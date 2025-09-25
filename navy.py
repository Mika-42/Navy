import threading, time, sys, queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import re
import os

class Navy:
    def __init__(self, filepath):
        # --- Chargement des vocabulaires et état ---
        self.vocabularies = self.load_vocabularies()
        self.active_theme = "general"  # par défaut
        self.filepath = filepath
        # --- Patterns bannis ---
        self.BANNED_PATTERNS = [
            r"Sous-?titrage.*",
            r"Sous-?titres.*",
            r"ST'? ?501",
            r"Translated by.*",
        ]

        # --- Paramètres ---
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.DTYPE = "float32"

        self.CHUNK_S = 0.8
        self.CORR_WINDOW_S = 20.0
        self.CORR_PERIOD_S = 8.0

        self.MODEL_NAME = "large-v3"
        self.DEVICE = "cuda"          # ou "cpu"
        self.COMPUTE_TYPE = "float16"

        self.BEAM_FAST = 6
        self.BEAM_CORR = 8
        self.TEMPERATURE = 0.0

        # --- Buffers partagés ---
        self.audio_q = queue.Queue()
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.tail_seconds_to_keep = int(self.CORR_WINDOW_S * self.SAMPLE_RATE)

        self.validated_text = ""   # texte figé
        self.pending_text = ""     # texte en cours
        self.render_lock = threading.Lock()
        self.last_render = ""
        self.last_corrected_shown = ""

        self.model = WhisperModel(self.MODEL_NAME, device=self.DEVICE, compute_type=self.COMPUTE_TYPE)

        # threads et stream
        self._worker_thread = None
        self._correction_thread = None
        self._stream = None
        self._running = False

    # --- Fonctions utilitaires (inchangées) ---
    def load_vocabularies(self):
        vocabs = {}
        folder = "vocabularies"
        for fname in os.listdir(folder):
            if fname.startswith("vocab_") and fname.endswith(".txt"):
                theme = fname.replace("vocab_", "").replace(".txt", "")
                with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                    raw = f.read().replace("\n", ",")
                    vocabs[theme] = [w.strip() for w in raw.split(",") if w.strip()]
        return vocabs

    def detect_theme(self, text, vocabularies):
        scores = {}
        for theme, vocab in vocabularies.items():
            count = sum(1 for w in vocab if w.lower() in text.lower())
            scores[theme] = count
        # Retourne le thème avec le plus de correspondances
        return max(scores, key=scores.get) if scores else None

    def clean_text(self, text: str) -> str:
        for pat in self.BANNED_PATTERNS:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
        return " ".join(text.split())

    def append_audio(self, samples: np.ndarray):
        """Stocke l'audio pour la correction périodique."""
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        self.audio_buffer = np.concatenate((self.audio_buffer, samples))
        if len(self.audio_buffer) > self.tail_seconds_to_keep:
            self.audio_buffer = self.audio_buffer[-self.tail_seconds_to_keep:]

    def transcribe_chunk(self, chunk: np.ndarray, beam: int, prompt: str = None) -> str:
        segments, _ = self.model.transcribe(
            audio=chunk,
            language="fr",
            beam_size=beam,
            temperature=self.TEMPERATURE,
            vad_filter=True,
            no_speech_threshold=0.7,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            initial_prompt=prompt,
        )
        return "".join(seg.text for seg in segments).strip()

    def render(self, text: str):
        text = " ".join(text.split())
        if text == self.last_render:
            return
        self.last_render = text
        # sys.stdout.write("\n" + text)
        # sys.stdout.flush()

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def normalize_text(self, s: str) -> str:
        return " ".join(s.split())

    # --- Workers (inchangés) ---
    def worker(self):
        """Lit la queue et transcrit en flux immédiat (brut)."""
        buffer_chunk = np.zeros(0, dtype=np.float32)
        chunk_samples = int(self.CHUNK_S * self.SAMPLE_RATE)

        while self._running:
            data = self.audio_q.get()
            buffer_chunk = np.concatenate((buffer_chunk, data))
            self.append_audio(data)
            while len(buffer_chunk) >= chunk_samples:
                chunk = buffer_chunk[:chunk_samples]
                buffer_chunk = buffer_chunk[chunk_samples:]
                text_chunk = self.transcribe_chunk(chunk, self.BEAM_FAST)
                if text_chunk:
                    if not self.pending_text.endswith(text_chunk):
                        self.pending_text = self.normalize_text(self.pending_text + " " + text_chunk)
                        # self.render("[BRUT] " + self.pending_text) # debug

    def overlap_merge(self, old: str, new: str) -> str:
        """Fusionne new dans old en supprimant les chevauchements."""
        old_n = self.normalize_text(old)
        new_n = self.normalize_text(new)
        max_overlap = min(len(old_n), len(new_n))
        for i in range(max_overlap, 0, -1):
            if old_n.endswith(new_n[:i]):
                return self.normalize_text(old_n + new_n[i:])
        if new_n in old_n:
            return old_n
        return self.normalize_text(old_n + " " + new_n)

    def correction_worker(self):
        """Corrige périodiquement la fin du texte et le fige."""
        while self._running:
            time.sleep(self.CORR_PERIOD_S)
            with self.render_lock:
                tail_audio = self.audio_buffer.copy()
            if len(tail_audio) < int(self.SAMPLE_RATE * 2):
                continue

                # Détection automatique du thème

            corrected_tail = self.transcribe_chunk(tail_audio, self.BEAM_CORR)
            detected = self.detect_theme(corrected_tail, self.vocabularies)
            if detected:
                self.active_theme = detected
            else:
                self.active_theme = "general" if "general" in self.vocabularies else list(self.vocabularies.keys())[0]

            light_prompt = ",".join(self.vocabularies[self.active_theme][:10])
            corrected_tail = self.transcribe_chunk(tail_audio, self.BEAM_CORR, prompt=light_prompt)

            with self.render_lock:
                if not corrected_tail:
                    continue

                new_validated = self.overlap_merge(self.validated_text, corrected_tail)

                if new_validated != self.validated_text:
                    self.render(self.clean_text(corrected_tail))
                    self.validated_text = new_validated
                    self.pending_text = ""
                    self.audio_buffer = np.zeros(0, dtype=np.float32)  # reset
                    # self.render("[VALIDÉ] " + self.validated_text) # to write in file

    # --- Callback audio (inchangé) ---
    def callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        mono = indata[:, 0] if indata.ndim > 1 else indata
        self.audio_q.put(mono.copy())   # juste push dans la queue

    # --- Méthodes de contrôle équivalentes à main() ---
    def listen(self):
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self.worker, daemon=True)
        self._correction_thread = threading.Thread(target=self.correction_worker, daemon=True)
        self._worker_thread.start()
        self._correction_thread.start()

        self._stream = sd.InputStream(samplerate=self.SAMPLE_RATE,
                                      channels=self.CHANNELS,
                                      dtype=self.DTYPE,
                                      callback=self.callback,
                                      blocksize=0,
                                      latency="high")
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

    def erase(self):
        os.remove(self.filepath)

    def set_source(self, source):
        pass