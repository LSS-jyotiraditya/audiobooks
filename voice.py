from pathlib import Path
from typing import List, Optional
import os
import tempfile
import subprocess
from openai import OpenAI

class VoiceIO:
    """
    STT (Whisper) + TTS (OpenAI TTS) utilities.

    - STT: Use when the user wants to speak a query about the book.
    - TTS: Use to read the book aloud and to vocalize LLM answers.
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        stt_model: str = "base",  # local whisper model name
        tts_model: str = "gpt-4o-mini-tts",
        default_voice: str = "alloy",
        use_local_whisper: bool = True,
    ):
        # Do not auto-instantiate OpenAI client here to avoid API key requirement on init
        self.client = client
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.default_voice = default_voice
        self.use_local_whisper = use_local_whisper
        self._whisper_model = None
        # Use OpenAI TTS only if an API key is present or a client was injected
        self.use_openai_tts = bool(client) or bool(os.environ.get("OPENAI_API_KEY"))

    def _ensure_whisper(self):
        if not self.use_local_whisper:
            return None
        if self._whisper_model is None:
            if whisper is None:
                raise RuntimeError(
                    "openai-whisper is not installed. Please `pip install openai-whisper` and ensure ffmpeg is available."
                )
            self._whisper_model = whisper.load_model(self.stt_model)
        return self._whisper_model

    def _ensure_openai_client(self) -> OpenAI:
        if self.client is None:
            self.client = OpenAI()
        return self.client

    def _tts_local_to_wav(self, text: str, wav_path: str, voice: Optional[str] = None):
        """
        Use espeak-ng to render `text` into `wav_path` safely by writing text to a temp file
        and invoking espeak-ng -f <file> -w <wav_path>. Only include -v if voice is non-empty.
        """
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)

        tf = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt")
        try:
            tf.write(text)
            tf.flush()
            tf.close()

            cmd = ["espeak-ng", "-w", wav_path, "-f", tf.name]
            # only add voice option if provided and non-empty
            if voice and str(voice).strip():
                cmd.extend(["-v", str(voice).strip()])

            proc = subprocess.run(cmd, capture_output=True, text=True)
            # check return code (including negative codes from signals)
            if proc.returncode != 0:
                stderr = proc.stderr.strip()
                raise RuntimeError(f"espeak-ng failed (code={proc.returncode}): {stderr}")
            if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                raise RuntimeError("espeak-ng did not produce a valid wav file")
        finally:
            try:
                os.remove(tf.name)
            except Exception:
                pass

    def _wav_to_target(self, wav_path: str, out_path: str):
        """
        Convert wav -> target audio (mp3/ogg) with ffmpeg, capturing stderr for clearer errors.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cmd = [
            "/usr/bin/ffmpeg",
            "-y",
            "-i",
            wav_path,
            "-vn",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-b:a",
            "192k",
            out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            # include stderr for debugging (ffmpeg prints reason here)
            raise RuntimeError(f"ffmpeg failed (code={proc.returncode}): {proc.stderr.strip()}")
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("ffmpeg conversion did not produce output file")

    def transcribe(self, audio_path: str, language: Optional[str] = None, prompt: Optional[str] = None) -> str:
        """
        Speech-to-text using local openai-whisper by default.
        """
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.use_local_whisper:
            model = self._ensure_whisper()
            result = model.transcribe(str(p), language=language, prompt=prompt)
            return (result.get("text") or "").strip()

        # Fallback: OpenAI API Whisper (requires OPENAI_API_KEY)
        with p.open("rb") as f:
            client = self._ensure_openai_client()
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
                prompt=prompt,
            )
        return getattr(result, "text", "") or ""

    def tts_bytes(self, text: str, voice: Optional[str] = None, audio_format: str = "mp3") -> bytes:
        """
        Text-to-speech; returns audio bytes. Uses OpenAI if available, otherwise offline TTS.
        """
        if not text:
            return b""
        v = voice or self.default_voice

        if self.use_openai_tts:
            client = self._ensure_openai_client()
            resp = client.audio.speech.create(
                model=self.tts_model,
                voice=v,
                input=text,
                format=audio_format,
            )
            return resp.read()

        with tempfile.TemporaryDirectory() as td:
            wav_path = str(Path(td) / "out.wav")
            out_path = str(Path(td) / f"out.{audio_format}")
            self._tts_local_to_wav(text, wav_path, voice=v)
            if audio_format.lower() in ["wav", "wave"]:
                return Path(wav_path).read_bytes()
            self._wav_to_target(wav_path, out_path)
            return Path(out_path).read_bytes()

    def tts_to_file(self, text: str, output_path: str, voice: Optional[str] = None) -> str:
        """
        Text-to-speech; writes audio to file. Uses OpenAI if available, otherwise offline TTS.
        """
        if not text:
            raise ValueError("No text provided for TTS.")
        v = voice or self.default_voice
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if self.use_openai_tts:
            client = self._ensure_openai_client()
            with client.audio.speech.with_streaming_response.create(
                model=self.tts_model,
                voice=v,
                input=text,
            ) as response:
                response.stream_to_file(str(out))
            return str(out)

        # Offline path
        with tempfile.TemporaryDirectory() as td:
            wav_path = str(Path(td) / "out.wav")
            self._tts_local_to_wav(text, wav_path, voice=v)
            self._wav_to_target(wav_path, str(out))
        return str(out)
    def tts_book(
        self,
        text: str,
        output_dir: str,
        base_name: str = "book_chunk",
        max_chars: int = 2000,
        voice: Optional[str] = None,
    ) -> List[str]:
        if not text:
            return []

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Prefer paragraph-based splitting, then length fallback
        parts: List[str] = []
        paragraphs = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
        buf = ""
        for p in paragraphs:
            if len(buf) + len(p) + 2 <= max_chars:
                buf = f"{buf}\n\n{p}".strip() if buf else p
            else:
                if buf:
                    parts.append(buf)
                if len(p) <= max_chars:
                    parts.append(p)
                    buf = ""
                else:
                    # Hard wrap long paragraph
                    start = 0
                    while start < len(p):
                        end = min(start + max_chars, len(p))
                        # try to break on space
                        cut = p.rfind(" ", start, end)
                        if cut == -1 or cut <= start + int(0.5 * max_chars):
                            cut = end
                        parts.append(p[start:cut].strip())
                        start = cut
                    buf = ""
        if buf:
            parts.append(buf)

        paths: List[str] = []
        for i, chunk in enumerate(parts):
            out_path = out_dir / f"{base_name}_{i+1:04d}.mp3"
            self.tts_to_file(chunk, str(out_path), voice=voice)
            paths.append(str(out_path))
        return paths