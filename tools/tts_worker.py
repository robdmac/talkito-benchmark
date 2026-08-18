#!/usr/bin/env python3

# Talkito - Universal TTS wrapper that works with any command
# Copyright (C) 2025 Robert Macrae
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Runs one TTS backend in its own interpreter, for models whose pins conflict with talkito's."""

# Executed by a separate virtualenv's python, so it must not import talkito.
# Protocol: one JSON request per line on stdin, one JSON response per line on stdout.
#   -> {"text": "hello", "out": "/tmp/x.wav"}
#   <- {"ok": true, "out": "/tmp/x.wav", "seconds": 1.2, "sample_rate": 24000, "audio_seconds": 2.0}
# The model stays resident between requests so timings measure synthesis, not loading.

import argparse
import contextlib
import json
import os
import sys
import time
import wave


# MeCab and other C extensions write straight to file descriptor 1, which Python-level
# redirection cannot catch and which would corrupt the protocol. Keep a private duplicate of the
# real stdout for responses and point descriptor 1 at stderr, so every stray write lands there.
_PROTOCOL = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)


def emit(payload):
    """Write one response line and flush, so the caller never blocks on a buffer."""
    _PROTOCOL.write(json.dumps(payload) + "\n")
    _PROTOCOL.flush()


def write_wav(path, samples, sample_rate):
    """Write mono 16-bit PCM, avoiding torchaudio's backend requirements."""
    import numpy as np

    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return len(pcm)


def pick_device(requested):
    """Choose the best available torch device."""
    import torch

    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ChatterboxBackend:
    """Resemble AI Chatterbox, a 0.5B Llama-backbone model taking raw text."""

    def __init__(self, device):
        from chatterbox.tts import ChatterboxTTS

        self.device = pick_device(device)
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sample_rate = self.model.sr

    def synthesize(self, request):
        kwargs = {}
        for name in ("exaggeration", "cfg_weight", "temperature"):
            if request.get(name) is not None:
                kwargs[name] = request[name]
        wav = self.model.generate(request["text"], **kwargs)
        return wav.cpu()


class StyleTTS2Backend:
    """StyleTTS2, a diffusion style model fed by gruut phonemes."""

    def __init__(self, device):
        import torch
        from styletts2 import tts

        self.device = pick_device(device)
        self.model = tts.StyleTTS2()
        self.sample_rate = 24000
        self.torch = torch

    def synthesize(self, request):
        audio = self.model.inference(request["text"], output_sample_rate=self.sample_rate)
        return self.torch.from_numpy(audio).unsqueeze(0)


class MeloTTSBackend:
    """MeloTTS, a VITS model with its own English front end."""

    def __init__(self, device):
        import torch
        from melo.api import TTS

        self.device = pick_device(device)
        self.model = TTS(language="EN", device=self.device)
        self.speaker_id = self.model.hps.data.spk2id["EN-US"]
        self.sample_rate = self.model.hps.data.sampling_rate
        self.torch = torch

    def synthesize(self, request):
        audio = self.model.tts_to_file(request["text"], self.speaker_id,
                                       output_path=None, quiet=True)
        return self.torch.from_numpy(audio).unsqueeze(0).float()


class XTTSBackend:
    """Coqui XTTS v2, a zero-shot multilingual cloner driven by a reference clip.

    It has no baked-in speaker, so a reference is mandatory: without one it produces nothing usable.
    The reference here is a sample this benchmark already generated, which keeps the run
    self-contained and uses a voice whose provenance is known.
    """

    MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(self, device):
        import os
        from TTS.api import TTS

        os.environ.setdefault("COQUI_TOS_AGREED", "1")   # non-interactive licence acceptance
        self.device = pick_device(device)
        if self.device == "mps":
            self.device = "cpu"                          # XTTS has no working MPS path
        self.model = TTS(self.MODEL).to(self.device)
        self.sample_rate = 24000
        self.speaker_wav = os.environ.get(
            "XTTS_SPEAKER_WAV",
            os.path.expanduser("~/.cache/talkito/benchmark-samples/piper.wav"))

    def synthesize(self, request):
        import numpy as np
        import torch

        wav = self.model.tts(text=request["text"], speaker_wav=self.speaker_wav, language="en")
        return torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)


class TortoiseBackend:
    """Tortoise TTS: a diffusion model that is famously slow, included because breadth was asked for.

    Runs its fastest preset. Even so it is expected to sit at the far end of the speed axis, which
    is the point of including it - the corpus is fixed, so a slow model simply takes longer.
    """

    def __init__(self, device):
        from tortoise.api import TextToSpeech

        self.device = pick_device(device)
        if self.device == "mps":
            self.device = "cpu"                    # tortoise has no working MPS path
        self.model = TextToSpeech(device=self.device)
        self.sample_rate = 24000

    def synthesize(self, request):
        wav = self.model.tts_with_preset(request["text"], preset="ultra_fast",
                                         voice_samples=None, conditioning_latents=None)
        return wav.squeeze(0).cpu()


BACKENDS = {
    "xtts": XTTSBackend,
    "tortoise": TortoiseBackend,
    "chatterbox": ChatterboxBackend,
    "styletts2": StyleTTS2Backend,
    "melotts": MeloTTSBackend,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    parser.add_argument("--device", default=os.environ.get("TTS_WORKER_DEVICE", "auto"))
    args = parser.parse_args()

    # These libraries print to stdout at import and during inference; stdout is our protocol
    # channel, so everything they emit has to be pushed to stderr instead
    with contextlib.redirect_stdout(sys.stderr):
        started = time.time()
        backend = BACKENDS[args.backend](args.device)
        load_seconds = time.time() - started

    emit({
        "ready": True,
        "backend": args.backend,
        "device": getattr(backend, "device", "?"),
        "sample_rate": backend.sample_rate,
        "load_seconds": load_seconds,
    })

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as e:
            emit({"ok": False, "error": f"bad request: {e}"})
            continue

        if request.get("command") == "quit":
            break

        try:
            out_path = request["out"]
            with contextlib.redirect_stdout(sys.stderr):
                started = time.time()
                wav = backend.synthesize(request)
                seconds = time.time() - started
                frames = write_wav(out_path, wav, backend.sample_rate)

            emit({
                "ok": True,
                "out": out_path,
                "seconds": seconds,
                "sample_rate": backend.sample_rate,
                "audio_seconds": frames / backend.sample_rate,
            })
        except Exception as e:
            emit({"ok": False, "error": f"{type(e).__name__}: {e}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
