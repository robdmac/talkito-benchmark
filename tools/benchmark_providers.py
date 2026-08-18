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

"""Extra TTS providers used only for benchmarking, deliberately kept out of the shipped package."""

# Nothing in talkito imports this module. It exists so the round-trip harness can score models
# that talkito itself will never ship:
#
#   * chatterbox / styletts2 / melotts pin numpy, torch or transformers versions that conflict
#     with talkito's, so each runs in its own virtualenv behind tools/tts_worker.py.
#   * chatterbox-q8 / chatterbox-q4 / pocket-tts are quantized GGUF models served by CrispASR,
#     a separate speech engine. It is a benchmark subject, not a dependency - talkito must not
#     ship on top of it.
#
# Import this module and call register() to add these providers to talkito's registry for the
# lifetime of the process.

import atexit
import contextlib
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.request import Request, urlopen

import io

from talkito import tts
from talkito.tts import PROVIDER_CLASSES, TTSProvider, log_message, normalize_for_bpe_speech

# Chatterbox pins numpy<2 and torch 2.6, so it cannot share talkito's environment and runs in its
# own interpreter. Point CHATTERBOX_PYTHON at that virtualenv's python.
chatterbox_python = os.environ.get('CHATTERBOX_PYTHON', str(Path(__file__).resolve().parent.parent / 'chatterbox_venv' / 'bin' / 'python'))
chatterbox_device = os.environ.get('CHATTERBOX_DEVICE', 'auto')
chatterbox_exaggeration = float(os.environ.get('CHATTERBOX_EXAGGERATION', '0.5'))
isolated_tts_device = os.environ.get('TTS_WORKER_DEVICE', 'auto')
# Quantized GGUF models served by the CrispASR C++ runtime, which has no Python bindings and so
# runs as a local HTTP server holding the model resident
crispasr_binary = os.environ.get('CRISPASR_BINARY',
                                 str(Path(__file__).resolve().parent.parent / 'crispasr' / 'build' / 'bin' / 'crispasr'))
crispasr_repo = os.environ.get('CRISPASR_REPO', 'cstr/chatterbox-GGUF')
# CrispASR prefixes voice-cloned output with a spoken AI disclosure and watermarks TTS output.
# Both are provenance obligations it will only waive behind --accept-marking-responsibility, an
# attestation that the marking duty is the operator's. That is the operator's call to make, never
# a default, so it is opt-in through the environment and off unless deliberately set.
crispasr_accept_marking = os.environ.get('TALKITO_BENCHMARK_ACCEPT_MARKING', '').lower() in ('1', 'true', 'yes')
# A 3B backbone can take longer than this to load, and 'auto' resolution downloads inside the
# startup window, so the budget is configurable rather than fixed
CRISPASR_START_TIMEOUT = float(os.environ.get('CRISPASR_START_TIMEOUT', '300'))
# 'model': 'auto' lets the runtime fetch its own default GGUF for that backend; an explicit
# filename is resolved from crispasr_repo instead
CRISPASR_MODELS: Dict[str, Dict[str, Any]] = {
    # Ships no built-in speaker: without a reference it ignores the input text entirely
    # (0/70, 100% WER, 42 runaways), so a reference voice is mandatory rather than optional
    'pocket-tts': {'backend': 'pocket-tts', 'model': 'auto', 'codec': None,
                   'port': 8767, 'normalize': True,
                   'voice': os.environ.get('POCKET_TTS_VOICE', '')},
    # Preset/baked-voice LM models in NeuTTS's size class, so none of them clone and none trip
    # the spoken-disclosure path
    'miotts': {'backend': 'miotts', 'model': 'auto', 'codec': None,
               'port': 8768, 'normalize': True},
    'vibevoice': {'backend': 'vibevoice-tts', 'model': 'auto', 'codec': None,
                  'port': 8769, 'normalize': True},
    'cosyvoice3': {'backend': 'cosyvoice3-tts', 'model': 'auto', 'codec': None,
                   'port': 8770, 'normalize': True},
    'csm': {'backend': 'csm', 'model': 'auto', 'codec': None,
            'port': 8771, 'normalize': True},
    'speecht5': {'backend': 'speecht5', 'model': 'auto', 'codec': None,
                 'port': 8772, 'normalize': True},
    'bark': {'backend': 'bark', 'model': 'auto', 'codec': None,
             'port': 8773, 'normalize': True},
    # Qwen3-TTS runs its LM at 12 Hz rather than the ~50 Hz most codecs use, which is the one
    # lever that actually moves RTF for an autoregressive model
    'qwen3-tts': {'backend': 'qwen3-tts', 'model': 'auto', 'codec': None,
                  'port': 8774, 'normalize': True},
    # Orpheus is a 3B Llama backbone over SNAC, so it is 4-16x the size of any NeuTTS variant.
    # Only three quants are published; q4_k is the smallest at 2.56 GB.
    'orpheus-q4': {'backend': 'orpheus', 'model': 'auto', 'model_quant': 'q4_k', 'codec': None,
                   'port': 8775, 'normalize': True},
    'orpheus-q8': {'backend': 'orpheus', 'model': 'auto', 'model_quant': 'q8_0', 'codec': None,
                   'port': 8776, 'normalize': True},
    # Sub-2-bit unsloth dynamic quants. These are the smallest published Orpheus builds and the
    # only ones close to real time here; whether a 3B TTS backbone survives 1-bit is the question.
    'orpheus-iq1s': {'backend': 'orpheus', 'repo': 'unsloth/orpheus-3b-0.1-ft-GGUF',
                     'model': 'orpheus-3b-0.1-ft-UD-IQ1_S.gguf', 'codec': None,
                     'port': 8777, 'normalize': True},
    'orpheus-iq1m': {'backend': 'orpheus', 'repo': 'unsloth/orpheus-3b-0.1-ft-GGUF',
                     'model': 'orpheus-3b-0.1-ft-UD-IQ1_M.gguf', 'codec': None,
                     'port': 8778, 'normalize': True},
    # Realtime candidates from the runtime's wider backend list. fastpitch is the interesting one:
    # non-autoregressive at 120 MB, so it should sit with piper and kokoro rather than the LMs.
    'fastpitch': {'backend': 'fastpitch', 'model': 'auto', 'codec': None,
                  'port': 8780, 'normalize': True},
    'f5-tts': {'backend': 'f5-tts', 'model': 'auto', 'codec': None,
               'port': 8781, 'normalize': True},
    'chatterbox-turbo': {'backend': 'chatterbox-turbo', 'model': 'auto', 'codec': None,
                         'port': 8782, 'normalize': True},
    # Wider open-weight sweep: everything the runtime can fetch that fits this machine,
    # regardless of speed. Ordered by download size.
    'bananamind-tts': {'backend': 'bananamind-tts', 'model': 'auto', 'codec': None,
                     'port': 8790, 'normalize': True},
    'cosyvoice3-rl': {'backend': 'cosyvoice3-tts-rl', 'model': 'auto', 'codec': None,
                    'port': 8791, 'normalize': True},
    # indextts is a zero-shot cloning model: with no reference voice it generates fluent audio
    # that ignores the input text entirely (0/12 twice, ~27s per phrase), exactly as pocket-tts
    # does. Unregistered so a re-run fails immediately rather than spending half an hour proving
    # it again; give it a reference voice to benchmark it properly.
    # 'indextts': {'backend': 'indextts', 'model': 'auto', 'codec': None,
    #            'port': 8792, 'normalize': True},
    'parler-tts': {'backend': 'parler-tts', 'model': 'auto', 'codec': None,
                 'port': 8793, 'normalize': True},
    'mini-omni2': {'backend': 'mini-omni2', 'model': 'auto', 'codec': None,
                 'port': 8794, 'normalize': True},
    'omnivoice': {'backend': 'omnivoice', 'model': 'auto', 'codec': None,
                'port': 8795, 'normalize': True},
    'outetts': {'backend': 'outetts', 'model': 'auto', 'codec': None,
              'port': 8796, 'normalize': True},
    'lfm2-audio': {'backend': 'lfm2-audio', 'model': 'auto', 'codec': None,
                 'port': 8797, 'normalize': True},
    'vibevoice-1.5b': {'backend': 'vibevoice-1.5b', 'model': 'auto', 'codec': None,
                     'port': 8798, 'normalize': True},
    'voxcpm2-tts': {'backend': 'voxcpm2-tts', 'model': 'auto', 'codec': None,
                  'port': 8799, 'normalize': True},
    'zonos': {'backend': 'zonos', 'model': 'auto', 'codec': None,
            'port': 8800, 'normalize': True},
    'vibevoice-bitnet': {'backend': 'vibevoice-bitnet', 'model': 'auto', 'codec': None,
                       'port': 8801, 'normalize': True},
    'tada-1b': {'backend': 'tada-1b', 'model': 'auto', 'codec': None,
              'port': 8802, 'normalize': True},
    'qwen3-tts-1.7b': {'backend': 'qwen3-tts-1.7b-base', 'model': 'auto', 'codec': None,
                     'port': 8803, 'normalize': True},
    'qwen3-tts-vd': {'backend': 'qwen3-tts-1.7b-voicedesign', 'model': 'auto', 'codec': None,
                   'port': 8804, 'normalize': True},
    'dia': {'backend': 'dia', 'model': 'auto', 'codec': None,
          'port': 8805, 'normalize': True},
    'dots-tts': {'backend': 'dots-tts', 'model': 'auto', 'codec': None,
               'port': 8806, 'normalize': True},
    'moss-tts': {'backend': 'moss-tts', 'model': 'auto', 'codec': None,
               'port': 8807, 'normalize': True},
    'tada-3b': {'backend': 'tada', 'model': 'auto', 'codec': None,
              'port': 8808, 'normalize': True},
    'moss-tts-local': {'backend': 'moss-tts-local', 'model': 'auto', 'codec': None,
                     'port': 8809, 'normalize': True},
    # 17.3 GB f16, by far the largest weights in the comparison. Deferred until disk allowed it.
    'kugelaudio': {'backend': 'kugelaudio', 'model': 'auto', 'codec': None,
                   'port': 8811, 'normalize': True},
    'chatterbox-q8': {'backend': 'chatterbox', 'model': 'chatterbox-t3-q8_0.gguf',
                      'codec': 'chatterbox-s3gen-q8_0.gguf', 'port': 8765, 'normalize': True},
    'chatterbox-q4': {'backend': 'chatterbox', 'model': 'chatterbox-t3-q4_k.gguf',
                      'codec': 'chatterbox-s3gen-q4_k.gguf', 'port': 8766, 'normalize': True},
}
# Models whose pins conflict with talkito's, each in its own virtualenv. 'normalize' says whether
# the model needs digits and acronyms spelled out before synthesis (raw-text models do).
ISOLATED_TTS_BACKENDS: Dict[str, Dict[str, Any]] = {
    'chatterbox': {'venv': 'chatterbox_venv', 'normalize': True},
    # gruut phonemizes but does not expand digits, so this needs the same treatment as a
    # raw-text model: the diagnostic measures a +20% WER gap on digits without it
    'styletts2': {'venv': 'styletts2_venv', 'normalize': True},
    # Despite bundling num2words and inflect, it still measures a gap on both digits and
    # acronyms, and normalizing lifts the technical category substantially
    'melotts': {'venv': 'melotts_venv', 'normalize': True},
    # XTTS v2 clones from a reference clip rather than shipping a speaker, so it needs one supplied
    'xtts': {'venv': 'xtts_venv', 'normalize': True},
    # Diffusion, and by far the slowest engine tried; included for breadth
    'tortoise': {'venv': 'tortoise_venv', 'normalize': True},
}
chatterbox_cfg_weight = float(os.environ.get('CHATTERBOX_CFG_WEIGHT', '0.5'))
CHATTERBOX_START_TIMEOUT = 300.0  # The 0.5B model takes over a minute to load on first use
CHATTERBOX_REQUEST_TIMEOUT = 600.0  # Sampling runs well over real time on CPU and MPS


class _IsolatedTTSWorker:
    """A resident model in its own virtualenv, spoken to in JSON lines."""

    def __init__(self, backend, process, info):
        self.backend = backend
        self.process = process
        self.info = info
        self.lock = threading.Lock()

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send one synthesis request and wait for its reply."""
        # Held across the exchange because the protocol pairs one reply to one request
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.backend} worker exited with code {self.process.returncode}")

            fd, out_path = tempfile.mkstemp(suffix='.wav', prefix=f'{self.backend}_')
            os.close(fd)
            payload = dict(payload, out=out_path)

            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()

            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.backend} worker closed its output")
            return json.loads(line)

    def shutdown(self) -> None:
        """Ask the worker to exit, then make sure it has."""
        with contextlib.suppress(Exception):
            self.process.stdin.write(json.dumps({'command': 'quit'}) + "\n")
            self.process.stdin.flush()
        with contextlib.suppress(Exception):
            self.process.wait(timeout=10)
        if self.process.poll() is None:
            with contextlib.suppress(Exception):
                self.process.kill()


_isolated_workers: Dict[str, '_IsolatedTTSWorker'] = {}
_isolated_worker_lock = threading.Lock()


def isolated_backend_python(backend: str) -> str:
    """Return the interpreter that runs a backend, overridable per backend by environment."""
    override = os.environ.get(f'TALKITO_{backend.upper()}_PYTHON')
    if override:
        return override
    venv = ISOLATED_TTS_BACKENDS[backend]['venv']
    return str(Path(__file__).resolve().parent.parent / venv / 'bin' / 'python')


def _get_isolated_worker(backend: str) -> Optional['_IsolatedTTSWorker']:
    """Start a backend's worker on first use and reuse it afterwards."""
    with _isolated_worker_lock:
        worker = _isolated_workers.get(backend)
        if worker is not None and worker.process.poll() is None:
            return worker

        python = isolated_backend_python(backend)
        script = Path(__file__).resolve().parent / 'tts_worker.py'
        if not os.path.isfile(python):
            log_message("ERROR", f"{backend} interpreter not found at {python}; "
                                 f"set TALKITO_{backend.upper()}_PYTHON")
            return None
        if not script.is_file():
            log_message("ERROR", f"TTS worker script not found at {script}")
            return None

        log_message("INFO", f"Starting {backend} worker: {python} {script}")
        process = subprocess.Popen(
            [python, str(script), '--backend', backend, '--device', isolated_tts_device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )

        line = process.stdout.readline()
        if not line:
            log_message("ERROR", f"{backend} worker exited before reporting ready")
            with contextlib.suppress(Exception):
                process.kill()
            return None

        info = json.loads(line)
        log_message("INFO", f"{backend} worker ready: {info}")
        worker = _IsolatedTTSWorker(backend, process, info)
        _isolated_workers[backend] = worker
        atexit.register(worker.shutdown)
        return worker


class IsolatedTTSProvider(TTSProvider):
    """Base for models that cannot share talkito's environment and run in a separate interpreter."""

    backend = ''

    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        try:
            if not text or not text.strip():
                log_message("WARNING", f"{self.backend}: Empty text provided, skipping synthesis")
                return None

            worker = _get_isolated_worker(self.backend)
            if worker is None:
                raise RuntimeError(f"{self.backend} worker unavailable")

            settings = ISOLATED_TTS_BACKENDS[self.backend]
            spoken = normalize_for_bpe_speech(text) if settings['normalize'] else text
            response = worker.request(dict(self.extra_request(), text=spoken))
            if not response.get('ok'):
                raise RuntimeError(response.get('error', 'unknown worker error'))

            out_path = response['out']
            try:
                with open(out_path, 'rb') as handle:
                    return handle.read(), ".wav"
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(out_path)
        except Exception as e:
            log_message("ERROR", f"{self.backend} synthesis error: {e}")
            return None

    def extra_request(self) -> Dict[str, Any]:
        """Backend-specific synthesis knobs."""
        return {}


# The 2E backbone ships only as q4; nano is the variant with a quantization ladder, and it is a
# phoneme-input cloning model rather than 2E's raw-text one, so it needs its own provider.
# The full NeuTTS matrix: every published backbone crossed with every device that backbone can
# actually use. The ONNX codec is CPU-only, so codec_device is always cpu; backbone_device is the
# variable. GGUF backbones go to llama.cpp (which has Metal kernels); torch backbones go to
# transformers (where MPS is the path Neuphonic's closed PR #123 targeted).
NEUTTS_VARIANTS = {
    # key                        kind        backbone repo                        backbone device
    "nt-2e-q4-cpu":     ("neutts2e", "neuphonic/neutts-2e-q4-gguf",   "cpu"),
    "nt-2e-q4-metal":   ("neutts2e", "neuphonic/neutts-2e-q4-gguf",   "metal"),
    # 2E ships a Q8_0 too, and it was missing here: the quantisation ladder was measured for
    # nano and Air but skipped for the current model at exactly the level where nano broke
    # worst (q8 scored worse than q4 there, which is still unexplained).
    "nt-2e-q8-cpu":     ("neutts2e", "neuphonic/neutts-2e-q8-gguf",   "cpu"),
    "nt-2e-q8-metal":   ("neutts2e", "neuphonic/neutts-2e-q8-gguf",   "metal"),
    "nt-2e-fp32-cpu":   ("neutts2e", "neuphonic/neutts-2e",           "cpu"),
    "nt-2e-fp32-mps":   ("neutts2e", "neuphonic/neutts-2e",           "mps"),
    "nt-nano-q4-cpu":   ("nano",     "neuphonic/neutts-nano-q4-gguf", "cpu"),
    "nt-nano-q4-metal": ("nano",     "neuphonic/neutts-nano-q4-gguf", "metal"),
    "nt-nano-q8-cpu":   ("nano",     "neuphonic/neutts-nano-q8-gguf", "cpu"),
    "nt-nano-q8-metal": ("nano",     "neuphonic/neutts-nano-q8-gguf", "metal"),
    "nt-nano-fp32-cpu": ("nano",     "neuphonic/neutts-nano",         "cpu"),
    "nt-nano-fp32-mps": ("nano",     "neuphonic/neutts-nano",         "mps"),
    # Air is the middle of the range at 0.748B, and phoneme-input like nano
    "nt-air-q4-cpu":    ("nano",     "neuphonic/neutts-air-q4-gguf",  "cpu"),
    "nt-air-q4-metal":  ("nano",     "neuphonic/neutts-air-q4-gguf",  "metal"),
    "nt-air-q8-cpu":    ("nano",     "neuphonic/neutts-air-q8-gguf",  "cpu"),
    "nt-air-q8-metal":  ("nano",     "neuphonic/neutts-air-q8-gguf",  "metal"),
    "nt-air-fp32-cpu":  ("nano",     "neuphonic/neutts-air",          "cpu"),
    "nt-air-fp32-mps":  ("nano",     "neuphonic/neutts-air",          "mps"),
}
neutts_nano_voice = os.environ.get("NEUTTS_VOICE", "emily")

_neutts_models: Dict[str, Any] = {}
_neutts_lock = threading.Lock()


_penalty_patched = False


def _apply_repeat_penalty():
    """Honour NEUTTS_REPEAT_PENALTY, which NeuTTS.infer() has no parameter for.

    The two backbone paths need different knobs and both have to be covered, or a sweep silently
    measures the penalty on the GGUF rows and nothing on the fp32 ones: quantised backbones
    sample through llama.cpp (`repeat_penalty`), fp32 and mps through HF `generate`
    (`repetition_penalty`). Both default to 1.0, i.e. off.
    """
    global _penalty_patched
    penalty = float(os.environ.get("NEUTTS_REPEAT_PENALTY", "1.0"))
    # min_p/top_p have to be able to apply on their own: the whole point of testing them is at
    # penalty 1.0, and returning early there would silently measure the baseline twice.
    wants_sampler = os.environ.get("NEUTTS_MIN_P") or os.environ.get("NEUTTS_TOP_P")
    if (penalty == 1.0 and not wants_sampler) or _penalty_patched:
        return
    _penalty_patched = True

    try:
        import llama_cpp
        original = llama_cpp.Llama.__call__

        announced = []
        # NeuTTS passes only temperature and top_k, so llama-cpp-python's own top_p=0.95 and
        # min_p=0.05 apply silently -- and the torch path, which gets HF's defaults, has
        # neither. NEUTTS_MIN_P / NEUTTS_TOP_P make that difference testable.
        min_p = os.environ.get("NEUTTS_MIN_P")
        top_p = os.environ.get("NEUTTS_TOP_P")

        def patched(self, prompt, *a, **kw):
            kw["repeat_penalty"] = penalty
            if min_p is not None:
                kw["min_p"] = float(min_p)
            if top_p is not None:
                kw["top_p"] = float(top_p)
            # Confirm once that the value actually reaches the sampler. Installing the patch and
            # the sampler honouring it are different claims, and a silent no-op here would make
            # a whole sweep look like "the penalty does nothing".
            if not announced:
                announced.append(1)
                print(f"  llama.cpp call: repeat_penalty={kw['repeat_penalty']} "
                      f"min_p={kw.get('min_p', 'default 0.05')} "
                      f"top_p={kw.get('top_p', 'default 0.95')}", flush=True)
            return original(self, prompt, *a, **kw)

        llama_cpp.Llama.__call__ = patched
    except ImportError:
        pass

    # The torch path calls backbone.generate(); patch the model class that neutts loads rather
    # than a specific instance, since each variant constructs its own.
    try:
        import transformers

        original_generate = transformers.GenerationMixin.generate

        def patched_generate(self, *a, **kw):
            kw.setdefault("repetition_penalty", penalty)
            return original_generate(self, *a, **kw)

        transformers.GenerationMixin.generate = patched_generate
    except (ImportError, AttributeError):
        pass

    print(f"  NEUTTS_REPEAT_PENALTY = {penalty} (llama.cpp and HF generate)", flush=True)


def _build_neutts(kind: str, backbone: str, backbone_device: str = "cpu"):
    """Construct a NeuTTS backbone directly, bypassing talkito's single-variant model cache."""
    from neutts import NeuTTS, NeuTTS2E

    _apply_repeat_penalty()

    model_class = NeuTTS2E if kind == "neutts2e" else NeuTTS
    resolved = tts._resolve_neutts_backbone(backbone)

    extra = {}
    if model_class is NeuTTS:
        # Resolving a GGUF repo to a file path loses the repo-id language lookup, so the language
        # has to be passed explicitly; 2E is BPE and loads no phonemizer at all
        try:
            from neutts import BACKBONE_LANGUAGE_MAP
            language = BACKBONE_LANGUAGE_MAP.get(backbone)
        except ImportError:
            language = None
        if language and resolved != backbone:
            extra["language"] = language

    return model_class(
        backbone_repo=resolved,
        backbone_device=backbone_device,
        codec_repo=tts._resolve_neutts_codec(tts.neutts_codec),
        codec_device=tts.neutts_device,
        **extra,
    )


def _neutts_reference(model, voice):
    """Load a bundled pre-encoded speaker reference, caching the costly encode on the model."""
    import torch
    from neutts import NeuTTS2E

    cache = getattr(model, "_bench_ref_cache", None)
    if cache is None:
        cache = {}
        setattr(model, "_bench_ref_cache", cache)
    if voice in cache:
        return cache[voice]

    sample_dir = Path(NeuTTS2E.SAMPLE_DIR)
    codes_path, text_path = sample_dir / f"{voice}.pt", sample_dir / f"{voice}.txt"
    if not codes_path.is_file():
        available = ", ".join(sorted(p.stem for p in sample_dir.glob("*.pt")))
        raise ValueError(f"Unknown NeuTTS voice {voice!r}; bundled voices: {available}")
    cache[voice] = (torch.load(codes_path), text_path.read_text().strip())
    return cache[voice]


# KittenTTS ships three sizes and talkito only ever loads one of them. The benchmark showed
# "kittentts" without saying which, so these make the ladder explicit -- and the family is small
# enough that all three fit comfortably.
KITTEN_VARIANTS = {
    "kittentts-micro": "KittenML/kitten-tts-micro-0.8",
    "kittentts-nano":  "KittenML/kitten-tts-nano-0.8-fp32",
    # int8 of the same nano weights: 23 MB against 54, and the one case here where a
    # quantisation can be judged on a model that does no sampling at all.
    "kittentts-nano-int8": "KittenML/kitten-tts-nano-0.8-int8",
    "kittentts-mini":  "KittenML/kitten-tts-mini-0.8",
}
_kitten_models: Dict[str, Any] = {}
_kitten_lock = threading.Lock()


class KittenVariantProvider(TTSProvider):
    """A specific KittenTTS size, so micro/nano/mini can be compared directly."""

    variant = ""

    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        try:
            import io
            import numpy as np
            import soundfile as sf

            if not text or not text.strip():
                return None
            repo = KITTEN_VARIANTS[self.variant]

            with _kitten_lock:
                model = _kitten_models.get(self.variant)
                if model is None:
                    log_message("INFO", f"Loading {self.variant} ({repo})")
                    from kittentts import KittenTTS
                    model = KittenTTS(repo)
                    _kitten_models[self.variant] = model

            voice = str(self.config.get('voice') or tts.kittentts_voice)
            audio = np.asarray(model.generate(text, voice=voice), dtype=np.float32)
            buf = io.BytesIO()
            # Clipped deliberately: these are ONNX models with no output constraint, and PCM16
            # wraps rather than saturates on overflow.
            sf.write(buf, np.clip(audio, -1.0, 1.0), 24000, format='WAV')
            return buf.getvalue(), ".wav"
        except Exception as e:
            log_message("ERROR", f"{self.variant} synthesis error: {e}")
            return None


class NeuTTSVariantProvider(TTSProvider):
    """A specific NeuTTS backbone, so the quantization ladder can be compared directly."""

    variant = ""

    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        try:
            import soundfile as sf

            if not text or not text.strip():
                return None
            kind, backbone, device = NEUTTS_VARIANTS[self.variant]

            with _neutts_lock:
                model = _neutts_models.get(self.variant)
                if model is None:
                    log_message("INFO", f"Loading {self.variant} ({backbone} on {device})")
                    model = _build_neutts(kind, backbone, device)
                    _neutts_models[self.variant] = model

            if kind == "neutts2e":
                # 2E has no phonemizer, so digits and acronyms are spelled out here
                audio = model.infer(normalize_for_bpe_speech(text), speaker=tts.neutts2e_voice,
                                    emotion=tts.neutts2e_emotion)
            else:
                ref_codes, ref_text = _neutts_reference(model, neutts_nano_voice)
                audio = model.infer(text, ref_codes, ref_text)

            buf = io.BytesIO()
            sf.write(buf, audio, tts.NEUTTS_SAMPLE_RATE, format="WAV")
            return buf.getvalue(), ".wav"
        except Exception as e:
            log_message("ERROR", f"{self.variant} synthesis error: {e}")
            return None


class ChatterboxProvider(IsolatedTTSProvider):
    """Resemble AI Chatterbox, a 0.5B raw-text model."""

    backend = 'chatterbox'

    def extra_request(self) -> Dict[str, Any]:
        return {
            'exaggeration': self.config.get('exaggeration', chatterbox_exaggeration),
            'cfg_weight': self.config.get('cfg_weight', chatterbox_cfg_weight),
        }


class StyleTTS2Provider(IsolatedTTSProvider):
    """StyleTTS2, a diffusion style model with a gruut phoneme front end."""

    backend = 'styletts2'


class MeloTTSProvider(IsolatedTTSProvider):
    """MeloTTS, a VITS model with its own English text front end."""

    backend = 'melotts'


class XTTSProvider(IsolatedTTSProvider):
    """Coqui XTTS v2, cloning from a reference clip rather than a baked-in speaker."""

    backend = 'xtts'


class TortoiseProvider(IsolatedTTSProvider):
    """Tortoise TTS, a diffusion model run at its fastest preset."""

    backend = 'tortoise'


class _CrispASRServer:
    """A CrispASR process holding one GGUF model resident, driven over its HTTP API."""

    def __init__(self, name, process, port):
        self.name = name
        self.process = process
        self.port = port
        self.lock = threading.Lock()

    def synthesize(self, text: str) -> bytes:
        """Ask the server for audio, returning WAV bytes."""
        payload = json.dumps({'input': text}).encode('utf-8')
        request = Request(f'http://127.0.0.1:{self.port}/v1/audio/speech', data=payload,
                          headers={'Content-Type': 'application/json'})
        # One request at a time: the server holds a single model instance
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name} server exited with code {self.process.returncode}")
            with urlopen(request, timeout=CRISPASR_REQUEST_TIMEOUT) as response:
                return response.read()

    def shutdown(self) -> None:
        """Stop the server process."""
        with contextlib.suppress(Exception):
            self.process.terminate()
            self.process.wait(timeout=10)
        if self.process.poll() is None:
            with contextlib.suppress(Exception):
                self.process.kill()


# Quantized sampling still runs well over real time, and this sweep deliberately includes models
# that are nowhere near it. Configurable so a slow model is measured rather than recorded as a
# failure: a timeout discards the phrase, which looks identical to the model producing nothing.
CRISPASR_REQUEST_TIMEOUT = float(os.environ.get('CRISPASR_REQUEST_TIMEOUT', '600'))

_crispasr_servers: Dict[str, '_CrispASRServer'] = {}
_crispasr_server_lock = threading.Lock()


def _crispasr_model_path(filename: str, repo: Optional[str] = None) -> str:
    """Resolve a GGUF file from the local Hub cache, downloading it if needed.

    Quants below q4 are published by third parties rather than in the default repo, so a model
    can name its own source.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo or crispasr_repo, filename=filename)


def _get_crispasr_server(name: str) -> Optional['_CrispASRServer']:
    """Start a model's server on first use and reuse it afterwards."""
    with _crispasr_server_lock:
        server = _crispasr_servers.get(name)
        if server is not None and server.process.poll() is None:
            return server

        settings = CRISPASR_MODELS[name]
        if not os.path.isfile(crispasr_binary):
            log_message("ERROR", f"CrispASR binary not found at {crispasr_binary}; "
                                 f"build it or set CRISPASR_BINARY")
            return None

        try:
            model_path = ('auto' if settings['model'] == 'auto'
                          else _crispasr_model_path(settings['model'], settings.get('repo')))
            codec_path = _crispasr_model_path(settings['codec']) if settings['codec'] else None
        except Exception as e:
            log_message("ERROR", f"Could not resolve {name} GGUF files: {e}")
            return None

        port = settings['port']
        log_message("INFO", f"Starting CrispASR server for {name} on port {port}")
        command = [crispasr_binary, '--server', '--port', str(port),
                   '--backend', settings['backend'], '-m', model_path]
        if codec_path:
            command += ['--codec-model', codec_path]

        # Registry auto-resolution picks a default quant; naming one keeps the comparison honest
        if settings.get('model_quant'):
            command += ['--model-quant', settings['model_quant']]

        # Models with no built-in speaker can only synthesize by cloning a reference
        voice = settings.get('voice')
        if voice:
            command += ['--voice', voice, '--i-have-rights']
            if not crispasr_accept_marking:
                log_message("WARNING",
                            f"{name} clones a reference voice, so its output carries a spoken AI "
                            f"disclosure that will be transcribed and counted against WER. Set "
                            f"TALKITO_BENCHMARK_ACCEPT_MARKING=1 to drop it.")

        if crispasr_accept_marking:
            # Measurement purity: the disclosure prefix lands in the transcript, and the watermark
            # perturbs the very audio the ASR scores
            command += ['--accept-marking-responsibility', '--no-spoken-disclaimer', '--no-watermark']
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        deadline = time.time() + CRISPASR_START_TIMEOUT
        while time.time() < deadline:
            if process.poll() is not None:
                log_message("ERROR", f"{name} server exited during startup")
                return None
            try:
                with urlopen(f'http://127.0.0.1:{port}/health', timeout=2):
                    break
            except Exception:
                time.sleep(1.0)
        else:
            log_message("ERROR", f"{name} server did not become healthy within {CRISPASR_START_TIMEOUT}s")
            with contextlib.suppress(Exception):
                process.kill()
            return None

        server = _CrispASRServer(name, process, port)
        _crispasr_servers[name] = server
        atexit.register(server.shutdown)
        log_message("INFO", f"CrispASR server for {name} is ready")
        return server


class CrispASRProvider(TTSProvider):
    """Base for quantized GGUF models served by the CrispASR runtime."""

    model_key = ''

    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        try:
            if not text or not text.strip():
                log_message("WARNING", f"{self.model_key}: Empty text provided, skipping synthesis")
                return None

            server = _get_crispasr_server(self.model_key)
            if server is None:
                raise RuntimeError(f"{self.model_key} server unavailable")

            settings = CRISPASR_MODELS[self.model_key]
            spoken = normalize_for_bpe_speech(text) if settings['normalize'] else text
            return server.synthesize(spoken), ".wav"
        except Exception as e:
            log_message("ERROR", f"{self.model_key} synthesis error: {e}")
            return None


class PocketTTSProvider(CrispASRProvider):
    """Kyutai Pocket TTS, a 100M continuous-latent autoregressive model at 12.5 Hz."""

    model_key = 'pocket-tts'


def _make_crispasr_provider(key):
    """Build a provider class for one CrispASR-served model."""
    return type(f"{key.title().replace('-', '')}Provider", (CrispASRProvider,), {
        'model_key': key,
        '__doc__': f"{key} served by the CrispASR runtime.",
    })


class ChatterboxQ8Provider(CrispASRProvider):
    """Chatterbox with 8-bit quantized weights."""

    model_key = 'chatterbox-q8'


class ChatterboxQ4Provider(CrispASRProvider):
    """Chatterbox with 4-bit quantized weights."""

    model_key = 'chatterbox-q4'

class MMSTTSProvider(TTSProvider):
    """Meta's MMS-TTS: a VITS checkpoint per language, run in-process through transformers.

    Included because it is the easiest open-weight engine to obtain that is not served by the C++
    runtime - no separate install, no virtualenv - and because it is a plain VITS, which the
    comparison is otherwise thin on.
    """

    repo = 'facebook/mms-tts-eng'
    _model = None
    _tokenizer = None

    def synthesize(self, text: str, **kwargs) -> Tuple[bytes, str]:
        import numpy as np
        import torch

        if MMSTTSProvider._model is None:
            from transformers import VitsModel, AutoTokenizer
            MMSTTSProvider._tokenizer = AutoTokenizer.from_pretrained(self.repo)
            MMSTTSProvider._model = VitsModel.from_pretrained(self.repo).eval()

        # VITS reads characters, not phonemes, so digits and acronyms need spelling out first
        inputs = MMSTTSProvider._tokenizer(normalize_for_bpe_speech(text), return_tensors="pt")
        with torch.no_grad():
            wave = MMSTTSProvider._model(**inputs).waveform[0].cpu().numpy()

        rate = MMSTTSProvider._model.config.sampling_rate
        buf = io.BytesIO()
        import wave as wavemod
        with wavemod.open(buf, 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes((np.clip(wave, -1, 1) * 32767).astype(np.int16).tobytes())
        return buf.getvalue(), '.wav'


BENCHMARK_PROVIDERS = {
    'mms-tts': MMSTTSProvider,
    'xtts': XTTSProvider,
    'tortoise': TortoiseProvider,
    'chatterbox': ChatterboxProvider,
    'styletts2': StyleTTS2Provider,
    'melotts': MeloTTSProvider,
    'chatterbox-q8': ChatterboxQ8Provider,
    'chatterbox-q4': ChatterboxQ4Provider,
    'pocket-tts': PocketTTSProvider,
    **{key: _make_crispasr_provider(key)
       for key in ('miotts', 'vibevoice', 'cosyvoice3', 'csm', 'speecht5', 'bark',
                   'qwen3-tts', 'orpheus-q4', 'orpheus-q8', 'fastpitch', 'f5-tts', 'kugelaudio',
                   'bananamind-tts', 'cosyvoice3-rl', 'indextts', 'parler-tts', 'mini-omni2', 'omnivoice', 'outetts', 'lfm2-audio', 'vibevoice-1.5b', 'voxcpm2-tts', 'zonos', 'vibevoice-bitnet', 'tada-1b', 'qwen3-tts-1.7b', 'qwen3-tts-vd', 'dia', 'dots-tts', 'moss-tts', 'tada-3b', 'moss-tts-local',
                   'chatterbox-turbo',
                   'orpheus-iq1s', 'orpheus-iq1m')},
    **{key: type(f"{key.title().replace('-', '')}Provider", (KittenVariantProvider,),
                 {'variant': key, '__doc__': f"KittenTTS {key}."})
       for key in KITTEN_VARIANTS},
    **{key: type(f"{key.title().replace('-', '')}Provider", (NeuTTSVariantProvider,),
                 {'variant': key, '__doc__': f"NeuTTS backbone {key}."})
       for key in NEUTTS_VARIANTS},
}


def register() -> Dict[str, Any]:
    """Add the benchmark-only providers to talkito's registry for this process."""
    PROVIDER_CLASSES.update(BENCHMARK_PROVIDERS)
    return BENCHMARK_PROVIDERS
