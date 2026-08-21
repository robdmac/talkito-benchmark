#!/usr/bin/env python3

"""End-to-end round-trip test: TTS synthesizes text, ASR transcribes it back, results are scored."""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import wave
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing talkito.tts applies patch_phonemizer_espeak_api(), which kokoro/kittentts need at import time
import talkito.tts as tts  # noqa: E402
from talkito.asr import ASRConfig, PROVIDERS as ASR_PROVIDERS  # noqa: E402

# Models talkito does not ship - separate virtualenvs, and GGUF served by CrispASR, which is a
# benchmark subject rather than a dependency. Absent unless that tooling has been set up locally.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
try:
    import benchmark_providers  # noqa: E402

    benchmark_providers.register()
except ImportError:
    pass

# ASR is fed 16 kHz mono; faster-whisper assumes this rate when handed a raw array
ASR_SAMPLE_RATE = 16000

@dataclass
class Phrase:
    """One benchmark item: the text to speak plus every transcription we would accept as correct."""
    text: str
    category: str
    accept: List[str] = field(default_factory=list)
    raw: Optional[str] = None  # Terminal-style source text, run through talkito's own preprocessing

    def references(self) -> List[str]:
        return [self.text] + self.accept


# Corpus grouped by what each group stresses. `accept` lists alternative readings that are
# genuinely correct, so the score measures intelligibility rather than formatting choices.
CORPUS: List[Phrase] = [
    # --- core: plain prose, the regression baseline ---
    Phrase("Hello, world.", "core"),
    Phrase("The quick brown fox jumps over the lazy dog.", "core"),
    Phrase("Running tests now, please wait a moment.", "core"),
    Phrase("I found three errors in the configuration file.", "core"),
    Phrase("She sells sea shells by the sea shore.", "core",
           ["She sells seashells by the seashore."]),
    Phrase("The rain in Spain falls mainly on the plain.", "core"),

    # --- numeric: digits, versions, money, time, sizes ---
    Phrase("The build finished successfully after 42 seconds.", "numeric"),
    Phrase("Pi is approximately 3.14159.", "numeric",
           ["Pi is approximately 3.14159", "Pie is approximately 3.14159"]),
    Phrase("Upgrade from version 1.2.3 to 2.0.1.", "numeric",
           ["Upgrade from version one point two point three to two point oh one"]),
    Phrase("Coverage increased to 87.5 percent.", "numeric", ["Coverage increased to 87.5%"]),
    Phrase("The total came to 1,234 dollars and 56 cents.", "numeric",
           ["The total came to $1,234.56", "The total came to 1234 dollars and 56 cents"]),
    Phrase("The job started at 9:45 and ran for 200 milliseconds.", "numeric",
           ["The job started at nine forty five and ran for 200 milliseconds",
            "The job started at 9 45 and ran for 200 milliseconds"]),
    Phrase("Allocate 16 gigabytes of memory.", "numeric", ["Allocate 16 GB of memory"]),

    # --- technical: acronyms, product names, identifiers ---
    Phrase("The HTTP request returned a JSON payload.", "technical"),
    Phrase("Connect to the PostgreSQL database through nginx.", "technical",
           ["Connect to the Postgres SQL database through nginx",
            "Connect to the PostgreSQL database through engine X"]),
    Phrase("The CPU usage spiked while the GPU stayed idle.", "technical"),
    Phrase("Deploy the container to Kubernetes.", "technical"),
    Phrase("Check the API key before calling the SDK.", "technical"),
    Phrase("Parse the YAML file and validate the schema.", "technical",
           ["Parse the yaml file and validate the schema"]),

    # --- prosody: questions, emphasis, clauses, length ---
    Phrase("Would you like me to commit these changes?", "prosody"),
    Phrase("Wait, that is not what I expected at all!", "prosody",
           ["Wait, that's not what I expected at all!"]),
    Phrase("If the tests pass, and the reviewer approves, then we can merge on Friday.", "prosody"),
    Phrase("The migration, which touched every table in the database, completed without errors.",
           "prosody"),
    Phrase("First, back up the data. Second, run the migration. Third, verify the results.",
           "prosody"),
    Phrase("Are you sure? This cannot be undone.", "prosody", ["Are you sure? This can not be undone."]),

    # --- edge: short, repeated, homographs, awkward punctuation ---
    Phrase("Done.", "edge"),
    Phrase("Ok", "edge", ["OK", "Okay"]),
    Phrase("The lead developer will read the report she read yesterday.", "edge",
           ["The lead developer will read the report she read yesterday"]),
    Phrase("I had had enough of that that day.", "edge"),
    Phrase("It is live, and the crowd is live.", "edge", ["It's live, and the crowd is live."]),
    Phrase("Well... maybe. Perhaps not.", "edge", ["Well, maybe. Perhaps not.", "Well maybe perhaps not"]),

    # --- pipeline: raw terminal output through talkito's own preprocessing.
    # The spoken form is whatever extract_speakable_text produces, so these measure how well a
    # voice renders real post-processed output; `accept` only widens tolerance.
    Phrase("", "pipeline", ["Updated parse config file in loader dot pie"],
           raw="Updated `parse_config_file()` in **src/config/loader.py**"),
    Phrase("", "pipeline", ["Tests passed equals 12 failed equals 0 skipped equals 3"],
           raw="Tests: passed=12 failed=0 skipped=3"),
    Phrase("", "pipeline", ["All checks completed successfully"],
           raw="## All checks completed successfully (took 4s)"),
    Phrase("", "pipeline", ["Warning deprecated API will be removed in version 3"],
           raw="WARNING: deprecated API will be removed in v3.0"),
    # --- core (expanded) ---
    Phrase("The server restarted without any warning.", "core"),
    Phrase("Please confirm before deleting the branch.", "core"),
    Phrase("Nothing changed since the last commit.", "core"),
    Phrase("A quick check of the logs found the cause.", "core"),
    Phrase("The cat sat on the mat and refused to move.", "core"),
    Phrase("Peter Piper picked a peck of pickled peppers.", "core"),
    Phrase("How much wood would a woodchuck chuck.", "core"),
    Phrase("The sixth sick sheikh's sixth sheep is sick.", "core",
           ["The sixth sick sheiks sixth sheep is sick."]),

    # --- numeric (expanded) ---
    Phrase("The job ran for 1 hour and 17 minutes.", "numeric"),
    Phrase("Memory use peaked at 3.7 gigabytes.", "numeric"),
    Phrase("There were 1,024 rows in the result.", "numeric",
           ["There were 1024 rows in the result."]),
    Phrase("Set the timeout to 0.25 seconds.", "numeric"),
    Phrase("The release is scheduled for June 3rd, 2027.", "numeric",
           ["The release is scheduled for June third, 2027."]),
    Phrase("Coverage dropped from 91% to 88%.", "numeric",
           ["Coverage dropped from 91 percent to 88 percent."]),
    Phrase("Call me at 555 0199 after 6 pm.", "numeric",
           ["Call me at 555 0199 after 6 p.m."]),

    # --- technical (expanded) ---
    Phrase("The API returned a 404 for that endpoint.", "technical",
           ["The A P I returned a 404 for that endpoint."]),
    Phrase("Run npm install before starting the server.", "technical"),
    Phrase("The SSH key was rejected by the host.", "technical",
           ["The S S H key was rejected by the host."]),
    Phrase("Check the JSON schema for a missing comma.", "technical"),
    Phrase("The CPU stayed idle while the GPU worked.", "technical",
           ["The C P U stayed idle while the G P U worked."]),
    Phrase("Rebase onto main and force push the branch.", "technical"),

    # --- prosody (expanded) ---
    Phrase("Wait, did you really mean to do that?", "prosody"),
    Phrase("That is absolutely not what I expected!", "prosody"),
    Phrase("Are you sure, or should we check again?", "prosody"),
    Phrase("Well, that explains everything.", "prosody"),
    Phrase("Stop. Read the error. Then try again.", "prosody"),
    Phrase("It works, but only sometimes.", "prosody"),

    # --- edge (expanded) ---
    Phrase("No.", "edge"),
    Phrase("Done", "edge"),
    Phrase("Yes, exactly that.", "edge"),
    Phrase("Hmm.", "edge", ["Hm.", "Hmm", "Um."]),
    Phrase("A", "edge", ["A.", "Ay."]),
    Phrase("It is what it is.", "edge"),

    # --- pipeline (expanded) ---
    Phrase("Wrote 122 lines to talkito slash tts dot py", "pipeline"),
    Phrase("Reading configuration from the home directory", "pipeline"),
    Phrase("Skipped 3 files that had not changed", "pipeline"),
    Phrase("Applied the patch and reran the failing test", "pipeline"),

    # --- longform: sustained generation, where autoregressive models drift or run away ---
    Phrase("The deployment finished in the middle of the afternoon, and although the logs "
           "showed no errors, the team decided to roll it back until the following morning "
           "so that the database migration could be checked once more.", "longform"),
    Phrase("When the build first failed we assumed the cache was stale, but after clearing "
           "it twice and rerunning the whole suite from scratch it became clear that the "
           "problem was a missing environment variable in the continuous integration "
           "configuration.", "longform"),
    Phrase("If you are reading this message it means the process completed, though several "
           "warnings were printed along the way, and you should review them before treating "
           "the result as final.", "longform"),
    Phrase("The report covers three areas: how long each stage took, how much memory was "
           "used at the peak, and which of the tests were skipped because their "
           "dependencies were unavailable.", "longform"),
    Phrase("She explained that the change was small but the consequences were not, because "
           "every downstream service parsed the same field and none of them had been "
           "updated to expect the new format.", "longform"),
    Phrase("Start by reading the summary at the top, then look at the failures in order, "
           "and only after that should you open the individual log files, which are long "
           "and mostly repetitive.", "longform"),

    # --- punctuation: pausing and phrasing marks the corpus otherwise never exercises ---
    Phrase("The tests passed; the linter did not.", "punctuation"),
    Phrase("He said \"the build is green\" and left.", "punctuation",
           ["He said the build is green and left."]),
    Phrase("The result (after retrying twice) was the same.", "punctuation"),
    Phrase("It failed again - the third time today.", "punctuation",
           ["It failed again, the third time today."]),
    Phrase("Everything worked... eventually.", "punctuation",
           ["Everything worked, eventually.", "Everything worked eventually."]),
    Phrase("Three things matter: speed, accuracy, and cost.", "punctuation"),
]

CATEGORIES = ["core", "numeric", "technical", "prosody", "edge", "pipeline",
              "longform", "punctuation"]

# Paired probes for the normalization diagnostic. Each pair says the same thing twice: once as a
# user would write it, once pre-normalized the way eSpeak would hand it to a phoneme model. A
# provider that scores much better on the normalized form is not "bad" - it is a raw-text (BPE)
# model that needs normalization applied before synthesis.
NORMALIZATION_PROBES: List[Tuple[str, str, str]] = [
    ("digits", "The total came to 1,234 dollars and 56 cents.",
     "The total came to one thousand two hundred thirty four dollars and fifty six cents."),
    ("digits", "The build finished after 42 seconds.",
     "The build finished after forty two seconds."),
    ("digits", "Upgrade from version 1.2.3 to 2.0.1.",
     "Upgrade from version one point two point three to two point zero point one."),
    ("acronyms", "The CPU usage spiked while the GPU stayed idle.",
     "The C P U usage spiked while the G P U stayed idle."),
    ("acronyms", "Check the API key before calling the SDK.",
     "Check the A P I key before calling the S D K."),
    ("acronyms", "The HTTP request returned a payload.",
     "The H T T P request returned a payload."),
]

# Architecture class, which accounts for most of the RTF and run-to-run variation differences.
#   det-ff   one forward pass with a deterministic duration predictor: same timing every run,
#            cannot hallucinate, tiny RTF
#   stoch-ff same one-pass architecture but sampling inside it (VITS stochastic duration
#            predictor, or diffusion style sampling): output length varies between runs
#   ar-lm    a transformer samples audio tokens one at a time and a codec decodes them: length is
#            emergent, emotion and cloning are controllable, and RTF is floored by sequential
#            decoding of roughly 50 tokens per second of audio
#   os       whatever the operating system ships
ARCHITECTURE_CLASSES = {
    "kokoro": "det-ff",
    "kittentts": "det-ff",
    "piper": "stoch-ff",
    "melotts": "stoch-ff",
    "styletts2": "stoch-ff",
    "neutts2e": "ar-lm",
    "neutts2e-fp32": "ar-lm",
    "nt-2e-q4-cpu": "ar-lm",
    "nt-2e-q4-metal": "ar-lm",
    "nt-2e-fp32-cpu": "ar-lm",
    "nt-2e-fp32-mps": "ar-lm",
    "nt-nano-q4-cpu": "ar-lm",
    "nt-nano-q4-metal": "ar-lm",
    "nt-nano-q8-cpu": "ar-lm",
    "nt-nano-q8-metal": "ar-lm",
    "nt-nano-fp32-cpu": "ar-lm",
    "nt-nano-fp32-mps": "ar-lm",
    "nt-air-q4-cpu": "ar-lm",
    "nt-air-q4-metal": "ar-lm",
    "nt-air-q8-cpu": "ar-lm",
    "nt-air-q8-metal": "ar-lm",
    "nt-air-fp32-cpu": "ar-lm",
    "nt-air-fp32-mps": "ar-lm",
    "neutts-nano": "ar-lm",
    "neutts-nano-q4": "ar-lm",
    "neutts-nano-q8": "ar-lm",
    "chatterbox": "ar-lm",
    "chatterbox-q8": "ar-lm",
    "chatterbox-q4": "ar-lm",
    "pocket-tts": "ar-lm",
    "vibevoice": "ar-lm",
    "cosyvoice3": "ar-lm",
    "csm": "ar-lm",
    "speecht5": "ar-lm",
    "bark": "ar-lm",
    "qwen3-tts": "ar-lm",
    "miotts": "ar-lm",
    "system": "os",
    # Remote services: the architecture is not observable from here, and RTF measures the network
    # round trip rather than the model
    "openai": "cloud",
    "aws": "cloud",
    "polly": "cloud",
    "azure": "cloud",
    "gcloud": "cloud",
    "elevenlabs": "cloud",
    "deepgram": "cloud",
}
UNKNOWN_ARCHITECTURE = "?"  # A newly dropped-in model until it is classified above

# A provider whose raw-vs-normalized WER gap exceeds this is flagged as needing normalization
NORMALIZATION_GAP_THRESHOLD = 0.15

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

# Symbols that TTS engines speak as words; expanded on both sides so either spelling scores equal
SYMBOL_WORDS = {"%": " percent ", "$": " dollars ", "&": " and ", "+": " plus ", "=": " equals "}

# Unit abbreviations ASR may emit where the reference spelled the word out (or vice versa)
UNIT_WORDS = {
    "gb": "gigabytes", "mb": "megabytes", "kb": "kilobytes", "tb": "terabytes",
    "ms": "milliseconds", "ghz": "gigahertz", "mhz": "megahertz",
}

RE_INNER_DOT = re.compile(r"(?<=[a-z])\.(?=[a-z])")   # loader.py -> loader dot py
RE_DIGIT_LETTER = re.compile(r"(?<=\d)(?=[a-z])")     # 16gb -> 16 gb


@dataclass
class RoundTripResult:
    """Outcome of a single synthesize-then-transcribe cycle."""
    tts_provider: str
    asr_provider: str
    reference: str
    hypothesis: str
    wer: float
    tts_seconds: float
    asr_seconds: float
    audio_seconds: float
    speech_seconds: float = 0.0
    from_cache: bool = False
    category: str = "core"
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.wer <= WER_THRESHOLD

    @property
    def rtf(self) -> float:
        """Real-time factor: compute seconds per second of audio produced."""
        return self.tts_seconds / self.audio_seconds if self.audio_seconds else float("nan")

    @property
    def speech_rtf(self) -> float:
        """Real-time factor against speech alone, ignoring silence at either end.

        Plain RTF divides by everything the model emitted, so padding inflates the denominator and
        a model is rewarded for the silence it should not have produced. Dividing by the speech
        instead measures the work done per second of speech actually delivered.
        """
        return self.tts_seconds / self.speech_seconds if self.speech_seconds else float("nan")


@dataclass
class ProviderReport:
    """Aggregated results for one TTS/ASR provider pairing."""
    tts_provider: str
    asr_provider: str
    results: List[RoundTripResult] = field(default_factory=list)

    @property
    def mean_wer(self) -> float:
        # Per-phrase WER is unbounded above: one runaway generation emitting 100 words for a
        # 12-word reference scores ~800% and swamps the mean. Cap each phrase at 100% so the
        # average reflects typical quality; runaways are still counted by `runaways` below.
        scored = [min(r.wer, 1.0) for r in self.results if r.error is None]
        return sum(scored) / len(scored) if scored else 1.0

    @property
    def runaways(self) -> int:
        """Phrases where the model emitted more words than the reference contained."""
        return sum(1 for r in self.results if r.error is None and r.wer > 1.0)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def rtf(self) -> float:
        """Aggregate real-time factor across all scored phrases."""
        timed = [r for r in self.results if r.error is None and not r.from_cache]
        synth = sum(r.tts_seconds for r in timed)
        audio = sum(r.audio_seconds for r in timed)
        return synth / audio if audio else float("nan")

    @property
    def speech_rtf(self) -> float:
        """Aggregate real-time factor measured against speech rather than total output."""
        timed = [r for r in self.results if r.error is None and not r.from_cache]
        synth = sum(r.tts_seconds for r in timed)
        speech = sum(r.speech_seconds for r in timed)
        return synth / speech if speech else float("nan")

    def by_category(self) -> Dict[str, List[RoundTripResult]]:
        grouped: Dict[str, List[RoundTripResult]] = {}
        for r in self.results:
            grouped.setdefault(r.category, []).append(r)
        return grouped


WER_THRESHOLD = 0.34  # Allow roughly one wrong word in three; this is a smoke test, not an accuracy benchmark


def int_to_words(n: int) -> str:
    """Spell out a non-negative integer below 1000; larger values are read digit by digit."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return (TENS[n // 10] + (" " + ONES[n % 10] if n % 10 else "")).strip()
    if n < 1000:
        rest = n % 100
        return (ONES[n // 100] + " hundred" + (" " + int_to_words(rest) if rest else "")).strip()
    if n < 1_000_000:
        rest = n % 1000
        return (int_to_words(n // 1000) + " thousand"
                + (" " + int_to_words(rest) if rest else "")).strip()
    return " ".join(ONES[int(d)] for d in str(n))


def expand_numbers(token: str) -> str:
    """Expand a numeric token into words, handling decimals and dotted version strings."""
    if "." in token:
        head, *tail = token.split(".")
        parts = [expand_numbers(head)] if head else []
        for chunk in tail:
            parts.append("point")
            # A lone decimal tail is read digit by digit; version segments read as numbers
            # A leading zero is part of the value, so "05" is "zero five" and not "five"
            if len(chunk) > 2 or chunk.startswith("0"):
                parts.append(" ".join(ONES[int(d)] for d in chunk))
            else:
                parts.append(expand_numbers(chunk))
        return " ".join(p for p in parts if p)
    if token.isdigit():
        return int_to_words(int(token))
    return token


def normalize(text: str) -> List[str]:
    """Lowercase, expand symbols and digits, and strip punctuation for a fair word comparison."""
    text = text.lower()
    for symbol, word in SYMBOL_WORDS.items():
        text = text.replace(symbol, word)
    text = text.replace(",", "")  # 1,234 -> 1234 before punctuation stripping
    # "loader.py" and "loader dot py" are the same utterance; so are "16gb" and "16 GB"
    text = RE_INNER_DOT.sub(" dot ", text)
    text = RE_DIGIT_LETTER.sub(" ", text)
    cleaned = "".join(c if (c.isalnum() or c.isspace() or c == ".") else " " for c in text)

    words: List[str] = []
    for word in cleaned.split():
        stripped = word.strip(".")
        if not stripped:
            continue
        if any(c.isdigit() for c in stripped):
            words.extend(expand_numbers(stripped).split())
        else:
            words.append(UNIT_WORDS.get(stripped, stripped))
    return words


def _wer_single(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, divided by reference length."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1] / len(ref)


def word_error_rate(references: List[str], hypothesis: str) -> float:
    """Score against the closest acceptable reading, so valid variants are not penalised."""
    candidates = [r for r in references if r]
    if not candidates:
        return 1.0
    return min(_wer_single(ref, hypothesis) for ref in candidates)


def decode_audio(audio_bytes: bytes, ext: str) -> Tuple[np.ndarray, int]:
    """Decode synthesized audio to a mono float32 array plus its sample rate."""
    if ext == ".wav":
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            if wav.getsampwidth() != 2:
                raise ValueError(f"Expected 16-bit PCM, got {wav.getsampwidth() * 8}-bit")
            rate, channels = wav.getframerate(), wav.getnchannels()
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    else:
        # Cloud providers return mp3; soundfile handles it via libsndfile >= 1.2
        import soundfile as sf
        data, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        samples, channels = (data * 32767).astype(np.int16), (1 if data.ndim == 1 else data.shape[1])

    audio = samples.astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear-interpolation resampler, adequate for speech intelligibility checks."""
    if source_rate == target_rate:
        return audio
    duration = len(audio) / source_rate
    target_positions = np.linspace(0, duration, int(duration * target_rate), endpoint=False)
    return np.interp(target_positions, np.arange(len(audio)) / source_rate, audio).astype(np.float32)


def to_audio_data(audio: np.ndarray, rate: int):
    """Wrap a float32 array in the speech_recognition AudioData that ASR providers expect."""
    import speech_recognition as sr
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    return sr.AudioData(pcm.tobytes(), rate, 2)


def synthesize(provider_name: str, text: str) -> Tuple[bytes, str]:
    """Synthesize text with a talkito TTS provider, returning raw audio bytes and extension."""
    if provider_name == "system":
        return _synthesize_system(text)

    provider = tts.create_tts_provider(provider_name)
    if provider is None:
        raise RuntimeError(f"No provider class registered for '{provider_name}'")
    result = provider.synthesize(text)
    if not result:
        raise RuntimeError(f"{provider_name} returned no audio")
    return result


def _synthesize_system(text: str) -> Tuple[bytes, str]:
    """Capture macOS `say` output to a WAV file so system TTS can join the round trip."""
    import shutil
    import subprocess
    import tempfile

    if not shutil.which("say"):
        raise RuntimeError("system TTS capture is only implemented for macOS `say`")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        # speak_with_default applies no expansion for system engines, so neither does this
        subprocess.run(
            ["say", "-o", path, "--data-format=LEI16@22050", text],
            check=True, capture_output=True,
        )
        with open(path, "rb") as f:
            return f.read(), ".wav"
    finally:
        os.unlink(path)


def make_asr_provider(provider_name: str, model: Optional[str]):
    """Instantiate a talkito ASR provider by name."""
    provider_class = ASR_PROVIDERS.get(provider_name)
    if provider_class is None:
        raise RuntimeError(f"Unknown ASR provider '{provider_name}'")
    return provider_class(ASRConfig(provider=provider_name, model=model))


def spoken_form(phrase: Phrase) -> str:
    """Resolve what actually gets synthesized, running talkito's own preprocessing for raw items."""
    if phrase.raw is None:
        return phrase.text
    text = tts.extract_speakable_text(phrase.raw)
    if isinstance(text, tuple):  # extract_speakable_text is annotated as a pair
        text = text[0]
    return tts.clean_punctuation_sequences(text)


TRIM_LEADING_SILENCE = False  # set by --trim-silence
DURATIONS_ONLY = False  # set by --durations-only
AUDIO_CACHE = None  # set by --audio-cache
STRIP_DISCLOSURE = False  # set by --strip-disclosure


def audio_cache_name(provider: str, text: str, ext: str) -> str:
    """Stable filename for one provider/phrase pair.

    Deliberately not hash(): Python randomises it per process, so the existing --keep-audio names
    differ between runs and could never be looked up again.
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{provider}_{digest}{ext}"


def cached_synthesize(provider: str, text: str, cache_dir: Optional[str]) -> Tuple[bytes, str, bool]:
    """Return audio for one phrase, reusing a cached file when there is one.

    The third value says whether it came from cache. Timing must be taken only from real synthesis:
    a cache hit measures a file read, and averaging that into RTF would report a model as far
    faster than it is.
    """
    if cache_dir:
        for ext in (".wav", ".mp3", ".flac", ".ogg"):
            path = os.path.join(cache_dir, audio_cache_name(provider, text, ext))
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                with open(path, "rb") as handle:
                    return handle.read(), ext, True

    audio_bytes, ext = synthesize(provider, text)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, audio_cache_name(provider, text, ext))
        tmp = path + ".part"
        with open(tmp, "wb") as handle:
            handle.write(audio_bytes)
        os.replace(tmp, path)          # atomic, so an interrupted run leaves no half file to reuse
    return audio_bytes, ext, False


def speech_duration(audio: "np.ndarray", rate: int) -> float:
    """Seconds between the first and last audible sample, so padding is not counted as output.

    Uses the same amplitude threshold as trim_lead, so the two agree about where speech starts.
    A clip with nothing above the threshold returns its full length rather than zero: that is a
    model producing no speech at all, and reporting it as instant would flatter it enormously.
    """
    if audio.size == 0:
        return 0.0
    threshold = max(float(np.abs(audio).max()) * 0.02, 1e-4)
    voiced = np.nonzero(np.abs(audio) > threshold)[0]
    if voiced.size == 0:
        return len(audio) / rate
    return float(voiced[-1] - voiced[0] + 1) / rate


def strip_disclosure(audio: "np.ndarray", rate: int) -> "np.ndarray":
    """Drop a spoken AI disclosure prepended by the runtime, so only the requested text is scored.

    The disclosure is a fixed sentence, separated from the requested text by a pause, so the cut is
    the first silence of at least 150 ms falling in the window where that sentence can end. It is
    removed from scoring only: the audio a caller receives still carries it.
    """
    threshold = max(float(np.abs(audio).max()) * 0.02, 1e-4)
    silent = np.abs(audio) <= threshold
    window = int(rate * 0.15)
    index = 0
    while index < len(silent) - window:
        if silent[index:index + window].all():
            end = index
            while end < len(silent) and silent[end]:
                end += 1
            if 2.5 < index / rate < 6.0:
                return audio[end:]
            index = end
        else:
            index += 1
    return audio


def trim_lead(audio: "np.ndarray", rate: int, keep_ms: int = 100) -> "np.ndarray":
    """Drop silence before the first audible sample, leaving a short lead-in.

    Whisper never returns an empty string: fed silence it emits a word ("You", "Thank you"),
    which scores as an insertion. A model that pads heavily before speaking is therefore
    penalised for audio that sounds perfect, so trimming separates that from real errors.
    """
    if audio.size == 0:
        return audio
    threshold = max(float(np.abs(audio).max()) * 0.02, 1e-4)
    voiced = np.nonzero(np.abs(audio) > threshold)[0]
    if voiced.size == 0:
        return audio
    start = max(0, int(voiced[0]) - int(rate * keep_ms / 1000))
    return audio[start:]


def round_trip(tts_provider: str, asr_provider, asr_name: str, phrase: Phrase,
               keep_dir: Optional[str] = None) -> RoundTripResult:
    """Synthesize one phrase, transcribe it back, and score the result."""
    text = spoken_form(phrase)
    result = RoundTripResult(tts_provider, asr_name, text, "", 1.0, 0.0, 0.0, 0.0,
                             category=phrase.category)

    try:
        if not text.strip():
            raise RuntimeError(f"preprocessing produced no speakable text from {phrase.raw!r}")

        start = time.monotonic()
        audio_bytes, ext, result.from_cache = cached_synthesize(tts_provider, text, AUDIO_CACHE)
        result.tts_seconds = time.monotonic() - start

        audio, rate = decode_audio(audio_bytes, ext)
        result.audio_seconds = len(audio) / rate
        result.speech_seconds = speech_duration(audio, rate)
        if result.audio_seconds < 0.2:
            raise RuntimeError(f"Audio suspiciously short ({result.audio_seconds:.2f}s)")

        if keep_dir:
            name = f"{tts_provider}_{phrase.category}_{abs(hash(text)) % 10000}{ext}"
            with open(os.path.join(keep_dir, name), "wb") as f:
                f.write(audio_bytes)

        if DURATIONS_ONLY:
            # Timing needs no recogniser. Scored as a pass so the run aggregates normally; WER
            # from this mode is meaningless and must not be read as a quality result.
            result.wer = 0.0
            return result

        scored = strip_disclosure(audio, rate) if STRIP_DISCLOSURE else audio
        scored = trim_lead(scored, rate) if TRIM_LEADING_SILENCE else scored
        audio_data = to_audio_data(resample(scored, rate, ASR_SAMPLE_RATE), ASR_SAMPLE_RATE)

        start = time.monotonic()
        result.hypothesis = asr_provider.recognize(audio_data)
        result.asr_seconds = time.monotonic() - start

        # Score against the spoken form plus every accepted alternative reading
        references = [text] + phrase.accept if phrase.raw is None else phrase.accept + [text]
        result.wer = word_error_rate(references, result.hypothesis)
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"

    return result


def available_tts_providers(requested: Optional[List[str]]) -> List[str]:
    """Resolve which TTS providers to exercise, skipping any that are not usable here."""
    if requested:
        return requested
    accessible = tts.check_tts_provider_accessibility()
    return [name for name, info in accessible.items()
            if info.get("available") and name != "polly"]


def preload(tts_name: str, timeout: float = 900.0) -> None:
    """Load a local model up front, as talkito does at startup, so phrase timings exclude the load."""
    if tts_name not in tts.LOCAL_MODEL_PROVIDERS:
        # Benchmark-only providers load lazily on first use, which would otherwise be charged to
        # the first scored phrase. That matters most for slow models measured in small chunks,
        # where one model load can outweigh all the synthesis in the chunk.
        provider = tts.create_tts_provider(tts_name)
        if provider is not None:
            print(f"  warming up {tts_name}...", end="", flush=True)
            start = time.monotonic()
            provider.synthesize("Warm up.")
            print(f" ready in {time.monotonic() - start:.1f}s")
        return

    print(f"  preloading {tts_name}...", end="", flush=True)
    start = time.monotonic()
    tts.preload_local_model(tts_name)

    # preload_local_model returns immediately; wait here for the background thread to finish
    while time.monotonic() - start < timeout:
        with tts._local_model_cache_lock:
            entry = tts._local_models.get((tts_name, tts._model_variant(tts_name)))
            if entry is not None and entry.model is not None:
                print(f" ready in {time.monotonic() - start:.1f}s")
                return
            if entry is not None and entry.error and not entry.loading:
                print(f" FAILED: {entry.error}")
                return
        time.sleep(0.5)
    print(f" timed out after {timeout:.0f}s")


def run(tts_providers: List[str], asr_name: str, asr_model: Optional[str],
        phrases: List[Phrase], keep_dir: Optional[str], repeat: int = 1,
        verbose: bool = True) -> List[ProviderReport]:
    """Run every phrase through every TTS provider and report per-provider accuracy."""
    asr_provider = make_asr_provider(asr_name, asr_model)
    reports = []

    for tts_name in tts_providers:
        print(f"\n=== {tts_name} -> {asr_name} ===")
        preload(tts_name)
        report = ProviderReport(tts_name, asr_name)

        for pass_no in range(repeat):
            if repeat > 1:
                print(f"  --- pass {pass_no + 1}/{repeat} ---")
            for phrase in phrases:
                result = round_trip(tts_name, asr_provider, asr_name, phrase, keep_dir)
                report.results.append(result)

                if result.error:
                    print(f"  ERROR  [{result.category}] {phrase.text or phrase.raw!r}")
                    print(f"         {result.error}")
                elif result.passed and not verbose:
                    continue
                else:
                    mark = "ok  " if result.passed else "FAIL"
                    print(f"  {mark}   WER {result.wer:>5.0%}  rtf {result.rtf:>5.2f}  "
                          f"[{result.category}]")
                    print(f"         ref: {result.reference}")
                    print(f"         hyp: {result.hypothesis}")

        reports.append(report)

    return reports


def print_reports(reports: List[ProviderReport]) -> int:
    """Print the per-category matrix and the summary table; return the number of failures."""
    categories = [c for c in CATEGORIES
                  if any(c in r.by_category() for r in reports)]

    print("\n" + "=" * 78)
    print("PER-CATEGORY  (passed / total, mean WER)")
    print("-" * 78)
    header = f"{'provider':<16}" + "".join(f"{c:>11}" for c in categories)
    print(header)
    for report in reports:
        grouped = report.by_category()
        row = f"{report.tts_provider:<16}"
        for cat in categories:
            items = grouped.get(cat, [])
            if not items:
                row += f"{'-':>11}"
                continue
            ok = sum(1 for r in items if r.passed)
            wer = sum(min(r.wer, 1.0) for r in items if r.error is None)
            wer = wer / max(1, len([r for r in items if r.error is None]))
            row += f"{f'{ok}/{len(items)} {wer:.0%}':>11}"
        print(row)

    print("\n" + "=" * 78)
    print(f"{'TTS provider':<16} {'class':>9} {'passed':>10} {'mean WER':>9} "
          f"{'mean synth':>11} {'RTF':>7} {'runaway':>8}")
    print("-" * 78)
    failed = 0
    for report in reports:
        synth = [r.tts_seconds for r in report.results if r.error is None]
        mean_synth = sum(synth) / len(synth) if synth else 0.0
        architecture = ARCHITECTURE_CLASSES.get(report.tts_provider, UNKNOWN_ARCHITECTURE)
        print(f"{report.tts_provider:<16} {architecture:>9} {report.passed:>4}/{len(report.results):<5} "
              f"{report.mean_wer:>8.0%} {mean_synth:>10.2f}s {report.rtf:>6.2f}x "
              f"{report.runaways:>8}")
        failed += len(report.results) - report.passed
    print("=" * 78)
    return failed


def diagnose_normalization(tts_providers: List[str], asr_name: str, asr_model: Optional[str],
                           repeat: int = 1) -> Dict[str, Dict[str, float]]:
    """Detect whether each provider needs text normalization, by scoring raw vs pre-normalized text.

    This is the portable part of the harness: drop in any new provider and it will say whether that
    model reads digits and acronyms on its own, or needs them spelled out first.
    """
    asr_provider = make_asr_provider(asr_name, asr_model)
    findings: Dict[str, Dict[str, float]] = {}

    print("\n" + "=" * 78)
    print("NORMALIZATION DIAGNOSTIC  (raw input vs pre-normalized input)")
    print("-" * 78)

    for tts_name in tts_providers:
        preload(tts_name)
        sums: Dict[str, List[float]] = {}
        for _ in range(repeat):
            for kind, raw_text, normalized_text in NORMALIZATION_PROBES:
                for label, text in (("raw", raw_text), ("normalized", normalized_text)):
                    # Both forms are scored against the raw wording: we care whether the words
                    # arrive intact, not which spelling was fed to the model
                    phrase = Phrase(text, kind, accept=[raw_text, normalized_text])
                    result = round_trip(tts_name, asr_provider, asr_name, phrase)
                    sums.setdefault(f"{kind}_{label}", []).append(
                        result.wer if result.error is None else 1.0)

        summary = {k: sum(v) / len(v) for k, v in sums.items() if v}
        findings[tts_name] = summary

        print(f"\n  {tts_name}")
        needs = []
        for kind in ("digits", "acronyms"):
            raw_wer = summary.get(f"{kind}_raw")
            norm_wer = summary.get(f"{kind}_normalized")
            if raw_wer is None or norm_wer is None:
                continue
            gap = raw_wer - norm_wer
            flag = "  <-- needs normalization" if gap > NORMALIZATION_GAP_THRESHOLD else ""
            print(f"    {kind:<10} raw {raw_wer:>5.0%}   normalized {norm_wer:>5.0%}   "
                  f"gap {gap:>+5.0%}{flag}")
            if gap > NORMALIZATION_GAP_THRESHOLD:
                needs.append(kind)

        verdict = ("raw-text (BPE-style) input: normalize " + " and ".join(needs) + " before synthesis"
                   if needs else "handles raw text natively (phoneme-style front end)")
        print(f"    verdict: {verdict}")
        findings[tts_name]["needs_normalization"] = float(bool(needs))

    print("=" * 78)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-provider", action="append", dest="tts_providers",
                        help="TTS provider to test (repeatable); defaults to all accessible")
    parser.add_argument("--asr-provider", default="local_whisper", help="ASR provider to transcribe with")
    parser.add_argument("--asr-model", default="base.en", help="ASR model name")
    parser.add_argument("--phrase", action="append", dest="phrases", help="Override test phrases (repeatable)")
    parser.add_argument("--category", action="append", dest="categories",
                        choices=CATEGORIES, help="Only run these corpus categories (repeatable)")
    parser.add_argument("--trim-silence", action="store_true",
                        help="Trim leading silence before transcription, so a model is not "
                             "penalised for silence the recogniser hallucinates words onto.")
    parser.add_argument("--audio-cache", metavar="DIR",
                        help="Reuse synthesized audio from DIR, storing anything missing. Scoring "
                             "and audio measurements can then be repeated without re-synthesizing; "
                             "timing is taken only from phrases actually synthesized.")
    parser.add_argument("--strip-disclosure", action="store_true",
                        help="Drop a spoken AI disclosure the runtime prepends, before scoring "
                             "only. Without it the recogniser transcribes the disclosure as part "
                             "of the utterance and the model is charged for words it was not asked "
                             "to say.")
    parser.add_argument("--durations-only", action="store_true",
                        help="Synthesize and measure durations without transcribing. Recording "
                             "speech time needs no recogniser, so this re-measures timing across "
                             "the corpus at a fraction of the cost of a scored sweep.")
    parser.add_argument("--phrase-index", action="append", dest="phrase_indices", type=int,
                        help="Restrict to these CORPUS indices (repeatable), so a slow provider "
                             "can be measured in chunks that each finish in reasonable time")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Run the corpus N times to expose run-to-run variance")
    parser.add_argument("--quiet", action="store_true", help="Only print failures and the summary")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run the normalization diagnostic (raw vs pre-normalized input)")
    parser.add_argument("--diagnose-only", action="store_true",
                        help="Run only the diagnostic and skip the corpus")
    parser.add_argument("--json", metavar="FILE", help="Write machine-readable results to FILE")
    parser.add_argument("--raw-model", action="store_true",
                        help="Disable talkito's BPE text normalization, to measure a model's "
                             "native handling of digits and acronyms")
    parser.add_argument("--keep-audio", metavar="DIR", help="Write synthesized audio to DIR for inspection")
    parser.add_argument("--neutts-codec", help="Override the codec repo used by the NeuTTS providers")
    parser.add_argument("--neutts-voice", help="Override the NeuTTS reference voice or speaker")
    args = parser.parse_args()

    if args.trim_silence:
        global TRIM_LEADING_SILENCE
        TRIM_LEADING_SILENCE = True
        print("Leading silence trimmed before scoring (--trim-silence)")

    if args.audio_cache:
        global AUDIO_CACHE
        AUDIO_CACHE = args.audio_cache
        os.makedirs(AUDIO_CACHE, exist_ok=True)
        print(f"Audio cache: {AUDIO_CACHE}")

    if args.strip_disclosure:
        global STRIP_DISCLOSURE
        STRIP_DISCLOSURE = True
        print("Spoken AI disclosure removed before scoring (--strip-disclosure)")

    if args.durations_only:
        global DURATIONS_ONLY
        DURATIONS_ONLY = True
        print("Timing only, nothing transcribed (--durations-only): WER from this run is not a "
              "quality measurement")

    if args.phrase_indices:
        selected = [CORPUS[i] for i in args.phrase_indices if 0 <= i < len(CORPUS)]
        CORPUS[:] = selected

    if args.raw_model:
        # Measure the model itself rather than talkito's wiring around it
        tts.normalize_for_bpe_speech = lambda text: text
        print("Provider-side BPE normalization DISABLED (--raw-model)")
    if args.neutts_codec:
        tts.neutts_codec = args.neutts_codec
        print(f"NeuTTS codec: {args.neutts_codec}")
    if args.neutts_voice:
        tts.neutts_voice = tts.neutts2e_voice = args.neutts_voice

    if args.keep_audio:
        os.makedirs(args.keep_audio, exist_ok=True)

    tts_providers = available_tts_providers(args.tts_providers)
    if not tts_providers:
        print("No TTS providers available to test")
        return 1

    if args.phrases:
        phrases = [Phrase(text, "core") for text in args.phrases]
    else:
        phrases = [p for p in CORPUS
                   if not args.categories or p.category in args.categories]

    counts = {}
    for p in phrases:
        counts[p.category] = counts.get(p.category, 0) + 1
    print(f"TTS providers: {', '.join(tts_providers)}")
    print(f"ASR: {args.asr_provider} ({args.asr_model})")
    print(f"Corpus: {len(phrases)} phrases x {args.repeat} pass(es) "
          f"({', '.join(f'{k}={v}' for k, v in sorted(counts.items()))})")

    diagnostics = {}
    reports = []
    failed = 0

    if not args.diagnose_only:
        reports = run(tts_providers, args.asr_provider, args.asr_model, phrases,
                      args.keep_audio, args.repeat, verbose=not args.quiet)
        failed = print_reports(reports)

    if args.diagnose or args.diagnose_only:
        diagnostics = diagnose_normalization(tts_providers, args.asr_provider,
                                             args.asr_model, args.repeat)

    if args.json:
        payload = {
            "asr": {"provider": args.asr_provider, "model": args.asr_model},
            "corpus_size": len(phrases),
            "repeat": args.repeat,
            "providers": {
                r.tts_provider: {
                    "passed": r.passed,
                    "total": len(r.results),
                    "mean_wer": round(r.mean_wer, 4),
                    "rtf": round(r.rtf, 4),
                    # RTF is a ratio of totals, so combining runs needs the totals themselves;
                    # averaging per-run RTF over-weights short phrases, where fixed overhead
                    # dominates and the ratio is worst
                    "synth_seconds": round(sum(x.tts_seconds for x in r.results
                                               if x.error is None), 4),
                    "audio_seconds": round(sum(x.audio_seconds for x in r.results
                                               if x.error is None), 4),
                    # Speech alone, so RTF can be recomputed without crediting a model for the
                    # silence it padded with. Summed for the same reason as the two above.
                    "speech_seconds": round(sum(x.speech_seconds for x in r.results
                                                if x.error is None), 4),
                    # Timing covers only these; anything served from cache measures a file read
                    "synthesized": sum(1 for x in r.results
                                       if x.error is None and not x.from_cache),
                    "by_category": {
                        cat: {
                            "passed": sum(1 for x in items if x.passed),
                            "total": len(items),
                            # Capped exactly as ProviderReport.mean_wer and the printed table
                            # do; raw WER is unbounded, so one runaway otherwise swamps the mean
                            # and the JSON disagrees with what was printed
                            "mean_wer": round(
                                sum(min(x.wer, 1.0) for x in items if x.error is None)
                                / max(1, len([x for x in items if x.error is None])), 4),
                            "runaways": sum(1 for x in items
                                            if x.error is None and x.wer > 1.0),
                        } for cat, items in r.by_category().items()
                    },
                } for r in reports
            },
            "normalization": diagnostics,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nWrote {args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
