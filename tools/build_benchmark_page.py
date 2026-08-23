#!/usr/bin/env python3
"""Regenerates the benchmark table in the published page from the sweep state files."""

import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmark_sweep as sw

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tts_benchmark.html"
import glob as _glob
# Every pass contributes its units. Totals are summed and WER weighted by sample count, so
# adding a second pass averages the two rather than replacing the first.
STATES = sorted(f for f in _glob.glob(os.path.join(HERE, "sweep_r*.json")) +
                              _glob.glob(os.path.join(HERE, "sweep_candidates.json")) +
                              _glob.glob(os.path.join(HERE, "sweep_wide.json")) +
                              _glob.glob(os.path.join(HERE, "neutts_matrix_r*.json"))
                if "partial" not in f and "_r3" not in f)

# frame rate (tokens per second of audio), release date, and the note for each configuration.
# Frame rates are measured or read from GGUF metadata; "-" means the model declares none.
META = {
 "piper":            ("-",     "2023-06", "No torch at all. Peaks at full scale - check for clipping."),
 "kittentts":        ("-",     "2026-02", "Identical duration every run; only numerical jitter."),
 "kittentts-micro":  ("-",     "2026-02", "The most accurate KittenTTS here at 43 MB. Non-autoregressive, so no sampling and no runaways."),
 "kittentts-nano":   ("-",     "2026-02", "The fp32 middle size of KittenTTS's four. Not the default - talkito loads mini."),
 "kittentts-nano-int8": ("-",  "2026-02", "int8 of the same nano weights at 26 MB. Non-autoregressive, so quantisation cannot cause the sampling failures seen in the LM models."),
 "kittentts-mini":   ("-",     "2026-02", "The largest KittenTTS, and what talkito loads by default."),
 "chatterbox":       ("25 Hz", "2025-04", "MIT, and the family's best PESQ at 3.61 - but turbo is more accurate and 4x faster. PerTh watermarking; own venv."),
 "kokoro":           ("-",     "2024-12", "Deterministic timing. Weakest on very short utterances."),
 "system":           ("-",     "-",       "Ships with macOS. Nothing to install."),
 "fastpitch":        ("-",     "2021-06", "Non-autoregressive, 120 MB. Output carries a spoken AI disclosure, removed before scoring."),
 "f5-tts":           ("-",     "2024-10", "Flow matching: non-autoregressive, but iterative."),
 "bananamind-tts":   ("-",     "-",       "38 MB. Output carries a spoken AI disclosure, removed before scoring."),
 "cosyvoice3-rl":    ("50 Hz", "2025-01", "CosyVoice 3 with RL post-training, a third the size of the base LLM."),
 "parler-tts":       ("-",     "2024-08", "Full corpus at last, and the worst WER here at 59%. Unusually recogniser-sensitive: 41% on whisper small."),
 "chatterbox-turbo": ("50 Hz", "2025-08", "Distilled chatterbox, and joint-best accuracy here at 4%. Base q8 measures 2.22x."),
 "chatterbox-q8":    ("25 Hz", "2025-04", "Quantization costs 4 points, refuting the ASR-identical claim."),
 "vibevoice":        ("-",     "2025-08", "Small deployed footprint for an LM engine, at 686 MB. 1.02B despite a 0.5b filename."),
 "chatterbox-q4":    ("25 Hz", "2025-04", "6 points worse than fp32 and no faster than q8."),
 "styletts2":        ("-",     "~2023",   "Phoneme model that still needs digit normalization (+20% gap)."),
 "melotts":          ("-",     "2024-02", "198 MB of weights in a 3 GB install; 774 MB is a Japanese dictionary."),
 "cosyvoice3":       ("-",     "2025-07", "9 languages plus 18 Chinese dialects; 8-voice baked bank."),
 "csm":              ("12.5 Hz","~2025",  "1.78B as served; the CSM-1B name covers the backbone only."),
 "speecht5":         ("-",     "2023-02", "16 kHz output - duller than everything else regardless of accuracy."),
 "bark":             ("-",     "2023-04", "Unusable. Invents lead-in words never present in the input."),
 "qwen3-tts":        ("12 Hz", "2026-01", "Alibaba's 0.6B at a 12 Hz codec - the direct architectural rival."),
 "nt-2e-q8-cpu":     ("50 Hz", "2026-07", "The most accurate 2E configuration at 5%, and absent from this table until now - Neuphonic publish a Q8_0 that was never measured."),
 "nt-2e-q8-metal":   ("50 Hz", "2026-07", "The fastest 2E configuration at 0.46x, but 3.5 points behind the same weights on CPU."),
 "nt-2e-q4-metal":   ("50 Hz", "2026-07", "0.48x, just behind q8-metal. Reachable only via an undocumented split-device call."),
 "nt-2e-q4-cpu":     ("50 Hz", "2026-07", "What you get by default. ~1 point behind fp32 - an excellent conversion."),
 "nt-2e-fp32-cpu":   ("50 Hz", "2026-07", "4.4x slower than the same weights quantized on Metal."),
 "nt-2e-fp32-mps":   ("50 Hz", "2026-07", "A dead heat with CPU: the path PR #123 targeted has nothing to win."),
 "nt-nano-fp32-cpu": ("50 Hz", "2025-11", "Accurate at fp32 - and the model whose quantizations are broken."),
 "nt-nano-fp32-mps": ("50 Hz", "2025-11", "The one torch config where MPS helps, 1.4x over CPU."),
 "nt-nano-q4-metal": ("50 Hz", "2025-11", "7 points worse than the fp32 it was converted from."),
 "nt-nano-q4-cpu":   ("50 Hz", "2025-11", "Same damage on CPU as Metal - the fault is the conversion."),
 "nt-nano-q8-cpu":   ("50 Hz", "2025-11", "Higher precision than q4, worse result. Reproduced five times."),
 "nt-nano-q8-metal": ("50 Hz", "2025-11", "Worst NeuTTS result; the q8 build looks mis-converted."),
 "nt-air-q4-metal":  ("50 Hz", "2025-09", "Neuphonic's best. Beats Qwen3-TTS on both axes at 4x the frame rate."),
 "nt-air-q4-cpu":    ("50 Hz", "2025-09", "Air's quantizations are clean - no degradation, no q8/q4 inversion."),
 "nt-air-q8-cpu":    ("50 Hz", "2025-09", "q8 behaves correctly here, unlike nano's."),
 "nt-air-q8-metal":  ("50 Hz", "2025-09", "Confirms the nano defect is specific to nano's GGUF build."),
 "nt-air-fp32-cpu":  ("50 Hz", "2025-09", "7x slower than its own q4 for no measurable quality gain."),
 "nt-air-fp32-mps":  ("50 Hz", "2025-09", "Torch on MPS again fails to beat CPU."),
 "omnivoice":        ("-",     "-",       "Speech is fluent; no upstream model card found, so licence is unverified."),
 "vibevoice-1.5b":   ("-",     "2025-08", "Larger than the 1.02B vibevoice and markedly worse."),
 "zonos":            ("-",     "2025-02", "Second-best PESQ here after piper, but 8x real time."),
 "mms-tts":          ("-",     "2023-05", "Meta MMS: a VITS per language, 1100+ of them, and one of eighteen rows run outside the C++ runtime."),
 "tada-1b":           ("-",     "2025-09", "1B multilingual. Middling accuracy, unremarkable throughout."),
 "tada-3b":           ("-",     "2025-09", "3B sibling of tada-1b: three times the weights, three points better."),
 "dots-tts":          ("-",     "2025-06", "4.4 GB f16 and 11x real time, for 6% WER - mid-pack accuracy."),
 "pocket-tts":       ("-",     "2026-01", "Kyutai Pocket TTS, 100M parameters. Faster than real time in 457 MB of peak RSS. Clones from the piper sample; with no reference it scores 0/70 with 42 runaways."),
 "voxcpm2-tts":      ("-",     "2024-12",       "OpenBMB VoxCPM on MiniCPM-4: tokenizer-free, modelling continuous speech space rather than discrete acoustic tokens. Upstream claim is 0.17x on GPU; this is q4_k on an M2. Clones from the piper sample."),
 "qwen3-tts-vd":     ("12 Hz", "2026-01", "Qwen3-TTS VoiceDesign at 1.7B: the speaker comes from a written description, not a reference clip."),
 "orpheus-q4":       ("-",     "2025-03", "Llama-3.2-3B finetune over SNAC: the largest LM backbone here, and the slowest at 8.1x real time."),
 "orpheus-q8":       ("-",     "2025-03", "The same 3B at q8. Only three quants are published and none reach real time."),
 "supertonic":       ("-",     "2026-03", "99M params across four fp32 ONNX graphs, so 380 MB on disk despite the parameter count. Native 44.1 kHz, resampled to 24 kHz here. Weights are OpenRAIL-M, not the MIT of the sample code."),
 "xtts":             ("-",     "2023-11", "Coqui XTTS v2. Clones from a reference clip; scored against a piper sample as reference."),
}
# Licence as declared on each model's own card, not inferred from the runtime that serves it. Two
# are restrictive in ways a reader choosing a model needs to see rather than discover later.
LICENSE = {
 "piper": "MIT", "kokoro": "Apache-2.0", "kittentts": "Apache-2.0", "melotts": "MIT",
 "kittentts-micro": "Apache-2.0", "kittentts-nano": "Apache-2.0", "kittentts-mini": "Apache-2.0",
 "kittentts-nano-int8": "Apache-2.0",
 "styletts2": "MIT", "bark": "MIT", "speecht5": "MIT", "csm": "Apache-2.0",
 "chatterbox": "MIT", "chatterbox-q8": "MIT", "chatterbox-q4": "MIT", "chatterbox-turbo": "MIT",
 "cosyvoice3": "Apache-2.0", "cosyvoice3-rl": "Apache-2.0",
 "vibevoice": "MIT", "vibevoice-1.5b": "MIT",
 "qwen3-tts": "Apache-2.0", "bananamind-tts": "Apache-2.0",
 "parler-tts": "Apache-2.0", "zonos": "Apache-2.0",
 "fastpitch": "CC-BY-4.0",
 # Non-commercial. The best-scoring model here that cannot be shipped in a product.
 "f5-tts": "CC-BY-NC-4.0", "xtts": "CPML (non-commercial)",
 # Code is MIT but the weights are not: OpenRAIL-M carries use restrictions, which is the half
 # that matters to someone choosing a model from this table.
 "supertonic": "OpenRAIL-M",
 "indextts": "Apache-2.0", "pocket-tts": "CC-BY-4.0", "voxcpm2-tts": "Apache-2.0",
 "orpheus-q4": "Apache-2.0", "orpheus-q8": "Apache-2.0",
 "qwen3-tts-vd": "Apache-2.0",
 # Custom "NeuTTS License": free for research, commercial use permitted only below $5M annual
 # revenue. Air was Apache-2.0; 2E is not.
 "nt-2e-q8-cpu": "NeuTTS, under $5M", "nt-2e-q8-metal": "NeuTTS, under $5M",
 "nt-2e-q4-metal": "NeuTTS, under $5M", "nt-2e-q4-cpu": "NeuTTS, under $5M",
 "nt-2e-fp32-cpu": "NeuTTS, under $5M", "nt-2e-fp32-mps": "NeuTTS, under $5M",
 "omnivoice": "?", "mms-tts": "CC-BY-NC-4.0",
 # tada and dots-tts publish no licence I could verify from their cards
 "tada-1b": "?", "tada-3b": "?", "dots-tts": "?",
}

LIBS = {"piper":(391,338),"kittentts":(764,608),"chatterbox":(1200,3573),"kokoro":(1147,1650),"system":(0,0),"chatterbox-q8":(18,1443),"vibevoice":(18,1333),"chatterbox-q4":(18,1181),"styletts2":(1100,1828),"melotts":(2800,587),"cosyvoice3":(18,999),"csm":(18,2459),"speecht5":(18,474),"bark":(18,624),"qwen3-tts":(18,2211)}
# Peak RSS in MB. Entries added by measure_gaps are ESTIMATES: the run includes the
# recogniser sharing the process, and the ~1055 MB subtracted for it varies by provider
# (718 MB on mms-tts), so they carry roughly +/-200 MB.
# Peak RSS in MB. Values for providers measured by measure_rss.py are taken with
# --durations-only, so no recogniser is loaded; earlier figures in this table were
# measured differently and are left as they were.
# Peak RSS in MB. Providers measured by measure_rss_rusage.py use the kernel's own
# high-water mark (/usr/bin/time -l) with --durations-only, so no recogniser is loaded
# and no sampling can miss an allocation spike. Older entries were measured another way
# and are not strictly comparable.
NEU_RSS = {"bananamind-tts":246,"bark":635,"chatterbox-turbo":1724,"cosyvoice3-rl":1023,"dots-tts":5056,"f5-tts":1169,"fastpitch":376,"kittentts-micro":681,"kittentts-mini":785,"kittentts-nano":616,"kittentts-nano-int8":589,"mms-tts":742,"nt-2e-fp32-cpu":2025,"nt-2e-fp32-mps":1116,"nt-2e-q4-cpu":1862,"nt-2e-q4-metal":1734,"nt-2e-q8-cpu":2048,"nt-2e-q8-metal":1895,"nt-air-fp32-cpu":0,"nt-air-fp32-mps":0,"nt-air-q4-cpu":1988,"nt-air-q4-metal":1895,"nt-air-q8-cpu":2388,"nt-air-q8-metal":2058,"nt-nano-fp32-cpu":2210,"nt-nano-fp32-mps":2253,"nt-nano-q4-cpu":1895,"nt-nano-q4-metal":1542,"nt-nano-q8-cpu":1886,"nt-nano-q8-metal":1598,"omnivoice":2275,"orpheus-q4":2762,"parler-tts":2285,"qwen3-tts-vd":3368,"supertonic":882,"tada-1b":3102,"tada-3b":8559,"vibevoice-1.5b":2474,"xtts":4075,"zonos":2495}

def mb(v):
    if not v: return "-"
    return f"{v/1024:.1f} GB" if v >= 1024 else f"{v} MB"

SAMPLES = os.environ.get(
    "TTS_SAMPLE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "talkito", "benchmark-samples", "mp3"))


def sample_uri(name):
    """Inline the sample as a data URI; the page CSP blocks any external audio host."""
    import base64
    f = os.path.join(SAMPLES, f"{name}.mp3")
    if not os.path.isfile(f): return ""
    return "data:audio/mpeg;base64," + base64.b64encode(open(f, "rb").read()).decode()


def sortnum(text):
    """Numeric sort key for a rendered cell, so the browser never parses '2.2 GB' itself."""
    if not text or text == "-": return -1
    t = str(text).replace(",", "")
    m = re.match(r"^~?(\d{4})-?(\d{2})?$", t)          # release dates
    if m: return int(m.group(1)) * 100 + int(m.group(2) or 0)
    m = re.match(r"^([\d.]+)\s*([MB])$", t)             # params: 16M / 0.75B
    if m: return float(m.group(1)) * (1e3 if m.group(2) == "B" else 1)
    m = re.match(r"^([\d.]+)\s*(MB|GB)$", t)            # sizes
    if m: return float(m.group(1)) * (1024 if m.group(2) == "GB" else 1)
    m = re.match(r"^(\d+)/(\d+)$", t)                   # passed: rank by pass rate
    if m: return int(m.group(1)) / int(m.group(2))
    m = re.match(r"^([\d.]+)", t)                       # 50 Hz, 4%, 0.50x
    return float(m.group(1)) if m else -1


def sev(kind, v):
    # Test the worst band first: ordering these the other way round makes "bad" unreachable
    if kind == "wer":  return "good" if v < .08 else ("bad" if v >= .40 else ("warn" if v >= .20 else ""))
    if kind == "per":  return "good" if v <= 0.6 else ("bad" if v > 5.0 else "")
    # Only excess is flagged: unusually short output can mean truncation, so it is not "good"
    if kind == "aper": return "bad" if v > 5.5 else ""
    # Silence before the first audible sample. Anything past a second is audible dead air, and it
    # also inflates RTF and confuses ASR, so it is flagged even though it is not a quality defect
    if kind == "lead": return "good" if v <= .15 else ("bad" if v > 1.5 else ("warn" if v > .5 else ""))
    # RTF is synth/audio, so lower is better. Over 1.0x is slower than real time and cannot keep
    # up with continuous speech; over 2.0x is unusable for anything interactive.
    return "good" if v <= .20 else ("bad" if v > 2.0 else ("warn" if v > 1.0 else ""))

# Merge every pass first, then emit one row per provider. Keeping the passes separate here
# would produce a duplicate row per provider instead of an average across them.
SQUIM = os.path.join(HERE, "squim_scores.json")
squim = json.load(open(SQUIM)) if os.path.exists(SQUIM) else {}

# Lead-in silence, and PESQ recomputed on trimmed audio. Scoring the untrimmed clip lets padding
# pull PESQ toward the ceiling - measured at up to +0.78 - so the models that pad the most were
# being rewarded for it. Trimmed scores are the quality of the speech alone.
AQ = os.path.join(HERE, "audio_quality.json")
aq = json.load(open(AQ)) if os.path.exists(AQ) else {}

# Real-time factor measured against speech rather than total output, from a timing-only pass. RTF
# divides by everything a model emits, so silence inflates the divisor; this says what the ratio
# would be without it. Shown on hover rather than as a column: across these rows the correction is
# immaterial, and only one of them changes meaning.
DUR = os.path.join(HERE, "durations.json")
dur = json.load(open(DUR)).get("providers", {}) if os.path.exists(DUR) else {}

MEDIUM = os.path.join(HERE, "sweep_medium.json")
med = {}
if os.path.exists(MEDIUM):
    for k, v in json.load(open(MEDIUM))["units"].items():
        if "error" in v: continue
        e = med.setdefault(k.split("|")[0], [0, 0.0])
        e[0] += v["total"]; e[1] += v["mean_wer"] * v["total"]

by_provider = {}
for path in STATES:
    if not os.path.exists(path): continue
    for k, v in json.load(open(path))["units"].items():
        if "error" in v: continue
        by_provider.setdefault(k.split("|")[0], []).append(v)

# Excluded from the table rather than deleted: the measurements are real, but the rows do not
# belong in a comparison of current models.
#
# "system" is not reproducible. macOS resolves a voice to whichever quality tier that machine has
# downloaded - Compact, Enhanced and Premium are different models, not bitrates of one - so it
# identifies a per-device configuration rather than a model.
#
# NeuTTS nano (Nov 2025) and Air (Sep 2025) are superseded generations; 2E (Jul 2026) replaces
# them. Ranking a vendor's older models against everyone else's current ones misrepresents the
# vendor, and the quantization and padding faults found in the nano GGUFs are properly a report to
# Neuphonic rather than a row in a cross-vendor table. Their measurements stay in the raw data.
#
# Plain "kittentts" is the same weights as kittentts-mini: talkito defaults KITTENTTS_MODEL to
# KittenML/kitten-tts-mini-0.8 (tts.py:210), so the row is that checkpoint under a name that does
# not say which of the four it is. The named variants measure it explicitly.
#
# parler-tts does not synthesize the text it is given. Transcribing its clips shows fluent English
# unrelated to the reference -- one longform phrase came back as "I don't have a major fleet in the
# world" repeated fourteen times -- while short phrases mostly come through verbatim. A row scoring
# 59% implies a model that speaks badly; this one is not attempting the sentence, so its WER is not
# the same measurement as everyone else's and does not belong beside them. Its measurements stay in
# the raw data and its audio still ships.
# dia is excluded on a measurement fault, not a model one. Its output is quantised into ~2.2s
# blocks -- one per dialogue turn -- and a turn is capped there however much text it holds, so
# 198 characters of input yields 4.4s of audio where the text is roughly 12s of speech. Long
# phrases truncate mid-word ("Running tests now, please wait a moment" comes back as "Running
# test now, please"), short ones fill the block with silence or a repeated token. The corpus
# score of 78.5% measures that cap. Raising --max-new-tokens from 512 to 2048 does not move it.
EXCLUDED = {"system", "kittentts", "parler-tts", "dia"}
EXCLUDED_PREFIXES = ("nt-nano-", "nt-air-", "neutts-nano")

# Rows whose RTF comes from the timing-only pass in durations.json instead of the scored sweep.
#
# Deliberately not "everything durations.json covers". Preferring it everywhere restates RTF for
# sixteen other rows, 15 of 20 by more than 10% -- and in both directions, cosyvoice3 reading 34%
# slower on the timing pass while styletts2 reads 30% faster. A correction that changes sign
# between providers is measuring run-to-run variance, not removing a bias, so applying it wholesale
# would trade one set of noisy numbers for another while looking like a fix.
#
# These four are different: their sweep ran while a PESQ job held a recogniser and a second model
# server on the machine, they all read about 2x slow, and the re-measurement was taken with the
# process table verified empty. A known cause and a consistent direction is what justifies the
# override -- the rest of the table stays on the sweep until it has the same.
RETIMED = {"kittentts-micro", "kittentts-nano-int8", "kittentts-nano", "kittentts-mini"}


def superseded(provider):
    """True for a model generation the vendor has since replaced."""
    return provider in EXCLUDED or provider.startswith(EXCLUDED_PREFIXES)

rows = []
if True:
    for p in sorted(by_provider):
        if p not in META or superseded(p): continue
        us = by_provider[p]
        tot = sum(u["total"] for u in us)
        if not tot: continue
        passed = sum(u["passed"] for u in us)
        wer = sum(u["mean_wer"]*u["total"] for u in us)/tot
        aud = sum(u.get("audio_seconds",0) for u in us); syn = sum(u.get("synth_seconds",0) for u in us)
        if p in RETIMED and dur.get(p, {}).get("audio_seconds"):
            aud, syn = dur[p]["audio_seconds"], dur[p]["synth_seconds"]
        rtf = syn/aud if aud else 0
        # Seconds of compute per phrase. Every model speaks the same corpus, so this compares
        # directly - and unlike RTF it cannot be lowered by padding the output with silence.
        per = syn/tot if tot else 0
        # Audio produced per phrase. Identical text for every engine, so a high value means the
        # model is emitting more than the sentence needs - padding, or failing to stop.
        aper = aud/tot if tot else 0
        cls, params, size = sw.PROVIDER_META.get(p, ("?","-",0))
        libs, rss = LIBS.get(p, (1453 if p.startswith("nt-") else 0, NEU_RSS.get(p,0)))
        if p.startswith("nt-"): libs = 1434 if "fp32" in p else 1453
        fr, rel, note = META[p]
        license_name = LICENSE.get(p, "?")
        quality = aq.get(p, {})
        timing = dur.get(p, {})
        speech_rtf = speech_share = None
        if timing.get("speech_seconds") and timing.get("audio_seconds"):
            speech_rtf = timing["synth_seconds"] / timing["speech_seconds"]
            speech_share = timing["speech_seconds"] / timing["audio_seconds"]
        sq = quality.get("pesq", squim.get(p))
        lead = quality.get("lead")
        m = med.get(p)
        wer_med = (m[1] / m[0]) if m and m[0] else None
        partial = tot < 84
        rows.append(dict(name=p, cls=cls, params=params, fr=fr, rel=rel, size=size, libs=libs,
                         total=size+libs, rss=rss, passed=passed, n=tot, wer=wer, rtf=rtf, per=per, aper=aper, wer_med=wer_med, squim=sq,
                         lead=lead, license=license_name,
                         speech_rtf=speech_rtf, speech_share=speech_share,
                         note=note, partial=partial, neu=p.startswith("nt-")))

rows.sort(key=lambda r: (r["wer"], r["rtf"]))
out = []
for r in rows:
    tr = ' class="neu"' if r["neu"] else ''
    tag = ' <em>(partial)</em>' if r["partial"] else ''
    cells = [mb(r['size']), mb(r['libs']), mb(r['total']), mb(r['rss'])]
    uri = sample_uri(r['name'])
    medtxt = f"{r['wer_med']:.0%}" if r['wer_med'] is not None else "&middot;"
    medcls = sev('wer', r['wer_med']) if r['wer_med'] is not None else "pending"
    medsort = f"{r['wer_med']:.4f}" if r['wer_med'] is not None else "99"
    sqtxt = f"{r['squim']:.2f}" if r['squim'] is not None else "&middot;"
    sqcls = ("good" if r['squim'] >= 3.8 else "bad" if r['squim'] < 3.0 else "") if r['squim'] is not None else "pending"
    sqsort = f"{r['squim']:.2f}" if r['squim'] is not None else "-1"
    # Flag anything that is not a plain permissive licence, since that is the case where a good
    # score is not something a reader can actually act on
    liccls = "" if r['license'] in ("MIT", "Apache-2.0", "CC-BY-4.0") else " restricted"
    leadtxt = f"{r['lead']:.2f}s" if r['lead'] is not None else "&middot;"
    leadcls = sev('lead', r['lead']) if r['lead'] is not None else "pending"
    leadsort = f"{r['lead']:.2f}" if r['lead'] is not None else "99"
    if r['speech_rtf'] is not None:
        rtfcls = " measured"
        rtftip = (f' title="{r["speech_rtf"]:.2f}x against speech alone &#10;'
                  f'{r["speech_share"]:.0%} of this model&#39;s output is speech, the rest silence"')
    else:
        rtfcls, rtftip = "", ""
    play = (f'<button class="play" data-src="{uri}" aria-label="Play {r["name"]} sample">'
            f'<span aria-hidden="true">&#9654;</span></button>') if uri else \
           '<span class="play noplay" aria-hidden="true"></span>' 
    out.append(f"""      <tr{tr}>
        <td class="l name" data-sort="{r['name']}">{play}{r['name']}{tag}</td>
        <td class="l" data-sort="{r['cls']}"><span class="cls">{r['cls']}</span></td>
        <td class="l lic{liccls}" data-sort="{r['license']}">{r['license']}</td>
        <td class="num" data-sort="{sortnum(r['params'])}">{r['params']}</td>
        <td class="num" data-sort="{sortnum(r['fr'])}">{r['fr']}</td>
        <td class="num" data-sort="{sortnum(r['rel'])}">{r['rel']}</td>
        <td class="num" data-sort="{sortnum(cells[0])}">{cells[0]}</td>
        <td class="num" data-sort="{sortnum(cells[1])}">{cells[1]}</td>
        <td class="num" data-sort="{sortnum(cells[2])}">{cells[2]}</td>
        <td class="num" data-sort="{sortnum(cells[3])}">{cells[3]}</td>
        <td class="num" data-sort="{r['passed']/r['n']:.4f}">{r['passed']}/{r['n']}</td>
        <td class="num {sev('wer',r['wer'])}" data-sort="{r['wer']:.4f}">{r['wer']:.0%}</td>
        <td class="num {medcls}" data-sort="{medsort}">{medtxt}</td>
        <td class="num {sqcls}" data-sort="{sqsort}">{sqtxt}</td>
        <td class="num {leadcls}" data-sort="{leadsort}">{leadtxt}</td>
        <td class="num {sev('rtf',r['rtf'])}{rtfcls}"{rtftip} data-sort="{r['rtf']:.4f}">{r['rtf']:.2f}x</td>
        <td class="num {sev('per',r['per'])}" data-sort="{r['per']:.4f}">{r['per']:.2f}s</td>
        <td class="num {sev('aper',r['aper'])}" data-sort="{r['aper']:.4f}">{r['aper']:.2f}s</td>
        <td class="note" data-sort="{r['note'][:40]}">{r['note']}</td>
      </tr>""")

cols = [("Sample · Configuration","l"),("Class","l"),("Licence","l"),("Params",""),("Frame rate",""),("Released",""),
        ("Weights",""),("Libs",""),("Total disk",""),("Peak RSS",""),("Passed",""),
        ("WER base",""),("WER med",""),("PESQ",""),("Lead-in",""),("RTF",""),("Avg synth",""),("Avg audio",""),("Notes","l")]
head = "      <tr>\n" + "\n".join(
    f'        <th class="{c}" tabindex="0" role="columnheader" aria-sort="none">{n}</th>'
    for n, c in cols) + "\n      </tr>"
page = open(PAGE).read()
page = re.sub(r'<thead>.*?</thead>', f'<thead>\n{head}\n    </thead>', page, flags=re.S)
page = re.sub(r'<tbody>.*?</tbody>', '<tbody>\n' + "\n".join(out) + '\n    </tbody>', page, flags=re.S)
page = re.sub(r'<title>[^<]*</title>',
              f'<title>TTS Engine Benchmark \u2014 {len(rows)} configurations</title>', page)
page = re.sub(r'<h1>[^<]*</h1>', f'<h1>{len(rows)} text-to-speech configurations, measured end to end</h1>', page)
open(PAGE, "w").write(page)
print(f"  wrote {len(rows)} rows")

def _warn_stale_notes(rows):
    """Warn when a note ranks itself against the table, or quotes a figure its own cell contradicts.

    audit_notes.py has existed for a while and does this better, but it has to be remembered, and
    twice it was not: zonos kept "second-best PESQ here after piper" through the addition of a row
    that beat it, and that note was written during the pass meant to remove exactly this. A check
    that only runs when someone thinks to run it does not prevent the failure it was built for.

    Warnings, never an error. A note legitimately citing a neighbouring row -- chatterbox-turbo
    quoting the q8 RTF -- is correct and would fail any automatic rule.
    """
    import re as _re
    ranks = _re.compile(r"\b(best|worst|fastest|slowest|largest|smallest|highest|lowest|only|"
                        r"second-best|most|least)\b", _re.I)
    issues = []
    for r in rows:
        note = META.get(r["name"], ("", "", ""))[2]
        for word in sorted({m.group(0).lower() for m in ranks.finditer(note)}):
            issues.append(f"{r['name']}: ranks itself ({word!r}) - a new row can falsify this")
        for figure in _re.findall(r"(\d+(?:\.\d+)?)\s*x\b", note):
            if r["rtf"] and abs(float(figure) - r["rtf"]) > 0.05 * max(1.0, r["rtf"]):
                issues.append(f"{r['name']}: note says {figure}x, its cell is {r['rtf']:.2f}x")
    if issues:
        print(f"  {len(issues)} note(s) to check:")
        for line in issues:
            print(f"    {line}")


_warn_stale_notes(rows)
