#!/usr/bin/env python3

"""Bundle the benchmark into a single zip: the page, the measurements, and the audio.

Run again whenever the measurements change; it rebuilds the archive from scratch each time.
"""

import glob
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # the benchmark repo: FINDINGS.md and the page live here
SAMPLES = os.environ.get("TTS_SAMPLE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "talkito", "benchmark-samples"))

# Rows the page deliberately leaves out; their audio still ships, kept apart so the archive does
# not imply they are part of the comparison
SUPERSEDED_PREFIXES = ("nt-nano-", "nt-air-", "neutts-nano")
# neutts2e and neutts2e-fp32 are earlier names for configurations the nt-2e-* rows already cover,
# so listing both would double-count the same model under two labels
# "kittentts" is the same weights as kittentts-mini -- talkito defaults KITTENTTS_MODEL to
# KittenML/kitten-tts-mini-0.8 -- under a name that does not say which of the four variants it is
SUPERSEDED_EXACT = {"system", "neutts2e", "neutts2e-fp32", "kittentts", "parler-tts"}


def superseded(name):
    return name in SUPERSEDED_EXACT or name.startswith(SUPERSEDED_PREFIXES)


def load_units():
    """Merge every sweep state file into one provider -> totals mapping."""
    merged = {}
    for path in sorted(glob.glob(os.path.join(HERE, "sweep*.json"))) + \
                sorted(glob.glob(os.path.join(HERE, "neutts_matrix*.json"))):
        if "partial" in path:
            continue
        for key, unit in json.load(open(path)).get("units", {}).items():
            if "error" in unit:
                continue
            provider = key.split("|")[0]
            entry = merged.setdefault(provider, {"passed": 0, "total": 0, "wer_sum": 0.0,
                                                 "synth": 0.0, "audio": 0.0, "files": set()})
            entry["passed"] += unit.get("passed", 0)
            entry["total"] += unit.get("total", 0)
            entry["wer_sum"] += unit.get("mean_wer", 0) * unit.get("total", 0)
            entry["synth"] += unit.get("synth_seconds", 0)
            entry["audio"] += unit.get("audio_seconds", 0)
            entry["files"].add(os.path.basename(path))
    return merged


# Licence as declared on each model's own card. Two are non-commercial, which a reader comparing
# scores needs to see next to the score rather than discover afterwards.
LICENSE = {
 "piper": "MIT", "kokoro": "Apache-2.0", "kittentts": "Apache-2.0", "melotts": "MIT",
 "kittentts-micro": "Apache-2.0", "kittentts-nano": "Apache-2.0",
 "kittentts-nano-int8": "Apache-2.0", "kittentts-mini": "Apache-2.0",
 "styletts2": "MIT", "bark": "MIT", "speecht5": "MIT", "csm": "Apache-2.0",
 "chatterbox": "MIT", "chatterbox-q8": "MIT", "chatterbox-q4": "MIT", "chatterbox-turbo": "MIT",
 "cosyvoice3": "Apache-2.0", "cosyvoice3-rl": "Apache-2.0",
 "vibevoice": "MIT", "vibevoice-1.5b": "MIT",
 "qwen3-tts": "Apache-2.0", "bananamind-tts": "Apache-2.0",
 "parler-tts": "Apache-2.0", "zonos": "Apache-2.0", "fastpitch": "CC-BY-4.0",
 "f5-tts": "CC-BY-NC-4.0", "mms-tts": "CC-BY-NC-4.0",
 "nt-2e-q4-metal": "NeuTTS <$5M", "nt-2e-q4-cpu": "NeuTTS <$5M",
 "nt-2e-fp32-cpu": "NeuTTS <$5M", "nt-2e-fp32-mps": "NeuTTS <$5M",
 "omnivoice": "?", "tada-1b": "?", "tada-3b": "?", "dots-tts": "?", "kugelaudio": "?",
}


def results_table(merged, quality, durations):
    """Render every measurement as fixed-width text, so the data is readable without tooling."""
    lines = []
    head = (f"{'configuration':<22}{'licence':<16}{'passed':>9}{'WER':>7}{'PESQ':>7}{'lead-in':>9}"
            f"{'RTF':>8}{'speech-RTF':>12}{'synth/phrase':>14}{'audio/phrase':>14}")
    lines.append(head)
    lines.append("-" * len(head))
    for name in sorted(merged):
        if superseded(name):
            continue
        m = merged[name]
        if not m["total"]:
            continue
        q = quality.get(name, {})
        d = durations.get("providers", {}).get(name, {})
        wer = m["wer_sum"] / m["total"]
        rtf = m["synth"] / m["audio"] if m["audio"] else 0
        srtf = (d["synth_seconds"] / d["speech_seconds"]
                if d.get("speech_seconds") else None)
        lines.append(
            f"{name:<22}{LICENSE.get(name, '?'):<16}{m['passed']:>4}/{m['total']:<4}{wer:>7.1%}"
            f"{q.get('pesq', float('nan')):>7.2f}{q.get('lead', float('nan')):>8.2f}s"
            f"{rtf:>7.2f}x{(f'{srtf:.2f}x' if srtf else '-'):>12}"
            f"{m['synth'] / m['total']:>13.2f}s{m['audio'] / m['total']:>13.2f}s")
    return "\n".join(lines)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "tts-benchmark-export.zip")
    page = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("TTS_BENCHMARK_PAGE", "")

    merged = load_units()
    quality_path = os.path.join(HERE, "audio_quality.json")
    quality = json.load(open(quality_path)) if os.path.exists(quality_path) else {}
    dur_path = os.path.join(HERE, "durations.json")
    durations = json.load(open(dur_path)) if os.path.exists(dur_path) else {"providers": {}}

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    table = results_table(merged, quality, durations)
    included = [n for n in sorted(merged) if not superseded(n) and merged[n]["total"]]

    readme = f"""TTS round-trip benchmark
========================
Exported {stamp}. Apple M2, macOS.

{len(included)} configurations, each synthesizing the same 84-phrase corpus and having its own
output transcribed back by Whisper. Word error rate therefore measures intelligibility end to end,
not transcription accuracy in isolation.

CONTENTS
  benchmark.html        The full comparison, self-contained. Audio samples are embedded as data
                        URIs, so it needs no other files and no network access to view or host.
  FINDINGS.md           What the measurements mean, what was learned, and which to trust.
  data/results.txt      Every measurement as plain text (the table below).
  data/*.json           Raw per-unit state, as produced by the sweep runner. Each unit is one
                        (provider, category) slice, so partial runs are visible.
  samples/              One sample per configuration, all speaking the same sentence.
  samples/superseded/   Audio for rows the comparison excludes, kept for reference.

HOW TO READ IT
  WER           Round-trip word error rate over the corpus, capped at 1.0 per phrase so a single
                runaway cannot swamp the mean.
  PESQ          No-reference perceptual quality (torchaudio SQUIM), scored on trimmed audio.
                Scoring untrimmed rewards models that pad with silence, by up to 0.78.
  lead-in       Silence before the first audible sample. Measured on one clip per configuration,
                so treat it as indicative rather than a mean.
  RTF           Compute seconds per second of audio emitted. The conventional measure, and it
                flatters any model that emits excess silence, because that inflates the divisor.
  speech-RTF    The same ratio against speech alone. Prefer it: it cannot be lowered by padding.
  synth/phrase  Compute seconds per phrase. Every engine speaks the same corpus, so this compares
                directly and is immune to padding.
  licence       As declared on the model's own card. Weights for every row are publicly
                downloadable, so all of these are open weight - but F5-TTS and MMS-TTS are
                CC-BY-NC (non-commercial), and NeuTTS 2E permits commercial use only below $5M
                annual revenue. "?" means no licence could be verified from the card.

BACKENDS THAT DID NOT WORK
  Seven engines were tried and could not be measured, each for a stated reason:

    mini-omni2            the language model runs, then the SNAC decoder exhausts GPU memory
    lfm2-audio            loads, generates frames, cannot find its detokenizer, returns silence
                          while reporting a successful synthesis
    vibevoice-bitnet      the published GGUF was converted without decoder tensors
    qwen3-tts-1.7b-base   crashes the server on a ggml bounds assertion reading a tensor
    orpheus-iq1s / iq1m   built for llama.cpp with unprefixed tensors; this runtime's orpheus
                          backend needs a talker-prefixed conversion, which is a different build
    dia                   output is quantised into ~2.2s blocks, one per dialogue turn, so long
                          phrases truncate mid-sentence. Measured, then excluded: its 78.5% is
                          that cap rather than the model.

  Three more are wired but unmeasured -- moss-tts, moss-tts-local and outetts -- all zero-shot
  cloners awaiting a reference voice, which is the same condition that kept indextts, pocket-tts
  and voxcpm2-tts out of the table until they were given one. A cloner with no reference does not
  fail loudly: it produces fluent, well-formed audio of the wrong words, which reads as a poor
  model rather than a misconfiguration.

  Two recent models could not be attempted at all. Gepard requires CUDA with vLLM and a
  PostgreSQL voice store; Echo requires CUDA with at least 8GB of VRAM. Neither has a CPU or
  Apple Silicon path, so the absence of the newest MeanFlow and continuous-latent work reflects
  the hardware this benchmark runs on rather than a judgement about those models.

  Some rows are partial: a model too slow to finish the 84-phrase corpus within its time budget
  contributes what it managed, and its "passed/total" shows the smaller denominator.

WHAT IS EXCLUDED, AND WHY
  macOS system TTS is not reproducible: macOS resolves a voice to whichever quality tier a given
  machine has downloaded, and Compact, Enhanced and Premium are different models rather than
  bitrates of one. The row described one laptop.

  NeuTTS nano and Air are superseded generations, replaced by 2E in July 2026. Ranking a vendor's
  older models against everyone else's current ones misrepresents the vendor. Their measurements
  remain in data/ and their audio in samples/superseded/.

LIMITATIONS
  One English corpus, one machine, one recogniser family. Word error rate measures intelligibility
  and nothing else - not naturalness, expressiveness or speaker similarity, which are the axes
  LM-backed engines are actually sold on. This benchmark systematically understates them.

  Autoregressive engines are averaged over two passes; feed-forward engines over one. Differences
  under roughly two points are not resolvable at this sample size.

RESULTS
{table}
"""

    staging = os.path.join(REPO, ".export-staging")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(os.path.join(staging, "data"))
    os.makedirs(os.path.join(staging, "samples", "superseded"))

    open(os.path.join(staging, "README.txt"), "w").write(readme)
    open(os.path.join(staging, "data", "results.txt"), "w").write(table + "\n")
    # The numbers without the reasoning are hard to act on, so the findings ship alongside them
    findings = os.path.join(REPO, "FINDINGS.md")
    if os.path.isfile(findings):
        shutil.copy(findings, os.path.join(staging, "FINDINGS.md"))
    for path in glob.glob(os.path.join(HERE, "*.json")):
        shutil.copy(path, os.path.join(staging, "data", os.path.basename(path)))
    if page and os.path.isfile(page):
        shutil.copy(page, os.path.join(staging, "benchmark.html"))
    else:
        print("  WARNING: no page supplied, benchmark.html omitted")

    # Ship audio only for configurations the data covers. A sample with no row invites the reader
    # to treat it as a result, and several exist here from exploratory work that was never scored.
    scored = set(merged)
    audio = 0
    for path in sorted(glob.glob(os.path.join(SAMPLES, "mp3", "*.mp3"))):
        name = os.path.basename(path)[:-4]
        if name not in scored:
            continue
        target = "superseded" if superseded(name) else ""
        shutil.copy(path, os.path.join(staging, "samples", target, os.path.basename(path)))
        audio += 1

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(staging):
            for name in sorted(files):
                full = os.path.join(root, name)
                archive.write(full, os.path.relpath(full, staging))
    shutil.rmtree(staging, ignore_errors=True)

    size = os.path.getsize(out_path) / 1e6
    print(f"  {out_path}  ({size:.1f} MB)")
    print(f"  {len(included)} configurations, {audio} audio samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
