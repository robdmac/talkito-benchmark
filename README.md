# talkito-benchmark

Round-trip TTS benchmark: synthesize a fixed 84-phrase corpus with each engine, transcribe the
result with Whisper, and score what came back. Word error rate therefore measures intelligibility
end to end rather than transcription accuracy in isolation.

This repo holds only what is needed to produce the benchmark. No results, no findings, no page --
those are outputs, and everything here regenerates them from nothing.

    python3 tools/benchmark.py

Everything is ticked by default. Toggle with numbers, ranges (`4-9`) or substrings (`kitten` hits
all four), then press Enter. Each model prints its own line as it finishes and the run ends with an
`open` command for the page. A full sweep is around 11 hours; a first run on a clean machine is
longer, because each engine downloads its weights.

    python3 tools/benchmark.py --only piper,kokoro    # no picker
    python3 tools/benchmark.py --yes                  # everything, no picker
    python3 tools/benchmark.py --keep                 # add to existing measurements

## Finding talkito

The harness that does the synthesizing lives in talkito's `tests-repo` submodule, and the providers
import `talkito` directly, so this repo needs to know where a checkout is. `tools/paths.py` looks in
`TALKITO_ROOT`, then `../..` (this repo as a submodule inside talkito), then `../talkito` (the two
side by side). It checks for the harness file rather than a directory of the right name, because
the failure otherwise surfaces as an import error three subprocesses deep.

Setting `TALKITO_ROOT` to something that does not resolve is an error rather than a hint. Falling
back to the search would quietly measure a different checkout than the one asked for, and the
results would look completely normal.

Rebuilding the page needs none of this. It reads only stored measurements, so results can always be
republished from data alone.

## What each piece does

    tools/benchmark.py              the picker and the whole run
    tools/run_benchmark_sweep.py    one resumable unit per provider|category|chunk
    tools/benchmark_providers.py    how each engine is actually called
    tools/tts_worker.py             runs an engine in its own venv, for conflicting pins
    tools/build_benchmark_page.py   measurements -> the table
    tools/build_export.py           page + measurements + audio -> a single zip
    tools/page_template.html        the empty page the builder substitutes into
    tools/paths.py                  where talkito is

Three passes produce columns the sweep does not, because each needs a quiet machine for a different
reason and folding them in would roughly triple the runtime:

    tools/measure_gaps.py           PESQ and lead-in silence, over the corpus
    tools/measure_rss_rusage.py     peak RSS from the kernel, not sampled
    tools/make_samples.py           the one listening clip per model the page embeds

Run `make_samples.py` at least once, or the page is numbers with nothing to play and the export
reports fewer samples than configurations.

## Outputs, none of them committed

    tools/sweep_*.json        per-unit WER and timing, resumable
    tools/durations.json      timing-only passes
    tools/squim_scores.json   PESQ
    tools/audio_quality.json  PESQ on trimmed audio, plus lead-in
    tts-benchmark.html        the page; over 90% of its size is embedded sample audio
    tts-benchmark-export.zip  the shareable archive

## Adding a model

Four registries, and missing one leaves a blank cell rather than an error:

    tools/run_benchmark_sweep.py   PROVIDER_META   class, parameter count, weights on disk
    tools/build_benchmark_page.py  META            frame rate, release date, the note
    tools/build_benchmark_page.py  LICENSE         as declared on the model's own card
    tools/benchmark_providers.py                   how to actually call it

Notes claiming a superlative -- smallest here, best PESQ -- are claims about the whole table at the
moment they were written, and adding a row falsifies them silently rather than failing.

## Measurement notes worth keeping

  - RTF is wall-clock. Models run one at a time, and a run refuses to start if something heavy is
    already going; measured under load it reads about 2x high, unevenly enough to reorder models.
  - `--repeat` defaults to 3 in the sweep runner. Coverage is summed from each unit's `total`, so a
    unit recorded at repeat 3 counts its phrases three times and can push a row past 84.
    `benchmark.py` pins it to 1.
  - fastpitch, bananamind-tts and parler-tts speak an AI disclosure before the text. Unstripped
    they read about 70 points too high.
  - PESQ over a 20-phrase subset agrees with the full corpus to within about 0.1. WER does not
    subset safely: it is driven by a few catastrophic phrases a small sample can miss entirely.
  - A peak RSS below a model's own weights is impossible for the torch engines and unremarkable for
    the ones served by crispasr, which memory-maps them.
