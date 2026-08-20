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

"""Runs the round-trip benchmark across every provider, one resumable slice at a time."""

# A full sweep is hours of compute, which is longer than any single shell invocation survives.
# Work is split into (provider, category) units, each run as its own harness subprocess, and the
# result of every completed unit is written to a state file immediately. Re-running picks up
# wherever the last invocation stopped, so an interrupted sweep never loses finished work.

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HARNESS = os.path.join(REPO, "tests-repo", "test_tts_asr_roundtrip.py")

PROVIDERS = [
    "piper", "kokoro", "kittentts", "neutts2e",
    "melotts", "styletts2", "speecht5", "vibevoice", "cosyvoice3",
    "neutts2e-fp32",
    "chatterbox", "chatterbox-q8", "chatterbox-q4", "csm", "bark",
]


# Slow providers are split into chunks of this many phrases, so no single unit outlives the
# shell invocation running it. Long passages at 7x real time otherwise exceed any timeout.
PROVIDER_CHUNK = {
    "bark": 2,
    "chatterbox": 2,
    "csm": 3,
    "chatterbox-q8": 3,
    "chatterbox-q4": 3,
    "neutts-nano": 4,
    "neutts2e-fp32": 4,
    "nt-2e-fp32-cpu": 4, "nt-2e-fp32-mps": 4,
    "nt-nano-fp32-cpu": 4, "nt-nano-fp32-mps": 4,
    "nt-air-fp32-cpu": 4, "nt-air-fp32-mps": 4,
    # Wider sweep: these run well over real time, and an unchunked category means one long passage
    # can hold a unit open for half an hour and lose the whole unit if it times out. Small chunks
    # keep partial progress, which for a slow model is the difference between a row and nothing.
    "parler-tts": 3, "mini-omni2": 3, "omnivoice": 3, "outetts": 3,
    "lfm2-audio": 3, "vibevoice-1.5b": 3, "voxcpm2-tts": 3, "zonos": 3,
    "vibevoice-bitnet": 3, "tada-1b": 3, "qwen3-tts-1.7b": 3, "qwen3-tts-vd": 3,
    "xtts": 3, "tortoise": 2, "kugelaudio": 2, "dia": 2, "dots-tts": 2, "moss-tts": 2, "tada-3b": 2, "moss-tts-local": 2,
}


# Architecture class, published parameter count, and the on-disk weights this machine actually
# holds. Sizes are measured rather than quoted; "-" means the figure is not published upstream and
# is not worth guessing at. Sizes cover model weights only, not the runtime or its dependencies.
PROVIDER_META = {
    "piper":         ("stoch-ff", "16M",    60),
    "kokoro":        ("det-ff",   "82M",  313),
    "kittentts":     ("det-ff",   "73M",    78),
    # KittenTTS publishes three sizes; the row above is mini. Params scale with the ONNX
    # file, which is the whole model -- these are single-forward-pass ONNX graphs.
    "kittentts-micro": ("det-ff", "40M",    43),
    "kittentts-nano":  ("det-ff", "54M",    57),
    "kittentts-nano-int8": ("det-ff", "54M", 26),
    "kittentts-mini":  ("det-ff", "73M",    78),
    "system":        ("os",       "-",    0),
    "neutts2e":      ("ar-lm",    "0.24B", 287 + 298),
    "neutts2e-fp32": ("ar-lm",    "0.5B", 473 + 298),
    "neutts-nano":   ("ar-lm",    "0.5B", 913 + 298),
    "neutts-nano-q4":("ar-lm",    "0.5B", 186 + 298),
    "neutts-nano-q8":("ar-lm",    "0.5B", 241 + 298),
    "melotts":       ("stoch-ff", "-",    198),
    "styletts2":     ("stoch-ff", "-",    870),
    "speecht5":      ("ar-lm",    "157M",  300),
    "vibevoice":     ("ar-lm",    "1.02B", 668),
    "cosyvoice3":    ("ar-lm",    "0.64B", 1223),
    "chatterbox":    ("ar-lm",    "0.54B", 3072),
    "chatterbox-q8": ("ar-lm",    "0.54B", 1004),
    "chatterbox-q4": ("ar-lm",    "0.54B", 637),
    "csm":           ("ar-lm",    "1.78B",   1369),
    "bark":          ("ar-lm",    "404M",    414),
    "qwen3-tts":     ("ar-lm",    "0.6B", 1300),
    # Llama-3B backbone over SNAC; the largest LM in the comparison by a wide margin
    "orpheus-q4":    ("ar-lm",    "3.3B", 2560),
    "orpheus-q8":    ("ar-lm",    "3.3B", 3970),
    "orpheus-iq1s":  ("ar-lm",    "3.3B",  970),
    "orpheus-iq1m":  ("ar-lm",    "3.3B", 1020),
    "fastpitch":     ("det-ff",   "-",     120),
    "f5-tts":        ("stoch-ff", "0.34B", 953),
    "chatterbox-turbo": ("ar-lm", "0.54B", 980),
    "mms-tts":           ("stoch-ff", "36M", 145),
    "kugelaudio":        ("ar-lm", "-", 17300),
    "xtts":              ("ar-lm", "0.47B", 1870),
    "tortoise":          ("ar-lm", "-", 3200),
    "bananamind-tts":    ("det-ff", "-", 38),
    "cosyvoice3-rl":     ("ar-lm", "0.64B", 384),
    "indextts":          ("ar-lm", "-", 870),
    "parler-tts":        ("ar-lm", "0.88B", 900),
    "mini-omni2":        ("ar-lm", "-", 1000),
    "omnivoice":         ("ar-lm", "-", 1200),
    "outetts":           ("ar-lm", "1B", 1270),
    "lfm2-audio":        ("ar-lm", "1.5B", 1600),
    "vibevoice-1.5b":    ("ar-lm", "1.5B", 1600),
    "voxcpm2-tts":       ("ar-lm", "-", 1600),
    "zonos":             ("ar-lm", "1.6B", 1600),
    "vibevoice-bitnet":  ("ar-lm", "1.5B", 1600),
    "tada-1b":           ("ar-lm", "1B", 1700),
    "qwen3-tts-1.7b":    ("ar-lm", "1.7B", 1900),
    "qwen3-tts-vd":      ("ar-lm", "1.7B", 1900),
    "dia":               ("ar-lm", "1.6B", 3000),
    "dots-tts":          ("ar-lm", "-", 4400),
    "moss-tts":          ("ar-lm", "-", 5000),
    "tada-3b":           ("ar-lm", "3B", 6600),
    "moss-tts-local":    ("ar-lm", "-", 9100),

    # Same 12 Hz codec as the 0.6B qwen3-tts row, three times the backbone, and conditioned on a
    # written voice description rather than a reference clip.
    "qwen3-tts-vd":  ("ar-lm", "1.7B", 1948),
    # A 3B Llama-3.2 backbone over SNAC, so 4-16x any NeuTTS variant and the largest LM here.
    # q8_0 size is filled in once the sweep has fetched it.
    "orpheus-q4":    ("ar-lm",   "3B",  2440),
    "orpheus-q8":    ("ar-lm",   "3B",     0),
    # Flow matching over ONNX: not autoregressive, so it has no sampler and cannot run away.
    # The only 44.1 kHz engine here; output is resampled to 24 kHz to stay comparable.
    "supertonic":    ("det-ff",  "99M", 380),
    # Q8_0 weights measure 347 MB on disk; +298 for the codec, as every NeuTTS row counts it.
    "nt-2e-q8-cpu":     ("ar-lm", "0.24B", 347 + 298),
    "nt-2e-q8-metal":   ("ar-lm", "0.24B", 347 + 298),
    "nt-2e-q4-cpu":     ("ar-lm", "0.24B", 287 + 298),
    "nt-2e-q4-metal":   ("ar-lm", "0.24B", 287 + 298),
    "nt-2e-fp32-cpu":   ("ar-lm", "0.24B", 473 + 298),
    "nt-2e-fp32-mps":   ("ar-lm", "0.24B", 473 + 298),
    "nt-nano-q4-cpu":   ("ar-lm", "0.23B", 186 + 298),
    "nt-nano-q4-metal": ("ar-lm", "0.23B", 186 + 298),
    "nt-nano-q8-cpu":   ("ar-lm", "0.23B", 241 + 298),
    "nt-nano-q8-metal": ("ar-lm", "0.23B", 241 + 298),
    "nt-nano-fp32-cpu": ("ar-lm", "0.23B", 913 + 298),
    "nt-nano-fp32-mps": ("ar-lm", "0.23B", 913 + 298),
    "nt-air-q4-cpu":    ("ar-lm", "0.75B", 527 + 298),
    "nt-air-q4-metal":  ("ar-lm", "0.75B", 527 + 298),
    "nt-air-q8-cpu":    ("ar-lm", "0.75B", 803 + 298),
    "nt-air-q8-metal":  ("ar-lm", "0.75B", 803 + 298),
    "nt-air-fp32-cpu":  ("ar-lm", "0.75B", 3039 + 298),
    "nt-air-fp32-mps":  ("ar-lm", "0.75B", 3039 + 298),
}


def _swap_free_mb():
    """Free swap in MB, or -1 where the platform does not report it."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        match = re.search(r"free\s*=\s*([\d.]+)M", out)
        return round(float(match.group(1))) if match else -1
    except Exception:
        return -1


def units_for(provider, category, corpus):
    """Yield (unit_key, phrase_indices) covering one provider/category pair."""
    indices = [i for i, p in enumerate(corpus) if p.category == category]
    size = PROVIDER_CHUNK.get(provider, len(indices)) or len(indices)
    for start in range(0, len(indices), size):
        chunk = indices[start:start + size]
        suffix = "" if size >= len(indices) else f"|{start // size}"
        yield f"{provider}|{category}{suffix}", chunk


def load_state(path):
    """Read previously completed units."""
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    return {"units": {}}


def save_state(path, state):
    """Persist after every unit so an interrupted sweep keeps its finished work.

    Merges with whatever is on disk rather than overwriting it. Each process holds the whole state
    in memory, so two running at once would otherwise clobber each other: the second to save wipes
    every unit the first recorded after it started. That has already cost one provider's results,
    and it fails silently - the units simply are not there afterwards.
    """
    merged = dict(load_state(path).get("units", {}))
    merged.update(state.get("units", {}))
    out = dict(state)
    out["units"] = merged
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
    os.replace(tmp, path)
    state["units"] = merged


HARNESS_ARGS = []  # extra flags forwarded to every harness invocation, set from --harness-arg


def run_unit(provider, category, repeat, indices=None, asr_model=None):
    """Run one provider against one category (or a chunk of it) and return its scores."""
    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    command = [sys.executable, "-u", HARNESS, "--tts-provider", provider,
               "--category", category, "--repeat", str(repeat), "--quiet", "--json", out_path]
    if asr_model:
        command += ["--asr-model", asr_model]
    command += HARNESS_ARGS
    for index in indices or []:
        command += ["--phrase-index", str(index)]
    # RTF is wall-clock sensitive: a busy machine inflates it several-fold, so record the load
    # the measurement was taken under and let contaminated units be spotted and redone
    load_before = os.getloadavg()[0]
    swap_free_before = _swap_free_mb()
    started = time.time()
    proc = subprocess.run(command, capture_output=True, text=True, cwd=REPO)
    elapsed = time.time() - started
    load = max(load_before, os.getloadavg()[0])
    # Swap exhaustion slows synthesis several-fold and creeps up as large models are loaded and
    # unloaded, so record the low-water mark alongside the timing it produced
    swap_free = min(swap_free_before, _swap_free_mb())

    try:
        with open(out_path) as handle:
            report = json.load(handle)
    except Exception:
        return {"error": (proc.stderr or proc.stdout or "no output")[-300:], "seconds": elapsed}
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)

    entry = (report.get("providers") or {}).get(provider)
    if not entry:
        return {"error": "provider missing from report", "seconds": elapsed}

    cat = (entry.get("by_category") or {}).get(category, {})
    return {
        "passed": cat.get("passed", entry.get("passed", 0)),
        "total": cat.get("total", entry.get("total", 0)),
        "mean_wer": cat.get("mean_wer", entry.get("mean_wer", 0.0)),
        "rtf": entry.get("rtf", 0.0),
        "synth_seconds": entry.get("synth_seconds", 0.0),
        "audio_seconds": entry.get("audio_seconds", 0.0),
        "seconds": elapsed,
        "load": round(load, 2),
        "swap_free_mb": swap_free,
    }


def summarize(state, repeat):
    """Aggregate finished units into a per-provider table."""
    rows = []
    for provider in PROVIDERS:
        units = [(k, v) for k, v in state["units"].items()
                 if k.split("|")[0] == provider and "error" not in v]
        if not units:
            continue
        total = sum(v["total"] for _, v in units)
        if not total:
            continue
        passed = sum(v["passed"] for _, v in units)
        wer = sum(v["mean_wer"] * v["total"] for _, v in units) / total
        # Ratio of summed totals, not a mean of per-unit ratios
        audio = sum(v.get("audio_seconds", 0.0) for _, v in units)
        synth = sum(v.get("synth_seconds", 0.0) for _, v in units)
        rtf = (synth / audio) if audio else sum(v["rtf"] * v["total"] for _, v in units) / total
        loads = [v.get("load", 0.0) for _, v in units]
        rows.append((provider, passed, total, wer, rtf, len(units), max(loads) if loads else 0.0))

    rows.sort(key=lambda r: r[3])
    print(f"\n{'provider':<16} {'class':>9} {'params':>7} {'size':>8} {'passed':>10} "
          f"{'mean WER':>9} {'RTF':>8} {'units':>6}")
    print("-" * 82)
    for provider, passed, total, wer, rtf, n, peak in rows:
        cls, params, size_mb = PROVIDER_META.get(provider, ("?", "-", 0))
        size = "-" if not size_mb else (f"{size_mb/1024:.1f}GB" if size_mb >= 1024 else f"{size_mb}MB")
        flag = "!" if peak > 12 else " "
        print(f"{provider:<16} {cls:>9} {params:>7} {size:>8} {passed:>4}/{total:<5} "
              f"{wer:>8.0%} {rtf:>7.2f}x {n:>5}{flag}")
    print("  class: det-ff deterministic feed-forward | stoch-ff sampling feed-forward | "
          "ar-lm autoregressive LM + codec")
    print("  size: model weights on disk, excluding runtime and dependencies. "
          "! = measured under high system load")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=os.path.join(HERE, "sweep_state.json"))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--budget", type=float, default=480.0,
                        help="seconds of work to do before returning, so the caller never times out")
    parser.add_argument("--summary", action="store_true", help="print the table and exit")
    parser.add_argument("--providers", help="comma-separated subset to run instead of all")
    parser.add_argument("--harness-arg", action="append", dest="harness_args", default=[],
                        help="Forward a flag to the harness, e.g. --harness-arg --strip-disclosure "
                             "(repeatable)")
    parser.add_argument("--asr-model", help="ASR model to score with; use a separate state file "
                                            "per model, since scores are not comparable across them")
    args = parser.parse_args()

    if args.harness_args:
        global HARNESS_ARGS
        HARNESS_ARGS = args.harness_args

    if args.providers:
        global PROVIDERS
        PROVIDERS = [p.strip() for p in args.providers.split(",") if p.strip()]

    state = load_state(args.state)
    if args.summary:
        summarize(state, args.repeat)
        return 0

    sys.path.insert(0, os.path.join(REPO, "tests-repo"))
    import test_tts_asr_roundtrip as harness  # noqa: E402

    all_units = [(p, c, key, idx) for p in PROVIDERS for c in harness.CATEGORIES
                 for key, idx in units_for(p, c, harness.CORPUS)]
    pending = [u for u in all_units if u[2] not in state["units"]]
    done = len(all_units) - len(pending)
    print(f"{done} units done, {len(pending)} pending, budget {args.budget:.0f}s")

    deadline = time.time() + args.budget
    for provider, category, key, indices in pending:
        if time.time() >= deadline:
            print("budget reached, stopping cleanly")
            break
        result = run_unit(provider, category, args.repeat, indices, args.asr_model)
        state["units"][key] = result
        save_state(args.state, state)
        if "error" in result:
            print(f"  {provider:<15} {key.split('|',1)[1]:<14} ERROR {result['error'][:70]}")
        else:
            print(f"  {provider:<15} {key.split('|',1)[1]:<14} {result['passed']}/{result['total']} "
                  f"WER {result['mean_wer']:.0%} ({result['seconds']:.0f}s)")

    remaining = sum(1 for u in all_units if u[2] not in state["units"])
    print(f"\n{remaining} units remaining")
    return 0


if __name__ == "__main__":
    sys.exit(main())
