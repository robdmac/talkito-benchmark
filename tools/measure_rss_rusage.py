#!/usr/bin/env python3
"""Peak RSS per provider, read from the kernel rather than sampled.

Polling the process tree every 200ms swung +/-670 MB between identical runs, in both directions,
because model loading allocates in brief spikes that a sampler catches or misses by luck.
/usr/bin/time -l reports maximum resident set size from rusage, which the kernel maintains
exactly; repeated runs agree to within 2-65 MB.

One phrase is enough. The peak is set by loading the weights, not by synthesizing, which is why
eight long phrases cost minutes per provider and bought nothing.

--durations-only keeps the recogniser out of the process; it would otherwise contribute up to a
gigabyte of its own.

Two runs per provider, reporting the higher and the spread, so an unstable reading is visible
instead of averaged away.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
ROOT = paths.TALKITO

TARGETS = ["fastpitch", "mms-tts", "bananamind-tts", "nt-2e-q8-metal", "nt-2e-q8-cpu",
           "chatterbox-turbo", "cosyvoice3-rl", "tada-1b", "vibevoice-1.5b", "omnivoice",
           "tada-3b", "f5-tts", "parler-tts", "bark", "xtts", "zonos", "dots-tts"]


def rusage_peak(prov):
    """Max RSS in MB for one short synthesis, or None if the run produced nothing."""
    cmd = ["/usr/bin/time", "-l", sys.executable, paths.HARNESS,
           "--tts-provider", prov, "--durations-only", "--quiet", "--phrase-index", "0"]
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None
    m = re.search(r'(\d+)\s+maximum resident set size', r.stderr)
    return round(int(m.group(1)) / 1048576) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/Users/robertmacrae/.claude/jobs/0df08634/tmp/rss2")
    ap.add_argument("--providers", default=",".join(TARGETS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"  {'provider':<18}{'run 1':>8}{'run 2':>8}{'peak':>8}{'spread':>8}{'time':>7}")
    print("  " + "-" * 58)
    for prov in args.providers.split(","):
        path = os.path.join(args.out, f"{prov}.json")
        if os.path.exists(path):
            rec = json.load(open(path))
            print(f"  {prov:<18}{'':>8}{'':>8}{rec.get('peak_rss_mb', '-'):>8}"
                  f"{'':>8}{'cached':>7}")
            continue
        t0 = time.time()
        a = rusage_peak(prov)
        b = rusage_peak(prov)
        el = round(time.time() - t0)
        vals = [v for v in (a, b) if v]
        peak = max(vals) if vals else None
        spread = (max(vals) - min(vals)) if len(vals) == 2 else None
        json.dump({"provider": prov, "peak_rss_mb": peak, "runs": [a, b],
                   "spread_mb": spread, "method": "rusage max RSS, 1 phrase, no recogniser",
                   "elapsed_s": el}, open(path, "w"))
        print(f"  {prov:<18}{a if a else '-':>8}{b if b else '-':>8}"
              f"{peak if peak else '-':>8}{spread if spread is not None else '-':>8}{el:>6}s",
              flush=True)


if __name__ == "__main__":
    main()
