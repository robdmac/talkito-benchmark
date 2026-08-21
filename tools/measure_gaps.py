#!/usr/bin/env python3
"""Fill the benchmark's missing peak-RSS and PESQ figures, in one corpus pass per provider.

Sixteen of the 34 displayed rows carry neither number: both come from hand-maintained sources --
two dicts in the page builder for RSS, squim_scores.json for PESQ -- so every configuration
added since those passes inherits blanks without anything failing.

Both needed a corpus run, so both are taken from the same one:

  PESQ  averaged over the corpus, not measured on the listening sample. Checking a few providers
        that already have a score showed why: consistent models agree either way, but bark reads
        2.10 across the corpus and 3.03 on one clip, because a single take of an inconsistent
        model is not representative.
  RSS   peak across the whole process tree, not just this process. Several providers synthesize
        in a subprocess -- a CrispASR server or a venv worker -- whose memory would otherwise go
        uncounted, which is likely why those rows were left blank in the first place.

Cheapest providers run first, so a long sweep yields usable rows early.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")

# The harness is imported for strip_disclosure; without this on the path the import raises
# mid-run and takes the whole sweep down with it, which is exactly what happened once.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ordered by measured RTF: minutes of synthesis, roughly, for the 84-phrase corpus.
TARGETS = ["fastpitch", "mms-tts", "bananamind-tts", "nt-2e-q8-metal", "nt-2e-q8-cpu",
           "chatterbox-turbo", "cosyvoice3-rl", "tada-1b", "vibevoice-1.5b", "omnivoice",
           "tada-3b", "f5-tts", "parler-tts", "bark", "xtts", "zonos", "dots-tts"]

# Runtimes that prepend a spoken AI disclosure. Scored without --strip-disclosure, fastpitch
# reads 85% instead of its published 15.3%, because the recogniser transcribes a sentence the
# model was never asked to say. The flag fixes WER; PESQ needs the same cut applied to the audio
# separately, since the harness strips for scoring only and writes the file untouched.
DISCLOSURE = {"fastpitch", "bananamind-tts", "parler-tts"}


def peak_rss_of_tree(proc, stop, out):
    """Track the high-water RSS of the process and everything it spawns."""
    import psutil
    try:
        parent = psutil.Process(proc.pid)
    except Exception:
        return
    peak = 0
    while not stop.is_set():
        try:
            total = parent.memory_info().rss
            for child in parent.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except Exception:
                    pass
            peak = max(peak, total)
        except Exception:
            break
        time.sleep(0.25)
    out["peak_mb"] = round(peak / (1024 * 1024))


def squim_pesq(paths, model, strip=False):
    import numpy as np, soundfile as sf, torch, torchaudio
    vals = []
    for p in paths:
        try:
            a, r = sf.read(p, dtype="float32")
        except Exception:
            continue
        if a.ndim > 1:
            a = a.mean(axis=1)
        if strip:
            # The saved file still carries the disclosure; scoring it would measure the
            # runtime's preamble as much as the model.
            import test_tts_asr_roundtrip as H
            a = H.strip_disclosure(a, r)
        if a.size < r // 4:
            continue
        # Trimmed, as every stored PESQ was: scoring untrimmed rewards whoever pads most.
        th = max(float(np.abs(a).max()) * 0.02, 1e-4)
        idx = int(np.argmax(np.abs(a) > th))
        a = a[max(0, idx - int(0.1 * r)):]
        t = torch.from_numpy(a).unsqueeze(0)
        if r != 16000:
            t = torchaudio.functional.resample(t, r, 16000)
        with torch.no_grad():
            _, pesq, _ = model(t)
        vals.append(float(pesq[0]))
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/Users/robertmacrae/.claude/jobs/0df08634/tmp/gaps")
    ap.add_argument("--providers", default=",".join(TARGETS))
    # PESQ is a mean over clips, so a subset trades a little precision for hours on the models
    # that run at 10x real time. Recorded in the result as corpus_subset so an estimate is never
    # mistaken for a full-corpus score.
    ap.add_argument("--phrases", type=int, default=0,
                    help="score this many corpus phrases instead of all 84")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    import torchaudio
    model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()

    for prov in args.providers.split(","):
        result_path = os.path.join(args.out, f"{prov}.json")
        if os.path.exists(result_path):
            print(f"  {prov:<18} already done", flush=True)
            continue
        audio_dir = os.path.join(args.out, "audio", prov)
        # Clear first: an interrupted earlier attempt leaves clips behind, and --keep-audio
        # writes hashed filenames that do not collide, so the leftovers silently join the PESQ
        # average (fastpitch scored over 123 clips for an 84-phrase corpus this way).
        if os.path.isdir(audio_dir):
            shutil.rmtree(audio_dir)
        os.makedirs(audio_dir, exist_ok=True)
        run_json = os.path.join(args.out, f"{prov}_run.json")

        cmd = [sys.executable, paths.HARNESS,
               "--tts-provider", prov, "--asr-model", "small", "--quiet",
               "--keep-audio", audio_dir, "--json", run_json]
        if args.phrases:
            for i in range(args.phrases):
                cmd += ["--phrase-index", str(i)]
        if prov in DISCLOSURE:
            cmd.append("--strip-disclosure")
        print(f"  {prov:<18} running ...", flush=True)
        started = time.time()
        proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        stop, mem = threading.Event(), {}
        watcher = threading.Thread(target=peak_rss_of_tree, args=(proc, stop, mem), daemon=True)
        watcher.start()
        proc.wait()
        stop.set()
        watcher.join(timeout=5)

        wavs = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
        pesq, n = squim_pesq(wavs, model, strip=prov in DISCLOSURE)
        wer = None
        if os.path.exists(run_json):
            try:
                wer = json.load(open(run_json))["providers"][prov]["mean_wer"]
            except (ValueError, KeyError):
                pass
        rec = {"provider": prov, "peak_rss_mb": mem.get("peak_mb"), "pesq": pesq,
               "corpus_subset": args.phrases or None,
               "pesq_clips": n, "mean_wer": wer, "elapsed_s": round(time.time() - started, 1)}
        json.dump(rec, open(result_path, "w"))
        print(f"  {prov:<18} RSS {rec['peak_rss_mb']} MB   PESQ "
              f"{pesq if pesq is None else round(pesq, 2)} over {n} clips   "
              f"WER {'-' if wer is None else round(100*wer, 1)}%", flush=True)


if __name__ == "__main__":
    main()
