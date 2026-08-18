#!/usr/bin/env python3
"""Synthesize the one listening clip per model that the page embeds and the export ships.

Without this the table is numbers with no way to hear what they describe, and build_export.py
reports "N configurations, fewer audio samples" -- which is how four KittenTTS rows shipped with
every measurement and nothing to play.

Two details that are easy to get wrong and silent when you do:

  --keep-audio writes {provider}_{category}_{hash}.wav and never the canonical name, so the clip
  has to be promoted onto it. Two earlier regeneration passes reported success having replaced
  nothing, because that step was missing and the old files were still sitting there.

  The three runtimes that speak an AI disclosure need it cut from the saved clip too, not just
  from scoring, or the sample a reader plays opens with a sentence the model was never asked to
  say.
"""

import argparse
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

SAMPLES = os.environ.get("TTS_SAMPLE_DIR",
                         os.path.expanduser("~/.cache/talkito/benchmark-samples"))
PHRASE = "she sells seashells by the seashore"
DISCLOSURE = {"fastpitch", "bananamind-tts", "parler-tts"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("providers", nargs="+", help="models to synthesize a sample for")
    ap.add_argument("--phrase", default=PHRASE)
    args = ap.parse_args()

    os.makedirs(os.path.join(SAMPLES, "mp3"), exist_ok=True)
    for prov in args.providers:
        cmd = [sys.executable, paths.HARNESS, "--tts-provider", prov,
               "--phrase", args.phrase, "--keep-audio", SAMPLES,
               "--asr-model", "base.en", "--quiet"]
        if prov in DISCLOSURE:
            cmd.append("--strip-disclosure")
        subprocess.run(cmd, cwd=paths.TALKITO, stdout=subprocess.DEVNULL)

        produced = sorted(glob.glob(os.path.join(SAMPLES, f"{prov}_*.wav")),
                          key=os.path.getmtime)
        if not produced:
            print(f"  {prov:<22} no audio produced")
            continue
        wav = os.path.join(SAMPLES, f"{prov}.wav")
        os.replace(produced[-1], wav)
        for stale in produced[:-1]:
            os.unlink(stale)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-ac", "1",
                        "-b:a", "64k", os.path.join(SAMPLES, "mp3", f"{prov}.mp3")])
        print(f"  {prov:<22} {os.path.getsize(wav) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
