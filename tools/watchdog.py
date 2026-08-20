#!/usr/bin/env python3
"""Watch work for progress rather than bounding it by the clock.

A fixed timeout answers "has too much time passed", when the question is "is it still getting
anywhere". Both failures are on record in this session:

  dia            killed at 1800s while actively downloading at 0.5 MB/s. It needed 100 minutes and
                 was making steady progress the whole time. The timeout destroyed 2.2 GB of work,
                 twice, because the download does not resume.
  orpheus PESQ   sat at near-zero CPU for two hours having written nothing, and no timeout fired
                 at all. Progress had stopped; the clock had not.

So: watch a progress signal, not elapsed time. Sample it periodically, and act only when it stops
moving. Slow is fine, indefinitely. Stuck is not, for more than a few minutes.

    python3 watchdog.py --glob '~/.cache/crispasr/dia*.part.*' --stall-minutes 10 --pid 30901

Progress is the total size of files matching --glob, or their count with --count. Either way it is
a number that must keep rising; when it stops for --stall-minutes the process is reported and,
with --kill, terminated.
"""

import argparse
import glob as globmod
import os
import signal
import sys
import time


def measure(pattern, by_count):
    paths = globmod.glob(os.path.expanduser(pattern))
    if by_count:
        return float(len(paths))
    return float(sum(os.path.getsize(p) for p in paths if os.path.exists(p)))


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True, help="files whose size (or count) means progress")
    ap.add_argument("--pid", type=int, help="process to watch; exits when it does")
    ap.add_argument("--stall-minutes", type=float, default=10.0)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--count", action="store_true", help="count files instead of summing bytes")
    ap.add_argument("--kill", action="store_true", help="terminate the pid when stalled")
    ap.add_argument("--label", default="work")
    args = ap.parse_args()

    unit = "files" if args.count else "MB"
    scale = 1.0 if args.count else 1048576.0

    last, last_change, started = measure(args.glob, args.count), time.time(), time.time()
    print(f"  watching {args.label}: {last / scale:.0f} {unit}, stall limit "
          f"{args.stall_minutes:.0f} min", flush=True)

    while True:
        time.sleep(args.interval)
        if args.pid and not alive(args.pid):
            print(f"  {args.label}: process {args.pid} exited after "
                  f"{(time.time() - started) / 60:.0f} min, "
                  f"{measure(args.glob, args.count) / scale:.0f} {unit}", flush=True)
            return 0

        now = measure(args.glob, args.count)
        if now > last:
            rate = (now - last) / scale / ((time.time() - last_change) or 1)
            print(f"  {args.label}: {now / scale:.0f} {unit} "
                  f"(+{(now - last) / scale:.1f}, {rate * 60:.1f} {unit}/min, "
                  f"{(time.time() - started) / 60:.0f} min elapsed)", flush=True)
            last, last_change = now, time.time()
            continue

        stalled = (time.time() - last_change) / 60
        if stalled >= args.stall_minutes:
            print(f"  {args.label}: STALLED - no progress for {stalled:.0f} min at "
                  f"{now / scale:.0f} {unit}", flush=True)
            if args.pid and args.kill:
                os.kill(args.pid, signal.SIGKILL)
                print(f"  {args.label}: killed {args.pid}", flush=True)
            return 1


if __name__ == "__main__":
    sys.exit(main())
