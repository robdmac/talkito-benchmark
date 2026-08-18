#!/usr/bin/env python3
"""Where talkito is, from inside the benchmark repo.

Every script here reaches into talkito for two things: the package itself, which the providers
import, and tests-repo/test_tts_asr_roundtrip.py, which is the harness that actually synthesizes
and scores. Before the split those were simply the parent directory. Now they are not, and each
script guessing separately is how a relocation turns into eight different failures.

Resolution order, first hit wins:

  TALKITO_ROOT        set it explicitly; a checkout anywhere, or a second one to compare against
  ../..               the submodule position, benchmark/ inside a talkito checkout
  ../talkito          the sibling position, for working on both repos side by side

A wrong answer here fails late and confusingly -- an import error three subprocesses deep, or a
harness that runs against a different talkito than the one being measured -- so the check is for
the harness file itself rather than a directory that merely has the right name.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(HERE)

HARNESS_REL = os.path.join("tests-repo", "test_tts_asr_roundtrip.py")


def _valid(path):
    return path and os.path.exists(os.path.join(path, HARNESS_REL))


def talkito_root():
    """Absolute path to the talkito checkout, or raise with what was tried."""
    # An explicit setting that does not resolve is an error, not a hint. Falling through to the
    # search would quietly measure a different checkout than the one asked for, and the results
    # would look entirely normal -- the worst kind of wrong answer this module can give.
    override = os.environ.get("TALKITO_ROOT")
    if override and not _valid(override):
        raise RuntimeError(f"TALKITO_ROOT={override!r} does not contain {HARNESS_REL}")

    tried = []
    for candidate in (override,
                      os.path.dirname(BENCHMARK_ROOT),
                      os.path.join(os.path.dirname(BENCHMARK_ROOT), "talkito")):
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        tried.append(candidate)
        if _valid(candidate):
            return candidate
    raise RuntimeError(
        "cannot find talkito. Set TALKITO_ROOT to a checkout containing "
        f"{HARNESS_REL}.\nTried:\n  " + "\n  ".join(tried))


def __getattr__(name):
    """Resolve lazily, so importing this module never requires talkito to be present.

    Rebuilding the page reads only stored measurements, and tying that to a checkout being
    findable would mean the results could not be republished from the data alone.
    """
    if name == "TALKITO":
        return talkito_root()
    if name == "HARNESS":
        return os.path.join(talkito_root(), HARNESS_REL)
    if name == "TESTS_REPO":
        return os.path.join(talkito_root(), os.path.dirname(HARNESS_REL))
    raise AttributeError(name)
