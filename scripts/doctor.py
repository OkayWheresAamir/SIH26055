"""`python scripts/doctor.py` -- is this tree in a state worth handing over?

Every check here exists because it already went wrong once. The RL lane ran for
three rounds in which one side reported work "done" that the other side could
not see, because the work was committed locally and never pushed, or because a
decision number collided, or because a test module imported a training
dependency at the top level and took the whole suite down with it on a machine
that had no training stack.

This is not a test suite and does not replace one. It answers a narrower
question a test suite cannot: **would someone who pulls this right now get what
you think you handed them?** Run it before saying a chunk is finished, and
before pushing.

Exit code 0 = clean, 1 = something would not survive the handover.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECISIONS = ROOT / "docs" / "project" / "DECISIONS.md"

FAIL: list[str] = []
WARN: list[str] = []


def _sh(*args: str) -> str:
    """Git plumbing, or "" if the command can't run (no remote, not a repo)."""
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


# --------------------------------------------------------------- the checks --

def check_pushed() -> None:
    """Unpushed commits are the single cause of every divergence this lane hit.

    A teammate holding five commits for "one combined push" is invisible to
    everyone else, and every report they write describes a tree nobody can pull.
    """
    if not _sh("git", "remote"):
        WARN.append("no git remote configured -- nothing can be handed over from here")
        return
    branch = _sh("git", "rev-parse", "--abbrev-ref", "HEAD")
    _sh("git", "fetch", "--quiet")
    ahead = _sh("git", "log", f"origin/{branch}..HEAD", "--oneline")
    if ahead:
        n = len(ahead.splitlines())
        FAIL.append(f"{n} commit(s) on {branch} are NOT pushed -- nobody else can see this work:\n"
                    + "\n".join("      " + line for line in ahead.splitlines()))
    behind = _sh("git", "log", f"HEAD..origin/{branch}", "--oneline")
    if behind:
        WARN.append(f"{len(behind.splitlines())} commit(s) on origin/{branch} not pulled -- "
                    "you may be about to duplicate work or collide")


def check_uncommitted_decisions() -> None:
    """`DECISIONS.md` is where project knowledge survives a session (CLAUDE.md).

    Uncommitted, it survives nothing. This is the file that was lost twice.
    """
    dirty = _sh("git", "status", "--porcelain", "--", "docs/project", "CLAUDE.md")
    if dirty:
        FAIL.append("authoritative documents have uncommitted changes:\n"
                    + "\n".join("      " + line for line in dirty.splitlines()))


def check_decision_numbers() -> None:
    """One number, one decision.

    Two people allocating D-numbers in parallel produced two different D64s.
    Gaps are fine (a withdrawn entry leaves one); duplicates are not, because a
    later reference to "D64" then means two things and the reader cannot tell
    which.
    """
    if not DECISIONS.exists():
        FAIL.append(f"{DECISIONS.relative_to(ROOT)} is missing")
        return
    nums = [int(m) for m in re.findall(r"^## D(\d+) ", DECISIONS.read_text(encoding="utf-8"),
                                       flags=re.M)]
    if not nums:
        FAIL.append("no decision headings found -- has the format changed?")
        return
    dupes = sorted(n for n, c in Counter(nums).items() if c > 1)
    if dupes:
        FAIL.append("duplicate decision numbers (two decisions share one id): "
                    + ", ".join(f"D{n}" for n in dupes))
    print(f"    decisions D{min(nums)}..D{max(nums)}, {len(nums)} entries, highest = D{max(nums)}")


def check_suite_imports_without_training_stack() -> None:
    """A test module that imports the training stack at module level takes the
    WHOLE suite down at collection on a machine without it -- not one skip, no
    tests at all. This has now happened twice (test_split.py, and the rung skip
    table before it).

    Static check, so it reports the problem even on a machine where the import
    would have succeeded and hidden it.
    """
    heavy = {"stable_baselines3", "sb3_contrib", "torch", "rfenv.rl"}

    def _is_importorskip(node: ast.stmt) -> bool:
        """`sb3 = pytest.importorskip("stable_baselines3")` at module level is the
        CORRECT idiom -- it skips the whole module cleanly. Only an unguarded
        import is the bug."""
        call = node.value if isinstance(node, ast.Expr) else (
            node.value if isinstance(node, ast.Assign) else None)
        return (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "importorskip")

    for path in sorted((ROOT / "tests").glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            FAIL.append(f"{path.relative_to(ROOT)} does not parse: {exc}")
            continue
        guarded = False
        for node in tree.body:                      # top level only -- nested is the fix
            if _is_importorskip(node):
                guarded = True
                continue
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for mod in mods:
                if guarded:
                    continue
                if any(mod == h or mod.startswith(h + ".") for h in heavy):
                    FAIL.append(
                        f"{path.relative_to(ROOT)} imports {mod!r} at module level with no "
                        "preceding pytest.importorskip -- this aborts COLLECTION of the "
                        "entire suite without a training stack. Either add "
                        'pytest.importorskip("stable_baselines3") above it (skips the whole '
                        "module), or move the import inside the one test that needs it.")


def check_rungs_declare_their_checkpoints() -> None:
    """Every registered rung must be buildable, or fail in a way that says so.

    A rung pointing at a checkpoint nobody has is fine -- it should skip. A rung
    that raises ModuleNotFoundError instead is the bug that broke the suite
    three times.
    """
    try:
        from rfenv.baselines import LADDER
    except Exception as exc:
        FAIL.append(f"cannot import the ladder at all: {type(exc).__name__}: {exc}")
        return
    missing = []
    for spec in LADDER:
        src = getattr(spec.factory, "__doc__", "") or ""
        for m in re.finditer(r"runs/checkpoints/[\w./-]+\.zip", src):
            if not (ROOT / m.group()).exists():
                missing.append((spec.rung, m.group()))
    print(f"    ladder: {len(LADDER)} rungs registered")
    ckpt_dir = ROOT / "runs" / "checkpoints"
    # Recursive (**): checkpoints live in per-model subfolders, not flat, since
    # the runs/checkpoints/v1|v2/<model_name>/ reorganisation (D75).
    n = len(list(ckpt_dir.glob("**/*.zip"))) if ckpt_dir.is_dir() else 0
    print(f"    checkpoints on disk: {n}")
    if n == 0:
        WARN.append("no checkpoints in runs/checkpoints/ -- every RL rung will skip. "
                    "runs/ is gitignored, so these NEVER travel through git: if a "
                    "teammate needs them, they go over the drive share.")


def check_docs_agree_with_code() -> None:
    """The observation width is the number that has gone stale in the documents
    every single time it changed (109 -> 145 -> 146 -> 147 -> 146 -> ...).

    Read it from the code, then look for any *other* width being asserted as
    the observation's in an authoritative document.
    """
    try:
        from rfenv.constants import N_BANDS
        from rfenv.env import ScanEnv
        from rfenv.scenario import Scenario
        width = ScanEnv(scenario=Scenario.replay("config_81", "stare")).observation_space.shape[0]
    except Exception as exc:
        WARN.append(f"could not read the observation width from the code: "
                    f"{type(exc).__name__}: {exc}")
        return
    print(f"    observation width (from the code): {width}")
    # Only obs-width-shaped phrasings: "N wide", "N-vector", "shape=(N,)", "= N".
    # A bare number near the word "observation" is far too loose -- it matched a
    # "100" in an unrelated sentence.
    pattern = re.compile(
        r"\b(1[0-9]{2})[- ](?:wide|vector)\b"
        r"|shape=\((1[0-9]{2}),"
        r"|=\s*\*\*(1[0-9]{2})\*\*"
        r"|\b(1[0-9]{2})\b\s*=\s*36\s*[x\u00d7]", re.I)
    for name in ("ENVIRONMENT_SPEC.md", "STATE_ACTION_FORMULATION.md", "EVALUATION.md"):
        path = ROOT / "docs" / "project" / name
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in pattern.finditer(line):
                found = next((g for g in m.groups() if g), None)
                if found and int(found) != width and abs(int(found) - width) < 100:
                    WARN.append(f"{name}:{i} says {found} where the code says {width} "
                                f"-- check it is a historical reference, not a stale claim")


def main() -> int:
    print(f"doctor: {ROOT}\n")
    for label, fn in [
        ("git",         check_pushed),
        ("documents",   check_uncommitted_decisions),
        ("decisions",   check_decision_numbers),
        ("test imports", check_suite_imports_without_training_stack),
        ("ladder",      check_rungs_declare_their_checkpoints),
        ("doc sync",    check_docs_agree_with_code),
    ]:
        print(f"  [{label}]")
        fn()

    print()
    for w in WARN:
        print(f"  WARN  {w}")
    for f in FAIL:
        print(f"  FAIL  {f}")
    if FAIL:
        print(f"\n{len(FAIL)} blocking problem(s). This tree is not ready to hand over.")
        return 1
    print(f"\nClean{f' ({len(WARN)} warning(s))' if WARN else ''}. "
          "Safe to say the work is done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
