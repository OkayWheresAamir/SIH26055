"""`python3 scripts/slim_clone.py` -- how much disk does this clone use, and how
much of it is a second copy of something already on GitHub?

Every checkpoint under `runs/checkpoints/` is tracked, so a full clone holds each
one twice: once in the working tree, once in `.git`'s pack. Only the checkpoints
the code actually loads (a literal `runs/checkpoints/...zip` in `rfenv/`,
`tests/` or `scripts/`, outside a comment) are needed to run the ladder, the
suite and `doctor.py`; the rest are intermediate snapshots and dead observation
widths that live on the remote whether or not they are on this disk.

Three modes, and none of them deletes anything:

    python3 scripts/slim_clone.py                  # survey: report only, changes nothing
    python3 scripts/slim_clone.py --sparse         # in place: hide unreferenced checkpoints
    python3 scripts/slim_clone.py --restore        # undo --sparse
    python3 scripts/slim_clone.py --reclone DEST   # build a slim partial clone at DEST

`--sparse` is a git sparse checkout: the hidden files stay in `.git`, so it frees
the working-tree copy only and is undone by `--restore`. `--reclone` also drops
the `.git` copy -- a partial clone keeps full history but leaves blobs over 1 MB
on the remote until a checkout needs them -- and copies every untracked and
ignored path (`data/`, `.venv/`, `CLAUDE.local.md`, `runs/baselines/`, ...) across,
using copy-on-write clones on APFS so they cost no extra space. The old clone is
left untouched; the script prints the commands to swap the two over once you
have checked the new one.

A hidden snapshot is one command away in either mode (the `/**` matters: a bare
directory pattern loses to the file-level exclusions above it):

    git sparse-checkout add '/runs/checkpoints/v2p/<model_name>/**'

Both write modes refuse to run unless everything local is on the remote:
a clean tree, no stash, and every local branch pushed to its upstream.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT = "runs/checkpoints"
SCAN_DIRS = ("rfenv", "tests", "scripts")
REF = re.compile(r"runs/checkpoints/[\w./-]+\.zip")
BLOB_LIMIT = "1m"


def _git(*args: str, cwd: Path = ROOT, check: bool = True) -> str:
    done = subprocess.run(("git", *args), cwd=cwd, capture_output=True, text=True)
    if check and done.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{done.stderr.strip()}")
    return done.stdout


def _disk(path: Path, seen: set[tuple[int, int]] | None = None) -> int:
    """Bytes actually allocated on disk, each inode counted once (like `du`)."""
    seen = set() if seen is None else seen
    if path.is_dir() and not path.is_symlink():
        paths = (os.path.join(dp, f) for dp, _, fs in os.walk(path) for f in fs)
    else:
        paths = iter((str(path),))
    total = 0
    for p in paths:
        try:
            st = os.lstat(p)
        except OSError:
            continue
        key = (st.st_dev, st.st_ino)
        if key not in seen:
            seen.add(key)
            total += st.st_blocks * 512
    return total


def _gb(n: int) -> str:
    return f"{n / 1e9:6.2f} GB"


# ------------------------------------------------------------ what to keep --

def referenced_checkpoints(root: Path = ROOT) -> set[str]:
    """Every checkpoint the code names outright, plus its `.json` manifest.

    Commented-out rungs are skipped on purpose: they are not loaded. Only
    tracked files are returned, so a stale reference cannot become a pattern.
    """
    refs: set[str] = set()
    for base in SCAN_DIRS:
        for py in sorted((root / base).rglob("*.py")):
            for line in py.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lstrip().startswith("#"):
                    continue
                refs.update(REF.findall(line))
    tracked = set(_git("ls-files", "-z", CKPT, cwd=root).split("\0")) - {""}
    keep = {r for r in refs if r in tracked}
    keep |= {str(Path(r).with_suffix(".json")) for r in keep} & tracked
    return keep


def sparse_patterns(keep: set[str]) -> str:
    """Non-cone patterns: everything, minus every checkpoint file, plus the kept ones.

    Files (not directories) are excluded, so the per-file re-includes below
    are allowed to fire -- gitignore syntax cannot re-include beneath an
    excluded directory.
    """
    lines = ["/*", f"!/{CKPT}/**/*.zip", f"!/{CKPT}/**/*.json"]
    lines += [f"/{p}" for p in sorted(keep)]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- safety --

def unsafe_reasons() -> list[str]:
    """Everything that would exist only on this disk, and so must not be
    left behind by a reclone or hidden by a sparse checkout."""
    reasons = []
    _git("fetch", "--quiet", "--all", check=False)
    dirty = [l for l in _git("status", "--porcelain", "--untracked-files=no").splitlines() if l]
    if dirty:
        reasons.append(f"{len(dirty)} tracked file(s) modified or staged -- commit or stash them")
    if _git("stash", "list").strip():
        reasons.append("the stash is not empty -- it exists only in this clone")
    for line in _git("for-each-ref", "--format=%(refname:short)\t%(upstream:short)",
                     "refs/heads").splitlines():
        branch, _, upstream = line.partition("\t")
        if not upstream:
            if not _git("branch", "-r", "--contains", branch, check=False).strip():
                reasons.append(f"branch {branch!r} is on no remote branch -- push it first")
            continue
        ahead = _git("rev-list", "--count", f"{upstream}..{branch}", check=False).strip()
        if ahead and ahead != "0":
            reasons.append(f"branch {branch!r} is {ahead} commit(s) ahead of {upstream} -- push it")
    return reasons


def local_only_paths() -> list[str]:
    """Untracked and ignored top-level entries, as git collapses them."""
    out = _git("status", "--porcelain=v1", "-z", "--ignored=traditional",
               "--untracked-files=normal")
    return sorted(e[3:] for e in out.split("\0") if e[:2] in ("??", "!!"))


# ----------------------------------------------------------------- modes --

def survey() -> None:
    keep = referenced_checkpoints()
    tracked = set(_git("ls-files", "-z", CKPT).split("\0")) - {""}
    seen: set[tuple[int, int]] = set()
    kept = sum(_disk(ROOT / p, seen) for p in keep)
    hidden = sum(_disk(ROOT / p, seen) for p in tracked - keep)
    untracked = [p for p in local_only_paths() if p.startswith(CKPT)]
    rows = [
        (".git (history, incl. a 2nd copy of every checkpoint)", _disk(ROOT / ".git")),
        (f"{CKPT}: loaded by the code ({len(keep)} files)", kept),
        (f"{CKPT}: not loaded, tracked, on GitHub ({len(tracked - keep)} files)", hidden),
        (f"{CKPT}: UNTRACKED -- only on this disk", sum(_disk(ROOT / p) for p in untracked)),
        ("data/  (gated dataset, ignored -- keep, do not re-download)", _disk(ROOT / "data")),
        (".venv/ (rebuildable from requirements.txt)", _disk(ROOT / ".venv")),
    ]
    other = [p for p in local_only_paths()
             if not p.startswith((CKPT, "data", ".venv")) and "__pycache__" not in p]
    for p in other:
        rows.append((f"{p}  (untracked/ignored)", _disk(ROOT / p)))
    print(f"clone: {ROOT}\n")
    for label, size in rows:
        if size:
            print(f"  {_gb(size)}  {label}")
    print(f"  {'-' * 9}\n  {_gb(_disk(ROOT))}  total\n")
    if untracked:
        print("  untracked checkpoint files (NOT on GitHub; both modes keep them):")
        for p in untracked:
            print(f"      {p}")
        print()
    reasons = unsafe_reasons()
    if reasons:
        print("  NOT safe to slim yet:")
        for r in reasons:
            print(f"    - {r}")
    else:
        print("  safe to slim: tree clean, no stash, every local branch pushed.")
    git_bytes = _disk(ROOT / ".git")
    print(f"\n  --sparse  frees ~{_gb(hidden).strip()} (working tree only; undo with --restore)")
    print(f"  --reclone frees ~{_gb(hidden + git_bytes).strip()} minus the new clone's own "
          ".git (history of small files, typically well under 1 GB)")


def sparse(apply: bool) -> None:
    if not apply:
        _git("sparse-checkout", "disable")
        print("restored: every tracked file is checked out again.")
        return
    reasons = unsafe_reasons()
    if reasons:
        sys.exit("refusing --sparse:\n" + "\n".join(f"  - {r}" for r in reasons))
    keep = referenced_checkpoints()
    subprocess.run(("git", "sparse-checkout", "set", "--no-cone", "--stdin"), cwd=ROOT,
                   input=sparse_patterns(keep), text=True, check=True)
    missing = [p for p in keep if not (ROOT / p).exists()]
    if missing:
        sys.exit(f"sparse checkout left {len(missing)} referenced file(s) missing, "
                 f"e.g. {missing[0]} -- run --restore")
    print(f"sparse checkout applied: {len(keep)} checkpoint files kept.")
    print("undo:   python3 scripts/slim_clone.py --restore")


def _copy(src: Path, dst: Path) -> None:
    """Copy-on-write on APFS (`cp -c`: no extra space), plain copy elsewhere."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Darwin":
        done = subprocess.run(("cp", "-cRp", str(src), str(dst)), capture_output=True)
        if done.returncode == 0:
            return
    subprocess.run(("cp", "-Rp", str(src), str(dst)), check=True)


def reclone(dest: Path) -> None:
    dest = dest.expanduser().resolve()
    if dest.exists():
        sys.exit(f"{dest} already exists -- pick a path that does not")
    reasons = unsafe_reasons()
    if reasons:
        sys.exit("refusing --reclone:\n" + "\n".join(f"  - {r}" for r in reasons))
    url = _git("remote", "get-url", "origin").strip()
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
    head = _git("rev-parse", "HEAD").strip()
    keep = referenced_checkpoints()

    print(f"cloning {url} ({branch}) with --filter=blob:limit={BLOB_LIMIT} ...")
    subprocess.run(("git", "clone", f"--filter=blob:limit={BLOB_LIMIT}", "--no-checkout",
                    "--branch", branch, url, str(dest)), check=True)
    subprocess.run(("git", "sparse-checkout", "set", "--no-cone", "--stdin"), cwd=dest,
                   input=sparse_patterns(keep), text=True, check=True)
    subprocess.run(("git", "checkout", branch), cwd=dest, check=True)
    if _git("rev-parse", "HEAD", cwd=dest).strip() != head:
        sys.exit(f"new clone is not at {head[:10]} -- stopping, old clone untouched")
    missing = [p for p in keep if not (dest / p).exists()]
    if missing:
        sys.exit(f"{len(missing)} referenced checkpoint(s) missing in the new clone, "
                 f"e.g. {missing[0]} -- stopping, old clone untouched")

    for rel in local_only_paths():
        target = dest / rel.rstrip("/")
        if target.exists():
            print(f"  skip (exists in new clone): {rel}")
            continue
        print(f"  copy {rel}")
        _copy(ROOT / rel.rstrip("/"), target)

    old = ROOT
    print(f"\nnew clone: {dest}  ({_gb(_disk(dest)).strip()} on disk, "
          f"old: {_gb(_disk(old)).strip()})")
    print("check it, then swap the two over (keeps .venv's absolute paths valid):\n")
    print(f"  cd {dest} && python3 scripts/doctor.py && python -m pytest -q")
    print(f'  mv "{old}" "{old}-old" && mv "{dest}" "{old}"')
    print(f'  # once you are happy:  rm -rf "{old}-old"')
    print("\nhidden snapshots download on demand: "
          "git sparse-checkout add '/runs/checkpoints/<v>/<model_name>/**'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--sparse", action="store_true", help="hide unreferenced checkpoints in place")
    mode.add_argument("--restore", action="store_true", help="undo --sparse")
    mode.add_argument("--reclone", metavar="DEST", type=Path,
                      help="build a slim partial clone at DEST; this clone is left untouched")
    args = ap.parse_args()
    if args.sparse or args.restore:
        sparse(apply=args.sparse)
    elif args.reclone:
        reclone(args.reclone)
    else:
        survey()


if __name__ == "__main__":
    main()
