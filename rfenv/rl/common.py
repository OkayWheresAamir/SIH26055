"""Algorithm-agnostic pieces: the training env factory, the policy adapters,
and the training-time callback.

Every SB3 algorithm trains on the same env, so `make_train_env` belongs to
none of them specifically. Two policy adapters live here, one per `.predict()`
shape a scheduler might need to wrap -- not one per algorithm:

  `RLScheduler`           any feed-forward algorithm's plain
                          `predict(obs, deterministic=...)` -- DQN, PPO today.
  `RecurrentRLScheduler`  any recurrent (sb3-contrib) algorithm's
                          `predict(obs, state=..., episode_start=..., ...)` --
                          RecurrentPPO today (`recurrent_ppo.py`, rung 9).

`EpisodeMetricsCallback` is the third: `model.learn(callback=...)` accepts the
same `BaseCallback` regardless of algorithm, so it lives here too rather than
in any one `train()`. `training_callbacks()` assembles the callback list every
`train()` passes to `model.learn()` from the same two toggles each one
exposes (`print_episode_metrics`, `checkpoint_freq`) -- one shared place for
that assembly rather than three copies of the same few lines.

The fourth shared piece is the **checkpoint manifest** (`write_manifest`,
`read_manifest`, `require_loadable`). Every `.zip` this package writes gets a
`.json` sidecar beside it recording what produced it. This exists because it
was reconstructed the hard way: four RecurrentPPO training series were once
found on disk with identical seeds and identical recorded hyperparameters but
four different sets of weights, and nothing anywhere said what differed
between them -- SB3 serialises the algorithm's constructor arguments but not
the reward function, not the argv, not the commit, and not the observation
width the env had at the time. The width is the one that matters most: a
checkpoint trained against a narrower observation loads without complaint and
only fails later, at `predict()`, with a shape error that names no cause. The
manifest turns that into `require_loadable()` refusing up front with the
training command that would rebuild it.

Only `train()`, `load_checkpoint()` and the default checkpoint path (one file
per algorithm: `dqn.py`, `ppo.py`, `recurrent_ppo.py`) are algorithm-specific.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from rfenv.env import DEFAULT_REWARD, ScanEnv
from rfenv.scenario import EmitterPool


def make_train_env(*, reward: str = DEFAULT_REWARD) -> ScanEnv:
    """A fresh-scenario-per-reset training env (D25, D32) -- what RL trains on.

    `reward` is threaded straight through to ScanEnv, which is the whole
    "reward as a hyperparameter" requirement -- ScanEnv already validates it
    against `REWARDS`.
    """
    return ScanEnv(pool=EmitterPool.from_train(), reward=reward)


# --------------------------------------------------------------------------- #
# The checkpoint manifest
# --------------------------------------------------------------------------- #

# Read off `guard.py` rather than recomputed from N_BANDS here, so the width
# has exactly one definition: guard.py is already the module that names the
# observation's layout, and its last index + 1 *is* the width by construction.
def current_observation_width() -> int:
    """How wide `ScanEnv`'s observation is *right now*, without building one.

    Deliberately does not instantiate a `ScanEnv`: this is called from
    `require_loadable()` on the failure path, where constructing an env would
    load the emitter pool off disk, and where the env may not be constructible
    at all (a bad `DEFAULT_REWARD` raises in `ScanEnv.__init__`, which would
    then mask the shape mismatch this is trying to report).
    """
    from rfenv.baselines.guard import MEASURED_DBM
    return MEASURED_DBM + 1


def manifest_path(checkpoint: str | Path) -> Path:
    """The `.json` sidecar for a checkpoint: `foo.zip` -> `foo.json`."""
    return Path(checkpoint).with_suffix(".json")


def _git_state() -> dict:
    """The commit the training ran at, and whether the tree was dirty.

    Both `None` if this is not a git checkout or git is not on PATH -- a
    missing commit is recorded as missing, never guessed. `dirty` is what makes
    the commit honest: a SHA alone implies the code is recoverable from it,
    which is false the moment anything is uncommitted, and in this repository
    the RL lane has spent most of its life uncommitted.
    """
    def _git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ("git", *args), cwd=Path(__file__).resolve().parent,
                capture_output=True, text=True, timeout=15, check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip()

    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _command() -> str:
    """`sys.argv` rebuilt into something that can actually be pasted into a shell.

    `argv[0]` is not usable as-is: `python -m rfenv.rl.ppo` rewrites it to the
    module's *file path*, so a naive join produces a command that runs a file
    rather than a module -- which for `rfenv.rl` (whose entry point is
    `_cli.py`) is not even the same thing. The interpreter's own `__main__`
    spec is what still knows the module name, so use it when there is one.

    The verbatim `argv` is recorded separately and is the record of what ran;
    this field is only the convenience form, for pasting.
    """
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is None:
        head = [sys.executable, *sys.argv[:1]]
    else:
        name = spec.name.removesuffix(".__main__")
        head = [sys.executable, "-m", name]
    return subprocess.list2cmdline([*head, *sys.argv[1:]])


def _versions() -> dict:
    """Library versions that determine whether a checkpoint deserialises."""
    import gymnasium
    import stable_baselines3
    import torch

    versions = {
        "python": platform.python_version(),
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
    }
    try:
        import sb3_contrib
    except ImportError:
        pass
    else:
        versions["sb3_contrib"] = sb3_contrib.__version__
    return versions


def _hardware() -> dict:
    import torch

    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# Constructor arguments worth reading back off the trained model. Not every
# algorithm has all of them (DQN has no ent_coef, PPO no exploration_fraction);
# whatever is absent is simply skipped.
_RESOLVED_KEYS = (
    "learning_rate", "gamma", "n_steps", "batch_size", "n_epochs", "ent_coef",
    "gae_lambda", "vf_coef", "max_grad_norm", "target_update_interval",
    "exploration_fraction", "exploration_final_eps", "buffer_size",
    "learning_starts", "tau",
)


def write_manifest(
    checkpoint: str | Path,
    *,
    model,
    run: str,
    reward: str,
    hyperparameters: dict,
    started_at: float,
    description: str = "",
) -> Path:
    """Write `<checkpoint>.json` describing the run that produced it.

    Called for **every** `.zip` this package saves -- the final one each
    `train()` writes and every intermediate snapshot `--checkpoint-freq`
    produces -- so there is no such thing here as a checkpoint whose origin
    has to be reconstructed.

    Two separate hyperparameter records, and the distinction is the point:

      `hyperparameters`  what the caller actually passed to the algorithm's
                         constructor. Empty dict means "everything was left at
                         the library default", which is a real and reportable
                         fact rather than an absence of information.
      `resolved`         what the constructed model ended up holding for each
                         of `_RESOLVED_KEYS`. This is what a later reader
                         actually needs, because a library default is only
                         knowable if you also know the library version.

    `sha256` is of the `.zip` as written, so a manifest can be checked against
    its checkpoint rather than merely sitting next to one.
    """
    checkpoint = Path(checkpoint)
    resolved = {}
    for key in _RESOLVED_KEYS:
        if not hasattr(model, key):
            continue
        value = getattr(model, key)
        # learning_rate and clip_range are schedules by the time SB3 stores
        # them; a schedule reprs as a closure address, which is not a fact.
        resolved[key] = value if isinstance(value, (int, float, str, bool, type(None))) else repr(value)

    manifest = {
        "run": run,
        "description": description,
        "checkpoint": checkpoint.name,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "algorithm": type(model).__name__,
        "policy": type(model.policy).__name__,
        "reward": reward,
        "observation_width": int(model.observation_space.shape[0]),
        "action_space": str(model.action_space),
        "seed": model.seed,
        "total_timesteps": int(model.num_timesteps),
        "hyperparameters": hyperparameters,
        "resolved": resolved,
        "argv": list(sys.argv),
        "command": _command(),
        "git": _git_state(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "wall_clock_s": round(time.time() - started_at, 1),
        "versions": _versions(),
        "hardware": _hardware(),
    }
    path = manifest_path(checkpoint)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def parse_hyperparameters(pairs) -> dict:
    """`["ent_coef=0.01", "n_steps=512"]` -> `{"ent_coef": 0.01, "n_steps": 512}`.

    Values go through `json.loads` so `0.01` arrives as a float and `512` as an
    int -- SB3 type-checks several of these and a string `"512"` fails
    unhelpfully deep inside the constructor. Anything JSON cannot parse is kept
    as the literal string, which is what a policy name or a device string needs.

    Every `rfenv.rl` CLI exposes this as a repeatable `--hyperparam KEY=VALUE`.
    It is the only way to train anything here that is not at library defaults,
    and whatever it produces is exactly what the manifest records as "passed".
    """
    out = {}
    for pair in pairs or ():
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--hyperparam wants KEY=VALUE, got {pair!r}")
        try:
            out[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            out[key.strip()] = value
    return out


def add_manifest_arguments(ap) -> None:
    """The `--run-name`/`--description`/`--hyperparam` flags, on every RL CLI.

    Shared so a new algorithm's CLI gets the manifest's inputs by calling one
    function, in the same spirit as `training_callbacks` and `finish_training`.
    """
    ap.add_argument("--run-name", default=None,
                    help="name for this training run, used for the manifest and for "
                         "snapshot filenames (default: the checkpoint's stem)")
    ap.add_argument("--description", default="",
                    help="what this run is trying -- recorded in the manifest, which is "
                         "where a rung's label should come from")
    ap.add_argument("--hyperparam", action="append", metavar="KEY=VALUE",
                    help="a constructor argument to pass to the algorithm, repeatable "
                         "(e.g. --hyperparam ent_coef=0.01). Recorded verbatim.")


def read_manifest(checkpoint: str | Path) -> dict | None:
    """The checkpoint's manifest, or `None` if it has none.

    `None` is not an error: every checkpoint trained before manifests existed
    has no sidecar, and those still load. It only means the width check in
    `require_loadable()` cannot run.
    """
    path = manifest_path(checkpoint)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def checkpoint_observation_width(checkpoint: str | Path) -> int | None:
    """The observation width recorded *inside* the SB3 archive, or `None`.

    The manifest is the better source -- it also knows the reward and the argv
    -- but it only exists for checkpoints trained since manifests did, and the
    ones most likely to be stale are exactly the older ones. So fall back to
    this: every SB3 `.zip` carries a JSON `data` member holding the pickled
    `observation_space` the model was constructed with, which is
    non-fabricatable and present in every checkpoint ever written.

    Read straight out of the archive rather than by loading the model, because
    loading a stale checkpoint is the thing being avoided, and because this has
    to work on the failure path where torch may be the slow part.
    """
    import base64
    import pickle
    import zipfile

    try:
        with zipfile.ZipFile(checkpoint) as archive:
            data = json.loads(archive.read("data").decode("utf-8"))
        space = pickle.loads(base64.b64decode(data["observation_space"][":serialized:"]))
        return int(space.shape[0])
    except Exception:      # noqa: BLE001 -- any unreadable archive is simply unknown
        return None


def require_loadable(checkpoint: str | Path) -> None:
    """Refuse, loudly and before loading, if the checkpoint's shape is wrong.

    Every `load_checkpoint()` in this package calls this first. Without it a
    stale checkpoint loads perfectly happily -- `Algorithm.load()` binds no env
    and so checks nothing -- and only fails much later inside `predict()`,
    during a comparison run, with a bare `Unexpected observation shape` that
    names neither the checkpoint nor the reason. That failure mode is what this
    whole check exists to remove; measured, it accounted for 74 of the suite's
    failures at once when six stale checkpoints were restored to disk.

    Two sources for the width, in order: the manifest, which can also say what
    command would rebuild it, and failing that the archive's own recorded
    `observation_space`. A checkpoint with neither is allowed through -- there
    is then nothing to check against, and refusing on ignorance would break
    rungs that are actually fine.
    """
    current = current_observation_width()
    manifest = read_manifest(checkpoint) or {}
    recorded = manifest.get("observation_width")
    if recorded is None:
        recorded = checkpoint_observation_width(checkpoint)
    if recorded is None or recorded == current:
        return

    detail = ""
    if manifest:
        detail = (
            f"  run:       {manifest.get('run')}\n"
            f"  reward:    {manifest.get('reward')}\n"
            f"  trained:   {manifest.get('started_at')} at commit "
            f"{(manifest.get('git') or {}).get('commit')}\n"
            f"  retrain:   {manifest.get('command')}\n"
        )
    else:
        detail = (
            "  (no manifest -- width read from the archive itself. Nothing records "
            "what reward\n   or command produced this checkpoint; retrain it to get "
            "one.)\n"
        )
    raise ValueError(
        f"{Path(checkpoint).name} was trained against a {recorded}-wide observation, "
        f"but ScanEnv now builds a {current}-wide one -- this checkpoint cannot be "
        f"used and will fail inside predict() if forced.\n"
        f"{detail}"
        f"(D49 -- an observation change invalidates every checkpoint that predates it. "
        f"Retrain, or compare against the environment it was trained on.)"
    )


def checkpoint_is_usable(checkpoint: str | Path) -> bool:
    """Whether `load_checkpoint()` would accept this file: it exists and its
    observation width matches the current environment.

    For callers that need to *decide* rather than fail -- `compare.py` skipping
    an unbuildable rung, and the ladder's generic per-rung tests skipping one.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        return False
    try:
        require_loadable(checkpoint)
    except ValueError:
        return False
    return True


class RLScheduler:
    """A trained model as a `(obs, info) -> band` callable, per rollout.run_episode.

    Never reads `info` -- the policy conditions on the D34 observation alone,
    which is what makes it deployable. `guarded()` in baselines/ still strips
    `info` before this ever sees it; this class simply never asks.

    Wraps any SB3 algorithm that exposes plain `.predict()` -- today that's
    DQN and PPO (see `dqn.py`/`ppo.py`), but nothing here is specific to either.
    Stateless: any instance can be reused, or shared, across episodes, because
    a feed-forward policy carries nothing forward between calls.

    `deterministic` defaults to False -- see `RecurrentRLScheduler` for the
    measurement that settled that default. It applies to any policy whose action
    distribution stays broad, which is every on-policy algorithm here; a DQN's
    greedy argmax *is* its policy, so passing `deterministic=True` is the honest
    choice for rung 7 and the flag exists to say so explicitly.
    """

    def __init__(self, model: DQN, *, deterministic: bool = False):
        self.model = model
        self.deterministic = deterministic

    def __call__(self, obs, info) -> int:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)


class RecurrentRLScheduler:
    """A trained recurrent (LSTM) model as a `(obs, info) -> band` callable.

    Reads exactly one field of `info`: `slot`, already in `OBSERVABLE_INFO`
    (D19/D20 -- the receiver's own clock, not a truth-side quantity), used
    only to detect a new episode. Never conditions the action on anything else
    in `info`, so it's deployable by the same rule `RLScheduler` follows.

    **Not stateless, unlike `RLScheduler`.** A recurrent policy's `.predict()`
    takes the previous call's hidden state (`state=...`) and returns the next
    one; this class carries that state between calls so the LSTM actually
    accumulates memory across an episode instead of starting fresh every dwell.
    `episode_start` is derived from `info["slot"] == 0` -- true only on the
    very first call after `ScanEnv.reset()`, before any `step()` -- rather than
    tracked by hand, since `rollout.run_episode` itself gives a policy no other
    signal that a new episode has begun. Passing `episode_start=True` tells the
    recurrent policy to disregard whatever `self._lstm_state` still holds from
    the previous episode and start that hidden state from zero internally --
    this class doesn't need to reset the field itself for that to happen.

    **One instance per episode is the safe default.** Reusing one across
    episodes still works correctly (the `episode_start` signal resets the
    policy's own hidden state every time), but a fresh instance per episode is
    cheap and avoids ever wondering whether stale state leaked across a reset.

    **`deterministic` defaults to False, and that is load-bearing.** This class
    hardcoded `deterministic=True`, and every "the agent learned to camp" result
    on this rung came from it. Measured on `lstm_gamma997` over a seed-0
    episode, the policy's own action distribution has mean entropy 2.369 against
    3.584 for uniform-over-36 and a mean max-probability of 0.206 -- it has not
    collapsed, it is broad. But its argmax is *sticky*: the mode landed on the
    same band for 580 of 586 steps, so taking the argmax turns a broad policy
    into a one-band camper. Same checkpoint, same seeds: argmax visits 2 distinct
    bands and scores coverage 0.247/0.041, while sampling visits 31 and scores
    0.603/0.714. PPO optimises expected return under the sampled policy and never
    evaluates the mode, so sampling is also what the reported training return
    actually measured. Pass `deterministic=True` only to reproduce an older row.
    """

    def __init__(self, model, *, deterministic: bool = False):
        self.model = model
        self.deterministic = deterministic
        self._lstm_state = None

    def __call__(self, obs, info) -> int:
        episode_start = np.array([info.get("slot", 0) == 0])
        action, self._lstm_state = self.model.predict(
            obs,
            state=self._lstm_state,
            episode_start=episode_start,
            deterministic=self.deterministic,
        )
        return int(action)


class EpisodeMetricsCallback(BaseCallback):
    """Prints `ScanEnv.episode_metrics()` to stdout after every training episode.

    Reads it from `info["episode_metrics"]` (see `ScanEnv._info`), never by
    calling `env.episode_metrics()` from here directly. Every `rfenv.rl`
    trainer's env ends up wrapped in an SB3 `DummyVecEnv`, and a `DummyVecEnv`
    auto-resets a sub-env the instant its episode ends -- *before* any
    callback gets a chance to run. By the time `_on_step` fires, the
    underlying `ScanEnv` has already moved on to the next episode, so calling
    `episode_metrics()` here would silently describe the wrong one (measured:
    it reads back `n_steps=0`, the fresh episode, every time). `step()` avoids
    that by snapshotting the metrics into `info` itself, before any of that
    happens -- this callback only ever reads the snapshot.

    Pass to `model.learn(callback=EpisodeMetricsCallback())`; every `train()`
    in this package accepts a `print_episode_metrics: bool` that wires this in.
    """

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            metrics = info.get("episode_metrics")
            if metrics is not None:
                print(metrics)
        return True


class ManifestedCheckpointCallback(CheckpointCallback):
    """SB3's `CheckpointCallback`, with two changes.

    **It writes a manifest beside every snapshot**, via `write_manifest`, so an
    intermediate checkpoint is exactly as self-describing as a final one. This
    is the case that actually went wrong historically: the snapshots are the
    files that get registered as separate ladder rungs, and they were the ones
    nobody could later attribute to a run.

    **It names snapshots for the run, not the timestep count.** SB3's own
    convention is `<prefix>_<num_timesteps>_steps.zip`; this uses
    `<run>_s1.zip`, `<run>_s2.zip`, ... and puts `total_timesteps` in the
    manifest instead. A count in a filename reads as a label, gets copied into
    a rung key and a rung label by hand, and then drifts from the archive --
    which has already happened twice here (rung 8a says "800k" for a 600,064-step
    checkpoint; rung 7a says "5,000 timesteps" for a 20,000-step one). An
    ordinal cannot drift, because it claims nothing.
    """

    def __init__(self, *, run: str, manifest_kwargs: dict, **kwargs):
        super().__init__(**kwargs)
        self._run = run
        self._manifest_kwargs = manifest_kwargs
        self._n_saved = 0

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        return os.path.join(
            self.save_path, f"{self._run}_{checkpoint_type}s{self._n_saved + 1}.{extension}"
        )

    def _on_step(self) -> bool:
        due = self.n_calls % self.save_freq == 0
        super()._on_step()
        if due:
            path = self._checkpoint_path(extension="zip")
            write_manifest(path, model=self.model, run=self._run,
                           **self._manifest_kwargs)
            self._n_saved += 1
        return True


def training_callbacks(
    *,
    print_episode_metrics: bool,
    checkpoint_freq: int | None,
    checkpoint: str | Path,
    run: str,
    manifest_kwargs: dict,
) -> list[BaseCallback]:
    """The callback list every `train()` passes to `model.learn(callback=...)`,
    built from the two toggles every one of them exposes.

    `checkpoint_freq`, when not `None`, adds `ManifestedCheckpointCallback`,
    saving an *additional* snapshot every `checkpoint_freq` steps -- not a
    replacement for the final `model.save(checkpoint)` each `train()` still
    does at the end, but a way to get several checkpoints from one run instead
    of only the last one, e.g. to compare a rung at 200k/400k/600k timesteps
    without re-running training three times. Snapshots land in `checkpoint`'s
    own directory as `<run>_s1.zip`, `<run>_s2.zip`, ... each with its `.json`
    beside it; `runs/checkpoints/` stays flat, one `.zip` plus one `.json` per
    saved model, and the manifest -- not the filename -- carries the timestep
    count and the description. Each snapshot is registrable as its own ladder
    rung exactly like any other trained checkpoint (see run.md).

    `save_freq` is passed through as-is, not divided by `n_envs` (SB3's own
    docs flag that division as necessary for a multi-env `VecEnv`) -- every
    `rfenv.rl` trainer always trains on exactly one env, so `n_envs == 1` and
    the division would be a no-op anyway.

    An empty list if neither toggle is set, which `model.learn(callback=...)`
    accepts the same as `None` -- callers don't need to special-case it.
    """
    callbacks: list[BaseCallback] = []
    if print_episode_metrics:
        callbacks.append(EpisodeMetricsCallback())
    if checkpoint_freq is not None:
        callbacks.append(ManifestedCheckpointCallback(
            run=run,
            manifest_kwargs=manifest_kwargs,
            save_freq=checkpoint_freq,
            save_path=str(Path(checkpoint).parent),
        ))
    return callbacks


def finish_training(
    model,
    *,
    checkpoint: str | Path,
    run: str | None,
    reward: str,
    hyperparameters: dict,
    started_at: float,
    description: str = "",
) -> Path:
    """Save the final checkpoint and its manifest. The tail of every `train()`.

    One shared place so that DQN, PPO, RecurrentPPO and anything added later
    cannot save a checkpoint without also describing it -- the manifest is not
    an extra step a new algorithm's `train()` has to remember, it is part of
    what saving means here.

    `run` defaults to the checkpoint's own stem, which makes the common case
    (`--checkpoint runs/checkpoints/lstm_ppo4.zip`) name its run correctly with
    no extra flag.
    """
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    write_manifest(checkpoint, model=model, run=run or checkpoint.stem,
                   reward=reward, hyperparameters=hyperparameters,
                   started_at=started_at, description=description)
    return checkpoint
