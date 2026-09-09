Read these first, in full, before doing anything: CLAUDE.md,
docs/protocol/CLAUDE_CODE_RESEARCH_PROTOCOL.md, docs/project/RL_TEAM_HANDOFF.md
(especially §16, §17, §18), and skim docs/project/DECISIONS.md for house style.

THE SITUATION. Over the RL lane I built and trained rungs 7 (DQN) and 8 (PPO),
changed the observation vector from 109 to 146, rewrote reward_first_intercept,
and trained at least six checkpoint variants. Almost none of that was written
down as it happened. The code was merged upstream; the evidence and the
reasoning were not. I now need to reconstruct the record from whatever evidence
still exists on this machine, and package the results so my teammate can pick
it up.

THE ONE RULE THAT GOVERNS THIS WHOLE TASK. This repository exists because a
previous attempt accumulated numbers nobody could trace. A reconstructed record
that is confidently wrong is far worse than one with visible holes. So label
EVERY reconstructed fact with exactly one of these tags, inline:

  [RECOVERED]     I can point at a file, a command's output, or a commit.
                  Cite it in the same sentence.
  [RECALLED]      It appears in a session transcript but I cannot verify it
                  against the repo. Quote the transcript and say so.
  [UNRECOVERABLE] The evidence is gone. State plainly what is missing and
                  what it would take to recover it. LEAVE IT BLANK.

Never smooth over a gap. Never infer a hyperparameter from what "would be
typical." If you catch yourself writing a number you did not read off something,
stop and tag it [UNRECOVERABLE] instead. I would rather hand over ten solid
facts and three admitted holes than thirteen plausible ones.

Do not touch data/turing/**/test_* at any point.

---

PHASE 0 — PROTECT THE EVIDENCE. Do this before anything else, before you
read a single file. Copy runs/ to runs_backup_<today's date>/ and leave that
copy alone for the rest of this task. Phase 7 writes to runs/baselines and will
overwrite what is there; the checkpoints and logs in runs/ are the only surviving
record of the training journey and they are not in git. If a later phase needs
to compare against the pre-existing artefacts, read them from the backup.

Confirm the copy exists and report its size before continuing.

---

PHASE 1 — GATHER EVIDENCE. Do not edit a single document yet. Collect first,
into a scratch file. Sources, roughly in order of trustworthiness:

1. The checkpoints themselves — the best source, and non-fabricatable. Every
   SB3 .zip in runs/checkpoints/ is an archive containing a JSON `data` member
   with the model's actual constructor arguments. Unzip each one (or load it
   and inspect) and extract: learning_rate, gamma, n_steps, batch_size,
   n_epochs, ent_coef, clip_range, gae_lambda, vf_coef, max_grad_norm,
   policy_kwargs/net_arch, seed, num_timesteps, and the recorded
   observation_space and action_space. Also read system_info.txt and
   _stable_baselines3_version. Do this for EVERY checkpoint, not just the best
   one. Record each file's SHA-256 and its mtime.

   The observation_space width is a dating tool: a 109-wide checkpoint predates
   the observation change, a 146-wide one postdates it. Use that to order the
   training runs against the code history.

2. Training logs — runs/, tensorboard event files, monitor.csv, any nohup or
   console dumps. These give you the learning curves and the wall-clock times.

3. Git history — `git log --all --oneline --stat`, `git reflog`, and any
   stashes or unmerged local branches. `git log -p` on rfenv/env.py and
   rfenv/baselines/ladder.py specifically: the order in which rungs got
   registered IS the training journey, and the ladder's rung names encode the
   timestep counts.

4. Claude Code session transcripts — these persist on disk even though nothing
   was written to the repo. Look in ~/.claude/projects/ (on Windows:
   C:\Users\<you>\.claude\projects\) for the directory whose name matches this
   repo's path; each session is a .jsonl file. Read them oldest-first, ALL of
   them, not just the recent ones.

   This is the only place the reasoning survives, and it is also the only place
   that records decisions YOU (Claude) took autonomously without ever putting
   them to me for approval. Those are a first-class part of what I need
   recovered — see Phase 4b. Treat transcript content as [RECALLED], not
   [RECOVERED]: it records what was said, not what was true. Where a transcript
   claims a measurement, re-run the measurement rather than quoting it.

5. Anything else on disk — scratch scripts, notebooks, uncommitted files,
   shell history.

---

PHASE 2 — RECONSTRUCT THE TIMELINE. Write scratch/TRAINING_JOURNEY.md: what
happened, in the order it happened, one entry per meaningful step. For each:
what changed, why, what the measured effect was, what decision it embodied, and
its evidence tag.

The questions this must answer, because they are the ones being asked:
  - Why did the observation go 109 -> 146? What did current_band and camp_time
    buy, and was it ever measured against the 109 vector?
  - Why was reward_first_intercept rewritten from `len(newly)` to the flat
    +3/0/-0.5 shaping, and to per-(emitter,band) novelty?
  - What was weighted_camp, why was it tried, and why was it retired?
  - What order were the checkpoints trained in, and what changed between them?
  - Which one is the best, and by what measure, decided when?
  - What was tried that did NOT work? (This is worth as much as the wins.)

---

PHASE 3 — STOP AND SHOW ME. Present the timeline, the recovered hyperparameter
table, and the draft decision ledger headings before writing anything into
docs/. Flag every [UNRECOVERABLE] item explicitly so I can fill in from memory
while I still remember. Wait for me. Do not proceed on your own judgement here.

---

PHASE 4a — DRAFT THE CONTRACT DECISIONS. Write HANDOVER/DECISIONS_DRAFT.md
containing D-1 .. D-13 exactly as RL_TEAM_HANDOFF.md §16.1 specifies them, in
the §18 format (Status / The decision / Evidence / Alternatives rejected). Do
NOT write into docs/project/DECISIONS.md — §18 says my teammate assigns the
real D4x numbers.

Beyond the thirteen, three more are owed because they were taken silently and
are not in the brief's list:

  D-14  The observation change 109 -> 146. This modified D34, which is SETTLED,
        and D34's own text says extensions "get logged as decisions." It
        invalidates every 109-wide checkpoint and stales five documents. Give
        the reasoning and any measurement; if there is no ablation against the
        109 vector, say so outright.
  D-15  The reward_first_intercept rewrite. Note that it no longer matches
        D28's definition of first intercept (D28 is per-emitter; this is now
        per-(emitter,band)) and that this changes what D47's selection rule is
        selecting between.
  D-16  weighted_camp: tried and retired. A negative result — write it up as
        one, with the number that killed it if there is one.

D-8 (D30 — do AoA and PulseWidth enter the observation?) is the only OPEN
decision blocking the freeze and it is explicitly this lane's to answer. If it
was never addressed, say "not addressed" rather than closing it either way.

---

PHASE 4b — THE DECISION LEDGER. This is separate from 4a and just as important.
Write HANDOVER/DECISION_LEDGER.md: EVERY choice made over this lane, not only
the sixteen above. Including the small ones. Including the ones made inside an
already-agreed design. Including — especially — the ones you took on your own
without asking me, which I may not know were ever decided.

Three columns of provenance for each, and be honest about the third:

  WHO   me (explicitly asked for it) / you (took it and told me) /
        you (took it and never surfaced it)
  WHY   the reason at the time. A decision without a why is not reusable, and
        the "why" is what the SIH deck is made of.
  STILL STANDS?  yes / superseded by X / should be revisited

Non-exhaustive list of the kind of thing I mean, so you calibrate low enough:
why camp_slots counts slots rather than dwells; why camp_time normalises by
N_SLOTS; why the flat +3 rather than 3*len(newly); why every reward candidate
takes camp_slots even though none reads it; why newly widened to per-band and
why first_e was deliberately left alone; why baselines/metrics/render were split
into subpackages and why run_episode moved to rollout.py; why ladder.py uses
literal checkpoint paths instead of importing DEFAULT_CHECKPOINT; why the rung
factories import lazily; why rung keys encode timestep counts; why rung 8d was
registered at all; why deterministic=True (or not) at inference; why an
rgb_array render mode was added; why the encoding="utf-8" fixes were needed;
why gif_stride defaults to 8; why the Pareto figure was rekeyed by rung key.

Sort it: architectural decisions first (anything touching the environment, the
receiver, ground truth, the reward, the observation, the evaluation protocol, or
the scheduler design — CLAUDE.md gates all of these and several were taken
without a gate), then routine implementation choices. Mark clearly which of the
architectural ones went through no approval, because my teammate needs to
review exactly those.

If a transcript shows I approved something, quote it. If it shows you decided
and moved on, say so — that is not a criticism, it is the record. The failure
mode is a decision appearing in the code with no author and no reason.

---

PHASE 5 — FIX THE STALE DOCUMENTS. Work on a branch. These are authoritative
files, so change only what the code has actually made false, and list every
edit you make.

  docs/project/ENVIRONMENT_SPEC.md    §L3 says observation is 36x3+1
  docs/project/STATE_ACTION_FORMULATION.md   says 109 in ~6 places
  docs/project/EDGE_LANE_HANDOFF.md   says 109 in 3 places (already sent to
                                      the edge team — flag this in your summary)
  docs/project/RL_TEAM_HANDOFF.md     line ~431, Box(shape=(109,))
  docs/project/EVALUATION.md          §5 has no rungs 7/8 — add rows ONLY if
                                      Phase 7 produces real numbers; leave it
                                      alone otherwise
  CLAUDE.md                           build-status paragraph: test count and
                                      "rung 7 is the only thing missing"

Regenerate the PDFs with scripts/md2pdf.py for any markdown you changed.
Delete docs/technical-references/ — both PDFs there are byte-identical
duplicates of files in docs/project/ and both document the old 109 vector.

---

PHASE 6 — FIX WHAT IS BROKEN. On a clean checkout the suite is red and the
headline command crashes. Both are one-liners:

  HARD CONSTRAINT ON THIS PHASE: change NO behaviour in rfenv/env.py. Docstring
  corrections, deleting dead commented-out code, the test skip list and the
  compare skip guard are all safe. Anything that changes what an observation
  contains, what a reward returns, or what step() does invalidates every
  checkpoint I have — they were trained against the current behaviour and there
  is no budget to retrain them. If you think something in env.py is actually
  wrong, write it up as a finding and leave the code alone.

  a) `pytest tests -q` gives 7 failures. Rung "8d" (ppo_weighted_camp) is
     missing from _UNTRAINED_RUNG_CHECKPOINTS in tests/test_baselines.py, so it
     fails instead of skipping. Since its reward candidate is commented out of
     env.py and can no longer be reconstructed, the right fix is probably to
     delete the rung entirely — but say which you chose and why, and put it in
     the ledger.

  b) `python -m rfenv.compare --seeds 3 --sampled 10 --figures` hard-crashes
     with ModuleNotFoundError on the first RL rung when stable_baselines3 is
     absent or a checkpoint is missing. Make an unbuildable rung skip with a
     warning instead of killing the run. This command is the acceptance test in
     §1 of the brief; it must work on a machine with no training stack.

  c) Three doc-vs-code contradictions to correct while you are in there:
     - env.py's reward_first_intercept docstring says a dwell re-touching known
       occupancy "nets +0.5". The code nets 0.0. The docstring is wrong.
     - rollout.py's docstring says "obs[-1] is normalised episode time". After
       the observation change obs[-1] is camp_time; the clock is obs[-2].
     - dqn.train() defaults to 60_000 timesteps, its CLI defaults to 20_000,
       and the package docstring says 20,000 "the defaults". Pick one.

  d) Delete the commented-out reward_weighted_camp and reward_hybrid drafts
     from env.py. CLAUDE.md forbids keeping rejected approaches live, and
     reward_hybrid references obs/STALENESS/SWEEP_SLOTS which are not even in
     scope — it could not run. Their reasoning belongs in D-16 and the ledger,
     not in the file.

  e) Fix the small drift: the rung key deep_q_network_hit_y_20k describes
     itself as 5,000 timesteps; run.md's registration example disagrees with
     ladder.py on both key and rung number; both load_checkpoint error messages
     hardcode the Windows interpreter path; SCHEDULERS' comment still says
     "seven rungs" when there are twelve.

Then run the full suite and report the actual pass/fail/skip counts.

---

PHASE 7 — PRODUCE THE NUMBERS. This is the part with no substitute. Run:

  python -m rfenv.compare --seeds 3 --sampled 10 --figures --out runs/baselines

with every trained rung registered, then report the §5 table with the RL rows
in it.

  BEFORE YOU RUN IT: try loading every checkpoint and report which ones load.
  Any checkpoint saved before the observation change is 109-wide and will not
  load into the current 146-wide environment at all. Expect some to be dead.
  A dead checkpoint is an [UNRECOVERABLE] result — report it as one. Do NOT
  retrain to fill the gap: a multi-hundred-thousand-timestep run is hours of
  compute and I want to decide whether it is worth it, so ask me first.

  AFTER YOU RUN IT: verify the printed table actually contains rungs 7 and 8.
  Phase 6b makes an unbuildable rung skip with a warning, which means this
  command can now complete "successfully" with zero RL rows in it. That is not
  success, it is the failure this whole task exists to fix. If the RL rows are
  missing, say so at the top of your summary in plain words and do not describe
  Phase 7 as done.

Then, specifically:

  - Paired wins for the best RL rung against BOTH round_robin (the floor) and
    recency (the actual bar), via compare.paired_wins(). Rung 5 Pareto-dominates
    round-robin on 70.2% of episodes, so beating round-robin alone is not a
    result. If RL loses to rung 5, say that plainly — an honest negative result
    is a deliverable here, not a failure.
  - The reward comparison under D47's rule: the candidate that beats
    round-robin on BOTH headline metrics on the most paired episodes wins;
    within 5 pp, nothing is selected and it escalates. Show the "both" column.
  - The learning curve as a figure, from the logs.
  - Which component earns the gain: rung 7/8 vs rung 5 vs rung 6a.

Every number reported must come from a command run in this session, not from
memory or from a transcript.

---

PHASE 8 — BUILD HANDOVER/. Exactly the tree in RL_TEAM_HANDOFF.md §17:
README.md, DECISIONS_DRAFT.md, checkpoints/ (+ SHA256SUMS.txt), code/,
results/, figures/, logs/, NOTES.md — plus DECISION_LEDGER.md from Phase 4b.
HANDOVER/README.md must open with the one command that reproduces the results
table and the SHA-256 of the checkpoint it uses.

HANDOVER/ is too big for git and runs/ is gitignored — ship it over a drive
share. But put a short HANDOVER_INDEX.md IN the repo listing what is in the
share, every checkpoint's SHA-256, and the drive link, so the repo records that
the artefacts exist and where.

NOTES.md carries the negative results (E-6, E-7). This project already has four
recorded (D36, D43, D44, D45) and treats them as pitch material — add yours.

---

IF YOU RUN SHORT, OR THE EVIDENCE ISN'T THERE.

Two things can go wrong and neither should stall you.

  If ~/.claude/projects/ has no transcripts for this repo (cleared, different
  machine, sessions never persisted), Phases 2 and 4b get thin. Do not
  compensate by reasoning about what "must have" happened — that is exactly the
  fabrication this task forbids. Build both documents from the checkpoints, the
  logs and the git history alone, tag the reasoning gaps [UNRECOVERABLE], and
  tell me which questions only I can now answer.

  If time runs out, the priority order is strict, and stopping cleanly partway
  is much better than rushing all eight phases:

    1. Phase 1 + the hyperparameter table   — without this the training runs
                                              cannot be reproduced by anyone,
                                              ever. Everything else is optional
                                              next to it.
    2. Phase 7's numbers                    — without these there is no result,
                                              only code.
    3. Phase 8's HANDOVER/ with whatever    — so the artefacts physically reach
       exists so far                          my teammate.
    4. Phases 4a / 4b                       — the reasoning.
    5. Phases 5 / 6                         — doc sync and the two bug fixes;
                                              my teammate can do these himself.

  Say plainly where you stopped and what is left. An honest partial handover is
  a good outcome. A complete-looking one with invented content is the only
  genuinely bad one.

---

WHAT NOT TO DO.
  - Do not write into docs/project/DECISIONS.md. Draft only.
  - Do not invent a hyperparameter, a timestep count, a date, or a measurement.
  - Do not report a number you did not produce in this session.
  - Do not touch rfenv/constants.py or tests/test_freeze.py. The environment is
    frozen (D42); a change there re-opens all four validation gates.
  - Do not read anything under data/turing/**/test_* — that is the held-out set.
  - Do not commit or push. Stage nothing. Leave that to me.

DONE MEANS: the suite is green, `python -m rfenv.compare --seeds 3 --sampled 10
--figures` runs to completion on a clean checkout, the docs no longer say 109,
HANDOVER/ is built with both the decision draft and the decision ledger in it,
and every reconstructed claim carries one of the three tags. Then give me the
exact git commands to commit and push, one logical chunk per commit, and tell me
plainly which loose ends you could not close.
