#!/usr/bin/env bash
# D75 matched pair: "v2p" control vs "v3" treatment, seeds 0-2, strictly sequential.
# The two arms differ ONLY in --obs-version. band_priority stays off in both, so
# v2p's band_priority block is constant all-ones and v3 adds only the three
# in-context blocks. Interleaved so a matched pair finishes before the next starts.
# Hyperparameters mirror rung 23a/23b (512-wide LSTM, 800k, n_steps=8192).
set -u
cd "C:/Code/sih/SIH26055"
PY=venv/Scripts/python.exe

run() {  # $1=obs_version $2=run-name-prefix $3=dir $4=seed
  name="$2_seed$4"
  out="runs/checkpoints/$3/$name/$name.zip"
  if [ -f "$out" ]; then echo "skip $name (exists)"; return; fi
  echo "=== $(date +%T) start $name"
  $PY -m rfenv.rl.recurrent_ppo --reward reward_balance --obs-version "$1" \
    --timesteps 800000 --seed "$4" --checkpoint-freq 50000 \
    --hyperparam ent_coef=0.01 --hyperparam gamma=0.997 --hyperparam n_steps=8192 \
    --hyperparam 'policy_kwargs={"lstm_hidden_size": 512}' \
    --checkpoint "$out" --run-name "$name" --device auto \
    --description "D75 matched pair, $1 arm, seed $4: reward_balance, 512 LSTM, 800k. Differs from its partner only in --obs-version." \
    || echo "!!! $name FAILED"
  echo "=== $(date +%T) done  $name"
}

for seed in 0 1 2; do
  run v3  lstm_v3       v3  "$seed"
  run v2p lstm_v2p_ctrl v2p "$seed"
done
echo "ALL DONE"
