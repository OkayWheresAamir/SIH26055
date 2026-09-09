"""`python -m rfenv.baselines` -- print the ladder and what the two open-loop
rungs actually spend, per band (the D43 claim as a command's output)."""

from __future__ import annotations

import rfenv.baselines as baselines

summary = baselines.cycle_summary()
print(baselines.__doc__)
print("Rungs:")
for r in baselines.LADDER:
    print(f"  {r.rung:>2}  {r.key:<18} {r.label:<32} "
          f"{'reference line' if not r.deployable else 'scheduler'}")
print()
print(f"Turing sweep cycle : {summary['turing_sweep_cycle_slots']} slots "
      f"({summary['turing_sweep_cycle_s']:.2f} s), 1 slot/narrow band, 2/wide")
print(f"Round-robin cycle  : {summary['round_robin_cycle_slots']} slots "
      f"({summary['round_robin_cycle_s']:.2f} s), "
      f"{set(summary['round_robin_slots_per_band_per_cycle'].tolist())} slot(s) per band")
