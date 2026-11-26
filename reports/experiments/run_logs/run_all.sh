#!/usr/bin/env bash
# Runs the experiment scripts one after another; each writes its own log.
cd /f/DM_Project
for s in 13_ensembles 14_port_ablation 17_granularity 16_imbalance 15_day_split; do
  echo "=== $s start $(date +%T)"
  .venv/Scripts/python scripts/$s.py > reports/experiments/run_logs/$s.log 2>&1
  echo "=== $s exit $? $(date +%T)"
done
