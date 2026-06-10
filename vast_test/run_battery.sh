#!/bin/bash
# v2 battery: 3 seeds + stability/leanness ablations. Sequential, ~30 min total.
set -u
cd /root
PY=/venv/main/bin/python

run() {
  echo "=== RUN: $* ==="
  if [ -f base_acc.txt ]; then
    $PY -u joint_extraction_v2.py "$@" --base-acc "$(cat base_acc.txt)"
  else
    $PY -u joint_extraction_v2.py "$@"
  fi
  echo "=== DONE: $* (exit $?) ==="
}

run --name s42 --seed 42
run --name s43 --seed 43
run --name s44 --seed 44
run --name mlponly --seed 42 --lora-scope mlp
run --name ce0 --seed 42 --ce-weight 0.0
echo "BATTERY_COMPLETE"
