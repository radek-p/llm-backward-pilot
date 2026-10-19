#!/usr/bin/env bash
# Follow-up: clause-constrained searches ("[clause], I will ..."), started when the main pilot has finished.
# Hard cap 55 min; fold 1 first, fold 2 only if time remains. Output: results_clause/, run_followup.log.
cd "$(dirname "$0")"
until grep -q "== PILOT FINISHED" run.log 2>/dev/null; do sleep 20; done
echo "== main pilot finished, starting follow-up $(date -u)"
timeout 55m .venv/bin/python pilot.py --clause --steps 9 --skip-cheap --skip-a0 --budget-min 50 --out results_clause
.venv/bin/python report.py --out results_clause
tar czf results_clause.tar.gz results_clause
echo "== FOLLOWUP FINISHED $(date -u)"
