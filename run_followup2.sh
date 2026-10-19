#!/usr/bin/env bash
# Follow-up 2: "Because [clause], I will ..." beginnings (connective start + comma end, no single-letter or
# repeated tokens). Starts when follow-up 1 has finished. Hard cap 45 min. Output: results_clause2/.
cd "$(dirname "$0")"
until grep -q "== FOLLOWUP FINISHED" run_followup.log 2>/dev/null; do sleep 20; done
echo "== follow-up 1 finished, starting follow-up 2 $(date -u)"
timeout 45m .venv/bin/python pilot.py --starts --steps 9 --skip-cheap --skip-a0 --budget-min 40 --out results_clause2
.venv/bin/python report.py --out results_clause2
tar czf results_clause2.tar.gz results_clause2
echo "== FOLLOWUP2 FINISHED $(date -u)"
