set -e
cd /c/userfile/alcedo-perch/mech/tools/stage2
export MIDLINE=1 STACK_BUD_JOINT=8 STACK_NSUB=24
PY=../../../.venv/Scripts/python; L=../../results/stage2
$PY analyze.py 45 2e > $L/2f_analyze.log 2>&1
$PY check_2e.py > $L/2f_check.log 2>&1
$PY crossing_check.py > $L/2f_crossing.log 2>&1
$PY mirror_check.py > $L/2f_mirror.log 2>&1
$PY loads.py 2e > $L/2f_loads.log 2>&1
$PY feather_bend.py > $L/2f_feather.log 2>&1
rm -f $L/2f/stack_*.png $L/2f/mirror_*.png
IDS=$($PY -c "import json;print(' '.join(c['id'] for c in json.load(open('$L/2f/candidates.json',encoding='utf-8'))))")
($PY stack_figs.py && $PY cand_figs.py 2e && $PY mirror_figs.py $IDS && $PY export_sim.py 2e && $PY standalone.py) > $L/2f_figs.log 2>&1
$PY report_2f2.py > $L/2f_report.log 2>&1
echo PIPELINE_DONE $IDS
