set -e
cd /c/userfile/alcedo-perch/mech/tools/stage2
export MIDLINE=1 STACK_BUD_JOINT=8 STACK_NSUB=24 SWIVEL=20 STAGE_TAG=_sw20
PY=../../../.venv/Scripts/python; L=../../results/stage2; T=2f_sw20
$PY loads.py 2e > $L/${T}_loads.log 2>&1
$PY check_2e.py > $L/${T}_check.log 2>&1
$PY crossing_check.py > $L/${T}_crossing.log 2>&1
$PY mirror_check.py > $L/${T}_mirror.log 2>&1
$PY feather_bend.py > $L/${T}_feather.log 2>&1
IDS=$($PY -c "import json;print(' '.join(c['id'] for c in json.load(open('$L/$T/candidates.json',encoding='utf-8'))))")
($PY stack_figs.py && $PY cand_figs.py 2e && $PY mirror_figs.py $IDS) > $L/${T}_figs.log 2>&1
echo PIPELINE_DONE $IDS
