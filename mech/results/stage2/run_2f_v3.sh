set -e
cd /c/userfile/alcedo-perch/mech/tools/stage2
export MIDLINE=1 STACK_BUD_JOINT=8 STACK_NSUB=24
PY=../../../.venv/Scripts/python; L=../../results/stage2
$PY analyze.py 45 2e > $L/2f_analyze.log 2>&1
$PY screen_mc.py > $L/2f_screen.log 2>&1
$PY gate_2f.py > $L/2f_gate.log 2>&1
$PY analyze.py 45 2e > $L/2f_analyze.log 2>&1
$PY check_2e.py > $L/2f_check.log 2>&1
$PY crossing_check.py > $L/2f_crossing.log 2>&1
$PY mirror_check.py > $L/2f_mirror.log 2>&1
$PY loads.py 2e > $L/2f_loads.log 2>&1
$PY feather_bend.py > $L/2f_feather.log 2>&1
echo PIPELINE_DONE
