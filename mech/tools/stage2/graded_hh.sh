#!/bin/sh
# stage 2e follow-up: the two-bellcrank configuration (hum/hum) again with the graded stack measure (STACK_GRADED=1), seeds 100-103
LOG=../../results/stage2/batch_w45_2e.log
OUT=../../results/stage2/batch_w45_2e_graded.log
while [ "$(grep -c seed $LOG)" -lt 57 ]; do sleep 60; done
: > $OUT
for s in 100 101 102 103; do for d in dA dB; do
  ( STACK_GRADED=1 ../../../.venv/Scripts/python run.py hum hum $s 300 300 45 $d >> $OUT 2>&1 ) &
done; done
wait
echo done >> $OUT
