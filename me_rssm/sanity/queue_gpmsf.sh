#!/usr/bin/env bash
# GP-MSF BPTT0 resource gate. Serial after A2-R; owns/kills ONLY its spawned
# sessions. Fail closed: CPU -> real preflight -> matched B0/arm DDP S0.
# Formal continuation is intentionally NOT automatic: it must use the saved
# S0 verdict and an explicit --resume-from epoch_4.pth command.
set -eo pipefail
cd /home/lurui/workspace/R4Det
source .envrc
set -u
export CUDA_VISIBLE_DEVICES=5,6,7
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
WD=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_gpmsf_bptt0_s0_seed0
CFG=me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_gpmsf_activecells_bptt0.py
MAIN_CFG=configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_bptt0.py
B0=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed0
A2R=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA2R_seed0
QLOG=$WD/queue_gpmsf.log
CHECK=(python me_rssm/sanity/gpmsf_run_checks.py)
mkdir -p "$WD"
exec 9>"$WD/queue_gpmsf.lock"
flock -n 9 || { echo 'GP-MSF queue already armed'; exit 1; }
CHILD=0
say(){ echo "[gpmsf] $(date -u '+%F %T') $*" >> "$QLOG"; }
stop_owned(){
  if [ "$CHILD" -gt 0 ]; then
    # CHILD is a new setsid session created by this queue, never A2-R's group.
    kill -TERM -- "-$CHILD" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 -- "-$CHILD" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
    CHILD=0
  fi
}
on_exit(){
  code=$?
  trap - EXIT
  stop_owned
  if [ "$code" -ne 0 ]; then say "FAILED exit=$code; no automatic continuation"; fi
  exit "$code"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

drain(){
  local deadline=$((SECONDS + 14400))
  while [ "$SECONDS" -lt "$deadline" ]; do
    # Query failures and malformed/missing GPU rows are errors, not free GPUs.
    local memory
    memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,6,7)
    if MEMORY="$memory" python -c 'import os,sys; x=[int(v.strip()) for v in os.environ["MEMORY"].splitlines()]; assert len(x)==3 and min(x)>=0; sys.exit(0 if max(x)<1000 else 2)'; then
      say "GPU 5/6/7 drained"
      return 0
    else
      local code=$?
      [ "$code" -eq 2 ] || { say 'GPU query invalid'; return 1; }
    fi
    sleep 60
  done
  say 'GPU drain deadline exceeded; NOT launching'
  return 1
}

run_owned(){
  local limit=$1 logfile=$2 monitor=$3
  shift 3
  local deadline=$((SECONDS + limit)) early=0 mature=0 code=0
  setsid "$@" < /dev/null > "$logfile" 2>&1 &
  CHILD=$!
  say "launched pid/pgid=$CHILD log=$logfile"
  while kill -0 "$CHILD" 2>/dev/null; do
    if [ "$SECONDS" -ge "$deadline" ]; then say 'phase timed out'; return 1; fi
    if grep -Eqi 'CAUGHT AN EXCEPTION|Traceback \(most recent call last\)|CUDA out of memory' "$logfile"; then
      say 'phase exception; stopping owned session'
      return 1
    fi
    if [ "$monitor" = yes ]; then
      if [ "$early" -eq 0 ]; then
        if "${CHECK[@]}" ready --work-dir "$WD" --expected-epochs 6 > "$ATTEMPT/ready.log" 2>&1; then
          if "${CHECK[@]}" curve --work-dir "$WD" --b0 "$B0" --window 1-6 --early-stop --live \
                --out "$WD/gpmsf_ep6_report.json" >> "$QLOG" 2>&1; then
            early=1
            say 'ep1-6 complete; no >2 AP task harm'
          else
            say 'ep6 harm or evidence failure; stop for manual review (not final morphology rejection)'
            return 1
          fi
        else
          code=$?
          [ "$code" -eq 3 ] || { say 'invalid live training evidence'; return 1; }
        fi
      fi
      if [ "$mature" -eq 0 ]; then
        if "${CHECK[@]}" ready --work-dir "$WD" --expected-epochs 16 > "$ATTEMPT/ready.log" 2>&1; then
          "${CHECK[@]}" curve --work-dir "$WD" --b0 "$B0" --window 12-16 --live \
            --out "$WD/gpmsf_ep16_report.json" >> "$QLOG" 2>&1 || return 1
          mature=1
          say 'ep12-16 complete; report only, training continues to epoch 24'
        else
          code=$?
          [ "$code" -eq 3 ] || { say 'invalid live training evidence'; return 1; }
        fi
      fi
    fi
    sleep 60
  done
  wait "$CHILD" || return 1
  # A successful launcher must not leave a rank holding GPUs in its session.
  if kill -0 -- "-$CHILD" 2>/dev/null; then say 'orphaned phase ranks'; return 1; fi
  CHILD=0
}

say 'armed (reviewed gates); waiting for A2-R; F1 slot remains replaced'
while :; do
  if grep -q 'DONE' "$A2R/queue_semA2R_resume.log" 2>/dev/null && \
      ! pgrep -f '[s]emA2R_24e|[q]ueue_semA2R_resume.sh' >/dev/null; then break; fi
  sleep 60
done
[ -f "$A2R/epoch_16.pth" ] || { say 'predecessor checkpoint missing'; exit 1; }
"${CHECK[@]}" curve --work-dir "$A2R" --window 12-16 --expected-epochs 16 \
  --out "$WD/a2r_predecessor_checked.json" >> "$QLOG" 2>&1
say 'A2-R completed with full ep1-16 evidence; not relying on DONE alone'
# Fresh phases only. Never reuse old checkpoint or silently resume formal run.
if compgen -G "$WD/*.log.json" >/dev/null || compgen -G "$WD/epoch_*.pth" >/dev/null; then
  say 'existing formal artifacts; explicit resume/review required'
  exit 1
fi
ATTEMPT=$WD/s0_$(date -u '+%Y%m%dT%H%M%SZ')_$$
mkdir "$ATTEMPT"
python -m pytest -q -p no:hydra_pytest me_rssm/sanity/test_gpmsf_wiring.py \
  me_rssm/sanity/test_gpmsf_queue_checks.py >> "$QLOG" 2>&1
python me_rssm/sanity/gpmsf_cpu_checks.py >> "$QLOG" 2>&1
say 'CPU geometry/wiring/queue gates PASS'
drain
run_owned 3600 "$ATTEMPT/preflight.log" no env CUDA_VISIBLE_DEVICES=5 \
  python me_rssm/sanity/gpmsf_real_preflight.py --config "$CFG" \
  --baseline "$MAIN_CFG" --out-dir "$ATTEMPT/preflight"
for arm in baseline gpmsf; do
  drain
  PHASE_WD=$ATTEMPT/${arm}_450iter
  mkdir "$PHASE_WD"
  if [ "$arm" = baseline ]; then SMOKE_CFG=me_rssm/sanity/gpmsf_baseline_smoke_450iter.py
  else SMOKE_CFG=me_rssm/sanity/gpmsf_smoke_450iter.py; fi
  "${CHECK[@]}" start --work-dir "$PHASE_WD" --config "$SMOKE_CFG"
  run_owned 21600 "$ATTEMPT/${arm}_smoke.log" no bash tools/dist_train_runtime.sh \
    "$SMOKE_CFG" 3 --detect-anomaly off --seed 0 --deterministic \
    --work-dir "$PHASE_WD"
  "${CHECK[@]}" finish --work-dir "$PHASE_WD"
done
python me_rssm/sanity/gpmsf_s0_verdict.py --work-dir "$ATTEMPT/gpmsf_450iter" \
  --baseline-dir "$ATTEMPT/baseline_450iter" \
  --real-preflight "$ATTEMPT/preflight/real_preflight.json" \
  --arm-config "$CFG" --main-config "$MAIN_CFG" \
  --out "$ATTEMPT/s0_verdict.json" >> "$QLOG" 2>&1
say "S0_ALL_PASS including matched <=1.20x resources; artifacts=$ATTEMPT"
say 'S0 only: formal continuation requires manual review and explicit resume-from epoch_4.pth'
