#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"
if [[ -f local.env ]]; then source local.env; fi
task_id="task3"
mode="${1:-teleop}"
if (($#)); then shift; fi
case "$mode" in
  help|-h|--help)
    printf 'Usage: bash run.sh {teleop|eval|smoke} [extra Isaac arguments]\nConfigure local.env using env.example.\n'
    exit 0 ;;
  teleop|eval|smoke) ;;
  *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac
: "${ISAAC_SIM_ROOT:?Set ISAAC_SIM_ROOT in local.env or the environment}"
[[ -x "$ISAAC_SIM_ROOT/python.sh" ]] || { printf 'Missing Isaac Sim python.sh\n' >&2; exit 2; }
if [[ -n "${ISAACLAB_ROOT:-}" ]]; then
  [[ -d "$ISAACLAB_ROOT/source/isaaclab_tasks" ]] || { printf 'Invalid ISAACLAB_ROOT\n' >&2; exit 2; }
  export PYTHONPATH="$ISAACLAB_ROOT/source/isaaclab:$ISAACLAB_ROOT/source/isaaclab_tasks${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="$repo_dir/source/leisaac${PYTHONPATH:+:$PYTHONPATH}"
export LEISAAC_ASSETS_ROOT="$repo_dir/assets"
export PYTHONUNBUFFERED=1
case "$mode" in
  teleop)
    limit="${RUN_TIMEOUT_S:-3600}"
    args=(scripts/environments/teleoperation/teleop_se3_agent.py
      --task="$task_id" --teleop_device=bi_keyboard --num_envs=1
      --device=cuda --enable_cameras --multi_view)
    ;;
  eval)
    : "${POLICY_CHECKPOINT:?Set POLICY_CHECKPOINT to the path visible to the policy server}"
    limit="${RUN_TIMEOUT_S:-180}"
    args=(scripts/evaluation/policy_inference.py
      --task="$task_id" --seed="${SEED:-123}" --episode_length_s="${EPISODE_LENGTH_S:-40}"
      --eval_rounds=1 --policy_type=xtrainer_act
      --policy_host="${POLICY_HOST:-127.0.0.1}" --policy_port="${POLICY_PORT:-5555}"
      --policy_timeout_ms=15000 --policy_action_horizon="${POLICY_ACTION_HORIZON:-16}"
      --policy_language_instruction="Put items with Chinese labels into the red basket, and others into the blue basket."
      --policy_checkpoint_path="$POLICY_CHECKPOINT" --device=cuda --enable_cameras)
    ;;
  smoke)
    limit="${RUN_TIMEOUT_S:-180}"
    args=(scripts/smoke_test.py --task="$task_id" --headless --device=cuda)
    ;;
esac
mkdir -p logs
log_path="logs/${mode}_$(date +%Y%m%d_%H%M%S)_$$.log"
printf 'Task: %s | Mode: %s | Log: %s\n' "$task_id" "$mode" "$log_path"
set +e
timeout --signal=TERM --kill-after=10s "$limit" "$ISAAC_SIM_ROOT/python.sh" "${args[@]}" "$@" 2>&1 | tee "$log_path"
run_status=${PIPESTATUS[0]}
set -e
printf '\nRUN_EXIT_CODE=%s\n' "$run_status" | tee -a "$log_path"
exit "$run_status"
