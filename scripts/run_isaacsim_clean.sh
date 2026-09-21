#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "[ERROR] CONDA_PREFIX is empty. Please run: conda activate isaaclab" >&2
  exit 1
fi

if ! command -v isaacsim >/dev/null 2>&1; then
  echo "[ERROR] 'isaacsim' not found in PATH. Please run inside the isaaclab env." >&2
  exit 1
fi

LULA_TINY_DEFAULT="$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/exts/isaacsim.robot_motion.lula/pip_prebundle/_lula_libs/libtinyxml.so"
if [[ -f "${LULA_TINY_DEFAULT}" ]]; then
  LULA_TINY="${LULA_TINY_DEFAULT}"
else
  LULA_TINY="$(find "$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/exts" -path '*/isaacsim.robot_motion.lula/pip_prebundle/_lula_libs/libtinyxml.so' 2>/dev/null | head -n1 || true)"
fi

if [[ -z "${LULA_TINY:-}" || ! -f "${LULA_TINY}" ]]; then
  echo "[ERROR] Could not find Lula tinyxml library under $CONDA_PREFIX." >&2
  exit 1
fi

echo "[INFO] isaacsim: $(command -v isaacsim)"
echo "[INFO] preload: ${LULA_TINY}"

ROS_BRIDGE_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge"
ROS_DISTRO_INTERNAL="humble"
ROS_INTERNAL_PREFIX="$ROS_BRIDGE_DIR/$ROS_DISTRO_INTERNAL"
ROS_INTERNAL_LIB="$ROS_INTERNAL_PREFIX/lib"

if [[ ! -d "$ROS_INTERNAL_LIB" ]]; then
  echo "[ERROR] Internal ROS2 lib path not found: $ROS_INTERNAL_LIB" >&2
  exit 1
fi

echo "[INFO] internal ROS2: ${ROS_DISTRO_INTERNAL}"

env \
  -u PYTHONPATH \
  -u ROS_VERSION \
  -u CMAKE_PREFIX_PATH \
  ROS_DISTRO="${ROS_DISTRO_INTERNAL}" \
  RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  AMENT_PREFIX_PATH="${ROS_INTERNAL_PREFIX}" \
  LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${ROS_INTERNAL_LIB}" \
  LD_PRELOAD="${LULA_TINY}" \
  isaacsim "$@"
