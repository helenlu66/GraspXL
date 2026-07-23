#!/usr/bin/env bash
# GraspXL path defaults. Source from PhysAwareHOI scripts/paths.env.sh or directly.

if [ -n "${BASH_VERSION:-}" ]; then
  _graspxl_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [ -n "${ZSH_VERSION:-}" ]; then
  _graspxl_env_dir="$(cd "$(dirname "${(%):-%x}")" && pwd)"
else
  _graspxl_env_dir="$(cd "$(dirname "$0")" && pwd)"
fi
export GRASPXL_ROOT="${GRASPXL_ROOT:-$(cd "${_graspxl_env_dir}/.." && pwd)}"
export RAISIM_BUILD="${RAISIM_BUILD:-${HOME}/raisim/raisim_build}"
export GRASPXL_VENV="${GRASPXL_VENV:-${ARTIGRASP_VENV:-${HOME}/raisim/venv}}"
export GRASPXL_PYTHON="${GRASPXL_PYTHON:-${ARTIGRASP_PYTHON:-${GRASPXL_VENV}/bin/python}}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  export GRASPXL_UNITY="${GRASPXL_UNITY:-${GRASPXL_ROOT}/raisimUnity/m1/RaiSimUnity.app}"
else
  export GRASPXL_UNITY="${GRASPXL_UNITY:-${GRASPXL_ROOT}/raisimUnity/linux/raisimUnity.x86_64}"
fi
