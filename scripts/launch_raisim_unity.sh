#!/usr/bin/env bash
# Launch RaiSimUnity for the GraspXL demo.
#
# On macOS, RaiSimUnity is typically ONE window:
#   - Settings sidebar (IP/port, resource dirs, Auto-connect)
#   - 3D viewport (appears after a successful connect)
# Click inside the 3D viewport for mouse camera controls (not the sidebar) —
# scroll to zoom and drag to orbit already work out of the box; they're easy
# to miss if you never click into the viewport.
set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=graspxl_env.sh
source "${_script_dir}/graspxl_env.sh"

: "${GRASPXL_UNITY:?Set GRASPXL_UNITY (source scripts/graspxl_env.sh first)}"
if [[ ! -e "${GRASPXL_UNITY}" ]]; then
  echo "RaiSimUnity not found at ${GRASPXL_UNITY}" >&2
  exit 1
fi

resource_dir="${GRASPXL_ROOT}/rsc"

if [[ "$(uname -s)" == "Darwin" ]]; then
  plist="${HOME}/Library/Preferences/com.RaiSimTech.RaiSimUnity.plist"
  if pgrep -xq RaiSimUnity; then
    echo "RaiSimUnity is running; quit it (Cmd+Q) so resource-directory prefs can be reset."
  elif [[ -f "${plist}" ]]; then
    idx=1
    while /usr/libexec/PlistBuddy -c "Print :RscDir${idx}" "${plist}" >/dev/null 2>&1; do
      /usr/libexec/PlistBuddy -c "Delete :RscDir${idx}" "${plist}"
      idx=$((idx + 1))
    done
    if [[ -d "${resource_dir}" ]]; then
      if /usr/libexec/PlistBuddy -c "Print :RscDir0" "${plist}" >/dev/null 2>&1; then
        /usr/libexec/PlistBuddy -c "Set :RscDir0 ${resource_dir}" "${plist}"
      else
        /usr/libexec/PlistBuddy -c "Add :RscDir0 string ${resource_dir}" "${plist}"
      fi
      echo "RaiSimUnity prefs: resource directory set to ${resource_dir}"
    fi
  fi
fi

echo "Launch RaiSimUnity: ${GRASPXL_UNITY}"
echo ""
echo "RaiSimUnity layout (macOS):"
echo "  - Settings sidebar: port 8080, resource directories, Auto-connect"
echo "  - 3D viewport: shows the scene after connect (click here for mouse orbit)"
echo ""
if [[ -d "${resource_dir}" ]]; then
  echo "Add Resource Directory: ${resource_dir}"
  echo "(only this one — remove per-object paths)"
fi
echo "Press Enter in the terminal to start the sim server, then enable Auto-connect."
echo ""
echo "Camera (in the 3D viewport, not the sidebar):"
echo "  - Scroll wheel: zoom"
echo "  - Right-drag: orbit / rotate view"
echo "  - Left-drag: pan (may require clicking the scene first)"
echo ""

if [[ -x "${GRASPXL_UNITY}" ]]; then
  open "${GRASPXL_UNITY}" >/dev/null 2>&1 || "${GRASPXL_UNITY}" &
else
  "${GRASPXL_UNITY}" &
fi
sleep 3
