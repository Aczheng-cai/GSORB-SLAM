#!/bin/bash
#This file is part of GSORB-SLAM.
#For more information see <https://github.com/Aczheng-cai/GSORB-SLAM>

# filepath: run_all.sh
# Description: Sequentially runs a list of scripts. Terminates if any script fails.

# Optional: Enable strict mode for better error handling
set -euo pipefail

# Print start message
echo "[INFO] Starting to run all scripts..."

# Define the list of scripts to run
scripts=(
    "scripts/run_tum.sh"
    "scripts/run_replica.sh"
    "scripts/run_scannet.sh"
)

# Loop through and execute each script
for script in "${scripts[@]}"; do
    echo "[INFO] Running script: $script"

    if [ -f "$script" ]; then
        bash "$script"
        echo "[INFO] $script completed successfully ✅"
    else
        echo "[WARN] Script $script not found, skipping ❌"
    fi
done

# Print completion message
echo "[INFO] All scripts have been executed! 🎉"

