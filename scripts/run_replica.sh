#!/bin/bash
#This file is part of GSORB-SLAM.
#For more information see <https://github.com/Aczheng-cai/GSORB-SLAM>

# === Configuration ===
experiment_count=5   # Number of runs per scene

####################################################
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(realpath "$script_dir/..")"

# Dataset-specific path (e.g., Replica scenes)
scene_base="${workspace}/datasets/Replica"

yaml_name="replica"  # YAML file name (without extension)
yaml_base_path="${workspace}/Examples/RGB-D"  # Path to YAML configuration

vocab_path="${workspace}/Vocabulary/ORBvoc.txt"

# Define an array of scene names (just names, not full paths)
scene_names=(
    "room0"
    "room1"
    "room2"
    "office0"
    "office1"
    "office2"
    "office3"
    "office4"
)

# === Run multiple experiments per scene ===
for i in $(seq 0 $((experiment_count))); do
    echo "[INFO] Starting experiment round $i"
    cd "$workspace" || exit 1
    for scene_name in "${scene_names[@]}"; do
        scene_path="${scene_base}/${scene_name}"
        scene_tag="${scene_name}-${i}"

        echo "[INFO] Processing scene: ${scene_tag}"

        # Update scene name in the YAML config (for logging/saving)
        yq eval ".Dataset.name = \"${scene_tag}\"" \
            "${yaml_base_path}/${yaml_name}.yaml" -i

        # Execute the SLAM system
        ./Examples/RGB-D/rgbd_replica \
            ${vocab_path} \
            "${yaml_base_path}/${yaml_name}.yaml" \
            "${scene_path}" 

        echo "[INFO] Finished scene: ${scene_tag} ✅"
    done
done

echo "[INFO] All REPLICA experiments completed! 🎉"

