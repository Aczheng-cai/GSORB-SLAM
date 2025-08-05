#!/bin/bash
#This file is part of GSORB-SLAM.
#For more information see <https://github.com/Aczheng-cai/GSORB-SLAM>

# Configuration
experiment_count=0    # Number of runs per scene (0 to 5)

####################################################
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(realpath "$script_dir/..")"

scene_base="${workspace}/datasets/Scannet"   # ScanNet dataset base path

yaml_name="scannet"   # YAML filename without extension
yaml_base_path="${workspace}/Examples/RGB-D"  # Path to YAML configs
# Define scene names only (without full paths)

vocab_path="${workspace}/Vocabulary/ORBvoc.txt"

scene_names=(
    #"scene0000"
    #"scene0059"
    "scene0106"
    "scene0169"
    "scene0182"
    "scene0181"
    "scene0207"
    "scene0465"
)

# Loop through experiment runs
for i in $(seq 0 $((experiment_count))); do
    echo "[INFO] Starting experiment round $i"
    cd "$workspace" || exit 1
    # Loop through scenes
    for scene_name in "${scene_names[@]}"; do
        scene_path="${scene_base}/${scene_name}_00"
        scene_tag="${scene_name}-${i}"

        echo "[INFO] Processing scene: ${scene_tag}"

        # Update YAML with current scene name + iteration
        yq eval ".Dataset.name = \"${scene_tag}\"" "${yaml_base_path}/${yaml_name}.yaml" -i

        # Run SLAM system
        ./Examples/RGB-D/rgbd_scannet \
            ${vocab_path} \
            "${yaml_base_path}/${yaml_name}.yaml" \
            "${scene_path}"

        echo "[INFO] Finished scene: ${scene_tag} ✅"
    done
done

echo "[INFO] All SCANNET experiments completed! 🎉"

