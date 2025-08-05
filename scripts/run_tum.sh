#!/usr/bin/env bash
#This file is part of GSORB-SLAM.
#For more information see <https://github.com/Aczheng-cai/GSORB-SLAM>

# Configuration
experiment_count=5  # Number of runs per scene (1 to 5)


####################################################
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(realpath "$script_dir/..")"

scene_base="${workspace}/datasets/TUM_RGBD"     # Root dataset path

vocab_path="${workspace}/Vocabulary/ORBvoc.txt"
yaml_base_path="${workspace}/Examples/RGB-D/tum"
associations_path="${workspace}/Examples/RGB-D/associations"
# Define scenes: scene_name : dataset_path : yaml_name : association_file

scenes=(
    "tum_desk1:${scene_base}/rgbd_dataset_freiburg1_desk:TUM1:${associations_path}/fr1_desk.txt"
    "tum_xyz:${scene_base}/rgbd_dataset_freiburg2_xyz:TUM2:${associations_path}/fr2_xyz.txt"
    "tum_office:${scene_base}/rgbd_dataset_freiburg3_long_office_household:TUM3:${associations_path}/fr3_office.txt"
    
)

for i in $(seq 0 $experiment_count); do
    echo "[INFO] Starting experiment round $i"
    cd "$workspace" || exit 1

    for scene in "${scenes[@]}"; do
        IFS=":" read -r scene_name scene_path yaml_name association_file <<< "$scene"
        scene_tag="${scene_name}-${i}"

        echo "[INFO] Processing scene: $scene_tag"

        # Update YAML .Dataset.name field
        yq eval ".Dataset.name = \"${scene_tag}\"" "${yaml_base_path}/${yaml_name}.yaml" -i

        # Run SLAM for TUM dataset
        ./Examples/RGB-D/rgbd_tum \
            "$vocab_path" \
            "${yaml_base_path}/${yaml_name}.yaml" \
            "$scene_path" \
            "$association_file" 

        echo "[INFO] Finished scene: $scene_tag ✅"
    done
done

echo "[INFO] All TUM experiments completed! 🎉"

