id=(0000 0059 0106 0169 0181 0182 0207 0465)
data_dir="$HOME/GSORB_SLAM/datasets/Scannet"

convert_pose () {
    id=$1
    pose_dir=${data_dir}/scene${id}_00/pose
    pose_file=${data_dir}/scene${id}_00/groundtruth.txt
    rm -f $pose_file
    for i in `ls $pose_dir | sort -k1 -n`; do
        echo ${i%.*} $(sed -z 's/\n/\ /g' $pose_dir/$i) >> $pose_file
    done
}


for i in ${id[@]}; do
    convert_pose $i
done
