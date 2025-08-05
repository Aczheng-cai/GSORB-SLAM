#This file is part of GSORB-SLAM.
#For more information see <https://github.com/Aczheng-cai/GSORB-SLAM>
import sys
import torch
import os
import math
import argparse
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from plyfile import PlyData, PlyElement
import numpy as np
import cv2 as cv
import yaml
from torch import nn
from tqdm import tqdm
from pytorch_msssim import ms_ssim
lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

def lpips(img,gt):
    return lpips_model(torch.clamp(img.unsqueeze(0), 0.0, 1.0),
                    torch.clamp(gt.unsqueeze(0), 0.0, 1.0)).item()

class Gaussian:
    def __init__(self, sh_degree : int):
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._rgb = torch.empty(0)


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # features_dc = np.zeros((xyz.shape[0], 3, 1))
        # features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        # features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        # features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        
        rgb = np.stack((np.asarray(plydata.elements[0]["rgb_0"]),
                        np.asarray(plydata.elements[0]["rgb_1"]),
                        np.asarray(plydata.elements[0]["rgb_2"])),  axis=1)

        
        # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
        # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._rgb = nn.Parameter(torch.tensor(rgb, dtype=torch.float, device="cuda"))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda"))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda"))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda"))
        # embed()


   


    
def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    _tanfovx = (w / (2*fx))
    _tanfovy = (h / (2*fy))
    top = _tanfovy * near
    bottom = -top
    right = _tanfovx*near
    left = -right
    opengl_proj = torch.tensor([[2*near/(right-left), 0.0, (right+left)/(right-left), 0.0],        
                                [0.0, 2 * near / (top-bottom), (top+bottom)/(top-bottom), 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam

def transformed_params2rendervar(params, transformed_pts):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': params._rgb,
        'rotations': F.normalize(params._rotation),
        'opacities': torch.sigmoid(params._opacity),
        'scales': torch.exp(params._scaling),
        'means2D': torch.zeros_like(params._xyz, requires_grad=True, device="cuda") + 0
    }
    return rendervar

def get_depth_and_silhouette(pts_3D, w2c):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth of each gaussian center in camera frame
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]

    # embed()
    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 0.0
    depth_silhouette[:, 2] = 0.0
    
    return depth_silhouette

def transformed_params2depthplussilhouette(params, w2c, transformed_pts):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, w2c),
        'rotations': F.normalize(params._rotation),
        'opacities': torch.sigmoid(params._opacity),
        'scales': torch.exp(params._scaling),
        'means2D': torch.zeros_like(params._xyz, requires_grad=True, device="cuda") + 0
    }
    return rendervar

def read_file_to_2d_list(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # 读取所有行
        data_2d_list = [line.strip().split() for line in lines if not line.strip().startswith('#')]
    return data_2d_list

def quaternion2rot(quaternion):
    r = R.from_quat(quaternion)
    rot = r.as_matrix()
    return rot

def get_gt_pose_tum(gt_file: str, image_file: str):
    gt_pose_list = read_file_to_2d_list(gt_file)
    gt_im_list = read_file_to_2d_list(image_file)

    aligned_poses = []
    aligned_images = []
    aligned_depths = []

    for img_entry in gt_im_list:
        img_time = float(img_entry[0])
        for pose_entry in gt_pose_list:
            pose_time = float(pose_entry[0])
            if abs(img_time - pose_time) < 0.01:
                rotation = quaternion2rot(pose_entry[4:])
                translation = np.array(pose_entry[1:4], dtype=np.float32)

                T = torch.eye(4)
                T[:3, :3] = torch.from_numpy(rotation)
                T[:3, 3] = torch.from_numpy(translation)

                aligned_poses.append(T)
                aligned_images.append(img_entry[1])
                aligned_depths.append(img_entry[3])
                break

    return aligned_poses, aligned_images, aligned_depths

def get_gt_pose_replica(gt_file: str):
    pose_entries = read_file_to_2d_list(gt_file)
    image_count = len(pose_entries)
    aligned_poses = []
    aligned_images = [f"results/frame{str(i).zfill(6)}.jpg" for i in range(image_count)]
    aligned_depths = [f"results/depth{str(i).zfill(6)}.png" for i in range(image_count)]
    

    for pose_line in pose_entries:
        matrix_np = np.array(pose_line, dtype=np.float32).reshape(4, 4)
        T = torch.eye(4)
        T[:3, :3] = torch.from_numpy(matrix_np[:3, :3])
        T[:3, 3] = torch.from_numpy(matrix_np[:3, 3])
        aligned_poses.append(T)

    return aligned_poses, aligned_images, aligned_depths

def get_gt_pose_scannet(gt_file: str):
    pose_entries = read_file_to_2d_list(gt_file)
    image_count = len(pose_entries)
    aligned_poses = []
    aligned_images = [f"color/{str(i)}.jpg" for i in range(image_count)]
    aligned_depths = [f"depth/{str(i)}.png" for i in range(image_count)]

    for pose_line in pose_entries:
        matrix_np = np.array(pose_line[1:], dtype=np.float32).reshape(4, 4)
        T = torch.eye(4)
        T[:3, :3] = torch.from_numpy(matrix_np[:3, :3])
        T[:3, 3] = torch.from_numpy(matrix_np[:3, 3])
        aligned_poses.append(T)

    return aligned_poses, aligned_images, aligned_depths

def transform_to_frame(params, rel_w2c):

    pts = params._xyz.cuda()
    
    # Transform Centers and Unnorm Rots of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c.cuda() @ pts4.T).T[:, :3]

    return transformed_pts   

def calc_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def run_evaluation(config_path: str, config_path_association: str):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    

    # Optional: Enable video saving
    enable_video = False
    if enable_video:
        video_writer = cv.VideoWriter("render.mp4", cv.VideoWriter_fourcc(*'MJPG'), 30, (640, 480))
        
    experiments_path = config["Evalution"]["saveRootPath"] + '/' + config["Dataset"]["name"]
    
    estimation_pose_path = experiments_path + "/CarameTrajectory.txt" 
    gaussian_model_path = experiments_path + "/GaussianModel.ply"
    # Load dataset poses
    dataset_type = config["Dataset"]["type"]
    if dataset_type == "tum":
        if config_path_association is None:
            raise ValueError("Association file path is required for TUM dataset.")
        pose_list, im_list, depth_list = get_gt_pose_tum(estimation_pose_path, config_path_association)
    elif dataset_type == "replica":
        pose_list, im_list, depth_list = get_gt_pose_replica(estimation_pose_path)
    elif dataset_type == "scannet":
        pose_list, im_list, depth_list = get_gt_pose_scannet(estimation_pose_path)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Load intrinsics
    fx, fy = config["Camera"]["fx"], config["Camera"]["fy"]
    cx, cy = config["Camera"]["cx"], config["Camera"]["cy"]
    intrinsics = torch.tensor([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3, 3)

    # Load Gaussian model
    gaussian_model = Gaussian(0)
    print(f"[INFO] Loading ply file from: {gaussian_model_path}")
    gaussian_model.load_ply(gaussian_model_path)

    # Dataset paths
    frame_num = len(im_list)
    print(f"[INFO] Total frame count: {frame_num}")
    dataset_root = config["Dataset"]["path"]
    depth_scale = config["DepthMapFactor"]
    
    world_center = torch.eye(4)
    cam = None

    psnr_list, ssim_list, lpips_list, depthl1_list = [], [], [], []

    for i in tqdm(range(frame_num), desc="Evaluating"):
        # Load RGB and depth
        rgb_path = os.path.join(dataset_root, im_list[i])
        depth_path = os.path.join(dataset_root, depth_list[i])
        color_np = cv.imread(rgb_path, cv.IMREAD_UNCHANGED)
        depth_np = cv.imread(depth_path, cv.IMREAD_UNCHANGED).astype(np.float32)
        if dataset_type == "scannet":
            color_np = cv.resize(color_np,(640,480))
        color = torch.tensor(color_np).float().permute(2, 0, 1).cuda() / 255.0
        
        depth = torch.tensor(depth_np).float().unsqueeze(0).cuda() / depth_scale

        # First frame sets world coordinate
        w2c_raw = pose_list[i]
        if i == 0:
            world_center = torch.inverse(w2c_raw)
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c=torch.eye(4).numpy())

        w2c = torch.inverse(world_center @ w2c_raw).cuda()
        transformed_pts = transform_to_frame(gaussian_model, w2c)

        # Rendering color and depth
        render_rgb_input = transformed_params2rendervar(gaussian_model, transformed_pts)
        render_depth_input = transformed_params2depthplussilhouette(gaussian_model, w2c, transformed_pts)

        rendered_rgb, _, surface_depth, = Renderer(raster_settings=cam)(**render_rgb_input)
        rendered_depth, *_ = Renderer(raster_settings=cam)(**render_depth_input)

        # Mask and apply depth
        mask = (depth > 0)
        color = color * mask.tile([3,1,1])
        rendered_rgb = rendered_rgb * mask.tile([3,1,1])

        # Surface Depth evaluation
        render_depth = surface_depth[0].unsqueeze(0)
        valid_mask = mask[0].unsqueeze(0)
        rmse = torch.sqrt(((render_depth - depth[0]) ** 2)[valid_mask]).mean()
        depth_l1 = torch.abs((render_depth - depth[0])[valid_mask]).mean()

        # Metric evaluation
        psnr = calc_psnr(rendered_rgb, color).mean()
        ssim = ms_ssim(rendered_rgb.unsqueeze(0), color.unsqueeze(0), data_range=1.0, size_average=True)
        lpips_score = lpips_model(torch.clamp(rendered_rgb.unsqueeze(0), 0.0, 1.0),
                                  torch.clamp(color.unsqueeze(0), 0.0, 1.0)).item()

        # Collect metrics
        psnr_list.append(psnr.item())
        ssim_list.append(ssim.item())
        lpips_list.append(lpips_score)
        depthl1_list.append(depth_l1.item())

        # Show image
        numpy_im = (rendered_rgb * 255).clamp(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        cv.imshow("renderImage", numpy_im)
        if enable_video:
            video_writer.write(numpy_im)
        cv.waitKey(1)

    result_text = (
    f"--------\n"
    f"Replay Results:\n"
    f"[RESULT] Average PSNR:     {np.mean(psnr_list):.2f}\n"
    f"[RESULT] Average MS-SSIM:  {np.mean(ssim_list):.3f}\n"
    f"[RESULT] Average LPIPS:    {np.mean(lpips_list):.3f}\n"
    f"[RESULT] Average Depth L1: {np.mean(depthl1_list):.4f}\n"
    )

    res_path = experiments_path + "/result.txt"
    if not os.path.exists(res_path):
        with open(res_path, 'w') as f:
            f.write(result_text)
    else:
        with open(res_path, 'a') as f:
            f.write(result_text)

    print(result_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yamlPath", type=str,
                        default="./Examples/RGB-D/tum/TUM1.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--tumAss", type=str,
                        default="./Examples/RGB-D/associations/fr1_desk.txt",
                        help="Path to Associations config file")
    args = parser.parse_args()
    run_evaluation(args.yamlPath, args.tumAss)

