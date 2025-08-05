// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
#include "Thirdparty/diff_gaussian_rasterization/rasterize_points.h"
#include <math.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "Thirdparty/diff_gaussian_rasterization/cuda_rasterizer/config.h"
#include "Thirdparty/diff_gaussian_rasterization/cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>
	

namespace ORB_SLAM2{



std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
    const int device_num
    );

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);

struct GaussianRasterizationSettings {
    int image_height;
    int image_width;
    float tanfovx;
    float tanfovy;
    torch::Tensor bg;
    float scale_modifier;
    torch::Tensor viewmatrix;
    torch::Tensor projmatrix;
    int sh_degree;
    torch::Tensor camera_center;
    bool prefiltered;
};

torch::Tensor filter_radii(torch::Tensor means3D,
                            torch::Tensor scales,
                            torch::Tensor rotations,
                            int device_num,
                            GaussianRasterizationSettings raster_settings);

torch::Tensor
RasterizeGaussiansfilterCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
  const int image_height,
  const int image_width,
	const bool prefiltered,
    int device_num);



torch::autograd::tensor_list rasterize_gaussians(torch::Tensor means3D,
                                                 torch::Tensor means2D,
                                                 torch::Tensor sh,
                                                 torch::Tensor colors_precomp,
                                                 torch::Tensor opacities,
                                                 torch::Tensor scales,
                                                 torch::Tensor rotations,
                                                 torch::Tensor cov3Ds_precomp,
                                                 int device_num,
                                                 GaussianRasterizationSettings raster_settings);

class _RasterizeGaussians : public torch::autograd::Function<_RasterizeGaussians> {
public:
    static torch::autograd::tensor_list forward(torch::autograd::AutogradContext* ctx,
                                                torch::Tensor means3D,
                                                torch::Tensor means2D,
                                                torch::Tensor sh,
                                                torch::Tensor colors_precomp,
                                                torch::Tensor opacities,
                                                torch::Tensor scales,
                                                torch::Tensor rotations,
                                                torch::Tensor cov3Ds_precomp,
                                                torch::Tensor image_height,
                                                torch::Tensor image_width,
                                                torch::Tensor tanfovx,
                                                torch::Tensor tanfovy,
                                                torch::Tensor bg,
                                                torch::Tensor scale_modifier,
                                                torch::Tensor viewmatrix,
                                                torch::Tensor projmatrix,
                                                torch::Tensor sh_degree,
                                                torch::Tensor camera_center,
                                                torch::Tensor prefiltered,
                                                int device_num) {

        int image_height_val = image_height.item<int>();
        int image_width_val = image_width.item<int>();
        float tanfovx_val = tanfovx.item<float>();
        float tanfovy_val = tanfovy.item<float>();
        float scale_modifier_val = scale_modifier.item<float>();
        int sh_degree_val = sh_degree.item<int>();
        bool prefiltered_val = prefiltered.item<bool>();

        camera_center = camera_center.contiguous();
        int num_rendered;
        torch::Tensor color;
        torch::Tensor radii;
        torch::Tensor geomBuffer;
        torch::Tensor binningBuffer;
        torch::Tensor imgBuffer;
        torch::Tensor depth;
        // auto [num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, scaling] = RasterizeGaussiansCUDA(
        std::tie(num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth) = RasterizeGaussiansCUDA(
            bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            scale_modifier_val,
            cov3Ds_precomp,
            viewmatrix,
            projmatrix,
            tanfovx_val,
            tanfovy_val,
            image_height_val,
            image_width_val,
            sh,
            sh_degree_val,
            camera_center,
            prefiltered_val,
            device_num
            );

        ctx->save_for_backward({colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer});
        // TODO: Clean up. Too much data saved.
        ctx->saved_data["num_rendered"] = num_rendered;
        ctx->saved_data["background"] = bg;
        ctx->saved_data["scale_modifier"] = scale_modifier_val;
        ctx->saved_data["viewmatrix"] = viewmatrix;
        ctx->saved_data["projmatrix"] = projmatrix;
        ctx->saved_data["tanfovx"] = tanfovx_val;
        ctx->saved_data["tanfovy"] = tanfovy_val;
        ctx->saved_data["image_height"] = image_height_val;
        ctx->saved_data["image_width"] = image_width_val;
        ctx->saved_data["sh_degree"] = sh_degree_val;
        ctx->saved_data["camera_center"] = camera_center;
        ctx->saved_data["prefiltered"] = prefiltered_val;
        // assert(num_rendered!=0);
        return { color, radii, depth};
    }

    
    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs) {
        auto grad_out_color = grad_outputs[0];
        auto grad_out_radii = grad_outputs[1];


        auto num_rendered = ctx->saved_data["num_rendered"].to<int>();
        auto saved = ctx->get_saved_variables();
        auto colors_precomp = saved[0];
        auto means3D = saved[1];
        auto scales = saved[2];
        auto rotations = saved[3];
        auto cov3Ds_precomp = saved[4];
        auto radii = saved[5];
        auto sh = saved[6];
        auto geomBuffer = saved[7];
        auto binningBuffer = saved[8];
        auto imgBuffer = saved[9];
        torch::Tensor grad_means2D;
        torch::Tensor grad_colors_precomp;
        torch::Tensor grad_opacities;
        torch::Tensor grad_means3D;
        torch::Tensor grad_cov3Ds_precomp;
        torch::Tensor grad_sh;
        torch::Tensor grad_scales;
        torch::Tensor grad_rotations;

        std::tie(grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations) = RasterizeGaussiansBackwardCUDA(
            ctx->saved_data["background"].to<torch::Tensor>(),
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            ctx->saved_data["scale_modifier"].to<float>(),
            cov3Ds_precomp,
            ctx->saved_data["viewmatrix"].to<torch::Tensor>(),
            ctx->saved_data["projmatrix"].to<torch::Tensor>(),
            ctx->saved_data["tanfovx"].to<float>(),
            ctx->saved_data["tanfovy"].to<float>(),
            grad_out_color,
            sh,
            ctx->saved_data["sh_degree"].to<int>(),
            ctx->saved_data["camera_center"].to<torch::Tensor>(),
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer
            );

        // return gradients for all inputs, 19 in total. :D
        return {grad_means3D,
                grad_means2D,
                grad_sh,
                grad_colors_precomp,
                grad_opacities,
                grad_scales,
                grad_rotations,
                grad_cov3Ds_precomp,
                torch::Tensor(), // from here placeholder, not used: #forwards args = #backwards args.
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor()
                };
    }
};

class GaussianRasterizer : torch::nn::Module {
public:
    GaussianRasterizer(){}
    
    GaussianRasterizer(GaussianRasterizationSettings raster_settings) : raster_settings_(raster_settings) {}

    torch::Tensor mark_visible(torch::Tensor positions) {
        torch::NoGradGuard no_grad;
        auto visible = markVisible(
            positions,
            raster_settings_.viewmatrix,
            raster_settings_.projmatrix);

        return visible;
    }

    std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> forward(torch::Tensor means3D,
                                                     torch::Tensor means2D,
                                                     torch::Tensor opacities,
                                                     torch::Tensor shs = torch::Tensor(),
                                                     torch::Tensor colors_precomp = torch::Tensor(),
                                                     torch::Tensor scales = torch::Tensor(),
                                                     torch::Tensor rotations = torch::Tensor(),
                                                     torch::Tensor cov3D_precomp = torch::Tensor(),
                                                     int device_num=0) {

        if ((shs.defined() && colors_precomp.defined()) || (!shs.defined() && !colors_precomp.defined())) {
            throw std::invalid_argument("Please provide exactly one of either SHs or precomputed colors!");
        }
        if (((scales.defined() || rotations.defined()) && cov3D_precomp.defined()) ||
            (!scales.defined() && !rotations.defined() && !cov3D_precomp.defined())) {
            throw std::invalid_argument("Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!");
        }

        // Check if tensors are undefined, and if so, initialize them
        torch::Device device = {torch::kCUDA,device_num};
        if (!shs.defined()) {
            shs = torch::empty({0}, device);
        }
        if (!colors_precomp.defined()) {
            colors_precomp = torch::empty({0}, device);
        }
        if (!scales.defined()) {
            scales = torch::empty({0}, device);
        }
        if (!rotations.defined()) {
            rotations = torch::empty({0}, device);
        }
        if (!cov3D_precomp.defined()) {
            cov3D_precomp = torch::empty({0}, device);
        }
        
        auto result = rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            device_num,
            raster_settings_);

        return {result[0], result[1], result[2]};
    }

        std::tuple<torch::Tensor> Visable(torch::Tensor means3D,
                                                     torch::Tensor opacities,
                                                     torch::Tensor scales = torch::Tensor(),
                                                     torch::Tensor rotations = torch::Tensor(),
                                                     int device_num=0) {


        // Check if tensors are undefined, and if so, initialize them
        torch::Device device = {torch::kCUDA,device_num};
       
        if (!scales.defined()) {
            scales = torch::empty({0}, device);
        }
        if (!rotations.defined()) {
            rotations = torch::empty({0}, device);
        }

        auto result = filter_radii(
            means3D,
            scales,
            rotations,
            device_num,
            raster_settings_);

        return result;
    }

public:
    GaussianRasterizationSettings raster_settings_;
};


}//ORB_SLAM2