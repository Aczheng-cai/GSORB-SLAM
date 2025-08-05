#ifndef _UTILS_H_
#define _UTILS_H_
#include <torch/extension.h>
#include <torch/torch.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <tinyply.h>

#include <chrono>
#include<numeric>
#include <fstream>
#include <c10/cuda/CUDACachingAllocator.h>
#include <string>
#include "Gaussian.h"
#include "Render.h"
#include "Camera.h"
#include <filesystem>
#include <torch/script.h>


using namespace std;
using namespace cv;
using namespace std::filesystem;
namespace ORB_SLAM2
{
class Gaussian;
class Render;
class Camera;
class RenderFrame;
void printProgress(double percentage);

void WriteOutputPly(const path& file_path, const std::vector<torch::Tensor>& tensors, const std::vector<std::string>& attribute_names);
cv::Mat ImshowRGB(torch::Tensor tensor,std::string s=" ");
cv::Mat ImshowDepth(torch::Tensor tensor,std::string s=" ");
cv::Mat ImshowDepthFloat(torch::Tensor tensor,std::string s=" ");
std::tuple<torch::Tensor,torch::Tensor> ImAndDepth2tensor(const cv::Mat& cvImage,const cv::Mat& cvDepth,torch::Device device=GPU0);
torch::Tensor CvMat2Tensor(cv::Mat img,torch::Device device=GPU0);
torch::Tensor SSIM(const torch::Tensor& img1, const torch::Tensor& img2,torch::Device device=GPU0);
torch::Tensor SSIM(const torch::Tensor& img1, const torch::Tensor& img2, const torch::Tensor& mask,torch::Device device=GPU0);
torch::Tensor CreateWindow(int window_size=11, int channel=3);
torch::Tensor GaussianGenerator(int window_size, float sigma);
float PSNRMetric(const torch::Tensor& rendered_img, const torch::Tensor& gt_img);
torch::Tensor L1LossForMapping(const torch::Tensor& render_res, const torch::Tensor& gt,const torch::Tensor& mask=torch::Tensor());
torch::Tensor L1LossForTracking(const torch::Tensor& render_res, const torch::Tensor& gt,const torch::Tensor& mask=torch::Tensor());
torch::Tensor SmoothL1LossForTracking(const torch::Tensor& render_res, const torch::Tensor& gt,const torch::Tensor& mask=torch::Tensor());
unsigned long PrintCudaMenory();
torch::Tensor Rt2T(torch::Tensor Qua,torch::Tensor t);
void tic();
double toc(string func="time cost");
std::vector<std::string> ConstructListAttributes(Gaussian* pgaussian);
void SavePly(shared_ptr<Gaussian> pgaussian, const path& filePath, bool isLastIteration=true);
void Evalution(Render* pRender,const vector<cv::Mat>& Twc,const vector<cv::Mat>& vGtImage,const vector<cv::Mat>& vGtDepth,const string &filename);

inline torch::Tensor ToRotation(torch::Tensor r) {
    torch::Tensor norm = torch::sqrt(torch::sum(r.pow(2), 1));
    torch::Tensor q = r / norm.unsqueeze(-1);

    using Slice = torch::indexing::Slice;
    torch::Tensor R = torch::zeros({q.size(0), 3, 3}, torch::device(torch::kCUDA));
    torch::Tensor r0 = q.index({Slice(), 0});
    torch::Tensor x = q.index({Slice(), 1});
    torch::Tensor y = q.index({Slice(), 2});
    torch::Tensor z = q.index({Slice(), 3});

    R.index_put_({Slice(), 0, 0}, 1 - 2 * (y * y + z * z));
    R.index_put_({Slice(), 0, 1}, 2 * (x * y - r0 * z));
    R.index_put_({Slice(), 0, 2}, 2 * (x * z + r0 * y));
    R.index_put_({Slice(), 1, 0}, 2 * (x * y + r0 * z));
    R.index_put_({Slice(), 1, 1}, 1 - 2 * (x * x + z * z));
    R.index_put_({Slice(), 1, 2}, 2 * (y * z - r0 * x));
    R.index_put_({Slice(), 2, 0}, 2 * (x * z - r0 * y));
    R.index_put_({Slice(), 2, 1}, 2 * (y * z + r0 * x));
    R.index_put_({Slice(), 2, 2}, 1 - 2 * (x * x + y * y));
    return R;
}


inline torch::Tensor inverse_sigmoid(torch::Tensor x)
{
    return torch::log(x/(1-x));
}

inline float fov2focal(float fov, int pixels)
{
    return pixels / (2.0f * std::tan(fov / 2.0f));
}

inline float focal2fov(float focal, int pixels)
{
    return 2.0f * std::atan(pixels / (2.0f * focal));
}


}




#endif