#ifndef __RENDER_H_
#define __RENDER_H_
#define GPU0 {torch::kCUDA,0}
#define GPU1 {torch::kCUDA,1}
#define GPU2 {torch::kCUDA,2}
#define PRINT_SPINNER(msg)                             \
    do {                                               \
        static const char spinner[] = "|/-\\";         \
        static unsigned long i = 0;                    \
        printf("\r%s [%c]", msg, spinner[i++ % 4]);    \
        fflush(stdout);                                \
    } while (0)
#include <torch/extension.h>
#include <torch/torch.h>

#include <opencv2/core/core.hpp>
#include <iostream>
#include <cmath>
#include "eigen3/Eigen/Core"
#include <opencv2/core/eigen.hpp>
#include <mutex>
#include <condition_variable>
#include <tinyply.h>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include<numeric>
#include <fstream>
#include <filesystem>
#include <string>
#include <c10/cuda/CUDACachingAllocator.h>


#include "KeyFrame.h"
#include "Camera.h"
#include "Gaussian.h"
#include "Utils.h"
#include "Rasterizer.cuh"
#include "ORBextractor.h"
// #include"Thirdparty/wandb-cpp/wandbcpp.hpp"
#include "Map.h"
using namespace std;

namespace ORB_SLAM2
{
class Gaussian;
class Camera;
class GaussianRasterizer;
class MapPoint;
class PointCloud;
class Map;


class RenderFrame{
public:
    RenderFrame(){};
    RenderFrame(const Frame& F);
    RenderFrame(KeyFrame* pKF);
    RenderFrame& operator=(const RenderFrame& r)
    {
        return *this;
    }
    RenderFrame(const RenderFrame& F);

public:
    cv::Mat Tcw;
    cv::Mat mImRGB;
    cv::Mat mImDepth;

    long unsigned int mnId;

    cv::Mat GetPose(){return Tcw.clone();}
 
    vector<RenderFrame> mvBestCovisFrame;
    vector<RenderFrame> mvSecondCovisFrame;
    vector<RenderFrame> mvRandomCondidateFrame;
    vector<RenderFrame> mvRandomBestCovisFrame;

};


struct OptimizerGSParam
{
    torch::Tensor mean3D;//0
    torch::Tensor rgb;//1
    torch::Tensor unnorm_quat;//2
    torch::Tensor logit_opacities;//3
    torch::Tensor log_scales;//4
    // torch::Tensor mean2D;
    torch::Tensor Tcw;
};



class Render:public torch::nn::Module
{
public:
    Render(const string &strSettingsFile,const bool bMonocular);

    std::tuple<torch::Tensor, torch::Tensor,torch::Tensor>
    StartSplatting(OptimizerGSParam param,bool eval = false);

    std::tuple<torch::Tensor>
    StartSplattingRadii(OptimizerGSParam param);

    void InitGaussianPoint(RenderFrame& pRf);

    void ProjectPixel(RenderFrame& pRF,PointCloud& PtCloud,cv::Mat cvMask);

    void RemoveGaussian(float scalar=0.1);
  
    void GSParamRGBUpdata(OptimizerGSParam& param,bool isTracking=false);

    void GSParamDepthUpdata(OptimizerGSParam& param,bool isTracking=false);

    void GSParamRGBUpdata(OptimizerGSParam& param,torch::Tensor visual,bool isTracking=false);

    void GSParamDepthUpdata(OptimizerGSParam& param,torch::Tensor visual,bool isTracking=false);

    void GSParamRGB2Depth(OptimizerGSParam& paramIm,OptimizerGSParam& paramD,bool Tracking=false);

    void InitWorld(const Frame& F);

    void UpdataMaxZ();

    void SavePlyAndPrintTime();

    void AddGaussianForFrame(Frame& Frame);

    void RenderForFrame(RenderFrame& CurrentRF);

    void AddGaussian(RenderFrame& pRf, torch::Tensor& renderedIm, torch::Tensor& renderedDepth);

    void SetMap(Map* pMap_){mpMap = pMap_;}

    void RenderStartTraking(Frame& CurrentFrame,int TrackingIterTotal=0);

    void RemoveOutline(Frame& CurrentFrame);

    cv::Mat Viwer(const cv::Mat& Tcw,const int w,const int h);

public:


    shared_ptr<Camera> mpCam;
    shared_ptr<Gaussian> mpGaussian;


private:
    GaussianRasterizer mRasterizer;
    Map* mpMap;
    cv::RNG mRng;

    float _sceneRaduisDepthRatio;
    float mMaxZ;
    bool mbisInited;
    int _mappingIters;
    int _TrackingIters;
    float _medianMul;
    float _depthWeightMapping;
    float _imageWeightMapping;
    float _surDepthWeightMapping;
    float _regLongWeightMapping;
    float _regScalarWeightMapping;
    float _depthWeightTracking;
    float _imageWeightTracking;
    float _featureWeightTracking;
    float _lambda;
    bool _useWandb;
    bool _useSurDepth;
    bool _useRadiusFilter;
    bool _savePly;
    string _savePath;
    float _fx;
    float _fy;
    float _cx;
    float _cy;
    float _image_w;
    float _image_h;
    unsigned long _N = 0;
    
    YAML::Node _config;


public:

inline void Merge(vector<RenderFrame>& v1,vector<RenderFrame>& v2)
{
    v1.reserve(v1.size()+v2.size()+1);
    for(auto i:v2)
    v1.emplace_back(i);
}
    
};






}//ORB_SLAM2




#endif //