#ifndef _GAUSSIAN_H_
#define _GAUSSIAN_H_

#include<torch/extension.h>
#include <torch/torch.h>
#include<opencv2/core/core.hpp>
#include<iostream>
#include<cmath>
#include <eigen3/Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/quaternion.hpp>

#include <opencv2/core/eigen.hpp>
#include <tinyply.h>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include "Utils.h"
#include "spatial.h"
#include "KeyFrame.h"

using namespace std;
namespace ORB_SLAM2
{
struct OptimizerGSParam;
class KeyFrame;
class Render;


struct Point{
        int id;
        float x;
        float y;
        float z;
        float u;
        float v;
        float intensity;
};
struct GSPoint{
        float x;
        float y;
        float z;

};
struct Color{
    
    float r;
    float g;
    float b;
};

enum eInitScalarMethod{
        Distance=0,
        DistanceMean=1,
        SinglePixel=2,
    };


struct PointCloud{

    vector<GSPoint> Points;
    vector<Color> Colors;

    
    PointCloud& operator+=(const PointCloud& p) 
    {
        Points.insert(Points.end(),p.Points.begin(),p.Points.end());
        Colors.insert(Colors.end(),p.Colors.begin(),p.Colors.end());

        return *this;
    }
};



class Gaussian
{
public:

    Gaussian(const string &strSettingsFile);
        
    void AddGaussianPoints(PointCloud ptcloud);

    torch::Tensor InitCameraPose(const cv::Mat& Tcw);
   
    void StepUpdataForGaussian(bool set_zero = true);
    void StepUpdataForPose(bool set_zero=true);
    void CreateOptimizerForGaussian();
    void CreateOptimizerForPose();



    void UpdateOptimizerParams(torch::Tensor& newMean3D,
                                torch::Tensor& newRgb,
                                torch::Tensor& newLogitOpacities,
                                torch::Tensor& newLogScales,
                                torch::Tensor& newUnnormQua);

    void PruneOptimizer(const torch::Tensor& mask, torch::Tensor& oldTensor, int paramPosition);

    void CatTensorToOptimizer(torch::Tensor& extensionTensor,
                                torch::Tensor& oldTensor,
                                int paramPosition);

    void RemovePoints(const torch::Tensor& mask);
    void ResetOpacity();

    torch::Tensor RemoveLowOpcitiesGaussian();
    torch::Tensor RemoveBigGaussian(float th=0.1);





public:
   
    inline torch::Tensor GetXYZ()
    {
        unique_lock<mutex> lock(muxOptiParamUpdate);
        return this->mMean3D;
    }

    inline torch::Tensor GetRGB()
    {
        unique_lock<mutex> lock(muxOptiParamUpdate);
        return this->mRgb;
    }

      
    inline torch::Tensor GetLogScale()
    {
        unique_lock<mutex> lock(muxOptiParamUpdate);
        return this->mLogScales;
    }
    inline torch::Tensor GetUnnormQuat()
    {
        unique_lock<mutex> lock(muxOptiParamUpdate);
        return this->mUnnormQuat;
    }


    inline torch::Tensor GetLogitOpcity()
    {
        unique_lock<mutex> lock(muxOptiParamUpdate);
        return this->mLogitOpacities;
    }
    
    inline torch::Tensor GetScale()
    {
        unique_lock<mutex> lock(muxOptiParamUpdate);
        return torch::exp(this->mLogScales);
    }
 
    inline torch::Tensor GetQuat()
    {
        unique_lock<mutex> lock(muxOptiParamUpdate);
        return torch::nn::functional::normalize(this->mUnnormQuat);
    }


    inline torch::Tensor GetOpcity()
    {
        unique_lock<mutex> lock(muxOptiParamUpdate);
        return torch::sigmoid(this->mLogitOpacities);
    }


    inline torch::Tensor GetXYZ(const torch::Tensor& mask)
    {
        unique_lock<mutex> lock(muxOptiParamUpdate);
        return this->mMean3D.index_select(0,mask);
    }
    inline torch::Tensor GetRGB(const torch::Tensor& mask)
    {
        unique_lock<mutex> lock(muxOptiParamUpdate);
        return this->mRgb.index_select(0,mask);
    }

    inline torch::Tensor GetLogitOpcity(const torch::Tensor& mask)
    {
        unique_lock<mutex> lock(muxOptiParamUpdate);
        return this->mLogitOpacities.index_select(0,mask);
    }
    inline torch::Tensor GetLogScale(const torch::Tensor& mask)
    {
        unique_lock<mutex> lock(muxOptiParamUpdate);
        return this->mLogScales.index_select(0,mask);
    }
   
    inline torch::Tensor GetUnnormQuat(const torch::Tensor& mask)
    {
        unique_lock<mutex> lock(muxOptiParamUpdate);
        return this->mUnnormQuat.index_select(0,mask);
    }
    inline torch::Tensor GetIntrinsic()
    {
        return mK;
    }

    torch::Tensor mCamUnnormQuat;
    torch::Tensor mCamTrans;
    torch::Tensor _background;
    float _scaleModifier;
    float mSceneRadius;


private:
    std::mutex muxOptiParamUpdate;

    torch::Tensor mMean3D;//0
    torch::Tensor mRgb;//1
    torch::Tensor mUnnormQuat;//2
    torch::Tensor mLogitOpacities;//3
    torch::Tensor mLogScales;//4
    torch::Tensor mMean2D;

    



    float _lrMean3D = 0.0001;
    float _lrRgb = 0.0025;
    float _lrRotation = 0.001;
    float _lrOpacities = 0.05;
    float _lrScales = 0.001;

    float _lrCamQuat = 0.0004;
    float _lrCamTrans = 0.002;

    float _pruneOpcities = 0.005;

    float _initScalarMethod;


    unique_ptr<torch::optim::Adam> mpOptimizerPose;
    unique_ptr<torch::optim::Adam> mpOptimizerGaussian;

    float _fx;
    float _fy;
    float _cx;
    float _cy;

    torch::Tensor mK;
    YAML::Node config;



};



}//orbslam2

#endif