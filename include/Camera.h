#ifndef CAMERA_H_
#define CAMERA_H_

#include "Render.h"

using namespace std;
namespace ORB_SLAM2
{
class KeyFrame;
class Camera{
public:

    Camera();
    Camera(KeyFrame* pKf, float near = 0.01, float far = 100);
    Camera(int w,int h, cv::Mat K,cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F), float near = 0.01, float far = 100);
    void SetPose(cv::Mat Tcw);
    void SetPose(torch::Tensor Tcw);
    
public:
    int image_height;
    int image_width;
    float _tanfovx;
    float _tanfovy;
    
    torch::Tensor _viewmatrix;
    torch::Tensor _projmatrix;
    torch::Tensor _full_projmatrix;
    torch::Tensor _campos;


    int sh_degree;
    bool prefiltered;

    cv::Mat _cam_t;
    cv::Mat _cam_R;
    cv::Mat _cam_T;

    float fx;
    float fy;
    float cx;
    float cy;

};



}//ORB_SLAM2


#endif