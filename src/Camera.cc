#include "Camera.h"

namespace ORB_SLAM2{

    Camera::Camera(){}
    
    Camera::Camera(int w,int h, cv::Mat K,cv::Mat Tcw, float near, float far)
    :image_height(h),image_width(w),sh_degree(1),prefiltered(false)
    {
        float z_dir = 1;
        near*=z_dir;
        far*=z_dir;

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);

        _tanfovx = (image_width / (2*fx));
        _tanfovy = (image_height / (2*fy));

        float top = _tanfovy * near;
        float bottom = -top;
        float right = _tanfovx*near;
        float left = -right; 
        Eigen::Matrix4f eig_projmat;

        eig_projmat << 2*near/(right-left), 0.0,    (right+left)/(right-left),       0.0,
                       0.0,    2 * near / (top-bottom), (top+bottom)/(top-bottom),   0.0,
                        0.0,    0.0,        z_dir*far/(far-near),   -(far * near) / (far - near),
                        0.0,    0.0,             z_dir,                   0.0;

        _projmatrix = torch::from_blob(eig_projmat.data(),{4,4},torch::kFloat32).to(torch::kCUDA); 
        _viewmatrix = torch::from_blob(Tcw.data,{4,4},torch::kFloat32).to(torch::kCUDA); //N*H*W*C

        _full_projmatrix = _viewmatrix.unsqueeze(0).bmm(_projmatrix.unsqueeze(0)).squeeze(0).to(torch::kCUDA,torch::kFloat32);

        _campos = _viewmatrix.inverse()[3].slice(0,0,3); 


    }
    Camera::Camera(KeyFrame* pKf, float near, float far){}

    void Camera::SetPose(torch::Tensor Tcw)
    {
        this->_viewmatrix = Tcw.transpose(0,1).to(torch::kCUDA);
        this->_full_projmatrix = _viewmatrix.unsqueeze(0).bmm(this->_projmatrix.unsqueeze(0)).squeeze(0).to(torch::kCUDA,torch::kFloat32);
        // this->_campos = this->_viewmatrix.inverse()[3].slice(0,0,3);
        
        this->_campos = torch::linalg::pinv(Tcw.transpose(0,1))[3].slice(0,0,3);

    }

 



}//ORBS2