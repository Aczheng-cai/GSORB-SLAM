#include "Gaussian.h"
#define HISTORY_COUNT (int)10

using namespace std;
using namespace cv;
using namespace torch::indexing;
namespace ORB_SLAM2
{

Gaussian::Gaussian(const string &strSettingsFile,float opacities)
:_logit_opacities(torch::full({1,3},opacities)),
_means2D_gradient_acc(torch::zeros({1})),_timestep(0),_scale_modifier(1),
_background(torch::zeros({1,3})),_z_avg_acc_th(0.1),_GPU_device(GPU1)
{
    config = YAML::LoadFile(strSettingsFile);
    _lr_mean3D = config["Mapping"]["lrs_mean3D"].as<float>();
    _lr_rgb = config["Mapping"]["lrs_rgb"].as<float>();
    _lr_rotation = config["Mapping"]["lrs_unnorm_rotation"].as<float>();
    _lr_opacities = config["Mapping"]["lrs_logit_opacities"].as<float>();
    _lr_scales = config["Mapping"]["lrs_log_scales"].as<float>();

    _lr_cam_qua = config["Tracking"]["lrs_cam_qua"].as<float>();
    _lr_cam_trans = config["Tracking"]["lrs_cam_trans"].as<float>();

    _grad_th = config["Mapping"]["grad_th"].as<float>();
    _prune_opcities = config["Mapping"]["prune_opcities"].as<float>();

    int gpu_id = config["GPUdeivce"].as<int>();
    if(gpu_id==0)
        _GPU_device = GPU0;
    else if(gpu_id==1)
        _GPU_device = GPU1;
    else
        _GPU_device = GPU2;


    _fx = config["Camera"]["fx"].as<float>();
    _fy = config["Camera"]["fy"].as<float>();
    _cx = config["Camera"]["cx"].as<float>();
    _cy = config["Camera"]["cy"].as<float>();
    _image_w = config["Camera"]["width"].as<float>();
    _image_h = config["Camera"]["height"].as<float>();

    _K = torch::tensor({_fx,(float)0.0,_cx,(float)0.0,_fy,_cy,(float)0.0,(float)0.0,(float)1.0},torch::kFloat32).reshape({3,3});

    cout<<"-----------Learning Rate-------------------\n";
    cout<<"- lrs_mean3D: "<<_lr_mean3D<<endl;
    cout<<"- lrs_rgb: "<<_lr_rgb<<endl;
    cout<<"- lrs_unnorm_rotation: "<<_lr_rotation<<endl;
    cout<<"- lrs_logit_opacities: "<<_lr_opacities<<endl;
    cout<<"- lrs_log_scales: "<<_lr_scales<<endl;

    _background = _background.to(_GPU_device);


}

void Gaussian::AddGaussianPoints(PointCloud ptcloud)
{
    torch::Tensor add_mean3D = torch::from_blob(ptcloud.Points.data(),{ptcloud.Points.size(),3},torch::kFloat32).to(_GPU_device, true).set_requires_grad(true);
    torch::Tensor add_rgb = torch::from_blob(ptcloud.Colors.data(),{ptcloud.Colors.size(),3},torch::kFloat32).to(_GPU_device, true).set_requires_grad(true);
    torch::Tensor add_unnorm_qua = torch::zeros({add_mean3D.size(0),4},torch::kFloat32).index_put_({torch::indexing::Slice(),0},1).to(_GPU_device, true).set_requires_grad(true);
    torch::Tensor add_logit_opacities = torch::ones({add_mean3D.size(0),1},torch::kFloat32).to(_GPU_device, true).set_requires_grad(true);


    torch::Tensor add_z = torch::zeros({add_mean3D.size(0), HISTORY_COUNT}).to(_GPU_device);
    torch::Tensor add_z_denom = torch::zeros({add_mean3D.size(0), 1}).to(_GPU_device);

    torch::Tensor add_denom = torch::zeros({add_mean3D.size(0), 1}).to(_GPU_device);
    torch::Tensor add_brithday = torch::zeros({add_mean3D.size(0), 1}).to(_GPU_device);
    torch::Tensor add_states = torch::zeros({add_mean3D.size(0), 1}).to(_GPU_device);
    torch::Tensor add_change_count = torch::zeros({add_mean3D.size(0), 1}).to(_GPU_device);

    torch::Tensor add_gard_scalar_acc = torch::zeros({add_mean3D.size(0), 3}).to(_GPU_device);
    torch::Tensor add_gard_z_acc = torch::zeros({add_mean3D.size(0), 1}).to(_GPU_device);
    torch::Tensor add_gard_rgb_acc = torch::zeros({add_mean3D.size(0), 3}).to(_GPU_device);

    torch::Tensor add_s_weight = torch::zeros({add_mean3D.size(0), 3}).to(_GPU_device);
    torch::Tensor add_rgb_weight = torch::zeros({add_mean3D.size(0), 3}).to(_GPU_device);
    torch::Tensor add_z_weight = torch::zeros({add_mean3D.size(0), 1}).to(_GPU_device);

    torch::Tensor add_denom_radii = torch::zeros({add_mean3D.size(0), 1}).to(_GPU_device);

    if(!_z_error.defined())
    _z_error = add_z;
    else
    _z_error = torch::cat({_z_error,add_z});

    if(!_z_error_denom.defined())
    _z_error_denom = add_z_denom;
    else
    _z_error_denom = torch::cat({_z_error_denom,add_z_denom});

    if(!_denom.defined())
    _denom = add_denom;
    else
    _denom = torch::cat({_denom,add_denom});

    if(!_brithday.defined())
    _brithday = add_brithday;
    else
    _brithday = torch::cat({_brithday,add_brithday});

    if(!_states.defined())
    _states = add_states;
    else
    _states = torch::cat({_states,add_states});

    if(!_change_count.defined())
    _change_count = add_change_count;
    else
    _change_count = torch::cat({_states,add_change_count});


   if(!_s_weight.defined())
    _s_weight = add_s_weight;
    else
    _s_weight = torch::cat({_s_weight,add_s_weight});

    if(!_rgb_weight.defined())
    _rgb_weight = add_rgb_weight;
    else
    _rgb_weight = torch::cat({_rgb_weight,add_rgb_weight});
    if(!_z_weight.defined())
    _z_weight = add_z_weight;
    else
    _z_weight = torch::cat({_z_weight,add_z_weight});

    if(!_denom_radii.defined())
    _denom_radii = add_denom_radii;
    else
    _denom_radii = torch::cat({_denom_radii,add_denom_radii});

    if(!_gard_z_acc.defined())
    _gard_z_acc = add_gard_z_acc;
    else
    _gard_z_acc = torch::cat({_gard_z_acc,add_gard_z_acc});

    if(!_gard_rgb_acc.defined())
    _gard_rgb_acc = add_gard_rgb_acc;
    else
    _gard_rgb_acc = torch::cat({_gard_rgb_acc,add_gard_rgb_acc});

    if(!_gard_scalar_acc.defined())
    _gard_scalar_acc = add_gard_scalar_acc;
    else
    _gard_scalar_acc = torch::cat({_gard_scalar_acc,add_gard_scalar_acc});




    /*********************************************************
    auto dis = torch::clamp_min(distCUDA2(add_mean3D), 0.0000001);
    torch::Tensor add_log_scales = torch::log(torch::sqrt(dis)).unsqueeze(-1).repeat({1, 3}).to(_GPU_device, true).set_requires_grad(true);
    /*********************************************************/
    /*********************************************************
    auto dis = torch::clamp_min(distCUDA2(add_mean3D,_GPU_device), 0.0000001);
    auto sqrdis = torch::sqrt(dis);
    torch::Tensor add_log_scales = torch::log(torch::clamp_max(sqrdis,8*sqrdis.mean())).unsqueeze(-1).repeat({1, 3}).to(_GPU_device, true).set_requires_grad(true);
    /*********************************************************/

    /*********************************************************/
    torch::Tensor dis = torch::from_blob(ptcloud.Points.data(),{ptcloud.Points.size(),3},torch::kFloat32).select(1,2).clone();
    torch::Tensor add_log_scales = torch::log(torch::sqrt(torch::pow(dis.div((_fx+_fy)*0.5),2))).unsqueeze(-1).repeat({1, 3}).to(_GPU_device,torch::kFloat32, true).set_requires_grad(true);
    /*********************************************************/
    // auto meg_mean3D = torch::cat({_mean3D,add_mean3D});
    // auto meg_rgb = torch::cat({_rgb,add_rgb});
    // auto meg_unnorm_qua = torch::cat({_unnorm_qua,add_unnorm_qua});
    // auto meg_logit_opacities = torch::cat({_logit_opacities,add_logit_opacities});
    // auto meg_log_scales = torch::cat({_log_scales,add_log_scales});

    // _mean2D = torch::zeros_like(meg_mean3D).to(_GPU_device, true).set_requires_grad(true);
    // _max_radius2D = torch::zeros({meg_mean3D.size(0)}).to(_GPU_device);
    // _means2D_gradient_acc = torch::zeros({meg_mean3D.size(0),1}).to(_GPU_device);
    // _denom = torch::zeros({meg_mean3D.size(0),1}).to(_GPU_device);

    // add_mean3D = torch::autograd::make_variable(add_mean3D.contiguous(),true);
    // add_rgb = torch::autograd::make_variable(add_rgb.contiguous(),true);
    // add_unnorm_qua = torch::autograd::make_variable(add_unnorm_qua.contiguous(),true);    
    // add_logit_opacities = torch::autograd::make_variable(add_logit_opacities.contiguous(),true);    
    // add_log_scales = torch::autograd::make_variable(add_log_scales.contiguous(),true);
    if(!_mean3D.defined())
    {
        _mean3D = torch::autograd::make_variable(add_mean3D.contiguous(),true);
        _rgb = torch::autograd::make_variable(add_rgb.contiguous(),true);
        _unnorm_qua = torch::autograd::make_variable(add_unnorm_qua.contiguous(),true);    
        _logit_opacities = torch::autograd::make_variable(add_logit_opacities.contiguous(),true);    
        _log_scales = torch::autograd::make_variable(add_log_scales.contiguous(),true);
    }
    else
    UpdateOptimizerParams(add_mean3D, add_rgb, add_logit_opacities, add_log_scales, add_unnorm_qua);
  

}


void Gaussian::MergeGaussianPoints(PointCloud ptcloud)
{
    torch::Tensor add_mean3D = torch::from_blob(ptcloud.Points.data(),{ptcloud.Points.size(),3},torch::kFloat32).to(_GPU_device, true).set_requires_grad(true);
    torch::Tensor add_rgb = torch::from_blob(ptcloud.Colors.data(),{ptcloud.Colors.size(),3},torch::kFloat32).to(_GPU_device, true).set_requires_grad(true);
    torch::Tensor add_unnorm_qua = torch::zeros({add_mean3D.size(0),4},torch::kFloat32).index_put_({torch::indexing::Slice(),0},1).to(_GPU_device, true).set_requires_grad(true);
    torch::Tensor add_logit_opacities = torch::zeros({add_mean3D.size(0),1},torch::kFloat32).to(_GPU_device, true).set_requires_grad(true);

    /*********************************************************/
    auto dis = torch::clamp_min(distCUDA2(add_mean3D,_GPU_device), 0.0000001);
    torch::Tensor add_log_scales = torch::log(torch::sqrt(dis)).unsqueeze(-1).repeat({1, 3}).to(_GPU_device, true).set_requires_grad(true);
    /*********************************************************/

     _mean3D = torch::cat({_mean3D,add_mean3D});
     _rgb = torch::cat({_rgb,add_rgb});
     _unnorm_qua = torch::cat({_unnorm_qua,add_unnorm_qua});
     _logit_opacities = torch::cat({_logit_opacities,add_logit_opacities});
     _log_scales = torch::cat({_log_scales,add_log_scales});

    // _mean3D = torch::autograd::make_variable(meg_mean3D.contiguous(),true);
    // _rgb = torch::autograd::make_variable(meg_rgb.contiguous(),true);
    // _unnorm_qua = torch::autograd::make_variable(meg_unnorm_qua.contiguous(),true);    
    // _logit_opacities = torch::autograd::make_variable(meg_logit_opacities.contiguous(),true);    
    // _log_scales = torch::autograd::make_variable(meg_log_scales.contiguous(),true);

}
torch::Tensor Gaussian::InitCameraPose(const cv::Mat& Tcw)
{
    auto t_Tcw = torch::from_blob(Tcw.data,{4,4},torch::kFloat32);
    
    cv::Mat cv_R = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat cv_trans = Tcw.rowRange(0,3).col(3);
    cv_R.convertTo(cv_R,CV_64F);
    cv::Quatd cv_quatd = Quatd::createFromRotMat(cv_R);   

    std::vector<float> vec_qua({cv_quatd.w,cv_quatd.x,cv_quatd.y,cv_quatd.z});
     std::vector<float> vec_trans({cv_trans.at<float>(0),cv_trans.at<float>(1),cv_trans.at<float>(2)});

    _cam_qua_unnorm = torch::from_blob(vec_qua.data(),{4,1},torch::kFloat32).to(_GPU_device).set_requires_grad(true);
    _can_trans = torch::from_blob(vec_trans.data(),{3,1},torch::kFloat32).to(_GPU_device).set_requires_grad(true);

    _cam_qua_unnorm = torch::autograd::make_variable(_cam_qua_unnorm.contiguous(),true);
    _can_trans = torch::autograd::make_variable(_can_trans.contiguous(),true);

    return t_Tcw;
}
void Gaussian::CreateOptimizerForLocalPose()
{
    std::vector<torch::optim::OptimizerParamGroup> optim_group;
    optim_group.reserve(2);

    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_cam_qua_unnorm}, std::make_unique<torch::optim::AdamOptions>(_lr_cam_qua/2.0)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_can_trans},std::make_unique<torch::optim::AdamOptions>(_lr_cam_trans/2.0)));

    static_cast<torch::optim::AdamOptions&>(optim_group[0].options());
    static_cast<torch::optim::AdamOptions&>(optim_group[1].options());

    _optimizer_local_pose = std::make_unique<torch::optim::Adam>(optim_group, torch::optim::AdamOptions(0.f));
}

void Gaussian::CreateOptimizerForPose()
{
    std::vector<torch::optim::OptimizerParamGroup> optim_group;
    optim_group.reserve(2);

     optim_group.emplace_back(torch::optim::OptimizerParamGroup({_cam_qua_unnorm}, std::make_unique<torch::optim::AdamOptions>(_lr_cam_qua)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_can_trans},std::make_unique<torch::optim::AdamOptions>(_lr_cam_qua)));

    static_cast<torch::optim::AdamOptions&>(optim_group[0].options());
    static_cast<torch::optim::AdamOptions&>(optim_group[1].options());

    _optimizer_pose = std::make_unique<torch::optim::Adam>(optim_group, torch::optim::AdamOptions(0.f));
}

void Gaussian::StepUpdataForGaussian(bool set_zero)
{
    _optimizer_gaussian->step();
    _optimizer_gaussian->zero_grad();
}
void Gaussian::StepUpdataForPose(bool set_zero)
{
    _optimizer_pose->step();
    _optimizer_pose->zero_grad();
}
void Gaussian::StepUpdataForLocalPose(bool set_zero)
{
    _optimizer_local_pose->step();
    _optimizer_local_pose->zero_grad();
}
void Gaussian::StepUpdataForBA(bool set_zero)
{
    _optimizer_ba->step();
    _optimizer_ba->zero_grad();
}

void Gaussian::StepUpdataForLocalGaussian(bool set_zero)
{
    _optimizer_loacl_gaussian->step();
    _optimizer_loacl_gaussian->zero_grad();
}



void Gaussian::CreateOptimizerForBA()
{
    
    std::vector<torch::optim::OptimizerParamGroup> optim_group;
    optim_group.reserve(7);

    _mean3D = _mean3D.detach().set_requires_grad(true);
    _rgb = _rgb.detach().set_requires_grad(true);
    _unnorm_qua = _unnorm_qua.detach().set_requires_grad(true);
    _logit_opacities = _logit_opacities.detach().set_requires_grad(true);
    _log_scales = _log_scales.detach().set_requires_grad(true);


    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_mean3D}, std::make_unique<torch::optim::AdamOptions>(_lr_mean3D)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_rgb},std::make_unique<torch::optim::AdamOptions>(_lr_rgb)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_unnorm_qua},std::make_unique<torch::optim::AdamOptions>(_lr_rotation)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_logit_opacities},std::make_unique<torch::optim::AdamOptions>(_lr_opacities)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_log_scales},std::make_unique<torch::optim::AdamOptions>(_lr_scales)));


    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_cam_qua_unnorm}, std::make_unique<torch::optim::AdamOptions>(_lr_cam_qua)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_can_trans},std::make_unique<torch::optim::AdamOptions>(_lr_cam_trans)));

    static_cast<torch::optim::AdamOptions&>(optim_group[0].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[1].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[2].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[3].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[4].options()).eps(1e-15);

    static_cast<torch::optim::AdamOptions&>(optim_group[5].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[6].options()).eps(1e-15);

    
    _optimizer_ba = std::make_unique<torch::optim::Adam>(optim_group, torch::optim::AdamOptions(0.f).eps(1e-15));
}

void Gaussian::CreateOptimizerForLocalBA(vector<KeyFrame*> vpKF,int& opt_num,torch::Device device)
{
    
    std::vector<torch::optim::OptimizerParamGroup> optim_group;
    // optim_group.reserve(2*vpKF.size()+5);

    optim_group.reserve(5);

    _mean3D = _mean3D.detach().set_requires_grad(true);
    _rgb = _rgb.detach().set_requires_grad(true);
    _unnorm_qua = _unnorm_qua.detach().set_requires_grad(true);
    _logit_opacities = _logit_opacities.detach().set_requires_grad(true);
    _log_scales = _log_scales.detach().set_requires_grad(true);


    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_mean3D}, std::make_unique<torch::optim::AdamOptions>(_lr_mean3D)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_rgb},std::make_unique<torch::optim::AdamOptions>(_lr_rgb)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_unnorm_qua},std::make_unique<torch::optim::AdamOptions>(_lr_rotation)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_logit_opacities},std::make_unique<torch::optim::AdamOptions>(_lr_opacities)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_log_scales},std::make_unique<torch::optim::AdamOptions>(_lr_scales)));
    static_cast<torch::optim::AdamOptions&>(optim_group[0].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[1].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[2].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[3].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[4].options()).eps(1e-15);


    
    // for(int i=0; i<vpKF.size(); ++i)
    // {
    //     KeyFrame* pKF = vpKF[i];
    //     if(pKF->mnId==0) continue;

    //     cv::Mat Tcw = pKF->GetPose();
    //     auto t_Tcw = torch::from_blob(Tcw.data,{4,4},torch::kFloat32);
    //     cv::Mat cv_R = Tcw.rowRange(0,3).colRange(0,3);
    //     cv::Mat cv_trans = Tcw.rowRange(0,3).col(3);
    //     cv_R.convertTo(cv_R,CV_64F);
    //     cv::Quatd cv_quatd = Quatd::createFromRotMat(cv_R);   

    //     std::vector<float> vec_qua({cv_quatd.w,cv_quatd.x,cv_quatd.y,cv_quatd.z});
    //     std::vector<float> vec_trans({cv_trans.at<float>(0),cv_trans.at<float>(1),cv_trans.at<float>(2)});

    //     auto cam_qua_unnorm = torch::from_blob(vec_qua.data(),{4,1},torch::kFloat32).to(device).set_requires_grad(true);
    //     auto can_trans = torch::from_blob(vec_trans.data(),{3,1},torch::kFloat32).to(device).set_requires_grad(true);
    //     optim_group.emplace_back(torch::optim::OptimizerParamGroup({cam_qua_unnorm}, std::make_unique<torch::optim::AdamOptions>(_lr_cam_qua)));
    //     optim_group.emplace_back(torch::optim::OptimizerParamGroup({can_trans},std::make_unique<torch::optim::AdamOptions>(_lr_cam_trans)));
    //     static_cast<torch::optim::AdamOptions&>(optim_group[2*opt_num+5].options());
    //     static_cast<torch::optim::AdamOptions&>(optim_group[2*opt_num+6].options());
    //     ++opt_num;

    // }

    
    _optimizer_ba = std::make_unique<torch::optim::Adam>(optim_group);
}


void Gaussian::CreateOptimizerForGaussian()
{
    
    std::vector<torch::optim::OptimizerParamGroup> optim_group;
    optim_group.reserve(5);

    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_mean3D}, std::make_unique<torch::optim::AdamOptions>(_lr_mean3D)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_rgb},std::make_unique<torch::optim::AdamOptions>(_lr_rgb)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_unnorm_qua},std::make_unique<torch::optim::AdamOptions>(_lr_rotation)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_logit_opacities},std::make_unique<torch::optim::AdamOptions>(_lr_opacities)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({_log_scales},std::make_unique<torch::optim::AdamOptions>(_lr_scales)));

    static_cast<torch::optim::AdamOptions&>(optim_group[0].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[1].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[2].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[3].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[4].options()).eps(1e-15);
   
    _optimizer_gaussian = std::make_unique<torch::optim::Adam>(optim_group, torch::optim::AdamOptions(0.f).eps(1e-15));
}


void Gaussian::CreateOptimizerForLocalGaussian(OptimizerGSParam& param)
{
    _optimizer_loacl_gaussian = nullptr;
    std::vector<torch::optim::OptimizerParamGroup> optim_group;
    optim_group.reserve(5);
    param.mean3D = param.mean3D.detach().set_requires_grad(true);

    param.rgb = param.rgb.detach().set_requires_grad(true);
    param.unnorm_qua = param.unnorm_qua.detach().set_requires_grad(true);
    param.logit_opacities = param.logit_opacities.detach().set_requires_grad(true);
    param.log_scales = param.log_scales.detach().set_requires_grad(true);

    // optim_group.emplace_back(torch::optim::OptimizerParamGroup({param.mean3D.slice(1,2).set_requires_grad(true)}, std::make_unique<torch::optim::AdamOptions>(_lr_mean3D)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({param.mean3D}, std::make_unique<torch::optim::AdamOptions>(_lr_mean3D)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({param.rgb},std::make_unique<torch::optim::AdamOptions>(_lr_rgb)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({param.unnorm_qua},std::make_unique<torch::optim::AdamOptions>(_lr_rotation)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({param.logit_opacities},std::make_unique<torch::optim::AdamOptions>(_lr_opacities)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({param.log_scales},std::make_unique<torch::optim::AdamOptions>(_lr_scales)));

    static_cast<torch::optim::AdamOptions&>(optim_group[0].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[1].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[2].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[3].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optim_group[4].options()).eps(1e-15);
   
    _optimizer_loacl_gaussian = std::make_unique<torch::optim::Adam>(optim_group, torch::optim::AdamOptions(0.f).eps(1e-15));

    // _gard_scalar_acc = torch::zeros({param.mean3D.size(0), 3}).to(_GPU_device);
    // _gard_mean3D_acc = torch::zeros({param.mean3D.size(0), 3}).to(_GPU_device);
    // _gard_rgb_acc = torch::zeros({param.mean3D.size(0), 3}).to(_GPU_device);
}

void Gaussian::DensifyInfo()
{

    _means2D_gradient_acc = torch::zeros({_mean3D.size(0), 1}).to(_GPU_device);
    _denom = torch::zeros({_mean3D.size(0), 1}).to(_GPU_device);

    _means2D_gradient_acc.index_put_({_seen},_means2D_gradient_acc.index_select(0, _seen.nonzero().squeeze()) 
    + (_mean2D.grad().index_select(0, _seen.nonzero().squeeze()).slice(1,0,2).norm(2, -1, true)));
    _denom.index_put_({_seen}, _denom.index_select(0, _seen.nonzero().squeeze()) + 1);

}



void Gaussian::GetFrustumGaussian(torch::Tensor select_index,torch::Tensor Tcw,torch::Tensor& visiable,torch::Tensor& visablePoint)
{
    // cv::Mat depth = pRf.mImDepth;
    
    torch::Tensor select_mean3D = _mean3D.index_select(0, select_index).clone();

    select_mean3D = select_mean3D.to(GPU0);
    select_mean3D = torch::cat({select_mean3D,torch::ones({select_mean3D.size(0),1}).to(GPU0,torch::kFloat32)},1).unsqueeze(-1);

    Tcw = Tcw.unsqueeze(0).repeat({select_mean3D.size(0),1,1}).to(GPU0);

    torch::Tensor VisableMean3D = Tcw.bmm(select_mean3D);

    VisableMean3D= VisableMean3D.squeeze(-1).index({"...",Slice(None,3)});

    torch::Tensor GS_z = VisableMean3D.index({"...",Slice(2,None)});

    torch::Tensor tbK_ = _K.unsqueeze(0).repeat({VisableMean3D.size(0),1,1}).to(GPU0);
    
    torch::Tensor VisableMean2D3 = tbK_.bmm(VisableMean3D.unsqueeze(-1)).squeeze(-1);

    visablePoint = VisableMean2D3.index({"...",Slice(None,2)}).div(GS_z).to(torch::kLong).clone();

    visiable = torch::where(visablePoint.index({"...",Slice(0,1)})<_image_w&visablePoint.index({"...",Slice(0,1)})>0&visablePoint.index({"...",Slice(1,2)})<_image_h&visablePoint.index({"...",Slice(1,2)})>0)[0].to(torch::kLong).to(GPU0);


}

torch::Tensor Gaussian::DepthErrorFilter(torch::Tensor Depth,torch::Tensor Tcw,torch::Tensor Mask)
{

    Depth.masked_fill_(~Mask,-1);

    auto seen_index = _seen.nonzero().squeeze().to(torch::kLong);

     torch::Tensor normZ_VisableMean2D;
    torch::Tensor visiable;

    GetFrustumGaussian(seen_index,Tcw,visiable,normZ_VisableMean2D);

    auto val_v = normZ_VisableMean2D.index({visiable,Slice(0,1)});
    auto val_u = normZ_VisableMean2D.index({visiable,Slice(1,2)});

    auto obs_z = Depth.squeeze(0).index({val_u,val_v}).squeeze(-1);

    visiable = visiable.index({torch::where(obs_z!=-1)[0]});

    val_v = normZ_VisableMean2D.index({visiable,Slice(0,1)});
    val_u = normZ_VisableMean2D.index({visiable,Slice(1,2)});

    obs_z = Depth.squeeze(0).index({val_u,val_v}).squeeze(-1);

    auto local_index = seen_index.index({visiable});

    _z_error_denom.index_put_({local_index}, _z_error_denom.index_select(0, local_index) + 1);

    auto seen_count = (_z_error_denom.index({local_index}).squeeze()).to(torch::kLong);

    auto gaussian_z = _mean3D.index({local_index,Slice(2,3)}).squeeze(-1);


    _z_error.index_put_({local_index,(seen_count%HISTORY_COUNT)},abs(obs_z-gaussian_z));

    auto bug_z = _z_error.clone();

    auto sum_z = _z_error.index({local_index}).sum(1);

    torch::Tensor vis_count= torch::where(seen_count >= HISTORY_COUNT-1,
                                                   HISTORY_COUNT,
                                                   seen_count+1);


    torch::Tensor mean_z = sum_z/seen_count;
    auto repeat_mean_z = mean_z.unsqueeze(-1).repeat({1,HISTORY_COUNT});

    _z_error.index_put_({local_index},pow(_z_error.index({local_index})-repeat_mean_z,2));

    auto var_ = _z_error.index({local_index}).sum(1);

    var_ = var_/seen_count;

    torch::Tensor outline = seen_index.index({visiable.index({torch::where((var_ > 10 | (seen_count>HISTORY_COUNT/2 & mean_z==0)) )[0]})});

    auto to_move = torch::full({_mean3D.size(0)},false).to(_GPU_device,torch::kBool);

    auto a = torch::where(var_ > 10 | (seen_count>HISTORY_COUNT/2 & mean_z==0))[0];
     cv::Mat shmask = cv::Mat::zeros(Size(640,480),CV_8U);

        for(int i=0; i<a.size(0); ++i)
        {
            int id = a[i].item<int>();
            float gs_u = val_u[id].item<float>();
            float gs_v = val_v[id].item<float>();
        
            shmask.at<uchar>(gs_u,gs_v)=255;

           
                
        }
        imshow("sh",shmask);
                waitKey(0);

    to_move.index_put_({outline},true);
    cout<<"error remove: "<<to_move.sum().item<int>()<<endl;
    RemovePoints(to_move);
    c10::cuda::CUDACachingAllocator::emptyCache();


    torch::Tensor static_index = seen_index.index({visiable.index({torch::where((var_ < 0.001 ))[0]})});


    return static_index;
}

// void Gaussian::KeyframeVisableCount(torch::Tensor Depth,torch::Tensor Tcw,torch::Tensor Mask)
// {

//     Depth.masked_fill_(~Mask,-1);

//     auto seen_index = _seen.nonzero().squeeze().to(torch::kLong);

//     torch::Tensor normZ_VisableMean2D;
//     torch::Tensor visiable;

//     GetFrustumGaussian(seen_index,Tcw,visiable,normZ_VisableMean2D);

//     auto val_v = normZ_VisableMean2D.index({visiable,Slice(0,1)});
//     auto val_u = normZ_VisableMean2D.index({visiable,Slice(1,2)});

//     auto obs_z = Depth.squeeze(0).index({val_u,val_v}).squeeze(-1);

//     visiable = visiable.index({torch::where(obs_z!=-1)[0]});

//      val_v = normZ_VisableMean2D.index({visiable,Slice(0,1)});
//      val_u = normZ_VisableMean2D.index({visiable,Slice(1,2)});

//      obs_z = Depth.squeeze(0).index({val_u,val_v}).squeeze(-1);

//     auto seen_count = (_denom_keyframe.index({seen_index.index({visiable})}).squeeze()).to(torch::kLong);

//    _denom_keyframe.index_put_({seen_index.index({visiable})}, _denom_keyframe.index_select(0, seen_index.index({visiable})) + 1);

//     auto gaussian_z = _mean3D.index({seen_index.index({visiable}),Slice(2,3)}).squeeze(-1);
//     auto error = abs(obs_z-gaussian_z.data()).clone();
//     _z_avg_acc_keyframe.index_put_({seen_index.index({visiable}),seen_count},error);

// }

// torch::Tensor Gaussian::GaussianError(torch::Tensor seen_index,OptimizerGSParam afterParam)
// {
//     _gard_mean3D_acc.index_put_({seen_index},_gard_mean3D_acc.index_select(0,seen_index)/20);
//     _gard_rgb_acc.index_put_({seen_index},_gard_rgb_acc.index_select(0,seen_index)/20);


//     auto opt_mean3D = afterParam.mean3D;
//     auto vis_mean3D = _mean3D.index_select(0,seen_index);

//     auto opt_rgb = afterParam.rgb;
//     auto vis_rgb = _rgb.index_select(0,seen_index);

//     auto error_mean3D = sqrt(pow((opt_mean3D - vis_mean3D),2)).sum(1);
//     auto error_rgb = abs(opt_rgb - vis_rgb).sum(1);


//     // auto error_index = seen_index.index_select(0,{torch::where(error_rgb<0.2&error_mean3D<0.7)[0]});
//     auto error_index = torch::where(_gard_mean3D_acc<0.8*1e-4)[0];

// _gard_mean3D_acc.fill_(0);
// _gard_rgb_acc.fill_(0);

//     return error_index;
    
// }
void Gaussian::UpdateGrad(torch::Tensor seen_index)
{
    // _gard_scalar_acc.index_select(0, seen_index).index_put_({_gard_scalar_acc.index_select(0, seen_index).isnan()}, 0.0);
    // _gard_z_acc.index_select(0, seen_index).index_put_({_gard_z_acc.index_select(0, seen_index).isnan()}, 0.0);
    // _gard_rgb_acc.index_select(0, seen_index).index_put_({_gard_rgb_acc.index_select(0, seen_index).isnan()}, 0.0);

    _gard_scalar.index_put_({_gard_scalar.isnan()}, 0.0);
    _gard_z.index_put_({_gard_z.isnan()}, 0.0);
    _gard_rgb.index_put_({_gard_rgb.isnan()}, 0.0);

    _gard_scalar_acc.index_put_({seen_index},_gard_scalar_acc.index_select(0, seen_index) 
    + (_gard_scalar.grad().index_select(0, seen_index)));

    _gard_z_acc.index_put_({seen_index},_gard_z_acc.index_select(0, seen_index) 
    + (_gard_z.grad().index_select(0, seen_index).index({"...",2})).unsqueeze(-1));

    _gard_rgb_acc.index_put_({seen_index},_gard_rgb_acc.index_select(0, seen_index) 
    + (_gard_rgb.grad().index_select(0, seen_index)));

  
    // _gard_scalar_acc.index_put_({seen_index},_gard_scalar_acc.index_select(0, seen_index) 
    // + (_gard_scalar.grad().norm(2, -1, true)));

    // _gard_z_acc.index_put_({seen_index},_gard_z_acc.index_select(0, seen_index) 
    // + (_gard_z.grad().norm(2, -1, true)));

    // _gard_rgb_acc.index_put_({seen_index},_gard_rgb_acc.index_select(0, seen_index) 
    // + (_gard_rgb.grad().norm(2, -1, true)));




}
// void Gaussian::AccGrad()
// {
    

//     _scales_grad.index_put_({_scales_grad.grad().isnan()}, 0.0);
//     _mean3D_grad.index_put_({_mean3D_grad.grad().isnan()}, 0.0);
//     _rgb_grad.index_put_({_rgb_grad.grad().isnan()}, 0.0);
    

//     _gard_scalar_acc.index_put_({"..."},_gard_scalar_acc + (_scales_grad.grad()));

//     _gard_mean3D_acc.index_put_({"..."},_gard_mean3D_acc + (_mean3D_grad.grad()));

//     _gard_rgb_acc.index_put_({"..."},_gard_rgb_acc + (_rgb_grad.grad()));

 
 
// }

void Gaussian::StateCheck(torch::Tensor& seen_index,torch::Tensor& stable_index)
{

    // _gard_mean3D_acc.index_put_({"..."},_gard_mean3D_acc);
    _gard_rgb_acc.index_put_({"..."},_gard_rgb_acc);
    _gard_scalar_acc.index_put_({"..."},_gard_scalar_acc);

    auto stable_local_index = torch::where(_gard_rgb_acc.norm(2, -1, true)>config["Debug"]["rgb_th"].as<float>())[0];

    // auto stable_local_index = torch::where(_gard_mean3D_acc.norm(2, -1, true)>config["Debug"]["mean3D_th"].as<float>())[0];

    // auto stable_local_index = torch::where(_gard_scalar_acc.norm(2, -1, true)>config["Debug"]["scales_th"].as<float>())[0];


    stable_index = seen_index.index({stable_local_index});


}

void Gaussian::ClearGrad()
{
    _gard_scalar_acc.fill_(0);
    // _gard_mean3D_acc.fill_(0);
    _gard_rgb_acc.fill_(0);
}

void Gaussian::SetGradFalse(torch::Tensor false_index)
{
    false_index = false_index.to(torch::kLong);
    _mean3D.index({false_index}).detach();
    _rgb.index({false_index}).detach();
    _unnorm_qua.index({false_index}).detach();
    _logit_opacities.index({false_index}).detach();
    _log_scales.index({false_index}).detach();
    // _optimizer_gaussian->param_groups()[0].params()[0].index({false_index}) = _optimizer_gaussian->param_groups()[0].params()[0].index({false_index}).set_requires_grad(false);

    // _optimizer_gaussian->param_groups()[1].params()[0].index({false_index})= _optimizer_gaussian->param_groups()[1].params()[0].index({false_index}).set_requires_grad(false);

    // _optimizer_gaussian->param_groups()[2].params()[0].index({false_index})= _optimizer_gaussian->param_groups()[2].params()[0].index({false_index}).set_requires_grad(false);

    // _optimizer_gaussian->param_groups()[3].params()[0].index({false_index}) = _optimizer_gaussian->param_groups()[3].params()[0].index({false_index}).set_requires_grad(false);

    // _optimizer_gaussian->param_groups()[4].params()[0].index({false_index}) = _optimizer_gaussian->param_groups()[4].params()[0].index({false_index}).set_requires_grad(false);
    

}
void Gaussian::SetGradAllTrue()
{
    _optimizer_gaussian->param_groups()[0].params()[0].set_requires_grad(false);
    _optimizer_gaussian->param_groups()[1].params()[0].set_requires_grad(false);
    _optimizer_gaussian->param_groups()[2].params()[0].set_requires_grad(false);
    _optimizer_gaussian->param_groups()[3].params()[0].set_requires_grad(false);
    _optimizer_gaussian->param_groups()[4].params()[0].set_requires_grad(false);

}

void Gaussian::FixedGaussianInMask(torch::Tensor radii,torch::Tensor Depth,torch::Tensor Mask,torch::Tensor Tcw)
{
    Depth.masked_fill_(~Mask,-1);

    auto seen_index = (radii>0).nonzero().squeeze().to(torch::kLong);

    torch::Tensor normZ_VisableMean2D;
    torch::Tensor visiable;

    GetFrustumGaussian(seen_index,Tcw,visiable,normZ_VisableMean2D);

    auto val_v = normZ_VisableMean2D.index({visiable,Slice(0,1)});
    auto val_u = normZ_VisableMean2D.index({visiable,Slice(1,2)});

    auto obs_z = Depth.squeeze(0).index({val_u,val_v}).squeeze(-1);

    visiable = visiable.index({torch::where(obs_z==-1)[0]});

    auto fixed_index = seen_index.index({visiable});
    SetGradFalse(fixed_index);

}
void Gaussian::CloneAndSplit(int iters)
{
    torch::Tensor grads = _means2D_gradient_acc / _denom;
    grads.index_put_({grads.isnan()}, 0.0);

    DensifyAndClone(grads);
    DensifyAndSplit(grads,iters);


}
torch::Tensor Gaussian::Clone()
{
    torch::Tensor grads = _means2D_gradient_acc / _denom;
    grads.index_put_({grads.isnan()}, 0.0);
    return DensifyAndClone(grads);
}


torch::Tensor Gaussian::DensifyAndClone(torch::Tensor& grads) 
{
    torch::Tensor selected_pts_mask = torch::where(torch::linalg::vector_norm(grads, {2}, 1, true, torch::kFloat32) >= _grad_th,
                                                   torch::ones_like(grads.index({torch::indexing::Slice()})).to(torch::kBool),
                                                   torch::zeros_like(grads.index({torch::indexing::Slice()})).to(torch::kBool))
                                          .to(torch::kLong);

    selected_pts_mask = torch::logical_and(selected_pts_mask,
     std::get<0>(torch::exp(_log_scales).max(1)).unsqueeze(-1) <= 0.01 * _scene_radius);

    auto indices = torch::nonzero(selected_pts_mask.squeeze(-1) == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);
    torch::Tensor new_mean3D = _mean3D.index_select(0, indices);
    torch::Tensor new_rgb = _rgb.index_select(0, indices);
    torch::Tensor new_logit_opacities = _logit_opacities.index_select(0, indices);
    torch::Tensor new_log_scales = _log_scales.index_select(0, indices);
    torch::Tensor new_unnorm_qua = _unnorm_qua.index_select(0, indices);


    UpdateOptimizerParams(new_mean3D, new_rgb, new_logit_opacities, new_log_scales, new_unnorm_qua);

    return new_mean3D;
}
void Gaussian::DensifyAndSplit(torch::Tensor& grads, int total_iters) {
    static const int N = 2;
    const int n_init_points = _mean3D.size(0);
    torch::Tensor padded_grad = torch::zeros({n_init_points}).to(_GPU_device);
    padded_grad.slice(0, 0, grads.size(0)) = grads.squeeze();

    torch::Tensor selected_pts_mask = torch::where(padded_grad >= _grad_th,
     torch::ones_like(padded_grad).to(torch::kBool), torch::zeros_like(padded_grad).to(torch::kBool));

    selected_pts_mask = torch::logical_and(selected_pts_mask, get<0>(torch::exp(_optimizer_gaussian->param_groups()[4].params()[0]).max(1)) > 0.01 * _scene_radius);
    

    auto indices = torch::nonzero(selected_pts_mask.squeeze(-1) == true)
    .index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);
    
    torch::Tensor stds = torch::exp(_log_scales).index_select(0, indices).repeat({N, 1});
    torch::Tensor means = torch::zeros({stds.size(0), 3}).to(_GPU_device);
    torch::Tensor samples = torch::randn({stds.size(0), stds.size(1)}).to(_GPU_device) * stds + means;
    torch::Tensor rots = ToRotation(_unnorm_qua.index_select(0, indices)).repeat({N, 1, 1});


    torch::Tensor new_mean3D = torch::bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + _mean3D.index_select(0, indices).repeat({N, 1});
    torch::Tensor new_log_scales = torch::log(torch::exp(_log_scales).index_select(0, indices).repeat({N, 1}) / (0.8 * N));
    torch::Tensor new_unnorm_qua = _unnorm_qua.index_select(0, indices).repeat({N, 1});
    torch::Tensor new_rgb = _rgb.index_select(0, indices).repeat({N, 1});
    torch::Tensor new_logit_opacities = _logit_opacities.index_select(0, indices).repeat({N, 1});


    torch::Tensor to_remove = torch::cat({selected_pts_mask.squeeze(-1), torch::zeros({N * selected_pts_mask.sum().item<int>()}).to(torch::kBool).to(_GPU_device)});
    to_remove = torch::logical_or(to_remove, (torch::sigmoid(_logit_opacities) < _prune_opcities).squeeze(-1));
    cout<<"remove 3D Gaussian: "<<to_remove.sum()<<endl;
    // RemovePoints(to_remove);

    cout<<"split new 3D Gaussian: "<<new_mean3D.size(0)<<endl;
    UpdateOptimizerParams(new_mean3D, new_rgb, new_logit_opacities, new_log_scales, new_unnorm_qua);
    // if(total_iters > _prune_bigpt_iter)
    // {
    // to_remove = torch::logical_or(to_remove, std::get<0>(torch::exp(_log_scales).max(1)) > 0.01 * _scene_radius).squeeze(-1);
    // }

}

torch::Tensor Gaussian::RemoveLowOpcitiesGaussian()
{
    
    //torch::Tensor to_remove = (torch::sigmoid(_logit_opacities) < _prune_opcities).squeeze(-1);
    torch::Tensor to_remove = (torch::sigmoid(_optimizer_gaussian->param_groups()[3].params()[0]) < _prune_opcities).squeeze(-1);
    return to_remove;
}
torch::Tensor Gaussian::RemoveBigGaussian(float th)
{
    // torch::Tensor to_remove = (get<0>(torch::exp(_log_scales).max(1)) > th * _scene_radius);
    // float avg = (get<0>(torch::exp(_optimizer_gaussian->param_groups()[4].params()[0]).max(1))).sum().item<float>()/_optimizer_gaussian->param_groups()[4].params()[0].size(0);
    // torch::Tensor to_remove = (get<0>(torch::exp(_optimizer_gaussian->param_groups()[4].params()[0]).max(1)) > 20*avg);
    torch::Tensor to_remove = (get<0>(torch::exp(_optimizer_gaussian->param_groups()[4].params()[0]).max(1)) > th * _scene_radius);

    return to_remove;
}

void Gaussian::UpdateOptimizerParams(torch::Tensor& new_mean3D,
                                          torch::Tensor& new_rgb,
                                          torch::Tensor& new_logit_opacities,
                                          torch::Tensor& new_log_scales,
                                          torch::Tensor& new_unnorm_qua)
{
    CatTensorToOptimizer( new_mean3D, _mean3D, 0);
    CatTensorToOptimizer( new_rgb, _rgb, 1);
    CatTensorToOptimizer( new_unnorm_qua, _unnorm_qua, 2);
    CatTensorToOptimizer( new_logit_opacities, _logit_opacities, 3);
    CatTensorToOptimizer( new_log_scales, _log_scales, 4);

cout<<_mean3D.is_leaf()<<endl;
cout<<_rgb.is_leaf()<<endl;
cout<<_unnorm_qua.is_leaf()<<endl;
cout<<_logit_opacities.is_leaf()<<endl;
cout<<_mean3D.is_leaf()<<endl;


    // _means2D_gradient_acc = torch::zeros({new_mean3D.size(0), 1}).to(_GPU_device);
    // _denom = torch::zeros({new_mean3D.size(0), 1}).to(_GPU_device);
    // _max_radius2D = torch::zeros({new_mean3D.size(0)}).to(_GPU_device);
}




void Gaussian::RemovePoints(torch::Tensor mask) {
    auto valid_point_mask = ~mask;
    int true_count = valid_point_mask.sum().item<int>();
    auto indices = torch::nonzero(valid_point_mask == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);


    PruneOptimizer(indices, _mean3D, 0);
    PruneOptimizer(indices, _rgb, 1);
    PruneOptimizer(indices, _unnorm_qua, 2);
    PruneOptimizer(indices, _logit_opacities, 3);
    PruneOptimizer(indices, _log_scales, 4);
    
    _z_error_denom = _z_error_denom.index_select(0, indices);
    _z_error = _z_error.index_select(0, indices);
    _denom = _denom.index_select(0, indices);

    _brithday = _brithday.index_select(0, indices);
    _states = _states.index_select(0, indices);

    _change_count = _change_count.index_select(0, indices);

    _denom_radii = _denom_radii.index_select(0, indices);

    _s_weight = _s_weight.index_select(0, indices);
    _rgb_weight = _rgb_weight.index_select(0, indices);
    _z_weight = _z_weight.index_select(0, indices);

    _gard_scalar_acc = _gard_scalar_acc.index_select(0, indices);
    _gard_z_acc = _gard_z_acc.index_select(0, indices);
    _gard_rgb_acc = _gard_rgb_acc.index_select(0, indices);




    //_denom = _denom.index_select(0, indices);
    // _max_radius2D = _max_radius2D.index_select(0, indices);
    // _means2D_gradient_acc = _means2D_gradient_acc.index_select(0, indices);
    // _mean2D = _mean2D.index_select(0, indices).set_requires_grad(true);



}
void Gaussian::PruneOptimizer(const torch::Tensor& mask, torch::Tensor& old_tensor, int param_position) {
    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *_optimizer_gaussian->state()[c10::guts::to_string(_optimizer_gaussian->param_groups()[param_position].params()[0].unsafeGetTensorImpl())]));

    adamParamStates->exp_avg(adamParamStates->exp_avg().index({mask}));
    adamParamStates->exp_avg_sq(adamParamStates->exp_avg_sq().index({mask}));

    _optimizer_gaussian->state().erase(c10::guts::to_string(_optimizer_gaussian->param_groups()[param_position].params()[0].unsafeGetTensorImpl()));

    _optimizer_gaussian->param_groups()[param_position].params()[0] = old_tensor.index_select(0, mask).set_requires_grad(true);
    
    old_tensor = _optimizer_gaussian->param_groups()[param_position].params()[0]; // update old tensor
    
    _optimizer_gaussian->state()[c10::guts::to_string(_optimizer_gaussian->param_groups()[param_position].params()[0].unsafeGetTensorImpl())] = std::move(adamParamStates);

   
}

void Gaussian::CatTensorToOptimizer(torch::Tensor& extension_tensor,
                                    torch::Tensor& old_tensor,
                                    int param_position)
{
    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *_optimizer_gaussian->state()[c10::guts::to_string(_optimizer_gaussian->param_groups()[param_position].params()[0].unsafeGetTensorImpl())]));

    adamParamStates->exp_avg(torch::cat({adamParamStates->exp_avg(), torch::zeros_like(extension_tensor)}, 0));
    adamParamStates->exp_avg_sq(torch::cat({adamParamStates->exp_avg_sq(), torch::zeros_like(extension_tensor)}, 0));

    _optimizer_gaussian->state().erase(c10::guts::to_string(_optimizer_gaussian->param_groups()[param_position].params()[0].unsafeGetTensorImpl()));
    
    _optimizer_gaussian->param_groups()[param_position].params()[0] = torch::cat({old_tensor, extension_tensor}, 0).set_requires_grad(true);
    old_tensor = _optimizer_gaussian->param_groups()[param_position].params()[0];

    _optimizer_gaussian->state()[c10::guts::to_string(_optimizer_gaussian->param_groups()[param_position].params()[0].unsafeGetTensorImpl())] = std::move(adamParamStates);

}


void Gaussian::Reset_opacity() {
    // opacitiy activation
    auto new_opacity = torch::ones_like(inverse_sigmoid(_logit_opacities), torch::TensorOptions().dtype(torch::kFloat32)) * 0.01f;

    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *_optimizer_gaussian->state()[c10::guts::to_string(_optimizer_gaussian->param_groups()[3].params()[0].unsafeGetTensorImpl())]));

    _optimizer_gaussian->state().erase(c10::guts::to_string(_optimizer_gaussian->param_groups()[3].params()[0].unsafeGetTensorImpl()));

    adamParamStates->exp_avg(torch::zeros_like(new_opacity));
    adamParamStates->exp_avg_sq(torch::zeros_like(new_opacity));
    // replace tensor
    _optimizer_gaussian->param_groups()[3].params()[0] = new_opacity.set_requires_grad(true);
    _logit_opacities = _optimizer_gaussian->param_groups()[3].params()[0];

    _optimizer_gaussian->state()[c10::guts::to_string(_optimizer_gaussian->param_groups()[3].params()[0].unsafeGetTensorImpl())] = std::move(adamParamStates);
}






}//orbslam2