#include "Gaussian.h"

using namespace std;
using namespace cv;
using namespace torch::indexing;
namespace ORB_SLAM2
{

Gaussian::Gaussian(const string &strSettingsFile)
{
    config = YAML::LoadFile(strSettingsFile);
    _lrMean3D = config["Mapping"]["lrsMean3D"].as<float>();
    _lrRgb = config["Mapping"]["lrsRgb"].as<float>();
    _lrRotation = config["Mapping"]["lrsUnnormRotation"].as<float>();
    _lrOpacities = config["Mapping"]["lrsLogitOpacities"].as<float>();
    _lrScales = config["Mapping"]["lrsLogScales"].as<float>();

    _lrCamQuat = config["Tracking"]["lrsCamQuat"].as<float>();
    _lrCamTrans = config["Tracking"]["lrsCamTrans"].as<float>();

    _pruneOpcities = config["Mapping"]["pruneOpcities"].as<float>();

    _background = torch::full({1,3},config["Mapping"]["backgroundColor"].as<float>()).to(GPU0, torch::kFloat32);
    _scaleModifier = config["Mapping"]["scaleModifier"].as<float>();

    _initScalarMethod = config["Mapping"]["initScalarMethod"].as<int>();

    _fx = config["Camera"]["fx"].as<float>();
    _fy = config["Camera"]["fy"].as<float>();
    _cx = config["Camera"]["cx"].as<float>();
    _cy = config["Camera"]["cy"].as<float>();


    mK = torch::tensor({_fx,(float)0.0,_cx,(float)0.0,_fy,_cy,(float)0.0,(float)0.0,(float)1.0},torch::kFloat32).reshape({3,3});

    cout<<"-----------Learning Rate-------------------\n";
    cout<<"- Position: "<<_lrMean3D<<endl;
    cout<<"- RGB: "<<_lrRgb<<endl;
    cout<<"- Rotation: "<<_lrRotation<<endl;
    cout<<"- Opacities: "<<_lrOpacities<<endl;
    cout<<"- Scales: "<<_lrScales<<endl;

    cout<<"- Quaternion: "<<_lrCamQuat<<endl;
    cout<<"- Translation: "<<_lrCamTrans<<endl;



}

void Gaussian::AddGaussianPoints(PointCloud ptcloud)
{
    torch::Tensor addMean3D = torch::from_blob(ptcloud.Points.data(),{ptcloud.Points.size(),3},torch::kFloat32).to(GPU0, true).set_requires_grad(true);
    torch::Tensor addRgb = torch::from_blob(ptcloud.Colors.data(),{ptcloud.Colors.size(),3},torch::kFloat32).to(GPU0, true).set_requires_grad(true);
    torch::Tensor addUnnormQua = torch::zeros({addMean3D.size(0),4},torch::kFloat32).index_put_({torch::indexing::Slice(),0},1).to(GPU0, true).set_requires_grad(true);
    torch::Tensor addLogitOpcities = torch::ones({addMean3D.size(0),1},torch::kFloat32).to(GPU0, true).set_requires_grad(true);


    torch::Tensor addLogScales;
    if(_initScalarMethod == eInitScalarMethod::Distance)
    {
        auto dis = torch::clamp_min(distCUDA2(addMean3D,GPU0), 0.0000001);
        addLogScales = torch::log(torch::sqrt(dis)).unsqueeze(-1).repeat({1, 3}).to(GPU0, true).set_requires_grad(true);
    }
    else if(_initScalarMethod == eInitScalarMethod::DistanceMean)
    {
        auto dis = torch::clamp_min(distCUDA2(addMean3D,GPU0), 0.0000001);
        auto sqrdis = torch::sqrt(dis);
        addLogScales = torch::log(torch::clamp_max(sqrdis,8*sqrdis.mean())).unsqueeze(-1).repeat({1, 3}).to(GPU0, true).set_requires_grad(true);
    }
    else if(_initScalarMethod == eInitScalarMethod::SinglePixel)
    {
        torch::Tensor dis = torch::from_blob(ptcloud.Points.data(),{ptcloud.Points.size(),3},torch::kFloat32).select(1,2).clone();
        addLogScales = torch::log(torch::sqrt(torch::pow(dis.div((_fx+_fy)*0.5),2))).unsqueeze(-1).repeat({1, 3}).to(GPU0,torch::kFloat32, true).set_requires_grad(true);
    }
    else
    {
        cerr<<"Unknown Init Scalar Method"<<endl;
        exit(-1);
    }

    

    if(!mMean3D.defined())
    {
        mMean3D = torch::autograd::make_variable(addMean3D.contiguous(),true);
        mRgb = torch::autograd::make_variable(addRgb.contiguous(),true);
        mUnnormQuat = torch::autograd::make_variable(addUnnormQua.contiguous(),true);    
        mLogitOpacities = torch::autograd::make_variable(addLogitOpcities.contiguous(),true);    
        mLogScales = torch::autograd::make_variable(addLogScales.contiguous(),true);
    }
    else
        UpdateOptimizerParams(addMean3D, addRgb, addLogitOpcities, addLogScales, addUnnormQua);
  

}


torch::Tensor Gaussian::InitCameraPose(const cv::Mat& Tcw)
{
    torch::Tensor t_Tcw = torch::from_blob(Tcw.data, {4, 4}, torch::kFloat32).clone().to(GPU0);

    cv::Mat cv_R = Tcw(cv::Range(0, 3), cv::Range(0, 3));
    cv::Mat cv_trans = Tcw(cv::Range(0, 3), cv::Range(3, 4));

    cv_R.convertTo(cv_R, CV_64F);  
    const cv::Quatd quat = Quatd::createFromRotMat(cv_R);

    const std::vector<float> quat_vec = {static_cast<float>(quat.w),
                                         static_cast<float>(quat.x),
                                         static_cast<float>(quat.y),
                                         static_cast<float>(quat.z)};

    const std::vector<float> trans_vec = {
        cv_trans.at<float>(0, 0),
        cv_trans.at<float>(1, 0),
        cv_trans.at<float>(2, 0)
    };

    mCamUnnormQuat = torch::from_blob((void*)quat_vec.data(), {4, 1}, torch::kFloat32).clone()
                         .to(GPU0)
                         .set_requires_grad(true);

    mCamTrans = torch::from_blob((void*)trans_vec.data(), {3, 1}, torch::kFloat32).clone()
                    .to(GPU0)
                    .set_requires_grad(true);

    return t_Tcw;
}


void Gaussian::StepUpdataForGaussian(bool set_zero)
{
    mpOptimizerGaussian->step();
    mpOptimizerGaussian->zero_grad();
}
void Gaussian::StepUpdataForPose(bool set_zero)
{
    mpOptimizerPose->step();
    mpOptimizerPose->zero_grad();
}



void Gaussian::CreateOptimizerForPose()
{
    std::vector<torch::optim::OptimizerParamGroup> optim_group;
    optim_group.reserve(2);

    optim_group.emplace_back(torch::optim::OptimizerParamGroup({mCamUnnormQuat}, std::make_unique<torch::optim::AdamOptions>(_lrCamQuat)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({mCamTrans},std::make_unique<torch::optim::AdamOptions>(_lrCamQuat)));

    for(int i=0; i<optim_group.size(); ++i)
        static_cast<torch::optim::AdamOptions&>(optim_group[i].options()).eps(1e-15);
        
    mpOptimizerPose = std::make_unique<torch::optim::Adam>(optim_group, torch::optim::AdamOptions(0.f));
}

void Gaussian::CreateOptimizerForGaussian()
{
    
    std::vector<torch::optim::OptimizerParamGroup> optim_group;
    optim_group.reserve(5);

    optim_group.emplace_back(torch::optim::OptimizerParamGroup({mMean3D}, std::make_unique<torch::optim::AdamOptions>(_lrMean3D)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({mRgb},std::make_unique<torch::optim::AdamOptions>(_lrRgb)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({mUnnormQuat},std::make_unique<torch::optim::AdamOptions>(_lrRotation)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({mLogitOpacities},std::make_unique<torch::optim::AdamOptions>(_lrOpacities)));
    optim_group.emplace_back(torch::optim::OptimizerParamGroup({mLogScales},std::make_unique<torch::optim::AdamOptions>(_lrScales)));

    for(int i=0; i<optim_group.size(); ++i)
        static_cast<torch::optim::AdamOptions&>(optim_group[i].options()).eps(1e-15);
        
   
    mpOptimizerGaussian = std::make_unique<torch::optim::Adam>(optim_group, torch::optim::AdamOptions(0.f).eps(1e-15));
}




torch::Tensor Gaussian::RemoveLowOpcitiesGaussian()
{
    //torch::Tensor to_remove = (torch::sigmoid(mLogitOpacities) < _pruneOpcities).squeeze(-1);
    torch::Tensor toRemove = (torch::sigmoid(mpOptimizerGaussian->param_groups()[3].params()[0]) < _pruneOpcities).squeeze(-1);
    return toRemove;
}

torch::Tensor Gaussian::RemoveBigGaussian(float th)
{
    torch::Tensor toRemove = (get<0>(torch::exp(mpOptimizerGaussian->param_groups()[4].params()[0]).max(1)) > th * mSceneRadius);
    return toRemove;
}

void Gaussian::UpdateOptimizerParams(torch::Tensor& newMean3D,
                                          torch::Tensor& newRgb,
                                          torch::Tensor& newLogitOpacities,
                                          torch::Tensor& newLogScales,
                                          torch::Tensor& newUnnormQua)
{
    unique_lock<mutex> lock(muxOptiParamUpdate);
    CatTensorToOptimizer( newMean3D, mMean3D, 0);
    CatTensorToOptimizer( newRgb, mRgb, 1);
    CatTensorToOptimizer( newUnnormQua, mUnnormQuat, 2);
    CatTensorToOptimizer( newLogitOpacities, mLogitOpacities, 3);
    CatTensorToOptimizer( newLogScales, mLogScales, 4);

}


void Gaussian::RemovePoints(const torch::Tensor& mask) {
    auto valid_point_mask = ~mask;
    int true_count = valid_point_mask.sum().item<int>();
    auto indices = torch::nonzero(valid_point_mask == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);

    unique_lock<mutex> lock(muxOptiParamUpdate);
    PruneOptimizer(indices, mMean3D, 0);
    PruneOptimizer(indices, mRgb, 1);
    PruneOptimizer(indices, mUnnormQuat, 2);
    PruneOptimizer(indices, mLogitOpacities, 3);
    PruneOptimizer(indices, mLogScales, 4);

}

void Gaussian::PruneOptimizer(const torch::Tensor& mask, torch::Tensor& oldTensor, int paramPosition) {
    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *mpOptimizerGaussian->state()[c10::guts::to_string(mpOptimizerGaussian->param_groups()[paramPosition].params()[0].unsafeGetTensorImpl())]));

    adamParamStates->exp_avg(adamParamStates->exp_avg().index({mask}));
    adamParamStates->exp_avg_sq(adamParamStates->exp_avg_sq().index({mask}));

    mpOptimizerGaussian->state().erase(c10::guts::to_string(mpOptimizerGaussian->param_groups()[paramPosition].params()[0].unsafeGetTensorImpl()));

    mpOptimizerGaussian->param_groups()[paramPosition].params()[0] = oldTensor.index_select(0, mask).set_requires_grad(true);
    
    oldTensor = mpOptimizerGaussian->param_groups()[paramPosition].params()[0]; // update old tensor
    
    mpOptimizerGaussian->state()[c10::guts::to_string(mpOptimizerGaussian->param_groups()[paramPosition].params()[0].unsafeGetTensorImpl())] = std::move(adamParamStates);

   
}

void Gaussian::CatTensorToOptimizer(torch::Tensor& extensionTensor,
                                    torch::Tensor& oldTensor,
                                    int paramPosition)
{
    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *mpOptimizerGaussian->state()[c10::guts::to_string(mpOptimizerGaussian->param_groups()[paramPosition].params()[0].unsafeGetTensorImpl())]));

    adamParamStates->exp_avg(torch::cat({adamParamStates->exp_avg(), torch::zeros_like(extensionTensor)}, 0));
    adamParamStates->exp_avg_sq(torch::cat({adamParamStates->exp_avg_sq(), torch::zeros_like(extensionTensor)}, 0));

    mpOptimizerGaussian->state().erase(c10::guts::to_string(mpOptimizerGaussian->param_groups()[paramPosition].params()[0].unsafeGetTensorImpl()));
    
    mpOptimizerGaussian->param_groups()[paramPosition].params()[0] = torch::cat({oldTensor, extensionTensor}, 0).set_requires_grad(true);
    oldTensor = mpOptimizerGaussian->param_groups()[paramPosition].params()[0];

    mpOptimizerGaussian->state()[c10::guts::to_string(mpOptimizerGaussian->param_groups()[paramPosition].params()[0].unsafeGetTensorImpl())] = std::move(adamParamStates);

}


void Gaussian::ResetOpacity() {
    // opacitiy activation
    auto new_opacity = torch::ones_like(inverse_sigmoid(mLogitOpacities), torch::TensorOptions().dtype(torch::kFloat32)) * 0.01f;

    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *mpOptimizerGaussian->state()[c10::guts::to_string(mpOptimizerGaussian->param_groups()[3].params()[0].unsafeGetTensorImpl())]));

    mpOptimizerGaussian->state().erase(c10::guts::to_string(mpOptimizerGaussian->param_groups()[3].params()[0].unsafeGetTensorImpl()));

    adamParamStates->exp_avg(torch::zeros_like(new_opacity));
    adamParamStates->exp_avg_sq(torch::zeros_like(new_opacity));
    // replace tensor
    mpOptimizerGaussian->param_groups()[3].params()[0] = new_opacity.set_requires_grad(true);
    mLogitOpacities = mpOptimizerGaussian->param_groups()[3].params()[0];

    mpOptimizerGaussian->state()[c10::guts::to_string(mpOptimizerGaussian->param_groups()[3].params()[0].unsafeGetTensorImpl())] = std::move(adamParamStates);
}






}//orbslam2