
#include <math.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/types_c.h>

#include "Thirdparty/diff_gaussian_rasterization/cuda_rasterizer/config.h"
#include "Thirdparty/diff_gaussian_rasterization/cuda_rasterizer/rasterizer.h"

#include"eigen3/Eigen/Geometry"
#include"eigen3/Eigen/Dense"
#include"opencv2/core/eigen.hpp"

#include "Render.h"

 


using namespace std;
using namespace cv;
using namespace pybind11::detail;
using namespace torch::indexing;


static unsigned long MaxGPUUseOfMemory = 0;
static unsigned long sTotle_iters = 0;
static double Tracking_times=0;
static double Mapping_times=0;
static double Tracking_counts=0;
static double Mapping_counts=0;

static double mTotalGpuByte;

namespace ORB_SLAM2
{
RenderFrame::RenderFrame(const Frame& F)
:Tcw(F.mTcw),mImRGB(F.mImRGB),mImDepth(F.mImDepth),mnId(F.mnId)
{}

RenderFrame::RenderFrame(KeyFrame* pKF)
:Tcw(pKF->GetPose()),mImRGB(pKF->mImRGB.clone()),mnId(pKF->mnFrameId),mImDepth(pKF->mImDepth.clone())
{}

RenderFrame::RenderFrame(const RenderFrame& F):Tcw(F.Tcw),mImRGB(F.mImRGB),mImDepth(F.mImDepth),mnId(F.mnId),mvBestCovisFrame(F.mvBestCovisFrame),mvSecondCovisFrame(F.mvSecondCovisFrame),mvRandomCondidateFrame(F.mvRandomCondidateFrame),mvRandomBestCovisFrame(F.mvRandomBestCovisFrame)
{}


Render::Render(const string &strSettingsFile,const bool bMonocular):
mMaxZ(FLT_MIN),mbisInited(false)
{
    cudaDeviceReset();
    
    if (!torch::cuda::is_available()) {
    std::cout << "CUDA is not available! Training on CPU." << std::endl;
    exit(-1);
    }
    size_t totle_byte;
    size_t free_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte,&totle_byte);
    mTotalGpuByte=(double)totle_byte - 1024*1024*1024;
    cout<<"Reading Config Renser Parameters!\n";
    _config = YAML::LoadFile(strSettingsFile);
    cout<<"Config Renser!\n";
    _image_w = _config["Camera"]["width"].as<float>();
    _image_h = _config["Camera"]["height"].as<float>();

    _fx = _config["Camera"]["fx"].as<float>();
    _fy = _config["Camera"]["fy"].as<float>();
    _cx = _config["Camera"]["cx"].as<float>();
    _cy = _config["Camera"]["cy"].as<float>();
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = _fx;
    K.at<float>(1,1) = _fy;
    K.at<float>(0,2) = _cx;
    K.at<float>(1,2) = _cy;
    

    _sceneRaduisDepthRatio = _config["Mapping"]["raduisDepthRatio"].as<float>();
    _mappingIters = _config["Mapping"]["numIters"].as<int>();

    _depthWeightMapping = _config["Mapping"]["depthWeight"].as<float>();
    _imageWeightMapping = _config["Mapping"]["imWeight"].as<float>();
    _surDepthWeightMapping = _config["Mapping"]["surDepthWeight"].as<float>();
    _regLongWeightMapping = _config["Mapping"]["regLongWeight"].as<float>();
    _regScalarWeightMapping = _config["Mapping"]["regScalarWeight"].as<float>();

    _lambda = _config["Mapping"]["lambda"].as<float>();
    _medianMul = _config["Mapping"]["madienMul"].as<float>();

    _useWandb = _config["Debug"]["useWandb"].as<bool>();

    _useRadiusFilter = _config["Mapping"]["useRadiusFilter"].as<bool>();



    _featureWeightTracking = _config["Tracking"]["featureWeight"].as<float>();
    _imageWeightTracking = _config["Tracking"]["imWeight"].as<float>();
    _depthWeightTracking = _config["Tracking"]["depthWeight"].as<float>();

    if(_config["Dataset"]["path"].as<string>() == "dataset/Scannet/scene0181_00")
    _TrackingIters = 60;
    else
    _TrackingIters = _config["Tracking"]["numIters"].as<int>();

    _useSurDepth = _config["Tracking"]["useSurDepth"].as<bool>();

    _savePly = _config["Evalution"]["savePly"].as<bool>();
    _savePath = std::filesystem::current_path() / "experiments" / _config["Dataset"]["name"].as<std::string>();

    cudaError_t status;
    cudaError_t ct = cudaSetDevice(0);
    if (status != cudaSuccess) {
        printf("cudaSetDevice failed!\n");
        printf("%s",ct);
    }

    
    mpGaussian = make_shared<Gaussian>(strSettingsFile);
    mpCam = make_shared<Camera>(_image_w, _image_h, K);



    // cout<<"-----------Mapping Parameters-------------------\n";
    // cout<<"- Scene.RaduisDepthRatio: "<<_sceneRaduisDepthRatio<<endl;
    // cout<<"- Mapping.num_iters: "<<_mapping_iters<<endl;
    // cout<<"- Mapping.depth_loss_weight: "<<_depthLossWeight<<endl;
    // cout<<"- Mapping.im_loss_weight: "<<_imageLossWeight<<endl;
    // cout<<"- Mapping.lambda: "<<_lambda<<endl;
    // cout<<"- Mapping.feature_weight: "<<_config["Tracking"]["feature_weight"].as<float>()<<endl;


    

    GaussianRasterizationSettings RasterSettings_ = GaussianRasterizationSettings{
    .image_height = static_cast<int>(mpCam->image_height),
    .image_width = static_cast<int>(mpCam->image_width),
    .tanfovx = mpCam->_tanfovx,
    .tanfovy = mpCam->_tanfovy,
    .bg = mpGaussian->_background,
    .scale_modifier = mpGaussian->_scaleModifier,
    .viewmatrix = mpCam->_viewmatrix,
    .projmatrix = mpCam->_full_projmatrix,
    .sh_degree = mpCam->sh_degree,
    .camera_center = mpCam->_campos,
    .prefiltered = false
    };

    mRasterizer = GaussianRasterizer(RasterSettings_);


    // if(_config["Debug"]["useWandb"].as<bool>())
    //     wandbcpp::init({.project = "GSORBSLAM", .tags = {"basic"}, .group = "loss"});
     
}



void Render::SavePlyAndPrintTime()
{
    cout<<"MaxGPUUseOfMemory: "<<MaxGPUUseOfMemory<<endl;
    cout<<"Total Gaussian: "<<mpGaussian->GetXYZ().size(0)<<endl;
    cout<<"Avg Tracking Times: "<<Tracking_times/Tracking_counts<<" s"<<endl;
    cout<<"Avg Mapping Times: "<<Mapping_times/Mapping_counts<<" s"<<endl;
    cout<<"Total Tracking Times: "<<Tracking_times<<" s"<<endl;
    cout<<"Total Mapping Times: "<<Mapping_times<<" s"<<endl;
    if(_savePly)
        SavePly(mpGaussian,_savePath);
}

cv::Mat Render::Viwer(const cv::Mat& Tcw,const int w,const int h)
{
    if(!mbisInited) return cv::Mat();
    torch::NoGradGuard no_grad;
    OptimizerGSParam paramGS;

    paramGS.Tcw = torch::from_blob(Tcw.data,{4,4},torch::kFloat32).cuda().clone();

    torch::Tensor renderedImage;
    
    GSParamRGBUpdata(paramGS);
    std::tie(renderedImage, ignore, ignore) = StartSplatting(paramGS);

    return  ImshowRGB(renderedImage);
}



void Render::AddGaussianForFrame(Frame& Frame)
{

    RenderFrame pRF(Frame);
    OptimizerGSParam paramGS;

    cv::Mat im = pRF.mImRGB;
    cv::Mat depth = pRF.mImDepth;
    paramGS.Tcw = torch::from_blob(pRF.GetPose().data,{4,4},torch::kFloat32).to(GPU0);
    GSParamDepthUpdata(paramGS);
    auto[renderedDepth, _1, _0]  = StartSplatting(paramGS);
    GSParamRGBUpdata(paramGS);
    auto[renderedImage, _2, radii] = StartSplatting(paramGS);

    if(Frame.mnId % 50 ==0)
    {
        RemoveGaussian();
        UpdataMaxZ(); 
        if(PrintCudaMenory() > mTotalGpuByte)
            c10::cuda::CUDACachingAllocator::emptyCache();
    }

    AddGaussian(pRF,renderedImage,renderedDepth); 

    KeyFrame* pRefKF = Frame.mpReferenceKF;
    set<int> sReaptId;
    set<int> sReaptTrackingId;

    set<KeyFrame*> spVisReaptId;

    cv::Mat RFTwc = cv::Mat::eye(4,4,CV_32F);
    pRefKF->GetPoseInverse().rowRange(0,3).colRange(0,3).copyTo(RFTwc.rowRange(0,3).colRange(0,3));
    pRefKF->GetCameraCenter().copyTo(RFTwc.rowRange(0,3).col(3));
    cv::Mat RFTcw = RFTwc.inv();
    vector<KeyFrame*> vpKeyFrameDataset = mpMap->GetAllKeyFrames();
    torch::Tensor tK = mpGaussian->GetIntrinsic().reshape({3,3}).to(GPU0,torch::kFloat32);
    int RandomNum = pRefKF->mvReferentRandomPoints.size()/4;
    auto tbK = tK.unsqueeze(0).repeat({RandomNum,1,1});
    torch::Tensor tXw4 = torch::from_blob(pRefKF->mvReferentRandomPoints.data(),{RandomNum,4,1},torch::kFloat32).to(GPU0);

    int KF_dataset_num = vpKeyFrameDataset.size();
    int nn = 9;
    int n = 11;
    vector<KeyFrame*> vpNeighKFs = pRefKF->GetVectorCovisibleKeyFrames();
    int edge = 20;
    auto edge_h = (_image_w-20)*torch::ones({RandomNum,1},torch::kFloat32).to(GPU0);
    auto edge_w = (_image_h-20)*torch::ones({RandomNum,1},torch::kFloat32).to(GPU0);
    pRF.mvBestCovisFrame.emplace_back(pRF);   
    sReaptId.insert(pRF.mnId);

    for(auto pKF:vpKeyFrameDataset)
    {
        if(sReaptId.count(pKF->mnFrameId)>=1) continue;
        RenderFrame pRFBest(pKF);

        pRF.mvRandomBestCovisFrame.emplace_back(pRFBest);
        sReaptId.insert(pKF->mnFrameId);
        spVisReaptId.insert(pKF);

        if(pRF.mvRandomBestCovisFrame.size()>=4) break;

    }

    float LastOverlopRatio=0;

    for(int i=0;i<vpNeighKFs.size();++i)
    {
        KeyFrame* pKF_ = vpNeighKFs[i];
        if(sReaptId.count(pKF_->mnFrameId)>=1) continue;
        cv::Mat LastRFTwc = cv::Mat::eye(4,4,CV_32F);
        pKF_->GetPoseInverse().rowRange(0,3).colRange(0,3).copyTo(LastRFTwc.rowRange(0,3).colRange(0,3));
        pKF_->GetCameraCenter().copyTo(LastRFTwc.rowRange(0,3).col(3));
        cv::Mat Velocity = (RFTcw*LastRFTwc).inv();
        torch::Tensor tVelocity = CvMat2Tensor(Velocity,GPU0);
        auto tbTcw = tVelocity.repeat({RandomNum,1,1});
        auto tXc4 = (tbTcw.bmm(tXw4)).transpose(1,2);
        auto tXc3 = tXc4.index({"...",Slice({None,3})});
        tXc3 = tXc3.div((tXc3.index({"...",0,2})+1e-6).unsqueeze(-1).unsqueeze(-1).repeat({1,1,3}));
        auto pixel_uv = (tbK.bmm(tXc3.transpose(1,2)));

        auto inside = (pixel_uv.index({"...",0,0})>edge
        &pixel_uv.index({"...",0,0})<_image_w-edge
        &pixel_uv.index({"...",1,0})>edge
        &pixel_uv.index({"...",1,0})<_image_h-edge);

        float inside_num = inside.sum().item<float>();

        float Ratio = (float)inside_num/RandomNum;
        if(abs(Ratio-LastOverlopRatio)>0.07)
        {
            if(pKF_->isBad()) continue;
            RenderFrame pRFBest(pKF_);
            pRF.mvBestCovisFrame.emplace_back(pRFBest);
            sReaptId.insert(pKF_->mnFrameId);
            LastOverlopRatio = Ratio;
            spVisReaptId.insert(pKF_);

        }
       

        if(pRF.mvBestCovisFrame.size()>=n) break;

    }
    
    // for(auto r:sReaptId)
    // cout<<r<<" ";
    // cout<<vpNeighKFs.size();
    // cout<<endl;
    LastOverlopRatio=0;
    if(pRF.mvBestCovisFrame.size()<n)
    {
        for(auto pKF:vpKeyFrameDataset)
        {
            if(sReaptId.count(pKF->mnFrameId)>=1) continue;
            cv::Mat LastRFTwc = cv::Mat::eye(4,4,CV_32F);
            pKF->GetPoseInverse().rowRange(0,3).colRange(0,3).copyTo(LastRFTwc.rowRange(0,3).colRange(0,3));
            pKF->GetCameraCenter().copyTo(LastRFTwc.rowRange(0,3).col(3));
            cv::Mat Velocity = (RFTcw*LastRFTwc).inv();
            torch::Tensor tVelocity = CvMat2Tensor(Velocity,GPU0);

            auto tbTcw = tVelocity.repeat({RandomNum,1,1});
            auto tXc4 = (tbTcw.bmm(tXw4)).transpose(1,2);
            auto tXc3 = tXc4.index({"...",Slice({None,3})});
            tXc3 = tXc3.div((tXc3.index({"...",0,2})+1e-6).unsqueeze(-1).unsqueeze(-1).repeat({1,1,3}));

            auto pixel_uv = (tbK.bmm(tXc3.transpose(1,2)));

            auto inside = (pixel_uv.index({"...",0,0})>edge
            &pixel_uv.index({"...",0,0})<_image_w-edge
            &pixel_uv.index({"...",1,0})>edge
            &pixel_uv.index({"...",1,0})<_image_h-edge);

            float inside_num = inside.sum().item<float>();

            float Ratio = (float)inside_num/RandomNum;

            if(Ratio > 0.3 && abs(Ratio-LastOverlopRatio)>0.05)
            {
                if(pKF->isBad()) continue;
                RenderFrame pRFBest(pKF);
                pRF.mvBestCovisFrame.emplace_back(pRFBest);
                sReaptId.insert(pKF->mnFrameId); 
                spVisReaptId.insert(pKF);
                LastOverlopRatio = Ratio;
            }
           

            if(pRF.mvBestCovisFrame.size()>=n) break;
            
        }
    }
    
//    for(auto r:sReaptId)
//     cout<<r<<" ";
//     cout<<endl;

    sort(vpKeyFrameDataset.begin(),vpKeyFrameDataset.end(),[](KeyFrame* k1,KeyFrame* k2){return k1->mRenderedNum > k2->mRenderedNum;});
    for(auto pKF:vpKeyFrameDataset)
    {
        if(pKF->mRenderedNum==0) break;
        
            if(pKF->isBad()) continue;
            RenderFrame pRFBest(pKF);
            pRF.mvSecondCovisFrame.emplace_back(pRFBest);
            sReaptId.insert(pKF->mnFrameId); 
            spVisReaptId.insert(pKF);
            pKF->mRenderedNum=0;
        
        if(pRF.mvSecondCovisFrame.size()>=5) break;
        
    }

    // for(auto r:sReaptId)
    //     cout<<r<<" ";
    //     cout<<endl;
    // }
    

    while(KF_dataset_num!=0 && pRF.mvRandomCondidateFrame.size()+pRF.mvBestCovisFrame.size()<n+nn)
    {
        int k = mRng.uniform(0,vpKeyFrameDataset.size());
        KeyFrame* pKF_ = vpKeyFrameDataset[k];
        if(pKF_->mnFrameId > pRF.mnId) continue;
        if(sReaptId.count(pKF_->mnFrameId)<1)
        {
            if(pKF_->isBad()) continue;
            pRF.mvRandomCondidateFrame.emplace_back(RenderFrame(pKF_));
            sReaptId.insert(pKF_->mnFrameId);
            spVisReaptId.insert(pKF_);
        }
        --KF_dataset_num;
        
    }
    // for(auto r:sReaptId)
    // cout<<r<<" ";
    // cout<<endl;

    // mpMap->UpdateRenderFrame(spVisReaptId);


    RenderForFrame(pRF);

}


void Render::RenderForFrame(RenderFrame& CurrentRF)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    vector<RenderFrame> CondidateRenderFrame;
    CondidateRenderFrame.emplace_back(CurrentRF);
    Merge(CondidateRenderFrame,CurrentRF.mvBestCovisFrame);
    Merge(CondidateRenderFrame,CurrentRF.mvSecondCovisFrame);
    Merge(CondidateRenderFrame,CurrentRF.mvRandomCondidateFrame);
    Merge(CondidateRenderFrame,CurrentRF.mvRandomBestCovisFrame);

    int N_iters = CondidateRenderFrame.size();
    torch::Tensor renderedImage;
    torch::Tensor renderedDepth;
    torch::Tensor renderedSurdepth;

    auto maxScalar = 0.1*mpGaussian->mSceneRadius;

    for(int iter=0; iter<_mappingIters; ++iter)
    {
        OptimizerGSParam paramGS,paramGSD;
        int k = mRng.uniform(0,N_iters);
        RenderFrame pRF  = CondidateRenderFrame[k];
        // if(pRF_.mnId==0) continue;
        cv::Mat depth = pRF.mImDepth;
        cv::Mat im = pRF.mImRGB;

        auto[tGtImage,tGtDepth] = ImAndDepth2tensor(im,depth,GPU0);

        paramGS.Tcw = torch::from_blob(pRF.GetPose().data,{4,4},torch::kFloat32).to(GPU0);

        if(_useRadiusFilter)
        {
            GSParamRGBUpdata(paramGS);
            auto[radii] = StartSplattingRadii(paramGS);
            torch::Tensor visibleIndex = (radii>0).nonzero().squeeze().to(torch::kLong); 
            GSParamDepthUpdata(paramGS,visibleIndex);
            std::tie(renderedDepth, ignore, ignore) = StartSplatting(paramGS);
            GSParamRGBUpdata(paramGS,visibleIndex);
            std::tie(renderedImage, renderedSurdepth, ignore) = StartSplatting(paramGS);
        }
        else
        {
            GSParamDepthUpdata(paramGS);
            std::tie(renderedDepth, ignore, ignore) = StartSplatting(paramGS);
            GSParamRGBUpdata(paramGS);
            std::tie(renderedImage, renderedSurdepth, ignore) = StartSplatting(paramGS);
        }
        
        


        auto tvaildMask = (tGtDepth>0).to(GPU0);
        auto tvaildMasksur = (tGtDepth>0&renderedDepth[1]>0.99).to(GPU0);

        auto image_loss = _lambda * L1LossForMapping(renderedImage,tGtImage) + (1-_lambda) * (1.0 - SSIM(renderedImage,tGtImage,GPU0));
        auto depth_loss = L1LossForMapping(renderedDepth[0],tGtDepth,tvaildMask.detach());

        auto surdepth_loss = L1LossForMapping(renderedSurdepth,tGtDepth,tvaildMasksur.detach());
 
        auto big_scalar_mask = where(torch::exp(paramGS.log_scales)>maxScalar)[0];
        auto reg_scalar = (get<0>(torch::exp(paramGS.log_scales.index_select(0,big_scalar_mask)).max(1)) - maxScalar).sum();

        auto max_scale  = get<0>(torch::exp(paramGS.log_scales.index_select(0,big_scalar_mask)).max(1));
        auto min_scale = get<0>(torch::exp(paramGS.log_scales.index_select(0,big_scalar_mask)).min(1));
        auto reg_long = (max_scale - min_scale).mean();
       
       torch::Tensor loss = _imageWeightMapping*image_loss + _depthWeightMapping*depth_loss + _surDepthWeightMapping*surdepth_loss + _regLongWeightMapping*reg_long + _regScalarWeightMapping*reg_scalar;

        loss.backward();
        
        {
            torch::NoGradGuard no_grad;
            mpGaussian->StepUpdataForGaussian();

            ++Mapping_counts;

            if(PrintCudaMenory() > mTotalGpuByte)
                c10::cuda::CUDACachingAllocator::emptyCache();

        }//no_grad
    }
     
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double tmapping= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    // cout<<"Tracking Time Cost: "<<tmapping<<endl;
    // cout<<"Mapping Avg Time Cost: "<<tmapping/_mappingIters<<endl;
    Mapping_times += tmapping;



}


void Render::InitWorld(const Frame& F)
{  
   
    RenderFrame pCurrentRf = RenderFrame(F);
    // cout<<"Gaussian Init\n";

    InitGaussianPoint(pCurrentRf);

    mpGaussian->CreateOptimizerForGaussian();

    cv::Mat im = pCurrentRf.mImRGB;
    cv::Mat depth = pCurrentRf.mImDepth;
    OptimizerGSParam paramGS;
    
    paramGS.Tcw = torch::from_blob(pCurrentRf.GetPose().data,{4,4},torch::kFloat32).to(GPU0).clone();
    auto[tGtImage,tGtDepth] = ImAndDepth2tensor(im,depth,GPU0);

    auto tVailMask = (tGtDepth>0).to(GPU0);
 
    torch::Tensor renderedImage;
    torch::Tensor renderedDepth;
    torch::Tensor renderedSurdepth;


    for(int iter=0; iter<200; ++iter)
    {

        GSParamRGBUpdata(paramGS);
        std::tie(renderedImage, renderedSurdepth, ignore) = StartSplatting(paramGS);

        GSParamDepthUpdata(paramGS);
        std::tie(renderedDepth, ignore, ignore) = StartSplatting(paramGS);

        auto imageLoss = _lambda * L1LossForMapping(renderedImage,tGtImage) + (1-_lambda) * (1.0 - SSIM(renderedImage,tGtImage,GPU0));

        auto depthLoss = L1LossForMapping(renderedDepth[0],tGtDepth,tVailMask.detach());

        auto surdepthLoss = L1LossForMapping(renderedSurdepth,tGtDepth,tVailMask.detach());

        auto loss = _imageWeightMapping*imageLoss + 0.1*surdepthLoss + _depthWeightMapping*depthLoss;


        loss.backward();
        
        {
            torch::NoGradGuard no_grad;
            mpGaussian->StepUpdataForGaussian();

            ++sTotle_iters;

        PRINT_SPINNER("Init World.");
        }//no_grad

    }
    mbisInited=true;


}



void Render::AddGaussian(RenderFrame& pRf, torch::Tensor& renderedIm, torch::Tensor& renderedDepth)
{
   
    auto gray = (renderedIm[0]*299+renderedIm[1]*587+renderedIm[2]*114)/1000;
    auto tBlackMask = (gray < 50/255.0);

    cv::Mat depth = pRf.mImDepth.clone();
    auto tGtDepth = CvMat2Tensor(depth,GPU0);
  
    auto tDiffDepth = abs(tGtDepth - renderedDepth[0]);

    auto tDiffDepthMask = torch::logical_and(tDiffDepth<0.05,tGtDepth>0);
    tDiffDepthMask = tDiffDepthMask & (renderedDepth[0]>0);

    auto sum_val = tDiffDepth.masked_select(tDiffDepthMask).sum();
    auto avg_val = sum_val/tDiffDepthMask.sum();
    
    auto medie_val = tDiffDepth.masked_select(tDiffDepthMask).median();
    auto th = avg_val.item<float>()+_medianMul*medie_val.item<float>();
    
    if(th<0.01) th=0.01;
    // cout<<"  Add th:"<<th<<endl;


    auto c1 = ~(renderedDepth[1]>0.99) & tBlackMask & tDiffDepth > th;
    auto c2 = (renderedDepth[1]<0.8);
    auto tAddMask = c1 | c2;
    cv::Mat AddMask = ImshowDepth(tAddMask).clone();

    {
        torch::NoGradGuard no_grad;
        PointCloud PtCloud;
        ProjectPixel(pRf,PtCloud,AddMask);
        mpGaussian->AddGaussianPoints(PtCloud);
    }


}



void Render::RemoveGaussian(float scalar)
{
 
    auto to_remove_op = mpGaussian->RemoveLowOpcitiesGaussian();
    // auto to_remove_sc = mpGaussian->RemoveBigGaussian(scalar);

    // auto to_move = torch::logical_or(to_remove_op,to_remove_sc);
    auto to_move = to_remove_op;
 
    if(to_move.sum().item<int>()>0)
    {
    //    cout<<"Remove num:"<<to_move.sum().item<int>()<<endl;
        mpGaussian->RemovePoints(to_move);
        
    }

    c10::cuda::CUDACachingAllocator::emptyCache();

}
void Render::ProjectPixel(RenderFrame& pRF, PointCloud& PtCloud, cv::Mat cvMask)
 {
    const cv::Mat depth = pRF.mImDepth;
    cv::Mat im = pRF.mImRGB;
    cv::Mat Twc = pRF.GetPose().inv();
    cv::Mat split_im[3];
    im.convertTo(im,CV_32FC3,1.0/255.0);
    split(im,split_im);

    for(int i=0; i<im.rows; ++i)
        for(int j=0; j<im.cols; ++j)
        {
            if(cvMask.at<uchar>(i,j) < 250) continue;
            float z = depth.at<float>(i,j);
            float x=0;
            float y=0;
            if(z > 0 )
            {
                x = (j - _cx)*z / _fx;
                y = (i - _cy)*z / _fy;
                cv::Mat xyz_c = (cv::Mat_<float>(4,1)<<x,y,z,1);
                cv::Mat xyz_w = Twc*xyz_c;
                GSPoint pt = {
                    .x = xyz_w.at<float>(0),
                    .y = xyz_w.at<float>(1),
                    .z = xyz_w.at<float>(2)};
                Color rgb = {
                .r = split_im[0].at<float>(i,j),
                .g = split_im[1].at<float>(i,j),
                .b = split_im[2].at<float>(i,j)
                };
               
                mMaxZ = max(mMaxZ,z);
                PtCloud.Points.emplace_back(pt);
                PtCloud.Colors.emplace_back(rgb);

            }
        }
 }
 


void Render::UpdataMaxZ()
{
    mpGaussian->mSceneRadius = mMaxZ/_sceneRaduisDepthRatio;
}



void Render::InitGaussianPoint(RenderFrame& pRf)
{
    const cv::Mat depth = pRf.mImDepth;
    cv::Mat im = pRf.mImRGB;

    cv::Mat Twc = pRf.GetPose().inv();
    PointCloud pt_cloud;
    cv::Mat split_im[3];
    im.convertTo(im,CV_32FC3,1.0/255.0);
    split(im,split_im);

    for(int i=0; i<im.rows; ++i)
        for(int j=0; j<im.cols; ++j)
        {
            float z = depth.at<float>(i,j);
            if(z > 0 )
            {
                float x = (j - _cx)*z / _fx;
                float y = (i - _cy)*z / _fy;
                cv::Mat xyz_c = (cv::Mat_<float>(4,1)<<x,y,z,1);
                cv::Mat xyz_w = Twc*xyz_c;
                GSPoint pt = {
                    .x = xyz_w.at<float>(0),
                    .y = xyz_w.at<float>(1),
                    .z = xyz_w.at<float>(2)};
                Color rgb = {
                    .r = split_im[0].at<float>(i,j),
                    .g = split_im[1].at<float>(i,j),
                    .b = split_im[2].at<float>(i,j)
                };
                
                pt_cloud.Points.emplace_back(pt);
                pt_cloud.Colors.emplace_back(rgb);

                mMaxZ = max(mMaxZ,z);
            }
        }
    mpGaussian->AddGaussianPoints(pt_cloud);
    mpCam->_cam_T = pRf.GetPose();
    mpGaussian->mSceneRadius = mMaxZ/_sceneRaduisDepthRatio;
    cout<<"Init Gaussian Num: "<<mpGaussian->GetXYZ().size(0)<<endl;
}



std::tuple<torch::Tensor, torch::Tensor,torch::Tensor>
Render::StartSplatting(OptimizerGSParam param,bool eval)
{

    auto shs = torch::Tensor();
    auto pre_cov3D = torch::Tensor();
    auto rgb = torch::Tensor();
    auto mean3D = torch::Tensor();
    
    auto Tcw = param.Tcw;
    GaussianRasterizer Rasterizer = mRasterizer;
    
    if(eval)
     mean3D = param.mean3D.detach();
    else
     mean3D = param.mean3D.to(GPU0);

    if(_useRadiusFilter)
    {
        mpCam->SetPose(param.Tcw);

        GaussianRasterizationSettings RasterSettings_ = GaussianRasterizationSettings{
        .image_height = static_cast<int>(mpCam->image_height),
        .image_width = static_cast<int>(mpCam->image_width),
        .tanfovx = mpCam->_tanfovx,
        .tanfovy = mpCam->_tanfovy,
        .bg = mpGaussian->_background,
        .scale_modifier = mpGaussian->_scaleModifier,
        .viewmatrix = mpCam->_viewmatrix,
        .projmatrix = mpCam->_full_projmatrix,
        .sh_degree = mpCam->sh_degree,
        .camera_center = mpCam->_campos,
        .prefiltered = false
        };

        Rasterizer = GaussianRasterizer(RasterSettings_);
    }
    else
    {
        Tcw = Tcw.unsqueeze(0).repeat({mean3D.size(0),1,1}).to(GPU0, torch::kFloat32,true);
        mean3D = torch::cat({mean3D,torch::ones({mean3D.size(0),1}).to(GPU0,torch::kFloat32)},1).unsqueeze(-1);
        mean3D = Tcw.bmm(mean3D).squeeze(-1).index({torch::indexing::Slice(),torch::indexing::Slice(0,3)}).set_requires_grad(true);

    }

    auto mean2D = torch::zeros_like(mean3D).to(GPU0, true).set_requires_grad(true);
    auto opacities = torch::sigmoid(param.logit_opacities).to(GPU0);
    auto norm_qua = torch::nn::functional::normalize(param.unnorm_quat).to(GPU0);
    auto scales = torch::exp(param.log_scales).set_requires_grad(true).to(GPU0);
    rgb = param.rgb.set_requires_grad(true).to(GPU0);

    MaxGPUUseOfMemory = max(PrintCudaMenory(),MaxGPUUseOfMemory);
    mean2D.retain_grad();

    
    auto [rendererd_image, radii, depth] = Rasterizer.forward(
    mean3D,
    mean2D,
    opacities,
    shs,
    rgb,
    scales,
    norm_qua,
    pre_cov3D,
    0);
    torch::cuda::synchronize();

    return { rendererd_image, depth, radii};

    
}


std::tuple<torch::Tensor>
Render::StartSplattingRadii(OptimizerGSParam param)
{
    GaussianRasterizer Rasterizer;
      
    mpCam->SetPose(param.Tcw);

        GaussianRasterizationSettings RasterSettings_ = GaussianRasterizationSettings{
            .image_height = static_cast<int>(mpCam->image_height*1.2),
            .image_width = static_cast<int>(mpCam->image_width*1.2),
            .tanfovx = mpCam->_tanfovx,
            .tanfovy = mpCam->_tanfovy,
            .bg = mpGaussian->_background,
            .scale_modifier = mpGaussian->_scaleModifier,
            .viewmatrix = mpCam->_viewmatrix,
            .projmatrix = mpCam->_full_projmatrix,
            .sh_degree = mpCam->sh_degree,
            .camera_center = mpCam->_campos,
            .prefiltered = false,
            };
    Rasterizer = GaussianRasterizer(RasterSettings_);

    auto shs = torch::Tensor();
    auto pre_cov3D = torch::Tensor();
    auto rgb = torch::Tensor();
    auto mean3D = torch::Tensor();
    auto Tcw = param.Tcw;
    
    mean3D = param.mean3D.detach();

    auto opacities = torch::sigmoid(param.logit_opacities).detach().to(GPU0).detach();
    auto norm_qua = torch::nn::functional::normalize(param.unnorm_quat).detach().to(GPU0).detach();
    auto scales = torch::exp(param.log_scales).detach().to(GPU0).detach();


    auto radii = Rasterizer.Visable(
    mean3D,
    opacities,
    scales,
    norm_qua,
    0);

    torch::cuda::synchronize();

    return {radii};

    
}


void Render::GSParamRGBUpdata(OptimizerGSParam& param,torch::Tensor visual,bool isTracking)
{

    if(isTracking)
    {
    param.mean3D = mpGaussian->GetXYZ(visual).detach();  
    param.rgb = mpGaussian->GetRGB(visual).detach();
    param.unnorm_quat = mpGaussian->GetUnnormQuat(visual).detach();
    param.logit_opacities = mpGaussian->GetLogitOpcity(visual).detach();
    param.log_scales =mpGaussian->GetLogScale(visual).detach();
    }
    else
    {
    param.mean3D = mpGaussian->GetXYZ();  
    param.rgb = mpGaussian->GetRGB();
    param.unnorm_quat = mpGaussian->GetUnnormQuat();
    param.logit_opacities = mpGaussian->GetLogitOpcity();
    param.log_scales =mpGaussian->GetLogScale();
    }

}

void Render::GSParamRGB2Depth(OptimizerGSParam& paramIm,OptimizerGSParam& paramD,bool Tracking)
{
    if(Tracking)
    {
    paramD.unnorm_quat = paramIm.unnorm_quat.detach();
    paramD.logit_opacities = paramIm.logit_opacities.detach();
    paramD.log_scales = paramIm.log_scales.detach();
    paramD.mean3D = paramIm.mean3D.detach(); 
    paramD.rgb = torch::zeros_like(paramIm.rgb).to(GPU0,torch::kFloat32);
    auto Tcw_ = paramIm.Tcw.unsqueeze(0).repeat({paramD.mean3D.size(0),1,1}).to(GPU0,torch::kFloat32);
    auto one_ = torch::ones({paramD.mean3D.size(0),1}).to(GPU0);
    auto mean3D_d = torch::cat({paramD.mean3D,one_},1).unsqueeze(-1).to(GPU0,torch::kFloat32); 
    paramD.rgb = Tcw_.bmm(mean3D_d).squeeze(-1).index({"...",Slice(2,3)}).repeat({1,3}).clone().detach();
    // paramD.rgb.index_put_({"...",0},0);
    paramD.rgb.index_put_({"...",1},1);
    paramD.rgb.index_put_({"...",2},0);
    paramD.Tcw=paramIm.Tcw.clone();
    }
    else
    {
    paramD.unnorm_quat = paramIm.unnorm_quat;
    paramD.logit_opacities = paramIm.logit_opacities;
    paramD.log_scales = paramIm.log_scales;
    paramD.mean3D = paramIm.mean3D; 
            
    auto Tcw_ = paramIm.Tcw.unsqueeze(0).repeat({paramD.mean3D.size(0),1,1}).to(GPU0,torch::kFloat32);
    auto one_ = torch::ones({paramD.mean3D.size(0),1}).to(GPU0);
    auto mean3D_d = torch::cat({paramD.mean3D,one_},1).unsqueeze(-1).to(GPU0,torch::kFloat32); 
    paramD.rgb = Tcw_.bmm(mean3D_d).squeeze(-1).index({"...",Slice(2,3)}).repeat({1,3}).clone();
    paramD.rgb.index_put_({"...",1},1);
    paramD.rgb.index_put_({"...",2},0);
    paramD.Tcw=paramIm.Tcw.clone();
    }
    

}


void Render::GSParamDepthUpdata(OptimizerGSParam& param,torch::Tensor visual,bool isTracking)
{
     
    if(isTracking)
    {
        param.mean3D = mpGaussian->GetXYZ(visual).detach();
        auto Tcw_ = param.Tcw.unsqueeze(0).repeat({param.mean3D.size(0),1,1}).to(GPU0,torch::kFloat32, true);
        auto mean3D_d = torch::cat({param.mean3D,torch::ones({param.mean3D.size(0),1}).to(GPU0,torch::kFloat32)},1).unsqueeze(-1);
        param.rgb = Tcw_.bmm(mean3D_d).squeeze(-1).index({"...",Slice(2,3)}).repeat({1,3}).detach();
        param.rgb.index_put_({"...",1},1);
        param.rgb.index_put_({"...",2},0);

        param.unnorm_quat = mpGaussian->GetUnnormQuat(visual).detach();
        param.logit_opacities = mpGaussian->GetLogitOpcity(visual).detach();
        param.log_scales = mpGaussian->GetLogScale(visual).detach();

    }
    else
    {
        param.unnorm_quat = mpGaussian->GetUnnormQuat(visual);
        param.logit_opacities = mpGaussian->GetLogitOpcity(visual);
        param.log_scales =mpGaussian->GetLogScale(visual);
        param.mean3D = mpGaussian->GetXYZ(visual);         
        auto one_ = torch::ones({param.mean3D.size(0),1}).to(GPU0);
        auto mean3D_d = torch::cat({param.mean3D,one_},1).unsqueeze(-1);
        auto Tcw_ = param.Tcw.unsqueeze(0).repeat({param.mean3D.size(0),1,1});
        param.rgb = Tcw_.bmm(mean3D_d).squeeze(-1).index({"...",Slice(2,3)}).repeat({1,3}).clone();
        param.rgb.index_put_({"...",1},1);
        param.rgb.index_put_({"...",2},0);

    }
}

void Render::GSParamRGBUpdata(OptimizerGSParam& param,bool isTracking)
{

    if(isTracking)
    {
    param.mean3D = mpGaussian->GetXYZ().detach();  
    param.rgb = mpGaussian->GetRGB().detach();
    param.unnorm_quat = mpGaussian->GetUnnormQuat().detach();
    param.logit_opacities = mpGaussian->GetLogitOpcity().detach();
    param.log_scales =mpGaussian->GetLogScale().detach();
    }
    else
    {
    param.mean3D = mpGaussian->GetXYZ();  
    param.rgb = mpGaussian->GetRGB();
    param.unnorm_quat = mpGaussian->GetUnnormQuat();
    param.logit_opacities = mpGaussian->GetLogitOpcity();
    param.log_scales =mpGaussian->GetLogScale();
    }
}


void Render::GSParamDepthUpdata(OptimizerGSParam& param,bool isTracking)
{
     
    if(isTracking)
    {
        param.mean3D = mpGaussian->GetXYZ().detach();
        auto Tcw_ = param.Tcw.unsqueeze(0).repeat({param.mean3D.size(0),1,1}).to(GPU0,torch::kFloat32, true);
        auto mean3D_d = torch::cat({param.mean3D,torch::ones({param.mean3D.size(0),1}).to(GPU0,torch::kFloat32)},1).unsqueeze(-1);
        param.rgb = Tcw_.bmm(mean3D_d).squeeze(-1).index({"...",Slice(2,3)}).repeat({1,3}).detach();
        param.rgb.index_put_({"...",1},1);
        param.rgb.index_put_({"...",2},0);

        param.unnorm_quat = mpGaussian->GetUnnormQuat().detach();
        param.logit_opacities = mpGaussian->GetLogitOpcity().detach();
        param.log_scales = mpGaussian->GetLogScale().detach();
    }
    else
    {
        param.unnorm_quat = mpGaussian->GetUnnormQuat();
        param.logit_opacities = mpGaussian->GetLogitOpcity();
        param.log_scales =mpGaussian->GetLogScale();
        param.mean3D = mpGaussian->GetXYZ();         
        auto one_ = torch::ones({param.mean3D.size(0),1}).to(GPU0);
        auto mean3D_d = torch::cat({param.mean3D,one_},1).unsqueeze(-1);
        auto Tcw_ = param.Tcw.unsqueeze(0).repeat({param.mean3D.size(0),1,1});
        param.rgb = Tcw_.bmm(mean3D_d).squeeze(-1).index({"...",Slice(2,3)}).repeat({1,3}).clone();
        param.rgb.index_put_({"...",1},1);
        param.rgb.index_put_({"...",2},0);
 


    }
}



void Render::RenderStartTraking(Frame& CurrentFrame,int TrackingIterTotal)
{
    OptimizerGSParam paramGS;
    paramGS.Tcw = mpGaussian->InitCameraPose(CurrentFrame.mTcw);

    mpGaussian->CreateOptimizerForPose();
    torch::Tensor bestQuat = mpGaussian->mCamUnnormQuat.clone();
    torch::Tensor bestTrans = mpGaussian->mCamTrans.clone();
    torch::Tensor renderedImage;
    torch::Tensor renderedDepth;
    torch::Tensor renderedRadii;
    torch::Tensor renderedSurdepth;

    cv::Mat im = CurrentFrame.mImRGB.clone();
    cv::Mat depth = CurrentFrame.mImDepth.clone();

    auto[tGtImage,tGtDepth] = ImAndDepth2tensor(im,depth,GPU0);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    torch::Tensor minLoss = torch::tensor({FLT_MAX},torch::kFloat32).to(GPU0);
    torch::Tensor tK = mpGaussian->GetIntrinsic().reshape({3,3}).to(GPU0,torch::kFloat32);
    
    
    if(TrackingIterTotal==0)
        TrackingIterTotal = _TrackingIters;
   
    vector<float> vobs;
    vector<float> vXw4;
    vector<float> vinvSigma2;
    int vilad_match_num = 0;

    for(int i=0; i<CurrentFrame.N; ++i)
    {
        MapPoint* pMP = CurrentFrame.mvpMapPoints[i];
        if(pMP)
        {

            CurrentFrame.mvbOutlier[i]=false;
            const cv::KeyPoint &kpUn = CurrentFrame.mvKeysUn[i];
            vobs.emplace_back(kpUn.pt.x);
            vobs.emplace_back(kpUn.pt.y);
            vobs.emplace_back(1);
            vinvSigma2.emplace_back(CurrentFrame.mvInvLevelSigma2[kpUn.octave]);
            cv::Mat Xw = pMP->GetWorldPos();
            vXw4.emplace_back(Xw.at<float>(0));
            vXw4.emplace_back(Xw.at<float>(1));
            vXw4.emplace_back(Xw.at<float>(2));
            vXw4.emplace_back(1);
            ++vilad_match_num;

        }
    }


    torch::Tensor tobs = torch::from_blob(vobs.data(),{vilad_match_num,3,1},torch::kFloat32).to(GPU0);
    torch::Tensor tXw4 = torch::from_blob(vXw4.data(),{vilad_match_num,4,1},torch::kFloat32).to(GPU0);
    torch::Tensor tinvSigma2 = torch::from_blob(vinvSigma2.data(),{vilad_match_num,1},torch::kFloat32).to(GPU0);
    tinvSigma2 = tinvSigma2.unsqueeze(-1).repeat({1,1,2});
    torch::Tensor diagonal_tinvSigma2 = torch::diag_embed(tinvSigma2);

    auto tbK = tK.unsqueeze(0).repeat({vilad_match_num,1,1});
    torch::Tensor delta = torch::tensor({15},torch::kFloat32).to(GPU0);
    torch::Tensor Lrpj = torch::zeros({1},torch::kFloat32).to(GPU0);
    torch::Tensor ImageL1 = torch::zeros({1},torch::kFloat32).to(GPU0);
    torch::Tensor DepthL1 = torch::zeros({1},torch::kFloat32).to(GPU0);

    torch::Tensor inline_mask = torch::ones({vilad_match_num,1},torch::kBool).to(GPU0);
    int feature_clear = TrackingIterTotal/2.0;

    torch::Tensor lastLoss = torch::zeros({1},torch::kFloat32).to(GPU0);
    for(int iter=0; iter<TrackingIterTotal; ++iter)
    {

        paramGS.Tcw = Rt2T(mpGaussian->mCamUnnormQuat.clone(),mpGaussian->mCamTrans.clone()).to(GPU0);
        auto tbTcw = paramGS.Tcw.unsqueeze(0).repeat({vilad_match_num,1,1});
        auto tXc4 = (tbTcw.bmm(tXw4)).transpose(1,2);
        auto tXc3 = tXc4.index({"...",Slice({None,3})});
        tXc3 = tXc3.div(tXc3.index({"...",0,2}).unsqueeze(-1).unsqueeze(-1).repeat({1,1,3}));
        auto pixel_uv = (tbK.bmm(tXc3.transpose(1,2)));
        auto pixel_error = (pixel_uv - tobs).index({"...",Slice(0,2),0}).unsqueeze(-1).unsqueeze(1);
        auto weightError = pixel_error.transpose(2,3).matmul(diagonal_tinvSigma2);
        weightError = weightError.matmul(pixel_error).squeeze(-1).squeeze(-1);

        
        GSParamRGBUpdata(paramGS,true);
        std::tie(renderedImage, renderedSurdepth, ignore) = StartSplatting(paramGS);
        GSParamDepthUpdata(paramGS,true);
        std::tie(renderedDepth, ignore, ignore) = StartSplatting(paramGS);
        torch::cuda::synchronize();


        auto uncertainDepth = (renderedDepth[1]>0.99&~isnan(tGtDepth));
        
        // ImshowDepth(uncertainDepth,"uncertainDepth");

        // cv::Mat renderedIm = ImshowRGB(renderedImage,"Track");
    
        if(iter == feature_clear)
        {
            inline_mask = weightError < 5.991;
        }
        Lrpj = weightError.masked_select(inline_mask).sum();

        ImageL1 = L1LossForTracking(renderedImage,tGtImage,uncertainDepth.tile({3,1,1}).detach());

        if(_useSurDepth)
            DepthL1 = L1LossForTracking(renderedSurdepth,tGtDepth,uncertainDepth.detach()); 
        else
            DepthL1 = L1LossForTracking(renderedDepth[0],tGtDepth,uncertainDepth.detach()); 


        
        torch::Tensor loss = _imageWeightTracking*ImageL1 + _depthWeightTracking*DepthL1 + _featureWeightTracking*Lrpj;
        loss.backward();
        {
            torch::NoGradGuard no_grad;
            
            if(!std::isnan(loss.item<float>()) && loss.item<float>() < minLoss.item<float>())
            {
                bestQuat = mpGaussian->mCamUnnormQuat.detach().clone();
                bestTrans = mpGaussian->mCamTrans.detach().clone();
                minLoss = loss;
                
            }
            if(abs(lastLoss-loss).item<float>()<10e-4)
                break;

            lastLoss = loss;

            // if(_config["Debug"]["useWandb"].as<bool>())
            // wandbcpp::log({{to_string(CurrentFrame.mnId)+"Loss", loss.item<float>()},{to_string(CurrentFrame.mnId)+"Image", ImageL1.item<float>()}
            // ,{to_string(CurrentFrame.mnId)+"Depth", DepthL1.item<float>()},{to_string(CurrentFrame.mnId)+"Lrpj", Lrpj.item<float>()}});

            mpGaussian->StepUpdataForPose();
            ++Tracking_counts;
            if(PrintCudaMenory() > mTotalGpuByte)
                c10::cuda::CUDACachingAllocator::emptyCache();



        }//no_grad

    }

    auto tT = Rt2T(bestQuat,bestTrans);
    cv::Mat pose(tT.size(0),tT.size(1),CV_32FC1,tT.data_ptr());
    CurrentFrame.SetPose(pose);

    RemoveOutline(CurrentFrame);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    // cout<<"Tracking Time Cost: "<<ttrack<<endl;
    // cout<<"Tracking Avg Time Cost: "<<ttrack/TrackingIterTotal<<endl;
    Tracking_times += ttrack;


}

void Render::RemoveOutline(Frame& CurrentFrame)
{
    float sigma2 = 1.25;
    float chi2Th = 3.99;
    for(int i=0; i<CurrentFrame.N; ++i)
    {
        MapPoint* pMP = CurrentFrame.mvpMapPoints[i];
        if(pMP)
        {
            const cv::KeyPoint &kpUn = CurrentFrame.mvKeysUn[i];
            cv::Mat obs = (cv::Mat_<float>(2,1) << kpUn.pt.x, kpUn.pt.y);
            cv::Mat Xw = pMP->GetWorldPos();
            cv::Mat Xw4 = (cv::Mat_<float>(4,1) << Xw.at<float>(0),Xw.at<float>(1),Xw.at<float>(2),1);
            cv::Mat Xc = CurrentFrame.mTcw*Xw4;
            float invz = 1.0/Xc.at<float>(2);
            float u = _fx*Xc.at<float>(0)*invz + _cx;
            float v = _fy*Xc.at<float>(1)*invz + _cy;

            cv::Mat est = (cv::Mat_<float>(2,1) << u, v);
            cv::Mat info = cv::Mat::eye(2,2,CV_32F)*CurrentFrame.mvInvLevelSigma2[kpUn.octave];
            cv::Mat error = obs-est;
            // float squareError= sqrt((est.at<float>(0)-obs.at<float>(0))*(est.at<float>(0)-obs.at<float>(0))
            // +(est.at<float>(1)-obs.at<float>(1))*(est.at<float>(1)-obs.at<float>(1)));
            float chi2 = error.dot(info*error);
            
            // if(squareError > 4*sigma2)
            if(chi2 > chi2Th)
                CurrentFrame.mvbOutlier[i]=true;
        }
    }
}




} // namespace ORB_SLAM2
