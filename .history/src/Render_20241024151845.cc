
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
#include"eigen3/Eigen/Geometry"
#include"eigen3/Eigen/Dense"
#include"opencv2/core/eigen.hpp"

#include "Render.h"
#include "MaskNet.h"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/types_c.h>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace pybind11::detail;
using namespace torch::indexing;
using namespace cv::xfeatures2d;


static unsigned long MaxGPUUseOfMemory = 0;
static unsigned long sTotle_iters = 0;
static long LastGaussianNum = 0;
static unsigned long Tracking_times=0;
static unsigned long Tracking_counts=0;
static unsigned long Current_iters=0;
static double mTotalGpuByte;

namespace ORB_SLAM2
{
RenderFrame::RenderFrame(const Frame& F)
:Tcw(F.mTcw),mImRGB(F.mImRGB),mImDepth(F.mImDepth),mnId(F.mnId),mImMask(F.mImMask),mvMask(F.mvMask)
{
    // mtMask = torch::from_blob(F.mImMask.data,{F.mImMask.rows,F.mImMask.cols,F.mImMask.channels()},torch::kBool).permute({2,0,1}).to(GPU1);
}

RenderFrame::RenderFrame(KeyFrame* pKF)
:Tcw(pKF->GetPose()),mImRGB(pKF->mImRGB.clone()),mnId(pKF->mnFrameId),mImMask(pKF->mImMask)
{
    if(!pKF->mImDepth.empty()) 
    mImDepth = pKF->mImDepth.clone();
    // mtMask = torch::from_blob(pKF->mImMask.data,{pKF->mImMask.rows,pKF->mImMask.cols,pKF->mImMask.channels()},torch::kBool).permute({2,0,1}).to(GPU1);
}

RenderFrame::RenderFrame(const RenderFrame& F):Tcw(F.Tcw),mImRGB(F.mImRGB),mImDepth(F.mImDepth),mnId(F.mnId),mvBestCovisFrame(F.mvBestCovisFrame),mvSecondCovisFrame(F.mvSecondCovisFrame),mvRandomCondidateFrame(F.mvRandomCondidateFrame),mvRandomBestCovisFrame(F.mvRandomBestCovisFrame),mImMask(F.mImMask),mtMask(F.mtMask)
{}


Render::Render(const string &strSettingsFile,const bool bMonocular):
_shutdown_flag(false),_use_reset_opcities(false),_use_gaussian_splatting(false),
mMax_z(FLT_MIN),_path(strSettingsFile),mbisInited(false),mbRenderFinish(false),mbFinishRequested(false),mbFinished(false),_GPU_device(GPU0)
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
     image_w = _config["Camera"]["width"].as<float>();
     image_h = _config["Camera"]["height"].as<float>();
    mTotalPixelCount = image_w*image_h;
        fx = _config["Camera"]["fx"].as<float>();
        fy = _config["Camera"]["fy"].as<float>();
        cx = _config["Camera"]["cx"].as<float>();
        cy = _config["Camera"]["cy"].as<float>();
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    
    mSceneRaduisDepthRatio = _config["Scene"]["RaduisDepthRatio"].as<float>();
    _mapping_iters = _config["Mapping"]["num_iters"].as<int>();

    _depth_loss_weight = _config["Mapping"]["depth_loss_weight"].as<float>();
    _image_loss_weight = _config["Mapping"]["im_loss_weight"].as<float>();
    _lambda = _config["Mapping"]["lambda"].as<float>();

    _iter_print_count = _config["Debug"]["iter_print_count"].as<int>();

    _use_reset_opcities = _config["Mapping"]["use_reset_opcities"].as<bool>();
    _use_gaussian_splatting = _config["Mapping"]["use_gaussian_splatting"].as<bool>();

    _reset_opcities_every = _config["Mapping"]["reset_opcities_every"].as<int>();
    _prune_every = _config["Mapping"]["prune_every"].as<int>();
    _use_metric = _config["Debug"]["use_metric"].as<bool>();

    int gpu_id = _config["GPUdeivce"].as<int>();
    if(gpu_id==0)
        _GPU_device = GPU0;
    else if(gpu_id==1)
        _GPU_device = GPU1;
    else
        _GPU_device = GPU2;

    cudaError_t status;
    cudaError_t ct = cudaSetDevice(gpu_id);
    if (status != cudaSuccess) {
        printf("cudaSetDevice failed!\n");
        printf("%s",ct);
    }

    
    mpGaussian = make_shared<Gaussian>(strSettingsFile);
    mpCam = make_shared<Camera>(image_w, image_h, K);



    cout<<"-----------Mapping Parameters-------------------\n";
    cout<<"- Scene.RaduisDepthRatio: "<<mSceneRaduisDepthRatio<<endl;
    cout<<"- Mapping.num_iters: "<<_mapping_iters<<endl;
    cout<<"- Mapping.depth_loss_weight: "<<_depth_loss_weight<<endl;
    cout<<"- Mapping.im_loss_weight: "<<_image_loss_weight<<endl;
    cout<<"- Mapping.lambda: "<<_lambda<<endl;
    cout<<"- Mapping.feature_weight: "<<_config["Tracking"]["feature_weight"].as<float>()<<endl;


    
    _thread_render = make_unique<thread>(bind(&Render::Run,this));

    GaussianRasterizationSettings RasterSettings_ = GaussianRasterizationSettings{
    .image_height = static_cast<int>(mpCam->image_height),
    .image_width = static_cast<int>(mpCam->image_width),
    .tanfovx = mpCam->_tanfovx,
    .tanfovy = mpCam->_tanfovy,
    .bg = mpGaussian->_background,
    .scale_modifier = mpGaussian->_scale_modifier,
    .viewmatrix = mpCam->_viewmatrix,
    .projmatrix = mpCam->_full_projmatrix,
    .sh_degree = mpCam->sh_degree,
    .camera_center = mpCam->_campos,
    .prefiltered = false
    };

    mRasterizer = GaussianRasterizer(RasterSettings_);


    if(_config["Debug"]["use_wandb"].as<bool>())
        wandbcpp::init({.project = "ORBGaussianMap", .tags = {"basic"}, .group = "loss"});
     
}

void Render::SetMaskNet(MaskNet* pMaskNet_)
{
    mpNet = pMaskNet_;
}


void Render::Run()
{
    static unsigned long last_N = 0;
    
    while(1)
    {
        
       if(mMutexTrackingIdle.try_lock())
        {
            if(!CheckNewKeyFrame())
            {
                auto start_time = std::chrono::steady_clock::now();
                // ProcessNewKeyFrame();

                if(!mbisInited)
                {
                    ProcessNewKeyFrame();
                    InitWorld();
                    mbisInited=true;
                }
                else
                {
                    // MaintainMap();
                }

                c10::cuda::CUDACachingAllocator::emptyCache();
                
                auto end_time = std::chrono::steady_clock::now();
                std::chrono::duration<double> time_elapsed = end_time - start_time;
                _vec_avg_time.emplace_back(time_elapsed.count());
            }
            

            mMutexTrackingIdle.unlock();
        }
      
        if(CheckFinish())
        if(mMutexTrackingIdle.try_lock())
        {
            {
                if(_config["Eval"]["SavePly"].as<bool>())
                {
                    cout<<"MaxGPUUseOfMemory: "<<MaxGPUUseOfMemory<<endl;
                    cout<<"Total Gaussian: "<<mpGaussian->_mean3D.size(0)<<endl;
                    SavePly(mpGaussian,_config["Dataset"]["name"].as<std::string>());
                    cout<<"Avg Tracking Times: "<<Tracking_times/Tracking_counts<<" ms"<<endl;


                }
                mMutexTrackingIdle.unlock();

                break;
            }
            
        }

    }//while(1)
    if(_config["Debug"]["use_wandb"].as<bool>())
         wandbcpp::finish();
    SetFinish();

   
}



void Render::MaintainMap(bool requestFinish)
{

    if(requestFinish && mqNewRenderFrame.empty())
    {
        mbRenderFinish=true;
        return;
    }

    if(mqNewRenderFrame.empty())
        return;

    RenderFrame CurrentRF = mqNewRenderFrame.front();
    mqNewRenderFrame.pop_front();



    cv::Mat im = CurrentRF.mImRGB;
    cv::Mat depth = CurrentRF.mImDepth;
    OptimizerGSParam paramGS;

    im.convertTo(im,CV_32FC3,1.0 / 255.0);

    paramGS.Tcw = torch::from_blob(CurrentRF.GetPose().data,{4,4},torch::kFloat32).to(_GPU_device);

    torch::Tensor rendered_image;
    torch::Tensor rendered_depth;
    torch::Tensor rendered_radii;

    
    vector<RenderFrame> CondidateRenderFrame;
    CondidateRenderFrame.emplace_back(CurrentRF);
    Merge(CondidateRenderFrame,CurrentRF.mvBestCovisFrame);
    // Merge(CondidateRenderFrame,CurrentRF.mvSecondCovisFrame);
    Merge(CondidateRenderFrame,CurrentRF.mvRandomCondidateFrame);
    Merge(CondidateRenderFrame,CurrentRF.mvRandomBestCovisFrame);

    int N_iters = CondidateRenderFrame.size();


    for(int iter=0; iter<N_iters; ++iter)
    {
        // int k = mRng.uniform(0,N_iters);
        RenderFrame pRF_ = CondidateRenderFrame[iter];

        cv::Mat depth = pRF_.mImDepth;
        cv::Mat im = pRF_.mImRGB;

        auto[tGtImage,tGtDepth] = ImAndDepth2tensor(im,depth,_GPU_device);

        paramGS.Tcw = torch::from_blob(pRF_.GetPose().data,{4,4},torch::kFloat32).to(_GPU_device);

        GSParamDepthUpdata(paramGS);
        std::tie(rendered_depth, ignore, ignore) = Start(paramGS);
        GSParamRGBUpdata(paramGS);
        std::tie(rendered_image, ignore, rendered_radii) = Start(paramGS);

        auto tVaildMask = (tGtDepth>0).to(_GPU_device);

         auto image_loss = _lambda * L1LossForMapping(rendered_image,tGtImage,tVaildMask.tile({3,1,1}).detach()) + (1-_lambda) * (1.0 - SSIM(rendered_image,tGtImage,_GPU_device));

        auto depth_loss = L1LossForMapping(rendered_depth[0],tGtDepth,tVaildMask.detach());
        auto loss = _image_loss_weight*image_loss + _depth_loss_weight*depth_loss ;

        loss.backward();
        
        {
            torch::NoGradGuard no_grad;
            mpGaussian->StepUpdataForGaussian();
            // ++sTotle_iters;
            // ImshowRGB(rendered_image,"Render");
        }//no_grad
    }

    {
        torch::NoGradGuard no_grad;
        UpdataMaxZ();
        RemoveGaussian();
    }

    
   if(_use_metric)
    {
        // cout<<"Total Gaussian: "<<mpGaussian->_mean3D.size(0)<<"\tChange num: "<<mpGaussian->_mean3D.size(0)-LastGaussianNum<<"\tTotal Iters: "<< sTotle_iters<<endl;
        LastGaussianNum = mpGaussian->_mean3D.size(0);
        // MaxGPUUseOfMemory = max(PrintCudaMenory(),MaxGPUUseOfMemory);
        // wandbcpp::log({{"loss", avg_loss/(N_iters*2)},{"Gaussian Num:",mpGaussian->_mean3D.size(0)}});
        
    }
   


}



void Render::AddGaussianForFrame(Frame& Frame)
{
    unique_lock<mutex> lock(mMutexTrackingIdle);

    RenderFrame pRF(Frame);

    OptimizerGSParam paramGS;

    cv::Mat im = pRF.mImRGB;
    cv::Mat depth = pRF.mImDepth;
    paramGS.Tcw = torch::from_blob(pRF.GetPose().data,{4,4},torch::kFloat32).to(_GPU_device);
    paramGS.cTcw = pRF.GetPose().clone();
    GSParamDepthUpdata(paramGS);
    auto[rendered_depth, _1, _0]  = Start(paramGS);

    GSParamRGBUpdata(paramGS);
    auto[rendered_image, render_mdepth, radii] = Start(paramGS);

    // auto[tGtImage,tGtDepth] = ImAndDepth2tensor(im,depth,_GPU_device);
    // cv::Mat diff = ImshowDepth(abs(render_mdepth - tGtDepth));
    // ImshowDepth(tGtDepth>0,"zeros");
    // cv::Mat colorDifference;
    //  cv::applyColorMap(diff, colorDifference, cv::COLORMAP_JET);
    //  imshow("colorDifference",colorDifference);
    // waitKey(0);

    //  GSParamRGBUpdata(paramGS);
    // auto[radii] = Start_Radii(paramGS);
 
    // auto visible_index = (radii>0).nonzero().squeeze().to(torch::kLong);

    // mTrackingParams.mean3D = mpGaussian->_mean3D.index_select(0, visible_index).clone();
    // mTrackingParams.rgb = mpGaussian->_rgb.index_select(0, visible_index).clone();
    // mTrackingParams.unnorm_qua = mpGaussian->_unnorm_qua.index_select(0, visible_index).clone();
    // mTrackingParams.logit_opacities = mpGaussian->_logit_opacities.index_select(0, visible_index).clone();
    // mTrackingParams.log_scales = mpGaussian->_log_scales.index_select(0, visible_index).clone();
 

    if(Frame.mnId % 50 ==0)
    {
        RemoveGaussian();
        UpdataMaxZ(); 
    }


    AddGaussian(pRF,rendered_image,rendered_depth); 


    KeyFrame* pRefKF = Frame.mpReferenceKF;
    set<int> sReaptId;
    set<int> sReaptTrackingId;

    set<KeyFrame*> spVisReaptId;

    cv::Mat RFTwc = cv::Mat::eye(4,4,CV_32F);
    pRefKF->GetPoseInverse().rowRange(0,3).colRange(0,3).copyTo(RFTwc.rowRange(0,3).colRange(0,3));
    pRefKF->GetCameraCenter().copyTo(RFTwc.rowRange(0,3).col(3));
    cv::Mat RFTcw = RFTwc.inv();
    vector<KeyFrame*> vpKeyFrameDataset = mpMap->GetAllKeyFrames();
    torch::Tensor tK_ = torch::tensor({fx,(float)0.0,cx,(float)0.0,fy,cy,(float)0.0,(float)0.0,(float)1.0},torch::kFloat32).reshape({3,3}).to(_GPU_device);
    int RandomNum = pRefKF->mvReferentRandomPoints.size()/4;
    auto tbK_ = tK_.unsqueeze(0).repeat({RandomNum,1,1});
    torch::Tensor tXw4 = torch::from_blob(pRefKF->mvReferentRandomPoints.data(),{RandomNum,4,1},torch::kFloat32).to(_GPU_device);

    int KF_dataset_num = vpKeyFrameDataset.size();
    int nn = 9;
    int n = 11;
    vector<KeyFrame*> vpNeighKFs = pRefKF->GetVectorCovisibleKeyFrames();
    int edge = 20;
    auto edge_h = (image_w-20)*torch::ones({RandomNum,1},torch::kFloat32).to(_GPU_device);
    auto edge_w = (image_h-20)*torch::ones({RandomNum,1},torch::kFloat32).to(_GPU_device);
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
        torch::Tensor tVelocity = CvMat2Tensor(Velocity,_GPU_device);
        auto tbTcw = tVelocity.repeat({RandomNum,1,1});
        auto tXc4 = (tbTcw.bmm(tXw4)).transpose(1,2);
        auto tXc3 = tXc4.index({"...",Slice({None,3})});
        tXc3 = tXc3.div((tXc3.index({"...",0,2})+1e-6).unsqueeze(-1).unsqueeze(-1).repeat({1,1,3}));
        auto pixel_uv = (tbK_.bmm(tXc3.transpose(1,2)));

        auto inside = (pixel_uv.index({"...",0,0})>edge
        &pixel_uv.index({"...",0,0})<image_w-edge
        &pixel_uv.index({"...",1,0})>edge
        &pixel_uv.index({"...",1,0})<image_h-edge);

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
            torch::Tensor tVelocity = CvMat2Tensor(Velocity,_GPU_device);

            auto tbTcw = tVelocity.repeat({RandomNum,1,1});
            auto tXc4 = (tbTcw.bmm(tXw4)).transpose(1,2);
            auto tXc3 = tXc4.index({"...",Slice({None,3})});
            tXc3 = tXc3.div((tXc3.index({"...",0,2})+1e-6).unsqueeze(-1).unsqueeze(-1).repeat({1,1,3}));

            auto pixel_uv = (tbK_.bmm(tXc3.transpose(1,2)));

            auto inside = (pixel_uv.index({"...",0,0})>edge
            &pixel_uv.index({"...",0,0})<image_w-edge
            &pixel_uv.index({"...",1,0})>edge
            &pixel_uv.index({"...",1,0})<image_h-edge);

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
//    for(auto r:sReaptId)
//     cout<<r<<" ";
    
    // if(vpKeyFrameDataset.size() > 40)
    // {
    //     sort(vpKeyFrameDataset.begin(),vpKeyFrameDataset.end(),[](KeyFrame* k1,KeyFrame* k2){return k1->mRenderedNum < k2->mRenderedNum;});
    //     for(auto pKF:vpKeyFrameDataset)
    //     {

    //         if(sReaptId.count(pKF->mnId)<1)
    //         {
    //             if(pKF->isBad()) continue;
    //             pRF.mvRandomBestCovisFrame.emplace_back(RenderFrame(pKF));
    //             sReaptId.insert(pKF->mnId);
    //             spVisReaptId.insert(pKF);
    //             pKF->mRenderedNum++;
    //         }
    //         if(pRF.mvRandomBestCovisFrame.size()>5) break;
    //     }

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
            // if(vpKeyFrameDataset.size() > 50)
            // {
            //     vector<KeyFrame*> vpNeighKFs_ = pKF_->GetBestCovisibilityKeyFrames(3);
            //     if(sReaptId.count(vpNeighKFs_.back()->mnId)>0) continue;
            //     pRF.mvRandomBestCovisFrame.emplace_back(vpNeighKFs_.back());
            //     sReaptId.insert(vpNeighKFs_.back()->mnId);
            // }
        }
        --KF_dataset_num;
        
    }
    // for(auto r:sReaptId)
    // cout<<r<<" ";
    // cout<<endl;

    mpMap->UpdateRenderFrame(spVisReaptId);

//    for(int i=0;i<vpNeighKFs.size();++i)
//     {
//         KeyFrame* pKF_ = vpNeighKFs[i];
        
//         if(pKF_->isBad() || sReaptId.count(pKF_->mnId)<1) continue;
//         RenderFrame pRFBest(pKF_);
//         pRF.mvBestCovisFrame.emplace_back(pRFBest);
//         sReaptId.insert(pKF_->mnFrameId);
//         spVisReaptId.insert(pKF_);

//         if(pRF.mvBestCovisFrame.size()>=23) break;

//     }
//     for(int i=0; i<7;++i)
//     {
//         int k = mRng.uniform(0,vpKeyFrameDataset.size());
//         KeyFrame* pKF_ = vpKeyFrameDataset[k];
//         if(pKF_->mnFrameId > pRF.mnId) continue;
//         if(sReaptId.count(pKF_->mnFrameId)<1)
//         {
//             if(pKF_->isBad()) continue;
//             pRF.mvRandomCondidateFrame.emplace_back(RenderFrame(pKF_));
//             sReaptId.insert(pKF_->mnFrameId);
//             spVisReaptId.insert(pKF_);
//         }
//     }

    RenderForFrame(pRF);

}


void Render::RenderForFrame(RenderFrame& CurrentRF)
{

    vector<RenderFrame> CondidateRenderFrame;
    CondidateRenderFrame.emplace_back(CurrentRF);
    Merge(CondidateRenderFrame,CurrentRF.mvBestCovisFrame);
    Merge(CondidateRenderFrame,CurrentRF.mvSecondCovisFrame);
    Merge(CondidateRenderFrame,CurrentRF.mvRandomCondidateFrame);
    Merge(CondidateRenderFrame,CurrentRF.mvRandomBestCovisFrame);

    int N_iters = CondidateRenderFrame.size();
    torch::Tensor rendered_image;
    torch::Tensor rendered_depth;
    torch::Tensor rendered_surdepth;

    // torch::Tensor to_remove = (torch::sigmoid(mTrackingParams.logit_opacities) < 0.005).squeeze(-1);
    // auto valid_point_mask = ~to_remove;
    // int true_count = valid_point_mask.sum().item<int>();
    // cout<<"Local Remove: "<<true_count<<endl;
    // auto indices = torch::nonzero(valid_point_mask == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);
    // mTrackingParams.mean3D = mTrackingParams.mean3D.index_select(0, indices);
    // mTrackingParams.rgb = mTrackingParams.rgb.index_select(0, indices);
    // mTrackingParams.log_scales = mTrackingParams.log_scales.index_select(0, indices);
    // mTrackingParams.logit_opacities = mTrackingParams.logit_opacities.index_select(0, indices);
    // mTrackingParams.unnorm_qua = mTrackingParams.unnorm_qua.index_select(0, indices);
    // mpGaussian->CreateOptimizerForLocalGaussian(mTrackingParams);

    // for(int iter=0; iter<30; ++iter)
    // {

    //     cv::Mat im = CurrentRF.mImRGB;
    //     cv::Mat depth = CurrentRF.mImDepth;
        
    //     auto[tGtImage,tGtDepth] = ImAndDepth2tensor(im,depth,_GPU_device);

    //     auto tVaildMask = (tGtDepth>0).to(_GPU_device);
    //     OptimizerGSParam paramD;

    //     GSParamRGB2Depth(mTrackingParams,paramD);
        
    //     auto[rendered_depth, _3, _4] = Start(paramD);
    //     auto[rendered_image, _6, _5] = Start(mTrackingParams);

    //     auto image_loss = _lambda * L1LossForMapping(rendered_image,tGtImage) + (1-_lambda) * (1.0 - SSIM(rendered_image,tGtImage,_GPU_device));

    //     auto depth_loss = L1LossForMapping(rendered_depth[0],tGtDepth,tVaildMask.detach());

    //     auto loss = _image_loss_weight*image_loss + _depth_loss_weight*depth_loss;

    //     loss.backward();
    //     {
    //         torch::NoGradGuard no_grad;

    //         ImshowRGB(rendered_image,"Render");

    //         mpGaussian->StepUpdataForLocalGaussian();

    //     }

    // }
    auto max_scalar = 0.1*mpGaussian->_scene_radius;

    torch::Tensor loss;
    for(int iter=0; iter<_mapping_iters; ++iter)
    {
        OptimizerGSParam paramGS;
        int k = mRng.uniform(0,N_iters);
        RenderFrame pRF_  = CondidateRenderFrame[k];
        // if(pRF_.mnId==0) continue;
        cv::Mat depth = pRF_.mImDepth;
        cv::Mat im = pRF_.mImRGB;

        auto[tGtImage,tGtDepth] = ImAndDepth2tensor(im,depth,_GPU_device);

        paramGS.Tcw = torch::from_blob(pRF_.GetPose().data,{4,4},torch::kFloat32).to(_GPU_device);
        //GSParamRGBUpdata(paramGS);
        //auto[radii] = Start_Radii(paramGS);
        //auto visible_index = (radii>0).nonzero().squeeze().to(torch::kLong);
        GSParamDepthUpdata(paramGS);
        std::tie(rendered_depth, ignore, ignore) = Start(paramGS);
        GSParamRGBUpdata(paramGS);
        std::tie(rendered_image, rendered_surdepth, ignore) = Start(paramGS);

        auto tvaildMask = (tGtDepth>0).to(_GPU_device);
        auto tvaildMasksur = (tGtDepth>0&rendered_depth[1]>0.99).to(_GPU_device);


        auto image_loss = _lambda * L1LossForMapping(rendered_image,tGtImage) + (1-_lambda) * (1.0 - SSIM(rendered_image,tGtImage,_GPU_device));
        auto depth_loss = L1LossForMapping(rendered_depth[0],tGtDepth,tvaildMask.detach());

        auto surdepth_loss = L1LossForMapping(rendered_surdepth,tGtDepth,tvaildMasksur.detach());
 
        auto big_scalar_mask = where(torch::exp(paramGS.log_scales)>max_scalar)[0];
        auto reg_scalar = (get<0>(torch::exp(paramGS.log_scales.index_select(0,big_scalar_mask)).max(1)) - max_scalar).sum();

        auto max_scale  = get<0>(torch::exp(paramGS.log_scales.index_select(0,big_scalar_mask)).max(1));
        auto min_scale = get<0>(torch::exp(paramGS.log_scales.index_select(0,big_scalar_mask)).min(1));
        auto reg_long = (max_scale - min_scale).mean();
        // cout<<max_scalar<<endl;
        // if(CurrentRF.mnId>4)
        // loss = 1.0*image_loss + 0.7*depth_loss + 0.1*surdepth_loss;

        loss = 1.0*image_loss + 0.7*depth_loss + 0.1*surdepth_loss + 10*reg_scalar + 5*reg_long;
        // loss = 1.0*image_loss + 0.7*depth_loss + 10*reg_scalar + 5*reg_long;

    
        // else
        // loss = 1.0*image_loss + 0.7*depth_loss + 0.1*surdepth_loss +0.01*reg_scalar;

        // else
         //loss = _image_loss_weight*image_loss + _depth_loss_weight*depth_loss + reg_scalar;


        loss.backward();
        
        {
            torch::NoGradGuard no_grad;
            mpGaussian->StepUpdataForGaussian();
            // mpGaussian->_mean3D.index_put_({visible_index},paramGS.mean3D);
            // mpGaussian->_rgb.index_put_({visible_index},paramGS.rgb);
            // mpGaussian->_unnorm_qua.index_put_({visible_index},paramGS.unnorm_qua);
            // mpGaussian->_logit_opacities.index_put_({visible_index},paramGS.logit_opacities);
            // mpGaussian->_log_scales.index_put_({visible_index},paramGS.log_scales);
            // ++sTotle_iters;
            ImshowRGB(rendered_image,"Render");
            if(_config["Debug"]["use_wandb"].as<bool>())
            wandbcpp::log({{"Loss", loss.item<float>()},{"Image", image_loss.item<float>()}
            ,{"Depth", depth_loss.item<float>()},{"sur", surdepth_loss.item<float>()}});

            if(PrintCudaMenory() > mTotalGpuByte)
                c10::cuda::CUDACachingAllocator::emptyCache();

        }//no_grad
    }
     
                // c10::cuda::CUDACachingAllocator::emptyCache();

   if(_use_metric)
    {
        // cout<<"RenderForFrame ->> Total Gaussian: "<<mpGaussian->_mean3D.size(0)<<"\tChange num: "<<mpGaussian->_mean3D.size(0)-LastGaussianNum<<"\tTotal Iters: "<< sTotle_iters<<endl;
        LastGaussianNum = mpGaussian->_mean3D.size(0);
        // MaxGPUUseOfMemory = max(PrintCudaMenory(),MaxGPUUseOfMemory);
        // wandbcpp::log({{"loss", avg_loss/(N_iters*2)},{"Gaussian Num:",mpGaussian->_mean3D.size(0)}});
        // cout<<PrintCudaMenory()<<","<<mpGaussian->_mean3D.size(0)<<endl;

    }
   


}


void Render::InitWorld()
{  
   
    RenderFrame pCurrentRf = mqNewRenderFrame.front();
    mqNewRenderFrame.pop_front();
    // cout<<"Gaussian Init\n";

    InitGaussianPoint(pCurrentRf);

    mpGaussian->CreateOptimizerForGaussian();

    cv::Mat im = pCurrentRf.mImRGB;
    cv::Mat depth = pCurrentRf.mImDepth;
    OptimizerGSParam paramGS;
    
    paramGS.Tcw = torch::from_blob(pCurrentRf.GetPose().data,{4,4},torch::kFloat32).to(_GPU_device).clone();
    paramGS.cTcw = pCurrentRf.GetPose().clone();
    auto[tGtImage,tGtDepth] = ImAndDepth2tensor(im,depth,_GPU_device);

    auto tVailMask = (tGtDepth>0).to(_GPU_device);
 
    torch::Tensor rendered_image;
    torch::Tensor rendered_depth;
    torch::Tensor rendered_surdepth;


    for(int iter=0; iter<200; ++iter)
    {

        GSParamRGBUpdata(paramGS);
        std::tie(rendered_image, rendered_surdepth, ignore) = Start(paramGS);

        GSParamDepthUpdata(paramGS);
        std::tie(rendered_depth, ignore, ignore) = Start(paramGS);

        auto image_loss = _lambda * L1LossForMapping(rendered_image,tGtImage) + (1-_lambda) * (1.0 - SSIM(rendered_image,tGtImage,_GPU_device));

        auto depth_loss = L1LossForMapping(rendered_depth[0],tGtDepth,tVailMask.detach());

        auto surdepth_loss = L1LossForMapping(rendered_surdepth,tGtDepth,tVailMask.detach());

        auto loss = _image_loss_weight*image_loss + 0.1*surdepth_loss + _depth_loss_weight*depth_loss;


        loss.backward();
        
        {
            torch::NoGradGuard no_grad;
            mpGaussian->StepUpdataForGaussian();
            ImshowRGB(rendered_image,"Render");

            ++sTotle_iters;

        }//no_grad

    }
    // GSParamRGBUpdata(mTrackingParams);
    


}



void Render::AddGaussian(RenderFrame& pRf, torch::Tensor& rendered_im, torch::Tensor& rendered_depth)
{
    
    auto gray = (rendered_im[0]*299+rendered_im[1]*587+rendered_im[2]*114)/1000;
    auto tBlackMask = (gray < 50/255.0);

    cv::Mat depth = pRf.mImDepth.clone();
    auto tGtDepth = CvMat2Tensor(depth,_GPU_device);
  
    auto tDiffDepth = abs(tGtDepth - rendered_depth[0]);

    auto tDiffDepthMask = torch::logical_and(tDiffDepth<0.05,tGtDepth>0);
    tDiffDepthMask = tDiffDepthMask & (rendered_depth[0]>0);

    auto sum_val = tDiffDepth.masked_select(tDiffDepthMask).sum();
    auto avg_val = sum_val/tDiffDepthMask.sum();
    
    auto medie_val = tDiffDepth.masked_select(tDiffDepthMask).median();
    auto th = avg_val.item<float>()+_config["Mapping"]["madie_mul"].as<float>()*medie_val.item<float>();
    
    if(th<0.01) th=0.01;
    // cout<<"  Add th:"<<th<<endl;


    // // auto tAddMask = ((tDiffDepth > th) & tBlackMask) | (rendered_depth[1]<0.8);
    // auto finish_edge = (~(rendered_depth[1]>0.99) & tBlackMask);
    // auto tAddMask = ((tDiffDepth > th) & tBlackMask) | (rendered_depth[1]<0.8);
    // tAddMask = (tAddMask & rendered_depth[1]>0.99) | finish_edge;
    auto c1 = ~(rendered_depth[1]>0.99) & tBlackMask & tDiffDepth > th;
    auto c2 = (rendered_depth[1]<0.8);
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
void Render::ProjectPixel(RenderFrame& pRF,PointCloud& PtCloud,cv::Mat cvMask)
 {
    const cv::Mat depth = pRF.mImDepth;
    cv::Mat im = pRF.mImRGB;
    cv::Mat Twc = pRF.GetPose().inv();
    cv::Mat split_im[3];
    im.convertTo(im,CV_32FC3,1.0/255.0);
    split(im,split_im);
    // imshow("add_mask",cvMask);

    for(int i=0; i<im.rows; ++i)
        for(int j=0; j<im.cols; ++j)
        {
            if(cvMask.at<uchar>(i,j) < 250) continue;
            float z = depth.at<float>(i,j);
            float x=0;
            float y=0;
            if(z > 0 )
            {
                x = (j - cx)*z / fx;
                y = (i - cy)*z / fy;
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
               
                mMax_z = max(mMax_z,z);
                PtCloud.Points.emplace_back(pt);
                PtCloud.Colors.emplace_back(rgb);

            }
        }
 }
 


void Render::ProcessNewKeyFrame()
{
    deque<KeyFrame*> vdKeyFrame;
    {
        unique_lock<mutex> lck(mMutexKeyFrameUpdata);
    
        while(mdpKeyFrame.front()->isBad())
        {
            mdpKeyFrame.pop_front();
        } 
        
        vdKeyFrame.push_back(mdpKeyFrame.front());      
        mdpKeyFrame.pop_front();
    }
    vector<KeyFrame*> vpKeyFrameDataset = mpMap->GetAllKeyFrames();
    int KF_dataset_num = vpKeyFrameDataset.size();
    int nn = 5;
    int n = 25;
    set<int> sReaptId;

    for(auto pKF:vdKeyFrame)
    {
        if(pKF->isBad()) continue;
        vector<KeyFrame*> vpNeighKFs;
        vpNeighKFs.emplace_back(pKF);

        RenderFrame pRF(pKF);
        
        vpNeighKFs = pKF->GetVectorCovisibleKeyFrames();

        
        for(int i=0;i<vpNeighKFs.size();++i)
        {
            KeyFrame* pKF_ = vpNeighKFs[i];
            if(pKF_->isBad()) continue;
            if(sReaptId.count(pKF_->mnId)<1)
            {
                RenderFrame pRFBest(pKF_);
                pRF.mvBestCovisFrame.emplace_back(pRFBest);
                sReaptId.insert(pRFBest.mnId);
                
                
            }

            if(pRF.mvBestCovisFrame.size()>n-1) break;

        }
        for(auto r:sReaptId)
        cout<<r<<" ";
        cout<<endl;

        if(pRF.mvBestCovisFrame.size()<n)
        {
            if(pKF->isBad()) continue;
            vector<KeyFrame*> vpNeighKFs;
            KeyFrame* pKF=nullptr;
            int x=1;
            while(vpKeyFrameDataset.size()>x)
            {
                pKF = vpKeyFrameDataset[vpKeyFrameDataset.size()-x];
                ++x;
                if(pKF->isBad() || pKF->mnId > pRF.mnId)
                {
                    continue;
                } 
                
                RenderFrame RF(pKF);
                if(sReaptId.count(RF.mnId)<1)
                {
                    pRF.mvBestCovisFrame.emplace_back(RF);
                    sReaptId.insert(RF.mnId);                   
                }
                vpNeighKFs = pKF->GetBestCovisibilityKeyFrames(5);
                for(int i=0;i<vpNeighKFs.size();++i)
                {
                    KeyFrame* pKF_ = vpNeighKFs[i];
                    if(pKF_->isBad()) continue;
                    if(sReaptId.count(pKF_->mnId)<1)
                    {
                        RenderFrame pRFBest(pKF_);
                        pRF.mvBestCovisFrame.emplace_back(pRFBest);
                        sReaptId.insert(pRFBest.mnId);                   
                    }
                    if(pRF.mvBestCovisFrame.size()>n) break;
                }
                if(pRF.mvBestCovisFrame.size()>n) break;             
                
            }
           
        }


        while(KF_dataset_num!=0 && pRF.mvRandomCondidateFrame.size()+pRF.mvBestCovisFrame.size()<n+nn)
        {
            int k = mRng.uniform(0,KF_dataset_num);
            KeyFrame* pKF_ = vpKeyFrameDataset[k];
            if(pKF_->isBad() || pKF_->mnId > pRF.mnId) continue;
            if(sReaptId.count(pKF_->mnId)<1)
            {
                pRF.mvRandomCondidateFrame.emplace_back(RenderFrame(pKF_));
                sReaptId.insert(pKF_->mnId);
                if(vpKeyFrameDataset.size() > 50)
                {
                    vector<KeyFrame*> vpNeighKFs_ = pKF_->GetBestCovisibilityKeyFrames(3);
                    if(sReaptId.count(vpNeighKFs_.back()->mnId)>0) continue;
                    pRF.mvRandomBestCovisFrame.emplace_back(vpNeighKFs_.back());
                    sReaptId.insert(vpNeighKFs_.back()->mnId);
                }
            }
            --KF_dataset_num;
            
        }
        for(auto r:sReaptId)
        cout<<r<<" ";
        cout<<endl;

        
        mqNewRenderFrame.push_back(pRF);
    }


}


//using gaussian splatting
void Render::GaussianSplatting(int iters)
{

    mpGaussian->DensifyInfo();
    mpGaussian->CloneAndSplit(iters);
    // if(_use_reset_opcities && iters > mpGaussian->_reset_opcities_iter&&iters%15==0)
    // {
    //     cout<<"Reset Opacity\n";
    //     mpGaussian->Reset_opacity();
    // }
}

void Render::UpdataMaxZ()
{
    mpGaussian->_scene_radius = mMax_z/mSceneRaduisDepthRatio;
}

void Render::Shutdown()
{
    {
        unique_lock<mutex> lck(mMutexKeyFrameUpdata);
        unique_lock<mutex> lck1(mMutexShutdown);
        // _update_keyframe.notify_one();
        _shutdown_flag = true;
    }

    _thread_render->join();
}

bool Render::CheckNewKeyFrame()
{
    {
        unique_lock<mutex> lck(mMutexKeyFrameUpdata);
        return mdpKeyFrame.empty();
    }
}

void Render::InsertKeyFrame(KeyFrame* pKf)
{
    {
    unique_lock<mutex> lck(mMutexKeyFrameUpdata);
        mdpKeyFrame.push_back(pKf);
    }
}

void Render::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Render::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Render::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
}

bool Render::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
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
                float x = (j - cx)*z / fx;
                float y = (i - cy)*z / fy;
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

                mMax_z = max(mMax_z,z);
            }
        }
    mpGaussian->AddGaussianPoints(pt_cloud);
    mpCam->_cam_T = pRf.GetPose();
    mpGaussian->_scene_radius = mMax_z/mSceneRaduisDepthRatio;

}

std::tuple<torch::Tensor, torch::Tensor,torch::Tensor,torch::Tensor, torch::Tensor,torch::Tensor>
Render::Start_Grad(OptimizerGSParam param,bool eval)
{
    auto shs = torch::Tensor();
    auto pre_cov3D = torch::Tensor();
    auto rgb = torch::Tensor();
    auto mean3D = torch::Tensor();
    auto Tcw = param.Tcw;

    
    if(eval)
     mean3D = param.mean3D.detach();
    else
     mean3D = param.mean3D;

    Tcw = Tcw.unsqueeze(0).repeat({mean3D.size(0),1,1}).to(_GPU_device, torch::kFloat32,true);

    mean3D = torch::cat({mean3D,torch::ones({mean3D.size(0),1}).to(_GPU_device,torch::kFloat32)},1).unsqueeze(-1);
    mean3D = Tcw.bmm(mean3D).squeeze(-1).index({torch::indexing::Slice(),torch::indexing::Slice(0,3)}).set_requires_grad(true);

    auto mean2D = torch::zeros_like(mean3D).to(_GPU_device, true).set_requires_grad(true);
    auto opacities = torch::sigmoid(param.logit_opacities);
    auto norm_qua = torch::nn::functional::normalize(param.unnorm_qua);
    auto scales = torch::exp(param.log_scales).set_requires_grad(true);
    rgb = param.rgb.set_requires_grad(true);

    mean2D.retain_grad();
    mean3D.retain_grad();
    rgb.retain_grad();
    scales.retain_grad();


    auto [rendererd_image, radii, depth] = mRasterizer.forward(
    mean3D,
    mean2D,
    opacities,
    shs,
    rgb,
    scales,
    norm_qua,
    pre_cov3D,
    _GPU_device.index());
    torch::cuda::synchronize();


    // mpGaussian->_mean2D = mean2D.to(_GPU_device, true).set_requires_grad(true);

    // return {rendererd_image, means2D, radii > 0, radii};
    return { rendererd_image, mean2D, radii,mean3D,rgb,scales};

    
}


std::tuple<torch::Tensor, torch::Tensor,torch::Tensor>
Render::Start(OptimizerGSParam param,bool eval)
{

    auto shs = torch::Tensor();
    auto pre_cov3D = torch::Tensor();
    auto rgb = torch::Tensor();
    auto mean3D = torch::Tensor();
    auto Tcw = param.Tcw;

    
    if(eval)
     mean3D = param.mean3D.detach();
    else
     mean3D = param.mean3D.to(_GPU_device);

    // mpCam->SetPose(param.Tcw);

    // GaussianRasterizationSettings RasterSettings_ = GaussianRasterizationSettings{
    // .image_height = static_cast<int>(mpCam->image_height),
    // .image_width = static_cast<int>(mpCam->image_width),
    // .tanfovx = mpCam->_tanfovx,
    // .tanfovy = mpCam->_tanfovy,
    // .bg = mpGaussian->_background,
    // .scale_modifier = mpGaussian->_scale_modifier,
    // .viewmatrix = mpCam->_viewmatrix,
    // .projmatrix = mpCam->_full_projmatrix,
    // .sh_degree = mpCam->sh_degree,
    // .camera_center = mpCam->_campos,
    // .prefiltered = false
    // };

    // auto Rasterizer = GaussianRasterizer(RasterSettings_);

    Tcw = Tcw.unsqueeze(0).repeat({mean3D.size(0),1,1}).to(_GPU_device, torch::kFloat32,true);

    mean3D = torch::cat({mean3D,torch::ones({mean3D.size(0),1}).to(_GPU_device,torch::kFloat32)},1).unsqueeze(-1);
    mean3D = Tcw.bmm(mean3D).squeeze(-1).index({torch::indexing::Slice(),torch::indexing::Slice(0,3)}).set_requires_grad(true);
    // mean3D = param.mean3D;
    auto mean2D = torch::zeros_like(mean3D).to(_GPU_device, true).set_requires_grad(true);
    auto opacities = torch::sigmoid(param.logit_opacities).to(_GPU_device);
    auto norm_qua = torch::nn::functional::normalize(param.unnorm_qua).to(_GPU_device);
    auto scales = torch::exp(param.log_scales).to(_GPU_device).set_requires_grad(true);
    rgb = param.rgb.to(_GPU_device).set_requires_grad(true);

    mean2D.retain_grad();

    MaxGPUUseOfMemory = max(PrintCudaMenory(),MaxGPUUseOfMemory);
    auto [rendererd_image, radii, depth] = mRasterizer.forward(
    mean3D,
    mean2D,
    opacities,
    shs,
    rgb,
    scales,
    norm_qua,
    pre_cov3D,
    _GPU_device.index());
    torch::cuda::synchronize();

    return { rendererd_image, depth, radii};

    
}


std::tuple<torch::Tensor, torch::Tensor,torch::Tensor>
Render::Start_Tracking(OptimizerGSParam param,bool eval)
{

    auto shs = torch::Tensor();
    auto pre_cov3D = torch::Tensor();
    auto rgb = torch::Tensor();
    auto mean3D = torch::Tensor();
    auto Tcw = param.Tcw;

    
    if(eval)
     mean3D = param.mean3D.detach();
    else
     mean3D = param.mean3D.to(_GPU_device);

    // mpCam->SetPose(param.Tcw);

    // GaussianRasterizationSettings RasterSettings_ = GaussianRasterizationSettings{
    // .image_height = static_cast<int>(mpCam->image_height),
    // .image_width = static_cast<int>(mpCam->image_width),
    // .tanfovx = mpCam->_tanfovx,
    // .tanfovy = mpCam->_tanfovy,
    // .bg = mpGaussian->_background,
    // .scale_modifier = mpGaussian->_scale_modifier,
    // .viewmatrix = mpCam->_viewmatrix,
    // .projmatrix = mpCam->_full_projmatrix,
    // .sh_degree = mpCam->sh_degree,
    // .camera_center = Tcw.inverse().index({Slice(0,3),Slice(3,None)}),
    // .prefiltered = false
    // };

    // auto Rasterizer = GaussianRasterizer(RasterSettings_);

    Tcw = Tcw.unsqueeze(0).repeat({mean3D.size(0),1,1}).to(_GPU_device, torch::kFloat32,true);

    mean3D = torch::cat({mean3D,torch::ones({mean3D.size(0),1}).to(_GPU_device,torch::kFloat32)},1).unsqueeze(-1);
    mean3D = Tcw.bmm(mean3D).squeeze(-1).index({torch::indexing::Slice(),torch::indexing::Slice(0,3)}).set_requires_grad(true);
    auto mean2D = torch::zeros_like(mean3D).to(_GPU_device, true).set_requires_grad(true);
    auto opacities = torch::sigmoid(param.logit_opacities).to(_GPU_device);
    auto norm_qua = torch::nn::functional::normalize(param.unnorm_qua).to(_GPU_device);
    auto scales = torch::exp(param.log_scales).to(_GPU_device).set_requires_grad(true);
    rgb = param.rgb.to(_GPU_device).set_requires_grad(true);

    mean2D.retain_grad();

    MaxGPUUseOfMemory = max(PrintCudaMenory(),MaxGPUUseOfMemory);
    auto [rendererd_image, radii, depth] = mRasterizer.forward(
    mean3D,
    mean2D,
    opacities,
    shs,
    rgb,
    scales,
    norm_qua,
    pre_cov3D,
    _GPU_device.index());
    torch::cuda::synchronize();

    return { rendererd_image, depth, radii};

    
}

std::tuple<torch::Tensor>
Render::Start_Radii(OptimizerGSParam param)
{

    mpCam->SetPose(param.Tcw);

    GaussianRasterizationSettings RasterSettings_ = GaussianRasterizationSettings{
    .image_height = static_cast<int>(mpCam->image_height),
    .image_width = static_cast<int>(mpCam->image_width),
    .tanfovx = mpCam->_tanfovx,
    .tanfovy = mpCam->_tanfovy,
    .bg = mpGaussian->_background,
    .scale_modifier = mpGaussian->_scale_modifier,
    .viewmatrix = mpCam->_viewmatrix,
    .projmatrix = mpCam->_full_projmatrix,
    .sh_degree = mpCam->sh_degree,
    .camera_center = mpCam->_campos,
    .prefiltered = false
    };

    GaussianRasterizer Rasterizer(RasterSettings_);

    auto shs = torch::Tensor();
    auto pre_cov3D = torch::Tensor();
    auto rgb = torch::Tensor();
    auto mean3D = torch::Tensor();
    auto Tcw = param.Tcw;

    mean3D = param.mean3D.detach();

    // Tcw = Tcw.unsqueeze(0).repeat({mean3D.size(0),1,1}).to(_GPU_device, torch::kFloat32,true);

    // mean3D = torch::cat({mean3D,torch::ones({mean3D.size(0),1}).to(_GPU_device,torch::kFloat32)},1).unsqueeze(-1);
    // mean3D = Tcw.bmm(mean3D).squeeze(-1).index({torch::indexing::Slice(),torch::indexing::Slice(0,3)}).detach();

    auto opacities = torch::sigmoid(param.logit_opacities).detach().to(_GPU_device);
    auto norm_qua = torch::nn::functional::normalize(param.unnorm_qua).detach().to(_GPU_device);
    auto scales = torch::exp(param.log_scales).detach().to(_GPU_device);


    auto radii = Rasterizer.Visable(
    mean3D,
    opacities,
    scales,
    norm_qua,
    _GPU_device.index());

    torch::cuda::synchronize();

    return {radii};

    
}


void Render::GSParamRGBUpdata(OptimizerGSParam& param,torch::Tensor visual,bool isTracking)
{


    if(isTracking)
    {
    param.mean3D = mpGaussian->_mean3D.index_select(0, visual).detach();
    param.rgb = mpGaussian->_rgb.index_select(0, visual).detach();
    param.unnorm_qua = mpGaussian->_unnorm_qua.index_select(0, visual).detach();
    param.logit_opacities = mpGaussian->_logit_opacities.index_select(0, visual).detach();
    param.log_scales = mpGaussian->_log_scales.index_select(0, visual).detach();
    }
    else
    {
    param.mean3D = mpGaussian->_mean3D.index_select(0, visual);
    param.rgb = mpGaussian->_rgb.index_select(0, visual);
    param.unnorm_qua = mpGaussian->_unnorm_qua.index_select(0, visual);
    param.logit_opacities = mpGaussian->_logit_opacities.index_select(0, visual);
    param.log_scales = mpGaussian->_log_scales.index_select(0, visual);

    }

}

void Render::GSParamRGB2Depth(OptimizerGSParam& paramIm,OptimizerGSParam& paramD,bool Tracking)
{
    if(Tracking)
    {
    paramD.unnorm_qua = paramIm.unnorm_qua.detach();
    paramD.logit_opacities = paramIm.logit_opacities.detach();
    paramD.log_scales = paramIm.log_scales.detach();
    paramD.mean3D = paramIm.mean3D.detach(); 
            
    auto Tcw_ = paramIm.Tcw.unsqueeze(0).repeat({paramD.mean3D.size(0),1,1}).to(_GPU_device,torch::kFloat32);
    auto one_ = torch::ones({paramD.mean3D.size(0),1}).to(_GPU_device);
    auto mean3D_d = torch::cat({paramD.mean3D,one_},1).unsqueeze(-1).to(_GPU_device,torch::kFloat32); 
    paramD.rgb = Tcw_.bmm(mean3D_d).squeeze(-1).index({"...",Slice(2,3)}).repeat({1,3}).clone().detach();
    paramD.rgb.index_put_({"...",1},1);
    paramD.rgb.index_put_({"...",2},0);
    paramD.Tcw=paramIm.Tcw.clone();
    }
    else
    {
    paramD.unnorm_qua = paramIm.unnorm_qua;
    paramD.logit_opacities = paramIm.logit_opacities;
    paramD.log_scales = paramIm.log_scales;
    paramD.mean3D = paramIm.mean3D; 
            
    auto Tcw_ = paramIm.Tcw.unsqueeze(0).repeat({paramD.mean3D.size(0),1,1}).to(_GPU_device,torch::kFloat32);
    auto one_ = torch::ones({paramD.mean3D.size(0),1}).to(_GPU_device);
    auto mean3D_d = torch::cat({paramD.mean3D,one_},1).unsqueeze(-1).to(_GPU_device,torch::kFloat32); 
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
        param.mean3D = mpGaussian->_mean3D.index_select(0, visual).detach();
        auto Tcw_ = param.Tcw.unsqueeze(0).repeat({param.mean3D.size(0),1,1}).to(_GPU_device,torch::kFloat32, true);
        auto mean3D_d = torch::cat({param.mean3D,torch::ones({param.mean3D.size(0),1}).to(_GPU_device,torch::kFloat32)},1).unsqueeze(-1);
        param.rgb = Tcw_.bmm(mean3D_d).squeeze(-1).index({"...",Slice(2,3)}).repeat({1,3}).detach();
        param.rgb.index_put_({"...",1},1);
        param.rgb.index_put_({"...",2},0);

        param.unnorm_qua = mpGaussian->_unnorm_qua.index_select(0, visual).detach();
        param.logit_opacities = mpGaussian->_logit_opacities.index_select(0, visual).detach();
        param.log_scales = mpGaussian->_log_scales.index_select(0, visual).detach();
    }
    else
    {
        param.unnorm_qua = mpGaussian->_unnorm_qua.index_select(0, visual);
        param.logit_opacities = mpGaussian->_logit_opacities.index_select(0, visual);
        param.log_scales = mpGaussian->_log_scales.index_select(0, visual);
        param.mean3D = mpGaussian->_mean3D.index_select(0, visual);                  
        auto Tcw_ = param.Tcw.unsqueeze(0).repeat({param.mean3D.size(0),1,1});
        auto one_ = torch::ones({param.mean3D.size(0),1}).to(_GPU_device);
        auto mean3D_d = torch::cat({param.mean3D,one_},1).unsqueeze(-1);
        param.rgb = Tcw_.bmm(mean3D_d).squeeze(-1).index({"...",Slice(2,3)}).repeat({1,3}).clone();
        param.rgb.index_put_({"...",1},1);
        param.rgb.index_put_({"...",2},0);
 


    }
}

void Render::GSParamRGBUpdata(OptimizerGSParam& param,bool isTracking)
{


    if(isTracking)
    {
    param.mean3D = mpGaussian->_mean3D.detach();
    param.rgb = mpGaussian->_rgb.detach();
    param.unnorm_qua = mpGaussian->_unnorm_qua.detach();
    param.logit_opacities = mpGaussian->_logit_opacities.detach();
    param.log_scales = mpGaussian->_log_scales.detach();
    }
    else
    {
    param.mean3D = mpGaussian->_mean3D;
    param.rgb = mpGaussian->_rgb;
    param.unnorm_qua = mpGaussian->_unnorm_qua;
    param.logit_opacities = mpGaussian->_logit_opacities;
    param.log_scales = mpGaussian->_log_scales;

    }

    

}


void Render::GSParamDepthUpdata(OptimizerGSParam& param,bool isTracking)
{
     
    if(isTracking)
    {
        param.mean3D = mpGaussian->_mean3D.detach();
        auto Tcw_ = param.Tcw.unsqueeze(0).repeat({param.mean3D.size(0),1,1}).to(_GPU_device,torch::kFloat32, true);
        auto mean3D_d = torch::cat({param.mean3D,torch::ones({param.mean3D.size(0),1}).to(_GPU_device,torch::kFloat32)},1).unsqueeze(-1);
        param.rgb = Tcw_.bmm(mean3D_d).squeeze(-1).index({"...",Slice(2,3)}).repeat({1,3}).detach();
        param.rgb.index_put_({"...",1},1);
        param.rgb.index_put_({"...",2},0);

        param.unnorm_qua = mpGaussian->_unnorm_qua.detach();
        param.logit_opacities = mpGaussian->_logit_opacities.detach();
        param.log_scales = mpGaussian->_log_scales.detach();
    }
    else
    {
        param.unnorm_qua = mpGaussian->_unnorm_qua;
        param.logit_opacities = mpGaussian->_logit_opacities;
        param.log_scales =mpGaussian->_log_scales;
        param.mean3D = mpGaussian->_mean3D;         
        auto one_ = torch::ones({param.mean3D.size(0),1}).to(_GPU_device);
        auto mean3D_d = torch::cat({param.mean3D,one_},1).unsqueeze(-1);
        auto Tcw_ = param.Tcw.unsqueeze(0).repeat({param.mean3D.size(0),1,1});
        param.rgb = Tcw_.bmm(mean3D_d).squeeze(-1).index({"...",Slice(2,3)}).repeat({1,3}).clone();
        param.rgb.index_put_({"...",1},1);
        param.rgb.index_put_({"...",2},0);
 


    }
}


cv::Mat GetEdge(cv::Mat depth)
{
    int64_t rows = depth.rows;
    int64_t cols = depth.cols;

    // 创建一个与矩阵相同大小的布尔 mask
    cv::Mat mask = cv::Mat::zeros(Size(cols,rows),CV_8UC1);

    // 计算四领域差值并生成 mask
    for (int64_t i = 1; i < rows-1; ++i) {
        for (int64_t j = 1; j < cols-1; ++j) {

            float value = depth.at<float>(i,j);
            if(value <= 0||depth.at<float>(i-1,j)==0 || depth.at<float>(i+1,j)==0||depth.at<float>(i,j+1)==0||depth.at<float>(i,j-1)==0) continue;
            // 计算上、下、左、右的差值
            float top_diff = std::abs(value - depth.at<float>(i-1,j));
            float bottom_diff =  std::abs(value - depth.at<float>(i+1,j));
            float left_diff =  std::abs(value - depth.at<float>(i,j-1));
            float right_diff = std::abs(value - depth.at<float>(i,j+1));

            // 判断差值是否大于 0.5
            if (top_diff > 0.05 || bottom_diff > 0.05 || left_diff > 0.05 || right_diff > 0.05) {
                mask.at<uchar>(i,j) = 255;
            }
        }
    }
    cv::dilate(mask,mask,getStructuringElement(MORPH_ELLIPSE, Size(10, 10)));
    return mask.clone();
}


void Render::RenderStartLocalTraking(Frame& CurrentF_,int iter_total,torch::Device device)
{
//     torch::Tensor rendered_image;
//     torch::Tensor rendered_depth;
//     torch::Tensor rendered_radii;
//     OptimizerGSParam paramGS;
//     paramGS.Tcw = mpGaussian->InitCameraPose(CurrentF_.mTcw);
//     cv::Mat im = CurrentF_.mImRGB.clone();
//     cv::Mat depth = CurrentF_.mImDepth.clone();
   
//     mpGaussian->CreateOptimizerForLocalPose();
//     torch::Tensor best_Qua = mpGaussian->_cam_qua_unnorm.clone();
//     torch::Tensor best_trans = mpGaussian->_can_trans.clone();


//     // cv::Mat im = CurrentF_.mImRGB.clone();
//     // cv::Mat depth = CurrentF_.mImDepth.clone();

//     // cv::Mat edge = GetEdge(depth);
//     // edge.convertTo(edge,CV_32F,1.0/255.0);
//     // auto EdgeMask = CvMat2Tensor(edge,_GPU_device).to(torch::kBool);

//     auto[tGtImage,tGtDepth] = ImAndDepth2tensor(im,depth,_GPU_device);

//     torch::Tensor min_loss = torch::tensor({FLT_MAX},torch::kFloat32).to(_GPU_device);
//     tic();
//     float last_loss = 0;
//     torch::Tensor tK_ = torch::tensor({fx,(float)0.0,cx,(float)0.0,fy,cy,(float)0.0,(float)0.0,(float)1.0},torch::kFloat32).reshape({3,3}).to(_GPU_device);
//     float feature_weight = _config["Tracking"]["feature_weight"].as<float>();
//     float im_weight = _config["Tracking"]["im_weight"].as<float>();
//     float depth_weight = _config["Tracking"]["depth_weight"].as<float>();
//     if(iter_total==0)
//      iter_total = _config["Tracking"]["iters"].as<int>();
//     cout<<"Tracking Process: ";

//     vector<float> vobs;
//     vector<float> vXw4;
//     vector<float> vinvSigma2;
//     int vilad_match_num = 0;


//     for(int i=0; i<CurrentF_.N; ++i)
//     {
//         MapPoint* pMP = CurrentF_.mvpMapPoints[i];
//         if(pMP)
//         {

//             CurrentF_.mvbOutlier[i]=false;
//             const cv::KeyPoint &kpUn = CurrentF_.mvKeysUn[i];
//             vobs.emplace_back(kpUn.pt.x);
//             vobs.emplace_back(kpUn.pt.y);
//             vobs.emplace_back(1);
//             vinvSigma2.emplace_back(CurrentF_.mvInvLevelSigma2[kpUn.octave]);
//             cv::Mat Xw = pMP->GetWorldPos();
//             vXw4.emplace_back(Xw.at<float>(0));
//             vXw4.emplace_back(Xw.at<float>(1));
//             vXw4.emplace_back(Xw.at<float>(2));
//             vXw4.emplace_back(1);
//             ++vilad_match_num;

//         }
//     }

//     torch::Tensor tobs = torch::from_blob(vobs.data(),{vilad_match_num,3,1},torch::kFloat32).to(_GPU_device);

//     torch::Tensor tXw4 = torch::from_blob(vXw4.data(),{vilad_match_num,4,1},torch::kFloat32).to(_GPU_device);
//     torch::Tensor tinvSigma2 = torch::from_blob(vinvSigma2.data(),{vilad_match_num,1},torch::kFloat32).to(_GPU_device);
//     tinvSigma2 = tinvSigma2.unsqueeze(-1).repeat({1,1,2});
//     torch::Tensor diagonal_tinvSigma2 = torch::diag_embed(tinvSigma2);


//     auto tbK_ = tK_.unsqueeze(0).repeat({vilad_match_num,1,1});
//     torch::Tensor delta = torch::tensor({15},torch::kFloat32).to(_GPU_device);
//     torch::Tensor Lrpj = torch::zeros({1},torch::kFloat32).to(_GPU_device).set_requires_grad(true);
//         cout<<vilad_match_num<<endl;
//    int feature_clear = iter_total/3.0;
//    torch::Tensor inline_mask = torch::ones({vilad_match_num,1},torch::kBool).to(_GPU_device);
//     for(int iter=0; iter<iter_total; ++iter)
//     {

//         mTrackingParams.Tcw = Rt2T(mpGaussian->_cam_qua_unnorm.clone(),mpGaussian->_can_trans.clone()).to(_GPU_device);
 
//         auto tbTcw = mTrackingParams.Tcw.unsqueeze(0).repeat({vilad_match_num,1,1});
//         auto tXc4 = (tbTcw.bmm(tXw4)).transpose(1,2);
//         auto tXc3 = tXc4.index({"...",Slice({None,3})});
//         tXc3 = tXc3.div(tXc3.index({"...",0,2}).unsqueeze(-1).unsqueeze(-1).repeat({1,1,3}));
//         auto pixel_uv = (tbK_.bmm(tXc3.transpose(1,2)));
//         auto pixel_error = (pixel_uv - tobs).index({"...",Slice(0,2),0}).unsqueeze(-1).unsqueeze(1);
//         auto weightError = pixel_error.transpose(2,3).matmul(diagonal_tinvSigma2);
//         weightError = weightError.matmul(pixel_error).squeeze(-1).squeeze(-1);

       
//         // GSParamRGBUpdata(paramGS,true);
//         // std::tie(rendered_image, ignore, ignore) = Start(paramGS);
//         // GSParamDepthUpdata(paramGS,true);
//         // std::tie(rendered_depth, ignore, ignore) = Start(paramGS);
       
        
//         auto uncertain_depth = (rendered_depth[1]>0.99&~isnan(tGtDepth));
//         auto invail_depth_mask = (uncertain_depth).to(_GPU_device);
        
       
//         cv::Mat rendered_im = ImshowRGB(rendered_image,"Track");
      
//         if(iter % feature_clear == 0)
//         {
//             inline_mask = weightError < 7.815;
//         }
//         Lrpj = weightError.masked_select(inline_mask).sum();
//         // EdgeMask = EdgeMask&uncertain_depth;


//         auto ImageL1 = L1LossForTracking(rendered_image,tGtImage,uncertain_depth.tile({3,1,1}).detach());
//         auto DepthL1 = L1LossForTracking(rendered_depth[0],tGtDepth,uncertain_depth.detach());
//         // auto DepthEdgeL1 = L1LossForTracking(rendered_depth[0],tGtDepth,EdgeMask.detach());

//         // torch::Tensor loss = im_weight*ImageL1 + depth_weight*DepthL1 + feature_weight*Lrpj;
//         torch::Tensor loss = im_weight*ImageL1 + depth_weight*DepthL1 ;
       

//         loss.backward();
//         {
//             torch::NoGradGuard no_grad;
            
//             if(!std::isnan(loss.item<float>()) && loss.item<float>() < min_loss.item<float>())
//             {
//                 best_Qua = mpGaussian->_cam_qua_unnorm.detach().clone();
//                 best_trans = mpGaussian->_can_trans.detach().clone();
//                 min_loss = loss;
                
//             }
//             if(_config["Debug"]["use_wandb"].as<bool>())
//             wandbcpp::log({{to_string(CurrentF_.mnId)+"Loss", loss.item<float>()},{to_string(CurrentF_.mnId)+"Image", ImageL1.item<float>()}
//             ,{to_string(CurrentF_.mnId)+"Depth", DepthL1.item<float>()},{to_string(CurrentF_.mnId)+"Lrpj", Lrpj.item<float>()},{to_string(CurrentF_.mnId)+"mask", inline_mask.sum().item<int>()}});
//             // for(int j=0;j<weightError.size(0);++j)
//             // {
//             //     wandbcpp::log({{to_string(CurrentF_.mnId)+"errb", weightError[j].item<float>()}});
//             // }
//                 mpGaussian->StepUpdataForLocalPose();
            

//         }//no_grad
//         // waitKey(200);

//         cout<<"I";
//     }
//     cout<<endl;
//     toc("Tracking Time Cost: ");

//     auto t_T = Rt2T(best_Qua,best_trans);
//     cv::Mat pose(t_T.size(0),t_T.size(1),CV_32FC1,t_T.data_ptr());
//     CurrentF_.SetPose(pose);

    RemoveOutline(CurrentF_);
  
    
    c10::cuda::CUDACachingAllocator::emptyCache();






}


void Render::RenderStartTraking(Frame& CurrentF_,int iter_total,torch::Device device)
{
    OptimizerGSParam ParamGs;
    ParamGs.Tcw = mpGaussian->InitCameraPose(CurrentF_.mTcw);

    mpGaussian->CreateOptimizerForPose();
    torch::Tensor best_Qua = mpGaussian->_cam_qua_unnorm.clone();
    torch::Tensor best_trans = mpGaussian->_can_trans.clone();
    torch::Tensor rendered_image;
    torch::Tensor rendered_depth;
    torch::Tensor rendered_radii;
    torch::Tensor rendered_surdepth;

    cv::Mat im = CurrentF_.mImRGB.clone();
    cv::Mat depth = CurrentF_.mImDepth.clone();

    auto[tGtImage,tGtDepth] = ImAndDepth2tensor(im,depth,_GPU_device);

    torch::Tensor min_loss = torch::tensor({FLT_MAX},torch::kFloat32).to(_GPU_device);
    tic();
    torch::Tensor tK_ = torch::tensor({fx,(float)0.0,cx,(float)0.0,fy,cy,(float)0.0,(float)0.0,(float)1.0},torch::kFloat32).reshape({3,3}).to(_GPU_device);
    float feature_weight = _config["Tracking"]["feature_weight"].as<float>();
    float im_weight = _config["Tracking"]["im_weight"].as<float>();
    float depth_weight = _config["Tracking"]["depth_weight"].as<float>();
    if(iter_total==0)
     iter_total = _config["Tracking"]["iters"].as<int>();
    // cout<<"Tracking Process: ";
    tic();
    vector<float> vobs;
    vector<float> vXw4;
    vector<float> vinvSigma2;
    int vilad_match_num = 0;

    for(int i=0; i<CurrentF_.N; ++i)
    {
        MapPoint* pMP = CurrentF_.mvpMapPoints[i];
        if(pMP)
        {

            CurrentF_.mvbOutlier[i]=false;
            const cv::KeyPoint &kpUn = CurrentF_.mvKeysUn[i];
            vobs.emplace_back(kpUn.pt.x);
            vobs.emplace_back(kpUn.pt.y);
            vobs.emplace_back(1);
            vinvSigma2.emplace_back(CurrentF_.mvInvLevelSigma2[kpUn.octave]);
            cv::Mat Xw = pMP->GetWorldPos();
            vXw4.emplace_back(Xw.at<float>(0));
            vXw4.emplace_back(Xw.at<float>(1));
            vXw4.emplace_back(Xw.at<float>(2));
            vXw4.emplace_back(1);
            ++vilad_match_num;

        }
    }


    torch::Tensor tobs = torch::from_blob(vobs.data(),{vilad_match_num,3,1},torch::kFloat32).to(_GPU_device);
    torch::Tensor tXw4 = torch::from_blob(vXw4.data(),{vilad_match_num,4,1},torch::kFloat32).to(_GPU_device);
    torch::Tensor tinvSigma2 = torch::from_blob(vinvSigma2.data(),{vilad_match_num,1},torch::kFloat32).to(_GPU_device);
    tinvSigma2 = tinvSigma2.unsqueeze(-1).repeat({1,1,2});
    torch::Tensor diagonal_tinvSigma2 = torch::diag_embed(tinvSigma2);

    auto tbK_ = tK_.unsqueeze(0).repeat({vilad_match_num,1,1});
    torch::Tensor delta = torch::tensor({15},torch::kFloat32).to(_GPU_device);
    torch::Tensor Lrpj = torch::zeros({1},torch::kFloat32).to(_GPU_device).set_requires_grad(true);
    torch::Tensor inline_mask = torch::ones({vilad_match_num,1},torch::kBool).to(_GPU_device);
    int feature_clear = iter_total/2.0;

    torch::Tensor last_loss = torch::zeros({1},torch::kFloat32).to(_GPU_device);

    for(int iter=0; iter<iter_total; ++iter)
    {

        ParamGs.Tcw = Rt2T(mpGaussian->_cam_qua_unnorm.clone(),mpGaussian->_can_trans.clone()).to(_GPU_device);
        auto tbTcw = ParamGs.Tcw.unsqueeze(0).repeat({vilad_match_num,1,1});
        auto tXc4 = (tbTcw.bmm(tXw4)).transpose(1,2);
        auto tXc3 = tXc4.index({"...",Slice({None,3})});
        tXc3 = tXc3.div(tXc3.index({"...",0,2}).unsqueeze(-1).unsqueeze(-1).repeat({1,1,3}));
        auto pixel_uv = (tbK_.bmm(tXc3.transpose(1,2)));
        auto pixel_error = (pixel_uv - tobs).index({"...",Slice(0,2),0}).unsqueeze(-1).unsqueeze(1);
        auto weightError = pixel_error.transpose(2,3).matmul(diagonal_tinvSigma2);
        weightError = weightError.matmul(pixel_error).squeeze(-1).squeeze(-1);

        //GSParamRGBUpdata(ParamGs,true);
        //auto[radii] = Start_Radii(ParamGs);
        //auto visible_index = (radii>0).nonzero().squeeze().to(torch::kLong);
        GSParamRGBUpdata(ParamGs,true);
        std::tie(rendered_image, rendered_surdepth, ignore) = Start_Tracking(ParamGs);
        GSParamDepthUpdata(ParamGs,true);
        std::tie(rendered_depth, ignore, ignore) = Start_Tracking(ParamGs);


        auto uncertain_depth = (rendered_depth[1]>0.99&~isnan(tGtDepth));
        
        // ImshowDepth(uncertain_depth,"unc");

        // cv::Mat rendered_im = ImshowRGB(rendered_image,"Track");
    
        if(iter == feature_clear)
        {
            inline_mask = weightError < 5.991;
        }
        Lrpj = weightError.masked_select(inline_mask).sum();
        // Lrpj = weightError.sum();
        torch::Tensor DepthL1;
        auto ImageL1 = L1LossForTracking(rendered_image,tGtImage,uncertain_depth.tile({3,1,1}).detach());
        if(_config["Debug"]["use_sur"].as<bool>())
        DepthL1 = L1LossForTracking(rendered_surdepth,tGtDepth,uncertain_depth.detach()); 
        else
        DepthL1 = L1LossForTracking(rendered_depth[0],tGtDepth,uncertain_depth.detach()); 
        
        torch::Tensor loss = im_weight*ImageL1 + depth_weight*DepthL1 + feature_weight*Lrpj;

        loss.backward();
        {
            torch::NoGradGuard no_grad;
            
            if(!std::isnan(loss.item<float>()) && loss.item<float>() < min_loss.item<float>())
            {
                best_Qua = mpGaussian->_cam_qua_unnorm.detach().clone();
                best_trans = mpGaussian->_can_trans.detach().clone();
                min_loss = loss;
                
            }
            if(abs(last_loss-loss).item<float>()<10e-5)
            {
                break;
            }

            last_loss = loss;

            if(_config["Debug"]["use_wandb"].as<bool>())
            wandbcpp::log({{to_string(CurrentF_.mnId)+"Loss", loss.item<float>()},{to_string(CurrentF_.mnId)+"Image", ImageL1.item<float>()}
            ,{to_string(CurrentF_.mnId)+"Depth", DepthL1.item<float>()},{to_string(CurrentF_.mnId)+"Lrpj", Lrpj.item<float>()}});

            mpGaussian->StepUpdataForPose();
            ++Tracking_counts;
            if(PrintCudaMenory() > mTotalGpuByte)
                c10::cuda::CUDACachingAllocator::emptyCache();


        }//no_grad
        // waitKey(200);

        // cout<<"I";
    }
    // cout<<endl;
    Tracking_times += toc("Tracking Time Cost: ");

    auto t_T = Rt2T(best_Qua,best_trans);
    cv::Mat pose(t_T.size(0),t_T.size(1),CV_32FC1,t_T.data_ptr());
    CurrentF_.SetPose(pose);

    RemoveOutline(CurrentF_);


}

void Render::RemoveOutline(Frame& CurrentF_)
{
    float sigma2 = 1.25;
    float chi2Th = 3.99;
    for(int i=0; i<CurrentF_.N; ++i)
    {
        MapPoint* pMP = CurrentF_.mvpMapPoints[i];
        if(pMP)
        {
            const cv::KeyPoint &kpUn = CurrentF_.mvKeysUn[i];
            cv::Mat obs = (cv::Mat_<float>(2,1) << kpUn.pt.x, kpUn.pt.y);
            cv::Mat Xw = pMP->GetWorldPos();
            cv::Mat Xw4 = (cv::Mat_<float>(4,1) << Xw.at<float>(0),Xw.at<float>(1),Xw.at<float>(2),1);
            cv::Mat Xc = CurrentF_.mTcw*Xw4;
            float invz = 1.0/Xc.at<float>(2);
            float u = fx*Xc.at<float>(0)*invz + cx;
            float v = fy*Xc.at<float>(1)*invz + cy;

            cv::Mat est = (cv::Mat_<float>(2,1) << u, v);
            cv::Mat info = cv::Mat::eye(2,2,CV_32F)*CurrentF_.mvInvLevelSigma2[kpUn.octave];
            cv::Mat error = obs-est;
            // float squareError= sqrt((est.at<float>(0)-obs.at<float>(0))*(est.at<float>(0)-obs.at<float>(0))
            // +(est.at<float>(1)-obs.at<float>(1))*(est.at<float>(1)-obs.at<float>(1)));
            float chi2 = error.dot(info*error);
            
            // if(squareError > 4*sigma2)
            if(chi2 > chi2Th)
                CurrentF_.mvbOutlier[i]=true;
        }
    }
}



} // namespace ORB_SLAM2
