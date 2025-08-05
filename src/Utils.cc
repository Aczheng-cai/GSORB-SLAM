#include "Utils.h"
using namespace std;
using namespace cv;
using namespace torch::indexing;

namespace ORB_SLAM2
{


#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
const char* spinner = "|/-\\";
static int i = 0;

static const float C1 = 0.01 * 0.01;
static const float C2 = 0.03 * 0.03;
static std::chrono::steady_clock::time_point t_s;
static std::chrono::steady_clock::time_point t_e;

void printProgress(double percentage)
{
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s] %c", val, lpad, PBSTR, rpad, "", spinner[i++ % 4]);
    if (val == 100) {
        printf("\n");
    }
    fflush(stdout);
}


float PSNRMetric(const torch::Tensor& rendered_img, const torch::Tensor& gt_img) {
torch::Tensor squared_diff = (rendered_img - gt_img).pow(2);
torch::Tensor mse_val = squared_diff.view({rendered_img.size(0), -1}).mean(1, true);
return (20.f * torch::log10(1.0 / mse_val.sqrt())).mean().item<float>();
}

torch::Tensor L1LossForMapping(const torch::Tensor& render_res, const torch::Tensor& gt,const torch::Tensor& mask) {
    if(!mask.defined())
    return torch::abs((render_res - gt)).mean();
    else
    return torch::abs((render_res - gt)).masked_select(mask).mean();
}
torch::Tensor L1LossForTracking(const torch::Tensor& render_res, const torch::Tensor& gt,const torch::Tensor& mask)
{
    if(!mask.defined())
    return torch::abs((render_res - gt)).sum();
    else
    return torch::abs((render_res - gt)).masked_select(mask).sum();

}

torch::Tensor SmoothL1LossForTracking(const torch::Tensor& render_res, const torch::Tensor& gt,const torch::Tensor& mask)
{
    torch::Tensor error;
    float beta = 0.5;
    if(!mask.defined())
    {
        error = abs(render_res - gt);
    }else{
        error = abs((render_res - gt)).masked_select(mask);
    }
    auto ret = where(error < beta,pow((0.5*error),2)/beta, error-0.5*beta);
    return ret.sum();

}
torch::Tensor GaussianGenerator(int window_size, float sigma) {
    torch::Tensor gauss = torch::empty(window_size);
    for (int x = 0; x < window_size; ++x) {
        gauss[x] = std::exp(-(std::pow(std::floor(static_cast<float>(x - window_size) / 2.f), 2)) / (2.f * sigma * sigma));
    }
    return gauss / gauss.sum();
}
torch::Tensor CreateWindow(int window_size, int channel) {
    auto _1D_window = GaussianGenerator(window_size, 1.5).unsqueeze(1);
    auto _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0);
    return _2D_window.expand({channel, 1, window_size, window_size}).contiguous();
}

torch::Tensor SSIM(const torch::Tensor& img1, const torch::Tensor& img2,torch::Device device) {
    
    int window_size = 11;
    int channel = 3;
    auto window = CreateWindow(window_size,channel).to(device);

    auto mu1 = torch::nn::functional::conv2d(img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
    auto mu1_sq = mu1.pow(2);
    auto sigma1_sq = torch::nn::functional::conv2d(img1 * img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu1_sq;

    auto mu2 = torch::nn::functional::conv2d(img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
    auto mu2_sq = mu2.pow(2);
    auto sigma2_sq = torch::nn::functional::conv2d(img2 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu2_sq;

    auto mu1_mu2 = mu1 * mu2;
    auto sigma12 = torch::nn::functional::conv2d(img1 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu1_mu2;
    auto ssim_map = ((2.f * mu1_mu2 + C1) * (2.f * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *(sigma1_sq + sigma2_sq + C2));

    return ssim_map.mean();
}
torch::Tensor SSIM(const torch::Tensor& img1, const torch::Tensor& img2, const torch::Tensor& mask,torch::Device device) {
    
    int window_size = 11;
    int channel = 3;
    auto window = CreateWindow(window_size,channel).to(device);

    auto mu1 = torch::nn::functional::conv2d(img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
    auto mu1_sq = mu1.pow(2);
    auto sigma1_sq = torch::nn::functional::conv2d(img1 * img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu1_sq;

    auto mu2 = torch::nn::functional::conv2d(img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
    auto mu2_sq = mu2.pow(2);
    auto sigma2_sq = torch::nn::functional::conv2d(img2 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu2_sq;

    auto mu1_mu2 = mu1 * mu2;
    auto sigma12 = torch::nn::functional::conv2d(img1 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu1_mu2;
    auto ssim_map = ((2.f * mu1_mu2 + C1) * (2.f * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *(sigma1_sq + sigma2_sq + C2));

    return ssim_map.masked_select(mask).mean();
}

torch::Tensor CvMat2Tensor(cv::Mat img,torch::Device device)
{
    return torch::from_blob(img.data,{img.rows,img.cols,img.channels()},torch::kFloat32).permute({2,0,1}).to(device);
}

cv::Mat ImshowRGB(torch::Tensor tensor,std::string s)
{
    auto render_im = tensor.div(max(tensor)).mul(255).permute({1,2,0}).clamp(0,255).to(torch::kCPU,torch::kU8).contiguous().detach();
    cv::Mat render_image(render_im.size(0),render_im.size(1),CV_8UC3,render_im.data_ptr());
    if(s != " ")
        cv::imshow(s,render_image);
    return render_image.clone();
}
cv::Mat ImshowDepth(torch::Tensor tensor,std::string s)
{
   torch::Tensor render_depth;
    if(tensor.dim()!=3)
     render_depth = tensor.unsqueeze(0).index({0,"..."}).div(max(tensor.unsqueeze(0).index({0,"..."}))).mul(255).to(torch::kCPU,torch::kU8).contiguous();
    else 
     render_depth = tensor.index({0,"..."}).div(max(tensor.index({0,"..."}))).mul(255).to(torch::kCPU,torch::kU8).contiguous();

    cv::Mat render_depth_im(render_depth.size(0),render_depth.size(1),CV_8UC1,render_depth.data_ptr());
    if(s != " ")
        cv::imshow(s,render_depth_im);
    return render_depth_im.clone();
}
cv::Mat ImshowDepthFloat(torch::Tensor tensor,std::string s)
{
   torch::Tensor render_depth;

     render_depth = tensor.to(torch::kCPU,torch::kFloat32).contiguous();

    cv::Mat render_depth_im(render_depth.size(0),render_depth.size(1),CV_32F,render_depth.data_ptr());
    if(s != " ")
        cv::imshow(s,render_depth_im);
    return render_depth_im.clone();
}

std::tuple<torch::Tensor,torch::Tensor> ImAndDepth2tensor(const cv::Mat& cvImage,const cv::Mat& cvDepth,torch::Device device)
{
    cv::Mat scaleIm;
    cvImage.convertTo(scaleIm,CV_32FC3,1.0 / 255.0);
    auto tGtImage = CvMat2Tensor(scaleIm,device);
    auto tGtDepth = CvMat2Tensor(cvDepth,device);
    return {tGtImage.clone(),tGtDepth.clone()};
}


torch::Tensor Rt2T(torch::Tensor Qua,torch::Tensor t)
{
    torch::Tensor t_T = torch::eye(4).to(torch::kFloat32);
    // cout<<Qua<<endl;
    auto t_R = ToRotation(Qua.transpose(0,1));
    // cout<<t_R<<endl;
    t_T.index_put_({Slice(0,3),Slice(0,3)},t_R.index({"..."}));
    t_T.index_put_({Slice(0,3),Slice(3)},t.index({"..."}));
    return t_T.clone();
}


void SavePly(shared_ptr<Gaussian> pgaussian, const path& filePath, bool isLastIteration) {
    if (!std::filesystem::exists(filePath)) {
        std::filesystem::create_directories(filePath);
    }
    std::cout << "Saving at " << filePath << " \n";
    // folder = file_path / ("point_cloud/");

    auto xyz = pgaussian->GetXYZ().contiguous().cpu();
    auto rgb = pgaussian->GetRGB().contiguous().cpu();
    auto opacities = pgaussian->GetLogitOpcity().cpu();
    auto scale = pgaussian->GetLogScale().cpu();
    auto rotation = pgaussian->GetUnnormQuat().cpu();

    std::vector<torch::Tensor> tensor_attributes = {xyz.clone(),
                                                    rgb.clone(),
                                                    opacities.clone(),
                                                    scale.clone(),
                                                    rotation.clone()};
    auto attributes = ConstructListAttributes(pgaussian.get());
    std::thread t = std::thread([filePath, tensor_attributes, attributes]() {
        WriteOutputPly(filePath / ("GaussianModel.ply"), tensor_attributes, attributes);
    });

    if (isLastIteration) {
        t.join();
    } else {
        t.detach();
    }
}

std::vector<std::string> ConstructListAttributes(Gaussian* pgaussian) {
    std::vector<std::string> attributes = {"x", "y", "z"};

    for (int i = 0; i < pgaussian->GetRGB().size(1); ++i)
        attributes.push_back("rgb_" + std::to_string(i));

    attributes.emplace_back("opacity");

    for (int i = 0; i < pgaussian->GetLogScale().size(1); ++i)
        attributes.push_back("scale_" + std::to_string(i));

    for (int i = 0; i < pgaussian->GetUnnormQuat().size(1); ++i)
    {
        attributes.push_back("rot_" + std::to_string(i));
    }

    return attributes;
}

void WriteOutputPly(const std::filesystem::path& file_path,
                    const std::vector<torch::Tensor>& tensors,
                    const std::vector<std::string>& attribute_names) {
    tinyply::PlyFile plyFile;
    size_t attribute_offset = 0;

    // Check attribute size
    size_t total_columns = 0;
    for (const auto& t : tensors) total_columns += t.size(1);
    if (attribute_names.size() < total_columns) {
        std::cerr << "Error: attribute_names too few!" << std::endl;
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto t = tensors[i];
        if (t.dim() != 2) {
            std::cerr << "Tensor " << i << " is not 2D." << std::endl;
            return;
        }
        if (!t.device().is_cpu()) {
            t = t.cpu();
        }

        size_t cols = t.size(1);
        std::vector<std::string> curr_attrs;
        for (size_t j = 0; j < cols; ++j)
            curr_attrs.push_back(attribute_names[attribute_offset + j]);

        plyFile.add_properties_to_element(
            "vertex", curr_attrs,
            tinyply::Type::FLOAT32, t.size(0),
            reinterpret_cast<uint8_t*>(t.data_ptr<float>()),
            tinyply::Type::INVALID, 0);

        attribute_offset += cols;
    }

    // std::filesystem::create_directories(file_path.parent_path());
    std::filebuf fb;
    fb.open(file_path, std::ios::out | std::ios::binary);
    if (!fb.is_open()) {
        std::cerr << "Cannot open file: " << file_path << std::endl;
        return;
    }
    std::ostream outputStream(&fb);
    std::cout << "Writing to file: " << file_path << std::endl;

    plyFile.write(outputStream, true);
}

unsigned long PrintCudaMenory()
{
    size_t free_byte;
    size_t totle_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte,&totle_byte);

    if(cuda_status!=cudaSuccess)
    {
        cout<<"Error: Get CUDA Memory Fail!\n";
    }
    double used_mem = ((double)totle_byte - (double)free_byte)/1024.0/1024.0;

    // cout<<"Total GPU Memory "<<totle_byte/1024.0/1024.0<<" MB\n";

    // cout<<"Now Used GPU Memory "<<used_mem<<" MB\n";
    return used_mem;
}

void tic()
{
    t_s = std::chrono::steady_clock::now();
}
double toc(string func)
{
    t_e = std::chrono::steady_clock::now();
    double t = std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(t_e-t_s).count();
    cout<<func<<": "<<t<<" ms"<<endl;
    return t;
}

// void RGB2HSV(torch::Tensor inIm, torch::Tensor &outIm)
// {
//   img_hsv = Mat::zeros(img_rgb.size(), CV_8UC3);
 
 
//   for (int i = 0; i < img_rgb.rows; i++)
//   {
//     Vec3b *p0 = inIm[];   //B--p[0]  G--p[1]  R--p[2]
//     Vec3b *p1 = img_hsv.ptr<Vec3b>(i);   //B--p[0]  G--p[1]  R--p[2]
 
 
//     for (int j = 0; j < img_rgb.cols; j++)
//     {
//       float B = p0[j][0] / 255.0;
//       float G = p0[j][1] / 255.0;
//       float R = p0[j][2] / 255.0;
      
//       float V = (float)std::max({ B, G, R });     //B/G/R
//       float vmin = (float)std::min({ B, G, R });
//       float diff = V - vmin;
 
 
//       float S, H;
//       S = diff / (float)(fabs(V) + FLT_EPSILON);
//       diff = (float)(60.0 / (diff + FLT_EPSILON));
 
 
//       if (V == B)   //V=B
//       {
//         H = 240.0 + (R - G) * diff;
//       }
//       else if (V == G)  //V=G
//       {
//         H = 120.0 + (B - R) * diff;
//       }
//       else if (V == R)   //V=R
//       {
//         H = (G - B) * diff;
//       }
 
 
//       H = (H < 0.0) ? (H + 360.0) : H;
 
 
//       p1[j][0] = (uchar)(H / 2);
//       p1[j][1] = (uchar)(S * 255);
//       p1[j][2] = (uchar)(V * 255);
//     }
//   }
// }



void Evalution(Render* pRender,const vector<cv::Mat>& Twc,const vector<cv::Mat>& vGtImage,const vector<cv::Mat>& vGtDepth,const string &filename)
{
    torch::jit::script::Module SSIMModel = torch::jit::load("/home/zwc/GSORB_SLAM/scripts/ms_ssim_model.pt");
    std::cout << "load ssim model is successed!" << std::endl;
    SSIMModel.to(GPU0);
    std::cout << "load ssim model to device!" << std::endl;
    SSIMModel.eval();

    torch::jit::script::Module LPIPSModel = torch::jit::load("/home/zwc/GSORB_SLAM/scripts/lpips_model.pt");
    std::cout << "load lpips model is successed!" << std::endl;
    LPIPSModel.to(GPU0);
    std::cout << "load lpips model to device!" << std::endl;
    LPIPSModel.eval();
    cout<<"Evalution ... "<<endl;
    int N = vGtImage.size();
    vector<float> vMeanPSNR(N,0);
    vector<float> vMeanSSIM(N,0);
    vector<float> vMeanMS_SSIM(N,0);
    vector<float> vMeanLPIPS(N,0);


    vector<cv::Mat> vRenderImg;
    vector<cv::Mat> vRenderDepth;

    string resultFilename= filename +"/result.txt";
    ofstream f;
    f.open(resultFilename.c_str());
    f << fixed;
    string renderim_save_path = filename+"/rendeImage";
    string renderdepth_save_path = filename+"/renderDepth";
    if (!std::filesystem::exists(renderim_save_path)) {
        std::filesystem::create_directories(renderim_save_path);
    }
    if (!std::filesystem::exists(renderdepth_save_path)) {
        std::filesystem::create_directories(renderdepth_save_path);
    }
  
    OptimizerGSParam paramGS;
    torch::Tensor renderedIm;
    torch::Tensor renderedDepth;
    torch::Tensor renderedSurdepth;
    for(int in=0; in<N; ++in)
    {
        torch::NoGradGuard no_grad;
        auto[tsImage,tsDepth] = ImAndDepth2tensor(vGtImage[in],vGtDepth[in],GPU0);
        
        paramGS.Tcw = torch::from_blob(Twc[in].data,{4,4},torch::kFloat32).cuda();
        
        pRender->GSParamRGBUpdata(paramGS,true);
        std::tie(renderedIm, renderedSurdepth, ignore) = pRender->StartSplatting(paramGS);

        pRender->GSParamDepthUpdata(paramGS,true);
        std::tie(renderedDepth, ignore, ignore) = pRender->StartSplatting(paramGS);

        auto mask = (tsDepth>0 & ~isnan(tsDepth));

        cv::Mat renderSaveIm = ImshowRGB(renderedIm);
        vRenderImg.emplace_back(renderSaveIm);

        renderedIm = renderedIm * mask.tile({3,1,1});
        tsImage = tsImage * mask.tile({3,1,1});

        // imshow("rendered_im",renderSaveIm);
        // waitKey(1);


        float psnr = PSNRMetric(tsImage,renderedIm);
        cv::Mat cvDepth = ImshowDepth(renderedSurdepth);
        applyColorMap(cvDepth,cvDepth,cv::COLORMAP_JET);
        vRenderDepth.emplace_back(cvDepth);
 
        
        std::vector<torch::jit::IValue> inputs1,inputs2;
        inputs1.push_back(renderedIm.unsqueeze(0));
        inputs1.push_back(tsImage.unsqueeze(0));
        float lpips = LPIPSModel.forward(inputs1).toTensor().item<float>();
        inputs2.push_back(renderedIm.div(255).unsqueeze(0).contiguous().cuda());
        inputs2.push_back(tsImage.div(255).unsqueeze(0).contiguous().cuda());
        float ms_ssim = SSIMModel.forward(inputs2).toTensor().item<float>();
        float ssim = SSIM(renderedIm,tsImage,GPU0).item<float>();

        vMeanPSNR[in] = psnr;
        vMeanSSIM[in] = ssim;
        vMeanLPIPS[in] = lpips;
        vMeanMS_SSIM[in] = ms_ssim;
        // f<<psnr<<endl;
        printProgress(static_cast<double>(in) / (static_cast<double>(N) - 1));
    }
    cout<<endl;
    cout<<"Mean PSNR: "<<accumulate(vMeanPSNR.begin(),vMeanPSNR.end(),0.0)/N<<endl;
    cout<<"Mean SSIM: "<<accumulate(vMeanSSIM.begin(),vMeanSSIM.end(),0.0)/N<<endl;
    cout<<"Mean MS_SSIM: "<<accumulate(vMeanMS_SSIM.begin(),vMeanMS_SSIM.end(),0.0)/N<<endl;
    cout<<"Mean LPIPS: "<<accumulate(vMeanLPIPS.begin(),vMeanLPIPS.end(),0.0)/N<<endl;

    f<<"Mean PSNR: "<<accumulate(vMeanPSNR.begin(),vMeanPSNR.end(),0.0)/N<<endl;
    f<<"Mean SSIM: "<<accumulate(vMeanSSIM.begin(),vMeanSSIM.end(),0.0)/N<<endl;
    f<<"Mean MS_SSIM: Use replay.py for evaluation."<<endl;
    f<<"Mean LPIPS: "<<accumulate(vMeanLPIPS.begin(),vMeanLPIPS.end(),0.0)/N<<endl;

    f.close();

    for(int i=0; i<N; ++i)
    {
        // cv::imwrite(renderdepth_save_path+'/'+to_string(i)+".png",vRenderDepth[i]);
        cv::imwrite(renderim_save_path+'/'+to_string(i)+".png",vRenderImg[i]);
    }
	

}








}


