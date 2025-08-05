#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>
#include <sys/wait.h>

using namespace std;
using namespace cv;
int main(int argc, char **argv)
{


    if(argc != 4)
    {
        cerr << endl << "Usage: ./rgbd_replica path_to_vocabulary path_to_settings path_to_sequence " << endl;
        return 1;
    }

    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);
    YAML::Node config = YAML::LoadFile(argv[2]);
    int total_im_frame = config["Dataset"]["num"].as<int>();

    int nImages = total_im_frame;
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;
    vector<string> vImRGBflieName(nImages);
    vector<string> vImDepthfileName(nImages);
     for(int ni=0; ni<nImages; ni++)
    {

        string sequence = "000000";
        string str_id = to_string(ni);
        sequence.replace(sequence.size()-str_id.size(),str_id.size(),str_id);
        
        cv::Mat imRGB = imread(string(argv[3])+"/results/frame"+sequence+".jpg",cv::IMREAD_UNCHANGED);
        cv::Mat imD = imread(string(argv[3])+"/results/depth"+sequence+".png",cv::IMREAD_UNCHANGED);
        vImRGBflieName[ni] = "/frame"+sequence+".jpg";
        vImDepthfileName[ni] = "/depth"+sequence+".png";
        double tframe = ni;

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/frame" << ni << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        SLAM.TrackRGBD(imRGB,imD,tframe);
        // SLAM.TrackMonocular(imRGB,tframe);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        ORB_SLAM2::printProgress(static_cast<double>(ni) / (static_cast<double>(nImages) - 1));



    }
    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

     // Stop all threads
    SLAM.Shutdown();
    std::string savePath = SLAM.GetSavePath();

   
    if(SLAM._enableEvalution)
    {
        vector<cv::Mat> vImage;
        vector<cv::Mat> vDepth;
        cv::Mat imRGB, imD;

        vImage.reserve(nImages);
        vDepth.reserve(nImages);

        std::vector<cv::Mat> vPose = SLAM.GetPoseForEvalution();
    
        float depthFactor = SLAM.GetDepthFactor();
        for(int ni=0; ni<nImages; ni++)
        {
            // Read image and depthmap from file
            imRGB = cv::imread(string(argv[3]) + "/results/" + vImRGBflieName[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
            imD = cv::imread(string(argv[3]) + "/results/" + vImDepthfileName[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);

            vImage.emplace_back(imRGB);
            imD.convertTo(imD,CV_32F,depthFactor);
            vDepth.emplace_back(imD);
        }
        Evalution(SLAM.GetRender(),vPose,vImage,vDepth,savePath);

    } 

     int pid = fork();
    if (pid < 0)
    {
        cout << "fork failed" << endl;
    }
    else if( pid == 0 )
    {
        auto gtString = string(argv[3]) + "/traj.txt";
        auto trajPathString = savePath + "/CarameTrajectory.txt";
        char *gtPath = (char *)(gtString.c_str());
        char *trajPath = (char *)(trajPathString.c_str());
        char *savePathc = (char *)(savePath.c_str());

        std::cout << "ATE RMSE: " << std::endl;
        char *execArgs[] = {"python3", "scripts/eval_ate.py", gtPath, trajPath, "--save_path", savePathc, NULL};
        execvp("python3", execArgs);
    }
    wait(NULL);

    return 0;
}