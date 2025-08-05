/**
* This file modified from ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <sys/wait.h>
#include<unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <dirent.h>


#include<opencv2/core/core.hpp>
#include<opencv2/imgcodecs/legacy/constants_c.h>

#include<System.h>
#include <sys/wait.h>

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames);
string GetDatasetName(const string &strSequencePath);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: " << argv[0] << " path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<string> vstrDepthFilenames;
    string strFile = string(argv[3])+"/color";
    LoadImages(strFile, vstrImageFilenames);
    strFile = string(argv[3]) + "/depth";
    LoadImages(strFile, vstrDepthFilenames);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD, true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;

    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        imRGB = cv::imread(string(argv[3])+"/color/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        cv::resize(imRGB, imRGB, cv::Size(640, 480));
        imD = cv::imread(string(argv[3]) + "/depth/" + vstrDepthFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        
        double tframe = ni;

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << strFile << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

        if(imD.empty())
        {
            cerr << endl << "Failed to load depth at: "
                 << strFile << "/" << vstrDepthFilenames[ni] << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB,imD,tframe);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesTrack[ni]=ttrack;
        ORB_SLAM2::printProgress(static_cast<double>(ni) / (static_cast<double>(nImages) - 1));
        
        }

    // Stop orb-viewer and tracking. 
    // The user can watch the Nerf screen
    SLAM.Shutdown();

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

    std::string savePath = SLAM.GetSavePath();

    
    if(SLAM._enableEvalution)
    {
        vector<cv::Mat> vImage;
        vector<cv::Mat> vDepth;

        vImage.reserve(nImages);
        vDepth.reserve(nImages);

        std::vector<cv::Mat> vPose = SLAM.GetPoseForEvalution();
    
        float depthFactor = SLAM.GetDepthFactor();
        for(int ni=0; ni<nImages; ni++)
        {
            // Read image and depthmap from file
            imRGB = cv::imread(string(argv[3]) + "/color/" + vstrImageFilenames[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
            cv::resize(imRGB, imRGB, cv::Size(640, 480));
            imD = cv::imread(string(argv[3]) + "/depth/" + vstrDepthFilenames[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);

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
        auto gtString = string(argv[3]) + "/groundtruth.txt";
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

void ls(const string &path, vector<string> &vstrImageFilenames) {
    DIR *mydir;
    struct dirent *myfile;
    struct stat mystat;

    mydir = opendir(path.c_str());
    if (!mydir) {
        cerr << "Unable to open " << path << endl;
        exit(1);
    }
    while((myfile = readdir(mydir)) != NULL)
    {
        stat(myfile->d_name, &mystat); 
        std::string filename = std::string(myfile->d_name);
        if (filename != "." && filename != "..")
            vstrImageFilenames.push_back(filename);
    }
    closedir(mydir);
    
    sort(vstrImageFilenames.begin(), 
        vstrImageFilenames.end(), 
        [](string a, string b) {
            return stoi(a) < stoi(b);
    });
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames)
{
    ls(strFile, vstrImageFilenames);
}

string GetDatasetName(const string &strSequencePath) 
{
    string s(strSequencePath);
    std::string delimiter = "/";

    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        s.erase(0, pos + delimiter.length());
    }

    if (s.length() == 0)
        return token;
    else
        return s;
}