
#include "Viewer2.h"
namespace ORB_SLAM2
{
static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "[ImGuiViewer]GLFW Error %d: %s\n", error, description);
}

ImGuiViewer::ImGuiViewer(FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer,const string &strSettingPath)
    : glfw_window_width_(1200),
      glfw_window_height_(680),
      panel_width_(372),
      display_panel_height_(144),
      training_panel_height_(440),
      camera_panel_height_(144),
      SLAM_image_viewer_scale_(1.0f)
{

    YAML::Node _config = YAML::LoadFile(strSettingPath);

    image_width_ = _config["Camera"]["width"].as<int>();
    image_height_ = _config["Camera"]["height"].as<int>();

    viewpointX_ = _config["Viewer2"]["ViewpointX"].as<float>();
    viewpointY_ = _config["Viewer2"]["ViewpointY"].as<float>();
    viewpointZ_ = _config["Viewer2"]["ViewpointZ"].as<float>();
    viewpointF_ = _config["Viewer2"]["ViewpointF"].as<float>();


    glfw_window_width_ =
        _config["Viewer2"]["glfw_window_width"].as<int>();
    glfw_window_height_ =
        _config["Viewer2"]["glfw_window_height"].as<int>();


    rendered_image_viewer_scale_ =
        _config["Viewer2"]["image_scale"].as<float>();
    rendered_image_height_ = image_height_ * rendered_image_viewer_scale_;
    rendered_image_width_ = image_width_ * rendered_image_viewer_scale_;

    int temp = rendered_image_width_ % 4;
    padded_sub_image_width_ = rendered_image_width_ + 4 - (temp == 0 ? 4 : temp);

    rendered_image_viewer_scale_main_ =
        _config["Viewer2"]["image_scale_main"].as<float>();
    rendered_image_height_main_ = image_height_ * rendered_image_viewer_scale_main_;
    rendered_image_width_main_ = image_width_ * rendered_image_viewer_scale_main_;

    temp = rendered_image_width_main_ % 4;
    padded_main_image_width_ = rendered_image_width_main_ + 4 - (temp == 0 ? 4 : temp); 

    camera_watch_dist_ =
    _config["Viewer2"]["camera_watch_dist"].as<float>();

    SLAM_image_viewer_scale_ = static_cast<float>(rendered_image_width_) / image_width_;

    float fovy = focal2fov(viewpointF_, image_height_);
    cam_proj_ = glm::perspective(
        fovy < M_PIf32 ? fovy : M_PIf32, (float)glfw_window_width_ / (float)glfw_window_height_, 0.01f, 100.0f);

    up_ = glm::vec3(0.0f, -1.0f, 0.0f);
    up_aligned_ = glm::vec4(up_, 1.0f);
    behind_ = glm::vec4(0.0f, 0.0f, -camera_watch_dist_, 1.0f);
    cam_pos_ = glm::vec3(viewpointX_, viewpointY_, viewpointZ_);
    cam_target_ = glm::vec3(0.0f, 0.0f, 0.0f);
    cam_view_ = glm::lookAt(cam_pos_, cam_target_, up_);
    cam_trans_ = cam_proj_ * cam_view_;


    pSlamFrameDrawer_ = pFrameDrawer;
    pSlamMapDrawer_ = pMapDrawer;
 

}


void ImGuiViewer::Run()
{
    // Initialize glfw
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        throw std::runtime_error("[ImGuiViewer]Fails to initialize!");

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    // Create window with graphics context
    GLFWwindow* window =
        glfwCreateWindow(glfw_window_width_, glfw_window_height_,
                         "GSORB-SLAM", nullptr, nullptr);
    if (window == nullptr)
        throw std::runtime_error("[ImGuiViewer]Fails to create window!");
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    glEnable(GL_DEPTH_TEST); // Enable 3D Mouse handler

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Variables for tracking
    cv::Mat Tcw, TcwInit;
    cam_pos_ = glm::vec3(viewpointX_, viewpointY_, viewpointZ_);
    glm::vec4 cam_pos_aligned = glm::vec4(cam_pos_, 1.0f);
    cam_target_ = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 cam_direction = cam_pos_ - cam_target_;
    glm::vec3 cam_right = glm::normalize(glm::cross(up_, cam_direction));
    glm::vec3 cam_up = glm::cross(cam_direction, cam_right);
    glm::vec4 cam_up_aligned = glm::vec4(cam_up, 1.0f);
    cam_view_ = glm::lookAt(cam_pos_, cam_target_, cam_up);
    glm::mat4 glmTwc, Twr, glmTwcInit;
    glmTwc = glm::mat4(1.0f);
    glmTwcInit = glm::mat4(1.0f);
    glmTwc_main_ = glm::mat4(1.0f);
    glm::mat4 Ow, OwInit;
    Ow = glm::mat4(1.0f);
    OwInit = glm::mat4(1.0f);

    // Variables for showing images
    cv::Rect image_rect_sub(0, 0, rendered_image_width_, rendered_image_height_);
    cv::Rect image_rect_main(0, 0, rendered_image_width_main_, rendered_image_height_main_);

    GLuint SLAM_img_texture, rendered_img_texture, main_img_texture;

    glGenTextures(1, &SLAM_img_texture);
    glBindTexture(GL_TEXTURE_2D, SLAM_img_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &rendered_img_texture);
    glBindTexture(GL_TEXTURE_2D, rendered_img_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &main_img_texture);
    glBindTexture(GL_TEXTURE_2D, main_img_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Main loop
    while(!isStopped() && !glfwWindowShouldClose(window))
    {
        //--------------Poll and handle events (inputs, window resize, etc.)--------------
        glfwPollEvents();

        //--------------Start the Dear ImGui frame--------------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //--------------Get pose of current tracked frame--------------

        if(Tcw.empty())
            Tcw = cv::Mat::eye(4,4,CV_32F);
        pSlamMapDrawer_->GetOpenGLCameraMatrix(Tcw, glmTwc, Ow);
        
        
        if (tracking_vision_)
        {
            glm::vec3 cam_target = glm::vec3(Ow[3][0], Ow[3][1], Ow[3][2]);
            cam_pos_aligned = glmTwc * behind_;
            glm::vec3 cam_pos = glm::vec3(cam_pos_aligned.x, cam_pos_aligned.y, cam_pos_aligned.z);
            cam_direction = cam_pos - cam_target;
            cam_up_aligned = glmTwc * up_aligned_;
            cam_up = glm::normalize(glm::vec3(cam_up_aligned.x, cam_up_aligned.y, cam_up_aligned.z) - cam_target);
            cam_right = glm::normalize(glm::cross(cam_up, cam_direction));
            cam_up = glm::cross(cam_direction, cam_right);
            cam_view_ = glm::lookAt(cam_pos, cam_target, cam_up);
            cam_trans_ = cam_proj_ * cam_view_;
        }
        else
        {

            Tcw_main_ = Converter::Glm2Eigen4f(glmTwc_main_).inverse();
            handleUserInput();
            glmTwc_main_ = Converter::Eigen2Glm4f(Tcw_main_.inverse().matrix());
            cam_target_ = glm::vec3(glmTwc_main_[3][0], glmTwc_main_[3][1], glmTwc_main_[3][2]);
            cam_pos_aligned = glmTwc_main_ * behind_;

            cam_pos_ = glm::vec3(cam_pos_aligned.x, cam_pos_aligned.y, cam_pos_aligned.z);
            cam_direction = cam_pos_ - cam_target_;
            cam_up_aligned = glmTwc_main_ * up_aligned_;
            cam_up = glm::normalize(glm::vec3(cam_up_aligned.x, cam_up_aligned.y, cam_up_aligned.z) - cam_target_);
            cam_right = glm::normalize(glm::cross(cam_up, cam_direction));
            cam_up = glm::cross(cam_direction, cam_right);
            cam_view_ = glm::lookAt(cam_pos_, cam_target_, cam_up);
            cam_trans_ = cam_proj_ * cam_view_;
        }


            //--------------Draw SLAM frame image--------------
            cv::Mat SLAM_img_to_show;
            cv::Mat SLAM_img_with_text = pSlamFrameDrawer_->DrawFrame();
            if (SLAM_image_viewer_scale_ != 1.0f)
            {
                int width = rendered_image_width_;
                int height = static_cast<int>(SLAM_img_with_text.rows * SLAM_image_viewer_scale_);
                cv::resize(SLAM_img_with_text, SLAM_img_with_text, cv::Size(width, height));
                SLAM_img_to_show = cv::Mat(height, padded_sub_image_width_, CV_8UC3, cv::Vec3f(0, 0, 0));
            }
            else
            {
                SLAM_img_to_show = cv::Mat(image_height_, padded_sub_image_width_, CV_8UC3, cv::Vec3f(0, 0, 0));
            }
            cv::Rect SLAM_image_rect(0, 0, SLAM_img_with_text.cols, SLAM_img_with_text.rows);
            SLAM_img_with_text.copyTo(SLAM_img_to_show(SLAM_image_rect));
            // Upload SLAM frame
            glBindTexture(GL_TEXTURE_2D, SLAM_img_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SLAM_img_to_show.cols, SLAM_img_to_show.rows,
                        0, GL_BGR, GL_UNSIGNED_BYTE, (unsigned char*)SLAM_img_to_show.data);
            // Create an ImGui window to show the SLAM frame
            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(rendered_image_width_ + 12, SLAM_img_to_show.rows + 40), ImGuiCond_Once);
            {
                ImGui::Begin("SLAM Frame");
                ImGui::Image((void *)(intptr_t)SLAM_img_texture,
                            ImVec2(SLAM_img_to_show.cols, SLAM_img_to_show.rows));
                ImGui::End();
            }

        

        //--------------Draw main window image--------------
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        // Draw main window image
        if (show_main_rendered_)
        {
            auto drawlist = ImGui::GetBackgroundDrawList();
            cv::Mat main_img;
            if (tracking_vision_)
            {
                main_img = pSlamMapDrawer_->RenderViewer(
                    Tcw, rendered_image_width_main_, rendered_image_height_main_);
            }
            else
            {

                 main_img = pSlamMapDrawer_->RenderViewer(
                    Converter::toCvMat(Tcw_main_), rendered_image_width_main_, rendered_image_height_main_);
            }
            cv::Mat main_img_to_show = cv::Mat(rendered_image_height_main_, padded_main_image_width_, CV_8UC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
                if(!main_img.empty()) 
                {
                    // main_img = main_img_to_show;
                    cvtColor(main_img,main_img,cv::COLOR_RGB2BGR);
                    main_img.copyTo(main_img_to_show(image_rect_main));
                }

                glBindTexture(GL_TEXTURE_2D, main_img_texture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, main_img_to_show.cols, main_img_to_show.rows,
                     0, GL_RGB, GL_UNSIGNED_BYTE, (unsigned char*)main_img_to_show.data);
                drawlist->AddImage((void *)(intptr_t)main_img_texture, ImVec2(0, 0),
                                   ImVec2(glfw_window_width_, glfw_window_height_));
        }
       
        
        //--------------Display mode panel--------------
        ImGui::SetNextWindowPos(ImVec2(glfw_window_width_ - panel_width_, 0), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(panel_width_, display_panel_height_), ImGuiCond_Once);
        {
            ImGui::Begin("Display Mode");

            if (training_)
            {
                ImGui::Checkbox("Tracking vision", &tracking_vision_);
                ImGui::Checkbox("Show KeyFrames", &show_keyframes_);
                ImGui::Checkbox("Show sparse MapPoints", &show_sparse_mappoints_);

            }
            

            ImGui::Checkbox("Show main window rendered", &show_main_rendered_);

            // ImGui::Text("Viewer average FPS %.1f", io.Framerate);
            ImGui::End();
        }

        //--------------Camera view panel--------------
        ImGui::SetNextWindowPos(ImVec2(glfw_window_width_ - panel_width_, (training_ ? display_panel_height_ + training_panel_height_ + 16 : display_panel_height_ + 8)), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(panel_width_, camera_panel_height_), ImGuiCond_Once);
        {
            ImGui::Begin("Camera View Velocity");

            ImGui::SliderFloat("Mouse Left", &mouse_left_sensitivity_, 0.01f, 1.0f, "%.2f");
            ImGui::SliderFloat("Mouse Right", &mouse_right_sensitivity_, 0.01f, 1.0f, "%.2f");
            ImGui::SliderFloat("Mouse Middle", &mouse_middle_sensitivity_, 0.01f, 1.0f, "%.2f");
            ImGui::SliderFloat("Keyboard t", &keyboard_velocity_, 0.01f, 1.0f, "%.2f");
            ImGui::SliderFloat("Keyboard R", &keyboard_anglular_velocity_, 0.01f, 1.0f, "%.2f");

            ImGui::End();
        }

        //--------------ImGui Rendering--------------
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        //--------------Draw main window SLAM--------------
        // Set relative viewpoint
        glPushMatrix();
        glMultMatrixf(&cam_trans_[0][0]);
        // Draw camera, KeyFrames and MapPoints
        if (show_keyframes_)
        {
            pSlamMapDrawer_->DrawCurrentCamera(tracking_vision_ ? glmTwc : glmTwc_main_);
            pSlamMapDrawer_->DrawKeyFrames(true, false);
        }
        if(show_sparse_mappoints_)
        {
            pSlamMapDrawer_->DrawMapPoints();
        }
       
        
        // Clear relative viewpoint
        glPopMatrix();

        glfwSwapBuffers(window);
        glfwPollEvents();

        if (requested_stop_)
            break;
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    SetStop();
    

}

bool ImGuiViewer::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}
void ImGuiViewer::RequestStop()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    requested_stop_ = true;
}
void ImGuiViewer::SetStop()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    stopped_ = true;
}

/**
 * We modify Twc_main_ then Tcw_main_ (Sophus::SE3f) to handle mouse and keyboard inputs
 */
void ImGuiViewer::handleUserInput()
{
    if (tracking_vision_)
    {
        free_view_enabled_ = false;
        return;
    }
    else
    {
        free_view_enabled_ = true;
    }

    Twc_main_ = Tcw_main_.inverse();

    // Only respond to mouse inputs when not interacting with ImGui
    if (!ImGui::IsAnyItemActive() && !ImGui::GetIO().WantCaptureMouse)
    {
        mouseWheel();
        mouseDrag();
    }

    // Respond to keyboard inputs
    keyboardEvent();

    Tcw_main_ = Twc_main_.inverse();
}

void ImGuiViewer::mouseWheel()
{
    float delta = ImGui::GetIO().MouseWheel;

    if (delta == 0)
        return;

    float scale_factor = std::pow(1.1f, -delta);

    // float mouse_x = ImGui::GetMousePos().x;
    // float mouse_y = ImGui::GetMousePos().y;

    // float focus3D_x = (mouse_x - main_cx_) / main_fx_;
    // float focus3D_y = (mouse_y - main_cy_) / main_fy_;

    //---Translation---
    // Eigen::Vector3f translating = Eigen::Vector3f::Zero();
    // translating.x() += focus3D_x * (1 - scale_factor);
    // translating.y() += focus3D_y * (1 - scale_factor);
    //-------------

    //---Apply---
    Eigen::Matrix3f R = Twc_main_.block<3,3>(0,0);
    Twc_main_.block<3,1>(0,3) *= scale_factor;
    // Twc_main_.translation() += (R * translating);
}

void ImGuiViewer::mouseDrag()
{
    float delta_rel_x = ImGui::GetIO().MouseDelta.x / glfw_window_width_;
    float delta_rel_y = ImGui::GetIO().MouseDelta.y / glfw_window_height_;
    float delta_l = delta_rel_x * delta_rel_x + delta_rel_y * delta_rel_y;

    //---Rotation---
    Eigen::Vector3f eulars = Eigen::Vector3f::Zero();
    // Left held
    if (ImGui::GetIO().MouseDown[0])
    {
        // Upward (delta_y < 0): pitch upward (Rx+)
        eulars.x() -= M_PI * delta_rel_y;
        // Leftward (delta_x < 0): yaw leftward (Ry-)
        eulars.y() += M_PI * delta_rel_x;
        // Angular Velocity
        eulars.x() *= mouse_left_sensitivity_;
        eulars.y() *= mouse_left_sensitivity_;
    }

    // Right held
    if (ImGui::GetIO().MouseDown[1])
    {
        // Leftward (delta_x < 0): roll counterclockwise (Rz-)
        eulars.z() += M_PI * (delta_rel_x < 0.0f ? -delta_l : delta_l);
        // Angular Velocity
        eulars.z() *= mouse_right_sensitivity_;
    }
    // To rotation matrix
    Eigen::AngleAxisf roll_angle(eulars.z(), Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yaw_angle(eulars.y(), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitch_angle(eulars.x(), Eigen::Vector3f::UnitX());
    Eigen::Quaternion<float> q = roll_angle * yaw_angle * pitch_angle;
    Eigen::Matrix3f rotating = q.matrix();
    //-------------

    //---Translation---
    // Middle held
    Eigen::Vector3f translating = Eigen::Vector3f::Zero();
    if (ImGui::GetIO().MouseDown[2])
    {
        translating.x() += delta_rel_x;
        translating.y() -= delta_rel_y;
        // Velocity
        translating *= mouse_middle_sensitivity_;
    }
    //-------------

    //---Apply---
    Eigen::Matrix3f R = Twc_main_.block<3,3>(0,0);
    Twc_main_.block<3,1>(0,3) += (R * translating);
    Twc_main_.block<3,3>(0,0) = (R * rotating);
}

void ImGuiViewer::keyboardEvent()
{
    if (ImGui::GetIO().WantCaptureKeyboard)
        return;

    // R: Reset camera view to init
    if (ImGui::IsKeyPressed(ImGuiKey_R))
        reset_main_to_init_ = true;

    //---Translation---
    Eigen::Vector3f translating = Eigen::Vector3f::Zero();
    // W: forward
    if (ImGui::IsKeyDown(ImGuiKey_W))
        translating.z() += 1.0f;
    // S: backward
    if (ImGui::IsKeyDown(ImGuiKey_S))
        translating.z() -= 1.0f;
    // A: leftward
    if (ImGui::IsKeyDown(ImGuiKey_A))
        translating.x() -= 1.0f;
    // D: rightward
    if (ImGui::IsKeyDown(ImGuiKey_D))
        translating.x() += 1.0f;
    // Velocity
    translating *= keyboard_velocity_;
    //-------------

    //---Rotation---
    Eigen::Vector3f eulars = Eigen::Vector3f::Zero();
    // I: pitch upward (Rx+)
    if (ImGui::IsKeyDown(ImGuiKey_I))
        eulars.x() += M_PI;
    // K: pitch downward (Rx-)
    if (ImGui::IsKeyDown(ImGuiKey_K))
        eulars.x() -= M_PI;
    // J: yaw leftward (Ry-)
    if (ImGui::IsKeyDown(ImGuiKey_J))
        eulars.y() -= M_PI;
    // L: yaw rightward (Ry+)
    if (ImGui::IsKeyDown(ImGuiKey_L))
        eulars.y() += M_PI;
    // U: roll counterclockwise (Rz-)
    if (ImGui::IsKeyDown(ImGuiKey_U))
        eulars.z() -= M_PI;
    // O: roll clockwise (Rz+)
    if (ImGui::IsKeyDown(ImGuiKey_O))
        eulars.z() += M_PI;
    // Angular Velocity
    eulars *= keyboard_anglular_velocity_;
    // To rotation matrix
    Eigen::AngleAxisf roll_angle(eulars.z(), Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yaw_angle(eulars.y(), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitch_angle(eulars.x(), Eigen::Vector3f::UnitX());
    Eigen::Quaternion<float> q = roll_angle * yaw_angle * pitch_angle;
    Eigen::Matrix3f rotating = q.matrix();
    //--------------

    //---Apply---
    Eigen::Matrix3f R = Twc_main_.block<3,3>(0,0);
    Twc_main_.block<3,1>(0,3) += (R * translating);
    Twc_main_.block<3,3>(0,0) = (R * rotating);
}

}//ORBSLAM2