
#ifndef __VIEWER2_H_
#define __VIEWER2_H_

#include <torch/torch.h>

#include "FrameDrawer.h"
#include "MapDrawer.h"

#include "src/imgui/imgui.h"
#include "src/imgui/imgui_impl_glfw.h"
#include "src/imgui/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Eigen/Core>
#include <mutex>
#include "Utils.h"
#include "Converter.h"
namespace ORB_SLAM2
{
class Tracking;
class FrameDrawer;
class MapDrawer;
class Converter;


class ImGuiViewer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImGuiViewer(FrameDrawer* pFrameDrawer, MapDrawer *pMapDrawer,const string &strSettingPath);

    void Run();

    bool isStopped();
    void RequestStop();
    void SetStop();



protected:
    void handleUserInput();
    void mouseWheel();
    void mouseDrag();
    void keyboardEvent();

protected:

    FrameDrawer* pSlamFrameDrawer_;
    MapDrawer* pSlamMapDrawer_;

    // Status
    bool free_view_enabled_ = true;
    bool init_Twc_set_ = false;
    Eigen::Matrix4f Tcw_main_, Twc_main_;
    glm::mat4 glmTwc_main_;

    // Configurations
    bool training_ = true;

    int panel_width_, display_panel_height_, training_panel_height_, camera_panel_height_;


    float viewpointX_ = 0.0f, viewpointY_ = 0.0f, viewpointZ_ = -1.0f, viewpointF_;
    int image_width_, image_height_;
    int glfw_window_width_, glfw_window_height_;
    float rendered_image_viewer_scale_ = 1.0f;
    int rendered_image_width_, rendered_image_height_;
    int padded_sub_image_width_;
    float rendered_image_viewer_scale_main_ = 1.0f;
    int rendered_image_width_main_, rendered_image_height_main_;
    int padded_main_image_width_;
    float SLAM_image_viewer_scale_;


    float camera_watch_dist_;

    glm::vec3 up_;
    glm::vec4 up_aligned_;
    glm::vec4 behind_;
    glm::vec3 cam_target_, cam_pos_;
    glm::mat4 cam_proj_;
    glm::mat4 cam_view_;
    glm::mat4 cam_trans_;

    float mouse_left_sensitivity_ = 0.05 * M_PI;
    float mouse_right_sensitivity_ = 0.2 * M_PI;
    float mouse_middle_sensitivity_ = 0.2;
    float keyboard_velocity_ = 0.1;
    float keyboard_anglular_velocity_ = 0.05;

    bool reset_main_to_init_ = false;
    bool tracking_vision_ = false;
    bool show_keyframes_ = false;
    bool show_sparse_mappoints_ = false;
    bool show_main_rendered_ = true;
    bool show_octmap_mappoints_=false;
    bool show_anchor_mappoints_=false;

    float position_lr_init_;
    float feature_lr_;
    float opacity_lr_;
    float scaling_lr_;
    float rotation_lr_;
    float percent_dense_;
    float lambda_dssim_;
    int opacity_reset_interval_;
    float densify_grad_th_;
    int densify_interval_;
    int new_kf_times_of_use_;
    int stable_num_iter_existence_; ///< loop closure correction

    bool keep_training_ = false;
    bool do_gaus_pyramid_training_;
    bool do_inactive_geo_densify_;

    // Status
    bool requested_stop_ = false;
    bool stopped_=false;

    // Mutex
    std::mutex mutex_status_;
};




} // namespace ORB_SLAM2


#endif