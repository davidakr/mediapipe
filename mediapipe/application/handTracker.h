#ifndef HANDTRACKER_H
#define HANDTRACKER_H

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/calculators/core/concatenate_vector_calculator.h"

#include <string>
#include <thread>
#include <mutex>

class handTracker
{
private:
    cv::Mat output_mat;
    bool presence;
    std::vector<float> landmarks;
    const int width = 1280;
    const int height = 720;
    char kInputStream[12] = "input_video";
    char videoOutputStream[13] = "output_video";
    char landmarkOutputStream[15] = "hand_landmarks";
    char presenceOutputStream[14] = "hand_presence";
    ::mediapipe::Status RunMPPGraph();
    cv::Mat camera_frame_raw;
    bool processing = false;
    bool newData = false;
    bool grab_frames = true;
    std::thread threadGraph;
    static std::mutex mtx;
    void waitForProcessFinished();

public:
    handTracker();
    void processImage(cv::Mat camera_frame);
    std::vector<float> getLandmarks();
    bool getPresence();
    cv::Mat getImage();
};

#endif