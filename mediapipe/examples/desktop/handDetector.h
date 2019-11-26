#ifndef HANDDETECTOR_H
#define HANDDETECTOR_H

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

#include <stdio.h>
#include <stdlib.h>

class handDetector
{
private:
    static constexpr char kInputStream[] = "input_video";
    static constexpr char videoOutputStream[] = "output_video";
    static constexpr char landmarkOutputStream[] = "hand_landmarks";
    static constexpr char presenceOutputStream[] = "hand_presence";
    mediapipe::CalculatorGraph graph;
    mediapipe::GlCalculatorHelper gpu_helper;
    bool handPresent;
    cv::Mat imageOutput;
    ::mediapipe::Status assignValues();
    ::mediapipe::Status insertFrame();
    size_t frame_timestamp = 0;
    //mediapipe::OutputStreamPoller pollerVideo;
    //mediapipe::OutputStreamPoller pollerPresence = mediapipe::OutputStreamPoller();
    //mediapipe::OutputStreamPoller pollerLandmark;

public:
    handDetector();
    void insertImage();
    cv::Mat getImage();
    bool getPresence();
    void getLandmarks();
    void showImage();
    void shutdown();
};

#endif