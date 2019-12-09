#ifndef KINECTCAMERA_H
#define KINECTCAMERA_H

#include "k4a/k4a.h"
#include "opencv2/opencv.hpp"

class kinectCamera
{
private:
    k4a_device_t device;
    size_t serial_size = 0;
    char *serial;
    k4a_capture_t capture;
    const int32_t TIMEOUT_IN_MS = 1000;
    k4a_image_t color_image;
    k4a_image_t depth_image;
    cv::Mat transformed_depth_mat;
    int second;
    k4a_device_configuration_t config;
    k4a_calibration_t calibration;


public:
    kinectCamera();
    bool captureFrame();
    void releaseFrame();
    void releaseImage();
    void shutdown();
    cv::Mat convertPerspectiveDepthToColor();
    cv::Mat getColorImage();
    cv::Mat getDepthImage();
    cv::Point3f convertTo3D(cv::Point2f point);
};

#endif