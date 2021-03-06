#include "kinectCamera.h"

kinectCamera::kinectCamera()
{
    // Open the first plugged in Kinect device
    k4a_device_open(K4A_DEVICE_DEFAULT, &device);

    // Get the size of the serial number
    k4a_device_get_serialnum(device, NULL, &serial_size);

    // Allocate memory for the serial, then acquire it
    serial = (char *)(malloc(serial_size));
    k4a_device_get_serialnum(device, serial, &serial_size);
    printf("Opened device: %s\n", serial);
    free(serial);

    // Configure a stream of 4096x3072 BRGA color data at 15 frames per second
    config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    config.color_resolution = K4A_COLOR_RESOLUTION_720P;
    config.color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
    config.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
    config.camera_fps = K4A_FRAMES_PER_SECOND_5;
    config.synchronized_images_only = true;

    // Start the camera with the given configuration
    if (K4A_FAILED(k4a_device_start_cameras(device, &config)))
    {
        printf("Failed to start cameras!\n");
        k4a_device_close(device);
    }

    // Get configuration
    k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, &calibration);
}

bool kinectCamera::captureFrame()
{
    switch (k4a_device_get_capture(device, &capture, TIMEOUT_IN_MS))
    {
    case K4A_WAIT_RESULT_SUCCEEDED:
        return true;
    case K4A_WAIT_RESULT_TIMEOUT:
        printf("Timed out waiting for a capture\n");
        return false;
    case K4A_WAIT_RESULT_FAILED:
        printf("Failed to read a capture\n");
        return false;
    }
    return false;
}

void kinectCamera::shutdown()
{
    // Shut down the camera when finished with application logic
    printf("Camera shutwdown\n");
    k4a_device_stop_cameras(device);
    k4a_device_close(device);
}

cv::Mat kinectCamera::getColorImage()
{
    color_image = k4a_capture_get_color_image(capture);
    cv::Mat mat;
    if (color_image != NULL)
    {
        auto height = k4a_image_get_height_pixels(color_image);
        auto width = k4a_image_get_width_pixels(color_image);
        auto buffer = k4a_image_get_buffer(color_image);
        cv::Mat imgbuf = cv::Mat(height, width, CV_8UC1, (void *)buffer, cv::Mat::AUTO_STEP);
        mat = cv::imdecode(imgbuf, CV_LOAD_IMAGE_COLOR);
    }
    else
    {
        printf("Color image is empty\n");
    }
    return mat;
}

cv::Mat kinectCamera::getDepthImage()
{
    depth_image = k4a_capture_get_depth_image(capture);
    cv::Mat mat;
    if (depth_image != NULL)
    {
        auto height = k4a_image_get_height_pixels(depth_image);
        auto width = k4a_image_get_width_pixels(depth_image);
        auto buffer = k4a_image_get_buffer(depth_image);
        mat = cv::Mat(height, width, CV_16U, (void *)buffer, cv::Mat::AUTO_STEP);
    }
    else
    {
        printf("Depth image is empty\n");
    }
    return mat;
}

void kinectCamera::releaseMemory()
{
    k4a_image_release(color_image);
    k4a_image_release(depth_image);
    k4a_capture_release(capture);
}

cv::Point3f kinectCamera::convertTo3D(cv::Point2f point2D)
{
    k4a_float2_t color2D;
    k4a_float2_t depth2D;
    k4a_float3_t color3D;
    cv::Point3f point3D = cv::Point3f(0, 0, 0);

    int valid = false;

    color2D.xy.x = (int)point2D.x;
    color2D.xy.y = (int)point2D.y;
    if (depth_image == NULL)
    {
        getDepthImage();
    }

    uint16_t depth = 0;

    //Get Depth Value at position
    k4a_calibration_color_2d_to_depth_2d(&calibration, &color2D, depth_image, &depth2D, &valid);
    if (valid)
    {
        uint16_t *buffer = (uint16_t *)(void *)k4a_image_get_buffer(depth_image);
        auto height = k4a_image_get_height_pixels(depth_image);
        auto width = k4a_image_get_width_pixels(depth_image);
        cv::Mat mat = cv::Mat(height, width, CV_16U, (void *)buffer, cv::Mat::AUTO_STEP);
        depth = mat.at<ushort>(depth2D.xy.y, depth2D.xy.x);
    }

    //Get 3D point in color camera coordinates. Input pixel from color camera. Assigns previous assigned depth.
    k4a_calibration_2d_to_3d(&calibration, &color2D, depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR, &color3D, &valid);
    if (valid)
    {
        point3D.x = color3D.xyz.x;
        point3D.y = color3D.xyz.y;
        point3D.z = color3D.xyz.z;
    }
    return point3D;
}
