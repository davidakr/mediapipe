#include "kinectCamera.h"
#include "k4a/k4a.h"

#include <stdio.h>
#include <stdlib.h>

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
    config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
    config.camera_fps = K4A_FRAMES_PER_SECOND_15;
    config.synchronized_images_only = true;

    // Start the camera with the given configuration
    if (K4A_FAILED(k4a_device_start_cameras(device, &config)))
    {
        printf("Failed to start cameras!\n");
        k4a_device_close(device);
    }
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

void kinectCamera::releaseFrame()
{
    k4a_capture_release(capture);
}

cv::Mat kinectCamera::convertPerspectiveDepthToColor()
{
    k4a_calibration_t calibration;
    k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, &calibration);
    k4a_image_t transformed_depth_image = NULL;
    int color_image_width_pixels = k4a_image_get_width_pixels(color_image);
    int color_image_height_pixels = k4a_image_get_height_pixels(color_image);
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
                                                 color_image_width_pixels,
                                                 color_image_height_pixels,
                                                 color_image_width_pixels * (int)sizeof(uint16_t),
                                                 &transformed_depth_image))
    {
        printf("Failed to create transformed depth image\n");
    }

    auto result = k4a_transformation_depth_image_to_color_camera(k4a_transformation_create(&calibration), depth_image, transformed_depth_image);
    if (K4A_RESULT_SUCCEEDED != result)
    {
        printf("unsucessfull\n");
    }

    auto height = k4a_image_get_height_pixels(transformed_depth_image);
    auto width = k4a_image_get_width_pixels(transformed_depth_image);
    auto buffer = k4a_image_get_buffer(transformed_depth_image);
    k4a_image_release(transformed_depth_image);

    uint8_t *pixel = buffer;
    pixel += (height / 2 + width / 2)*4;
    float r = pixel[3];
    std::cout << r << std::endl;

    /*for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j, pixel += 4)
        {
            float r = pixel[3];
            std::cout << r << std::endl;
        }
    }*/
    return cv::Mat(height, width, CV_16U, (void *)buffer, cv::Mat::AUTO_STEP);
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

void kinectCamera::releaseImage()
{
    k4a_image_release(color_image);
    k4a_image_release(depth_image);
}
