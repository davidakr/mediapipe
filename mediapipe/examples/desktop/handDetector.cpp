#include "handDetector.h"
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

handDetector::handDetector()
{
    assignValues();
}

::mediapipe::Status handDetector::assignValues()
{
    std::string calculator_graph_config_contents;
    mediapipe::file::GetContents(
        "mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt", &calculator_graph_config_contents);
    LOG(INFO) << "Get calculator graph config contents: "
              << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    LOG(INFO) << "Initialize the calculator graph.";
    graph.Initialize(config);

    LOG(INFO) << "Initialize the GPU.";
    ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
    graph.SetGpuResources(std::move(gpu_resources));
    gpu_helper.InitializeForTest(graph.GetGpuResources().get());

    LOG(INFO) << "Start running the calculator graph.";
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerPresence_, graph.AddOutputStreamPoller(presenceOutputStream));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerVideo_, graph.AddOutputStreamPoller(videoOutputStream));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerLandmark_, graph.AddOutputStreamPoller(landmarkOutputStream));

    //pollerPresence = pollerPresence_;
    //pollerVideo = &pollerVideo_;
    //pollerLandmark = &pollerLandmark_;

    graph.StartRun({});

    LOG(INFO) << "Ready to process frames.";
}

::mediapipe::Status handDetector::insertFrame()
{
    cv::Mat camera_frame_raw;
    if (camera_frame_raw.empty())
    {
        LOG(INFO) << "End of video";
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Prepare and add graph input packet.
    auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
    auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
    glFlush();
    texture.Release();
    // Send GPU image packet into the graph.
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(gpu_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp++))));

    /*// Process video.
    mediapipe::Packet packet;
    if (!pollerVideo.Next(&packet))
    {
        //break;
    }
    std::unique_ptr<mediapipe::ImageFrame> output_frame;

    // Convert GpuBuffer to ImageFrame.
    auto &gpu_frame = packet.Get<mediapipe::GpuBuffer>();
    auto texture = gpu_helper.CreateSourceTexture(gpu_frame);
    output_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
        gpu_frame.width(), gpu_frame.height(),
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    gpu_helper.BindFramebuffer(texture);
    const auto info =
        mediapipe::GlTextureInfoForGpuBufferFormat(gpu_frame.format(), 0);
    glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                 info.gl_type, output_frame->MutablePixelData());
    glFlush();
    texture.Release();*/
}

bool handDetector::getPresence()
{
    return handPresent;
}

cv::Mat handDetector::getImage()
{
    return imageOutput;
}

void handDetector::shutdown()
{
}