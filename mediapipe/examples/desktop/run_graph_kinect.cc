// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.

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

#include "kinectCamera.h"

constexpr char kInputStream[] = "input_video";
constexpr char videoOutputStream[] = "output_video";
constexpr char landmarkOutputStream[] = "hand_landmarks";
constexpr char presenceOutputStream[] = "hand_presence";

constexpr char kWindowName[] = "MediaPipe";

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");

void showMat(cv::Mat mat, std::string windowName)
{
  cv::namedWindow(windowName, /*flags=WINDOW_AUTOSIZE*/ 1);
  cv::imshow(windowName, mat);
  cv::waitKey(1);
}

cv::Mat convertDepthImage(cv::Mat mat)
{
  double min;
  double max;
  cv::minMaxIdx(mat, &min, &max);
  cv::Mat adjMap;
  cv::convertScaleAbs(mat, adjMap, 255 / max);
  return adjMap;
}

::mediapipe::Status RunMPPGraph()
{
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  mediapipe::GlCalculatorHelper gpu_helper;
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());

  LOG(INFO) << "Initialize the kinect.";
  kinectCamera kinect = kinectCamera();

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerPresence, graph.AddOutputStreamPoller(presenceOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerVideo, graph.AddOutputStreamPoller(videoOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerLandmark, graph.AddOutputStreamPoller(landmarkOutputStream));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  size_t frame_timestamp = 0;
  bool grab_frames = true;
  while (grab_frames)
  {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    cv::Mat depth_frame_raw;
    cv::Mat depth_frame_transformed;
    kinect.captureFrame();
    camera_frame_raw = kinect.getColorImage();
    depth_frame_raw = kinect.getDepthImage();
    //depth_frame_transformed = kinect.convertPerspectiveDepthToColor();
    kinect.releaseFrame();

    showMat(convertDepthImage(depth_frame_raw), "depth");

    if (camera_frame_raw.empty())
    {
      LOG(INFO) << "End of video";
      break; // End of video.
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
    MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext([&input_frame, &frame_timestamp, &graph,
                                   &gpu_helper]() -> ::mediapipe::Status {
          // Convert ImageFrame to GpuBuffer.
          auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
          auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          texture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(gpu_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp++))));
          return ::mediapipe::OkStatus();
        }));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!pollerVideo.Next(&packet))
      break;
    std::unique_ptr<mediapipe::ImageFrame> output_frame;

    // Convert GpuBuffer to ImageFrame.
    MP_RETURN_IF_ERROR(gpu_helper.RunInGlContext(
        [&packet, &output_frame, &gpu_helper]() -> ::mediapipe::Status {
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
          texture.Release();
          return ::mediapipe::OkStatus();
        }));

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

    // Process landmarks
    if (!pollerLandmark.Next(&packet))
    {
      break;
    }
    LOG(INFO) << "Landmark received.";
    auto landmark_frame = packet.Get<std::vector<mediapipe::NormalizedLandmark, std::allocator<mediapipe::NormalizedLandmark>>>();

    //Process Hand Presence
    if (!pollerPresence.Next(&packet))
    {
      break;
    }
    LOG(INFO) << "Presence received.";
    auto presence_frame = packet.Get<bool>();

    showMat(output_frame_mat, "color");

    // Press any key to exit.
    const int pressed_key = cv::waitKey(5);
    if (pressed_key >= 0 && pressed_key != 255)
      grab_frames = false;
  }

  LOG(INFO) << "Shutting down.";

  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok())
  {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
  }
  else
  {
    LOG(INFO) << "Success!";
  }
  return 0;
}
