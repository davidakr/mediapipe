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

#include <pistache/endpoint.h>
#include <pistache/net.h>
#include <pistache/http.h>
#include <pistache/client.h>
#include "base64.h"

#include <pthread.h>
#include <chrono>
#include <queue>
#include <mutex>
#include <stdint.h>
#include <stdlib.h>

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

using namespace Pistache;
using namespace std;

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");

cv::Mat image = cv::imread("mediapipe/examples/desktop/data/rsz_test_img.jpg", CV_LOAD_IMAGE_COLOR);
bool run_graph = true;
bool grab_frames = false;
bool test_variable = false;
bool process_image = false;
mutex mtx_queue;
queue<cv::Mat> image_queue;

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

  //cv::VideoWriter writer;
  //cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller, graph.AddOutputStreamPoller(kOutputStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  size_t frame_timestamp = 0;

  cv::Mat camera_frame;
  camera_frame = image;

  //if (image.empty()){ return mediapipe::Status::; };
  while (run_graph)
  {
    mtx_queue.lock();
    process_image = !image_queue.empty();
    
    mtx_queue.unlock();

    while (process_image)
    {
      camera_frame = image_queue.front();
      image_queue.pop();
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
      if (!poller.Next(&packet))
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
      cv::imwrite("mediapipe/examples/desktop/data/test_img_check.jpg", output_frame_mat);
      //cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

      mtx_queue.lock();
      process_image = !image_queue.empty();
      mtx_queue.unlock();
    }
  }

  LOG(INFO) << "Shutting down.";
  //if (writer.isOpened())
  //  writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

string prepareString(bool to_send)
{
  if (to_send)
  {
    return "true";
  }
  else
  {
    return "false";
  }
}

cv::Mat base64ToImg(string base64)
{
  string dec_jpg = base64_decode(base64);
  std::vector<uchar> data(dec_jpg.begin(), dec_jpg.end());
  cv::Mat img = cv::imdecode(data, cv::IMREAD_UNCHANGED);
  return img;
}

string imgToBase64(cv::Mat img)
{
  std::vector<uchar> buf;
  cv::imencode(".jpg", img, buf);
  auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
  std::string encoded = base64_encode(enc_msg, buf.size());
  return encoded;
}

string getValueFromBody(string body, string value)
{
  body = "&" + body;
  auto position = body.find("&" + value);
  auto position_cut = body.find("&", position + 1);
  auto length = position_cut - position - value.length() - 2;
  return body.substr(position + value.length() + 2, length);
}

class InputHandler : public Http::Handler
{
  HTTP_PROTOTYPE(InputHandler)
  void onRequest(const Http::Request &request, Http::ResponseWriter writer) override
  {
    mtx_queue.lock();
    if (request.resource() == "/push" && request.method() == Http::Method::Post)
    {
      string body_str = request.body();
      string image_str = getValueFromBody(body_str, "image");
      string timestamp_str = getValueFromBody(body_str, "timestamp");

      cv::Mat image_mat = base64ToImg(image_str);
      image_queue.push(image_mat);
      writer.send(Http::Code::Ok);
    }
    else if (request.resource() == "/status")
    {
      char queue_size[256];
      snprintf(queue_size, sizeof queue_size, "%zu", image_queue.size());
      std::string to_send = "queue empty:" + prepareString(image_queue.empty());
      to_send = to_send + " queue size:" + queue_size;
      to_send = to_send + " run_graph:" + prepareString(run_graph);
      to_send = to_send + " grab_frames:" + prepareString(grab_frames);
      to_send = to_send + " process_image:" + prepareString(process_image);
      writer.send(Http::Code::Ok, to_send);
    } else if(request.resource() == "/getbase64"){
      string base64 = imgToBase64(image);
      writer.send(Http::Code::Ok, base64);
    }
    mtx_queue.unlock();
  }
};

bool sendImage()
{
  Http::Client client;

  auto opts = Http::Client::options()
                  .threads(1)
                  .maxConnectionsPerHost(8);
  client.init(opts);
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::thread threadGraph(RunMPPGraph);

  Port port(9090);
  Address addr(Ipv4::any(), port);
  auto server = std::make_shared<Http::Endpoint>(addr);
  auto opts = Http::Endpoint::options()
                  .maxRequestSize(2538400);

  server->init(opts);
  server->setHandler(Http::make_handler<InputHandler>());
  server->serve();
  threadGraph.join();
  return 0;
}