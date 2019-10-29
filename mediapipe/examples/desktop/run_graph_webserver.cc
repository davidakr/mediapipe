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

#include <pistache/endpoint.h>
#include <pistache/net.h>
#include <pistache/http.h>
#include <pistache/client.h>

#include "base64.h"
#include "include/rapidjson/writer.h"
#include "include/rapidjson/stringbuffer.h"

#include <pthread.h>
#include <chrono>
#include <mutex>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

constexpr char kInputStream[] = "input_video";
constexpr char videoOutputStream[] = "output_video";
constexpr char landmarkOutputStream[] = "hand_landmarks";
constexpr char rectOutputStream[] = "hand_rect";
constexpr char palmOutputStream[] = "palm_detections";
constexpr char presenceOutputStream[] = "hand_presence";

constexpr char kWindowName[] = "MediaPipe";

using namespace Pistache;
using namespace std;
using namespace rapidjson;

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");

bool run_graph = true;
bool process_image = false;
bool new_data = false;
bool process_done = false;
string json;

cv::Mat new_image;
std::mutex mtx;
std::mutex mtx_json;

queue<cv::Mat> image_queue_input;

string boolToString(bool b)
{
  return b ? "true" : "false";
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
void graphUnsucessful()
{
  mtx.lock();
  process_done = false;
  process_image = false;
  mtx.unlock();
  LOG(INFO) << "Graph failed";
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

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerPresence, graph.AddOutputStreamPoller(presenceOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerVideo, graph.AddOutputStreamPoller(videoOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerLandmark, graph.AddOutputStreamPoller(landmarkOutputStream));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  size_t frame_timestamp = 0;

  cv::Mat image_frame;

  while (run_graph)
  {
    mtx.lock();
    if (new_data)
    {
      LOG(INFO) << "Block request.";
      image_frame = new_image;
      process_image = true;
      new_data = false;
    }
    mtx.unlock();

    while (process_image)
    {
      // Wrap Mat into an ImageFrame.
      auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
          mediapipe::ImageFormat::SRGB, image_frame.cols, image_frame.rows,
          mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
      cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
      image_frame.copyTo(input_frame_mat);
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

      //prepare JSON
      StringBuffer s;
      Writer<StringBuffer> writer(s);
      writer.StartObject();

      // Get the graph result packet, or stop if that fails.
      mediapipe::Packet packet;

      if (!pollerVideo.Next(&packet))
      {
        graphUnsucessful();
        break;
      }

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
      writer.Key("image"); // output a key,
      string base64img = imgToBase64(output_frame_mat);
      char *cstr = new char[base64img.length() + 1];
      strcpy(cstr, base64img.c_str());
      writer.String(cstr);
      //cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

      // Process landmarks
      if (!pollerLandmark.Next(&packet))
      {
        graphUnsucessful();
        break;
      }
      LOG(INFO) << "Landmark received.";
      auto landmark_frame = packet.Get<std::vector<mediapipe::NormalizedLandmark, std::allocator<mediapipe::NormalizedLandmark>>>();
      writer.Key("landmarks");
      writer.StartArray();
      for (auto i = landmark_frame.begin(); i != landmark_frame.end(); ++i)
      {
        writer.StartArray();
        mediapipe::NormalizedLandmark landmark = *i;
        writer.Uint(landmark.x());
        writer.Uint(landmark.y());
        writer.EndArray();
      }
      writer.EndArray();

      //Process Hand Presence
      if (!pollerPresence.Next(&packet))
      {
        graphUnsucessful();
        break;
      }
      LOG(INFO) << "Presence received.";
      auto presence_frame = packet.Get<bool>();
      writer.Key("present");
      writer.Bool(presence_frame);

      writer.EndObject();

      mtx.lock();
      LOG(INFO) << "Release request.";
      json = s.GetString();
      process_image = false;
      process_done = true;
      mtx.unlock();

      /*cv::namedWindow(kWindowName, 1);
      cv::imshow(kWindowName, output_frame_mat);
      cv::waitKey(5);*/
    }
  }

  LOG(INFO) << "Shutting down.";
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

class InputHandler : public Http::Handler
{
  HTTP_PROTOTYPE(InputHandler)
  void onRequest(const Http::Request &request, Http::ResponseWriter writer) override
  {
    if (request.resource() == "/" && request.method() == Http::Method::Post)
    {
      LOG(INFO) << "---------------------------------------New Request---------------------------------------";
      string body_str = request.body();

      auto start = std::chrono::system_clock::now();
      if (body_str.empty())
      {
        writer.send(Http::Code::No_Content, "body empty");
        return;
      }
      cv::Mat buffer_image;
      buffer_image = base64ToImg(body_str);

      if (!(buffer_image.cols > 0 && buffer_image.rows > 0))
      {
        writer.send(Http::Code::No_Content, "no image");
        return;
      }
      auto end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds_datacheck = end - start;

      start = std::chrono::system_clock::now();
      mtx.lock();
      new_image = buffer_image;
      new_data = true;
      mtx.unlock();

      bool graph_is_running = true;
      string jsonResponse;

      while (graph_is_running)
      {
        mtx.lock();
        graph_is_running = process_image;
        if (!graph_is_running)
        {
          jsonResponse = json;
        }
        mtx.unlock();
      }
      end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds_graph = end - start;
      LOG(INFO) << "Datacheck:" << elapsed_seconds_datacheck.count() << " seconds";
      LOG(INFO) << "Running graph: " << elapsed_seconds_graph.count() << " seconds";

      writer.send(Http::Code::Ok, jsonResponse);
    }
  }
};

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::thread threadGraph(RunMPPGraph);

  Port port(argv[1]);
  Address addr(Ipv4::any(), port);
  auto server = std::make_shared<Http::Endpoint>(addr);
  auto opts_server = Http::Endpoint::options()
                         .maxRequestSize(2538400);

  server->init(opts_server);
  server->setHandler(Http::make_handler<InputHandler>());
  server->serve();

  threadGraph.join();

  //cout << RunMPPGraph() << endl;
  return 0;
}