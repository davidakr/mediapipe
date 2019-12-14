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

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <franka_msgs/FrankaState.h>

constexpr char kInputStream[] = "input_video";
constexpr char videoOutputStream[] = "output_video";
constexpr char landmarkOutputStream[] = "hand_landmarks";
constexpr char presenceOutputStream[] = "hand_presence";

constexpr char kWindowName[] = "MediaPipe";

const int amoutOfLandmarks = 21;
const int positionsLandmarks = 3;

const int width = 1280;
const int height = 720;

double x;
double y;
double z;

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");

void showImage(cv::Mat mat, std::string name)
{
  cv::namedWindow(name, 1);
  cv::imshow(name, mat);
  cv::waitKey(1);
  return;
}

int checkPositive(int value)
{
  if (value < 0)
    return 0;
  return value;
}

void stateCallback(const franka_msgs::FrankaState &state_sub)
{
  auto pose = state_sub.O_T_EE.data();
  x = pose[12];
  y = pose[13];
  z = pose[14];
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

  LOG(INFO) << "Initialize the ros node.";
  ros::NodeHandle node;
  //ros::Publisher presencePublisher = node.advertise<std_msgs::Bool>("/hand_tracking/presence", 1000);
  //ros::Publisher landmarksPublisher = node.advertise<std_msgs::Float32MultiArray>("/hand_tracking/landmarks", 1);
  ros::Subscriber state_sub = node.subscribe("/franka_state_controller/franka_states", 10, stateCallback);
  ros::Publisher pose_pub = node.advertise<std_msgs::Float64MultiArray>("/cartesian_position_velocity_controller/command_cartesian_position", 10);

  LOG(INFO) << "Start running the calculator graph.";
  //ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerPresence, graph.AddOutputStreamPoller(presenceOutputStream));
  //ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerVideo, graph.AddOutputStreamPoller(videoOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerLandmark, graph.AddOutputStreamPoller(landmarkOutputStream));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  size_t frame_timestamp = 0;
  bool grab_frames = true;
  while (grab_frames)
  {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    auto start_kinect = std::chrono::system_clock::now();
    kinect.captureFrame();
    camera_frame_raw = kinect.getColorImage();
    kinect.getDepthImage();
    double x_current = x;
    double y_current = y;
    double z_current = z;
    auto end_kinect = std::chrono::system_clock::now();

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

    auto start_graph = std::chrono::system_clock::now();

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

    auto end_graph = std::chrono::system_clock::now();

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;

    auto start_image = std::chrono::system_clock::now();

    /*if (!pollerVideo.Next(&packet))
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
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);*/

    auto end_image = std::chrono::system_clock::now();

    //showImage(output_frame_mat, "color");

    auto start_landmark = std::chrono::system_clock::now();
    // Process landmarks
    if (!pollerLandmark.Next(&packet))
    {
      break;
    }
    auto landmark_frame = packet.Get<std::vector<mediapipe::NormalizedLandmark, std::allocator<mediapipe::NormalizedLandmark>>>();

    auto start_landmark_transformation = std::chrono::system_clock::now();

    std_msgs::Float32MultiArray landmarks;
    landmarks.layout.dim.push_back(std_msgs::MultiArrayDimension());
    landmarks.layout.dim.push_back(std_msgs::MultiArrayDimension());
    landmarks.layout.data_offset = 0;
    landmarks.layout.dim[0].label = "landmark";
    landmarks.layout.dim[0].size = amoutOfLandmarks;
    landmarks.layout.dim[0].stride = amoutOfLandmarks * positionsLandmarks;
    landmarks.layout.dim[1].label = "position";
    landmarks.layout.dim[1].size = positionsLandmarks;
    landmarks.layout.dim[1].size = positionsLandmarks;

    // Convert landmarks
    std::vector<float> vec;

    for (auto i = landmark_frame.begin(); i != landmark_frame.end(); ++i)
    {
      mediapipe::NormalizedLandmark landmark = *i;
      float pixelWidth = checkPositive(width * landmark.x());
      float pixelHeigth = checkPositive(height * landmark.y());
      cv::Point2f point2D;
      point2D.x = pixelWidth;
      point2D.y = pixelHeigth;

      if (point2D.x != 0 && point2D.y != 0)
      {
        cv::Point3f point3D = kinect.convertTo3D(point2D);
        vec.push_back(point3D.x);
        vec.push_back(point3D.y);
        vec.push_back(point3D.z);
      }
    }

    landmarks.data = vec;
    //landmarksPublisher.publish(landmarks);
    auto end_landmark = std::chrono::system_clock::now();

    auto start_presence = std::chrono::system_clock::now();
    //Process Hand Presence
    /*if (!pollerPresence.Next(&packet))
    {
      break;
    }
    std_msgs::Bool presence;
    presence.data = packet.Get<bool>();

    presencePublisher.publish(presence);*/
    auto end_presence = std::chrono::system_clock::now();

    //Release camera frames
    kinect.releaseMemory();
    auto end_total = std::chrono::system_clock::now();
    /*std::chrono::duration<double> elapsed_seconds_kinect = end_kinect - start_kinect;
    std::chrono::duration<double> elapsed_seconds_graph = end_graph - start_graph;
    std::chrono::duration<double> elapsed_seconds_landmark = end_landmark - start_landmark;
    std::chrono::duration<double> elapsed_seconds_landmark_transformation = end_landmark - start_landmark_transformation;
    std::chrono::duration<double> elapsed_seconds_presence = end_presence - start_presence;
    std::chrono::duration<double> elapsed_seconds_image = end_image - start_image;
    std::chrono::duration<double> elapsed_seconds_total = end_total - start_kinect;
    std::cout << "total in fps: " << 1 / elapsed_seconds_total.count()
              << "\t kinect: " << elapsed_seconds_kinect.count() / elapsed_seconds_total.count() << "% "
              << "\t graph: " << elapsed_seconds_graph.count() / elapsed_seconds_total.count() << "% "
              << "\t landmarks: " << elapsed_seconds_landmark.count() / elapsed_seconds_total.count() << "% "
              << "\t landmarks transformation: " << elapsed_seconds_landmark_transformation.count() / elapsed_seconds_total.count() << "% " << std::endl;*/

    float avgX = 0;
    float avgY = 0;
    float avgZ = 0;
    int amountLandmarks = 0;
    double xToSend = x_current;
    double yToSend = y_current;
    double zToSend = z_current;

    if (vec.size() > 0)
    {
      for (int i = 0; i < 1; i++)
      {
        if ((vec[i + 2] != 0) && (vec[i + 2] < 500))
        {
          if (vec[i + 2] < 500)
          {
            avgX = avgX + vec[i];
            avgY = avgY + vec[i + 1];
            avgZ = avgZ + vec[i + 2];

            amountLandmarks++;
          }
        }
      }
      avgX = avgX / amountLandmarks;
      avgY = avgY / amountLandmarks;
      avgZ = avgZ / amountLandmarks;
      // Conversion camera coordinates in robot coordinates
      xToSend = x_current - 0.1 - avgY/1000;
      yToSend = y_current - avgX / 1000;
      zToSend = z_current + 0.3 - avgZ / 1000; //meter
      if (isnan(xToSend))
      {
        xToSend = x_current;
      }
      if (isnan(yToSend))
      {
        yToSend = y_current;
      }
      if (isnan(zToSend))
      {
        zToSend = z_current;
      }
      std::cout << "avgx " << avgX << " current y " << y_current << " yToSend " << yToSend << " amoutLandmarks " << amountLandmarks << " first point " << vec[0] << std::endl;
    }

    std_msgs::Float64MultiArray msg;
    msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
    msg.layout.dim[0].label = "position";
    msg.layout.dim[0].size = 3;

    std::vector<_Float64> vecSend;

    vecSend.push_back(xToSend);
    vecSend.push_back(yToSend);
    vecSend.push_back(zToSend);

    msg.data = vecSend;

    pose_pub.publish(msg);

    ros::spinOnce();
  }

  LOG(INFO) << "Shutting down.";

  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ros::init(argc, argv, "hand_tracking_kinect");

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
