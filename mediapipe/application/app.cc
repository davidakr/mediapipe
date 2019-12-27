#include "kinectCamera.h"
#include "frankaEmika.h"
#include "handTracker.h"

const int amoutOfLandmarks = 21;
const int positionsLandmarks = 3;

const int width = 1280;
const int height = 720;

double x;
double y;
double z;

void showImage(cv::Mat mat, std::string name)
{
  cv::namedWindow(name, 1);
  cv::imshow(name, mat);
  cv::waitKey(1);
  return;
}

void stateCallback(const franka_msgs::FrankaState &state_sub)
{
  auto pose = state_sub.O_T_EE.data();
  x = pose[12];
  y = pose[13];
  z = pose[14];
} 

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  ros::init(argc, argv, "hand_tracking_kinect");

  LOG(INFO) << "Initialize the kinect.";
  kinectCamera kinect = kinectCamera();

  LOG(INFO) << "Initialize franka.";
  frankaEmika franka = frankaEmika();

  LOG(INFO) << "Initialize Handtracker.";
  handTracker tracker = handTracker();

  while (ros::ok())
  {

    cv::Mat camera_frame_raw;
    //Caputre color and depth frame
    kinect.captureFrame();
    camera_frame_raw = kinect.getColorImage();
    kinect.getDepthImage();

    // Capture current Position
    franka.setCurrentPosition();
    tracker.processImage(camera_frame_raw);

    showImage(tracker.getImage(), "color");

    std::vector<cv::Point3f> landmarks3D; 
    std::vector<cv::Point2f> landmarks2D = tracker.getLandmarks();
    auto current = franka.getCurrentPosition();
    double x_current = current[0];
    double y_current = current[1];
    double z_current = current[2];
    float avgX = 0;
    float avgY = 0;
    float avgZ = 0;
    int amountLandmarks = 0;
    double xToSend = x_current;
    double yToSend = y_current;
    double zToSend = z_current;

    if (landmarks3D.size() > 0)
    {
      for (int i = 0; i < 1; i++) //aktuell nur ein Punkt
      {
        if ((landmarks3D[i].z != 0) && (landmarks3D[i].z < 800))
        {
          avgX = avgX + landmarks3D[i].x;
          avgY = avgY + landmarks3D[i].y;
          avgZ = avgZ + landmarks3D[i].z;
          amountLandmarks++;
        }
      }
      avgX = avgX / amountLandmarks;
      avgY = avgY / amountLandmarks;
      avgZ = avgZ / amountLandmarks;
      // Conversion camera coordinates in robot coordinates
      xToSend = x_current - 0.1 - avgY / 1000;
      yToSend = y_current - avgX / 1000;
      zToSend = z_current + 0.3 - avgZ / 1000; //meter
      //std::cout << "avgz " << avgZ/1000 << " current z " << z_current << " zToSend " << zToSend << " amoutLandmarks " << amountLandmarks << " first point " << vec[2] << std::endl;
    }

    double cartesianPosition[3] = {xToSend, yToSend, zToSend};
    franka.goToPosition(cartesianPosition);

    kinect.releaseMemory();
    ros::spinOnce();
  }

  return 0;
}
