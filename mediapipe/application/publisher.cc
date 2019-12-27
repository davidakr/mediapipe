#include "ros/ros.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/String.h"
#include "franka_msgs/FrankaState.h"
#include "handTracker.h"

#include <sstream>

double x;
double y;
double z;

/*void stateCallback(const franka_msgs::FrankaState &state_sub)
{
  auto pose = state_sub.O_T_EE.data();
  x = pose[12];
  y = pose[13];
  z = pose[14];
  std::cout << "state update" << std::endl;
}*/

int main(int argc, char **argv)
{
  cv::VideoCapture camera(0);
  cv::Mat frame;

  cv::namedWindow("Webcam", CV_WINDOW_AUTOSIZE);
  handTracker tracker = handTracker();
  while (true)
  {
    camera >> frame;
    tracker.processImage(frame);
    cv::Mat mat = tracker.getImage();
    cv::imshow("Webcam", mat);
    cv::waitKey(10);
  }
  /*ros::init(argc, argv, "publisher_pose");
  ros::NodeHandle n;
  ros::Publisher pose_pub = n.advertise<std_msgs::Float64MultiArray>("/cartesian_position_velocity_controller/command_cartesian_position", 10);
  ros::Subscriber state_sub = n.subscribe("/franka_state_controller/franka_states", 10, stateCallback);

  std::cout << "node started" << std::endl;

  ros::Rate loop_rate(10);
  int count = 0;
  while (ros::ok())
  {
    std_msgs::Float64MultiArray msg;
    msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
    msg.layout.dim[0].label = "position";
    msg.layout.dim[0].size = 3;

    std::vector<_Float64> vec;

    vec.push_back(0.5);
    vec.push_back(0);
    vec.push_back(0.5);

    msg.data = vec;

    pose_pub.publish(msg);

    ros::spinOnce();

    loop_rate.sleep();
    ++count;
  }*/

  return 0;
}