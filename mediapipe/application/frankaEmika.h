#ifndef FRANKAEMIKA_H
#define FRANKAEMIKA_H

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <franka_msgs/FrankaState.h>

class frankaEmika
{
private:
    ros::NodeHandle node;
    ros::Subscriber state_sub;
    ros::Publisher pose_pub;
    ros::Publisher gripper_move_pub;
    ros::Publisher gripper_grasp_pub;
    double position [3];
    double currentPosition [3];
    void stateCallback(const franka_msgs::FrankaState &state_sub);

public:
    frankaEmika();
    void setCurrentPosition();
    double *getCurrentPosition();
    void goToPosition(double cartesianPosition[3]);
    void move(double moveCoordinates[3]);
    void moveGriper (double width, double speed);
    void grasp (double width, double epsilonn_inner, double epsilon_outer, double speed, double force);

};

#endif