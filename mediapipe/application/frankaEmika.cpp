#include "frankaEmika.h"

void frankaEmika::stateCallback(const franka_msgs::FrankaState &state_sub)
{
    auto pose = state_sub.O_T_EE.data();
    position[0] = pose[12];
    position[1] = pose[13];
    position[2] = pose[14];
}

frankaEmika::frankaEmika()
{
    state_sub = node.subscribe("/franka_state_controller/franka_states", 10, &frankaEmika::stateCallback, this);
    pose_pub = node.advertise<std_msgs::Float64MultiArray>("/cartesian_position_velocity_controller/command_cartesian_position", 10);
    //gripper_move_pub = node.advertise<std_msgs::Float64MultiArray>("/franka_gripper/move/goal", 10);
    //gripper_grasp_pub = node.advertise<std_msgs::Float64MultiArray>("/franka_gripper/gripper_action/goal", 10);
}

void frankaEmika::goToPosition(double cartesianPosition[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (isnan(cartesianPosition[i]))
            cartesianPosition[i] = currentPosition[i];
    }

    std_msgs::Float64MultiArray msg;
    msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
    msg.layout.dim[0].label = "position";
    msg.layout.dim[0].size = 3;

    std::vector<_Float64> vecSend;
    vecSend.push_back(cartesianPosition[0]);
    vecSend.push_back(cartesianPosition[1]);
    vecSend.push_back(cartesianPosition[2]);

    msg.data = vecSend;
    pose_pub.publish(msg);
}

void frankaEmika::setCurrentPosition()
{
    currentPosition[0] = position[0];
    currentPosition[1] = position[1];
    currentPosition[2] = position[2];
}

void frankaEmika::move(double moveCoordinates[3])
{
    double cartesianPosition[3];
    cartesianPosition[0] = currentPosition[0] + moveCoordinates[0];
    cartesianPosition[1] = currentPosition[1] + moveCoordinates[1];
    cartesianPosition[2] = currentPosition[2] + moveCoordinates[2];
    goToPosition(cartesianPosition);
}

double *frankaEmika::getCurrentPosition()
{
    return currentPosition;
}
