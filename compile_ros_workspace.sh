#!/usr/bin/env bash

set -e

PACKAGES="roscpp rospy std_msgs franka_msgs"

rm -rf bundle_ws
mkdir bundle_ws
pushd bundle_ws


rosinstall_generator \
    --deps \
    --tar \
    --flat \
    --rosdistro melodic \
    $PACKAGES > ws.rosinstall
wstool init -j8 src ws.rosinstall


catkin config \
    --install \
    --source-space bundle_ws/src \
    --build-space bundle_ws/build \
    --devel-space bundle_ws/devel \
    --log-space bundle_ws/log \
    --install-space bundle_ws/install \
    --isolate-devel \
    --no-extend

catkin build