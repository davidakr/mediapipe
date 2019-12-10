
# Hand tracking using Google mediapipe

ROS node which sends out the hand landmarks of the detected hand.
Uses the Windows Kinect Camera.

## Build 

`bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS  mediapipe/application:hand_tracking` 

## Run

`GLOG_logtostderr=1 bazel-bin/mediapipe/application/hand_tracking   --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt` 


