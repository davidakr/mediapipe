bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS  mediapipe/application:hand_tracking
GLOG_logtostderr=1 bazel-bin/mediapipe/application/hand_tracking   --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt

Replace FrankaState.h manually since it is not updated to 0.7.0