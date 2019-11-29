cc_binary(
    name = "hand_tracking",
    deps = [
        "//mediapipe/application:app",
        "//mediapipe/graphs/hand_tracking:mobile_calculators",
    ],
)

cc_library(
    name = "app",
    srcs = ["app.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
        "//third_party:kinect",
        ":kinectCamera",
        "//mediapipe/calculators/core:concatenate_vector_calculator",
        "@ros",
    ],
)

cc_library(
    name = "kinectCamera",
    srcs = ["kinectCamera.cpp"],
    hdrs = ["kinectCamera.h"],
)

