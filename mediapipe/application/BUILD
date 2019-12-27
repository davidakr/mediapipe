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
        "//mediapipe/calculators/core:concatenate_vector_calculator",
        "//third_party:kinect",
        ":kinectCamera",
        ":frankaEmika",
        ":handTracker",
        "@ros",
    ],
)

cc_library(
    name = "kinectCamera",
    srcs = ["kinectCamera.cpp"],
    hdrs = ["kinectCamera.h"],
)

cc_library(
    name = "frankaEmika",
    srcs = ["frankaEmika.cpp"],
    hdrs = ["frankaEmika.h"],
    deps = [
        "@ros",
    ],
)

cc_binary(
    name = "publisher",
    srcs = ["publisher.cc"],
    deps = [
        "@ros",
        ":handTracker",
    ],
)

cc_library(
    name = "handTracker",
    srcs = ["handTracker.cpp"],
    hdrs = ["handTracker.h"],
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
        "//mediapipe/calculators/core:concatenate_vector_calculator",
    ],
)

cc_binary(
    name = "integration",
    deps = [
        "//mediapipe/application:publisher",
        "//mediapipe/graphs/hand_tracking:mobile_calculators",
    ],
)
