# Description:
#   OpenCV libraries for video/image processing on Linux

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

# The following build rule assumes that OpenCV is installed by
# 'apt-get install libopencv-core-dev libopencv-highgui-dev \'
# '                libopencv-imgproc-dev libopencv-video-dev' on Debian/Ubuntu.
# If you install OpenCV separately, please modify the build rule accordingly.
cc_library(
    name = "pistache",
    srcs = glob(
        [
            "lib/x86_64-linux-gnu/libpistache.so",
        ],
    ),
    hdrs = glob(["include/pistache/*.h*"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
