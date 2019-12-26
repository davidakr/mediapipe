#include "handTracker.h"

std::mutex handTracker::mtx;

handTracker::handTracker()
{
    threadGraph = std::thread(&handTracker::RunMPPGraph, this);
}

void handTracker::waitForProcessFinished()
{
    bool done = false;
    while (!done)
    {
        mtx.lock();
        done = !processing;
        mtx.unlock();
    }
    return;
}
void handTracker::processImage(cv::Mat camera_frame)
{
    mtx.lock();
    camera_frame_raw = camera_frame;
    newData = true;
    processing = true;
    mtx.unlock();
    return;
}

std::vector<float> handTracker::getLandmarks()
{
    waitForProcessFinished();
    return landmarks;
}
bool handTracker::getPresence()
{
    waitForProcessFinished();
    return presence;
}
cv::Mat handTracker::getImage()
{
    waitForProcessFinished();
    return output_mat;
}

::mediapipe::Status handTracker::RunMPPGraph()
{
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        "mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt", &calculator_graph_config_contents));
    LOG(INFO) << "Get calculator graph config contents: "
              << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    LOG(INFO) << "Initialize the calculator graph.";
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    LOG(INFO) << "Initialize the GPU.";
    ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
    MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
    mediapipe::GlCalculatorHelper gpu_helper;
    gpu_helper.InitializeForTest(graph.GetGpuResources().get());

    LOG(INFO) << "Start running the calculator graph.";
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerPresence, graph.AddOutputStreamPoller(presenceOutputStream));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerVideo, graph.AddOutputStreamPoller(videoOutputStream));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerLandmark, graph.AddOutputStreamPoller(landmarkOutputStream));

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    LOG(INFO) << "Start grabbing and processing frames.";
    size_t frame_timestamp = 0;

    while (grab_frames)
    {
        mtx.lock();
        bool processImage = newData;
        if (processImage)
        {
            camera_frame_raw = this->camera_frame_raw;
            newData = false;
            processing = true;
        }
        mtx.unlock();

        while (processImage)
        {
            // Capture opencv camera or video frame.

            if (camera_frame_raw.empty())
            {
                LOG(INFO) << "End of video";
                break;
            }
            cv::Mat camera_frame;
            cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

            // Wrap Mat into an ImageFrame.
            auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
                mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
                mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
            cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
            camera_frame.copyTo(input_frame_mat);

            // Prepare and add graph input packet.
            MP_RETURN_IF_ERROR(
                gpu_helper.RunInGlContext([&input_frame, &frame_timestamp, &graph,
                                           &gpu_helper, this]() -> ::mediapipe::Status {
                    // Convert ImageFrame to GpuBuffer.
                    auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
                    auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
                    glFlush();
                    texture.Release();
                    // Send GPU image packet into the graph.
                    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
                        kInputStream, mediapipe::Adopt(gpu_frame.release())
                                          .At(mediapipe::Timestamp(frame_timestamp++))));
                    return ::mediapipe::OkStatus();
                }));

            // Get the graph result packet, or stop if that fails.
            mediapipe::Packet packet;

            if (!pollerVideo.Next(&packet))
                break;
            std::unique_ptr<mediapipe::ImageFrame> output_frame;

            // Convert GpuBuffer to ImageFrame.
            MP_RETURN_IF_ERROR(gpu_helper.RunInGlContext(
                [&packet, &output_frame, &gpu_helper]() -> ::mediapipe::Status {
                    auto &gpu_frame = packet.Get<mediapipe::GpuBuffer>();
                    auto texture = gpu_helper.CreateSourceTexture(gpu_frame);
                    output_frame = absl::make_unique<mediapipe::ImageFrame>(
                        mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
                        gpu_frame.width(), gpu_frame.height(),
                        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
                    gpu_helper.BindFramebuffer(texture);
                    const auto info =
                        mediapipe::GlTextureInfoForGpuBufferFormat(gpu_frame.format(), 0);
                    glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                                 info.gl_type, output_frame->MutablePixelData());
                    glFlush();
                    texture.Release();
                    return ::mediapipe::OkStatus();
                }));

            // Convert back to opencv for display or saving.
            cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

            // Process landmarks
            if (!pollerLandmark.Next(&packet))
            {
                break;
            }
            auto landmark_frame = packet.Get<std::vector<mediapipe::NormalizedLandmark, std::allocator<mediapipe::NormalizedLandmark>>>();

            // Convert landmarks to vector
            std::vector<float> vec;

            for (auto i = landmark_frame.begin(); i != landmark_frame.end(); ++i)
            {
                mediapipe::NormalizedLandmark landmark = *i;
                float pixelWidth = width * landmark.x();
                float pixelHeigth = height * landmark.y();
                vec.push_back(pixelWidth);
                vec.push_back(pixelHeigth);
            }

            //Process Hand Presence
            if (!pollerPresence.Next(&packet))
            {
                break;
            }
            bool presence = packet.Get<bool>();

            mtx.lock();
            landmarks = vec;
            output_mat = output_frame_mat;
            this->presence = presence;
            processing = false;
            processImage = false;
            mtx.unlock();
        }
    }

    LOG(INFO) << "Shutting down.";

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}