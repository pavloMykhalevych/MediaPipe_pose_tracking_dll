#include "pose_tracking_lib.h"

MPPoseTrackingDetector::MPPoseTrackingDetector(const char *pose_landmark_model_path, const char *pose_detection_model_path) {
  const auto status = InitPoseTrackingDetector(pose_landmark_model_path, pose_detection_model_path);
  if (!status.ok()) {
    LOG(INFO) << "Failed constructing MPPoseTrackingDetector.";
    LOG(INFO) << status.message();
  }
}

absl::Status
MPPoseTrackingDetector::InitPoseTrackingDetector(const char *pose_landmark_model_path, const char *pose_detection_model_path) {

  if (pose_landmark_model_path == nullptr) {
    pose_landmark_model_path =
        "mediapipe/modules/pose_landmark/pose_landmark_full.tflite";
  }

  if (pose_detection_model_path == nullptr) {
    pose_detection_model_path =
        "mediapipe/modules/pose_detection/pose_detection.tflite";
  }

  // Prepare graph config.
  auto preparedGraphConfig = absl::StrReplaceAll(
    graphConfig,
    {{"$poseLandmarkModelPath", pose_landmark_model_path}});

  preparedGraphConfig = absl::StrReplaceAll(
    preparedGraphConfig,
    {{"$poseDetectionModelPath", pose_detection_model_path}});

  LOG(INFO) << "Get calculator graph config contents: " << preparedGraphConfig;

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          preparedGraphConfig);
  LOG(INFO) << "Initialize the calculator graph.";

  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Start running the calculator graph.";

  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmarks_poller,
                   graph.AddOutputStreamPoller(kOutputStream_landmarks));

/*  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller world_landmarks_poller,
                   graph.AddOutputStreamPoller(kOutputStream_world_landmarks));*/

  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pose_detected_poller,
                   graph.AddOutputStreamPoller(kOutputStream_pose_detected));

  landmarks_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
      std::move(landmarks_poller));

/*  world_landmarks_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
      std::move(world_landmarks_poller));*/

  pose_detected_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
      std::move(pose_detected_poller));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "MPPoseTrackingDetector constructed successfully.";

  return absl::OkStatus();
}

absl::Status
MPPoseTrackingDetector::DetectPosesWithStatus(const cv::Mat &camera_frame, bool *isPose,  const std::chrono::microseconds& timestamp){

  if (!isPose) {
    return absl::InvalidArgumentError(
        "MPPoseTrackingDetector::DetectPosesWithStatus requires notnull pointer to "
        "save results data.");
  }
  
  *isPose = false;
  pose_detected = false;

  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  camera_frame.copyTo(input_frame_mat);

  // Send image packet into the graph.
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(timestamp.count()))));

  mediapipe::Packet pose_detected_packet;
  if (!pose_detected_poller_ptr ||
      !pose_detected_poller_ptr->Next(&pose_detected_packet)) {
    return absl::CancelledError(
        "Failed during getting next pose_detected_packet.");
  }

  auto &pose_detected_packet_val = pose_detected_packet.Get<bool>();

  if (!pose_detected_packet_val) {
    return absl::OkStatus();
  }

  // Get pose landmarks.
  if (!landmarks_poller_ptr ||
      !landmarks_poller_ptr->Next(&pose_landmarks_packet)) {
    return absl::CancelledError("Failed during getting next landmarks_packet.");
  }

/*  if (!world_landmarks_poller_ptr ||
      !world_landmarks_poller_ptr->Next(&pose_world_landmarks_packet)) {
    return absl::CancelledError("Failed during getting next world_landmarks_packet.");
  }*/

  *isPose = true;
  pose_detected = true;

  return absl::OkStatus();
}

absl::Status MPPoseTrackingDetector::DetectLandmarksWithStatus(
    Point2DWithVisibility *poses_landmarks) {

  if (pose_landmarks_packet.IsEmpty()) {
    return absl::CancelledError("Pose landmarks packet is empty.");
  }
  if (!pose_detected)
  {
    return absl::OkStatus();
  }

  auto &pose_landmarks =
      pose_landmarks_packet
          .Get<::mediapipe::NormalizedLandmarkList>();

  // Convert landmarks to cv::Point2f**.
  const auto &normalizedLandmarkList = pose_landmarks;
  const auto landmarks_num = normalizedLandmarkList.landmark_size();

  if (landmarks_num != kLandmarksNum) {
    return absl::CancelledError("Detected unexpected landmarks number.");
  }

  for (int j = 0; j < landmarks_num; ++j) {
    const auto &landmark = normalizedLandmarkList.landmark(j);
    poses_landmarks[j].PointCoordinates.x = landmark.x();
    poses_landmarks[j].PointCoordinates.y = landmark.y();
    poses_landmarks[j].visibility = landmark.visibility();
  }

  return absl::OkStatus();
}

absl::Status MPPoseTrackingDetector::DetectLandmarksWithStatus(
    Point3DWithVisibility *poses_landmarks) {

  if (pose_landmarks_packet.IsEmpty()) {
    return absl::CancelledError("Pose world landmarks packet is empty.");
  }

  if (!pose_detected)
  {
    return absl::OkStatus();
  }

  auto &pose_landmarks =
      pose_landmarks_packet
          .Get<::mediapipe::NormalizedLandmarkList>();

  // Convert landmarks to cv::Point3f**.
  const auto &normalizedLandmarkList = pose_landmarks;
  const auto landmarks_num = normalizedLandmarkList.landmark_size();

  if (landmarks_num != kLandmarksNum) {
    return absl::CancelledError("Detected unexpected world landmarks number.");
  }

  for (int j = 0; j < landmarks_num; ++j) {
    const auto &landmark = normalizedLandmarkList.landmark(j);
    poses_landmarks[j].PointCoordinates.x = landmark.x();
    poses_landmarks[j].PointCoordinates.y = landmark.y();
    poses_landmarks[j].PointCoordinates.z = landmark.z();
    poses_landmarks[j].visibility = landmark.visibility();
  }

  return absl::OkStatus();
}

void MPPoseTrackingDetector::DetectPoses(const cv::Mat &camera_frame, bool *isPose, const std::chrono::microseconds& timestamp) {
  const auto status =
      DetectPosesWithStatus(camera_frame, isPose, timestamp);
  if (!status.ok()) {
    LOG(INFO) << "MPPoseTrackingDetector::DetectPoses failed: " << status.message();
  }
}

void MPPoseTrackingDetector::DetectLandmarks(Point2DWithVisibility *pose_landmarks) {
  const auto status = DetectLandmarksWithStatus(pose_landmarks);
  if (!status.ok()) {
    LOG(INFO) << "MPPoseTrackingDetector::DetectLandmarks failed: "
              << status.message();
  }
}

void MPPoseTrackingDetector::DetectLandmarks(Point3DWithVisibility *pose_landmarks) {
  const auto status = DetectLandmarksWithStatus(pose_landmarks);
  if (!status.ok()) {
    LOG(INFO) << "MPPoseTrackingDetector::DetectLandmarks failed: "
              << status.message();
  }
}

extern "C" {
DLLEXPORT MPPoseTrackingDetector *
MPPoseTrackingDetectorConstruct(const char *pose_landmark_model_path, const char *pose_detection_model_path) {
  return new MPPoseTrackingDetector(pose_landmark_model_path, pose_detection_model_path);
}

DLLEXPORT void MPPoseTrackingDetectorDestruct(MPPoseTrackingDetector *detector) {
  delete detector;
}

DLLEXPORT void MPPoseTrackingDetectorDetectPoses(
    MPPoseTrackingDetector *detector, const cv::Mat &camera_frame, bool *isPose, const std::chrono::microseconds& timestamp) {
  detector->DetectPoses(camera_frame, isPose, timestamp);
}

DLLEXPORT void
MPPoseTrackingDetectorDetect2DLandmarks(MPPoseTrackingDetector *detector,
                                    Point2DWithVisibility *pose_landmarks) {
  detector->DetectLandmarks(pose_landmarks);
}
DLLEXPORT void
MPPoseTrackingDetectorDetect3DLandmarks(MPPoseTrackingDetector *detector,
                                    Point3DWithVisibility *pose_landmarks) {
  detector->DetectLandmarks(pose_landmarks);
}

DLLEXPORT const int MPPoseTrackingDetectorLandmarksNum =
    MPPoseTrackingDetector::kLandmarksNum;
}

const std::string MPPoseTrackingDetector::graphConfig = R"pb(
# MediaPipe graph that performs pose tracking with TensorFlow Lite on CPU.

# CPU buffer. (ImageFrame)
input_stream: "input_video"

# Pose landmarks. (NormalizedLandmarkList)
output_stream: "pose_landmarks"

# Pose landmarks. (bool)
output_stream: "pose_detected"

# Generates side packet to enable segmentation.
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:enable_segmentation"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { bool_value: true }
    }
  }
}

# Defines side packets for further use in the graph.
node {
    calculator: "ConstantSidePacketCalculator"
    output_side_packet: "PACKET:pose_landmark_model_path"
    options: {
        [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
            packet { string_value: "$poseLandmarkModelPath" }
        }
    }
}

node {
    calculator: "ConstantSidePacketCalculator"
    output_side_packet: "PACKET:pose_detection_model_path"
    options: {
        [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
            packet { string_value: "$poseDetectionModelPath" }
        }
    }
}

node {
    calculator: "LocalFileContentsCalculator"
    input_side_packet: "FILE_PATH:0:pose_landmark_model_path"
    output_side_packet: "CONTENTS:0:pose_landmark_model_blob"
    input_side_packet: "FILE_PATH:1:pose_detection_model_path"
    output_side_packet: "CONTENTS:1:pose_detection_model_blob"
}

node {
    calculator: "TfLiteModelCalculator"
    input_side_packet: "MODEL_BLOB:pose_landmark_model_blob"
    output_side_packet: "MODEL:pose_landmark_model"
}

node {
    calculator: "TfLiteModelCalculator"
    input_side_packet: "MODEL_BLOB:pose_detection_model_blob"
    output_side_packet: "MODEL:pose_detection_model"
}

node {
  calculator: "FlowLimiterCalculator"
  input_stream: "input_video"
  input_stream: "FINISHED:pose_detected"
  input_stream_info: {
    tag_index: "FINISHED"
    back_edge: true
  }
  output_stream: "throttled_input_video"
}

# Subgraph that detects poses and corresponding landmarks.
node {
  calculator: "PoseLandmarkCpu"
  input_side_packet: "ENABLE_SEGMENTATION:enable_segmentation"
  input_side_packet: "MODEL:0:pose_landmark_model"
  input_side_packet: "MODEL:1:pose_detection_model"
  input_stream: "IMAGE:throttled_input_video"
  output_stream: "LANDMARKS:pose_landmarks"
}

# The image stream is only used to 
# make the calculator work even when there is no input vector.
node {
  calculator: "IsPosePresentCalculator"
  input_stream: "CLOCK:throttled_input_video"
  input_stream: "LANDMARKS:pose_landmarks"
  output_stream: "BOOL:pose_detected"
}
)pb";
