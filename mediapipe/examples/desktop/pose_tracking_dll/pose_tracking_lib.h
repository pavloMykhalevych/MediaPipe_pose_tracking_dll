#ifndef POSE_TRACKING_LIBRARY_H
#define POSE_TRACKING_LIBRARY_H

#ifdef COMPILING_DLL
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

#include <cstdlib>
#include <memory>
#include <string>
#include <windows.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_replace.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

struct Point3DWithVisibility{
  cv::Point3f PointCoordinates;
  double visibility;
};

struct Point2DWithVisibility{
  cv::Point2f PointCoordinates;
  double visibility;
};

class MPPoseTrackingDetector {
public:
  MPPoseTrackingDetector(const char *pose_landmark_model_path, const char *pose_detection_model_path);

  void DetectPoses(const cv::Mat &camera_frame, bool *isPose, const std::chrono::milliseconds& timestamp);

  void DetectLandmarks(Point2DWithVisibility *pose_landmarks);
  void DetectLandmarks(Point3DWithVisibility *pose_landmarks);

  static constexpr auto kLandmarksNum = 33;

private:
  absl::Status InitPoseTrackingDetector(const char *pose_landmark_model_path, const char *pose_detection_model_path);

  absl::Status DetectPosesWithStatus(const cv::Mat &camera_frame, bool *isPose, const std::chrono::milliseconds& timestamp);

  absl::Status DetectLandmarksWithStatus(Point2DWithVisibility *pose_landmarks);
  absl::Status DetectLandmarksWithStatus(Point3DWithVisibility *pose_landmarks);

  static constexpr auto kInputStream = "input_video";
  static constexpr auto kOutputStream_landmarks = "pose_landmarks";
  //static constexpr auto kOutputStream_world_landmarks = "pose_world_landmarks";
  static constexpr auto kOutputStream_pose_detected = "pose_detected";

  static const std::string graphConfig;

  mediapipe::CalculatorGraph graph;

  std::unique_ptr<mediapipe::OutputStreamPoller> landmarks_poller_ptr;
  //std::unique_ptr<mediapipe::OutputStreamPoller> world_landmarks_poller_ptr;
  std::unique_ptr<mediapipe::OutputStreamPoller> pose_detected_poller_ptr;

  mediapipe::Packet pose_landmarks_packet;
  //mediapipe::Packet pose_world_landmarks_packet;

  bool pose_detected;
};

#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT MPPoseTrackingDetector *
MPPoseTrackingDetectorConstruct(const char *pose_landmark_model_path, const char *pose_detection_model_path);

DLLEXPORT void MPPoseTrackingDetectorDestruct(MPPoseTrackingDetector *detector);

DLLEXPORT void MPPoseTrackingDetectorDetectPoses(
    MPPoseTrackingDetector *detector, const cv::Mat &camera_frame, bool *isPose, const std::chrono::milliseconds& timestamp);

DLLEXPORT void
MPPoseTrackingDetectorDetect2DLandmarks(MPPoseTrackingDetector *detector,
                                    Point2DWithVisibility *pose_landmarks);
DLLEXPORT void
MPPoseTrackingDetectorDetect3DLandmarks(MPPoseTrackingDetector *detector,
                                    Point3DWithVisibility *pose_landmarks);

DLLEXPORT extern const int MPPoseTrackingDetectorLandmarksNum;

#ifdef __cplusplus
};
#endif
#endif