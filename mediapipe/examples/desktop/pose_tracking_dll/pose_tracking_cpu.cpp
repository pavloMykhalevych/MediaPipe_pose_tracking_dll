#include "pose_tracking_lib.h"

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  cv::VideoCapture capture;
  capture.open(0);
  if (!capture.isOpened()) {
    return -1;
  }

  constexpr char kWindowName[] = "MediaPipe";

  cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
  capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  capture.set(cv::CAP_PROP_FPS, 30);
#endif

  LOG(INFO) << "VideoCapture initialized.";

  constexpr char pose_landmark_model_path[] =
      "mediapipe/modules/pose_landmark/pose_landmark_full.tflite";

  constexpr char pose_detection_model_path[] =
      "mediapipe/modules/pose_detection/pose_detection.tflite";

  MPPoseTrackingDetector *poseDetector = MPPoseTrackingDetectorConstruct(pose_landmark_model_path, pose_detection_model_path);

  // Allocate memory for pose landmarks.
  auto poseLandmarks = new Point3DWithVisibility[MPPoseTrackingDetectorLandmarksNum];

  LOG(INFO) << "PoseTrackingDetector constructed.";

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;

  cv::Mat camera_frame_raw;
  while (capture.read(camera_frame_raw)) {
    // Capture opencv camera.
    
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      LOG(INFO) << "Ignore empty frames from camera.";
      continue;
    }

    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

    bool isPose = false;
    const auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
    LOG(INFO) << "Timestamp: " << timestamp.count();

    MPPoseTrackingDetectorDetectPoses(poseDetector, camera_frame, &isPose, timestamp);

    if (isPose){
      MPPoseTrackingDetectorDetect3DLandmarks(poseDetector, poseLandmarks);
      
      auto &pose_landmarks = poseLandmarks;
      //auto &landmark = pose_landmarks;

      for (int i = 0; i<33; i++){
        LOG(INFO) << "landmark: x - " << pose_landmarks[i].PointCoordinates.x << ", y - "
                      << pose_landmarks[i].PointCoordinates.y << ", z - " << pose_landmarks[i].PointCoordinates.z;
      }
      LOG(INFO) << "----------------------------";
    }

    const int pressed_key = cv::waitKey(5);
    if (pressed_key >= 0 && pressed_key != 255)
      break;

    cv::imshow(kWindowName, camera_frame_raw);
  }

  LOG(INFO) << "Shutting down.";

  delete[] poseLandmarks;

  MPPoseTrackingDetectorDestruct(poseDetector);
}