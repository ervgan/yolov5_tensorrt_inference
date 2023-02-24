#ifndef DETECTOR_NODE_HPP
#define DETECTOR_NODE_HPP

#include "../include/yolo_detector.h"
#include "rclcpp/rclcpp.hpp"
#include "yolo_interface/msg/detection.hpp"

class DetectorNode : public rclcpp::Node {
 public:
  DetectorNode(int argc, char** argv);

 private:
  void PublishDetection(const cv::Mat& frame);

  rclcpp::Publisher<yolo_interface::msg::Detection>::SharedPtr
      detection_publisher_;
  YoloDetector yolo_detector_;
  int yolo_detector_state_;
};

#endif  // DETECTOR_NODE_HPP
