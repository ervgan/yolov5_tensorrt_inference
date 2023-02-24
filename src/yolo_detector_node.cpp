#include "../include/yolo_detector_node.h"

#include <glog/logging.h>

#include "std_msgs/msg/header.hpp"

class DetectorNode : public rclcpp::Node {
 public:
  DetectorNode(int argc, char** argv) : Node("detector_node") {
    detection_publisher_ =
        this->create_publisher<yolo_interface::msg::Detection>(
            "detection_topic", 10);
    yolo_detector_state_ = yolo_detector_.init(argc, argv);
  }

 private:
  void PublishDetection(const cv::Mat& frame) {
    if (yolo_detector_ == 0) {
      std::cerr << "You should run detection -d and not serialize -s"
                << std::endl;
      CHECK(false);
    }
    // TODO: create a new function Detect() to process one frame and return a
    // detection struct
    Detection detection = yolo_detector_.Detect();
    auto detection_msg = yolo_interface::msg::Detection();

    detection_msg.header.stamp = this->now();
    detection_msg.x = detection.bounding_box[0];
    detection_msg.y = detection.bounding_box[1];
    detection_msg.width = detection.bounding_box[2];
    detection_msg.height = detection.bounding_box[3];
    detection_msg.cam_id = "";

    RCLCPP_INFO(this->get_logger(), "Publishing detection stamp: %ld",
                this->now().nanoseconds());
    detection_publisher_->publish(detection_msg);
  };
  rclcpp::Publisher<yolo_interface::msg::Detection>::SharedPtr
      detection_publisher_;
  YoloDetector yolo_detector_;
  int yolo_detector_state_;
};
