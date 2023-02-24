#include "../include/yolo_detector_node.h"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto detector_node = std::make_shared<DetectorNode>(argc, argv);

  rclcpp::spin(detector_node);

  rclcpp::shutdown();
  return 0;
}
