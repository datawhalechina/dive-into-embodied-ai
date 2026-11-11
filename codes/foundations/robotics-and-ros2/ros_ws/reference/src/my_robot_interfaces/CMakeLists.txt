cmake_minimum_required(VERSION 3.8)
project(my_robot_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SolveIK.srv"
  "action/MoveTo.action"
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
