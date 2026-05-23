cmake_minimum_required(VERSION 3.8)
project(multi_robot_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

find_package(ament_cmake REQUIRED)

# Ordner für die Installation freigeben
install(
  DIRECTORY launch worlds urdf rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()