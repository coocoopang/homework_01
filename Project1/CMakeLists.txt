cmake_minimum_required(VERSION 3.10)
project(ComputerVisionAssignment)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable for original implementation (now includes custom_cv)
add_executable(original_cv main.cpp custom_cv.cpp)
target_link_libraries(original_cv ${OpenCV_LIBS})

# Add executable for custom implementation
add_executable(custom_cv main_updated.cpp custom_cv.cpp)
target_link_libraries(custom_cv ${OpenCV_LIBS})

# Set default target
set_property(TARGET custom_cv PROPERTY CXX_STANDARD 14)
set_property(TARGET original_cv PROPERTY CXX_STANDARD 14)

# Print OpenCV version
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")