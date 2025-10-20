cmake_minimum_required(VERSION 3.10)
project(FaceMatchingSystem)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV 찾기
find_package(OpenCV REQUIRED)

# 컴파일러 플래그
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# 디버그/릴리스 설정
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# 헤더 파일 디렉터리
include_directories(${OpenCV_INCLUDE_DIRS})

# 실행 파일 생성
add_executable(face_matcher
    main.cpp
    face_matcher.cpp
)

# 라이브러리 링크
target_link_libraries(face_matcher ${OpenCV_LIBS})

# 설치 설정
install(TARGETS face_matcher DESTINATION bin)

# 정보 출력
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")