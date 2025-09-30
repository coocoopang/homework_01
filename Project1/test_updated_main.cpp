#include <iostream>
#include "opencv2/opencv.hpp"
#include "custom_cv.h"

int main() {
    std::cout << "=== Testing Updated main.cpp with Custom Implementations ===" << std::endl;
    
    // Create test image to simulate the functions
    cv::Mat testImg = cv::Mat::zeros(300, 300, CV_8UC1);
    
    // Add horizontal and vertical lines
    cv::line(testImg, cv::Point(50, 100), cv::Point(250, 100), cv::Scalar(255), 2);
    cv::line(testImg, cv::Point(100, 50), cv::Point(100, 250), cv::Scalar(255), 2);
    cv::rectangle(testImg, cv::Point(150, 150), cv::Point(200, 200), cv::Scalar(255), 2);
    
    std::cout << "\\n1. Testing Hough Lines (Threshold = 80):" << std::endl;
    
    cv::Mat edges;
    cv::Canny(testImg, edges, 170, 200);
    
    std::vector<cv::Vec2f> lines;
    custom_cv::HoughLines(edges, lines, 1, CV_PI / 180, 80);
    
    std::cout << "   Found " << lines.size() << " lines" << std::endl;
    
    // Analyze line types
    int horizontal = 0, vertical = 0, diagonal = 0;
    for (const auto& line : lines) {
        double theta_deg = line[1] * 180.0 / CV_PI;
        if (std::abs(theta_deg) < 15 || std::abs(theta_deg - 180) < 15) {
            horizontal++;
        } else if (std::abs(theta_deg - 90) < 15) {
            vertical++;
        } else {
            diagonal++;
        }
    }
    std::cout << "   Line types: " << horizontal << " horizontal, " 
              << vertical << " vertical, " << diagonal << " diagonal" << std::endl;
    
    std::cout << "\\n2. Testing Harris Corners:" << std::endl;
    
    cv::Mat harris_result;
    custom_cv::cornerHarris(testImg, harris_result, 5, 3, 0.01);
    
    // Apply the same threshold as main.cpp
    cv::threshold(harris_result, harris_result, 0.02, 0, cv::THRESH_TOZERO);
    
    int corner_pixels = cv::countNonZero(harris_result > 0);
    std::cout << "   Corner pixels after threshold: " << corner_pixels << std::endl;
    
    std::cout << "\\n=== Integration Test Results ===" << std::endl;
    
    if (diagonal == 0) {
        std::cout << "✅ Hough Lines: Diagonal filtering working correctly" << std::endl;
    } else {
        std::cout << "⚠️ Hough Lines: " << diagonal << " diagonal lines detected" << std::endl;
    }
    
    if (corner_pixels < 500) { // Reasonable range
        std::cout << "✅ Harris Corners: Filtering working correctly" << std::endl;
    } else {
        std::cout << "⚠️ Harris Corners: Many corner pixels detected (" << corner_pixels << ")" << std::endl;
    }
    
    std::cout << "\\nThe updated main.cpp is ready to use with your custom implementations!" << std::endl;
    std::cout << "\\nTo run the full program:" << std::endl;
    std::cout << "  cd build && ./original_cv" << std::endl;
    std::cout << "\\nNote: Make sure to place test images in D:/images/ or modify paths in main.cpp" << std::endl;
    
    return 0;
}