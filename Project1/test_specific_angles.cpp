#include <iostream>
#include "opencv2/opencv.hpp"
#include "custom_cv.h"

std::vector<cv::Point> FindLocalExtrema_Simple(cv::Mat& src, double minThreshold = 0.02)
{
    cv::Mat dilatedImg, localMaxImg;
    cv::Size sz(7, 7);
    cv::Mat rectKernel = cv::getStructuringElement(cv::MORPH_RECT, sz);

    cv::dilate(src, dilatedImg, rectKernel);
    localMaxImg = (src == dilatedImg);

    cv::Mat erodedImg, localMinImg;
    cv::erode(src, erodedImg, rectKernel);
    localMinImg = (src > erodedImg);

    cv::Mat localExtremaImg;
    localExtremaImg = (localMaxImg & localMinImg);

    std::vector<cv::Point> points;

    for (int y = 0; y < localExtremaImg.rows; ++y) {
        for (int x = 0; x < localExtremaImg.cols; ++x) {
            uchar val = localExtremaImg.at<uchar>(y, x);
            float response = src.at<float>(y, x);
            
            if (val && response >= minThreshold) {
                points.push_back(cv::Point(x, y));
            }
        }
    }
    return points;
}

int main() {
    std::cout << "=== Testing Specific Angle Corner Detection Issues ===" << std::endl;
    
    // Test multiple angles that might cause problems
    std::vector<float> angles = {0, 15, 30, 45, 60, 75, 90};
    
    for (float angle : angles) {
        std::cout << "\\n🔍 Testing " << angle << "° rotation:" << std::endl;
        
        cv::Mat testImg = cv::Mat::zeros(300, 300, CV_8UC1);
        
        // Create rotated rectangle
        cv::Point2f center(150, 150);
        cv::Size2f size(80, 50);
        
        cv::RotatedRect rotRect(center, size, angle);
        cv::Point2f vertices[4];
        rotRect.points(vertices);
        
        for (int i = 0; i < 4; i++) {
            cv::line(testImg, vertices[i], vertices[(i+1)%4], cv::Scalar(255), 2);
        }
        
        // Test both implementations
        int blockSize = 5;
        int kSize = 3;
        double k = 0.01;
        
        // OpenCV
        cv::Mat R_opencv;
        cv::cornerHarris(testImg, R_opencv, blockSize, kSize, k);
        cv::threshold(R_opencv, R_opencv, 0.02, 0, cv::THRESH_TOZERO);
        std::vector<cv::Point> corners_opencv = FindLocalExtrema_Simple(R_opencv, 0.02);
        
        // Custom
        cv::Mat R_custom;
        custom_cv::cornerHarris(testImg, R_custom, blockSize, kSize, k);
        cv::threshold(R_custom, R_custom, 0.02, 0, cv::THRESH_TOZERO);
        std::vector<cv::Point> corners_custom = FindLocalExtrema_Simple(R_custom, 0.02);
        
        std::cout << "  OpenCV: " << corners_opencv.size() << " corners" << std::endl;
        std::cout << "  Custom: " << corners_custom.size() << " corners" << std::endl;
        
        if (corners_custom.size() < corners_opencv.size() * 0.5) {
            std::cout << "  ❌ Custom significantly worse at " << angle << "°" << std::endl;
        } else if (corners_custom.size() == 0 && corners_opencv.size() > 0) {
            std::cout << "  🔥 CRITICAL: Custom found no corners at " << angle << "°" << std::endl;
        } else {
            std::cout << "  ✅ Custom performance acceptable at " << angle << "°" << std::endl;
        }
    }
    
    // Test the main.cpp scenario more closely
    std::cout << "\\n\\n=== Testing main.cpp Scenario ===" << std::endl;
    
    // Load or create image similar to shapes1.jpg
    cv::Mat shapes;
    shapes = cv::imread("images/shapes1.jpg", cv::IMREAD_GRAYSCALE);
    
    if (shapes.empty()) {
        std::cout << "Creating synthetic shapes similar to shapes1.jpg..." << std::endl;
        shapes = cv::Mat::zeros(400, 400, CV_8UC1);
        
        // Rectangle
        cv::rectangle(shapes, cv::Point(50, 50), cv::Point(150, 120), cv::Scalar(255), 2);
        
        // Rotated rectangle (45 degrees)
        cv::Point2f center2(250, 100);
        cv::RotatedRect rotRect2(center2, cv::Size2f(70, 40), 45);
        cv::Point2f vertices2[4];
        rotRect2.points(vertices2);
        for (int i = 0; i < 4; i++) {
            cv::line(shapes, vertices2[i], vertices2[(i+1)%4], cv::Scalar(255), 2);
        }
        
        // Triangle
        std::vector<cv::Point> tri;
        tri.push_back(cv::Point(100, 200));
        tri.push_back(cv::Point(150, 150));
        tri.push_back(cv::Point(200, 200));
        cv::fillPoly(shapes, tri, cv::Scalar(255));
        
        // Rotated triangle
        cv::Point2f tri_center(300, 250);
        float tri_angle = 30;
        std::vector<cv::Point> rot_tri;
        for (int i = 0; i < 3; i++) {
            float base_angle = i * 2 * CV_PI / 3;
            float final_angle = base_angle + tri_angle * CV_PI / 180;
            int x = tri_center.x + 30 * cos(final_angle);
            int y = tri_center.y + 30 * sin(final_angle);
            rot_tri.push_back(cv::Point(x, y));
        }
        cv::fillPoly(shapes, rot_tri, cv::Scalar(255));
        
        // Circle (should not produce corners)
        cv::circle(shapes, cv::Point(100, 320), 30, cv::Scalar(255), 2);
    }
    
    // Apply the exact same parameters as main.cpp
    int blockSize = 5;
    int kSize = 3; 
    double k = 0.01;
    
    // OpenCV test
    cv::Mat R_opencv;
    cv::cornerHarris(shapes, R_opencv, blockSize, kSize, k);
    cv::threshold(R_opencv, R_opencv, 0.02, 0, cv::THRESH_TOZERO);
    std::vector<cv::Point> corners_opencv = FindLocalExtrema_Simple(R_opencv);
    
    // Custom test
    cv::Mat R_custom;
    custom_cv::cornerHarris(shapes, R_custom, blockSize, kSize, k);  
    cv::threshold(R_custom, R_custom, 0.02, 0, cv::THRESH_TOZERO);
    std::vector<cv::Point> corners_custom = FindLocalExtrema_Simple(R_custom);
    
    std::cout << "\\nMain.cpp scenario results:" << std::endl;
    std::cout << "OpenCV detected: " << corners_opencv.size() << " corners" << std::endl;
    std::cout << "Custom detected: " << corners_custom.size() << " corners" << std::endl;
    
    if (corners_custom.size() < corners_opencv.size() * 0.7) {
        std::cout << "\\n❌ CONFIRMED: Custom implementation missing rotated corners!" << std::endl;
        std::cout << "\\nPossible issues:" << std::endl;
        std::cout << "1. Morphological filtering too aggressive" << std::endl;
        std::cout << "2. Threshold normalization issue" << std::endl;
        std::cout << "3. Sobel kernel implementation problem" << std::endl;
    } else {
        std::cout << "\\n✅ Custom implementation performance seems acceptable" << std::endl;
    }
    
    return 0;
}