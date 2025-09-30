#include <iostream>
#include "opencv2/opencv.hpp"
#include "custom_cv.h"

std::vector<cv::Point> FindLocalExtrema_Debug(cv::Mat& src, double minThreshold = 0.01)
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
    std::cout << "=== Testing Rotated Shape Corner Detection ===" << std::endl;
    
    // Create test image with rotated shapes
    cv::Mat testImg = cv::Mat::zeros(400, 400, CV_8UC1);
    
    // 1. Axis-aligned rectangle (should work)
    cv::rectangle(testImg, cv::Point(50, 50), cv::Point(120, 120), cv::Scalar(255), 2);
    std::cout << "✓ Added axis-aligned rectangle" << std::endl;
    
    // 2. Rotated rectangle (problem case)
    cv::Point2f center(200, 100);
    cv::Size2f size(80, 50);
    float angle = 30.0; // 30 degree rotation
    
    cv::RotatedRect rotRect(center, size, angle);
    cv::Point2f vertices[4];
    rotRect.points(vertices);
    
    for (int i = 0; i < 4; i++) {
        cv::line(testImg, vertices[i], vertices[(i+1)%4], cv::Scalar(255), 2);
    }
    std::cout << "✓ Added 30° rotated rectangle" << std::endl;
    
    // 3. Rotated triangle (problem case)
    std::vector<cv::Point> triangle;
    triangle.push_back(cv::Point(300 + 30*cos(CV_PI/6), 200 + 30*sin(CV_PI/6)));     // rotated
    triangle.push_back(cv::Point(300 + 30*cos(CV_PI/6 + 2*CV_PI/3), 200 + 30*sin(CV_PI/6 + 2*CV_PI/3)));
    triangle.push_back(cv::Point(300 + 30*cos(CV_PI/6 + 4*CV_PI/3), 200 + 30*sin(CV_PI/6 + 4*CV_PI/3)));
    
    cv::fillPoly(testImg, triangle, cv::Scalar(255));
    std::cout << "✓ Added rotated triangle" << std::endl;
    
    // 4. L-shape (should work well)  
    cv::line(testImg, cv::Point(80, 250), cv::Point(80, 320), cv::Scalar(255), 4);
    cv::line(testImg, cv::Point(80, 320), cv::Point(150, 320), cv::Scalar(255), 4);
    std::cout << "✓ Added L-shape" << std::endl;
    
    // Test with both implementations
    int blockSize = 5;
    int kSize = 3;
    double k = 0.01;
    
    std::cout << "\\n🔬 Testing OpenCV Harris:" << std::endl;
    cv::Mat R_opencv;
    cv::cornerHarris(testImg, R_opencv, blockSize, kSize, k);
    cv::threshold(R_opencv, R_opencv, 0.02, 0, cv::THRESH_TOZERO);
    
    std::vector<cv::Point> corners_opencv = FindLocalExtrema_Debug(R_opencv, 0.02);
    std::cout << "OpenCV found " << corners_opencv.size() << " corners" << std::endl;
    
    // Analyze corner locations
    int axis_aligned = 0, rotated_rect = 0, rotated_tri = 0, l_shape = 0;
    for (const auto& corner : corners_opencv) {
        if (corner.x >= 40 && corner.x <= 130 && corner.y >= 40 && corner.y <= 130) {
            axis_aligned++;
        } else if (corner.x >= 160 && corner.x <= 240 && corner.y >= 60 && corner.y <= 140) {
            rotated_rect++;
        } else if (corner.x >= 270 && corner.x <= 330 && corner.y >= 170 && corner.y <= 230) {
            rotated_tri++;
        } else if (corner.x >= 70 && corner.x <= 160 && corner.y >= 240 && corner.y <= 330) {
            l_shape++;
        }
    }
    
    std::cout << "OpenCV corners by shape:" << std::endl;
    std::cout << "  Axis-aligned rect: " << axis_aligned << std::endl;
    std::cout << "  Rotated rect: " << rotated_rect << std::endl;
    std::cout << "  Rotated triangle: " << rotated_tri << std::endl;
    std::cout << "  L-shape: " << l_shape << std::endl;
    
    std::cout << "\\n🔬 Testing Custom Harris:" << std::endl;
    cv::Mat R_custom;
    custom_cv::cornerHarris(testImg, R_custom, blockSize, kSize, k);
    cv::threshold(R_custom, R_custom, 0.02, 0, cv::THRESH_TOZERO);
    
    std::vector<cv::Point> corners_custom = FindLocalExtrema_Debug(R_custom, 0.02);
    std::cout << "Custom found " << corners_custom.size() << " corners" << std::endl;
    
    // Analyze custom corner locations
    axis_aligned = rotated_rect = rotated_tri = l_shape = 0;
    for (const auto& corner : corners_custom) {
        if (corner.x >= 40 && corner.x <= 130 && corner.y >= 40 && corner.y <= 130) {
            axis_aligned++;
        } else if (corner.x >= 160 && corner.x <= 240 && corner.y >= 60 && corner.y <= 140) {
            rotated_rect++;
        } else if (corner.x >= 270 && corner.x <= 330 && corner.y >= 170 && corner.y <= 230) {
            rotated_tri++;
        } else if (corner.x >= 70 && corner.x <= 160 && corner.y >= 240 && corner.y <= 330) {
            l_shape++;
        }
    }
    
    std::cout << "Custom corners by shape:" << std::endl;
    std::cout << "  Axis-aligned rect: " << axis_aligned << std::endl;
    std::cout << "  Rotated rect: " << rotated_rect << std::endl;
    std::cout << "  Rotated triangle: " << rotated_tri << std::endl;
    std::cout << "  L-shape: " << l_shape << std::endl;
    
    // Analysis
    std::cout << "\\n📊 Problem Analysis:" << std::endl;
    
    if (rotated_rect == 0 || rotated_tri == 0) {
        std::cout << "❌ Custom implementation fails on rotated shapes!" << std::endl;
        std::cout << "\\nPossible causes:" << std::endl;
        std::cout << "1. Overly strict filtering removes rotated corner responses" << std::endl;
        std::cout << "2. Morphological filtering may be too aggressive" << std::endl;
        std::cout << "3. Threshold may be too high for rotated corners" << std::endl;
    } else {
        std::cout << "✅ Custom implementation handles rotated shapes well" << std::endl;
    }
    
    // Show response maps for debugging
    try {
        cv::Mat R_opencv_display, R_custom_display;
        cv::normalize(R_opencv, R_opencv_display, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        R_custom.convertTo(R_custom_display, CV_8UC1, 255.0);
        
        cv::imshow("Test Image", testImg);
        cv::imshow("OpenCV Harris Response", R_opencv_display);
        cv::imshow("Custom Harris Response", R_custom_display);
        
        // Draw detected corners
        cv::Mat result_opencv, result_custom;
        cv::cvtColor(testImg, result_opencv, cv::COLOR_GRAY2BGR);
        cv::cvtColor(testImg, result_custom, cv::COLOR_GRAY2BGR);
        
        for (const auto& corner : corners_opencv) {
            cv::circle(result_opencv, corner, 3, cv::Scalar(0, 0, 255), 2);
        }
        for (const auto& corner : corners_custom) {
            cv::circle(result_custom, corner, 3, cv::Scalar(0, 255, 0), 2);
        }
        
        cv::imshow("OpenCV Corners (Red)", result_opencv);
        cv::imshow("Custom Corners (Green)", result_custom);
        
        std::cout << "\\nPress any key to continue..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    } catch (const cv::Exception& e) {
        std::cout << "Display not available, skipping visual output." << std::endl;
    }
    
    return 0;
}