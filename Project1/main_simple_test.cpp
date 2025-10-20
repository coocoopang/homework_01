#include <iostream>
#include "opencv2/opencv.hpp"
#include "custom_cv.h"

// Simple test program to verify fixes
int main() {
    std::cout << "=== Testing Fixed Custom Implementations ===" << std::endl;
    
    // Create test image for Hough Lines
    cv::Mat testLines = cv::Mat::zeros(300, 300, CV_8UC1);
    cv::line(testLines, cv::Point(50, 50), cv::Point(250, 50), cv::Scalar(255), 2);
    cv::line(testLines, cv::Point(100, 0), cv::Point(100, 300), cv::Scalar(255), 2);
    cv::line(testLines, cv::Point(150, 100), cv::Point(250, 200), cv::Scalar(255), 2);
    
    // Test Hough Lines
    cv::Mat edges;
    cv::Canny(testLines, edges, 50, 150);
    
    std::vector<cv::Vec2f> lines_opencv, lines_custom;
    
    cv::HoughLines(edges, lines_opencv, 1, CV_PI/180, 50);
    custom_cv::HoughLines(edges, lines_custom, 1, CV_PI/180, 50);
    
    std::cout << "Hough Lines Test:" << std::endl;
    std::cout << "  OpenCV found: " << lines_opencv.size() << " lines" << std::endl;
    std::cout << "  Custom found: " << lines_custom.size() << " lines" << std::endl;
    
    // Create test image for Harris Corners
    cv::Mat testCorners = cv::Mat::zeros(300, 300, CV_8UC1);
    cv::rectangle(testCorners, cv::Point(50, 50), cv::Point(150, 150), cv::Scalar(255), 2);
    cv::rectangle(testCorners, cv::Point(180, 80), cv::Point(250, 180), cv::Scalar(255), -1);
    
    // Test Harris Corners
    cv::Mat harris_opencv, harris_custom;
    
    cv::cornerHarris(testCorners, harris_opencv, 5, 3, 0.04);
    custom_cv::cornerHarris(testCorners, harris_custom, 5, 3, 0.04);
    
    double min1, max1, min2, max2;
    cv::minMaxLoc(harris_opencv, &min1, &max1);
    cv::minMaxLoc(harris_custom, &min2, &max2);
    
    std::cout << "Harris Corners Test:" << std::endl;
    std::cout << "  OpenCV response range: [" << min1 << ", " << max1 << "]" << std::endl;
    std::cout << "  Custom response range: [" << min2 << ", " << max2 << "]" << std::endl;
    
    // Show results if display is available
    bool showImages = true;
    try {
        if (showImages) {
            // Show Hough Lines results
            cv::Mat linesResult = cv::Mat::zeros(300, 300, CV_8UC3);
            cvtColor(testLines, linesResult, cv::COLOR_GRAY2BGR);
            
            // Draw OpenCV lines in red
            for (const auto& line : lines_opencv) {
                float rho = line[0], theta = line[1];
                double x0 = rho * cos(theta), y0 = rho * sin(theta);
                cv::Point pt1(cvRound(x0 + 1000*(-sin(theta))), cvRound(y0 + 1000*(cos(theta))));
                cv::Point pt2(cvRound(x0 - 1000*(-sin(theta))), cvRound(y0 - 1000*(cos(theta))));
                cv::line(linesResult, pt1, pt2, cv::Scalar(0, 0, 255), 1);
            }
            
            // Draw custom lines in green  
            for (const auto& line : lines_custom) {
                float rho = line[0], theta = line[1];
                double x0 = rho * cos(theta), y0 = rho * sin(theta);
                cv::Point pt1(cvRound(x0 + 1000*(-sin(theta))), cvRound(y0 + 1000*(cos(theta))));
                cv::Point pt2(cvRound(x0 - 1000*(-sin(theta))), cvRound(y0 - 1000*(cos(theta))));
                cv::line(linesResult, pt1, pt2, cv::Scalar(0, 255, 0), 1);
            }
            
            cv::imshow("Hough Lines Comparison (Red=OpenCV, Green=Custom)", linesResult);
            
            // Show Harris corners
            cv::Mat harrisCombined;
            cv::normalize(harris_opencv, harrisCombined, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::imshow("OpenCV Harris Response", harrisCombined);
            
            cv::Mat harrisCustomDisplay;
            harris_custom.convertTo(harrisCustomDisplay, CV_8UC1, 255.0);
            cv::imshow("Custom Harris Response", harrisCustomDisplay);
            
            std::cout << "\\nPress any key to continue..." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    } catch (const cv::Exception& e) {
        std::cout << "Display not available, skipping visual output." << std::endl;
    }
    
    std::cout << "\\n=== Test Results ===" << std::endl;
    
    if (lines_custom.size() > 0 && lines_custom.size() < 20) {
        std::cout << "✅ Hough Lines: Fixed! Reasonable number of lines detected." << std::endl;
    } else {
        std::cout << "❌ Hough Lines: Still needs adjustment." << std::endl;
    }
    
    if (max2 > 0 && max2 < 100) {
        std::cout << "✅ Harris Corners: Fixed! Normalized response values." << std::endl;
    } else {
        std::cout << "❌ Harris Corners: Still needs adjustment." << std::endl;
    }
    
    return 0;
}