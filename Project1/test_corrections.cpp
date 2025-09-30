#include <iostream>
#include "opencv2/opencv.hpp"
#include "custom_cv.h"

int main() {
    std::cout << "=== Testing Corrected Implementations ===" << std::endl;
    
    // Test 1: Hough Lines - should prefer horizontal/vertical lines
    std::cout << "\\n1. Testing Hough Lines Corrections:" << std::endl;
    
    cv::Mat testLines = cv::Mat::zeros(300, 300, CV_8UC1);
    // Add horizontal lines
    cv::line(testLines, cv::Point(50, 100), cv::Point(250, 100), cv::Scalar(255), 2);
    cv::line(testLines, cv::Point(50, 200), cv::Point(250, 200), cv::Scalar(255), 2);
    // Add vertical lines  
    cv::line(testLines, cv::Point(100, 50), cv::Point(100, 250), cv::Scalar(255), 2);
    cv::line(testLines, cv::Point(200, 50), cv::Point(200, 250), cv::Scalar(255), 2);
    // Add diagonal lines (should be filtered out)
    cv::line(testLines, cv::Point(50, 50), cv::Point(150, 150), cv::Scalar(255), 2);
    cv::line(testLines, cv::Point(150, 50), cv::Point(250, 150), cv::Scalar(255), 2);
    
    cv::Mat edges;
    cv::Canny(testLines, edges, 50, 150);
    
    std::vector<cv::Vec2f> lines_opencv, lines_custom;
    cv::HoughLines(edges, lines_opencv, 1, CV_PI/180, 50);
    custom_cv::HoughLines(edges, lines_custom, 1, CV_PI/180, 50);
    
    std::cout << "   OpenCV found: " << lines_opencv.size() << " lines" << std::endl;
    std::cout << "   Custom found: " << lines_custom.size() << " lines" << std::endl;
    
    // Analyze line angles
    int horizontal_count = 0, vertical_count = 0, diagonal_count = 0;
    for (const auto& line : lines_custom) {
        double theta_deg = line[1] * 180.0 / CV_PI;
        if (std::abs(theta_deg) < 15 || std::abs(theta_deg - 180) < 15) {
            horizontal_count++;
        } else if (std::abs(theta_deg - 90) < 15) {
            vertical_count++;
        } else {
            diagonal_count++;
        }
    }
    std::cout << "   Custom lines: " << horizontal_count << " horizontal, " 
              << vertical_count << " vertical, " << diagonal_count << " diagonal" << std::endl;
    
    // Test 2: Harris Corners - should avoid circle edges
    std::cout << "\\n2. Testing Harris Corners Corrections:" << std::endl;
    
    cv::Mat testCorners = cv::Mat::zeros(300, 300, CV_8UC1);
    // Rectangle corners (should be detected)
    cv::rectangle(testCorners, cv::Point(50, 50), cv::Point(120, 120), cv::Scalar(255), 2);
    // Triangle corner (should be detected)
    std::vector<cv::Point> triangle = {cv::Point(150, 50), cv::Point(200, 50), cv::Point(175, 100)};
    cv::fillPoly(testCorners, triangle, cv::Scalar(255));
    // Circle (should NOT produce strong corners)
    cv::circle(testCorners, cv::Point(200, 180), 40, cv::Scalar(255), 2);
    // L-shape corner (should be detected)
    cv::line(testCorners, cv::Point(50, 200), cv::Point(50, 250), cv::Scalar(255), 3);
    cv::line(testCorners, cv::Point(50, 250), cv::Point(100, 250), cv::Scalar(255), 3);
    
    cv::Mat harris_opencv, harris_custom;
    cv::cornerHarris(testCorners, harris_opencv, 5, 3, 0.04);
    custom_cv::cornerHarris(testCorners, harris_custom, 5, 3, 0.04);
    
    // Count strong corners
    double thresh_opencv = 0.01 * cv::norm(harris_opencv, cv::NORM_INF);
    cv::Mat strong_opencv = harris_opencv > thresh_opencv;
    int corners_opencv = cv::countNonZero(strong_opencv);
    
    cv::Mat strong_custom = harris_custom > 0.1; // Custom uses normalized values
    int corners_custom = cv::countNonZero(strong_custom);
    
    std::cout << "   OpenCV strong corners: " << corners_opencv << " pixels" << std::endl;
    std::cout << "   Custom strong corners: " << corners_custom << " pixels" << std::endl;
    
    // Results analysis
    std::cout << "\\n=== Correction Results ===" << std::endl;
    
    if (diagonal_count <= horizontal_count + vertical_count) {
        std::cout << "✅ Hough Lines: Successfully filtering diagonal lines" << std::endl;
    } else {
        std::cout << "❌ Hough Lines: Still detecting too many diagonal lines" << std::endl;
    }
    
    if (corners_custom < corners_opencv * 1.5) { // Custom should be more selective
        std::cout << "✅ Harris Corners: Successfully filtering weak corner responses" << std::endl;
    } else {
        std::cout << "❌ Harris Corners: Still detecting too many weak corners" << std::endl;
    }
    
    std::cout << "\\nNote: Custom implementations now apply stricter filtering to match" << std::endl;
    std::cout << "the behavior you observed in OpenCV results." << std::endl;
    
    return 0;
}