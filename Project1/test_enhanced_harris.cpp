#include <iostream>
#include "opencv2/opencv.hpp"
#include "custom_cv_enhanced.h"

// Original FindLocalExtrema for comparison
std::vector<cv::Point> FindLocalExtrema_Original(cv::Mat& src) {
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
            if (val) points.push_back(cv::Point(x, y));
        }
    }
    return points;
}

int main() {
    std::cout << "=== Enhanced Harris Corner Detection Test ===" << std::endl;
    
    // Create comprehensive test image
    cv::Mat testImg = cv::Mat::zeros(500, 500, CV_8UC1);
    
    // 1. Regular shapes
    cv::rectangle(testImg, cv::Point(50, 50), cv::Point(120, 120), cv::Scalar(255), 2);
    std::cout << "✓ Added axis-aligned rectangle" << std::endl;
    
    // 2. Multiple rotated rectangles at different angles
    for (int angle = 15; angle <= 75; angle += 15) {
        cv::Point2f center(150 + angle * 3, 200);
        cv::Size2f size(60, 40);
        
        cv::RotatedRect rotRect(center, size, angle);
        cv::Point2f vertices[4];
        rotRect.points(vertices);
        
        for (int i = 0; i < 4; i++) {
            cv::line(testImg, vertices[i], vertices[(i+1)%4], cv::Scalar(255), 2);
        }
    }
    std::cout << "✓ Added rotated rectangles (15°, 30°, 45°, 60°, 75°)" << std::endl;
    
    // 3. Rotated triangles
    for (int angle = 0; angle <= 90; angle += 30) {
        float rad = angle * CV_PI / 180.0;
        cv::Point center(100 + angle * 2, 350);
        int radius = 30;
        
        std::vector<cv::Point> triangle;
        for (int i = 0; i < 3; i++) {
            float a = rad + i * 2 * CV_PI / 3;
            triangle.push_back(cv::Point(
                center.x + radius * cos(a),
                center.y + radius * sin(a)
            ));
        }
        
        for (int i = 0; i < 3; i++) {
            cv::line(testImg, triangle[i], triangle[(i+1)%3], cv::Scalar(255), 2);
        }
    }
    std::cout << "✓ Added rotated triangles (0°, 30°, 60°, 90°)" << std::endl;
    
    // 4. Complex L and T shapes
    cv::line(testImg, cv::Point(400, 50), cv::Point(400, 120), cv::Scalar(255), 3);
    cv::line(testImg, cv::Point(400, 120), cv::Point(470, 120), cv::Scalar(255), 3);
    
    cv::line(testImg, cv::Point(420, 180), cv::Point(470, 180), cv::Scalar(255), 3);
    cv::line(testImg, cv::Point(445, 180), cv::Point(445, 250), cv::Scalar(255), 3);
    std::cout << "✓ Added L and T shapes" << std::endl;
    
    // Test parameters
    int blockSize = 5;
    int kSize = 3;
    double k = 0.01;
    
    std::cout << "\\n🔬 Testing OpenCV Harris:" << std::endl;
    cv::Mat R_opencv;
    cv::cornerHarris(testImg, R_opencv, blockSize, kSize, k);
    cv::threshold(R_opencv, R_opencv, 0.02, 0, cv::THRESH_TOZERO);
    
    std::vector<cv::Point> corners_opencv = FindLocalExtrema_Original(R_opencv);
    std::cout << "OpenCV found " << corners_opencv.size() << " corners" << std::endl;
    
    std::cout << "\\n🔬 Testing Enhanced Harris:" << std::endl;
    cv::Mat R_enhanced;
    custom_cv_enhanced::cornerHarris(testImg, R_enhanced, blockSize, kSize, k);
    cv::threshold(R_enhanced, R_enhanced, 0.02, 0, cv::THRESH_TOZERO);
    
    // Test both original and enhanced extrema finding
    std::cout << "\\n--- Using Original FindLocalExtrema ---" << std::endl;
    std::vector<cv::Point> corners_enhanced_orig = FindLocalExtrema_Original(R_enhanced);
    std::cout << "Enhanced Harris + Original Extrema: " << corners_enhanced_orig.size() << " corners" << std::endl;
    
    std::cout << "\\n--- Using Enhanced FindLocalExtrema ---" << std::endl;
    std::vector<cv::Point> corners_enhanced_new = custom_cv_enhanced::FindLocalExtrema_Enhanced(R_enhanced, 0.01, 5, true);
    std::cout << "Enhanced Harris + Enhanced Extrema: " << corners_enhanced_new.size() << " corners" << std::endl;
    
    // Analyze corner quality in different regions
    auto analyzeRegion = [&](const std::vector<cv::Point>& corners, const std::string& name) {
        std::cout << "\\n" << name << " corner distribution:" << std::endl;
        
        int rotated_shapes = 0, regular_shapes = 0, triangles = 0, l_t_shapes = 0;
        
        for (const auto& corner : corners) {
            if (corner.x >= 40 && corner.x <= 130 && corner.y >= 40 && corner.y <= 130) {
                regular_shapes++;
            } else if (corner.x >= 140 && corner.x <= 380 && corner.y >= 160 && corner.y <= 240) {
                rotated_shapes++;
            } else if (corner.x >= 80 && corner.x <= 280 && corner.y >= 320 && corner.y <= 380) {
                triangles++;
            } else if (corner.x >= 390 && corner.x <= 480 && corner.y >= 40 && corner.y <= 260) {
                l_t_shapes++;
            }
        }
        
        std::cout << "  Regular shapes: " << regular_shapes << std::endl;
        std::cout << "  Rotated rectangles: " << rotated_shapes << std::endl;
        std::cout << "  Triangles: " << triangles << std::endl;
        std::cout << "  L/T shapes: " << l_t_shapes << std::endl;
        
        return rotated_shapes + triangles; // Count problematic cases
    };
    
    int opencv_rotated = analyzeRegion(corners_opencv, "OpenCV");
    int enhanced_orig_rotated = analyzeRegion(corners_enhanced_orig, "Enhanced+OrigExtrema");
    int enhanced_new_rotated = analyzeRegion(corners_enhanced_new, "Enhanced+NewExtrema");
    
    std::cout << "\\n📊 Performance Summary:" << std::endl;
    std::cout << "OpenCV rotated detection: " << opencv_rotated << " corners" << std::endl;
    std::cout << "Enhanced+Original rotated detection: " << enhanced_orig_rotated << " corners" << std::endl;
    std::cout << "Enhanced+New rotated detection: " << enhanced_new_rotated << " corners" << std::endl;
    
    // Determine the best approach
    if (enhanced_new_rotated >= opencv_rotated && enhanced_new_rotated >= enhanced_orig_rotated) {
        std::cout << "\\n🏆 Enhanced Harris + Enhanced Extrema performs best!" << std::endl;
    } else if (enhanced_orig_rotated >= opencv_rotated) {
        std::cout << "\\n🏆 Enhanced Harris + Original Extrema performs well!" << std::endl;
    } else {
        std::cout << "\\n⚠️ OpenCV still performs better in some cases" << std::endl;
    }
    
    // Test different parameter combinations for problematic cases
    std::cout << "\\n🧪 Testing parameter sensitivity:" << std::endl;
    
    struct TestParams {
        int blockSize;
        int kSize;
        double k;
        std::string name;
    };
    
    std::vector<TestParams> params = {
        {3, 3, 0.01, "Small window (3x3)"},
        {7, 3, 0.01, "Large window (7x7)"},
        {5, 5, 0.01, "Large Sobel (5x5)"},
        {5, 3, 0.005, "Small k (0.005)"},
        {5, 3, 0.02, "Large k (0.02)"}
    };
    
    for (const auto& param : params) {
        cv::Mat R_test;
        custom_cv_enhanced::cornerHarris(testImg, R_test, param.blockSize, param.kSize, param.k);
        cv::threshold(R_test, R_test, 0.02, 0, cv::THRESH_TOZERO);
        
        std::vector<cv::Point> corners_test = custom_cv_enhanced::FindLocalExtrema_Enhanced(R_test, 0.01, 5, true);
        int rotated_test = 0;
        
        for (const auto& corner : corners_test) {
            if ((corner.x >= 140 && corner.x <= 380 && corner.y >= 160 && corner.y <= 240) ||
                (corner.x >= 80 && corner.x <= 280 && corner.y >= 320 && corner.y <= 380)) {
                rotated_test++;
            }
        }
        
        std::cout << param.name << ": " << corners_test.size() << " total, " 
                  << rotated_test << " rotated" << std::endl;
    }
    
    // Show visualization if possible
    try {
        cv::Mat result_opencv, result_enhanced;
        cv::cvtColor(testImg, result_opencv, cv::COLOR_GRAY2BGR);
        cv::cvtColor(testImg, result_enhanced, cv::COLOR_GRAY2BGR);
        
        for (const auto& corner : corners_opencv) {
            cv::circle(result_opencv, corner, 3, cv::Scalar(0, 0, 255), 2);
        }
        for (const auto& corner : corners_enhanced_new) {
            cv::circle(result_enhanced, corner, 3, cv::Scalar(0, 255, 0), 2);
        }
        
        cv::imshow("Test Image", testImg);
        cv::imshow("OpenCV Corners (Red)", result_opencv);
        cv::imshow("Enhanced Corners (Green)", result_enhanced);
        
        custom_cv_enhanced::showHarrisResponse(R_opencv, "OpenCV Response");
        custom_cv_enhanced::showHarrisResponse(R_enhanced, "Enhanced Response");
        
        std::cout << "\\nPress any key to continue..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    } catch (const cv::Exception& e) {
        std::cout << "Display not available, skipping visualization." << std::endl;
    }
    
    return 0;
}