#include <iostream>
#include "opencv2/opencv.hpp"
#include "custom_cv.h"
#include "custom_cv_enhanced.h"

std::vector<cv::Point> FindLocalExtrema(cv::Mat& src)
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
            if (val)  points.push_back(cv::Point(x, y));
        }
    }
    return points;
}

// Enhanced FindLocalExtrema for better rotated shape detection
std::vector<cv::Point> FindLocalExtrema_Optimized(cv::Mat& src, double minThreshold = 0.01)
{
    // Use smaller kernel for better rotated corner detection
    cv::Mat dilatedImg, localMaxImg;
    cv::Size sz(5, 5); // Reduced from 7x7 to 5x5
    cv::Mat rectKernel = cv::getStructuringElement(cv::MORPH_RECT, sz);

    cv::dilate(src, dilatedImg, rectKernel);
    localMaxImg = (src == dilatedImg);

    cv::Mat erodedImg, localMinImg;
    cv::erode(src, erodedImg, rectKernel);
    localMinImg = (src > erodedImg);

    cv::Mat localExtremaImg = (localMaxImg & localMinImg);
    
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

int run_HoughLines_Original()
{
    cv::Mat src = cv::imread("D:/images/lg_building.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다." << std::endl;
        return -1;
    }

    cv::Mat src_out;
    cvtColor(src, src_out, cv::COLOR_GRAY2BGR);

    cv::Mat src_edge;
    cv::Canny(src, src_edge, 170, 200);

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(src_edge, lines, 1, CV_PI / 180, 400);

    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double x0 = rho * cos(theta), y0 = rho * sin(theta);
        pt1.x = round(x0 + 1000 * (-sin(theta)));
        pt1.y = round(y0 + 1000 * (cos(theta)));
        pt2.x = round(x0 - 1000 * (-sin(theta)));
        pt2.y = round(y0 - 1000 * (cos(theta)));

        cv::line(src_out, pt1, pt2, cv::Scalar(0, 0, 255), 2, 8);
    }

    cv::imshow("Original Image", src);
    cv::imshow("Edge Image", src_edge);
    cv::imshow("Line Image", src_out);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

int run_HoughLines_Custom()
{
    cv::Mat src = cv::imread("D:/images/lg_building.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다." << std::endl;
        return -1;
    }

    cv::Mat src_out;
    cvtColor(src, src_out, cv::COLOR_GRAY2BGR);

    cv::Mat src_edge;
    cv::Canny(src, src_edge, 170, 200);

    std::vector<cv::Vec2f> lines;
    custom_cv::HoughLines(src_edge, lines, 1, CV_PI / 180, 80);

    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double x0 = rho * cos(theta), y0 = rho * sin(theta);
        pt1.x = round(x0 + 1000 * (-sin(theta)));
        pt1.y = round(y0 + 1000 * (cos(theta)));
        pt2.x = round(x0 - 1000 * (-sin(theta)));
        pt2.y = round(y0 - 1000 * (cos(theta)));

        cv::line(src_out, pt1, pt2, cv::Scalar(0, 0, 255), 2, 8);
    }

    cv::imshow("Original Image", src);
    cv::imshow("Edge Image", src_edge);
    cv::imshow("Line Image", src_out);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

int run_HarrisCornerDetector_Original()
{
    cv::Mat src = cv::imread("D:/images/shapes1.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다." << std::endl;
        return -1;
    }

    int blockSize = 5;
    int kSize = 3;
    double k = 0.01;

    cv::Mat R;
    cv::cornerHarris(src, R, blockSize, kSize, k);
    cv::threshold(R, R, 0.02, 0, cv::THRESH_TOZERO);

    std::vector<cv::Point> cornerPoints = FindLocalExtrema(R);

    cv::Mat dst(src.size(), CV_8UC3);
    cvtColor(src, dst, cv::COLOR_GRAY2BGR);

    for (const auto& c : cornerPoints) {
        cv::circle(dst, c, 5, cv::Scalar(0, 0, 255), 2);
    }

    std::cout << "OpenCV found " << cornerPoints.size() << " corners" << std::endl;

    cv::imshow("Original Image", src);
    cv::imshow("Result Image", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

int run_HarrisCornerDetector_Custom()
{
    cv::Mat src = cv::imread("D:/images/shapes1.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다." << std::endl;
        return -1;
    }

    int blockSize = 5;
    int kSize = 3;
    double k = 0.01;

    cv::Mat R;
    custom_cv::cornerHarris(src, R, blockSize, kSize, k);
    cv::threshold(R, R, 0.02, 0, cv::THRESH_TOZERO);

    std::vector<cv::Point> cornerPoints = FindLocalExtrema(R);

    cv::Mat dst(src.size(), CV_8UC3);
    cvtColor(src, dst, cv::COLOR_GRAY2BGR);

    for (const auto& c : cornerPoints) {
        cv::circle(dst, c, 5, cv::Scalar(0, 0, 255), 2);
    }

    std::cout << "Custom found " << cornerPoints.size() << " corners" << std::endl;

    cv::imshow("Original Image", src);
    cv::imshow("Result Image", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

int run_HarrisCornerDetector_Enhanced()
{
    cv::Mat src = cv::imread("D:/images/shapes1.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다." << std::endl;
        return -1;
    }

    int blockSize = 5;
    int kSize = 3;
    double k = 0.01;

    std::cout << "\\n=== Enhanced Harris Corner Detection ===" << std::endl;
    cv::Mat R;
    custom_cv_enhanced::cornerHarris(src, R, blockSize, kSize, k);
    cv::threshold(R, R, 0.02, 0, cv::THRESH_TOZERO);

    // Use optimized extrema finding
    std::vector<cv::Point> cornerPoints = FindLocalExtrema_Optimized(R, 0.015);

    cv::Mat dst(src.size(), CV_8UC3);
    cvtColor(src, dst, cv::COLOR_GRAY2BGR);

    for (const auto& c : cornerPoints) {
        cv::circle(dst, c, 5, cv::Scalar(0, 255, 0), 2); // Green for enhanced
    }

    std::cout << "Enhanced found " << cornerPoints.size() << " corners" << std::endl;

    cv::imshow("Original Image", src);
    cv::imshow("Enhanced Result", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

int compare_HarrisCornerDetectors()
{
    cv::Mat src = cv::imread("D:/images/shapes1.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다. 테스트 이미지를 생성합니다." << std::endl;
        
        // Create test image if shapes1.jpg not available
        src = cv::Mat::zeros(400, 400, CV_8UC1);
        
        // Add various shapes for comprehensive testing
        cv::rectangle(src, cv::Point(50, 50), cv::Point(120, 120), cv::Scalar(255), 2);
        
        // Rotated rectangle
        cv::Point2f center(200, 100);
        cv::Size2f size(80, 50);
        cv::RotatedRect rotRect(center, size, 30.0);
        cv::Point2f vertices[4];
        rotRect.points(vertices);
        
        for (int i = 0; i < 4; i++) {
            cv::line(src, vertices[i], vertices[(i+1)%4], cv::Scalar(255), 2);
        }
        
        // Triangle
        std::vector<cv::Point> triangle = {
            cv::Point(300, 200),
            cv::Point(330, 250),
            cv::Point(270, 250)
        };
        cv::fillPoly(src, triangle, cv::Scalar(255));
    }

    int blockSize = 5;
    int kSize = 3;
    double k = 0.01;

    std::cout << "\\n=== Comprehensive Harris Comparison ===" << std::endl;
    
    // OpenCV
    cv::Mat R_opencv;
    cv::cornerHarris(src, R_opencv, blockSize, kSize, k);
    cv::threshold(R_opencv, R_opencv, 0.02, 0, cv::THRESH_TOZERO);
    std::vector<cv::Point> corners_opencv = FindLocalExtrema(R_opencv);
    
    // Original Custom
    cv::Mat R_custom;
    custom_cv::cornerHarris(src, R_custom, blockSize, kSize, k);
    cv::threshold(R_custom, R_custom, 0.02, 0, cv::THRESH_TOZERO);
    std::vector<cv::Point> corners_custom = FindLocalExtrema(R_custom);
    
    // Enhanced Custom
    cv::Mat R_enhanced;
    custom_cv_enhanced::cornerHarris(src, R_enhanced, blockSize, kSize, k);
    cv::threshold(R_enhanced, R_enhanced, 0.02, 0, cv::THRESH_TOZERO);
    std::vector<cv::Point> corners_enhanced = FindLocalExtrema_Optimized(R_enhanced, 0.015);
    
    std::cout << "\\n📊 Corner Detection Results:" << std::endl;
    std::cout << "OpenCV Harris: " << corners_opencv.size() << " corners" << std::endl;
    std::cout << "Custom Harris: " << corners_custom.size() << " corners" << std::endl;
    std::cout << "Enhanced Harris: " << corners_enhanced.size() << " corners" << std::endl;
    
    // Create comparison visualization
    cv::Mat result_comparison(src.rows, src.cols * 3, CV_8UC3);
    
    // OpenCV results (left)
    cv::Mat opencv_result;
    cvtColor(src, opencv_result, cv::COLOR_GRAY2BGR);
    for (const auto& c : corners_opencv) {
        cv::circle(opencv_result, c, 3, cv::Scalar(0, 0, 255), 2);
    }
    opencv_result.copyTo(result_comparison(cv::Rect(0, 0, src.cols, src.rows)));
    
    // Custom results (center)
    cv::Mat custom_result;
    cvtColor(src, custom_result, cv::COLOR_GRAY2BGR);
    for (const auto& c : corners_custom) {
        cv::circle(custom_result, c, 3, cv::Scalar(255, 0, 0), 2);
    }
    custom_result.copyTo(result_comparison(cv::Rect(src.cols, 0, src.cols, src.rows)));
    
    // Enhanced results (right)
    cv::Mat enhanced_result;
    cvtColor(src, enhanced_result, cv::COLOR_GRAY2BGR);
    for (const auto& c : corners_enhanced) {
        cv::circle(enhanced_result, c, 3, cv::Scalar(0, 255, 0), 2);
    }
    enhanced_result.copyTo(result_comparison(cv::Rect(src.cols * 2, 0, src.cols, src.rows)));
    
    // Add labels
    cv::putText(result_comparison, "OpenCV (Red)", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    cv::putText(result_comparison, "Custom (Blue)", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
    cv::putText(result_comparison, "Enhanced (Green)", cv::Point(src.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    cv::imshow("Harris Comparison: OpenCV | Custom | Enhanced", result_comparison);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

int main()
{
    std::cout << "Computer Vision Assignment - Optimized Implementation" << std::endl;
    std::cout << "====================================================" << std::endl;

    int choice;
    while (true) {
        std::cout << "\\nChoose an option:" << std::endl;
        std::cout << "1. Run Original OpenCV HoughLines" << std::endl;
        std::cout << "2. Run Custom HoughLines Implementation" << std::endl;
        std::cout << "3. Run Original OpenCV cornerHarris" << std::endl;
        std::cout << "4. Run Custom cornerHarris Implementation" << std::endl;
        std::cout << "5. Run Enhanced cornerHarris Implementation (BEST FOR ROTATED SHAPES)" << std::endl;
        std::cout << "6. Compare All Harris Implementations Side-by-Side" << std::endl;
        std::cout << "7. Compare Hough Lines (Original vs Custom)" << std::endl;
        std::cout << "0. Exit" << std::endl;
        std::cout << "Enter choice: ";

        std::cin >> choice;

        switch (choice) {
        case 1:
            run_HoughLines_Original();
            break;
        case 2:
            run_HoughLines_Custom();
            break;
        case 3:
            run_HarrisCornerDetector_Original();
            break;
        case 4:
            run_HarrisCornerDetector_Custom();
            break;
        case 5:
            run_HarrisCornerDetector_Enhanced();
            break;
        case 6:
            compare_HarrisCornerDetectors();
            break;
        case 7:
            run_HoughLines_Original();
            run_HoughLines_Custom();
            break;
        case 0:
            std::cout << "Exiting..." << std::endl;
            return 0;
        default:
            std::cout << "Invalid choice. Please try again." << std::endl;
        }
    }

    return 0;
}