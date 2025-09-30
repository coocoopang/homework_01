#include <iostream>
#include "opencv2/opencv.hpp"
#include "custom_cv.h"

// Improved FindLocalExtrema with stricter filtering
std::vector<cv::Point> FindLocalExtrema(cv::Mat& src, double minThreshold = 0.1)
{
    cv::Mat dilatedImg, localMaxImg;
    cv::Size sz(5, 5); // Smaller kernel for more precise detection
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
            
            // Only accept strong corner responses
            if (val && response >= minThreshold) {
                points.push_back(cv::Point(x, y));
            }
        }
    }
    return points;
}

int run_HoughLines()
{
    std::cout << "=== Testing Hough Lines (Original vs Custom) ===" << std::endl;
    
    // Create or load test image
    cv::Mat src;
    
    // Try to load image, create synthetic if not found
    src = cv::imread("images/lg_building.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cout << "Creating synthetic test image for Hough Lines..." << std::endl;
        src = cv::Mat::zeros(400, 400, CV_8UC1);
        
        // Draw strong horizontal and vertical lines
        cv::line(src, cv::Point(50, 100), cv::Point(350, 100), cv::Scalar(255), 3);
        cv::line(src, cv::Point(50, 200), cv::Point(350, 200), cv::Scalar(255), 3);
        cv::line(src, cv::Point(100, 50), cv::Point(100, 350), cv::Scalar(255), 3);
        cv::line(src, cv::Point(300, 50), cv::Point(300, 350), cv::Scalar(255), 3);
        
        // Add some rectangles to create more horizontal/vertical lines
        cv::rectangle(src, cv::Point(150, 250), cv::Point(250, 320), cv::Scalar(255), 2);
    }

    cv::Mat src_out_original, src_out_custom;
    cvtColor(src, src_out_original, cv::COLOR_GRAY2BGR);
    cvtColor(src, src_out_custom, cv::COLOR_GRAY2BGR);

    cv::Mat src_edge;
    cv::Canny(src, src_edge, 100, 200); // Stronger edge detection

    // Original OpenCV implementation
    std::vector<cv::Vec2f> lines_original;
    cv::HoughLines(src_edge, lines_original, 1, CV_PI / 180, 100); // Lower threshold for comparison

    for (size_t i = 0; i < lines_original.size(); i++) {
        float rho = lines_original[i][0], theta = lines_original[i][1];
        cv::Point pt1, pt2;
        double x0 = rho * cos(theta), y0 = rho * sin(theta);
        pt1.x = cvRound(x0 + 1000 * (-sin(theta)));
        pt1.y = cvRound(y0 + 1000 * (cos(theta)));
        pt2.x = cvRound(x0 - 1000 * (-sin(theta)));
        pt2.y = cvRound(y0 - 1000 * (cos(theta)));

        cv::line(src_out_original, pt1, pt2, cv::Scalar(0, 0, 255), 2, 8);
    }

    // Custom implementation
    std::vector<cv::Vec2f> lines_custom;
    custom_cv::HoughLines(src_edge, lines_custom, 1, CV_PI / 180, 100);

    for (size_t i = 0; i < lines_custom.size(); i++) {
        float rho = lines_custom[i][0], theta = lines_custom[i][1];
        cv::Point pt1, pt2;
        double x0 = rho * cos(theta), y0 = rho * sin(theta);
        pt1.x = cvRound(x0 + 1000 * (-sin(theta)));
        pt1.y = cvRound(y0 + 1000 * (cos(theta)));
        pt2.x = cvRound(x0 - 1000 * (-sin(theta)));
        pt2.y = cvRound(y0 - 1000 * (cos(theta)));

        cv::line(src_out_custom, pt1, pt2, cv::Scalar(0, 255, 0), 2, 8); // Green for custom
    }

    std::cout << "Original OpenCV found: " << lines_original.size() << " lines" << std::endl;
    std::cout << "Custom implementation found: " << lines_custom.size() << " lines" << std::endl;

    cv::imshow("Original Image", src);
    cv::imshow("Edge Image", src_edge);
    cv::imshow("Original Hough Lines", src_out_original);
    cv::imshow("Custom Hough Lines", src_out_custom);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}

int run_HarrisCornerDetector()
{
    std::cout << "\\n=== Testing Harris Corner Detector (Original vs Custom) ===" << std::endl;
    
    // Create or load test image
    cv::Mat src;
    
    // Try to load image, create synthetic if not found
    src = cv::imread("images/shapes1.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cout << "Creating synthetic test image for Harris Corners..." << std::endl;
        src = cv::Mat::zeros(400, 400, CV_8UC1);
        
        // Draw shapes with clear corners
        cv::rectangle(src, cv::Point(50, 50), cv::Point(150, 150), cv::Scalar(255), 2);
        cv::rectangle(src, cv::Point(200, 100), cv::Point(350, 200), cv::Scalar(255), -1);
        
        // Triangle with clear corner at apex
        std::vector<cv::Point> triangle;
        triangle.push_back(cv::Point(100, 250));
        triangle.push_back(cv::Point(200, 250));
        triangle.push_back(cv::Point(150, 300));
        cv::fillPoly(src, triangle, cv::Scalar(255));
        
        // Circle (should NOT produce corners)
        cv::circle(src, cv::Point(300, 320), 30, cv::Scalar(255), 2);
        
        // L-shape (should produce one strong corner)
        cv::line(src, cv::Point(250, 250), cv::Point(250, 350), cv::Scalar(255), 4);
        cv::line(src, cv::Point(250, 350), cv::Point(350, 350), cv::Scalar(255), 4);
    }

    int blockSize = 5;
    int kSize = 3;
    double k = 0.04; // Standard Harris parameter

    // Original OpenCV implementation
    cv::Mat R_original;
    cv::cornerHarris(src, R_original, blockSize, kSize, k);
    
    // Apply threshold similar to original
    double maxVal_orig;
    cv::minMaxLoc(R_original, nullptr, &maxVal_orig);
    cv::threshold(R_original, R_original, 0.01 * maxVal_orig, 0, cv::THRESH_TOZERO);

    std::vector<cv::Point> cornerPoints_original = FindLocalExtrema(R_original, 0.01 * maxVal_orig);

    cv::Mat dst_original(src.size(), CV_8UC3);
    cvtColor(src, dst_original, cv::COLOR_GRAY2BGR);

    for (const auto& c : cornerPoints_original) {
        cv::circle(dst_original, c, 5, cv::Scalar(0, 0, 255), 2);
    }

    // Custom implementation
    cv::Mat R_custom;
    custom_cv::cornerHarris(src, R_custom, blockSize, kSize, k);

    std::vector<cv::Point> cornerPoints_custom = FindLocalExtrema(R_custom, 0.1); // Stricter threshold

    cv::Mat dst_custom(src.size(), CV_8UC3);
    cvtColor(src, dst_custom, cv::COLOR_GRAY2BGR);

    for (const auto& c : cornerPoints_custom) {
        cv::circle(dst_custom, c, 5, cv::Scalar(0, 255, 0), 2); // Green for custom
    }

    std::cout << "Original OpenCV found: " << cornerPoints_original.size() << " corners" << std::endl;
    std::cout << "Custom implementation found: " << cornerPoints_custom.size() << " corners" << std::endl;

    // Show normalized response for comparison
    cv::Mat R_orig_display, R_custom_display;
    cv::normalize(R_original, R_orig_display, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    R_custom.convertTo(R_custom_display, CV_8UC1, 255.0);

    cv::imshow("Original Image", src);
    cv::imshow("Original Harris Response", R_orig_display);
    cv::imshow("Custom Harris Response", R_custom_display);
    cv::imshow("Original Harris Corners", dst_original);
    cv::imshow("Custom Harris Corners", dst_custom);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

int main()
{
    std::cout << "Computer Vision Assignment - Corrected Custom Implementation" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "\\nCorrections applied:" << std::endl;
    std::cout << "1. Hough Lines: Strict horizontal/vertical line filtering" << std::endl;
    std::cout << "2. Harris Corners: Strong corner filtering to eliminate curve responses" << std::endl;
    
    int choice;
    while (true) {
        std::cout << "\\nChoose an option:" << std::endl;
        std::cout << "1. Test Hough Lines (Original vs Custom)" << std::endl;
        std::cout << "2. Test Harris Corners (Original vs Custom)" << std::endl;
        std::cout << "3. Test Both" << std::endl;
        std::cout << "0. Exit" << std::endl;
        std::cout << "Enter choice: ";
        
        std::cin >> choice;
        
        switch (choice) {
        case 1:
            run_HoughLines();
            break;
        case 2:
            run_HarrisCornerDetector();
            break;
        case 3:
            run_HoughLines();
            run_HarrisCornerDetector();
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