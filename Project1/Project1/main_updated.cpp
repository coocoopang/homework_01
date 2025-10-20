#include <iostream>
#include "opencv2/opencv.hpp"
#include "custom_cv.h"

// Find local extrema for corner detection (existing helper function)
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
            if (val) points.push_back(cv::Point(x, y));
        }
    }
    return points;
}

int run_HoughLines_Original()
{
    std::cout << "=== Running Original OpenCV HoughLines ===" << std::endl;
    
    // Try different possible image paths
    std::vector<std::string> imagePaths = {
        "images/lg_building.jpg",
        "images/building.jpg", 
        "images/test_building.jpg"
    };
    
    cv::Mat src;
    std::string usedPath;
    
    for (const auto& path : imagePaths) {
        src = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (!src.empty()) {
            usedPath = path;
            break;
        }
    }
    
    // If no image found, create a simple test image
    if (src.empty()) {
        std::cout << "No test image found. Creating synthetic test image..." << std::endl;
        src = cv::Mat::zeros(400, 400, CV_8UC1);
        
        // Draw some lines for testing
        cv::line(src, cv::Point(50, 50), cv::Point(350, 50), cv::Scalar(255), 2);
        cv::line(src, cv::Point(100, 100), cv::Point(100, 350), cv::Scalar(255), 2);
        cv::line(src, cv::Point(200, 150), cv::Point(350, 300), cv::Scalar(255), 2);
        cv::rectangle(src, cv::Point(150, 200), cv::Point(250, 300), cv::Scalar(255), 2);
        
        usedPath = "synthetic_test_image";
    }

    cv::Mat src_out;
    cvtColor(src, src_out, cv::COLOR_GRAY2BGR);

    cv::Mat src_edge;
    cv::Canny(src, src_edge, 50, 150);

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(src_edge, lines, 1, CV_PI / 180, 80);  // Lower threshold for synthetic image

    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double x0 = rho * cos(theta), y0 = rho * sin(theta);
        pt1.x = cvRound(x0 + 1000 * (-sin(theta)));
        pt1.y = cvRound(y0 + 1000 * (cos(theta)));
        pt2.x = cvRound(x0 - 1000 * (-sin(theta)));
        pt2.y = cvRound(y0 - 1000 * (cos(theta)));

        cv::line(src_out, pt1, pt2, cv::Scalar(0, 0, 255), 2, 8);
    }

    std::cout << "Original OpenCV found " << lines.size() << " lines" << std::endl;

    cv::imshow("Original - Source Image", src);
    cv::imshow("Original - Edge Image", src_edge);
    cv::imshow("Original - Lines Result", src_out);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}

int run_HoughLines_Custom()
{
    std::cout << "\n=== Running Custom HoughLines Implementation ===" << std::endl;
    
    // Try different possible image paths
    std::vector<std::string> imagePaths = {
        "images/lg_building.jpg",
        "images/building.jpg", 
        "images/test_building.jpg"
    };
    
    cv::Mat src;
    std::string usedPath;
    
    for (const auto& path : imagePaths) {
        src = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (!src.empty()) {
            usedPath = path;
            break;
        }
    }
    
    // If no image found, create a simple test image
    if (src.empty()) {
        std::cout << "No test image found. Creating synthetic test image..." << std::endl;
        src = cv::Mat::zeros(400, 400, CV_8UC1);
        
        // Draw some lines for testing
        cv::line(src, cv::Point(50, 50), cv::Point(350, 50), cv::Scalar(255), 2);
        cv::line(src, cv::Point(100, 100), cv::Point(100, 350), cv::Scalar(255), 2);
        cv::line(src, cv::Point(200, 150), cv::Point(350, 300), cv::Scalar(255), 2);
        cv::rectangle(src, cv::Point(150, 200), cv::Point(250, 300), cv::Scalar(255), 2);
        
        usedPath = "synthetic_test_image";
    }

    cv::Mat src_out;
    cvtColor(src, src_out, cv::COLOR_GRAY2BGR);

    cv::Mat src_edge;
    cv::Canny(src, src_edge, 50, 150);

    std::vector<cv::Vec2f> lines;
    // Use our custom implementation instead of cv::HoughLines
    custom_cv::HoughLines(src_edge, lines, 1, CV_PI / 180, 80);

    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double x0 = rho * cos(theta), y0 = rho * sin(theta);
        pt1.x = cvRound(x0 + 1000 * (-sin(theta)));
        pt1.y = cvRound(y0 + 1000 * (cos(theta)));
        pt2.x = cvRound(x0 - 1000 * (-sin(theta)));
        pt2.y = cvRound(y0 - 1000 * (cos(theta)));

        cv::line(src_out, pt1, pt2, cv::Scalar(0, 255, 0), 2, 8);  // Green lines for custom
    }

    std::cout << "Custom implementation found " << lines.size() << " lines" << std::endl;

    cv::imshow("Custom - Source Image", src);
    cv::imshow("Custom - Edge Image", src_edge);
    cv::imshow("Custom - Lines Result", src_out);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}

int run_HarrisCornerDetector_Original()
{
    std::cout << "\n=== Running Original OpenCV cornerHarris ===" << std::endl;
    
    // Try different possible image paths
    std::vector<std::string> imagePaths = {
        "images/shapes1.jpg",
        "images/shapes.jpg", 
        "images/corners.jpg",
        "images/test_corners.jpg"
    };
    
    cv::Mat src;
    std::string usedPath;
    
    for (const auto& path : imagePaths) {
        src = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (!src.empty()) {
            usedPath = path;
            break;
        }
    }
    
    // If no image found, create a simple test image with corners
    if (src.empty()) {
        std::cout << "No test image found. Creating synthetic test image with corners..." << std::endl;
        src = cv::Mat::zeros(400, 400, CV_8UC1);
        
        // Draw shapes with corners
        cv::rectangle(src, cv::Point(50, 50), cv::Point(150, 150), cv::Scalar(255), 2);
        cv::rectangle(src, cv::Point(200, 100), cv::Point(350, 200), cv::Scalar(255), -1);
        
        // Triangle
        std::vector<cv::Point> triangle;
        triangle.push_back(cv::Point(100, 250));
        triangle.push_back(cv::Point(200, 250));
        triangle.push_back(cv::Point(150, 300));
        cv::fillPoly(src, triangle, cv::Scalar(255));
        
        usedPath = "synthetic_corner_test_image";
    }

    int blockSize = 5;
    int kSize = 3;
    double k = 0.04;

    cv::Mat R;
    cv::cornerHarris(src, R, blockSize, kSize, k);
    cv::threshold(R, R, 0.01 * R.at<float>(0, 0), 0, cv::THRESH_TOZERO);

    std::vector<cv::Point> cornerPoints = FindLocalExtrema(R);

    cv::Mat dst(src.size(), CV_8UC3);
    cvtColor(src, dst, cv::COLOR_GRAY2BGR);

    for (const auto& c : cornerPoints) {
        cv::circle(dst, c, 5, cv::Scalar(0, 0, 255), 2);
    }

    std::cout << "Original OpenCV found " << cornerPoints.size() << " corners" << std::endl;

    cv::imshow("Original - Source Image", src);
    cv::imshow("Original - Corner Response", R);
    cv::imshow("Original - Corners Result", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

int run_HarrisCornerDetector_Custom()
{
    std::cout << "\n=== Running Custom cornerHarris Implementation ===" << std::endl;
    
    // Try different possible image paths
    std::vector<std::string> imagePaths = {
        "images/shapes1.jpg",
        "images/shapes.jpg", 
        "images/corners.jpg",
        "images/test_corners.jpg"
    };
    
    cv::Mat src;
    std::string usedPath;
    
    for (const auto& path : imagePaths) {
        src = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (!src.empty()) {
            usedPath = path;
            break;
        }
    }
    
    // If no image found, create a simple test image with corners
    if (src.empty()) {
        std::cout << "No test image found. Creating synthetic test image with corners..." << std::endl;
        src = cv::Mat::zeros(400, 400, CV_8UC1);
        
        // Draw shapes with corners
        cv::rectangle(src, cv::Point(50, 50), cv::Point(150, 150), cv::Scalar(255), 2);
        cv::rectangle(src, cv::Point(200, 100), cv::Point(350, 200), cv::Scalar(255), -1);
        
        // Triangle
        std::vector<cv::Point> triangle;
        triangle.push_back(cv::Point(100, 250));
        triangle.push_back(cv::Point(200, 250));
        triangle.push_back(cv::Point(150, 300));
        cv::fillPoly(src, triangle, cv::Scalar(255));
        
        usedPath = "synthetic_corner_test_image";
    }

    int blockSize = 5;
    int kSize = 3;
    double k = 0.04;

    cv::Mat R;
    // Use our custom implementation instead of cv::cornerHarris
    custom_cv::cornerHarris(src, R, blockSize, kSize, k);
    
    // Normalize for display
    cv::Mat R_norm;
    cv::normalize(R, R_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    
    // Threshold
    cv::threshold(R, R, 0.01 * cv::norm(R, cv::NORM_INF), 0, cv::THRESH_TOZERO);

    std::vector<cv::Point> cornerPoints = FindLocalExtrema(R);

    cv::Mat dst(src.size(), CV_8UC3);
    cvtColor(src, dst, cv::COLOR_GRAY2BGR);

    for (const auto& c : cornerPoints) {
        cv::circle(dst, c, 5, cv::Scalar(0, 255, 0), 2);  // Green circles for custom
    }

    std::cout << "Custom implementation found " << cornerPoints.size() << " corners" << std::endl;

    cv::imshow("Custom - Source Image", src);
    cv::imshow("Custom - Corner Response", R_norm);
    cv::imshow("Custom - Corners Result", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

int main()
{
    std::cout << "Computer Vision Assignment - Custom Implementation" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    int choice;
    while (true) {
        std::cout << "\nChoose an option:" << std::endl;
        std::cout << "1. Run Original OpenCV HoughLines" << std::endl;
        std::cout << "2. Run Custom HoughLines Implementation" << std::endl;
        std::cout << "3. Run Original OpenCV cornerHarris" << std::endl;
        std::cout << "4. Run Custom cornerHarris Implementation" << std::endl;
        std::cout << "5. Compare Hough Lines (Original vs Custom)" << std::endl;
        std::cout << "6. Compare Harris Corners (Original vs Custom)" << std::endl;
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
            run_HoughLines_Original();
            run_HoughLines_Custom();
            break;
        case 6:
            run_HarrisCornerDetector_Original();
            run_HarrisCornerDetector_Custom();
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