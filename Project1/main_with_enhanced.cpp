#include <iostream>
#include "opencv2/opencv.hpp"
#include "custom_cv.h"

// Enhanced FindLocalExtrema for better rotated shape detection
std::vector<cv::Point> FindLocalExtrema_Enhanced(cv::Mat& src, double minThreshold = 0.01)
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

// Original FindLocalExtrema
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

int run_HoughLines_Original()
{
    cv::Mat src = cv::imread("./images/lg_building.jpg", cv::IMREAD_GRAYSCALE);
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

    std::cout << "OpenCV HoughLines 결과: " << lines.size() << "개 직선 검출" << std::endl;

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

    try {
        cv::imshow("Original Image", src);
        cv::imshow("Edge Image", src_edge);
        cv::imshow("Line Image", src_out);
        cv::waitKey(0);
        cv::destroyAllWindows();
    } catch(const cv::Exception& e) {
        std::cout << "디스플레이 사용 불가, 결과만 출력합니다." << std::endl;
    }
    return 0;
}

int run_HoughLines_Custom()
{
    cv::Mat src = cv::imread("./images/lg_building.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다." << std::endl;
        return -1;
    }

    cv::Mat src_out;
    cvtColor(src, src_out, cv::COLOR_GRAY2BGR);

    cv::Mat src_edge;
    cv::Canny(src, src_edge, 170, 200);

    std::vector<cv::Vec2f> lines;
    custom_cv::HoughLines(src_edge, lines, 1, CV_PI / 180.0, 80);  // 최적화된 threshold

    std::cout << "Custom HoughLines 결과: " << lines.size() << "개 직선 검출" << std::endl;

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

    try {
        cv::imshow("Original Image", src);
        cv::imshow("Edge Image", src_edge);
        cv::imshow("Line Image", src_out);
        cv::waitKey(0);
        cv::destroyAllWindows();
    } catch(const cv::Exception& e) {
        std::cout << "디스플레이 사용 불가, 결과만 출력합니다." << std::endl;
    }
    return 0;
}

int run_HarrisCornerDetector_Original()
{
    cv::Mat src = cv::imread("./images/shapes1.jpg", cv::IMREAD_GRAYSCALE);
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

    std::cout << "OpenCV cornerHarris 결과: " << cornerPoints.size() << "개 코너 검출" << std::endl;

    cv::Mat dst(src.size(), CV_8UC3);
    cvtColor(src, dst, cv::COLOR_GRAY2BGR);

    for (const auto& c : cornerPoints) {
        cv::circle(dst, c, 5, cv::Scalar(0, 0, 255), 2);
    }

    try {
        cv::imshow("Original Image", src);
        cv::imshow("Result Image", dst);
        cv::waitKey(0);
        cv::destroyAllWindows();
    } catch(const cv::Exception& e) {
        std::cout << "디스플레이 사용 불가, 결과만 출력합니다." << std::endl;
    }

    return 0;
}

int run_HarrisCornerDetector_Custom()
{
    cv::Mat src = cv::imread("./images/shapes1.jpg", cv::IMREAD_GRAYSCALE);
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

    std::cout << "Custom cornerHarris 결과: " << cornerPoints.size() << "개 코너 검출" << std::endl;

    cv::Mat dst(src.size(), CV_8UC3);
    cvtColor(src, dst, cv::COLOR_GRAY2BGR);

    for (const auto& c : cornerPoints) {
        cv::circle(dst, c, 5, cv::Scalar(0, 0, 255), 2);
    }

    try {
        cv::imshow("Original Image", src);
        cv::imshow("Result Image", dst);
        cv::waitKey(0);
        cv::destroyAllWindows();
    } catch(const cv::Exception& e) {
        std::cout << "디스플레이 사용 불가, 결과만 출력합니다." << std::endl;
    }

    return 0;
}

int run_HarrisCornerDetector_Enhanced()
{
    cv::Mat src = cv::imread("./images/shapes1.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다." << std::endl;
        return -1;
    }

    int blockSize = 5;
    int kSize = 3;
    double k = 0.01;

    std::cout << "Enhanced Harris Corner Detection 실행 중..." << std::endl;
    
    cv::Mat R;
    custom_cv::cornerHarris(src, R, blockSize, kSize, k);
    cv::threshold(R, R, 0.015, 0, cv::THRESH_TOZERO);  // 더 낮은 threshold

    // Enhanced FindLocalExtrema 사용
    std::vector<cv::Point> cornerPoints = FindLocalExtrema_Enhanced(R, 0.01);

    std::cout << "Enhanced cornerHarris 결과: " << cornerPoints.size() << "개 코너 검출" << std::endl;

    cv::Mat dst(src.size(), CV_8UC3);
    cvtColor(src, dst, cv::COLOR_GRAY2BGR);

    for (const auto& c : cornerPoints) {
        cv::circle(dst, c, 5, cv::Scalar(0, 255, 0), 2);  // 녹색으로 표시
    }

    try {
        cv::imshow("Original Image", src);
        cv::imshow("Enhanced Result Image", dst);
        cv::waitKey(0);
        cv::destroyAllWindows();
    } catch(const cv::Exception& e) {
        std::cout << "디스플레이 사용 불가, 결과만 출력합니다." << std::endl;
    }

    return 0;
}

int main()
{
    std::cout << "Computer Vision Assignment - 향상된 구현" << std::endl;
    std::cout << "=================================================" << std::endl;

    int choice;
    while (true) {
        std::cout << "\n옵션을 선택하세요:" << std::endl;
        std::cout << "1. OpenCV HoughLines 실행" << std::endl;
        std::cout << "2. Custom HoughLines 실행 (최적화됨)" << std::endl;
        std::cout << "3. OpenCV cornerHarris 실행" << std::endl;
        std::cout << "4. Custom cornerHarris 실행" << std::endl;
        std::cout << "5. Enhanced cornerHarris 실행 (회전된 도형 최적화)" << std::endl;
        std::cout << "6. Hough Lines 비교 (OpenCV vs Custom)" << std::endl;
        std::cout << "7. Harris Corners 비교 (OpenCV vs Custom vs Enhanced)" << std::endl;
        std::cout << "0. 종료" << std::endl;
        std::cout << "선택: ";

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
            std::cout << "\n=== HoughLines 비교 ===" << std::endl;
            run_HoughLines_Original();
            run_HoughLines_Custom();
            break;
        case 7:
            std::cout << "\n=== Harris Corners 비교 ===" << std::endl;
            run_HarrisCornerDetector_Original();
            run_HarrisCornerDetector_Custom();
            run_HarrisCornerDetector_Enhanced();
            break;
        case 0:
            std::cout << "프로그램을 종료합니다..." << std::endl;
            return 0;
        default:
            std::cout << "잘못된 선택입니다. 다시 시도해주세요." << std::endl;
        }
    }

    return 0;
}