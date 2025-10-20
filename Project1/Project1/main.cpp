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
        std::cerr << "�̹����� �ҷ��� �� �����ϴ�." << std::endl;
        return -1;
    }

    cv::Mat src_out;
    cvtColor(src, src_out, cv::COLOR_GRAY2BGR);

    cv::Mat src_edge;
    cv::Canny(src, src_edge, 170, 200);

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(src_edge, lines, 1, CV_PI / 180, 400);

    std::cout << "OpenCV HoughLines ���: " << lines.size() << "�� ���� ����" << std::endl;

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
    }
    catch (const cv::Exception& e) {
        std::cout << "���÷��� ��� �Ұ�, ����� ����մϴ�." << std::endl;
    }
    return 0;

}

int run_HoughLines_Custom()
{
    cv::Mat src = cv::imread("./images/lg_building.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "�̹����� �ҷ��� �� �����ϴ�." << std::endl;
        return -1;
    }

    cv::Mat src_out;
    cvtColor(src, src_out, cv::COLOR_GRAY2BGR);

    cv::Mat src_edge;
    cv::Canny(src, src_edge, 170, 200);


    std::vector<cv::Vec2f> lines;
    custom_cv::HoughLines(src_edge, lines, 1, CV_PI / 180.0, 80);  // ����ȭ�� threshold

    std::cout << "Custom HoughLines ���: " << lines.size() << "�� ���� ����" << std::endl;

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
    }
    catch (const cv::Exception& e) {
        std::cout << "���÷��� ��� �Ұ�, ����� ����մϴ�." << std::endl;
    }
    return 0;
}

int run_HarrisCornerDetector_Original()
{
    cv::Mat src = cv::imread("./images/shapes1.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "�̹����� �ҷ��� �� �����ϴ�." << std::endl;
        return -1;
    }

    int blockSize = 5;
    int kSize = 3;
    double k = 0.01;

    cv::Mat R;
    cv::cornerHarris(src, R, blockSize, kSize, k);
    cv::threshold(R, R, 0.02, 0, cv::THRESH_TOZERO);

    std::vector<cv::Point> cornerPoints = FindLocalExtrema(R);

    std::cout << "OpenCV cornerHarris ���: " << cornerPoints.size() << "�� �ڳ� ����" << std::endl;

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
    }
    catch (const cv::Exception& e) {
        std::cout << "���÷��� ��� �Ұ�, ����� ����մϴ�." << std::endl;
    }

    return 0;
}

int run_HarrisCornerDetector_Custom()
{
    cv::Mat src = cv::imread("./images/shapes1.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "�̹����� �ҷ��� �� �����ϴ�." << std::endl;
        return -1;
    }

    int blockSize = 5;
    int kSize = 3;
    double k = 0.01;

    cv::Mat R;
    custom_cv::cornerHarris(src, R, blockSize, kSize, k);
    cv::threshold(R, R, 0.02, 0, cv::THRESH_TOZERO);

    std::vector<cv::Point> cornerPoints = FindLocalExtrema(R);

    std::cout << "Custom cornerHarris ���: " << cornerPoints.size() << "�� �ڳ� ����" << std::endl;

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
    }
    catch (const cv::Exception& e) {
        std::cout << "���÷��� ��� �Ұ�, ����� ����մϴ�." << std::endl;
    }

    return 0;
}

int run_HarrisCornerDetector_Enhanced()
{
    cv::Mat src = cv::imread("./images/shapes1.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "�̹����� �ҷ��� �� �����ϴ�." << std::endl;
        return -1;
    }

    int blockSize = 5;
    int kSize = 3;
    double k = 0.01;

    std::cout << "Enhanced Harris Corner Detection ���� ��..." << std::endl;

    cv::Mat R;
    custom_cv::cornerHarris(src, R, blockSize, kSize, k);
    cv::threshold(R, R, 0.015, 0, cv::THRESH_TOZERO);  // �� ���� threshold

    // Enhanced FindLocalExtrema ���
    std::vector<cv::Point> cornerPoints = FindLocalExtrema_Enhanced(R, 0.01);

    std::cout << "Enhanced cornerHarris ���: " << cornerPoints.size() << "�� �ڳ� ����" << std::endl;

    cv::Mat dst(src.size(), CV_8UC3);
    cvtColor(src, dst, cv::COLOR_GRAY2BGR);

    for (const auto& c : cornerPoints) {
        cv::circle(dst, c, 5, cv::Scalar(0, 255, 0), 2);  // ������� ǥ��
    }

    try {
        cv::imshow("Original Image", src);
        cv::imshow("Enhanced Result Image", dst);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    catch (const cv::Exception& e) {
        std::cout << "���÷��� ��� �Ұ�, ����� ����մϴ�." << std::endl;
    }

    return 0;
}

int main()
{
    std::cout << "Computer Vision Assignment - ���� ����" << std::endl;
    std::cout << "=================================================" << std::endl;

    int choice;
    while (true) {
        std::cout << "\n�ɼ��� �����ϼ���:" << std::endl;
        std::cout << "1. OpenCV HoughLines ����" << std::endl;
        std::cout << "2. Custom HoughLines ���� (����ȭ��)" << std::endl;
        std::cout << "3. OpenCV cornerHarris ����" << std::endl;
        std::cout << "4. Custom cornerHarris ����" << std::endl;
        std::cout << "5. Enhanced cornerHarris ���� (ȸ���� ���� ����ȭ)" << std::endl;
        std::cout << "6. Hough Lines �� (OpenCV vs Custom)" << std::endl;
        std::cout << "7. Harris Corners �� (OpenCV vs Custom vs Enhanced)" << std::endl;
        std::cout << "0. ����" << std::endl;
        std::cout << "����: ";

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
            std::cout << "\n=== HoughLines �� ===" << std::endl;
            run_HoughLines_Original();
            run_HoughLines_Custom();
            break;
        case 7:
            std::cout << "\n=== Harris Corners �� ===" << std::endl;
            run_HarrisCornerDetector_Original();
            run_HarrisCornerDetector_Custom();
            run_HarrisCornerDetector_Enhanced();
            break;
        case 0:
            std::cout << "���α׷��� �����մϴ�..." << std::endl;
            return 0;
        default:
            std::cout << "�߸��� �����Դϴ�. �ٽ� �õ����ּ���." << std::endl;
        }
    }

    return 0;
}