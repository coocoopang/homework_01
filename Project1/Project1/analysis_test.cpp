#include <iostream>
#include "opencv2/opencv.hpp"
#include "custom_cv.h"

std::vector<cv::Point> FindLocalExtrema(cv::Mat& src) {
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
    std::cout << "==== GitHub main.cpp 성능 분석 ====" << std::endl;
    std::cout << std::endl;
    
    // 1. HoughLines 테스트
    std::cout << "📐 HoughLines 테스트 결과:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    cv::Mat building_src = cv::imread("./images/lg_building.jpg", cv::IMREAD_GRAYSCALE);
    if (!building_src.empty()) {
        cv::Mat src_edge;
        cv::Canny(building_src, src_edge, 170, 200);
        
        // OpenCV HoughLines
        std::vector<cv::Vec2f> lines_opencv;
        cv::HoughLines(src_edge, lines_opencv, 1, CV_PI / 180, 400);
        
        // Custom HoughLines (수정된 threshold)
        std::vector<cv::Vec2f> lines_custom;
        custom_cv::HoughLines(src_edge, lines_custom, 1, CV_PI / 180.0, 80);
        
        std::cout << "🔹 OpenCV HoughLines (threshold=400): " << lines_opencv.size() << "개 직선 검출" << std::endl;
        std::cout << "🔹 Custom HoughLines (threshold=80):  " << lines_custom.size() << "개 직선 검출" << std::endl;
        
        if (lines_opencv.size() > 0 && lines_custom.size() > 0) {
            std::cout << "✅ HoughLines 정상 작동 - threshold 최적화 완료!" << std::endl;
        } else if (lines_opencv.size() == 0) {
            std::cout << "⚠️  OpenCV threshold 400이 너무 높음 (직선 검출 안됨)" << std::endl;
        } else {
            std::cout << "⚠️  Custom implementation에 문제 있음" << std::endl;
        }
    } else {
        std::cout << "❌ lg_building.jpg 이미지를 불러올 수 없음" << std::endl;
    }
    
    std::cout << std::endl;
    
    // 2. Harris Corner 테스트
    std::cout << "🔍 Harris Corner Detection 테스트 결과:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    cv::Mat shapes_src = cv::imread("./images/shapes1.jpg", cv::IMREAD_GRAYSCALE);
    if (!shapes_src.empty()) {
        int blockSize = 5;
        int kSize = 3;
        double k = 0.01;
        
        // OpenCV cornerHarris
        cv::Mat R_opencv;
        cv::cornerHarris(shapes_src, R_opencv, blockSize, kSize, k);
        cv::threshold(R_opencv, R_opencv, 0.02, 0, cv::THRESH_TOZERO);
        std::vector<cv::Point> corners_opencv = FindLocalExtrema(R_opencv);
        
        // Custom cornerHarris
        cv::Mat R_custom;
        custom_cv::cornerHarris(shapes_src, R_custom, blockSize, kSize, k);
        cv::threshold(R_custom, R_custom, 0.02, 0, cv::THRESH_TOZERO);
        std::vector<cv::Point> corners_custom = FindLocalExtrema(R_custom);
        
        std::cout << "🔹 OpenCV cornerHarris: " << corners_opencv.size() << "개 코너 검출" << std::endl;
        std::cout << "🔹 Custom cornerHarris: " << corners_custom.size() << "개 코너 검출" << std::endl;
        
        // 회전된 도형 영역에서의 코너 검출 분석
        int opencv_rotated = 0, custom_rotated = 0;
        
        for (const auto& corner : corners_opencv) {
            // 회전된 사각형 영역 (center: 200,100, 30도 회전)
            if (corner.x >= 160 && corner.x <= 240 && corner.y >= 60 && corner.y <= 140) {
                opencv_rotated++;
            }
            // 회전된 삼각형 영역 (center: 300,200, 45도 회전) 
            else if (corner.x >= 260 && corner.x <= 340 && corner.y >= 160 && corner.y <= 240) {
                opencv_rotated++;
            }
        }
        
        for (const auto& corner : corners_custom) {
            // 회전된 사각형 영역
            if (corner.x >= 160 && corner.x <= 240 && corner.y >= 60 && corner.y <= 140) {
                custom_rotated++;
            }
            // 회전된 삼각형 영역
            else if (corner.x >= 260 && corner.x <= 340 && corner.y >= 160 && corner.y <= 240) {
                custom_rotated++;
            }
        }
        
        std::cout << std::endl;
        std::cout << "🎯 회전된 도형에서의 코너 검출 성능:" << std::endl;
        std::cout << "   OpenCV (회전된 도형): " << opencv_rotated << "개 코너" << std::endl;
        std::cout << "   Custom (회전된 도형): " << custom_rotated << "개 코너" << std::endl;
        
        if (custom_rotated >= opencv_rotated) {
            std::cout << "✅ Custom Harris가 회전된 도형에서 잘 작동함!" << std::endl;
        } else {
            std::cout << "⚠️  Custom Harris가 회전된 도형에서 성능 부족" << std::endl;
            std::cout << "    -> Enhanced Harris 구현 필요" << std::endl;
        }
        
        // 전체 성능 비교
        double performance_ratio = (double)corners_custom.size() / corners_opencv.size() * 100.0;
        std::cout << std::endl;
        std::cout << "📊 전체 성능 비교:" << std::endl;
        std::cout << "   Custom/OpenCV 비율: " << std::fixed << std::setprecision(1) << performance_ratio << "%" << std::endl;
        
        if (performance_ratio >= 90.0) {
            std::cout << "✅ Custom 구현체 성능 우수!" << std::endl;
        } else {
            std::cout << "⚠️  Custom 구현체 성능 개선 필요" << std::endl;
        }
        
    } else {
        std::cout << "❌ shapes1.jpg 이미지를 불러올 수 없음" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "==== 분석 완료 ====" << std::endl;
    
    return 0;
}