#include "opencv2/opencv.hpp"
#include <iostream>

int main() {
    // 1. lg_building.jpg 대체 이미지 생성 (Hough Lines 테스트용)
    cv::Mat building_img = cv::Mat::zeros(400, 600, CV_8UC1);
    
    // 건물 형태의 라인들 그리기
    // 수직선들 (건물 기둥)
    cv::line(building_img, cv::Point(100, 50), cv::Point(100, 350), cv::Scalar(255), 3);
    cv::line(building_img, cv::Point(200, 50), cv::Point(200, 350), cv::Scalar(255), 3);
    cv::line(building_img, cv::Point(300, 50), cv::Point(300, 350), cv::Scalar(255), 3);
    cv::line(building_img, cv::Point(400, 50), cv::Point(400, 350), cv::Scalar(255), 3);
    cv::line(building_img, cv::Point(500, 50), cv::Point(500, 350), cv::Scalar(255), 3);
    
    // 수평선들 (건물 층)
    cv::line(building_img, cv::Point(50, 100), cv::Point(550, 100), cv::Scalar(255), 3);
    cv::line(building_img, cv::Point(50, 200), cv::Point(550, 200), cv::Scalar(255), 3);
    cv::line(building_img, cv::Point(50, 300), cv::Point(550, 300), cv::Scalar(255), 3);
    cv::line(building_img, cv::Point(80, 50), cv::Point(520, 50), cv::Scalar(255), 3);
    
    // 약간의 노이즈 추가
    cv::randu(building_img, 0, 50);
    cv::threshold(building_img, building_img, 40, 255, cv::THRESH_BINARY);
    
    // 2. shapes1.jpg 대체 이미지 생성 (Harris Corner 테스트용)
    cv::Mat shapes_img = cv::Mat::zeros(400, 400, CV_8UC1);
    
    // 정사각형 (axis-aligned)
    cv::rectangle(shapes_img, cv::Point(50, 50), cv::Point(120, 120), cv::Scalar(255), 2);
    
    // 회전된 사각형 (문제가 되는 케이스)
    cv::Point2f center(200, 100);
    cv::Size2f size(80, 50);
    float angle = 30.0; // 30도 회전
    
    cv::RotatedRect rotRect(center, size, angle);
    cv::Point2f vertices[4];
    rotRect.points(vertices);
    
    for (int i = 0; i < 4; i++) {
        cv::line(shapes_img, vertices[i], vertices[(i+1)%4], cv::Scalar(255), 2);
    }
    
    // 회전된 삼각형 (또 다른 문제 케이스)
    std::vector<cv::Point> triangle;
    float tri_angle = 45 * CV_PI / 180.0; // 45도 회전
    cv::Point tri_center(300, 200);
    int radius = 40;
    
    for (int i = 0; i < 3; i++) {
        float a = tri_angle + i * 2 * CV_PI / 3;
        triangle.push_back(cv::Point(
            tri_center.x + radius * cos(a),
            tri_center.y + radius * sin(a)
        ));
    }
    
    for (int i = 0; i < 3; i++) {
        cv::line(shapes_img, triangle[i], triangle[(i+1)%3], cv::Scalar(255), 2);
    }
    
    // L자 모양 (직각 코너 테스트)
    cv::line(shapes_img, cv::Point(80, 250), cv::Point(80, 320), cv::Scalar(255), 3);
    cv::line(shapes_img, cv::Point(80, 320), cv::Point(150, 320), cv::Scalar(255), 3);
    
    // 원형 (코너가 없어야 함)
    cv::circle(shapes_img, cv::Point(300, 320), 30, cv::Scalar(255), 2);
    
    // 이미지 저장
    cv::imwrite("./images/lg_building.jpg", building_img);
    cv::imwrite("./images/shapes1.jpg", shapes_img);
    
    std::cout << "테스트 이미지들이 생성되었습니다:" << std::endl;
    std::cout << "- ./images/lg_building.jpg (HoughLines 테스트용)" << std::endl;
    std::cout << "- ./images/shapes1.jpg (Harris Corner 테스트용)" << std::endl;
    
    // 생성된 이미지 보기 (선택사항)
    try {
        cv::imshow("Generated Building Image", building_img);
        cv::imshow("Generated Shapes Image", shapes_img);
        cv::waitKey(3000); // 3초 대기
        cv::destroyAllWindows();
    } catch(const cv::Exception& e) {
        std::cout << "디스플레이를 사용할 수 없어 이미지 표시를 건너뜁니다." << std::endl;
    }
    
    return 0;
}