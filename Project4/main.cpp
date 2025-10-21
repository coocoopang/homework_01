#include "face_matcher.h"
#include <iostream>
#include <string>
#include <fstream>

void printUsage() {
    std::cout << "🎯 최근접 특징 매칭 기반 얼굴 인식 시스템 v3.0" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << std::endl;
    std::cout << "🔬 특징점 매칭 알고리즘:" << std::endl;
    std::cout << "   - SIFT (Scale-Invariant Feature Transform)" << std::endl;
    std::cout << "   - ORB (Oriented FAST and Rotated BRIEF)" << std::endl;
    std::cout << "   - 최근접 이웃 매칭 + 기하학적 검증" << std::endl;
    std::cout << std::endl;
    std::cout << "📋 지원하는 입력 소스:" << std::endl;
    std::cout << "   1. 웹캠 실시간 영상" << std::endl;
    std::cout << "   2. MP4 비디오 파일" << std::endl;
    std::cout << std::endl;
    std::cout << "📋 사용법:" << std::endl;
    std::cout << "   1. 자신의 얼굴 사진을 ./images/ 폴더에 준비" << std::endl;
    std::cout << "   2. 프로그램 실행 후 사진 경로 입력" << std::endl;
    std::cout << "   3. 입력 소스 선택 (웹캠 또는 비디오 파일)" << std::endl;
    std::cout << "   4. 얼굴 매칭 결과 확인" << std::endl;
    std::cout << std::endl;
    std::cout << "🎮 실행 중 조작법:" << std::endl;
    std::cout << "   - ESC 또는 'q': 프로그램 종료" << std::endl;
    std::cout << "   - SPACE: 일시정지/재생 (비디오 파일만)" << std::endl;
    std::cout << "   - 't': 매칭 임계값 조정 (기본: 70%)" << std::endl;
    std::cout << "   - 'd': 특징점 검출기 변경 (SIFT ↔ ORB)" << std::endl;
    std::cout << "   - 's': 현재 화면 스크린샷 저장" << std::endl;
    std::cout << std::endl;
    std::cout << "💡 특징점 매칭 팁:" << std::endl;
    std::cout << "   - 텍스처가 풍부한 얼굴 사진을 사용하세요" << std::endl;
    std::cout << "   - 조명이 적당하고 균일한 곳에서 테스트하세요" << std::endl;
    std::cout << "   - SIFT는 정확하지만 느리고, ORB는 빠르지만 덜 정확합니다" << std::endl;
    std::cout << "   - MP4 파일은 ./videos/ 폴더에 준비하세요" << std::endl;
    std::cout << std::endl;
}

void createSampleImages() {
    // images 폴더 생성
    system("mkdir -p ./images");
    system("mkdir -p ./videos");
    
    std::cout << "📁 ./images 폴더가 생성되었습니다." << std::endl;
    std::cout << "📁 ./videos 폴더가 생성되었습니다." << std::endl;
    std::cout << "💡 images 폴더에 자신의 얼굴 사진을 넣어주세요!" << std::endl;
    std::cout << "💡 videos 폴더에 테스트할 MP4 파일을 넣어주세요!" << std::endl;
    std::cout << std::endl;
    
    // 샘플 이미지 생성 (테스트용)
    cv::Mat sampleImage = cv::Mat::zeros(400, 300, CV_8UC3);
    
    // 간단한 얼굴 모양 그리기
    cv::Point center(150, 200);
    
    // 얼굴 윤곽
    cv::ellipse(sampleImage, center, cv::Size(80, 100), 0, 0, 360, cv::Scalar(220, 200, 180), -1);
    
    // 눈
    cv::circle(sampleImage, cv::Point(130, 170), 8, cv::Scalar(0, 0, 0), -1);
    cv::circle(sampleImage, cv::Point(170, 170), 8, cv::Scalar(0, 0, 0), -1);
    
    // 코
    cv::ellipse(sampleImage, cv::Point(150, 190), cv::Size(3, 8), 0, 0, 360, cv::Scalar(200, 180, 160), -1);
    
    // 입
    cv::ellipse(sampleImage, cv::Point(150, 220), cv::Size(20, 8), 0, 0, 180, cv::Scalar(180, 100, 100), -1);
    
    // 텍스트 추가
    cv::putText(sampleImage, "Sample Face", cv::Point(50, 350), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::putText(sampleImage, "Replace with your photo", cv::Point(30, 380), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
    
    cv::imwrite("./images/sample_face.jpg", sampleImage);
    
    std::cout << "📸 샘플 이미지 생성: ./images/sample_face.jpg" << std::endl;
    std::cout << "   (실제 얼굴 사진으로 교체해주세요)" << std::endl;
}

bool downloadHaarCascade() {
    std::cout << "📥 얼굴 검출기 파일 다운로드 중..." << std::endl;
    
    // OpenCV Haar cascade 파일 다운로드
    std::string url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml";
    std::string command = "wget -O haarcascade_frontalface_alt.xml '" + url + "'";
    
    int result = system(command.c_str());
    if (result == 0) {
        std::cout << "✅ 얼굴 검출기 다운로드 완료!" << std::endl;
        return true;
    } else {
        std::cout << "❌ 다운로드 실패. 시스템에 설치된 파일을 사용합니다." << std::endl;
        return false;
    }
}

int main() {
    printUsage();
    
    // 필요한 폴더 및 파일 생성
    createSampleImages();
    
    // Haar cascade 파일 확인 및 다운로드
    if (!std::ifstream("haarcascade_frontalface_alt.xml").good()) {
        downloadHaarCascade();
    }
    
    // FaceMatcher 초기화
    FaceMatcher matcher;
    
    // 기준 얼굴 이미지 로드
    std::string imagePath;
    std::cout << "👤 기준 얼굴 이미지 경로를 입력하세요 (예: ./images/my_face.jpg): ";
    std::getline(std::cin, imagePath);
    
    if (imagePath.empty()) {
        imagePath = "./images/sample_face.jpg";
        std::cout << "기본값 사용: " << imagePath << std::endl;
    }
    
    if (!matcher.loadReferenceFace(imagePath)) {
        std::cerr << "❌ 기준 얼굴 이미지 로드 실패!" << std::endl;
        std::cerr << "💡 해결방법:" << std::endl;
        std::cerr << "   1. 이미지 경로가 올바른지 확인" << std::endl;
        std::cerr << "   2. 이미지에 얼굴이 포함되어 있는지 확인" << std::endl;
        std::cerr << "   3. 이미지 형식이 지원되는지 확인 (jpg, png 등)" << std::endl;
        return -1;
    }
    
    // 입력 소스 선택
    std::cout << std::endl;
    std::cout << "🎥 입력 소스를 선택하세요:" << std::endl;
    std::cout << "   1. 웹캠 (실시간)" << std::endl;
    std::cout << "   2. MP4 비디오 파일" << std::endl;
    std::cout << "선택 (1 또는 2): ";
    
    std::string choice;
    std::getline(std::cin, choice);
    
    bool success = false;
    
    if (choice == "1") {
        // 웹캠 모드
        if (matcher.startWebcam(0)) {
            success = true;
        } else {
            std::cerr << "❌ 웹캠 시작 실패!" << std::endl;
            std::cerr << "💡 해결방법:" << std::endl;
            std::cerr << "   1. 웹캠이 연결되어 있는지 확인" << std::endl;
            std::cerr << "   2. 다른 프로그램에서 웹캠을 사용 중인지 확인" << std::endl;
            std::cerr << "   3. 웹캠 권한이 있는지 확인" << std::endl;
        }
    } else if (choice == "2") {
        // 비디오 파일 모드
        std::cout << "📁 비디오 파일 경로를 입력하세요 (예: ./videos/test.mp4): ";
        std::string videoPath;
        std::getline(std::cin, videoPath);
        
        if (videoPath.empty()) {
            std::cerr << "❌ 비디오 파일 경로가 입력되지 않았습니다!" << std::endl;
            return -1;
        }
        
        if (matcher.loadVideoFile(videoPath)) {
            success = true;
        } else {
            std::cerr << "❌ 비디오 파일 로드 실패!" << std::endl;
            std::cerr << "💡 해결방법:" << std::endl;
            std::cerr << "   1. 파일 경로가 올바른지 확인" << std::endl;
            std::cerr << "   2. 파일이 존재하는지 확인" << std::endl;
            std::cerr << "   3. 지원되는 비디오 형식인지 확인 (mp4, avi, mov 등)" << std::endl;
        }
    } else {
        std::cerr << "❌ 잘못된 선택입니다!" << std::endl;
        return -1;
    }
    
    if (!success) {
        return -1;
    }
    
    // 매칭 임계값 설정 (선택사항)
    std::cout << "🎚️ 매칭 임계값을 설정하시겠습니까? (기본값: 70%) [y/N]: ";
    std::string response;
    std::getline(std::cin, response);
    
    if (response == "y" || response == "Y") {
        int threshold;
        std::cout << "임계값 입력 (0-100): ";
        std::cin >> threshold;
        matcher.setMatchThreshold(threshold / 100.0);
        std::cin.ignore(); // 버퍼 클리어
    }
    
    std::cout << std::endl;
    std::cout << "🚀 얼굴 매칭 시작!" << std::endl;
    if (choice == "1") {
        std::cout << "📹 웹캠 화면이 나타나면 얼굴을 카메라에 비춰보세요." << std::endl;
    } else {
        std::cout << "📹 비디오가 재생되면서 얼굴 매칭이 진행됩니다." << std::endl;
        std::cout << "   - SPACE키로 일시정지/재생 가능" << std::endl;
    }
    std::cout << std::endl;
    
    // 실시간 얼굴 매칭 실행
    try {
        matcher.runFaceMatching();
    } catch (const cv::Exception& e) {
        std::cerr << "❌ OpenCV 오류: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "❌ 실행 오류: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "👋 프로그램을 종료합니다." << std::endl;
    return 0;
}