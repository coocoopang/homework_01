#include "face_matcher.h"
#include <iostream>
#include <string>
#include <fstream>

void printUsage() {
    std::cout << "얼굴 매칭 시스템 v2.0" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << std::endl;
    std::cout << "지원하는 입력 소스:" << std::endl;
    std::cout << " 1. 웹캠 실시간 영상" << std::endl;
    std::cout << " 2. MP4 비디오 파일" << std::endl;
    std::cout << std::endl;
    std::cout << "사용법:" << std::endl;
    std::cout << " 1. 자신의 얼굴 사진을 ./images/ 폴더에 준비" << std::endl;
    std::cout << " 2. 프로그램 실행 후 사진 경로 입력" << std::endl;
    std::cout << " 3. 입력 소스 선택 (웹캠 또는 비디오 파일)" << std::endl;
    std::cout << " 4. 얼굴 매칭 결과 확인" << std::endl;
    std::cout << std::endl;
    std::cout << "실행 중 조작법:" << std::endl;
    std::cout << " - ESC 또는 'q': 프로그램 종료" << std::endl;
    std::cout << " - SPACE: 일시정지/재생 (비디오 파일만)" << std::endl;
    std::cout << " - 't': 매칭 임계값 조정 (기본: 70%)" << std::endl;
    std::cout << " - 's': 현재 화면 스크린샷 저장" << std::endl;
    std::cout << std::endl;
    std::cout << "팁:" << std::endl;
    std::cout << " - 조명이 적당한 곳에서 테스트하세요" << std::endl;
    std::cout << " - 정면을 보는 얼굴 사진을 사용하세요" << std::endl;
    std::cout << " - MP4 파일은 ./videos/ 폴더에 준비하세요" << std::endl;
    std::cout << std::endl;
}

bool downloadHaarCascade() {
    std::cout << "얼굴 검출기 파일 다운로드 중..." << std::endl;
    
    // OpenCV Haar cascade 파일 다운로드
    std::string url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml";
    std::string command = "wget -O haarcascade_frontalface_alt.xml '" + url + "'";
    
    int result = system(command.c_str());
    if (result == 0) {
        std::cout << "얼굴 검출기 다운로드 완료!" << std::endl;
        return true;
    } else {
        std::cout << "다운로드 실패. 시스템에 설치된 파일을 사용합니다." << std::endl;
        return false;
    }
}

int main() 
{
    printUsage();
    
    // Haar cascade 파일 확인 및 다운로드
    if (!std::ifstream("haarcascade_frontalface_alt.xml").good()) {
        downloadHaarCascade();
    }
    
    // FaceMatcher 초기화
    FaceMatcher matcher;
    
    // 기준 얼굴 이미지 로드
    std::string imagePath;
    imagePath = "./images/my_face.png";
    
    if (!matcher.loadReferenceFace(imagePath)) {
        std::cerr << "기준 얼굴 이미지 로드 실패!" << std::endl;
        std::cerr << "해결방법:" << std::endl;
        std::cerr << " 1. 이미지 경로가 올바른지 확인" << std::endl;
        std::cerr << " 2. 이미지에 얼굴이 포함되어 있는지 확인" << std::endl;
        std::cerr << " 3. 이미지 형식이 지원되는지 확인 (jpg, png 등)" << std::endl;
        return -1;
    }
    
    // 입력 소스 선택
    std::cout << std::endl;
    std::cout << "입력 소스를 선택하세요:" << std::endl;
    std::cout << " 1. 웹캠 (실시간)" << std::endl;
    std::cout << " 2. MP4 비디오 파일" << std::endl;
    std::cout << "선택 (1 또는 2): ";
    
    std::string choice;
    std::getline(std::cin, choice);
    
    bool success = false;
    
    if (choice == "1") {
        // 웹캠 모드
        if (matcher.startWebcam(0)) {
            success = true;
        } else {
            std::cerr << "웹캠 시작 실패!" << std::endl;
            std::cerr << "해결방법:" << std::endl;
            std::cerr << " 1. 웹캠이 연결되어 있는지 확인" << std::endl;
            std::cerr << " 2. 다른 프로그램에서 웹캠을 사용 중인지 확인" << std::endl;
            std::cerr << " 3. 웹캠 권한이 있는지 확인" << std::endl;
        }
    } else if (choice == "2") {
        // 비디오 파일 모드
        std::cout << "비디오 파일 경로를 입력하세요 (예: ./videos/test.mp4): ";
        std::string videoPath;
        std::getline(std::cin, videoPath);
        
        if (videoPath.empty()) {
            std::cerr << "비디오 파일 경로가 입력되지 않았습니다!" << std::endl;
            return -1;
        }
        
        if (matcher.loadVideoFile(videoPath)) {
            success = true;
        } else {
            std::cerr << "비디오 파일 로드 실패!" << std::endl;
            std::cerr << "해결방법:" << std::endl;
            std::cerr << " 1. 파일 경로가 올바른지 확인" << std::endl;
            std::cerr << " 2. 파일이 존재하는지 확인" << std::endl;
            std::cerr << " 3. 지원되는 비디오 형식인지 확인 (mp4, avi, mov 등)" << std::endl;
        }
    } else {
        std::cerr << "잘못된 선택입니다!" << std::endl;
        return -1;
    }
    
    if (!success) {
        return -1;
    }
    
    // 매칭 임계값 설정 (선택사항)
    std::cout << "매칭 임계값을 설정하시겠습니까? (기본값: 70%) [y/N]: ";
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
    std::cout << "얼굴 매칭 시작!" << std::endl;
    if (choice == "1") {
        std::cout << "웹캠 화면이 나타나면 얼굴을 카메라에 비춰보세요." << std::endl;
    } else {
        std::cout << "비디오가 재생되면서 얼굴 매칭이 진행됩니다." << std::endl;
        std::cout << " - SPACE키로 일시정지/재생 가능" << std::endl;
    }
    std::cout << std::endl;
    
    // 실시간 얼굴 매칭 실행
    try {
        matcher.runFaceMatching();
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV 오류: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "실행 오류: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "프로그램을 종료합니다." << std::endl;
    return 0;
}