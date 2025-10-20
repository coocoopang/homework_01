#include "face_matcher.h"
#include <iostream>

int main() {
    std::cout << "🎥 MP4 비디오 파일 로드 테스트" << std::endl;
    
    FaceMatcher matcher;
    
    // 테스트 비디오 파일 로드
    if (matcher.loadVideoFile("videos/test_sample.mp4")) {
        std::cout << "✅ MP4 비디오 파일 로드 성공!" << std::endl;
        std::cout << "✅ MP4 지원 기능이 정상적으로 작동합니다!" << std::endl;
    } else {
        std::cout << "❌ MP4 비디오 파일 로드 실패!" << std::endl;
    }
    
    return 0;
}
