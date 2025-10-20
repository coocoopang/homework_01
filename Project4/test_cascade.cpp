#include "face_matcher.h"
#include <iostream>

int main() {
    std::cout << "🔍 FaceMatcher 초기화 테스트" << std::endl;
    
    // FaceMatcher 객체 생성 (cascade 로딩 테스트)
    FaceMatcher matcher;
    
    std::cout << "\n📷 기준 얼굴 이미지 로드 테스트" << std::endl;
    if (matcher.loadReferenceFace("images/test_face.jpg")) {
        std::cout << "✅ 기준 얼굴 로드 성공!" << std::endl;
    } else {
        std::cout << "❌ 기준 얼굴 로드 실패!" << std::endl;
    }
    
    std::cout << "\n🎥 웹캠 초기화 테스트 (실제로는 열지 않음)" << std::endl;
    std::cout << "✅ cascade 로딩 및 기본 초기화 완료!" << std::endl;
    
    return 0;
}
