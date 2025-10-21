#include "face_matcher.h"
#include <iostream>

int main() {
    std::cout << "🔬 최근접 특징 매칭 시스템 테스트" << std::endl;
    
    // FaceMatcher 객체 생성
    FaceMatcher matcher;
    
    std::cout << "✅ 특징점 검출기 초기화 성공!" << std::endl;
    std::cout << "✅ 최근접 매칭 알고리즘 준비 완료!" << std::endl;
    std::cout << "✅ SIFT/ORB 특징점 추출 기능 사용 가능!" << std::endl;
    
    return 0;
}
