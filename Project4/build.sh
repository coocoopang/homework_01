#!/bin/bash

echo "🔨 얼굴 매칭 시스템 빌드 스크립트"
echo "================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 빌드 디렉터리 생성
echo -e "${BLUE}📁 빌드 디렉터리 준비 중...${NC}"
mkdir -p build
cd build

# CMake 설정
echo -e "${BLUE}⚙️ CMake 설정 중...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ CMake 설정 실패!${NC}"
    echo -e "${YELLOW}💡 해결방법:${NC}"
    echo "   - OpenCV가 설치되어 있는지 확인하세요"
    echo "   - sudo apt-get install libopencv-dev"
    exit 1
fi

# 컴파일
echo -e "${BLUE}🔧 컴파일 중...${NC}"
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 빌드 성공!${NC}"
    echo -e "${GREEN}🚀 실행 파일: ./build/face_matcher${NC}"
    
    # 실행 파일 복사
    cp face_matcher ../
    echo -e "${GREEN}📋 실행 파일이 프로젝트 루트로 복사되었습니다.${NC}"
    
    echo ""
    echo -e "${YELLOW}🎯 사용법:${NC}"
    echo "   1. cd .. (프로젝트 루트로 이동)"
    echo "   2. ./face_matcher (프로그램 실행)"
    echo ""
    echo -e "${YELLOW}📝 준비사항:${NC}"
    echo "   - ./images/ 폴더에 자신의 얼굴 사진 준비"
    echo "   - 웹캠 연결 및 권한 확인"
else
    echo -e "${RED}❌ 빌드 실패!${NC}"
    echo -e "${YELLOW}💡 해결방법:${NC}"
    echo "   - 컴파일 오류 메시지를 확인하세요"
    echo "   - OpenCV 개발 패키지가 설치되어 있는지 확인하세요"
    exit 1
fi