# 사용법 가이드

## 프로그램 실행

### 1. 빌드된 프로그램 실행
```bash
cd build
./custom_cv
```

### 2. 메뉴 선택
프로그램 실행 시 다음과 같은 메뉴가 표시됩니다:

```
Choose an option:
1. Run Original OpenCV HoughLines
2. Run Custom HoughLines Implementation  
3. Run Original OpenCV cornerHarris
4. Run Custom cornerHarris Implementation
5. Compare Hough Lines (Original vs Custom)
6. Compare Harris Corners (Original vs Custom)
0. Exit
Enter choice: 
```

## 각 옵션 설명

### 옵션 1: Original OpenCV HoughLines
- OpenCV 라이브러리의 `cv::HoughLines` 함수를 사용
- 원본 구현의 성능과 정확도 확인
- 빨간색 선으로 검출된 직선 표시

### 옵션 2: Custom HoughLines Implementation
- 직접 구현한 `custom_cv::HoughLines` 함수를 사용
- Hough 변환 알고리즘의 세부 구현 확인
- 초록색 선으로 검출된 직선 표시

### 옵션 3: Original OpenCV cornerHarris  
- OpenCV 라이브러리의 `cv::cornerHarris` 함수를 사용
- 원본 구현의 코너 검출 결과 확인
- 빨간색 원으로 검출된 코너 표시

### 옵션 4: Custom cornerHarris Implementation
- 직접 구현한 `custom_cv::cornerHarris` 함수를 사용  
- Harris 코너 검출 알고리즘의 세부 구현 확인
- 초록색 원으로 검출된 코너 표시

### 옵션 5, 6: 비교 모드
- 원본과 커스텀 구현을 연속으로 실행하여 결과 비교
- 알고리즘 정확도와 성능 차이 확인

## 이미지 준비

### 권장 테스트 이미지

#### Hough Lines 테스트용:
- 건물, 도로, 기하학적 구조물 이미지
- 직선이 많이 포함된 이미지
- 파일명: `lg_building.jpg`, `building.jpg`, `test_building.jpg`

#### Harris Corners 테스트용:
- 코너와 모서리가 많은 이미지
- 기하학적 도형이 포함된 이미지  
- 파일명: `shapes1.jpg`, `shapes.jpg`, `corners.jpg`, `test_corners.jpg`

### 이미지 위치
```
Project1/
└── images/
    ├── lg_building.jpg     # Hough Lines 테스트
    ├── building.jpg
    ├── shapes1.jpg         # Harris Corners 테스트  
    ├── shapes.jpg
    └── corners.jpg
```

### 이미지가 없는 경우
- 프로그램이 자동으로 합성 테스트 이미지를 생성
- Hough Lines: 직선과 사각형이 포함된 이미지
- Harris Corners: 사각형과 삼각형이 포함된 이미지

## 결과 해석

### Hough Lines 결과
- **빨간색 선**: OpenCV 원본 구현 결과
- **초록색 선**: 커스텀 구현 결과
- 직선이 많이 검출될수록 임계값을 높여서 조정 가능

### Harris Corners 결과  
- **빨간색 원**: OpenCV 원본 구현 결과
- **초록색 원**: 커스텀 구현 결과
- 코너 응답값이 클수록 더 강한 코너

## 매개변수 조정

### Hough Lines 매개변수
```cpp
custom_cv::HoughLines(src_edge, lines, 1, CV_PI / 180, 80);
//                                     rho   theta      threshold
```
- `rho`: 거리 해상도 (픽셀 단위)
- `theta`: 각도 해상도 (라디안 단위)  
- `threshold`: 검출 임계값 (높을수록 강한 직선만 검출)

### Harris Corners 매개변수
```cpp
custom_cv::cornerHarris(src, R, blockSize, kSize, k);
//                              5          3     0.04
```
- `blockSize`: 윈도우 크기 (보통 3, 5, 7)
- `kSize`: Sobel 연산자 크기 (3 또는 5)
- `k`: Harris 검출기 매개변수 (0.04~0.06)

## 문제 해결

### 빌드 오류
```bash
# OpenCV 재설치
sudo apt-get install libopencv-dev libopencv-contrib-dev

# CMake 재설치  
sudo apt-get install cmake

# 빌드 디렉토리 초기화
rm -rf build
./build.sh
```

### 실행 오류
```bash
# 권한 확인
chmod +x build/custom_cv

# 라이브러리 경로 확인
ldd build/custom_cv

# X11 디스플레이 설정 (필요시)
export DISPLAY=:0
```

### 결과가 예상과 다른 경우
1. **매개변수 조정**: 임계값을 낮추거나 높여서 테스트
2. **이미지 품질**: 더 선명한 에지가 있는 이미지 사용
3. **전처리**: Canny 에지 검출의 임계값 조정

## 학습 포인트

### Hough Transform 이해
- 직교 좌표계에서 극좌표계로의 변환
- 누적기 배열의 동작 원리
- 로컬 최댓값 검출의 중요성

### Harris Corner Detector 이해  
- 구조 텐서(Structure Matrix) 계산
- 고유값과 고유벡터의 의미
- 가우시안 가중치의 역할

이 구현을 통해 컴퓨터 비전의 핵심 알고리즘을 깊이 이해할 수 있습니다!