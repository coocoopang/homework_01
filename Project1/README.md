# 컴퓨터비전및응용 과제 - Custom Implementation

본 프로젝트는 컴퓨터비전 강의의 과제를 위해 OpenCV의 핵심 함수들을 직접 구현한 프로젝트입니다.

## 구현된 기능

### 1. Hough 변환을 이용한 직선 검출 (Custom HoughLines)
- **원본 함수**: `cv::HoughLines(src_edge, lines, 1, CV_PI / 180, 400)`
- **커스텀 구현**: `custom_cv::HoughLines(src_edge, lines, 1, CV_PI / 180, 400)`

#### 알고리즘 특징:
- Hough 공간에서 누적기(Accumulator) 배열 생성
- (ρ, θ) 매개변수 공간에서 직선 검출
- 비최대 억제(Non-maximum Suppression)를 통한 로컬 최댓값 검출
- 임계값 기반 직선 필터링

### 2. Harris Corner Detector (Custom cornerHarris)
- **원본 함수**: `cv::cornerHarris(src, R, blockSize, kSize, k)`
- **커스텀 구현**: `custom_cv::cornerHarris(src, R, blockSize, kSize, k)`

#### 알고리즘 특징:
- Sobel 연산자를 이용한 이미지 gradient 계산
- Structure Matrix 계산 (Ixx, Iyy, Ixy)
- 가우시안 가중치 적용
- Harris 응답 함수: R = det(M) - k(trace(M))²

## 프로젝트 구조

```
Project1/
├── main.cpp              # 원본 OpenCV 구현
├── main_updated.cpp      # 비교 테스트가 포함된 개선된 메인
├── custom_cv.h          # 커스텀 구현 헤더 파일
├── custom_cv.cpp        # 커스텀 구현 소스 파일
├── CMakeLists.txt       # CMake 빌드 설정
├── README.md           # 프로젝트 설명서
└── images/             # 테스트 이미지 디렉토리
```

## 빌드 및 실행

### 필요 사항
- OpenCV 4.x
- CMake 3.10+
- C++14 지원 컴파일러

### 빌드 방법

#### CMake를 사용한 빌드 (권장)
```bash
# 빌드 디렉토리 생성
mkdir build
cd build

# CMake 설정
cmake ..

# 빌드
make

# 실행
./custom_cv    # 커스텀 구현 버전
./original_cv  # 원본 OpenCV 버전
```

#### 직접 컴파일
```bash
# 커스텀 구현 빌드
g++ -std=c++14 main_updated.cpp custom_cv.cpp -o custom_cv `pkg-config --cflags --libs opencv4`

# 원본 구현 빌드
g++ -std=c++14 main.cpp -o original_cv `pkg-config --cflags --libs opencv4`
```

## 사용법

실행하면 다음과 같은 메뉴가 표시됩니다:

```
Choose an option:
1. Run Original OpenCV HoughLines
2. Run Custom HoughLines Implementation  
3. Run Original OpenCV cornerHarris
4. Run Custom cornerHarris Implementation
5. Compare Hough Lines (Original vs Custom)
6. Compare Harris Corners (Original vs Custom)
0. Exit
```

- 옵션 1, 3: 원본 OpenCV 함수를 사용한 결과
- 옵션 2, 4: 직접 구현한 커스텀 함수를 사용한 결과
- 옵션 5, 6: 원본과 커스텀 구현 결과를 연속으로 비교

## 테스트 이미지

프로그램은 다음 순서로 이미지를 찾습니다:

### Hough Lines 테스트:
1. `images/lg_building.jpg`
2. `images/building.jpg`
3. `images/test_building.jpg`
4. 없으면 합성 테스트 이미지 생성

### Harris Corners 테스트:
1. `images/shapes1.jpg`
2. `images/shapes.jpg`
3. `images/corners.jpg`
4. `images/test_corners.jpg`
5. 없으면 합성 테스트 이미지 생성

## 구현 세부사항

### Hough Transform 구현
```cpp
void custom_cv::HoughLines(const cv::Mat& image, std::vector<cv::Vec2f>& lines, 
                          double rho, double theta, int threshold);
```

- **입력**: 에지 이미지, rho 해상도, theta 해상도, 임계값
- **출력**: 검출된 직선들의 (ρ, θ) 매개변수
- **핵심 알고리즘**: 
  1. 누적기 배열 초기화
  2. 각 에지 픽셀에 대해 모든 각도에서 ρ 계산
  3. 누적기에 투표
  4. 임계값 이상의 로컬 최댓값 검출

### Harris Corner Detector 구현
```cpp
void custom_cv::cornerHarris(const cv::Mat& src, cv::Mat& dst, int blockSize, 
                            int ksize, double k, int borderType);
```

- **입력**: 그레이스케일 이미지, 윈도우 크기, Sobel 커널 크기, Harris 매개변수 k
- **출력**: 각 픽셀의 Corner Response 값
- **핵심 알고리즘**:
  1. Sobel 필터로 Ix, Iy 계산
  2. Structure Matrix 요소 계산 (Ix², Iy², IxIy)
  3. 가우시안 가중치 적용
  4. Harris 응답 함수 계산

## 성능 비교

커스텀 구현은 OpenCV의 최적화된 구현에 비해:
- **정확도**: 알고리즘이 동일하므로 유사한 결과
- **속도**: OpenCV 네이티브 구현보다 느림 (교육 목적 구현)
- **메모리**: 비슷한 메모리 사용량

## 학습 목적

이 구현을 통해 다음을 학습할 수 있습니다:
1. Hough 변환의 수학적 원리와 구현 세부사항
2. Harris Corner Detector의 알고리즘 단계별 이해
3. 이미지 처리에서 convolution과 필터링의 중요성
4. 매개변수 튜닝이 결과에 미치는 영향

## 참고사항

- 실제 프로덕션에서는 OpenCV의 최적화된 구현을 사용하는 것이 권장됩니다
- 본 구현은 교육 목적으로 가독성과 이해를 우선으로 작성되었습니다
- 매개변수는 이미지에 따라 조정이 필요할 수 있습니다