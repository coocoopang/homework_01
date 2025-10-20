#include "face_matcher.h"
#include <algorithm>
#include <cmath>

FaceMatcher::FaceMatcher() 
    : matchThreshold(0.7), detectionScale(1.1), 
      videoSource(""), isVideoFile(false), cascadeLoaded(false) {
    
    // OpenCV의 사전 훈련된 얼굴 검출기 로드
    // Haar cascade 파일들은 보통 OpenCV 설치 디렉터리에 있습니다
    std::vector<std::string> cascadePaths = {
        "haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_default.xml"
    };
    
    for (const auto& path : cascadePaths) {
        if (faceClassifier.load(path)) {
            cascadeLoaded = true;
            std::cout << "얼굴 검출기 로드 성공: " << path << std::endl;
            break;
        }
    }
    
    if (!cascadeLoaded) {
        std::cerr << "얼굴 검출기 로드 실패! Haar cascade 파일을 찾을 수 없습니다." << std::endl;
        std::cerr << "해결방법: Haar cascade 파일을 다운로드하고 Project4 폴더에 복사하세요." << std::endl;
        std::cerr << " wget https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml" << std::endl;
    }
}

FaceMatcher::~FaceMatcher() {
    if (videoCapture.isOpened()) {
        videoCapture.release();
    }
}

bool FaceMatcher::loadReferenceFace(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "기준 얼굴 이미지를 로드할 수 없습니다: " << imagePath << std::endl;
        return false;
    }
    
    // 얼굴 검출
    std::vector<cv::Rect> faces = detectFaces(image);
    if (faces.empty()) {
        std::cerr << "기준 이미지에서 얼굴을 찾을 수 없습니다!" << std::endl;
        return false;
    }
    
    // 가장 큰 얼굴을 기준으로 선택
    cv::Rect largestFace = *std::max_element(faces.begin(), faces.end(), 
        [](const cv::Rect& a, const cv::Rect& b) {
            return a.area() < b.area();
        });
    
    // 얼굴 영역 확장 및 추출
    cv::Rect expandedFace = FaceMatchingUtils::expandFaceRect(largestFace, image.size());
    referenceFace = image(expandedFace).clone();
    
    // 전처리
    referenceFace = preprocessFace(referenceFace);
    
    std::cout << "기준 얼굴 이미지 로드 완료: " << referenceFace.size() << std::endl;
    return true;
}

bool FaceMatcher::startWebcam(int deviceId) {
    videoCapture.open(deviceId);
    if (!videoCapture.isOpened()) {
        std::cerr << "웹캠을 열 수 없습니다! (Device ID: " << deviceId << ")" << std::endl;
        return false;
    }
    
    // 웹캠 설정
    videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    videoCapture.set(cv::CAP_PROP_FPS, 30);
    
    videoSource = "webcam";
    isVideoFile = false;
    
    std::cout << "웹캠 시작 완료!" << std::endl;
    return true;
}

bool FaceMatcher::loadVideoFile(const std::string& videoPath) {
    videoCapture.open(videoPath);
    if (!videoCapture.isOpened()) {
        std::cerr << "비디오 파일을 열 수 없습니다: " << videoPath << std::endl;
        return false;
    }
    
    videoSource = videoPath;
    isVideoFile = true;
    
    // 비디오 정보 출력
    int totalFrames = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = videoCapture.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "비디오 파일 로드 완료!" << std::endl;
    std::cout << "파일: " << videoPath << std::endl;
    std::cout << "정보: " << width << "x" << height << ", " << fps << " FPS, " << totalFrames << " 프레임" << std::endl;
    
    return true;
}

void FaceMatcher::runFaceMatching() {
    if (!videoCapture.isOpened()) {
        std::cerr << "비디오 소스가 열려있지 않습니다!" << std::endl;
        return;
    }
    
    if (referenceFace.empty()) {
        std::cerr << "기준 얼굴 이미지가 로드되지 않았습니다!" << std::endl;
        return;
    }
    
    std::string sourceType = isVideoFile ? "비디오 파일" : "웹캠";
    std::cout << sourceType << " 얼굴 매칭 시작!" << std::endl;
    if (isVideoFile) {
        std::cout << "파일: " << videoSource << std::endl;
    }
    
    std::cout << "조작법:" << std::endl;
    std::cout << " - ESC 또는 'q': 종료" << std::endl;
    std::cout << " - SPACE: 일시정지/재생 (비디오 파일)" << std::endl;
    std::cout << " - 't': 매칭 임계값 조정" << std::endl;
    std::cout << " - 's': 스크린샷 저장" << std::endl;
    std::cout << std::endl;
    
    cv::Mat frame;
    int frameCount = 0;
    int totalFrames = isVideoFile ? static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_COUNT)) : 0;
    bool paused = false;
    
    while (true) {
        if (!paused || !isVideoFile) {
            videoCapture >> frame;
            if (frame.empty()) {
                if (isVideoFile) {
                    std::cout << "비디오 재생 완료!" << std::endl;
                }
                break;
            }
            frameCount++;
        }
        
        // 얼굴 검출
        std::vector<cv::Rect> faces = detectFaces(frame);
        
        // 각 검출된 얼굴에 대해 매칭 검사
        for (const auto& faceRect : faces) {
            // 얼굴 영역 확장 및 추출
            cv::Rect expandedFace = FaceMatchingUtils::expandFaceRect(faceRect, frame.size());
            cv::Mat detectedFace = frame(expandedFace);
            
            // 얼굴 매칭
            double matchScore = matchFace(referenceFace, detectedFace);
            bool isMatch = matchScore > matchThreshold;
            
            // 결과 표시
            drawMatchResult(frame, expandedFace, matchScore, isMatch);
        }
        
        // 정보 표시
        std::string title = isVideoFile ? "Face Matching - Video" : "Face Matching - Webcam";
        cv::putText(frame, title, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        cv::putText(frame, "Threshold: " + std::to_string(int(matchThreshold * 100)) + "%", 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::putText(frame, "Faces: " + std::to_string(faces.size()), 
                   cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // 비디오 파일인 경우 프레임 정보 표시
        if (isVideoFile) {
            std::string frameInfo = "Frame: " + std::to_string(frameCount) + "/" + std::to_string(totalFrames);
            cv::putText(frame, frameInfo, cv::Point(10, 100), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            
            if (paused) {
                cv::putText(frame, "PAUSED", cv::Point(frame.cols/2 - 50, 50), 
                           cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 3);
            }
        }
        
        // 화면 출력
        cv::imshow(title, frame);
        
        // 키 입력 처리 - 비디오 파일의 경우 적절한 지연시간 설정
        int waitTime = isVideoFile ? 30 : 1;  // 비디오 파일: 30ms, 웹캠: 1ms
        int key = cv::waitKey(waitTime) & 0xFF;
        
        if (key == 27 || key == 'q') { // ESC 또는 'q'
            break;
        } else if (key == ' ' && isVideoFile) { // SPACE - 일시정지/재생
            paused = !paused;
            std::cout << (paused ? "일시정지" : "▶️ 재생") << std::endl;
        } else if (key == 't') { // 임계값 조정
            std::cout << "현재 임계값: " << int(matchThreshold * 100) << "%" << std::endl;
            std::cout << "새로운 임계값 입력 (0-100): ";
            int newThreshold;
            std::cin >> newThreshold;
            matchThreshold = std::max(0, std::min(100, newThreshold)) / 100.0;
            std::cout << "임계값 변경: " << int(matchThreshold * 100) << "%" << std::endl;
        } else if (key == 's') { // 스크린샷
            std::string filename = "screenshot_" + std::to_string(frameCount) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "스크린샷 저장: " << filename << std::endl;
        }
    }
    
    cv::destroyAllWindows();
    std::cout << "얼굴 매칭 종료!" << std::endl;
}

void FaceMatcher::runVideoFaceMatching() {
    // 기본 runFaceMatching()과 동일하지만 비디오 파일 전용 기능 추가
    runFaceMatching();
}

std::vector<cv::Rect> FaceMatcher::detectFaces(const cv::Mat& frame) {
    std::vector<cv::Rect> faces;
    
    // cascade가 로드되지 않았으면 빈 벡터 반환
    if (!cascadeLoaded || faceClassifier.empty()) {
        static bool errorShown = false;
        if (!errorShown) {
            std::cerr << "얼굴 검출기가 로드되지 않아 얼굴 검출을 수행할 수 없습니다." << std::endl;
            errorShown = true;
        }
        return faces;
    }
    
    // 입력 프레임 검증
    if (frame.empty()) {
        std::cerr << "빈 프레임이 입력되었습니다." << std::endl;
        return faces;
    }
    
    cv::Mat grayFrame;
    try {
        if (frame.channels() == 3) {
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        } else {
            grayFrame = frame.clone();
        }
        
        // 히스토그램 균등화로 조명 보정
        cv::equalizeHist(grayFrame, grayFrame);
        
        // 얼굴 검출 - 안전한 파라미터 사용
        faceClassifier.detectMultiScale(
            grayFrame,
            faces,
            1.1,               // scale factor (안정적인 값)
            3,                 // min neighbors
            0 | cv::CASCADE_SCALE_IMAGE,
            cv::Size(30, 30),  // minimum size
            cv::Size()         // maximum size (기본값)
        );
        
    } catch (const cv::Exception& e) {
        std::cerr << "detectMultiScale 에러: " << e.what() << std::endl;
        std::cerr << " 프레임 크기: " << frame.size() << ", 채널: " << frame.channels() << std::endl;
    }
    
    return faces;
}

double FaceMatcher::matchFace(const cv::Mat& face1, const cv::Mat& face2) {
    // 두 가지 방법을 조합: 템플릿 매칭 + 히스토그램 비교
    double templateScore = calculateTemplateMatchScore(face1, face2);
    double histogramScore = calculateHistogramSimilarity(face1, face2);
    
    // 가중 평균 (템플릿 매칭 60%, 히스토그램 40%)
    return templateScore * 0.6 + histogramScore * 0.4;
}

double FaceMatcher::calculateTemplateMatchScore(const cv::Mat& face1, const cv::Mat& face2) {
    // 동일한 크기로 조정
    cv::Mat resized1, resized2;
    cv::resize(face1, resized1, cv::Size(100, 100));
    cv::resize(face2, resized2, cv::Size(100, 100));
    
    // 그레이스케일 변환
    cv::Mat gray1, gray2;
    if (resized1.channels() == 3) cv::cvtColor(resized1, gray1, cv::COLOR_BGR2GRAY);
    else gray1 = resized1.clone();
    
    if (resized2.channels() == 3) cv::cvtColor(resized2, gray2, cv::COLOR_BGR2GRAY);
    else gray2 = resized2.clone();
    
    // 템플릿 매칭
    cv::Mat result;
    cv::matchTemplate(gray1, gray2, result, cv::TM_CCOEFF_NORMED);
    
    double minVal, maxVal;
    cv::minMaxLoc(result, &minVal, &maxVal);
    
    // 0-1 범위로 정규화
    return std::max(0.0, maxVal);
}

double FaceMatcher::calculateHistogramSimilarity(const cv::Mat& face1, const cv::Mat& face2) {
    // HSV 변환
    cv::Mat hsv1, hsv2;
    cv::cvtColor(face1, hsv1, cv::COLOR_BGR2HSV);
    cv::cvtColor(face2, hsv2, cv::COLOR_BGR2HSV);
    
    // 히스토그램 계산
    int histSize[] = {32, 32}; // H, S 채널
    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    const float* ranges[] = {hRanges, sRanges};
    int channels[] = {0, 1};
    
    cv::Mat hist1, hist2;
    cv::calcHist(&hsv1, 1, channels, cv::Mat(), hist1, 2, histSize, ranges);
    cv::calcHist(&hsv2, 1, channels, cv::Mat(), hist2, 2, histSize, ranges);
    
    // 히스토그램 정규화
    cv::normalize(hist1, hist1, 0, 1, cv::NORM_MINMAX);
    cv::normalize(hist2, hist2, 0, 1, cv::NORM_MINMAX);
    
    // 코릴레이션 계산
    double correlation = cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
    
    return std::max(0.0, correlation);
}

cv::Mat FaceMatcher::preprocessFace(const cv::Mat& face) {
    cv::Mat processed;
    
    // 크기 정규화
    cv::resize(face, processed, cv::Size(150, 150));
    
    // 조명 보정
    cv::Mat lab;
    cv::cvtColor(processed, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labChannels;
    cv::split(lab, labChannels);
    
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(labChannels[0], labChannels[0]);
    
    cv::merge(labChannels, lab);
    cv::cvtColor(lab, processed, cv::COLOR_Lab2BGR);
    
    return processed;
}

void FaceMatcher::drawMatchResult(cv::Mat& frame, const cv::Rect& faceRect, 
                                 double matchScore, bool isMatch) {
    // 매칭 결과에 따른 색상 결정
    cv::Scalar color = isMatch ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255); // 빨강 or 노랑
    int thickness = isMatch ? 4 : 2;
    
    // 사각형 그리기
    cv::rectangle(frame, faceRect, color, thickness);
    
    // 매칭 점수 표시
    std::string scoreText = std::to_string(int(matchScore * 100)) + "%";
    if (isMatch) {
        scoreText = "MATCH " + scoreText;
    }
    
    cv::Point textPos(faceRect.x, faceRect.y - 10);
    cv::putText(frame, scoreText, textPos, cv::FONT_HERSHEY_SIMPLEX, 
               0.6, color, 2);
    
    // 매칭된 경우 추가 표시
    if (isMatch) {
        // 원형 테두리 추가
        cv::Point center(faceRect.x + faceRect.width/2, faceRect.y + faceRect.height/2);
        int radius = std::max(faceRect.width, faceRect.height) / 2 + 10;
        cv::circle(frame, center, radius, cv::Scalar(0, 0, 255), 3);
        
        // "MATCHED!" 텍스트
        cv::putText(frame, "MATCHED!", 
                   cv::Point(faceRect.x, faceRect.y + faceRect.height + 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }
}

// 유틸리티 함수들 구현
namespace FaceMatchingUtils {
    cv::Mat resizeImage(const cv::Mat& image, int targetWidth) {
        if (image.empty()) return cv::Mat();
        
        double aspectRatio = (double)image.rows / image.cols;
        int targetHeight = static_cast<int>(targetWidth * aspectRatio);
        
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(targetWidth, targetHeight));
        return resized;
    }
    
    cv::Rect expandFaceRect(const cv::Rect& face, const cv::Size& imageSize, double factor) {
        int newWidth = static_cast<int>(face.width * factor);
        int newHeight = static_cast<int>(face.height * factor);
        
        int newX = face.x - (newWidth - face.width) / 2;
        int newY = face.y - (newHeight - face.height) / 2;
        
        // 이미지 경계 내로 제한
        newX = std::max(0, newX);
        newY = std::max(0, newY);
        newWidth = std::min(imageSize.width - newX, newWidth);
        newHeight = std::min(imageSize.height - newY, newHeight);
        
        return cv::Rect(newX, newY, newWidth, newHeight);
    }
    
    double scoreToPercent(double score) {
        return std::max(0.0, std::min(100.0, score * 100.0));
    }
}