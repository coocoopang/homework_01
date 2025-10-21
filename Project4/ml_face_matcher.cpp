#include "ml_face_matcher.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>

MLFaceMatcher::MLFaceMatcher() 
    : currentExtractorType(FeatureExtractorType::ORB),
      currentDistanceMetric(DistanceMetric::EUCLIDEAN),
      matchThreshold(0.7),
      knnK(2),
      ratioThreshold(0.7),
      cascadeLoaded(false),
      isVideoFile(false),
      videoSource("") {
    
    initializeFeatureExtractor();
    initializeFaceDetector();
}

MLFaceMatcher::~MLFaceMatcher() {
    if (videoCapture.isOpened()) {
        videoCapture.release();
    }
}

// === 설정 관련 메서드 ===

void MLFaceMatcher::setFeatureExtractor(FeatureExtractorType type) {
    currentExtractorType = type;
    initializeFeatureExtractor();
    std::cout << "🔧 특징 추출기 변경: ";
    switch (type) {
        case FeatureExtractorType::SIFT:
            std::cout << "SIFT" << std::endl;
            break;
        case FeatureExtractorType::ORB:
            std::cout << "ORB" << std::endl;
            break;
        case FeatureExtractorType::AKAZE:
            std::cout << "AKAZE" << std::endl;
            break;
    }
}

void MLFaceMatcher::setDistanceMetric(DistanceMetric metric) {
    currentDistanceMetric = metric;
    std::cout << "📏 거리 측정 방법 변경: ";
    switch (metric) {
        case DistanceMetric::EUCLIDEAN:
            std::cout << "유클리드 거리" << std::endl;
            break;
        case DistanceMetric::COSINE:
            std::cout << "코사인 거리" << std::endl;
            break;
        case DistanceMetric::HAMMING:
            std::cout << "해밍 거리" << std::endl;
            break;
    }
}

void MLFaceMatcher::setMatchThreshold(double threshold) {
    matchThreshold = threshold;
    std::cout << "🎯 매칭 임계값 설정: " << threshold << std::endl;
}

void MLFaceMatcher::setKNNParameters(int k, double ratioThreshold) {
    knnK = k;
    this->ratioThreshold = ratioThreshold;
    std::cout << "🔍 KNN 파라미터 설정: K=" << k << ", Ratio=" << ratioThreshold << std::endl;
}

// === 학습 데이터 관리 ===

bool MLFaceMatcher::addReferenceFace(const std::string& imagePath, const std::string& label) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "❌ 이미지 로드 실패: " << imagePath << std::endl;
        return false;
    }
    
    MLFaceMatchingUtils::Timer timer;
    timer.start();
    
    // 얼굴 검출
    std::vector<cv::Rect> faces = detectFaces(image);
    if (faces.empty()) {
        std::cerr << "❌ 얼굴 검출 실패: " << imagePath << std::endl;
        return false;
    }
    
    // 가장 큰 얼굴 선택
    cv::Rect largestFace = *std::max_element(faces.begin(), faces.end(),
        [](const cv::Rect& a, const cv::Rect& b) {
            return a.area() < b.area();
        });
    
    // 얼굴 영역 확장
    cv::Rect expandedFace = expandFaceRect(largestFace, image.size());
    cv::Mat faceImage = image(expandedFace);
    
    // 특징 추출
    FaceFeatures features = extractFaceFeatures(faceImage, expandedFace);
    features.label = label;
    
    if (features.descriptors.empty()) {
        std::cerr << "❌ 특징 추출 실패: " << imagePath << std::endl;
        return false;
    }
    
    referenceFeatures.push_back(features);
    labelCounts[label]++;
    
    double processingTime = timer.getElapsedMs();
    std::cout << "✅ 기준 얼굴 추가: " << label 
              << " (" << features.keypoints.size() << " 특징점, "
              << processingTime << "ms)" << std::endl;
    
    return true;
}

bool MLFaceMatcher::loadMultipleReferences(const std::string& directory) {
    std::vector<std::string> imageFiles = MLFaceMatchingUtils::getImageFiles(directory);
    
    if (imageFiles.empty()) {
        std::cerr << "❌ 디렉터리에서 이미지를 찾을 수 없습니다: " << directory << std::endl;
        return false;
    }
    
    int successCount = 0;
    for (const auto& filePath : imageFiles) {
        // 파일명에서 라벨 추출 (확장자 제거)
        std::string filename = std::filesystem::path(filePath).stem().string();
        
        if (addReferenceFace(filePath, filename)) {
            successCount++;
        }
    }
    
    std::cout << "📚 다중 기준 얼굴 로드 완료: " 
              << successCount << "/" << imageFiles.size() 
              << " 개 성공" << std::endl;
    
    return successCount > 0;
}

void MLFaceMatcher::clearReferenceData() {
    referenceFeatures.clear();
    labelCounts.clear();
    std::cout << "🗑️ 기준 데이터 초기화 완료" << std::endl;
}

int MLFaceMatcher::getReferenceCount() const {
    return static_cast<int>(referenceFeatures.size());
}

// === 비디오 소스 관리 ===

bool MLFaceMatcher::startWebcam(int deviceId) {
    videoCapture.open(deviceId);
    if (!videoCapture.isOpened()) {
        std::cerr << "❌ 웹캠 시작 실패 (Device ID: " << deviceId << ")" << std::endl;
        return false;
    }
    
    // 웹캠 설정
    videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    videoCapture.set(cv::CAP_PROP_FPS, 30);
    
    videoSource = "webcam";
    isVideoFile = false;
    
    std::cout << "✅ 웹캠 시작 완료!" << std::endl;
    return true;
}

bool MLFaceMatcher::loadVideoFile(const std::string& videoPath) {
    videoCapture.open(videoPath);
    if (!videoCapture.isOpened()) {
        std::cerr << "❌ 비디오 파일 로드 실패: " << videoPath << std::endl;
        return false;
    }
    
    videoSource = videoPath;
    isVideoFile = true;
    
    // 비디오 정보 출력
    int totalFrames = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = videoCapture.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "✅ 비디오 파일 로드 완료!" << std::endl;
    std::cout << "📁 파일: " << videoPath << std::endl;
    std::cout << "📊 정보: " << width << "x" << height 
              << ", " << fps << " FPS, " << totalFrames << " 프레임" << std::endl;
    
    return true;
}

// === 실시간 매칭 실행 ===

void MLFaceMatcher::runFaceMatching() {
    if (!videoCapture.isOpened()) {
        std::cerr << "❌ 비디오 소스가 열려있지 않습니다!" << std::endl;
        return;
    }
    
    if (referenceFeatures.empty()) {
        std::cerr << "❌ 기준 얼굴 데이터가 없습니다!" << std::endl;
        return;
    }
    
    std::string sourceType = isVideoFile ? "비디오 파일" : "웹캠";
    std::cout << "🎥 ML 기반 " << sourceType << " 얼굴 매칭 시작!" << std::endl;
    std::cout << "📚 학습된 기준 얼굴: " << referenceFeatures.size() << "개" << std::endl;
    
    if (isVideoFile) {
        std::cout << "📁 파일: " << videoSource << std::endl;
    }
    
    std::cout << "\n📋 조작법:" << std::endl;
    std::cout << "   - ESC 또는 'q': 종료" << std::endl;
    std::cout << "   - SPACE: 일시정지/재생 (비디오 파일)" << std::endl;
    std::cout << "   - 't': 매칭 임계값 조정" << std::endl;
    std::cout << "   - 's': 스크린샷 저장" << std::endl;
    std::cout << "   - 'f': 특징점 표시 토글" << std::endl;
    std::cout << "   - 'p': 성능 통계 출력" << std::endl;
    std::cout << std::endl;
    
    cv::Mat frame;
    int frameCount = 0;
    int totalFrames = isVideoFile ? static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_COUNT)) : 0;
    bool paused = false;
    bool showFeatures = false;
    
    MLFaceMatchingUtils::Timer frameTimer;
    
    while (true) {
        frameTimer.start();
        
        if (!paused || !isVideoFile) {
            videoCapture >> frame;
            if (frame.empty()) {
                if (isVideoFile) {
                    std::cout << "📹 비디오 재생 완료!" << std::endl;
                }
                break;
            }
            frameCount++;
        }
        
        // 얼굴 매칭 수행
        std::vector<MatchResult> matchResults = matchFacesInImage(frame);
        
        // 결과 표시
        for (const auto& result : matchResults) {
            drawMatchResult(frame, result);
            if (showFeatures && result.isMatch) {
                // 매칭된 얼굴의 특징점 표시는 별도 구현 필요
            }
        }
        
        // 정보 표시
        std::string title = "ML Face Matching - " + (isVideoFile ? "Video" : "Webcam");
        cv::putText(frame, title, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        cv::putText(frame, "References: " + std::to_string(referenceFeatures.size()), 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::putText(frame, "Matches: " + std::to_string(matchResults.size()), 
                   cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        if (isVideoFile) {
            std::string frameInfo = "Frame: " + std::to_string(frameCount) + "/" + std::to_string(totalFrames);
            cv::putText(frame, frameInfo, cv::Point(10, 100), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            
            if (paused) {
                cv::putText(frame, "PAUSED", cv::Point(frame.cols/2 - 50, 50), 
                           cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 3);
            }
        }
        
        cv::imshow(title, frame);
        
        // 성능 통계 업데이트
        double frameTime = frameTimer.getElapsedMs();
        updatePerformanceStats(frameTime, 0, 0); // 세부 시간은 matchFacesInImage에서 측정
        
        // 키 입력 처리
        int waitTime = isVideoFile ? 30 : 1;
        int key = cv::waitKey(waitTime) & 0xFF;
        
        if (key == 27 || key == 'q') { // ESC 또는 'q'
            break;
        } else if (key == ' ' && isVideoFile) { // SPACE - 일시정지/재생
            paused = !paused;
            std::cout << (paused ? "⏸️ 일시정지" : "▶️ 재생") << std::endl;
        } else if (key == 't') { // 임계값 조정
            std::cout << "현재 매칭 임계값: " << matchThreshold << std::endl;
            std::cout << "새로운 임계값 입력 (0.0-1.0): ";
            double newThreshold;
            std::cin >> newThreshold;
            setMatchThreshold(std::max(0.0, std::min(1.0, newThreshold)));
        } else if (key == 's') { // 스크린샷
            std::string filename = "ml_screenshot_" + std::to_string(frameCount) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "📸 스크린샷 저장: " << filename << std::endl;
        } else if (key == 'f') { // 특징점 표시 토글
            showFeatures = !showFeatures;
            std::cout << "🎯 특징점 표시: " << (showFeatures ? "ON" : "OFF") << std::endl;
        } else if (key == 'p') { // 성능 통계 출력
            printPerformanceStats();
        }
    }
    
    cv::destroyAllWindows();
    printPerformanceStats();
    std::cout << "👋 ML 얼굴 매칭 종료!" << std::endl;
}

// === 단일 이미지 매칭 ===

std::vector<MatchResult> MLFaceMatcher::matchFacesInImage(const cv::Mat& image) {
    std::vector<MatchResult> results;
    
    MLFaceMatchingUtils::Timer extractionTimer;
    extractionTimer.start();
    
    // 얼굴 검출
    std::vector<cv::Rect> faces = detectFaces(image);
    stats.facesDetected += faces.size();
    
    double extractionTime = extractionTimer.getElapsedMs();
    
    MLFaceMatchingUtils::Timer matchingTimer;
    matchingTimer.start();
    
    // 각 검출된 얼굴에 대해 특징 매칭 수행
    for (const auto& faceRect : faces) {
        cv::Rect expandedFace = expandFaceRect(faceRect, image.size());
        cv::Mat faceImage = image(expandedFace);
        
        // 특징 추출
        FaceFeatures queryFeatures = extractFaceFeatures(faceImage, expandedFace);
        
        if (!queryFeatures.descriptors.empty()) {
            // KNN 매칭 수행
            MatchResult result = performKNNMatching(queryFeatures);
            result.faceRegion = expandedFace;
            
            if (validateMatch(result)) {
                results.push_back(result);
                if (result.isMatch) {
                    stats.facesMatched++;
                }
            }
        }
    }
    
    double matchingTime = matchingTimer.getElapsedMs();
    updatePerformanceStats(0, extractionTime, matchingTime);
    
    return results;
}

// === 내부 메서드 구현 ===

void MLFaceMatcher::initializeFeatureExtractor() {
    switch (currentExtractorType) {
        case FeatureExtractorType::SIFT:
            try {
                featureExtractor = cv::SIFT::create(500); // 최대 500개 특징점
                matcher = cv::BFMatcher::create(cv::NORM_L2, true);
            } catch (const cv::Exception& e) {
                std::cerr << "⚠️ SIFT 생성 실패, ORB로 대체: " << e.what() << std::endl;
                currentExtractorType = FeatureExtractorType::ORB;
                initializeFeatureExtractor();
            }
            break;
            
        case FeatureExtractorType::ORB:
            featureExtractor = cv::ORB::create(500);
            matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
            break;
            
        case FeatureExtractorType::AKAZE:
            featureExtractor = cv::AKAZE::create();
            matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
            break;
    }
    
    if (!featureExtractor) {
        std::cerr << "❌ 특징 추출기 생성 실패!" << std::endl;
    }
}

bool MLFaceMatcher::initializeFaceDetector() {
    std::vector<std::string> cascadePaths = {
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    };
    
    for (const auto& path : cascadePaths) {
        if (faceClassifier.load(path)) {
            cascadeLoaded = true;
            std::cout << "✅ 얼굴 검출기 로드 성공: " << path << std::endl;
            return true;
        }
    }
    
    std::cerr << "❌ 얼굴 검출기 로드 실패!" << std::endl;
    return false;
}

std::vector<cv::Rect> MLFaceMatcher::detectFaces(const cv::Mat& frame) {
    std::vector<cv::Rect> faces;
    
    if (!cascadeLoaded || faceClassifier.empty() || frame.empty()) {
        return faces;
    }
    
    cv::Mat grayFrame;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    } else {
        grayFrame = frame.clone();
    }
    
    cv::equalizeHist(grayFrame, grayFrame);
    
    try {
        faceClassifier.detectMultiScale(
            grayFrame, faces,
            1.1, 3,
            0 | cv::CASCADE_SCALE_IMAGE,
            cv::Size(30, 30)
        );
    } catch (const cv::Exception& e) {
        std::cerr << "❌ 얼굴 검출 오류: " << e.what() << std::endl;
    }
    
    return faces;
}

FaceFeatures MLFaceMatcher::extractFaceFeatures(const cv::Mat& faceImage, const cv::Rect& faceRegion) {
    FaceFeatures features;
    features.faceRegion = faceRegion;
    
    if (!featureExtractor || faceImage.empty()) {
        return features;
    }
    
    cv::Mat processedFace = faceImage.clone();
    if (!preprocessFace(processedFace)) {
        return features;
    }
    
    try {
        featureExtractor->detectAndCompute(processedFace, cv::noArray(), 
                                         features.keypoints, features.descriptors);
    } catch (const cv::Exception& e) {
        std::cerr << "❌ 특징 추출 오류: " << e.what() << std::endl;
    }
    
    return features;
}

bool MLFaceMatcher::preprocessFace(cv::Mat& face) {
    if (face.empty()) return false;
    
    // 크기 정규화
    cv::resize(face, face, cv::Size(128, 128));
    
    // 조명 정규화
    face = MLFaceMatchingUtils::normalizeIllumination(face);
    
    return true;
}

MatchResult MLFaceMatcher::performKNNMatching(const FaceFeatures& queryFeatures) {
    MatchResult result;
    
    if (queryFeatures.descriptors.empty() || referenceFeatures.empty()) {
        return result;
    }
    
    double bestDistance = std::numeric_limits<double>::max();
    std::string bestLabel;
    int bestMatchCount = 0;
    
    // 각 기준 얼굴과 비교
    for (const auto& refFeature : referenceFeatures) {
        if (refFeature.descriptors.empty()) continue;
        
        try {
            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher->knnMatch(queryFeatures.descriptors, refFeature.descriptors, knnMatches, knnK);
            
            // Lowe's ratio test 적용
            std::vector<cv::DMatch> goodMatches = filterMatches(knnMatches);
            
            if (goodMatches.size() >= 4) { // 최소 매칭 개수
                // 거리 계산
                double avgDistance = 0.0;
                for (const auto& match : goodMatches) {
                    avgDistance += match.distance;
                }
                avgDistance /= goodMatches.size();
                
                // 가장 좋은 매칭 업데이트
                if (avgDistance < bestDistance && goodMatches.size() > bestMatchCount) {
                    bestDistance = avgDistance;
                    bestLabel = refFeature.label;
                    bestMatchCount = goodMatches.size();
                    result.matches = goodMatches;
                }
            }
        } catch (const cv::Exception& e) {
            std::cerr << "❌ 매칭 오류: " << e.what() << std::endl;
        }
    }
    
    // 결과 설정
    if (bestMatchCount > 0) {
        result.distance = bestDistance;
        result.confidence = calculateConfidence(result.matches, bestDistance);
        result.matchedLabel = bestLabel;
        result.isMatch = (result.confidence > matchThreshold);
    }
    
    return result;
}

std::vector<cv::DMatch> MLFaceMatcher::filterMatches(const std::vector<std::vector<cv::DMatch>>& knnMatches) {
    std::vector<cv::DMatch> goodMatches;
    
    for (const auto& match : knnMatches) {
        if (match.size() == 2) {
            // Lowe's ratio test
            if (match[0].distance < ratioThreshold * match[1].distance) {
                goodMatches.push_back(match[0]);
            }
        } else if (match.size() == 1) {
            // 단일 매칭의 경우 거리가 충분히 작은지 확인
            if (match[0].distance < 100.0) { // 임계값은 특징 추출기에 따라 조정 필요
                goodMatches.push_back(match[0]);
            }
        }
    }
    
    return goodMatches;
}

double MLFaceMatcher::calculateConfidence(const std::vector<cv::DMatch>& matches, double distance) {
    if (matches.empty()) return 0.0;
    
    // 매칭 개수와 평균 거리를 고려한 신뢰도 계산
    double matchRatio = static_cast<double>(matches.size()) / 100.0; // 정규화
    double distanceScore = std::max(0.0, 1.0 - distance / 200.0); // 거리 점수
    
    return std::min(1.0, matchRatio * 0.6 + distanceScore * 0.4);
}

bool MLFaceMatcher::validateMatch(const MatchResult& result) {
    // 기본 검증: 최소 매칭 개수와 거리 임계값
    return result.matches.size() >= 3 && result.distance < 150.0;
}

void MLFaceMatcher::drawMatchResult(cv::Mat& frame, const MatchResult& result) {
    cv::Scalar color = result.isMatch ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255); // 빨강 또는 노랑
    int thickness = result.isMatch ? 4 : 2;
    
    // 사각형 그리기
    cv::rectangle(frame, result.faceRegion, color, thickness);
    
    // 레이블과 신뢰도 표시
    std::string text;
    if (result.isMatch) {
        text = result.matchedLabel + " (" + std::to_string(static_cast<int>(result.confidence * 100)) + "%)";
    } else {
        text = "Unknown (" + std::to_string(static_cast<int>(result.confidence * 100)) + "%)";
    }
    
    cv::Point textPos(result.faceRegion.x, result.faceRegion.y - 10);
    cv::putText(frame, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    
    // 매칭된 경우 추가 표시
    if (result.isMatch) {
        cv::Point center(result.faceRegion.x + result.faceRegion.width/2, 
                        result.faceRegion.y + result.faceRegion.height/2);
        int radius = std::max(result.faceRegion.width, result.faceRegion.height) / 2 + 10;
        cv::circle(frame, center, radius, cv::Scalar(0, 0, 255), 3);
        
        // 매칭 정보
        std::string matchInfo = "Matches: " + std::to_string(result.matches.size());
        cv::putText(frame, matchInfo, 
                   cv::Point(result.faceRegion.x, result.faceRegion.y + result.faceRegion.height + 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
}

cv::Rect MLFaceMatcher::expandFaceRect(const cv::Rect& face, const cv::Size& imageSize, double factor) {
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

void MLFaceMatcher::updatePerformanceStats(double frameTime, double extractionTime, double matchingTime) {
    stats.totalFrames++;
    stats.totalProcessingTime += frameTime;
    if (extractionTime > 0) stats.avgFeatureExtractionTime += extractionTime;
    if (matchingTime > 0) stats.avgMatchingTime += matchingTime;
}

void MLFaceMatcher::printPerformanceStats() {
    std::cout << "\n📊 성능 통계:" << std::endl;
    std::cout << "   - 총 프레임: " << stats.totalFrames << std::endl;
    std::cout << "   - 검출된 얼굴: " << stats.facesDetected << std::endl;
    std::cout << "   - 매칭된 얼굴: " << stats.facesMatched << std::endl;
    std::cout << "   - 매칭률: " << (stats.facesDetected > 0 ? 
                   (double)stats.facesMatched / stats.facesDetected * 100 : 0) << "%" << std::endl;
    std::cout << "   - 평균 프레임 처리 시간: " << 
                 (stats.totalFrames > 0 ? stats.totalProcessingTime / stats.totalFrames : 0) << "ms" << std::endl;
    std::cout << std::endl;
}

// === 유틸리티 함수 구현 ===

namespace MLFaceMatchingUtils {
    
    Timer::Timer() {}
    
    void Timer::start() {
        startTime = std::chrono::high_resolution_clock::now();
    }
    
    double Timer::getElapsedMs() {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        return duration.count() / 1000.0; // 밀리초로 변환
    }
    
    cv::Mat normalizeIllumination(const cv::Mat& image) {
        cv::Mat result;
        if (image.channels() == 3) {
            cv::Mat lab;
            cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
            
            std::vector<cv::Mat> labChannels;
            cv::split(lab, labChannels);
            
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
            clahe->apply(labChannels[0], labChannels[0]);
            
            cv::merge(labChannels, lab);
            cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
        } else {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
            clahe->apply(image, result);
        }
        return result;
    }
    
    std::vector<std::string> getImageFiles(const std::string& directory) {
        std::vector<std::string> imageFiles;
        std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
        
        try {
            for (const auto& entry : std::filesystem::directory_iterator(directory)) {
                if (entry.is_regular_file()) {
                    std::string filePath = entry.path().string();
                    std::string extension = entry.path().extension().string();
                    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                    
                    if (std::find(extensions.begin(), extensions.end(), extension) != extensions.end()) {
                        imageFiles.push_back(filePath);
                    }
                }
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "❌ 디렉터리 읽기 오류: " << e.what() << std::endl;
        }
        
        return imageFiles;
    }
}