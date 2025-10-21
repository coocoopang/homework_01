#include "face_matcher.h"
#include <algorithm>
#include <cmath>
#include <iomanip>

FaceMatcher::FaceMatcher() 
    : videoSource(""), isVideoFile(false), cascadeLoaded(false),
      matchThreshold(0.7), detectorType("SIFT"), matcherType("BF"),
      minMatchCount(10), maxDistanceRatio(0.75) {
    
    // OpenCV의 사전 훈련된 얼굴 검출기 로드
    std::vector<std::string> cascadePaths = {
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "./haarcascades/haarcascade_frontalface_alt.xml",
        "./haarcascades/haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_default.xml"
    };
    
    for (const auto& path : cascadePaths) {
        if (faceClassifier.load(path)) {
            cascadeLoaded = true;
            std::cout << "✅ 얼굴 검출기 로드 성공: " << path << std::endl;
            break;
        }
    }
    
    if (!cascadeLoaded) {
        std::cerr << "❌ 얼굴 검출기 로드 실패! Haar cascade 파일을 찾을 수 없습니다." << std::endl;
        std::cerr << "📝 해결방법: Haar cascade 파일을 다운로드하고 Project4 폴더에 복사하세요." << std::endl;
        std::cerr << "   wget https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml" << std::endl;
    }
    
    // 기본 특징점 검출기와 매처 생성
    detector = FeatureMatchingUtils::createFeatureDetector(detectorType);
    matcher = FeatureMatchingUtils::createDescriptorMatcher(matcherType, detectorType);
    
    std::cout << "🔧 특징점 기반 얼굴 매칭 시스템 초기화 완료!" << std::endl;
    std::cout << "   - 특징점 검출기: " << detectorType << std::endl;
    std::cout << "   - 매처: " << matcherType << std::endl;
    std::cout << "   - 최소 매칭 개수: " << minMatchCount << std::endl;
}

FaceMatcher::~FaceMatcher() {
    if (videoCapture.isOpened()) {
        videoCapture.release();
    }
}

bool FaceMatcher::loadReferenceFace(const std::string& imagePath) {
    referenceFaceImage = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (referenceFaceImage.empty()) {
        std::cerr << "❌ 기준 얼굴 이미지를 로드할 수 없습니다: " << imagePath << std::endl;
        return false;
    }
    
    // 얼굴 검출
    std::vector<cv::Rect> faces = detectFaces(referenceFaceImage);
    if (faces.empty()) {
        std::cerr << "❌ 기준 이미지에서 얼굴을 찾을 수 없습니다!" << std::endl;
        return false;
    }
    
    // 가장 큰 얼굴을 기준으로 선택
    cv::Rect largestFace = *std::max_element(faces.begin(), faces.end(), 
        [](const cv::Rect& a, const cv::Rect& b) {
            return a.area() < b.area();
        });
    
    // 얼굴 영역 확장 및 추출
    cv::Rect expandedFace = FeatureMatchingUtils::expandFaceRect(largestFace, referenceFaceImage.size());
    cv::Mat faceROI = referenceFaceImage(expandedFace).clone();
    
    // 전처리
    faceROI = preprocessFace(faceROI);
    
    // 기준 얼굴에서 특징점 추출
    extractFeatures(faceROI, referenceKeypoints, referenceDescriptors);
    
    if (referenceKeypoints.empty()) {
        std::cerr << "❌ 기준 얼굴에서 특징점을 추출할 수 없습니다!" << std::endl;
        return false;
    }
    
    double featureDensity = FeatureMatchingUtils::calculateFeatureDensity(referenceKeypoints, faceROI.size());
    
    std::cout << "✅ 기준 얼굴 특징점 추출 완료!" << std::endl;
    std::cout << "   - 이미지 크기: " << faceROI.size() << std::endl;
    std::cout << "   - 특징점 개수: " << referenceKeypoints.size() << std::endl;
    std::cout << "   - 특징점 밀도: " << std::fixed << std::setprecision(4) << featureDensity << " points/pixel²" << std::endl;
    
    return true;
}

bool FaceMatcher::startWebcam(int deviceId) {
    videoCapture.open(deviceId);
    if (!videoCapture.isOpened()) {
        std::cerr << "❌ 웹캠을 열 수 없습니다! (Device ID: " << deviceId << ")" << std::endl;
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

bool FaceMatcher::loadVideoFile(const std::string& videoPath) {
    videoCapture.open(videoPath);
    if (!videoCapture.isOpened()) {
        std::cerr << "❌ 비디오 파일을 열 수 없습니다: " << videoPath << std::endl;
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
    std::cout << "📊 정보: " << width << "x" << height << ", " << fps << " FPS, " << totalFrames << " 프레임" << std::endl;
    
    return true;
}

void FaceMatcher::runFaceMatching() {
    if (!videoCapture.isOpened()) {
        std::cerr << "❌ 비디오 소스가 열려있지 않습니다!" << std::endl;
        return;
    }
    
    if (referenceKeypoints.empty() || referenceDescriptors.empty()) {
        std::cerr << "❌ 기준 얼굴 특징점이 준비되지 않았습니다!" << std::endl;
        return;
    }
    
    std::string sourceType = isVideoFile ? "비디오 파일" : "웹캠";
    std::cout << "🎥 " << sourceType << " 특징점 매칭 시작!" << std::endl;
    if (isVideoFile) {
        std::cout << "📁 파일: " << videoSource << std::endl;
    }
    
    std::cout << "📋 조작법:" << std::endl;
    std::cout << "   - ESC 또는 'q': 종료" << std::endl;
    std::cout << "   - SPACE: 일시정지/재생 (비디오 파일)" << std::endl;
    std::cout << "   - 't': 매칭 임계값 조정" << std::endl;
    std::cout << "   - 's': 스크린샷 저장" << std::endl;
    std::cout << "   - 'd': 특징점 검출기 변경 (SIFT ↔ ORB)" << std::endl;
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
                    std::cout << "📹 비디오 재생 완료!" << std::endl;
                }
                break;
            }
            frameCount++;
        }
        
        // 얼굴 검출
        std::vector<cv::Rect> faces = detectFaces(frame);
        
        // 각 검출된 얼굴에 대해 특징점 매칭 수행
        for (const auto& faceRect : faces) {
            // 얼굴 영역 확장 및 추출
            cv::Rect expandedFace = FeatureMatchingUtils::expandFaceRect(faceRect, frame.size());
            cv::Mat detectedFace = frame(expandedFace);
            
            // 특징점 기반 얼굴 매칭
            double matchScore = matchFaceByFeatures(detectedFace);
            bool isMatch = matchScore >= matchThreshold;
            
            // 결과 표시
            int matchCount = static_cast<int>(matchScore * 100); // 임시로 백분율을 매칭 개수로 사용
            drawMatchResult(frame, expandedFace, matchScore, isMatch, matchCount);
        }
        
        // 정보 표시
        std::string title = isVideoFile ? "Feature Matching - Video" : "Feature Matching - Webcam";
        cv::putText(frame, title, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        cv::putText(frame, "Threshold: " + std::to_string(int(matchThreshold * 100)) + "%", 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::putText(frame, "Faces: " + std::to_string(faces.size()), 
                   cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::putText(frame, "Detector: " + detectorType, 
                   cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::putText(frame, "Ref Features: " + std::to_string(referenceKeypoints.size()), 
                   cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // 비디오 파일인 경우 프레임 정보 표시
        if (isVideoFile) {
            std::string frameInfo = "Frame: " + std::to_string(frameCount) + "/" + std::to_string(totalFrames);
            cv::putText(frame, frameInfo, cv::Point(10, 140), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            
            if (paused) {
                cv::putText(frame, "PAUSED", cv::Point(frame.cols/2 - 50, 50), 
                           cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 3);
            }
        }
        
        // 화면 출력
        cv::imshow(title, frame);
        
        // 키 입력 처리
        int waitTime = isVideoFile ? 30 : 1;
        int key = cv::waitKey(waitTime) & 0xFF;
        
        if (key == 27 || key == 'q') { // ESC 또는 'q'
            break;
        } else if (key == ' ' && isVideoFile) { // SPACE - 일시정지/재생
            paused = !paused;
            std::cout << (paused ? "⏸️ 일시정지" : "▶️ 재생") << std::endl;
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
            std::cout << "📸 스크린샷 저장: " << filename << std::endl;
        } else if (key == 'd') { // 특징점 검출기 변경
            detectorType = (detectorType == "SIFT") ? "ORB" : "SIFT";
            detector = FeatureMatchingUtils::createFeatureDetector(detectorType);
            matcher = FeatureMatchingUtils::createDescriptorMatcher(matcherType, detectorType);
            
            // 기준 얼굴 특징점 재추출
            if (!referenceFaceImage.empty()) {
                std::vector<cv::Rect> faces = detectFaces(referenceFaceImage);
                if (!faces.empty()) {
                    cv::Rect largestFace = *std::max_element(faces.begin(), faces.end(), 
                        [](const cv::Rect& a, const cv::Rect& b) { return a.area() < b.area(); });
                    cv::Rect expandedFace = FeatureMatchingUtils::expandFaceRect(largestFace, referenceFaceImage.size());
                    cv::Mat faceROI = preprocessFace(referenceFaceImage(expandedFace));
                    extractFeatures(faceROI, referenceKeypoints, referenceDescriptors);
                }
            }
            
            std::cout << "🔄 특징점 검출기 변경: " << detectorType << " (기준 특징점: " << referenceKeypoints.size() << "개)" << std::endl;
        }
    }
    
    cv::destroyAllWindows();
    std::cout << "👋 특징점 매칭 종료!" << std::endl;
}

void FaceMatcher::runVideoFaceMatching() {
    runFaceMatching();
}

std::vector<cv::Rect> FaceMatcher::detectFaces(const cv::Mat& frame) {
    std::vector<cv::Rect> faces;
    
    if (!cascadeLoaded || faceClassifier.empty()) {
        static bool errorShown = false;
        if (!errorShown) {
            std::cerr << "⚠️ 얼굴 검출기가 로드되지 않아 얼굴 검출을 수행할 수 없습니다." << std::endl;
            errorShown = true;
        }
        return faces;
    }
    
    if (frame.empty()) {
        std::cerr << "⚠️ 빈 프레임이 입력되었습니다." << std::endl;
        return faces;
    }
    
    cv::Mat grayFrame;
    try {
        if (frame.channels() == 3) {
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        } else {
            grayFrame = frame.clone();
        }
        
        cv::equalizeHist(grayFrame, grayFrame);
        
        faceClassifier.detectMultiScale(
            grayFrame,
            faces,
            1.1,
            3,
            0 | cv::CASCADE_SCALE_IMAGE,
            cv::Size(30, 30),
            cv::Size()
        );
        
    } catch (const cv::Exception& e) {
        std::cerr << "❌ detectMultiScale 에러: " << e.what() << std::endl;
    }
    
    return faces;
}

double FaceMatcher::matchFaceByFeatures(const cv::Mat& detectedFace) {
    if (detectedFace.empty() || referenceDescriptors.empty()) {
        return 0.0;
    }
    
    // 전처리
    cv::Mat processedFace = preprocessFace(detectedFace);
    
    // 특징점 추출
    std::vector<cv::KeyPoint> detectedKeypoints;
    cv::Mat detectedDescriptors;
    extractFeatures(processedFace, detectedKeypoints, detectedDescriptors);
    
    if (detectedKeypoints.empty() || detectedDescriptors.empty()) {
        return 0.0;
    }
    
    // 최근접 매칭
    std::vector<cv::DMatch> matches = findNearestMatches(referenceDescriptors, detectedDescriptors);
    
    if (matches.empty()) {
        return 0.0;
    }
    
    // 기하학적 검증
    int verifiedMatches = verifyGeometry(referenceKeypoints, detectedKeypoints, matches);
    
    // 매칭 점수 계산
    double score = calculateMatchScore(verifiedMatches, referenceKeypoints.size());
    
    // 디버그 정보 출력 (개발 시에만)
    static int debugCount = 0;
    if (debugCount++ % 30 == 0) { // 30프레임마다 한 번씩
        printMatchingInfo(matches.size(), verifiedMatches, score);
    }
    
    return score;
}

void FaceMatcher::extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    if (!detector) {
        std::cerr << "❌ 특징점 검출기가 초기화되지 않았습니다!" << std::endl;
        return;
    }
    
    try {
        detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    } catch (const cv::Exception& e) {
        std::cerr << "❌ 특징점 추출 에러: " << e.what() << std::endl;
        keypoints.clear();
        descriptors = cv::Mat();
    }
}

std::vector<cv::DMatch> FaceMatcher::findNearestMatches(const cv::Mat& descriptors1, const cv::Mat& descriptors2) {
    std::vector<cv::DMatch> goodMatches;
    
    if (!matcher || descriptors1.empty() || descriptors2.empty()) {
        return goodMatches;
    }
    
    try {
        if (detectorType == "SIFT") {
            // SIFT의 경우 k-NN 매칭 사용 (Lowe's ratio test)
            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);
            goodMatches = filterGoodMatches(knnMatches);
        } else {
            // ORB의 경우 단순 매칭 후 거리 기반 필터링
            std::vector<cv::DMatch> matches;
            matcher->match(descriptors1, descriptors2, matches);
            
            // 거리 기반 필터링
            double maxDist = 0; double minDist = 100;
            for (const auto& match : matches) {
                double dist = match.distance;
                if (dist < minDist) minDist = dist;
                if (dist > maxDist) maxDist = dist;
            }
            
            double threshold = std::max(2 * minDist, 30.0);
            for (const auto& match : matches) {
                if (match.distance <= threshold) {
                    goodMatches.push_back(match);
                }
            }
        }
    } catch (const cv::Exception& e) {
        std::cerr << "❌ 특징점 매칭 에러: " << e.what() << std::endl;
    }
    
    return goodMatches;
}

std::vector<cv::DMatch> FaceMatcher::filterGoodMatches(const std::vector<std::vector<cv::DMatch>>& knnMatches) {
    std::vector<cv::DMatch> goodMatches;
    
    for (const auto& matchPair : knnMatches) {
        if (matchPair.size() == 2) {
            // Lowe's ratio test
            if (matchPair[0].distance < maxDistanceRatio * matchPair[1].distance) {
                goodMatches.push_back(matchPair[0]);
            }
        }
    }
    
    return goodMatches;
}

int FaceMatcher::verifyGeometry(const std::vector<cv::KeyPoint>& keypoints1, 
                               const std::vector<cv::KeyPoint>& keypoints2,
                               const std::vector<cv::DMatch>& matches) {
    if (matches.size() < 4) { // RANSAC에는 최소 4개의 점이 필요
        return matches.size();
    }
    
    try {
        // 매칭된 점들 추출
        std::vector<cv::Point2f> points1, points2;
        for (const auto& match : matches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }
        
        // RANSAC을 사용하여 호모그래피 추정 및 inlier 찾기
        cv::Mat mask;
        cv::findHomography(points1, points2, cv::RANSAC, 3.0, mask);
        
        // inlier 개수 세기
        int inlierCount = 0;
        if (!mask.empty()) {
            for (int i = 0; i < mask.rows; ++i) {
                if (mask.at<uchar>(i) > 0) {
                    inlierCount++;
                }
            }
        }
        
        return inlierCount;
    } catch (const cv::Exception& e) {
        std::cerr << "❌ 기하학적 검증 에러: " << e.what() << std::endl;
        return 0;
    }
}

double FaceMatcher::calculateMatchScore(int goodMatches, int totalKeypoints) {
    if (totalKeypoints == 0) return 0.0;
    
    // 매칭 비율 계산
    double matchRatio = static_cast<double>(goodMatches) / totalKeypoints;
    
    // 최소 매칭 개수 조건 확인
    if (goodMatches < minMatchCount) {
        matchRatio *= 0.5; // 페널티 적용
    }
    
    // 0-1 범위로 클램핑
    return std::min(1.0, std::max(0.0, matchRatio));
}

cv::Mat FaceMatcher::preprocessFace(const cv::Mat& face) {
    cv::Mat processed;
    
    // 크기 정규화
    cv::resize(face, processed, cv::Size(200, 200));
    
    // 조명 보정 (CLAHE)
    cv::Mat lab;
    cv::cvtColor(processed, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labChannels;
    cv::split(lab, labChannels);
    
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(labChannels[0], labChannels[0]);
    
    cv::merge(labChannels, lab);
    cv::cvtColor(lab, processed, cv::COLOR_Lab2BGR);
    
    // 가우시안 블러로 노이즈 제거
    cv::GaussianBlur(processed, processed, cv::Size(3, 3), 0.5);
    
    return processed;
}

void FaceMatcher::drawMatchResult(cv::Mat& frame, const cv::Rect& faceRect, 
                                 double matchScore, bool isMatch, int matchCount) {
    // 매칭 결과에 따른 색상 결정
    cv::Scalar color = isMatch ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255); // 빨강 or 노랑
    int thickness = isMatch ? 4 : 2;
    
    // 사각형 그리기
    cv::rectangle(frame, faceRect, color, thickness);
    
    // 매칭 정보 표시
    std::string scoreText = std::to_string(int(matchScore * 100)) + "% (" + std::to_string(matchCount) + ")";
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
        
        // "FEATURE MATCHED!" 텍스트
        cv::putText(frame, "FEATURE MATCHED!", 
                   cv::Point(faceRect.x, faceRect.y + faceRect.height + 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }
}

void FaceMatcher::drawFeatureMatches(const cv::Mat& detectedFace, const std::vector<cv::KeyPoint>& keypoints, 
                                   const std::vector<cv::DMatch>& goodMatches) {
    // 특징점 매칭 시각화 (디버깅용)
    cv::Mat matchImg;
    cv::drawMatches(referenceFaceImage, referenceKeypoints, detectedFace, keypoints,
                   goodMatches, matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    cv::imshow("Feature Matches", matchImg);
}

void FaceMatcher::printMatchingInfo(int totalMatches, int goodMatches, double score) {
    std::cout << "🔍 매칭 정보: 전체=" << totalMatches 
              << ", 검증됨=" << goodMatches 
              << ", 점수=" << std::fixed << std::setprecision(3) << score << std::endl;
}

// 유틸리티 함수들 구현
namespace FeatureMatchingUtils {
    cv::Ptr<cv::Feature2D> createFeatureDetector(const std::string& type) {
        if (type == "SIFT") {
            return cv::SIFT::create(500); // 최대 500개 특징점
        } else if (type == "ORB") {
            return cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
        } else {
            std::cerr << "❌ 지원하지 않는 특징점 검출기: " << type << std::endl;
            return cv::SIFT::create(500); // 기본값으로 SIFT 사용
        }
    }
    
    cv::Ptr<cv::DescriptorMatcher> createDescriptorMatcher(const std::string& type, const std::string& detectorType) {
        if (type == "BF") {
            if (detectorType == "SIFT") {
                return cv::BFMatcher::create(cv::NORM_L2);
            } else { // ORB
                return cv::BFMatcher::create(cv::NORM_HAMMING);
            }
        } else if (type == "FLANN") {
            if (detectorType == "SIFT") {
                return cv::FlannBasedMatcher::create();
            } else {
                // ORB용 FLANN 설정
                auto indexParams = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1);
                auto searchParams = cv::makePtr<cv::flann::SearchParams>(50);
                return cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
            }
        } else {
            std::cerr << "❌ 지원하지 않는 매처 타입: " << type << std::endl;
            return cv::BFMatcher::create();
        }
    }
    
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
    
    double calculateFeatureDensity(const std::vector<cv::KeyPoint>& keypoints, const cv::Size& imageSize) {
        if (keypoints.empty() || imageSize.area() == 0) return 0.0;
        return static_cast<double>(keypoints.size()) / imageSize.area();
    }
}