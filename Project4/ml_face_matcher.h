#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/ml.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <map>

/**
 * 머신러닝 기반 얼굴 매칭 시스템
 * - 최근접 특징 매칭(Nearest Neighbor Feature Matching) 기법 활용
 * - 다중 특징 추출기 지원 (SIFT, ORB, AKAZE)
 * - KNN 기반 분류 및 매칭
 * - 웹캠 및 MP4 비디오 파일 지원
 */

// 특징 추출 방법 열거형
enum class FeatureExtractorType {
    SIFT,
    ORB,
    AKAZE
};

// 거리 측정 방법 열거형
enum class DistanceMetric {
    EUCLIDEAN,
    COSINE,
    HAMMING
};

// 얼굴 특징 정보 구조체
struct FaceFeatures {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Rect faceRegion;
    std::string label;
    double confidence;
    
    FaceFeatures() : confidence(0.0) {}
};

// 매칭 결과 구조체
struct MatchResult {
    bool isMatch;
    double confidence;
    double distance;
    std::string matchedLabel;
    cv::Rect faceRegion;
    std::vector<cv::DMatch> matches;
    
    MatchResult() : isMatch(false), confidence(0.0), distance(0.0) {}
};

class MLFaceMatcher {
public:
    MLFaceMatcher();
    ~MLFaceMatcher();
    
    // === 설정 관련 메서드 ===
    void setFeatureExtractor(FeatureExtractorType type);
    void setDistanceMetric(DistanceMetric metric);
    void setMatchThreshold(double threshold);
    void setKNNParameters(int k, double ratioThreshold = 0.7);
    
    // === 학습 데이터 관리 ===
    bool addReferenceFace(const std::string& imagePath, const std::string& label);
    bool loadMultipleReferences(const std::string& directory);
    void clearReferenceData();
    int getReferenceCount() const;
    
    // === 비디오 소스 관리 ===
    bool startWebcam(int deviceId = 0);
    bool loadVideoFile(const std::string& videoPath);
    
    // === 실시간 매칭 실행 ===
    void runFaceMatching();
    
    // === 단일 이미지 매칭 ===
    std::vector<MatchResult> matchFacesInImage(const cv::Mat& image);
    
    // === 모델 저장/로드 ===
    bool saveModel(const std::string& modelPath);
    bool loadModel(const std::string& modelPath);
    
    // === 성능 분석 ===
    void printPerformanceStats();
    
private:
    // === 핵심 구성요소 ===
    cv::Ptr<cv::Feature2D> featureExtractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::CascadeClassifier faceClassifier;
    cv::VideoCapture videoCapture;
    
    // === 설정 변수 ===
    FeatureExtractorType currentExtractorType;
    DistanceMetric currentDistanceMetric;
    double matchThreshold;
    int knnK;
    double ratioThreshold;
    bool cascadeLoaded;
    bool isVideoFile;
    std::string videoSource;
    
    // === 학습 데이터 ===
    std::vector<FaceFeatures> referenceFeatures;
    std::map<std::string, int> labelCounts;
    
    // === 성능 통계 ===
    struct PerformanceStats {
        int totalFrames;
        int facesDetected;
        int facesMatched;
        double totalProcessingTime;
        double avgFeatureExtractionTime;
        double avgMatchingTime;
        
        PerformanceStats() : totalFrames(0), facesDetected(0), facesMatched(0), 
                           totalProcessingTime(0.0), avgFeatureExtractionTime(0.0), avgMatchingTime(0.0) {}
    } stats;
    
    // === 내부 메서드 ===
    
    // 특징 추출 관련
    void initializeFeatureExtractor();
    FaceFeatures extractFaceFeatures(const cv::Mat& faceImage, const cv::Rect& faceRegion);
    bool preprocessFace(cv::Mat& face);
    
    // 얼굴 검출
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame);
    bool initializeFaceDetector();
    
    // 매칭 관련
    MatchResult performKNNMatching(const FaceFeatures& queryFeatures);
    double calculateDistance(const cv::Mat& desc1, const cv::Mat& desc2);
    double calculateConfidence(const std::vector<cv::DMatch>& matches, double distance);
    
    // 필터링 및 후처리
    std::vector<cv::DMatch> filterMatches(const std::vector<std::vector<cv::DMatch>>& knnMatches);
    bool validateMatch(const MatchResult& result);
    
    // 시각화
    void drawMatchResult(cv::Mat& frame, const MatchResult& result);
    void drawFeaturePoints(cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints);
    
    // 유틸리티
    cv::Mat createFeatureDescriptorMatrix();
    void updatePerformanceStats(double frameTime, double extractionTime, double matchingTime);
    
    // 얼굴 영역 전처리
    cv::Rect expandFaceRect(const cv::Rect& face, const cv::Size& imageSize, double factor = 1.2);
    cv::Mat normalizeDescriptors(const cv::Mat& descriptors);
};

// === 유틸리티 함수들 ===
namespace MLFaceMatchingUtils {
    // 거리 계산
    double euclideanDistance(const cv::Mat& desc1, const cv::Mat& desc2);
    double cosineDistance(const cv::Mat& desc1, const cv::Mat& desc2);
    double hammingDistance(const cv::Mat& desc1, const cv::Mat& desc2);
    
    // 이미지 전처리
    cv::Mat enhanceImage(const cv::Mat& image);
    cv::Mat normalizeIllumination(const cv::Mat& image);
    
    // 파일 관리
    std::vector<std::string> getImageFiles(const std::string& directory);
    bool createDirectory(const std::string& path);
    
    // 성능 측정
    class Timer {
    public:
        Timer();
        void start();
        double getElapsedMs();
        
    private:
        std::chrono::high_resolution_clock::time_point startTime;
    };
}