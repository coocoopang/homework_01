#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <vector>
#include <string>
#include <iostream>

/**
 * 최근접 특징 매칭 기반 얼굴 인식 시스템
 * - 웹캠 또는 MP4 비디오 파일 입력
 * - SIFT/ORB 특징점 추출 및 최근접 매칭
 * - 매칭되는 얼굴에 빨간색 테두리 표시
 */
class FaceMatcher {
public:
    FaceMatcher();
    ~FaceMatcher();
    
    // 기준 얼굴 이미지 로드
    bool loadReferenceFace(const std::string& imagePath);
    
    // 웹캠 시작
    bool startWebcam(int deviceId = 0);
    
    // MP4 비디오 파일 로드
    bool loadVideoFile(const std::string& videoPath);
    
    // 실시간 얼굴 매칭 실행 (웹캠 또는 비디오 파일)
    void runFaceMatching();
    
    // 비디오 파일로 얼굴 매칭 실행
    void runVideoFaceMatching();
    
    // 매칭 임계값 설정
    void setMatchThreshold(double threshold) { matchThreshold = threshold; }
    
    // 특징점 검출기 타입 설정
    void setFeatureDetectorType(const std::string& type) { detectorType = type; }
    
    // 매칭 알고리즘 타입 설정  
    void setMatcherType(const std::string& type) { matcherType = type; }

private:
    // 얼굴 검출 관련
    cv::CascadeClassifier faceClassifier;    // 얼굴 검출기
    cv::Mat referenceFaceImage;              // 기준 얼굴 원본 이미지
    cv::VideoCapture videoCapture;           // 웹캠 또는 비디오 파일
    std::string videoSource;                 // 비디오 소스 ("webcam" 또는 파일 경로)
    bool isVideoFile;                        // 비디오 파일 여부
    bool cascadeLoaded;                      // cascade 로드 상태
    
    // 특징점 매칭 관련
    cv::Ptr<cv::Feature2D> detector;         // 특징점 검출기 (SIFT/ORB)
    cv::Ptr<cv::DescriptorMatcher> matcher;  // 특징점 매처 (BF/FLANN)
    std::vector<cv::KeyPoint> referenceKeypoints;    // 기준 얼굴의 특징점들
    cv::Mat referenceDescriptors;            // 기준 얼굴의 특징 디스크립터들
    
    // 매칭 파라미터
    double matchThreshold;                   // 매칭 임계값 (좋은 매칭의 최소 개수)
    std::string detectorType;                // 특징점 검출기 타입 ("SIFT", "ORB")
    std::string matcherType;                 // 매처 타입 ("BF", "FLANN")
    int minMatchCount;                       // 최소 매칭 개수
    double maxDistanceRatio;                 // Lowe's ratio test 임계값
    
    // 얼굴 검출
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame);
    
    // 특징점 기반 얼굴 매칭
    double matchFaceByFeatures(const cv::Mat& detectedFace);
    
    // 특징점 추출
    void extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    
    // 최근접 특징 매칭
    std::vector<cv::DMatch> findNearestMatches(const cv::Mat& descriptors1, const cv::Mat& descriptors2);
    
    // 좋은 매칭 필터링 (Lowe's ratio test)
    std::vector<cv::DMatch> filterGoodMatches(const std::vector<std::vector<cv::DMatch>>& knnMatches);
    
    // 기하학적 검증 (RANSAC)
    int verifyGeometry(const std::vector<cv::KeyPoint>& keypoints1, 
                      const std::vector<cv::KeyPoint>& keypoints2,
                      const std::vector<cv::DMatch>& matches);
    
    // 매칭 점수 계산
    double calculateMatchScore(int goodMatches, int totalKeypoints);
    
    // 얼굴 이미지 전처리
    cv::Mat preprocessFace(const cv::Mat& face);
    
    // 매칭 결과를 화면에 표시
    void drawMatchResult(cv::Mat& frame, const cv::Rect& faceRect, double matchScore, bool isMatch, int matchCount);
    
    // 특징점 시각화
    void drawFeatureMatches(const cv::Mat& detectedFace, const std::vector<cv::KeyPoint>& keypoints, 
                           const std::vector<cv::DMatch>& goodMatches);
    
    // 디버그 정보 출력
    void printMatchingInfo(int totalMatches, int goodMatches, double score);
};

// 특징 매칭 유틸리티 함수들
namespace FeatureMatchingUtils {
    // 특징점 검출기 생성
    cv::Ptr<cv::Feature2D> createFeatureDetector(const std::string& type);
    
    // 디스크립터 매처 생성
    cv::Ptr<cv::DescriptorMatcher> createDescriptorMatcher(const std::string& type, const std::string& detectorType);
    
    // 이미지 크기 조정 (종횡비 유지)
    cv::Mat resizeImage(const cv::Mat& image, int targetWidth = 200);
    
    // 얼굴 영역 확장 (더 나은 특징점 추출을 위해)
    cv::Rect expandFaceRect(const cv::Rect& face, const cv::Size& imageSize, double factor = 1.2);
    
    // 매칭 점수를 퍼센트로 변환
    double scoreToPercent(double score);
    
    // 특징점 밀도 계산
    double calculateFeatureDensity(const std::vector<cv::KeyPoint>& keypoints, const cv::Size& imageSize);
}