#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include <string>
#include <iostream>

/**
 * 실시간 얼굴 매칭 시스템
 * - 웹캠 또는 MP4 비디오 파일 입력
 * - 기준 얼굴 이미지와 매칭 검사
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
    
    // 얼굴 검출 스케일 설정
    void setDetectionScale(double scale) { detectionScale = scale; }

private:
    cv::CascadeClassifier faceClassifier;    // 얼굴 검출기
    cv::Mat referenceFace;                   // 기준 얼굴 이미지
    cv::VideoCapture videoCapture;           // 웹캠 또는 비디오 파일
    std::string videoSource;                 // 비디오 소스 ("webcam" 또는 파일 경로)
    bool isVideoFile;                        // 비디오 파일 여부
    
    double matchThreshold;                   // 매칭 임계값
    double detectionScale;                   // 검출 스케일 팩터
    bool cascadeLoaded;                      // cascade 로드 상태
    
    // 얼굴 검출
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame);
    
    // 얼굴 매칭 (템플릿 매칭 + 히스토그램 비교)
    double matchFace(const cv::Mat& face1, const cv::Mat& face2);
    
    // 템플릿 매칭 점수 계산
    double calculateTemplateMatchScore(const cv::Mat& face1, const cv::Mat& face2);
    
    // 히스토그램 유사도 계산
    double calculateHistogramSimilarity(const cv::Mat& face1, const cv::Mat& face2);
    
    // 얼굴 이미지 전처리
    cv::Mat preprocessFace(const cv::Mat& face);
    
    // 매칭 결과를 화면에 표시
    void drawMatchResult(cv::Mat& frame, const cv::Rect& faceRect, double matchScore, bool isMatch);
};

// 유틸리티 함수들
namespace FaceMatchingUtils {
    // 이미지 크기 조정
    cv::Mat resizeImage(const cv::Mat& image, int targetWidth = 200);
    
    // 얼굴 영역 확장 (더 나은 매칭을 위해)
    cv::Rect expandFaceRect(const cv::Rect& face, const cv::Size& imageSize, double factor = 1.2);
    
    // 매칭 점수를 퍼센트로 변환
    double scoreToPercent(double score);
}