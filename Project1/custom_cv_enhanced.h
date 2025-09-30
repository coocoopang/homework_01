#pragma once
#include "opencv2/opencv.hpp"

namespace custom_cv_enhanced {

// Enhanced Harris corner detector with adaptive thresholding
void cornerHarris(const cv::Mat& src, cv::Mat& dst, int blockSize, 
                 int ksize, double k, int borderType = cv::BORDER_DEFAULT);

// Enhanced Hough Lines (same as original)
void HoughLines(const cv::Mat& image, std::vector<cv::Vec2f>& lines, 
               double rho, double theta, int threshold);

// Helper functions with enhanced features
void computeSobelDerivatives(const cv::Mat& src, cv::Mat& Ix, cv::Mat& Iy, int ksize);
void applyGaussianWeighting(cv::Mat& Ixx, cv::Mat& Iyy, cv::Mat& Ixy, int blockSize);
void computeHarrisResponse(const cv::Mat& Ixx, const cv::Mat& Iyy, 
                          const cv::Mat& Ixy, cv::Mat& dst, double k);

// Enhanced local extrema finder with multiple strategies
std::vector<cv::Point> FindLocalExtrema_Enhanced(cv::Mat& src, double minThreshold = 0.01, 
                                               int kernelSize = 5, bool useAdaptive = true);

// Adaptive thresholding for different image types
double computeAdaptiveThreshold(const cv::Mat& responseMap, double percentile = 0.95);

// Debug visualization helpers
void showHarrisResponse(const cv::Mat& response, const std::string& windowName);
void analyzeCornerDistribution(const std::vector<cv::Point>& corners, const cv::Mat& image);

}