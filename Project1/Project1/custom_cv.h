#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

namespace custom_cv {
    
    /**
     * Custom implementation of Hough Line Transform
     * Equivalent to cv::HoughLines function
     * 
     * @param image Input edge image (binary image from edge detection)
     * @param lines Output vector of lines in (rho, theta) format
     * @param rho Distance resolution of the accumulator in pixels
     * @param theta Angle resolution of the accumulator in radians
     * @param threshold Accumulator threshold parameter. Only those lines are returned that get enough votes (>threshold)
     */
    void HoughLines(const cv::Mat& image, std::vector<cv::Vec2f>& lines, 
                   double rho, double theta, int threshold);
    
    /**
     * Custom implementation of Harris Corner Detector
     * Equivalent to cv::cornerHarris function
     * 
     * @param src Input image (grayscale)
     * @param dst Output image with corner response values
     * @param blockSize Size of neighborhood considered for corner detection
     * @param ksize Aperture parameter for Sobel derivative
     * @param k Harris detector free parameter
     * @param borderType Border type for convolution
     */
    void cornerHarris(const cv::Mat& src, cv::Mat& dst, int blockSize, 
                     int ksize, double k, int borderType = cv::BORDER_DEFAULT);
    
    /**
     * Helper function to compute Sobel derivatives
     */
    void computeSobelDerivatives(const cv::Mat& src, cv::Mat& Ix, cv::Mat& Iy, int ksize);
    
    /**
     * Helper function to apply Gaussian weighting to derivatives
     */
    void applyGaussianWeighting(cv::Mat& Ixx, cv::Mat& Iyy, cv::Mat& Ixy, int blockSize);
    
    /**
     * Helper function to compute Harris response
     */
    void computeHarrisResponse(const cv::Mat& Ixx, const cv::Mat& Iyy, 
                              const cv::Mat& Ixy, cv::Mat& dst, double k);
}