#include "custom_cv.h"
#include <algorithm>
#include <iostream>

namespace custom_cv {

void HoughLines(const cv::Mat& image, std::vector<cv::Vec2f>& lines, 
               double rho, double theta, int threshold) {
    lines.clear();
    
    if (image.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return;
    }
    
    // Image dimensions
    int width = image.cols;
    int height = image.rows;
    
    // Calculate the maximum possible distance (diagonal of image)
    double maxDist = sqrt(width * width + height * height);
    
    // Accumulator dimensions
    int numAngles = static_cast<int>(CV_PI / theta);  // Number of angle bins
    int numRhos = static_cast<int>(2 * maxDist / rho); // Number of rho bins
    
    // Create accumulator array
    cv::Mat accumulator = cv::Mat::zeros(numRhos, numAngles, CV_32SC1);
    
    // Fill accumulator
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Check if pixel is an edge pixel
            if (image.at<uchar>(y, x) > 0) {
                // For each angle
                for (int t = 0; t < numAngles; t++) {
                    double angle = t * theta;
                    
                    // Calculate rho for this angle and point
                    double r = x * cos(angle) + y * sin(angle);
                    
                    // Convert rho to accumulator index
                    int rhoIdx = static_cast<int>(round((r + maxDist) / rho));
                    
                    // Check bounds
                    if (rhoIdx >= 0 && rhoIdx < numRhos) {
                        accumulator.at<int>(rhoIdx, t)++;
                    }
                }
            }
        }
    }
    
    // Find peaks in accumulator (non-maximum suppression)
    for (int r = 0; r < numRhos; r++) {
        for (int t = 0; t < numAngles; t++) {
            int votes = accumulator.at<int>(r, t);
            
            if (votes >= threshold) {
                // Check if this is a local maximum
                bool isLocalMax = true;
                
                // Check 3x3 neighborhood
                for (int dr = -1; dr <= 1 && isLocalMax; dr++) {
                    for (int dt = -1; dt <= 1 && isLocalMax; dt++) {
                        int nr = r + dr;
                        int nt = t + dt;
                        
                        // Skip center pixel and handle boundary
                        if ((dr == 0 && dt == 0) || nr < 0 || nr >= numRhos || 
                            nt < 0 || nt >= numAngles) {
                            continue;
                        }
                        
                        if (accumulator.at<int>(nr, nt) > votes) {
                            isLocalMax = false;
                        }
                    }
                }
                
                if (isLocalMax) {
                    // Convert back to rho, theta
                    double actualRho = (r * rho) - maxDist;
                    double actualTheta = t * theta;
                    
                    lines.push_back(cv::Vec2f(static_cast<float>(actualRho), 
                                            static_cast<float>(actualTheta)));
                }
            }
        }
    }
    
    std::cout << "Found " << lines.size() << " lines with threshold " << threshold << std::endl;
}

void computeSobelDerivatives(const cv::Mat& src, cv::Mat& Ix, cv::Mat& Iy, int ksize) {
    // Create Sobel kernels
    cv::Mat sobelX, sobelY;
    
    if (ksize == 3) {
        // 3x3 Sobel kernels
        sobelX = (cv::Mat_<float>(3, 3) << 
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1);
        
        sobelY = (cv::Mat_<float>(3, 3) << 
            -1, -2, -1,
             0,  0,  0,
             1,  2,  1);
    } else if (ksize == 5) {
        // 5x5 Sobel kernels
        sobelX = (cv::Mat_<float>(5, 5) << 
            -1, -2, 0, 2, 1,
            -4, -8, 0, 8, 4,
            -6, -12, 0, 12, 6,
            -4, -8, 0, 8, 4,
            -1, -2, 0, 2, 1) / 48.0;
        
        sobelY = sobelX.t(); // Transpose for Y direction
    } else {
        // Default to 3x3
        ksize = 3;
        sobelX = (cv::Mat_<float>(3, 3) << 
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1);
        
        sobelY = (cv::Mat_<float>(3, 3) << 
            -1, -2, -1,
             0,  0,  0,
             1,  2,  1);
    }
    
    // Apply convolution
    cv::filter2D(src, Ix, CV_32F, sobelX);
    cv::filter2D(src, Iy, CV_32F, sobelY);
}

void applyGaussianWeighting(cv::Mat& Ixx, cv::Mat& Iyy, cv::Mat& Ixy, int blockSize) {
    // Create Gaussian kernel
    cv::Mat gaussian = cv::getGaussianKernel(blockSize, -1, CV_32F);
    cv::Mat gaussianKernel = gaussian * gaussian.t();
    
    // Apply Gaussian weighting to each component
    cv::filter2D(Ixx, Ixx, CV_32F, gaussianKernel);
    cv::filter2D(Iyy, Iyy, CV_32F, gaussianKernel);
    cv::filter2D(Ixy, Ixy, CV_32F, gaussianKernel);
}

void computeHarrisResponse(const cv::Mat& Ixx, const cv::Mat& Iyy, 
                          const cv::Mat& Ixy, cv::Mat& dst, double k) {
    dst = cv::Mat::zeros(Ixx.size(), CV_32F);
    
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            // Get matrix elements
            float xx = Ixx.at<float>(y, x);
            float yy = Iyy.at<float>(y, x);
            float xy = Ixy.at<float>(y, x);
            
            // Compute determinant and trace
            float det = xx * yy - xy * xy;
            float trace = xx + yy;
            
            // Harris response: det - k * trace^2
            float response = det - k * trace * trace;
            
            dst.at<float>(y, x) = response;
        }
    }
}

void cornerHarris(const cv::Mat& src, cv::Mat& dst, int blockSize, 
                 int ksize, double k, int borderType) {
    if (src.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return;
    }
    
    // Convert to float if necessary
    cv::Mat srcFloat;
    if (src.type() != CV_32F) {
        src.convertTo(srcFloat, CV_32F);
    } else {
        srcFloat = src.clone();
    }
    
    // Step 1: Compute image derivatives using Sobel
    cv::Mat Ix, Iy;
    computeSobelDerivatives(srcFloat, Ix, Iy, ksize);
    
    // Step 2: Compute products of derivatives
    cv::Mat Ixx = Ix.mul(Ix);  // Ix^2
    cv::Mat Iyy = Iy.mul(Iy);  // Iy^2
    cv::Mat Ixy = Ix.mul(Iy);  // Ix * Iy
    
    // Step 3: Apply Gaussian weighting (windowing function)
    applyGaussianWeighting(Ixx, Iyy, Ixy, blockSize);
    
    // Step 4: Compute Harris response for each pixel
    computeHarrisResponse(Ixx, Iyy, Ixy, dst, k);
    
    std::cout << "Harris corner detection completed. Response range: [" 
              << "min=" << *std::min_element(dst.begin<float>(), dst.end<float>()) 
              << ", max=" << *std::max_element(dst.begin<float>(), dst.end<float>()) 
              << "]" << std::endl;
}

}