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
    int numRhos = static_cast<int>(2 * maxDist / rho) + 1; // Number of rho bins (add 1 for safety)
    
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
    
    // Find peaks using improved non-maximum suppression
    std::vector<std::pair<int, std::pair<int, int>>> candidates; // votes, (r, t)
    
    for (int r = 1; r < numRhos - 1; r++) {
        for (int t = 1; t < numAngles - 1; t++) {
            int votes = accumulator.at<int>(r, t);
            
            if (votes >= threshold) {
                // Check if this is a local maximum in 5x5 neighborhood
                bool isLocalMax = true;
                
                for (int dr = -2; dr <= 2 && isLocalMax; dr++) {
                    for (int dt = -2; dt <= 2 && isLocalMax; dt++) {
                        if (dr == 0 && dt == 0) continue;
                        
                        int nr = r + dr;
                        int nt = t + dt;
                        
                        // Handle theta wrapping
                        if (nt < 0) nt = numAngles - 1;
                        if (nt >= numAngles) nt = 0;
                        
                        // Check bounds for rho
                        if (nr >= 0 && nr < numRhos) {
                            if (accumulator.at<int>(nr, nt) > votes) {
                                isLocalMax = false;
                            }
                        }
                    }
                }
                
                if (isLocalMax) {
                    candidates.push_back({votes, {r, t}});
                }
            }
        }
    }
    
    // Sort candidates by votes (descending)
    std::sort(candidates.begin(), candidates.end(), std::greater<std::pair<int, std::pair<int, int>>>());
    
    // Take only the strongest candidates and apply strict filtering
    int maxLines = std::min(50, static_cast<int>(candidates.size())); // Reduced max lines
    
    for (int i = 0; i < maxLines; i++) {
        int r = candidates[i].second.first;
        int t = candidates[i].second.second;
        int votes = candidates[i].first;
        
        // Convert back to rho, theta
        double actualRho = (r * rho) - maxDist;
        double actualTheta = t * theta;
        
        // STRICT FILTERING: Only accept horizontal/vertical lines if needed
        // Check if line is approximately horizontal or vertical
        double theta_deg = actualTheta * 180.0 / CV_PI;
        bool isHorizontalOrVertical = false;
        
        // Check for horizontal lines (around 0° or 180°)
        if (std::abs(theta_deg) < 15 || std::abs(theta_deg - 180) < 15) {
            isHorizontalOrVertical = true;
        }
        // Check for vertical lines (around 90°)
        else if (std::abs(theta_deg - 90) < 15) {
            isHorizontalOrVertical = true;
        }
        
        // Skip diagonal lines if we want to match OpenCV behavior more closely
        // Comment out the next 3 lines if you want to keep diagonal lines
        if (!isHorizontalOrVertical) {
            continue;
        }
        
        // Additional filtering: avoid nearly identical lines
        bool tooSimilar = false;
        for (const auto& existingLine : lines) {
            double rho_diff = std::abs(actualRho - existingLine[0]);
            double theta_diff = std::abs(actualTheta - existingLine[1]);
            
            // Handle theta wrapping for comparison
            if (theta_diff > CV_PI / 2) {
                theta_diff = CV_PI - theta_diff;
            }
            
            if (rho_diff < 15 && theta_diff < 0.15) { // Stricter similarity threshold
                tooSimilar = true;
                break;
            }
        }
        
        if (!tooSimilar) {
            lines.push_back(cv::Vec2f(static_cast<float>(actualRho), 
                                    static_cast<float>(actualTheta)));
        }
        
        // Stop if we have enough lines
        if (lines.size() >= 20) break; // Reduced to 20 lines max
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
        // 5x5 Sobel kernels - corrected
        sobelX = (cv::Mat_<float>(5, 5) << 
            -1, -2, 0, 2, 1,
            -4, -8, 0, 8, 4,
            -6, -12, 0, 12, 6,
            -4, -8, 0, 8, 4,
            -1, -2, 0, 2, 1) / 48.0;
        
        sobelY = (cv::Mat_<float>(5, 5) << 
            -1, -4, -6, -4, -1,
            -2, -8, -12, -8, -2,
            0, 0, 0, 0, 0,
            2, 8, 12, 8, 2,
            1, 4, 6, 4, 1) / 48.0;
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
        src.convertTo(srcFloat, CV_32F, 1.0/255.0); // Normalize to [0,1]
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
    
    // Step 5: Apply strict corner filtering
    // Suppress weak responses that might come from curves
    double minVal, maxVal;
    cv::minMaxLoc(dst, &minVal, &maxVal);
    
    // Apply much stricter thresholding to eliminate curve responses
    double strictThreshold = maxVal * 0.1; // Only keep top 10% responses
    cv::Mat mask = dst > strictThreshold;
    dst.setTo(0, ~mask);
    
    // Additional morphological filtering to remove isolated points
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat filtered;
    cv::morphologyEx(dst, filtered, cv::MORPH_OPEN, kernel);
    
    // Only keep responses that survive morphological filtering
    cv::Mat finalMask = filtered > 0;
    dst.setTo(0, ~finalMask);
    
    // Renormalize after filtering
    cv::minMaxLoc(dst, &minVal, &maxVal);
    if (maxVal > 0) {
        dst = dst / maxVal;
    }
    
    std::cout << "Harris corner detection completed. Filtered response range: [" 
              << "min=" << minVal << ", max=" << maxVal << "]" << std::endl;
}

}