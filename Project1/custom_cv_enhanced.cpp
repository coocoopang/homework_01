#include "custom_cv_enhanced.h"
#include <algorithm>
#include <iostream>

namespace custom_cv_enhanced {

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

double computeAdaptiveThreshold(const cv::Mat& responseMap, double percentile) {
    std::vector<float> values;
    
    // Collect all non-zero response values
    for (int y = 0; y < responseMap.rows; y++) {
        for (int x = 0; x < responseMap.cols; x++) {
            float val = responseMap.at<float>(y, x);
            if (val > 0) {
                values.push_back(val);
            }
        }
    }
    
    if (values.empty()) return 0.0;
    
    // Sort and find percentile
    std::sort(values.begin(), values.end());
    int index = static_cast<int>(values.size() * percentile);
    index = std::min(index, static_cast<int>(values.size() - 1));
    
    return values[index];
}

std::vector<cv::Point> FindLocalExtrema_Enhanced(cv::Mat& src, double minThreshold, 
                                                int kernelSize, bool useAdaptive) {
    std::vector<cv::Point> points;
    
    // Strategy 1: Standard morphological approach
    cv::Mat dilatedImg, localMaxImg;
    cv::Size sz(kernelSize, kernelSize);
    cv::Mat rectKernel = cv::getStructuringElement(cv::MORPH_RECT, sz);

    cv::dilate(src, dilatedImg, rectKernel);
    localMaxImg = (src == dilatedImg);

    cv::Mat erodedImg, localMinImg;
    cv::erode(src, erodedImg, rectKernel);
    localMinImg = (src > erodedImg);

    cv::Mat localExtremaImg = (localMaxImg & localMinImg);
    
    // Adaptive threshold if requested
    double adaptiveThresh = minThreshold;
    if (useAdaptive) {
        adaptiveThresh = computeAdaptiveThreshold(src, 0.90); // Top 10%
        adaptiveThresh = std::max(adaptiveThresh, minThreshold);
        std::cout << "Using adaptive threshold: " << adaptiveThresh << std::endl;
    }

    // Strategy 2: Add peak-based detection for rotated shapes
    cv::Mat gaussianBlurred;
    cv::GaussianBlur(src, gaussianBlurred, cv::Size(3, 3), 1.0);
    
    for (int y = kernelSize; y < src.rows - kernelSize; ++y) {
        for (int x = kernelSize; x < src.cols - kernelSize; ++x) {
            float centerVal = src.at<float>(y, x);
            
            // Check both morphological and threshold conditions
            bool isMorphological = localExtremaImg.at<uchar>(y, x) > 0;
            bool isAboveThreshold = centerVal >= adaptiveThresh;
            
            // Additional check: is it significantly higher than neighborhood average?
            bool isPeak = false;
            if (isAboveThreshold) {
                float sum = 0;
                int count = 0;
                
                // Check 5x5 neighborhood
                for (int dy = -2; dy <= 2; dy++) {
                    for (int dx = -2; dx <= 2; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        sum += src.at<float>(y + dy, x + dx);
                        count++;
                    }
                }
                
                float avgNeighbor = sum / count;
                isPeak = centerVal > avgNeighbor * 1.5; // 50% higher than average
            }
            
            // Accept if either morphological OR peak detection finds a corner
            if ((isMorphological && isAboveThreshold) || isPeak) {
                points.push_back(cv::Point(x, y));
            }
        }
    }
    
    // Remove duplicates (points too close to each other)
    std::vector<cv::Point> filtered_points;
    const int min_distance = 3;
    
    for (const auto& point : points) {
        bool tooClose = false;
        for (const auto& existing : filtered_points) {
            double dist = cv::norm(point - existing);
            if (dist < min_distance) {
                tooClose = true;
                break;
            }
        }
        if (!tooClose) {
            filtered_points.push_back(point);
        }
    }
    
    return filtered_points;
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
    
    // Step 5: Enhanced filtering that preserves rotated corners
    double minVal, maxVal;
    cv::minMaxLoc(dst, &minVal, &maxVal);
    
    // More gentle filtering - keep more potential corners
    double gentleThreshold = maxVal * 0.05; // Keep top 5% instead of 10%
    
    // Apply threshold but don't zero out everything
    cv::Mat mask = dst > gentleThreshold;
    cv::Mat filtered = dst.clone();
    filtered.setTo(cv::Scalar(0.0), ~mask);
    
    // Much lighter morphological filtering - use smaller kernel
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::Mat morphFiltered;
    cv::morphologyEx(filtered, morphFiltered, cv::MORPH_OPEN, kernel);
    
    // Combine original strong responses with morphologically filtered ones
    cv::Mat strongMask = dst > maxVal * 0.15; // Very strong corners
    dst = morphFiltered.clone();
    filtered.copyTo(dst, strongMask); // Preserve strong corners even if morphology removes them
    
    // Final normalization
    cv::minMaxLoc(dst, &minVal, &maxVal);
    if (maxVal > 0) {
        dst = dst / maxVal;
    }
    
    std::cout << "Enhanced Harris corner detection completed. Range: [" 
              << "min=" << minVal << ", max=" << maxVal << "]" << std::endl;
}

void HoughLines(const cv::Mat& image, std::vector<cv::Vec2f>& lines, 
               double rho, double theta, int threshold) {
    // Use the same implementation as the original custom_cv
    // This is already working well, so we don't need to change it
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
    int numAngles = static_cast<int>(CV_PI / theta);
    int numRhos = static_cast<int>(2 * maxDist / rho) + 1;
    
    // Create accumulator array
    cv::Mat accumulator = cv::Mat::zeros(numRhos, numAngles, CV_32SC1);
    
    // Fill accumulator (same as original)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (image.at<uchar>(y, x) > 0) {
                for (int t = 0; t < numAngles; t++) {
                    double angle = t * theta;
                    double r = x * cos(angle) + y * sin(angle);
                    int rhoIdx = static_cast<int>(round((r + maxDist) / rho));
                    
                    if (rhoIdx >= 0 && rhoIdx < numRhos) {
                        accumulator.at<int>(rhoIdx, t)++;
                    }
                }
            }
        }
    }
    
    // Find peaks (same logic as original)
    std::vector<std::pair<int, std::pair<int, int>>> candidates;
    
    for (int r = 1; r < numRhos - 1; r++) {
        for (int t = 1; t < numAngles - 1; t++) {
            int votes = accumulator.at<int>(r, t);
            
            if (votes >= threshold) {
                bool isLocalMax = true;
                
                for (int dr = -2; dr <= 2 && isLocalMax; dr++) {
                    for (int dt = -2; dt <= 2 && isLocalMax; dt++) {
                        if (dr == 0 && dt == 0) continue;
                        
                        int nr = r + dr;
                        int nt = t + dt;
                        
                        if (nt < 0) nt = numAngles - 1;
                        if (nt >= numAngles) nt = 0;
                        
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
    
    std::sort(candidates.begin(), candidates.end(), 
              std::greater<std::pair<int, std::pair<int, int>>>());
    
    int maxLines = std::min(50, static_cast<int>(candidates.size()));
    
    for (int i = 0; i < maxLines; i++) {
        int r = candidates[i].second.first;
        int t = candidates[i].second.second;
        
        double actualRho = (r * rho) - maxDist;
        double actualTheta = t * theta;
        
        double theta_deg = actualTheta * 180.0 / CV_PI;
        bool isHorizontalOrVertical = false;
        
        if (std::abs(theta_deg) < 15 || std::abs(theta_deg - 180) < 15) {
            isHorizontalOrVertical = true;
        }
        else if (std::abs(theta_deg - 90) < 15) {
            isHorizontalOrVertical = true;
        }
        
        if (!isHorizontalOrVertical) {
            continue;
        }
        
        bool tooSimilar = false;
        for (const auto& existingLine : lines) {
            double rho_diff = std::abs(actualRho - existingLine[0]);
            double theta_diff = std::abs(actualTheta - existingLine[1]);
            
            if (theta_diff > CV_PI / 2) {
                theta_diff = CV_PI - theta_diff;
            }
            
            if (rho_diff < 15 && theta_diff < 0.15) {
                tooSimilar = true;
                break;
            }
        }
        
        if (!tooSimilar) {
            lines.push_back(cv::Vec2f(static_cast<float>(actualRho), 
                                    static_cast<float>(actualTheta)));
        }
        
        if (lines.size() >= 20) break;
    }
    
    std::cout << "Found " << lines.size() << " lines with threshold " << threshold << std::endl;
}

void showHarrisResponse(const cv::Mat& response, const std::string& windowName) {
    try {
        cv::Mat display;
        cv::normalize(response, display, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imshow(windowName, display);
    } catch (const cv::Exception& e) {
        std::cout << "Display not available for " << windowName << std::endl;
    }
}

void analyzeCornerDistribution(const std::vector<cv::Point>& corners, const cv::Mat& image) {
    std::cout << "Corner Analysis:" << std::endl;
    std::cout << "Total corners found: " << corners.size() << std::endl;
    
    if (corners.empty()) return;
    
    // Analyze distribution
    int left = 0, right = 0, top = 0, bottom = 0, center = 0;
    int midX = image.cols / 2;
    int midY = image.rows / 2;
    
    for (const auto& corner : corners) {
        if (corner.x < midX / 2) left++;
        else if (corner.x > image.cols - midX / 2) right++;
        
        if (corner.y < midY / 2) top++;
        else if (corner.y > image.rows - midY / 2) bottom++;
        
        if (corner.x > midX / 2 && corner.x < image.cols - midX / 2 &&
            corner.y > midY / 2 && corner.y < image.rows - midY / 2) {
            center++;
        }
    }
    
    std::cout << "Distribution: Left=" << left << ", Right=" << right 
              << ", Top=" << top << ", Bottom=" << bottom << ", Center=" << center << std::endl;
}

}