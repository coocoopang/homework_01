#include <iostream>
#include "opencv2/opencv.hpp"

int run_HoughLines()
{
	cv::Mat src = cv::imread("D:/images/lg_building.jpg", cv::IMREAD_GRAYSCALE);
	if (src.empty()) {
		std::cerr << "이미지를 불러올 수 없습니다." << std::endl;
		return -1;
	}

	cv::Mat src_out;
	cvtColor(src, src_out, cv::COLOR_GRAY2BGR);

	cv::Mat src_edge;
	cv::Canny(src, src_edge, 170, 200);

	std::vector<cv::Vec2f> lines;
	cv::HoughLines(src_edge, lines, 1, CV_PI / 180, 400);

	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		cv::Point pt1, pt2;
		double x0 = rho * cos(theta), y0 = rho * sin(theta);
		pt1.x = round(x0 + 1000 * (-sin(theta)));
		pt1.y = round(y0 + 1000 * (cos(theta)));
		pt2.x = round(x0 - 1000 * (-sin(theta)));
		pt2.y = round(y0 - 1000 * (cos(theta)));

		cv::line(src_out, pt1, pt2, cv::Scalar(0, 0, 255), 2, 8);
	}

	cv::imshow("Original Image", src);
	cv::imshow("Edge Image", src_edge);
	cv::imshow("Line Image", src_out);
	cv::waitKey(0); // 키 입력을 기다립니다.
	cv::destroyAllWindows(); // 모든 창을 닫습니다.
	return 0;
}

std::vector<cv::Point> FindLocalExtrema(cv::Mat& src)
{
	cv::Mat dilatedImg, localMaxImg;
	cv::Size sz(7, 7);
	cv::Mat rectKernel = cv::getStructuringElement(cv::MORPH_RECT, sz);

	cv::dilate(src, dilatedImg, rectKernel);
	localMaxImg = (src == dilatedImg);

	cv::Mat erodedImg, localMinImg;
	cv::erode(src, erodedImg, rectKernel);
	localMinImg = (src > erodedImg);

	cv::Mat localExtremaImg;
	localExtremaImg = (localMaxImg & localMinImg);

	std::vector<cv::Point> points;

	for (int y = 0; y < localExtremaImg.rows; ++y) {
		for (int x = 0; x < localExtremaImg.cols; ++x) {
			uchar val = localExtremaImg.at<uchar>(y, x);
			if (val)  points.push_back(cv::Point(x, y));
		}
	}
	return points;
}


int run_HarrisCornerDetector()
{
	cv::Mat src = cv::imread("D:/images/shapes1.jpg", cv::IMREAD_GRAYSCALE);
	if (src.empty()) {
		std::cerr << "이미지를 불러올 수 없습니다." << std::endl;
		return -1;
	}

	int blockSize = 5;
	int kSize = 3;
	double k = 0.01;

	cv::Mat R;
	cv::cornerHarris(src, R, blockSize, kSize, k);
	cv::threshold(R, R, 0.02, 0, cv::THRESH_TOZERO);

	std::vector<cv::Point> cornerPoints = FindLocalExtrema(R);

	cv::Mat dst(src.size(), CV_8UC3);
	cvtColor(src, dst, cv::COLOR_GRAY2BGR);

	for (const auto& c : cornerPoints) {
		cv::circle(dst, c, 5, cv::Scalar(0, 0, 255), 2);
	}

	cv::imshow("Original Image", src);
	cv::imshow("Result Image", dst);
	cv::waitKey(0); // 키 입력을 기다립니다.
	cv::destroyAllWindows(); // 모든 창을 닫습니다.

	return 0;
}

int main()
{
	run_HoughLines();
	run_HarrisCornerDetector();

	return 0;
}
