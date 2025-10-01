#include "main.h"

/**
 * @brief Harris 응답 맵에서 비최대 억제(NMS)와 임계값을 적용하여 최종 코너점을 추출합니다.
 * @param harrisResponse 각 픽셀의 코너 응답(R) 값이 저장된 2D 벡터
 * @param threshold 코너로 판단할 임계값 (최대 응답 값에 대한 비율, 예: 0.01은 상위 1%)
 * @return 검출된 코너(Corner)들의 벡터
 */
vecCorner GetCorners(const vecVecDouble& harrisResponse, double threshold)
{
	vecCorner corners;
	int height = harrisResponse.size();
	int width = harrisResponse[0].size();

	// 1. R 값의 최댓값을 찾아 임계값을 절대값으로 변환합니다.
	//    이렇게 하면 이미지의 전반적인 밝기나 대비에 상관없이 일관된 검출이 가능합니다.
	double maxResponse = 0;
	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			if (harrisResponse[y][x] > maxResponse) {
				maxResponse = harrisResponse[y][x];
			}
		}
	}
	double actualThreshold = maxResponse * threshold; // 실제 적용할 임계값

	// 2. 모든 픽셀을 순회하며 코너점을 찾습니다.
	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			double r = harrisResponse[y][x];
			// 현재 픽셀의 R 값이 계산된 임계값보다 클 경우에만 코너 후보로 간주
			if (r > actualThreshold) {
				// 3. 비최대 억제(Non-Maximum Suppression): 주변 3x3 영역에서 최댓값인지 확인
				bool isMax = true;
				for (int i = -1; i <= 1; ++i) {
					for (int j = -1; j <= 1; ++j) {
						// 이웃 픽셀의 R 값이 현재 픽셀보다 크면, 현재 픽셀은 진짜 코너가 아님
						if (harrisResponse[y + i][x + j] > r) {
							isMax = false;
							break;
						}
					}
					if (!isMax) break;
				}
				// 지역 최댓값으로 판명되면 최종 코너 목록에 추가
				if (isMax) {
					corners.push_back({ x, y, r });
				}
			}
		}
	}
	return corners;
}

/**
 * @brief 검출된 코너들을 원본 비트맵 위에 시각적으로 표시합니다.
 * @param bmp 원본 이미지에 대한 비트맵 포인터
 * @param corners 그릴 코너들의 정보를 담은 벡터
 */
void DrawCorners(Bitmap* bmp, const vecCorner& corners)
{
	Graphics graphics(bmp); // GDI+ 그리기 객체 생성
	SolidBrush brush(Color(255, 255, 0, 0)); // 채우기용 빨간색 브러시 생성

	for (const auto& corner : corners) {
		// 각 코너의 (x, y) 위치에 작은 원(타원)을 그려 표시
		graphics.FillEllipse(&brush, corner.x - 6, corner.y - 6, 12, 12);
	}
}

/**
 * @brief GDI+ Bitmap을 2D 흑백 이미지(double 타입 벡터)로 변환합니다.
 * @param bmp 원본 이미지에 대한 비트맵 포인터
 * @return 0.0 ~ 255.0 범위의 밝기 값을 가지는 2D double 벡터
 */
vecVecDouble ConvertToGrayscale(Bitmap* bmp)
{
	UINT width = bmp->GetWidth();
	UINT height = bmp->GetHeight();
	vecVecDouble grayImage(height, vecDouble(width));

	BitmapData bitmapData;
	Rect rect(0, 0, width, height);
	// LockBits: 픽셀 데이터에 직접 접근하여 처리 속도를 높이는 GDI+의 표준적인 방법
	bmp->LockBits(&rect, ImageLockModeRead, PixelFormat24bppRGB, &bitmapData);

	BYTE* p = (BYTE*)bitmapData.Scan0; // 픽셀 데이터의 시작 메모리 주소
	int stride = bitmapData.Stride;    // 이미지 한 줄(row)의 실제 메모리 크기 (byte)

	for (UINT y = 0; y < height; ++y) {
		for (UINT x = 0; x < width; ++x) {
			// 픽셀의 주소: 시작주소 + y * stride + x * 3 (픽셀당 3바이트)
			// R, G, B 값의 평균을 내어 간단하게 흑백 값으로 변환
			grayImage[y][x] = (double)(p[y * stride + x * 3] + p[y * stride + x * 3 + 1] + p[y * stride + x * 3 + 2]) / 3.0;
		}
	}

	bmp->UnlockBits(&bitmapData); // 메모리 잠금 해제
	return grayImage;
}

/**
 * @brief Sobel 연산자를 이용하여 이미지의 x, y 방향 그래디언트(밝기 변화율)를 계산합니다.
 * @param grayImage 흑백 이미지 데이터
 * @param gradX [출력] x방향 그래디언트가 저장될 2D 벡터
 * @param gradY [출력] y방향 그래디언트가 저장될 2D 벡터
 */
void ComputeGradients(const vecVecDouble& grayImage,
	vecVecDouble& gradX,
	vecVecDouble& gradY)
{
	int height = grayImage.size();
	int width = grayImage[0].size();

	// Sobel 커널(마스크): 이미지의 미분(엣지)을 근사하는 데 사용되는 행렬
	double sobelX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} }; // x축(세로선) 엣지 검출
	double sobelY[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} }; // y축(가로선) 엣지 검출

	// 이미지 경계(가장자리 1픽셀)를 제외하고 컨볼루션(convolution) 수행
	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			double gx = 0, gy = 0;
			// 현재 픽셀 (x,y)를 중심으로 3x3 윈도우에 Sobel 커널을 적용
			for (int i = -1; i <= 1; ++i) {
				for (int j = -1; j <= 1; ++j) {
					gx += grayImage[y + i][x + j] * sobelX[i + 1][j + 1];
					gy += grayImage[y + i][x + j] * sobelY[i + 1][j + 1];
				}
			}
			gradX[y][x] = gx;
			gradY[y][x] = gy;
		}
	}
}

/**
 * @brief 계산된 그래디언트를 이용하여 Harris 코너 응답(R score)을 계산합니다.
 * @param gradX x방향 그래디언트 맵
 * @param gradY y방향 그래디언트 맵
 * @param windowSize 코너 응답 계산 시 주변 픽셀을 고려할 윈도우 크기 (보통 3 또는 5)
 * @param k Harris 코너 검출기의 경험적 상수 (보통 0.04 ~ 0.06)
 * @return 각 픽셀의 코너 응답(R) 값이 담긴 2D 벡터
 */
vecVecDouble ComputeHarrisResponse(
	const vecVecDouble& gradX,
	const vecVecDouble& gradY,
	int windowSize, double k)
{
	int height = gradX.size();
	int width = gradX[0].size();
	vecVecDouble harrisResponse(height, vecDouble(width, 0.0));

	// 최적화를 위해 Ix*Ix, Iy*Iy, Ix*Iy 값을 미리 계산하여 저장
	vecVecDouble Ixx(height, vecDouble(width));
	vecVecDouble Iyy(height, vecDouble(width));
	vecVecDouble Ixy(height, vecDouble(width));
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			Ixx[y][x] = gradX[y][x] * gradX[y][x];
			Iyy[y][x] = gradY[y][x] * gradY[y][x];
			Ixy[y][x] = gradX[y][x] * gradY[y][x];
		}
	}

	int offset = windowSize / 2;
	for (int y = offset; y < height - offset; ++y) {
		for (int x = offset; x < width - offset; ++x) {
			double Sxx = 0, Syy = 0, Sxy = 0;
			// 지정된 윈도우 내의 그래디언트 값들을 모두 합산 (가우시안 필터링과 유사한 효과)
			for (int i = -offset; i <= offset; ++i) {
				for (int j = -offset; j <= offset; ++j) {
					Sxx += Ixx[y + i][x + j];
					Syy += Iyy[y + i][x + j];
					Sxy += Ixy[y + i][x + j];
				}
			}

			// 그래디언트 행렬 M의 determinant와 trace를 계산
			// M = [ Sxx Sxy ]
			//     [ Sxy Syy ]
			double det = Sxx * Syy - Sxy * Sxy;
			double trace = Sxx + Syy;
			// Harris 코너 응답 R 값 계산 공식: R = det(M) - k * (trace(M))^2
			harrisResponse[y][x] = det - k * trace * trace;
		}
	}
	return harrisResponse;
}


/**
 * @brief GDI+에서 이미지를 저장하기 위해 필요한 인코더의 CLSID를 얻는 표준 함수.
 */
int GetEncoderClsid(const WCHAR* format, CLSID* pClsid)
{
	UINT  num = 0;
	UINT  size = 0;

	ImageCodecInfo* pImageCodecInfo = NULL;
	GetImageEncodersSize(&num, &size);
	if (size == 0) return -1;

	pImageCodecInfo = (ImageCodecInfo*)(malloc(size));
	if (pImageCodecInfo == NULL) return -1;

	GetImageEncoders(num, size, pImageCodecInfo);

	for (UINT j = 0; j < num; ++j) {
		if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0) {
			*pClsid = pImageCodecInfo[j].Clsid;
			free(pImageCodecInfo);
			return j;
		}
	}

	free(pImageCodecInfo);
	return -1;
}

/**
 * @brief Harris 코너 응답 맵을 0-255 범위로 정규화하여 흑백 이미지 파일로 저장합니다.
 */
void SaveHarrisResponseMap(const vecVecDouble& harrisResponse, const WCHAR* filename)
{
	int height = harrisResponse.size();
	if (height == 0) return;
	int width = harrisResponse[0].size();
	if (width == 0) return;

	// 1. 시각화를 위해 R 값들을 0~255 범위로 변환하는 정규화 과정
	double minR = harrisResponse[0][0];
	double maxR = harrisResponse[0][0];
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			if (harrisResponse[y][x] < minR) minR = harrisResponse[y][x];
			if (harrisResponse[y][x] > maxR) maxR = harrisResponse[y][x];
		}
	}

	Bitmap responseBmp(width, height, PixelFormat24bppRGB);
	BitmapData bitmapData;
	Rect rect(0, 0, width, height);
	responseBmp.LockBits(&rect, ImageLockModeWrite, PixelFormat24bppRGB, &bitmapData);

	BYTE* p = (BYTE*)bitmapData.Scan0;
	int stride = bitmapData.Stride;
	double range = maxR - minR;
	if (range == 0) range = 1.0; // 0으로 나누는 오류 방지

	// 2. 각 픽셀의 R 값을 0~255 사이의 흑백 명암 값으로 변환
	for (int y = 0; y < height; ++y) {
		BYTE* row = p + y * stride;
		for (int x = 0; x < width; ++x) {
			// 정규화 공식: newValue = ( (currentValue - min) / (max - min) ) * 255
			double normalizedValue = (harrisResponse[y][x] - minR) / range;
			BYTE color = static_cast<BYTE>(normalizedValue * 255.0);
			row[x * 3] = color;
			row[x * 3 + 1] = color;
			row[x * 3 + 2] = color;
		}
	}
	responseBmp.UnlockBits(&bitmapData);

	// 3. 정규화된 맵을 이미지 파일로 저장
	CLSID pngClsid;
	GetEncoderClsid(L"image/png", &pngClsid);
	if (responseBmp.Save(filename, &pngClsid, NULL) == Ok) {
		std::wcout << L"Harris response map saved to: " << filename << std::endl;
	}
	else {
		std::wcerr << L"Failed to save Harris response map." << std::endl;
	}
}

// 프로그램의 시작점
int main()
{
	// main 함수가 시작될 때 GDI+를 초기화하고, 끝날 때 자동으로 해제
	GdiplusInitializer gdiplusInitializer;

	// --- 1. 이미지 로드 ---
	Bitmap* originalBmp = new Bitmap(L"./images/shapes_01.png");
	if (originalBmp->GetLastStatus() != Ok) {
		std::cerr << "이미지 파일을 열 수 없습니다." << std::endl;
		return -1;
	}
	UINT width = originalBmp->GetWidth();
	UINT height = originalBmp->GetHeight();

	// --- 2. 흑백 변환 (전처리) ---
	vecVecDouble grayImage = ConvertToGrayscale(originalBmp);

	// --- 3. 그래디언트 계산 ---
	vecVecDouble gradX(height, vecDouble(width, 0));
	vecVecDouble gradY(height, vecDouble(width, 0));
	ComputeGradients(grayImage, gradX, gradY);

	// --- 4. Harris 코너 응답 계산 ---
	int windowSize = 3;
	double k = 0.04;
	vecVecDouble harrisResponse = ComputeHarrisResponse(gradX, gradY, windowSize, k);

	// 중간 결과물인 Harris Response Map을 이미지 파일로 저장
	SaveHarrisResponseMap(harrisResponse, L"./images/result_harris_response.png");

	// --- 5. 최종 코너점 찾기 ---
	double cornerThreshold = 0.05; // 임계값을 5%로 설정하여 더 확실한 코너만 검출
	vecCorner detectedCorners = GetCorners(harrisResponse, cornerThreshold);

	std::cout << "Detected " << detectedCorners.size() << " corners." << std::endl;

	// --- 6. 결과 그리기 ---
	DrawCorners(originalBmp, detectedCorners);

	// --- 7. 결과 저장 ---
	CLSID pngClsid;
	GetEncoderClsid(L"image/png", &pngClsid);
	originalBmp->Save(L"./images/result_corners.png", &pngClsid, NULL);

	// 할당된 메모리 해제
	delete originalBmp;
	return 0;
}
