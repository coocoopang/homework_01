#include "main.h"

/**
 * @brief Harris ���� �ʿ��� ���ִ� ����(NMS)�� �Ӱ谪�� �����Ͽ� ���� �ڳ����� �����մϴ�.
 * @param harrisResponse �� �ȼ��� �ڳ� ����(R) ���� ����� 2D ����
 * @param threshold �ڳʷ� �Ǵ��� �Ӱ谪 (�ִ� ���� ���� ���� ����, ��: 0.01�� ���� 1%)
 * @return ����� �ڳ�(Corner)���� ����
 */
vecCorner GetCorners(const vecVecDouble& harrisResponse, double threshold)
{
	vecCorner corners;
	int height = harrisResponse.size();
	int width = harrisResponse[0].size();

	// 1. R ���� �ִ��� ã�� �Ӱ谪�� ���밪���� ��ȯ�մϴ�.
	//    �̷��� �ϸ� �̹����� �������� ��⳪ ��� ������� �ϰ��� ������ �����մϴ�.
	double maxResponse = 0;
	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			if (harrisResponse[y][x] > maxResponse) {
				maxResponse = harrisResponse[y][x];
			}
		}
	}
	double actualThreshold = maxResponse * threshold; // ���� ������ �Ӱ谪

	// 2. ��� �ȼ��� ��ȸ�ϸ� �ڳ����� ã���ϴ�.
	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			double r = harrisResponse[y][x];
			// ���� �ȼ��� R ���� ���� �Ӱ谪���� Ŭ ��쿡�� �ڳ� �ĺ��� ����
			if (r > actualThreshold) {
				// 3. ���ִ� ����(Non-Maximum Suppression): �ֺ� 3x3 �������� �ִ����� Ȯ��
				bool isMax = true;
				for (int i = -1; i <= 1; ++i) {
					for (int j = -1; j <= 1; ++j) {
						// �̿� �ȼ��� R ���� ���� �ȼ����� ũ��, ���� �ȼ��� ��¥ �ڳʰ� �ƴ�
						if (harrisResponse[y + i][x + j] > r) {
							isMax = false;
							break;
						}
					}
					if (!isMax) break;
				}
				// ���� �ִ����� �Ǹ�Ǹ� ���� �ڳ� ��Ͽ� �߰�
				if (isMax) {
					corners.push_back({ x, y, r });
				}
			}
		}
	}
	return corners;
}

/**
 * @brief ����� �ڳʵ��� ���� ��Ʈ�� ���� �ð������� ǥ���մϴ�.
 * @param bmp ���� �̹����� ���� ��Ʈ�� ������
 * @param corners �׸� �ڳʵ��� ������ ���� ����
 */
void DrawCorners(Bitmap* bmp, const vecCorner& corners)
{
	Graphics graphics(bmp); // GDI+ �׸��� ��ü ����
	SolidBrush brush(Color(255, 255, 0, 0)); // ä���� ������ �귯�� ����

	for (const auto& corner : corners) {
		// �� �ڳ��� (x, y) ��ġ�� ���� ��(Ÿ��)�� �׷� ǥ��
		graphics.FillEllipse(&brush, corner.x - 6, corner.y - 6, 12, 12);
	}
}

/**
 * @brief GDI+ Bitmap�� 2D ��� �̹���(double Ÿ�� ����)�� ��ȯ�մϴ�.
 * @param bmp ���� �̹����� ���� ��Ʈ�� ������
 * @return 0.0 ~ 255.0 ������ ��� ���� ������ 2D double ����
 */
vecVecDouble ConvertToGrayscale(Bitmap* bmp)
{
	UINT width = bmp->GetWidth();
	UINT height = bmp->GetHeight();
	vecVecDouble grayImage(height, vecDouble(width));

	BitmapData bitmapData;
	Rect rect(0, 0, width, height);
	// LockBits: �ȼ� �����Ϳ� ���� �����Ͽ� ó�� �ӵ��� ���̴� GDI+�� ǥ������ ���
	bmp->LockBits(&rect, ImageLockModeRead, PixelFormat24bppRGB, &bitmapData);

	BYTE* p = (BYTE*)bitmapData.Scan0; // �ȼ� �������� ���� �޸� �ּ�
	int stride = bitmapData.Stride;    // �̹��� �� ��(row)�� ���� �޸� ũ�� (byte)

	for (UINT y = 0; y < height; ++y) {
		for (UINT x = 0; x < width; ++x) {
			// �ȼ��� �ּ�: �����ּ� + y * stride + x * 3 (�ȼ��� 3����Ʈ)
			// R, G, B ���� ����� ���� �����ϰ� ��� ������ ��ȯ
			grayImage[y][x] = (double)(p[y * stride + x * 3] + p[y * stride + x * 3 + 1] + p[y * stride + x * 3 + 2]) / 3.0;
		}
	}

	bmp->UnlockBits(&bitmapData); // �޸� ��� ����
	return grayImage;
}

/**
 * @brief Sobel �����ڸ� �̿��Ͽ� �̹����� x, y ���� �׷����Ʈ(��� ��ȭ��)�� ����մϴ�.
 * @param grayImage ��� �̹��� ������
 * @param gradX [���] x���� �׷����Ʈ�� ����� 2D ����
 * @param gradY [���] y���� �׷����Ʈ�� ����� 2D ����
 */
void ComputeGradients(const vecVecDouble& grayImage,
	vecVecDouble& gradX,
	vecVecDouble& gradY)
{
	int height = grayImage.size();
	int width = grayImage[0].size();

	// Sobel Ŀ��(����ũ): �̹����� �̺�(����)�� �ٻ��ϴ� �� ���Ǵ� ���
	double sobelX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} }; // x��(���μ�) ���� ����
	double sobelY[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} }; // y��(���μ�) ���� ����

	// �̹��� ���(�����ڸ� 1�ȼ�)�� �����ϰ� �������(convolution) ����
	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			double gx = 0, gy = 0;
			// ���� �ȼ� (x,y)�� �߽����� 3x3 �����쿡 Sobel Ŀ���� ����
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
 * @brief ���� �׷����Ʈ�� �̿��Ͽ� Harris �ڳ� ����(R score)�� ����մϴ�.
 * @param gradX x���� �׷����Ʈ ��
 * @param gradY y���� �׷����Ʈ ��
 * @param windowSize �ڳ� ���� ��� �� �ֺ� �ȼ��� ����� ������ ũ�� (���� 3 �Ǵ� 5)
 * @param k Harris �ڳ� ������� ������ ��� (���� 0.04 ~ 0.06)
 * @return �� �ȼ��� �ڳ� ����(R) ���� ��� 2D ����
 */
vecVecDouble ComputeHarrisResponse(
	const vecVecDouble& gradX,
	const vecVecDouble& gradY,
	int windowSize, double k)
{
	int height = gradX.size();
	int width = gradX[0].size();
	vecVecDouble harrisResponse(height, vecDouble(width, 0.0));

	// ����ȭ�� ���� Ix*Ix, Iy*Iy, Ix*Iy ���� �̸� ����Ͽ� ����
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
			// ������ ������ ���� �׷����Ʈ ������ ��� �ջ� (����þ� ���͸��� ������ ȿ��)
			for (int i = -offset; i <= offset; ++i) {
				for (int j = -offset; j <= offset; ++j) {
					Sxx += Ixx[y + i][x + j];
					Syy += Iyy[y + i][x + j];
					Sxy += Ixy[y + i][x + j];
				}
			}

			// �׷����Ʈ ��� M�� determinant�� trace�� ���
			// M = [ Sxx Sxy ]
			//     [ Sxy Syy ]
			double det = Sxx * Syy - Sxy * Sxy;
			double trace = Sxx + Syy;
			// Harris �ڳ� ���� R �� ��� ����: R = det(M) - k * (trace(M))^2
			harrisResponse[y][x] = det - k * trace * trace;
		}
	}
	return harrisResponse;
}


/**
 * @brief GDI+���� �̹����� �����ϱ� ���� �ʿ��� ���ڴ��� CLSID�� ��� ǥ�� �Լ�.
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
 * @brief Harris �ڳ� ���� ���� 0-255 ������ ����ȭ�Ͽ� ��� �̹��� ���Ϸ� �����մϴ�.
 */
void SaveHarrisResponseMap(const vecVecDouble& harrisResponse, const WCHAR* filename)
{
	int height = harrisResponse.size();
	if (height == 0) return;
	int width = harrisResponse[0].size();
	if (width == 0) return;

	// 1. �ð�ȭ�� ���� R ������ 0~255 ������ ��ȯ�ϴ� ����ȭ ����
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
	if (range == 0) range = 1.0; // 0���� ������ ���� ����

	// 2. �� �ȼ��� R ���� 0~255 ������ ��� ��� ������ ��ȯ
	for (int y = 0; y < height; ++y) {
		BYTE* row = p + y * stride;
		for (int x = 0; x < width; ++x) {
			// ����ȭ ����: newValue = ( (currentValue - min) / (max - min) ) * 255
			double normalizedValue = (harrisResponse[y][x] - minR) / range;
			BYTE color = static_cast<BYTE>(normalizedValue * 255.0);
			row[x * 3] = color;
			row[x * 3 + 1] = color;
			row[x * 3 + 2] = color;
		}
	}
	responseBmp.UnlockBits(&bitmapData);

	// 3. ����ȭ�� ���� �̹��� ���Ϸ� ����
	CLSID pngClsid;
	GetEncoderClsid(L"image/png", &pngClsid);
	if (responseBmp.Save(filename, &pngClsid, NULL) == Ok) {
		std::wcout << L"Harris response map saved to: " << filename << std::endl;
	}
	else {
		std::wcerr << L"Failed to save Harris response map." << std::endl;
	}
}

// ���α׷��� ������
int main()
{
	// main �Լ��� ���۵� �� GDI+�� �ʱ�ȭ�ϰ�, ���� �� �ڵ����� ����
	GdiplusInitializer gdiplusInitializer;

	// --- 1. �̹��� �ε� ---
	Bitmap* originalBmp = new Bitmap(L"./images/shapes_01.png");
	if (originalBmp->GetLastStatus() != Ok) {
		std::cerr << "�̹��� ������ �� �� �����ϴ�." << std::endl;
		return -1;
	}
	UINT width = originalBmp->GetWidth();
	UINT height = originalBmp->GetHeight();

	// --- 2. ��� ��ȯ (��ó��) ---
	vecVecDouble grayImage = ConvertToGrayscale(originalBmp);

	// --- 3. �׷����Ʈ ��� ---
	vecVecDouble gradX(height, vecDouble(width, 0));
	vecVecDouble gradY(height, vecDouble(width, 0));
	ComputeGradients(grayImage, gradX, gradY);

	// --- 4. Harris �ڳ� ���� ��� ---
	int windowSize = 3;
	double k = 0.04;
	vecVecDouble harrisResponse = ComputeHarrisResponse(gradX, gradY, windowSize, k);

	// �߰� ������� Harris Response Map�� �̹��� ���Ϸ� ����
	SaveHarrisResponseMap(harrisResponse, L"./images/result_harris_response.png");

	// --- 5. ���� �ڳ��� ã�� ---
	double cornerThreshold = 0.05; // �Ӱ谪�� 5%�� �����Ͽ� �� Ȯ���� �ڳʸ� ����
	vecCorner detectedCorners = GetCorners(harrisResponse, cornerThreshold);

	std::cout << "Detected " << detectedCorners.size() << " corners." << std::endl;

	// --- 6. ��� �׸��� ---
	DrawCorners(originalBmp, detectedCorners);

	// --- 7. ��� ���� ---
	CLSID pngClsid;
	GetEncoderClsid(L"image/png", &pngClsid);
	originalBmp->Save(L"./images/result_corners.png", &pngClsid, NULL);

	// �Ҵ�� �޸� ����
	delete originalBmp;
	return 0;
}
