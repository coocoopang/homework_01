#include "main.h"

/**
 * @brief ����� �迭���� ���ִ� ����(NMS)�� �����Ͽ� �ǹ� �ִ� �������� �����մϴ�.
 * @param accumulator Hough ��ȯ�� �Ϸ�� ����� �迭
 * @param threshold �������� ������ �ּ� ����� �� (��ǥ ��)
 * @param maxLines ��ȯ�� �ִ� ������ ����
 * @return ����� ���� ����(Line)�� ���� ����
 */
vecLine GetLinesFromAccumulator(
	const vecVecInt& accumulator,
	int threshold, int maxLines)
{
	vecLine allCandidates; // ��� ���� �ĺ��� ������ ����
	int rhoSize = accumulator.size();         // ������� ���� (rho�� ����)
	int thetaSize = accumulator[0].size();    // ������� �ʺ� (theta�� ����)
	int rhoCenter = rhoSize / 2;              // rho �ε��� ����� ���� �߽���

	// 1. ���ִ� ����(NMS)�� ���� ���� �ִ�(local maxima) ã��
	//    �迭�� ���(�����ڸ�)�� 3x3 �񱳰� �Ұ����ϹǷ� ���� (1���� size-1���� ��ȸ)
	for (int r_idx = 1; r_idx < rhoSize - 1; ++r_idx) {
		for (int t_idx = 1; t_idx < thetaSize - 1; ++t_idx) {
			int current_score = accumulator[r_idx][t_idx];

			// ���� ������ ������ �Ӱ谪�� �Ѿ�߸� ���� �ĺ��� ���
			if (current_score > threshold) {
				bool is_max = true;
				// �ֺ� 3x3 �ȼ�(�̿�)�� ���� ��
				for (int dr = -1; dr <= 1; ++dr) {
					for (int dt = -1; dt <= 1; ++dt) {
						if (dr == 0 && dt == 0) continue; // �ڱ� �ڽŰ��� ������ ����
						// �̿� �� �ϳ��� ���� ������ ũ��, ���� ��ġ�� ���� �ִ��� �ƴ�
						if (accumulator[r_idx + dr][t_idx + dt] > current_score) {
							is_max = false;
							break;
						}
					}
					if (!is_max) break;
				}

				// 3x3 �������� ���� ū ��(����츮)���� Ȯ�εǸ� �ĺ��� �߰�
				if (is_max) {
					Line line;
					line.rho = (double)(r_idx - rhoCenter); // �ε����� ���� rho ������ ��ȯ
					line.theta = (double)t_idx * M_PI / thetaSize; // �ε����� ���� theta ��(����)���� ��ȯ
					line.score = current_score;
					allCandidates.push_back(line);
				}
			}
		}
	}

	// 2. ����(score) �������� �ĺ����� �������� ����
	//    ���� �Լ��� ����Ͽ� ���� ������ ����
	std::sort(allCandidates.begin(), allCandidates.end(), [](const Line& a, const Line& b) {
		return a.score > b.score;
	});

	// 3. ���� ������ ���� ���� maxLines ������ŭ�� ���θ� ���� ����
	vecLine finalLines;
	for (int i = 0; i < std::min((int)allCandidates.size(), maxLines); ++i) {
		finalLines.push_back(allCandidates[i]);
	}

	return finalLines;
}

/**
 * @brief GDI+ Bitmap�� 2D ���� ������ ���� ������ ��ȯ�մϴ�.
 * @param bmp ���� �̹����� ���� ��Ʈ�� ������
 * @param threshold ������ �Ǵ��� ��� �Ӱ谪
 * @return ������ 1, ����� 0���� ä���� 2D ���� ����
 */
vecVecInt CreateEdgeMap(Bitmap* bmp, BYTE threshold)
{
	UINT width = bmp->GetWidth();
	UINT height = bmp->GetHeight();
	vecVecInt edgeMap(height, vecInt(width, 0));

	BitmapData bitmapData;
	Rect rect(0, 0, width, height);
	// LockBits: GetPixel/SetPixel���� �ξ� ����, �ȼ� �����Ϳ� ���� �����ϴ� ���
	bmp->LockBits(&rect, ImageLockModeRead, PixelFormat24bppRGB, &bitmapData);

	BYTE* p = (BYTE*)bitmapData.Scan0; // �ȼ� �������� ���� �ּ�
	int stride = bitmapData.Stride;    // �� ���� �̹��� �����Ͱ� �����ϴ� ���� ����Ʈ ��

	for (UINT y = 0; y < height; ++y) {
		for (UINT x = 0; x < width; ++x) {
			// 24bppRGB �����̹Ƿ� �ȼ� �ϳ��� 3����Ʈ(Blue, Green, Red)�� ����
			// �ȼ��� �ּ�: p + y * stride + x * 3
			// ������ ��� ��ȯ: R, G, B ���� ����� ���
			BYTE gray = (p[y * stride + x * 3] + p[y * stride + x * 3 + 1] + p[y * stride + x * 3 + 2]) / 3;
			// ���� ��� ���� �Ӱ谪���� ũ�� ������ ����
			if (gray > threshold) {
				edgeMap[y][x] = 1; // ������ ǥ��
			}
		}
	}

	bmp->UnlockBits(&bitmapData); // LockBits�� ��� �޸� ����
	return edgeMap;
}

/**
 * @brief ���� ���� ������� Hough ��ȯ�� �����Ͽ� ����� �迭�� ä��ϴ�.
 * @param edgeMap ���� ������ ��� 2D ����
 * @param accumulator [���] Hough ��ȯ ����� ������ 2D ����
 * @param width �̹��� �ʺ�
 * @param height �̹��� ����
 * @param rhoMax [���] ���� rho�� �ִ�
 * @param sinTable [���] �̸� ���� sin �� ���̺�
 * @param cosTable [���] �̸� ���� cos �� ���̺�
 */
void PerformHoughTransform(
	const vecVecInt& edgeMap,
	vecVecInt& accumulator,
	int width, int height,
	double& rhoMax, vecDouble& sinTable, vecDouble& cosTable)
{
	int thetaSize = accumulator[0].size();
	// rho�� �ִ��� �̹����� �밢�� ����
	rhoMax = sqrt(width * width + height * height);
	int rhoSize = accumulator.size();
	int rhoCenter = rhoSize / 2;

	// ����ȭ: �ݺ��� �ȿ��� sin/cos �Լ��� ��� ȣ���ϴ� ���� ���ϱ� ���� �̸� ���
	for (int t = 0; t < thetaSize; ++t) {
		double theta = (double)t * M_PI / thetaSize; // ����(�ε���)�� �������� ��ȯ
		cosTable[t] = cos(theta);
		sinTable[t] = sin(theta);
	}

	// �̹����� ��� �ȼ��� ��ȸ
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			// ���� �ȼ�(���� 1�� �ȼ�)�� ���ؼ��� Hough ��ȯ ����
			if (edgeMap[y][x] != 0) {
				// ������ ��� ����(theta)�� ���� rho ���� ���
				for (int t_idx = 0; t_idx < thetaSize; ++t_idx) {
					// Hough ��ȯ�� �ٽ� ����: �� = x*cos(��) + y*sin(��)
					double rho = x * cosTable[t_idx] + y * sinTable[t_idx];
					// ���� rho ���� ����� �迭�� �ε����� ��ȯ (+0.5�� �ݿø� ȿ��)
					int r_idx = rhoCenter + static_cast<int>(rho + 0.5);
					// �ε����� ����� �迭 ���� ���� ���� ��쿡�� ��ǥ(�� ����)
					if (r_idx >= 0 && r_idx < rhoSize) {
						accumulator[r_idx][t_idx]++;
					}
				}
			}
		}
	}
}

/**
 * @brief ����� �������� ���� ��Ʈ�� ���� �׸��ϴ�.
 * @param bmp ���� �̹����� ���� ��Ʈ�� ������
 * @param lines �׸� �������� ������ ���� ����
 */
void DrawLines(Bitmap* bmp, const vecLine& lines)
{
	Graphics graphics(bmp); // GDI+�� �׸��� �۾��� ���� ��ü
	Pen pen(Color(255, 255, 0, 0), 3); // �β� 3�� ������ �� ����

	for (const auto& line : lines) {
		// ������ �Ķ����(rho, theta)�� �̿��Ͽ� ���� ���� �� ���� ���
		double a = cos(line.theta);
		double b = sin(line.theta);

		// (x0, y0)�� �������� ������ ���� ����� ���� ��ǥ
		double x0 = a * line.rho;
		double y0 = b * line.rho;

		// (x0, y0)�� ��������, ������ ���� ����(a,b)�� ������ ����(-b,a)�� ����
		// �������� ����� �ָ� ������ �� ��(x1,y1), (x2,y2)�� ����Ͽ� �� ������ �׸�
		int x1 = static_cast<int>(x0 + 2000 * (-b));
		int y1 = static_cast<int>(y0 + 2000 * (a));
		int x2 = static_cast<int>(x0 - 2000 * (-b));
		int y2 = static_cast<int>(y0 - 2000 * (a));

		graphics.DrawLine(&pen, x1, y1, x2, y2);
	}
}

/**
 * @brief GDI+���� �̹����� �����ϱ� ���� �ʿ��� ���ڴ��� CLSID�� ��� ǥ�� �Լ�.
 * @param format ã���� �ϴ� �̹��� ���� (��: L"image/png")
 * @param pClsid [���] ã�� ���ڴ��� CLSID�� ����� ������
 * @return ���� �� 0 �̻��� ��, ���� �� -1
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
 * @brief ���� ��(2D ����)�� �ð������� Ȯ�� �����ϵ��� ��� �̹��� ���Ϸ� �����մϴ�.
 * @param edgeMap ������ ���� �� ������
 * @param width �̹��� �ʺ�
 * @param height �̹��� ����
 * @param filename ������ ���� ���
 */
void SaveEdgeMap(const vecVecInt& edgeMap, UINT width, UINT height, const WCHAR* filename)
{
	Bitmap edgeBmp(width, height, PixelFormat24bppRGB);

	BitmapData bitmapData;
	Rect rect(0, 0, width, height);
	edgeBmp.LockBits(&rect, ImageLockModeWrite, PixelFormat24bppRGB, &bitmapData);

	BYTE* p = (BYTE*)bitmapData.Scan0;
	int stride = bitmapData.Stride;
	const BYTE white = 255;
	const BYTE black = 0;

	for (UINT y = 0; y < height; ++y) {
		BYTE* row = p + y * stride;
		for (UINT x = 0; x < width; ++x) {
			// edgeMap���� ���� 1�̸� ���, 0�̸� ���������� �ȼ��� ä��
			BYTE color = (edgeMap[y][x] == 1) ? white : black;
			row[x * 3] = color;
			row[x * 3 + 1] = color;
			row[x * 3 + 2] = color;
		}
	}

	edgeBmp.UnlockBits(&bitmapData);

	CLSID pngClsid;
	GetEncoderClsid(L"image/png", &pngClsid);
	if (edgeBmp.Save(filename, &pngClsid, NULL) == Ok) {
		std::wcout << L"Edge map saved to: " << filename << std::endl;
	}
	else {
		std::wcerr << L"Failed to save edge map." << std::endl;
	}
}

// ���� ���α׷� ���� �Լ�
int main()
{
	// GdiplusInitializer ��ü�� ������ �� GDI+�� ���۵ǰ�,
	// main �Լ��� ���� �� ��ü�� �Ҹ�Ǹ鼭 �ڵ����� GDI+�� �����.
	GdiplusInitializer gdiplusInitializer;

	// 1. �̹��� �ε�
	Bitmap* originalBmp = new Bitmap(L"./images/building_01.png");
	if (originalBmp->GetLastStatus() != Ok) {
		std::cerr << "�̹��� ������ �� �� �����ϴ�." << std::endl;
		return -1;
	}
	UINT width = originalBmp->GetWidth();
	UINT height = originalBmp->GetHeight();

	// 2. ���� ���� (�̹��� ��ó��)
	vecVecInt edgeMap = CreateEdgeMap(originalBmp, 128);
	// �߰� ������� ���� ���� ���Ϸ� �����Ͽ� Ȯ��
	SaveEdgeMap(edgeMap, width, height, L"./images/result_edge_map.png");

	// 3. Hough ��ȯ ����
	int thetaSize = 180; // theta�� 0~179������ 1�� ������ �˻�
	double rhoMax = sqrt(width * width + height * height);
	int rhoSize = static_cast<int>(2 * rhoMax); // rho�� -rhoMax ~ +rhoMax ����
	// ����� �迭�� 0���� �ʱ�ȭ
	vecVecInt accumulator(rhoSize, vecInt(thetaSize, 0));
	vecDouble sinTable(thetaSize); // sin, cos ���̺�
	vecDouble cosTable(thetaSize);
	PerformHoughTransform(edgeMap, accumulator, width, height, rhoMax, sinTable, cosTable);

	// 4. ���� ���� �� ���͸�
	// 4-1. ��� �ĺ� ���� ����
	int lineThreshold = 100; // �������� �Ǵ��� �ּ� ��ǥ ��
	int maxLinesToDraw = 50;   // ���� ������ ���� �� ���� 50���� ���
	vecLine allDetectedLines = GetLinesFromAccumulator(accumulator, lineThreshold, maxLinesToDraw);
	std::cout << "Total detected lines (before filtering): " << allDetectedLines.size() << std::endl;

	// 4-2. ���� ���͸�
	vecLine horizontalLines;
	const double angle_tolerance_degrees = 10.0; // 90������ +-10�� ������ ������ �������� ����
	const double angle_tolerance_radians = angle_tolerance_degrees * M_PI / 180.0;
	const double ninety_degrees_radians = M_PI / 2.0; // 90���� �������� ǥ��

	for (const auto& line : allDetectedLines) {
		// ������ theta ���� 90��(����)���� ���̰� ��� ���� �̳����� Ȯ��
		if (std::abs(line.theta - ninety_degrees_radians) <= angle_tolerance_radians) {
			horizontalLines.push_back(line);
		}
	}
	std::cout << "Detected horizontal lines (after filtering): " << horizontalLines.size() << std::endl;

	// 5. ���͸��� ���򼱸� �׸���
	DrawLines(originalBmp, horizontalLines);

	// 6. ���� ��� �̹��� ����
	CLSID pngClsid;
	GetEncoderClsid(L"image/png", &pngClsid);
	originalBmp->Save(L"./images/result_lines.png", &pngClsid, NULL);

	// �������� �Ҵ��� ��Ʈ�� ��ü ����
	delete originalBmp;
	return 0;
}