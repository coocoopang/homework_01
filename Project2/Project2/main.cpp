#include "main.h"

/**
 * @brief 누산기 배열에서 비최대 억제(NMS)를 적용하여 의미 있는 직선들을 추출합니다.
 * @param accumulator Hough 변환이 완료된 누산기 배열
 * @param threshold 직선으로 간주할 최소 누산기 값 (투표 수)
 * @param maxLines 반환할 최대 직선의 개수
 * @return 검출된 직선 정보(Line)를 담은 벡터
 */
vecLine GetLinesFromAccumulator(
	const vecVecInt& accumulator,
	int threshold, int maxLines)
{
	vecLine allCandidates; // 모든 직선 후보를 저장할 벡터
	int rhoSize = accumulator.size();         // 누산기의 높이 (rho의 범위)
	int thetaSize = accumulator[0].size();    // 누산기의 너비 (theta의 범위)
	int rhoCenter = rhoSize / 2;              // rho 인덱스 계산을 위한 중심점

	// 1. 비최대 억제(NMS)를 통해 지역 최댓값(local maxima) 찾기
	//    배열의 경계(가장자리)는 3x3 비교가 불가능하므로 제외 (1부터 size-1까지 순회)
	for (int r_idx = 1; r_idx < rhoSize - 1; ++r_idx) {
		for (int t_idx = 1; t_idx < thetaSize - 1; ++t_idx) {
			int current_score = accumulator[r_idx][t_idx];

			// 현재 지점의 점수가 임계값을 넘어야만 직선 후보로 고려
			if (current_score > threshold) {
				bool is_max = true;
				// 주변 3x3 픽셀(이웃)과 값을 비교
				for (int dr = -1; dr <= 1; ++dr) {
					for (int dt = -1; dt <= 1; ++dt) {
						if (dr == 0 && dt == 0) continue; // 자기 자신과는 비교하지 않음
						// 이웃 중 하나라도 현재 값보다 크면, 현재 위치는 지역 최댓값이 아님
						if (accumulator[r_idx + dr][t_idx + dt] > current_score) {
							is_max = false;
							break;
						}
					}
					if (!is_max) break;
				}

				// 3x3 영역에서 가장 큰 값(산봉우리)으로 확인되면 후보에 추가
				if (is_max) {
					Line line;
					line.rho = (double)(r_idx - rhoCenter); // 인덱스를 실제 rho 값으로 변환
					line.theta = (double)t_idx * M_PI / thetaSize; // 인덱스를 실제 theta 값(라디안)으로 변환
					line.score = current_score;
					allCandidates.push_back(line);
				}
			}
		}
	}

	// 2. 점수(score) 기준으로 후보들을 내림차순 정렬
	//    람다 함수를 사용하여 정렬 기준을 정의
	std::sort(allCandidates.begin(), allCandidates.end(), [](const Line& a, const Line& b) {
		return a.score > b.score;
	});

	// 3. 가장 점수가 높은 상위 maxLines 개수만큼의 라인만 최종 선택
	vecLine finalLines;
	for (int i = 0; i < std::min((int)allCandidates.size(), maxLines); ++i) {
		finalLines.push_back(allCandidates[i]);
	}

	return finalLines;
}

/**
 * @brief GDI+ Bitmap을 2D 벡터 형태의 엣지 맵으로 변환합니다.
 * @param bmp 원본 이미지에 대한 비트맵 포인터
 * @param threshold 엣지로 판단할 밝기 임계값
 * @return 엣지는 1, 배경은 0으로 채워진 2D 정수 벡터
 */
vecVecInt CreateEdgeMap(Bitmap* bmp, BYTE threshold)
{
	UINT width = bmp->GetWidth();
	UINT height = bmp->GetHeight();
	vecVecInt edgeMap(height, vecInt(width, 0));

	BitmapData bitmapData;
	Rect rect(0, 0, width, height);
	// LockBits: GetPixel/SetPixel보다 훨씬 빠른, 픽셀 데이터에 직접 접근하는 방법
	bmp->LockBits(&rect, ImageLockModeRead, PixelFormat24bppRGB, &bitmapData);

	BYTE* p = (BYTE*)bitmapData.Scan0; // 픽셀 데이터의 시작 주소
	int stride = bitmapData.Stride;    // 한 줄의 이미지 데이터가 차지하는 실제 바이트 수

	for (UINT y = 0; y < height; ++y) {
		for (UINT x = 0; x < width; ++x) {
			// 24bppRGB 포맷이므로 픽셀 하나는 3바이트(Blue, Green, Red)로 구성
			// 픽셀의 주소: p + y * stride + x * 3
			// 간단한 흑백 변환: R, G, B 값의 평균을 계산
			BYTE gray = (p[y * stride + x * 3] + p[y * stride + x * 3 + 1] + p[y * stride + x * 3 + 2]) / 3;
			// 계산된 밝기 값이 임계값보다 크면 엣지로 간주
			if (gray > threshold) {
				edgeMap[y][x] = 1; // 엣지로 표시
			}
		}
	}

	bmp->UnlockBits(&bitmapData); // LockBits로 잠근 메모리 해제
	return edgeMap;
}

/**
 * @brief 엣지 맵을 기반으로 Hough 변환을 수행하여 누산기 배열을 채웁니다.
 * @param edgeMap 엣지 정보가 담긴 2D 벡터
 * @param accumulator [출력] Hough 변환 결과가 누적될 2D 벡터
 * @param width 이미지 너비
 * @param height 이미지 높이
 * @param rhoMax [출력] 계산된 rho의 최댓값
 * @param sinTable [출력] 미리 계산된 sin 값 테이블
 * @param cosTable [출력] 미리 계산된 cos 값 테이블
 */
void PerformHoughTransform(
	const vecVecInt& edgeMap,
	vecVecInt& accumulator,
	int width, int height,
	double& rhoMax, vecDouble& sinTable, vecDouble& cosTable)
{
	int thetaSize = accumulator[0].size();
	// rho의 최댓값은 이미지의 대각선 길이
	rhoMax = sqrt(width * width + height * height);
	int rhoSize = accumulator.size();
	int rhoCenter = rhoSize / 2;

	// 최적화: 반복문 안에서 sin/cos 함수를 계속 호출하는 것을 피하기 위해 미리 계산
	for (int t = 0; t < thetaSize; ++t) {
		double theta = (double)t * M_PI / thetaSize; // 각도(인덱스)를 라디안으로 변환
		cosTable[t] = cos(theta);
		sinTable[t] = sin(theta);
	}

	// 이미지의 모든 픽셀을 순회
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			// 엣지 픽셀(값이 1인 픽셀)에 대해서만 Hough 변환 수행
			if (edgeMap[y][x] != 0) {
				// 가능한 모든 각도(theta)에 대해 rho 값을 계산
				for (int t_idx = 0; t_idx < thetaSize; ++t_idx) {
					// Hough 변환의 핵심 공식: ρ = x*cos(θ) + y*sin(θ)
					double rho = x * cosTable[t_idx] + y * sinTable[t_idx];
					// 계산된 rho 값을 누산기 배열의 인덱스로 변환 (+0.5는 반올림 효과)
					int r_idx = rhoCenter + static_cast<int>(rho + 0.5);
					// 인덱스가 누산기 배열 범위 내에 있을 경우에만 투표(값 증가)
					if (r_idx >= 0 && r_idx < rhoSize) {
						accumulator[r_idx][t_idx]++;
					}
				}
			}
		}
	}
}

/**
 * @brief 검출된 직선들을 원본 비트맵 위에 그립니다.
 * @param bmp 원본 이미지에 대한 비트맵 포인터
 * @param lines 그릴 직선들의 정보를 담은 벡터
 */
void DrawLines(Bitmap* bmp, const vecLine& lines)
{
	Graphics graphics(bmp); // GDI+의 그리기 작업을 위한 객체
	Pen pen(Color(255, 255, 0, 0), 3); // 두께 3의 빨간색 펜 생성

	for (const auto& line : lines) {
		// 직선의 파라미터(rho, theta)를 이용하여 직선 위의 두 점을 계산
		double a = cos(line.theta);
		double b = sin(line.theta);

		// (x0, y0)는 원점에서 직선에 가장 가까운 점의 좌표
		double x0 = a * line.rho;
		double y0 = b * line.rho;

		// (x0, y0)를 기준으로, 직선의 방향 벡터(a,b)에 수직인 벡터(-b,a)를 따라
		// 양쪽으로 충분히 멀리 떨어진 두 점(x1,y1), (x2,y2)를 계산하여 긴 직선을 그림
		int x1 = static_cast<int>(x0 + 2000 * (-b));
		int y1 = static_cast<int>(y0 + 2000 * (a));
		int x2 = static_cast<int>(x0 - 2000 * (-b));
		int y2 = static_cast<int>(y0 - 2000 * (a));

		graphics.DrawLine(&pen, x1, y1, x2, y2);
	}
}

/**
 * @brief GDI+에서 이미지를 저장하기 위해 필요한 인코더의 CLSID를 얻는 표준 함수.
 * @param format 찾고자 하는 이미지 포맷 (예: L"image/png")
 * @param pClsid [출력] 찾은 인코더의 CLSID가 저장될 포인터
 * @return 성공 시 0 이상의 값, 실패 시 -1
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
 * @brief 엣지 맵(2D 벡터)을 시각적으로 확인 가능하도록 흑백 이미지 파일로 저장합니다.
 * @param edgeMap 저장할 엣지 맵 데이터
 * @param width 이미지 너비
 * @param height 이미지 높이
 * @param filename 저장할 파일 경로
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
			// edgeMap에서 값이 1이면 흰색, 0이면 검은색으로 픽셀을 채움
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

// 메인 프로그램 실행 함수
int main()
{
	// GdiplusInitializer 객체가 생성될 때 GDI+가 시작되고,
	// main 함수가 끝날 때 객체가 소멸되면서 자동으로 GDI+가 종료됨.
	GdiplusInitializer gdiplusInitializer;

	// 1. 이미지 로드
	Bitmap* originalBmp = new Bitmap(L"./images/building_01.png");
	if (originalBmp->GetLastStatus() != Ok) {
		std::cerr << "이미지 파일을 열 수 없습니다." << std::endl;
		return -1;
	}
	UINT width = originalBmp->GetWidth();
	UINT height = originalBmp->GetHeight();

	// 2. 엣지 검출 (이미지 전처리)
	vecVecInt edgeMap = CreateEdgeMap(originalBmp, 128);
	// 중간 결과물인 엣지 맵을 파일로 저장하여 확인
	SaveEdgeMap(edgeMap, width, height, L"./images/result_edge_map.png");

	// 3. Hough 변환 수행
	int thetaSize = 180; // theta를 0~179도까지 1도 단위로 검사
	double rhoMax = sqrt(width * width + height * height);
	int rhoSize = static_cast<int>(2 * rhoMax); // rho는 -rhoMax ~ +rhoMax 범위
	// 누산기 배열을 0으로 초기화
	vecVecInt accumulator(rhoSize, vecInt(thetaSize, 0));
	vecDouble sinTable(thetaSize); // sin, cos 테이블
	vecDouble cosTable(thetaSize);
	PerformHoughTransform(edgeMap, accumulator, width, height, rhoMax, sinTable, cosTable);

	// 4. 직선 추출 및 필터링
	// 4-1. 모든 후보 직선 추출
	int lineThreshold = 100; // 직선으로 판단할 최소 투표 수
	int maxLinesToDraw = 50;   // 점수 순으로 정렬 후 상위 50개만 고려
	vecLine allDetectedLines = GetLinesFromAccumulator(accumulator, lineThreshold, maxLinesToDraw);
	std::cout << "Total detected lines (before filtering): " << allDetectedLines.size() << std::endl;

	// 4-2. 수평선 필터링
	vecLine horizontalLines;
	const double angle_tolerance_degrees = 10.0; // 90도에서 +-10도 범위의 직선을 수평선으로 간주
	const double angle_tolerance_radians = angle_tolerance_degrees * M_PI / 180.0;
	const double ninety_degrees_radians = M_PI / 2.0; // 90도를 라디안으로 표현

	for (const auto& line : allDetectedLines) {
		// 직선의 theta 값이 90도(라디안)와의 차이가 허용 오차 이내인지 확인
		if (std::abs(line.theta - ninety_degrees_radians) <= angle_tolerance_radians) {
			horizontalLines.push_back(line);
		}
	}
	std::cout << "Detected horizontal lines (after filtering): " << horizontalLines.size() << std::endl;

	// 5. 필터링된 수평선만 그리기
	DrawLines(originalBmp, horizontalLines);

	// 6. 최종 결과 이미지 저장
	CLSID pngClsid;
	GetEncoderClsid(L"image/png", &pngClsid);
	originalBmp->Save(L"./images/result_lines.png", &pngClsid, NULL);

	// 동적으로 할당한 비트맵 객체 해제
	delete originalBmp;
	return 0;
}