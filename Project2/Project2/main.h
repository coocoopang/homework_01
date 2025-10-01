#pragma once
#define NOMINMAX // std::min과 충돌을 막기 위해 windows.h include 전에 처리
#include <windows.h>
#include <gdiplus.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// GDI+ 네임스페이스 사용
using namespace Gdiplus;
#pragma comment (lib,"Gdiplus.lib")

const double M_PI = 3.14159265358979323846;

// 검출된 직선의 정보를 저장하기 위한 구조체
struct Line {
	double rho;   // 원점에서 직선까지의 수직 거리 (ρ)
	double theta; // 그 수직선이 x축과 이루는 각도 (θ, 라디안 단위)
	int score;    // 누산기 값 (얼마나 많은 점이 이 직선에 투표했는지)
};

// 자주 사용하는 타입 정의
typedef std::vector<Line>				vecLine;
typedef std::vector<int>				vecInt;
typedef std::vector<double>				vecDouble;
typedef std::vector<std::vector<int>>	vecVecInt;

// GDI+ 시작 및 종료를 위한 헬퍼 클래스
class GdiplusInitializer {
public:
	GdiplusInitializer() {
		GdiplusStartupInput gdiplusStartupInput;
		GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
	}
	~GdiplusInitializer() {
		GdiplusShutdown(gdiplusToken);
	}
private:
	ULONG_PTR gdiplusToken;
};