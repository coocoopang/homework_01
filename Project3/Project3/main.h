#pragma once
#define NOMINMAX // std::min�� �浹�� ���� ���� windows.h include ���� ó��
#include <windows.h>
#include <gdiplus.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// GDI+ ���ӽ����̽� ���
using namespace Gdiplus;
#pragma comment (lib,"Gdiplus.lib")

const double M_PI = 3.14159265358979323846;

// ����� �ڳ��� ������ �����ϱ� ���� ����ü
struct Corner {
	int x, y;         // �ڳ��� �̹��� �� ��ǥ
	double response;  // �ش� ������ Harris �ڳ� ����(R) ��
};

// ���� ����ϴ� Ÿ�� ����
typedef std::vector<Corner>				 vecCorner;
typedef std::vector<double>				 vecDouble;
typedef std::vector<std::vector<double>> vecVecDouble;

// GDI+ ���� �� ���Ḧ ���� ���� Ŭ����
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