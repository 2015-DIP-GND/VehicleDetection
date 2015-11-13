#include <iostream>
#include <Windows.h>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include<opencv\cv.h>
#include<opencv\highgui.h>

using namespace cv;
using namespace std;

Mat src,
dst,
temp;

int main() {
	//Image 로드
	src = imread("C:\\Users\\kk070\\Documents\\self.jpg", CV_LOAD_IMAGE_COLOR);
	
	//이미지 밝기 조절 TEST
	double alpha = 1.0;//1.0~3.0
	int beta = 50;//0~100

	//Image 로드 되었는지 확인
	if (src.data == NULL) {
		cout << "Error: Image could not be loaded, Invalid Path"
			<< endl;
		getchar();
		return -1;
	};
	

	cvNamedWindow("Original Image",
		CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Scaled",
		CV_WINDOW_AUTOSIZE);
	imshow("Original Image", src);
	temp = src;
	dst = temp;
	imshow("Scaled", dst);

	while (true) {
		int c = waitKey(10);
		if ((char)c == 27) { break; }
		if ((char)c == 'u') {
			pyrUp(temp, dst, Size(temp.cols * 2, temp.rows * 2));
		}
		if ((char)c == 'd') {
			pyrDown(temp, dst, Size(temp.cols / 2, temp.rows / 2));
		}
		if ((char)c == 'a') {
			//픽셀 접근법
			for (int y = 0; y < temp.rows; y++)
			{
				for (int x = 0; x < temp.cols; x++)
				{
					for (int c = 0; c < 3; c++)
					{
						dst.at<Vec3b>(y, x)[c] =
							saturate_cast<uchar>(alpha * (temp.at<Vec3b>(y, x)[c]) + beta);
					}
				}
			}
		}
		imshow("Scaled", dst);
		temp = dst;
	}
	return 0;
}