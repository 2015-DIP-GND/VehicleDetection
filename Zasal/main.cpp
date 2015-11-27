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

int *hist;

void hog(Mat source)
{
	int nwin_x = 3, nwin_y = 3, cont = 0;
	int B = 9, ch = source.channels();
	int row = source.rows;
	int col = source.cols;
	hist = (int*)calloc(nwin_x * nwin_y * B, sizeof(int));
	int m = sqrt(row / 2), bin = 0;
	if (col == 1){
		// error control
	}
	int step_x = floor(col / (nwin_x + 1));
	int step_y = floor(row / (nwin_y + 1));
	int hx[] = { -1, 0, 1 };
	int hy[] = { 1, 0, -1 };
	float norm_ = 0;
	float H2[9] {0, };
	float ang_lim = 0;
	Mat grad_xr(col, row, CV_8UC3);	Mat grad_yu(col, row, CV_8UC3);
	Mat angles(col, row, CV_8UC3);	Mat magnit(col, row, CV_8UC3);
	//
	for (int y = 0; y < col; y++){
		for (int x = 0; x < row; x++){
			for (int z = 0; z < 3; z++){
				if (x == 0){
					grad_xr.data[y*source.rows*ch + x*ch + z] = (int)source.data[y*source.rows*ch + (x + 1)*ch + z];
					grad_yu.data[y*source.rows*ch + x*ch + z] = -(int)source.data[y*source.rows*ch + (x + 1)*ch + z];
				}
				else if (x == row - 1){
					grad_xr.data[y*source.rows*ch + x*ch + z] = -(int)source.data[y*source.rows*ch + (x - 1)*ch + z];
					grad_yu.data[y*source.rows*ch + x*ch + z] = (int)source.data[y*source.rows*ch + (x - 1)*ch + z];
				}
				else{
					grad_xr.data[y*source.rows*ch + x*ch + z] = (int)source.data[y*source.rows*ch + (x + 1)*ch + z] - (int)source.data[y*source.rows*ch + (x - 1)*ch + z];
					grad_yu.data[y*source.rows*ch + x*ch + z] = -(int)source.data[y*source.rows*ch + (x + 1)*ch + z] + (int)source.data[y*source.rows*ch + (x - 1)*ch + z];
				}
			}
		}
	}

	for (int y = 0; y < col; y++){
		for (int x = 0; x < row; x++){
			for (int z = 0; z < ch; z++){
				angles.data[y*source.rows*ch + x*ch + z] = atan2(grad_xr.data[y*source.rows*ch + x*ch + z], grad_yu.data[y*source.rows*ch + x*ch + z]);
				magnit.data[y*source.rows*ch + x*ch + z] = grad_xr.data[y*source.rows*ch + x*ch + z] * grad_xr.data[y*source.rows*ch + x*ch + z] +
					grad_yu.data[y*source.rows*ch + x*ch + z] * grad_yu.data[y*source.rows*ch + x*ch + z];
				magnit.data[y*source.rows*ch + x*ch + z] = pow(magnit.data[y*source.rows*ch + x*ch + z], 5);
			}
		}
	}

	for (int y = 0; y < nwin_y - 1; y++){
		for (int x = 0; x < nwin_x - 1; x++){
			cont++;


			bin = 0;
			// H2 = zeros(b,1);
			for (int x = 0; x < 9; x++)
				H2[x] = 0;
			// 2 for loop
			for (ang_lim = PI + 2 * PI / B; ang_lim < 2; ang_lim += PI) {
				bin++;
				for (int x = 0; x < K; x++){
					if (v_angle[x] < ang_lim){
						v_angle[x] = 100;
						H2[bin] += v_magnit[x];
					}
				}
			}
			// H2 = H2/(norm(H2) +0.01);
			for (int x = 0; x < 9; x++)
				norm_ += H2[x];
			norm_ /= 9;
			// H((cont-1)*B*1 : cont*B,1) = H2;
			for (int x = 0; x < 9; x++){
				H2[x] /= (norm_ + 0.01);
			} norm_ = 0;


		}
	}

}

void GrayScale(Mat source){	//����̹����� ����� �帳�ϴ�! �ٵ� ä�μ��� �״���Դϴ�.

	int ch = source.channels();
	int sum = 0;
	for (int y = 0; y < source.cols; y++){
		for (int x = 0; x < source.rows; x++){
			sum = 0;
			for (int c = 0; c < ch; c++){
				sum += (int)source.data[y*source.rows*ch + x*ch + c];
			}
			for (int c = 0; c < ch; c++){
				source.data[y*source.rows*ch + x*ch + c] = (unsigned char)(sum/3);
			}
		}
	}
}

void EdgeDetection(Mat source, int threshold_){	//ä�μ��� ������� ���� ���ؼ��� �ص帳�ϴ�!
	Mat buffer;
	int mask[4][3][3]{
		{ { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } },
		{ { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } },
		{ { 2, 1, 0 }, { 1, 0, -1 }, { 0, -1, -2 } },
		{ { 0, 1, 2 }, { -1, 0, 1 }, { -2, -1, 0 } }
	};
	int sum = 0;
	int threshold = threshold_;
	int width = source.cols;
	int height = source.rows;
	int channel = source.channels();
	buffer = source.clone();

	for (int m = 0; m < 4; m++){
		for (int y = 0; y < height - 2; y++){
			for (int x = 0; x < width - 2; x++){
				sum = 0;
				for (int my = 0; my < 3; my++){
					for (int mx = 0; mx < 3; mx++){
						for (int c = 0; c < channel; c++){
							sum += buffer.data[(y + my)*width*channel + (x + mx)*channel + c] * mask[m][mx][my];
						}
					}
				}
				sum /= channel;
				if (sum > threshold || -sum > threshold){
					for (int c = 0; c < channel; c++){
						source.data[y*width*channel + x*channel + c] = 255;
					}
				}
			}
		}
	}
}

int main() {
	//Image �ε� (��� �ϵ��ڵ� �Ǿ�����)
	src = imread("C:\\Users\\CIEN\\Desktop\\car.jpg", CV_LOAD_IMAGE_COLOR);

	temp = src.clone();				//���纻�� ���� (�׳� = �����ڷ� ������ ��� �ּҸ� ������.)
	GrayScale(temp);
	temp.convertTo(temp, CV_8UC1);	//�̹����� ä���� �ٲ�
	EdgeDetection(temp, 128);

	//�̹��� ��� ���� TEST
	double alpha = 1.0;//1.0~3.0
	int beta = 50;//0~100

	//Image �ε� �Ǿ����� Ȯ��
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
			//�ȼ� ���ٹ�
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