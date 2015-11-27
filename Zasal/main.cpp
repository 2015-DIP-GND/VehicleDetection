#include <iostream>
#include <string>
#include <Windows.h>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include<opencv\cv.h>
#include<opencv\highgui.h>
#include<math.h>

#define COLOR_NON -1
#define COLOR_WHITE 0
#define COLOR_RED 1

#define MATCH_SIZE_X 80
#define MATCH_SIZE_Y 60

#define PI 3.14159

using namespace cv;
using namespace std;

Mat compareImage, src, dst,temp;

int *hist;

/*
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
			//	for (int x = 0; x < K; x++){
					//if (v_angle[x] < ang_lim){
					//	v_angle[x] = 100;
					//	H2[bin] += v_magnit[x];
					//}
				//}
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

*/

Mat GrayScale(Mat source){		//흑백이미지로 만듬. 채널도 1로 바뀜.
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

	Mat result(source.rows, source.cols, CV_8UC1);
	for (int y = 0; y < source.cols; y++){
		for (int x = 0; x < source.rows; x++){
			result.data[y*source.rows + x] = source.data[y*source.rows*ch + x*ch];
		}
	}
	return result;
}

Mat Clipping(Mat source, int sizeX, int sizeY, int x, int y){
	if (source.channels() != 1){
		cout << "Clipping Error - channel must be 1" << endl;
	}
	int width = source.cols;
	Mat result(sizeY, sizeX, CV_8UC1);
	for (int i = y; i < y + sizeY; i++){
		for (int j = x; j < x + sizeX; j++){
			result.data[(i - y)*sizeX + (j - x)] = source.data[i*width + j];
		}
	}
	return result;
}

void Binarization(Mat source, int threshold){	//채널 1개의 이미지에 대해서 이진화(0 또는 255)
	if (source.channels() != 1){
		cout << "binarization - this it not a grayscale image" << endl;
		return;
	}
	int width = source.cols;
	int height = source.rows;
	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			if (source.data[y*width + x] > threshold){
				source.data[y*width + x] = 255;
			}
			else{
				source.data[y*width + x] = 0;
			}
		}
	}
}

void EdgeDetection(Mat source, int threshold_, int colorType){	//채널수에 상관없이 엣지를 디텍션함. 엣지를 colorType 색으로 표시함
	//colorType : 0-흰색 1-빨강
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
	if (channel == 1){
		colorType = COLOR_WHITE;
	}
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
					switch (colorType){
					case COLOR_WHITE:
						for (int c = 0; c < channel; c++){
							source.data[y*width*channel + x*channel + c] = 255;
						}
						break;
					case COLOR_RED:
						source.data[y*width*channel + x*channel + 0] = 0;
						source.data[y*width*channel + x*channel + 1] = 0;
						source.data[y*width*channel + x*channel + 2] = 255;
						break;
					}
				}
				else{
					for (int c = 0; c < channel; c++){
						source.data[y*width*channel + x*channel + c] = 0;
					}
				}
			}
		}
	}
}

Mat Resizing(Mat source, int sizeX , int sizeY){
	int width = source.cols;
	int height = source.rows;
	float blockSizeX = source.cols / (float)sizeX;		//블럭의 사이즈
	float blockSizeY = source.rows / (float)sizeY;		//블럭의 사이즈
	int channel = source.channels();
	if (source.channels() != 1){
		cout << "Fail to Resizing - channel must be 1" << endl;
		return source;
	}

	int rowSum = 0;
	int colSum = 0;
	int rowCount = 0;
	int colCount = 0;
	int limitX = 0;
	int limitY = 0;
	int cumulateX = 0;
	int cumulateY = 0;

	Mat result(sizeY, sizeX, CV_8UC1);

	for (int y = 0; y < sizeY; y++){
		limitY = (blockSizeY * (y + 1));
		for (int x = 0; x < sizeX; x++){
			limitX = (blockSizeX * (x + 1));
			colSum = 0;
			colCount = 0;
			for (cumulateY = (int)(blockSizeY * y); cumulateY < limitY ; cumulateY++){
				rowSum = 0;
				rowCount = 0;
				for (cumulateX = (int)(blockSizeX * x); cumulateX < limitX; cumulateX++){
					rowSum += source.data[cumulateY * width + cumulateX];
					rowCount++;
				}
				colSum += rowSum / rowCount;
				colCount++;
			}
			result.data[y*sizeX + x] = colSum / colCount;
		}
	}
	return result;
}

Mat FeatureCatch(Mat source){
	source = GrayScale(source);
	//EdgeDetection(source, 128, COLOR_RED);
	//source = Simplification(source, MATCH_SIZE, COLOR_RED);
	source = Resizing(source, MATCH_SIZE_X, MATCH_SIZE_Y);
	return source;
}

int GetSimilarity4(Mat sample, Mat feature){
	if (sample.cols != feature.cols){
		cout << "Image not matched" << endl;
		return -1;
	}
	if (sample.rows != feature.rows){
		cout << "Image not matched" << endl;
		return -1;
	}

	int width = sample.cols;
	int height = sample.rows;
	int difference = 0;
	int rmse = 0;
	for (int y = 0; y < height - 1; y++){
		for (int x = 0; x < width - 1; x++){
			for (int i = 0; i < 2; i++){
				for (int j = 0; j < 2; j++){
					if (feature.data[(y + i)*width + (x + j)] != 0){
						difference = feature.data[(y + i)*width + (x + j)] - sample.data[y*width + x];
						rmse -= sqrt((double)(difference*difference));
					}
				}
			}
		}
	}
	return rmse / 10;
}

int GetSimilarity4(Mat sample, Mat feature, Mat antiFeature){
	if (sample.cols != feature.cols || sample.cols != antiFeature.cols){
		cout << "Image not matched" << endl;
		return -1;
	}
	if (sample.rows != feature.rows || sample.rows != antiFeature.rows){
		cout << "Image not matched" << endl;
		return -1;
	}

	int width = sample.cols;
	int height = sample.rows;
	int difference = 0;
	int rmse = 0;
	for (int y = 0; y < height - 1; y++){
		for (int x = 0; x < width - 1; x++){
			for (int i = 0; i < 2; i++){
				for (int j = 0; j < 2; j++){
					if (feature.data[(y + i)*width + (x + j)] != 255){
						difference = feature.data[(y + i)*width + (x + j)] - sample.data[y*width + x];
						rmse -= sqrt((double)(difference*difference));
						difference = antiFeature.data[(y + i)*width + (x + j)] - sample.data[y*width + x];
						rmse += sqrt((double)(difference*difference));
					}
				}
			}
		}
	}
	return rmse;


}

int GetRMSE4(Mat source1, Mat source2){		//root mean square error
	if (source1.cols != source2.cols){
		cout << "Image not matched" << endl;
		return -1;
	}
	if (source1.rows != source2.rows){
		cout << "Image not matched" << endl;
		return -1;
	}
	int width = source1.cols;
	int height = source1.rows;
	int difference = 0;
	int rmse = 0;
	for (int y = 0; y < height - 1; y++){
		for (int x = 0; x < width - 1; x++){
			for (int i = 0; i < 2; i++){
				for (int j = 0; j < 2; j++){
					difference = source1.data[(y + i)*width + (x + j)] - source2.data[y*width + x];
					rmse += sqrt((double)(difference*difference));
				}
			}
		}
	}
	return rmse;
}

void AddFeature(Mat edgeFeatureImage, Mat intFeatureImage, Mat antiIntFeatureImage, Mat sample, int* sampleCount){
	int intensitySum = 0;
	int reverseSum = 0;

	Mat edgeSample = sample.clone();
	sample = GrayScale(sample);
	Mat grayBuffer = sample.clone();

	sample = Resizing(sample, MATCH_SIZE_X, MATCH_SIZE_Y);
	for (int y = 0; y < MATCH_SIZE_Y; y++){
		for (int x = 0; x < MATCH_SIZE_X; x++){
			intensitySum = intFeatureImage.data[y*MATCH_SIZE_X + x] * (*sampleCount) + sample.data[y*MATCH_SIZE_X + x];
			reverseSum = antiIntFeatureImage.data[y*MATCH_SIZE_X + x] * (*sampleCount) + 255 - sample.data[y*MATCH_SIZE_X + x];
			intFeatureImage.data[y*MATCH_SIZE_X + x] = intensitySum / ((*sampleCount) + 1);
			antiIntFeatureImage.data[y*MATCH_SIZE_X + x] = reverseSum / ((*sampleCount) + 1);
		}
	}

	EdgeDetection(edgeSample, 128, COLOR_RED);
	int width = edgeSample.cols;
	int height = edgeSample.rows;
	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			if (edgeSample.data[y*width * 3 + x * 3 + 0] == 0 && edgeSample.data[y*width * 3 + x * 3 + 1] == 0 && edgeSample.data[y*width * 3 + x * 3 + 2] == 255){
				grayBuffer.data[y*width + x] = 255;
			}
			else{
				grayBuffer.data[y*width + x] = 0;
			}
		}
	}
	grayBuffer = Resizing(grayBuffer, MATCH_SIZE_X, MATCH_SIZE_Y);

	intensitySum = 0;
	for (int y = 0; y < MATCH_SIZE_Y; y++){
		for (int x = 0; x < MATCH_SIZE_X; x++){
			intensitySum = edgeFeatureImage.data[y*MATCH_SIZE_X + x] * (*sampleCount) + grayBuffer.data[y*MATCH_SIZE_X + x];
			edgeFeatureImage.data[y*MATCH_SIZE_X + x] = intensitySum / ((*sampleCount) + 1);
		}
	}
	(*sampleCount)++;
}

int main() {
	Mat intFeatureOfCar(MATCH_SIZE_Y, MATCH_SIZE_X, CV_8UC1);
	Mat antiIntFeatureOfCar(MATCH_SIZE_Y, MATCH_SIZE_X, CV_8UC1);
	Mat edgeFeatureOfCar(MATCH_SIZE_Y, MATCH_SIZE_X, CV_8UC1);
	int sampleCount = 0;
	int* scPtr = &sampleCount;
	//featureOfCar = Resizing(GrayScale(imread("C:\\Users\\CIEN\\Desktop\\carData\\car.jpg", CV_LOAD_IMAGE_COLOR)),MATCH_SIZE);

	string filePath = "..\\database\\carData\\uzLeft\\";
	string fileName = "uzLeft01.jpg";
	for (int i = 0; i < 19; i++) {
		AddFeature(edgeFeatureOfCar, intFeatureOfCar, antiIntFeatureOfCar, imread(filePath + fileName, CV_LOAD_IMAGE_COLOR), scPtr);
		fileName[7]++;
		if (fileName[7] > '9') {
			fileName[7] -= 10;
			fileName[6] ++;
		}
	}

	cout << "학습 데이터 구축" << endl;


	compareImage = GrayScale(imread("..\\samples\\Car\\10.jpg", CV_LOAD_IMAGE_COLOR));

	int matchCount = 0;
	int clipX = MATCH_SIZE_X;
	int clipY = MATCH_SIZE_Y;
	Mat compareClip(clipX, clipY, CV_8UC1);
	float similarity = 0;
	int width = compareImage.cols;
	int height = compareImage.rows;
	for (int y = 0; y < height - clipY; y+=3){
		for (int x = 0; x < width - clipX; x += 3){
			compareClip = Clipping(compareImage, clipX, clipY, x, y);

			similarity = GetSimilarity4(compareClip, intFeatureOfCar, antiIntFeatureOfCar);
			EdgeDetection(compareClip, 128, COLOR_WHITE);
			similarity += GetSimilarity4(compareClip, edgeFeatureOfCar);
			similarity /= (MATCH_SIZE_X * MATCH_SIZE_Y);
			cout << x << "," << y << " 유사도 : " << similarity << endl;
			if (similarity > 150){
				cvNamedWindow(x + "a" + y, CV_WINDOW_AUTOSIZE);
				imshow(x + "a" + y, compareClip);
				cout << "자동차 입니다!" << endl;
				matchCount++;

				//x += 10;
				//y += 10;
				if (y >= height - clipY){
					break;
				}
			}
		}
		if (matchCount > 10){
			break;
		}
	}
	
	cvNamedWindow("feature",CV_WINDOW_AUTOSIZE);
	cvNamedWindow("compare",CV_WINDOW_AUTOSIZE);
	cvNamedWindow("edge", CV_WINDOW_AUTOSIZE);

	imshow("feature", intFeatureOfCar);
	imshow("compare", compareImage);
	imshow("edge", edgeFeatureOfCar);

	char ch = waitKey();	// 무한 대기


	return 0;
}