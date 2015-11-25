#include <iostream>
#include <Windows.h>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include<opencv\cv.h>
#include<opencv\highgui.h>
#include<math.h>

#define COLOR_NON -1
#define COLOR_WHITE 0
#define COLOR_RED 1

#define MATCH_SIZE 80

using namespace cv;
using namespace std;

Mat src, compareImage;

Mat GrayScale(Mat source){	//흑백이미지로 만들어 드립니다! 근데 채널수도 1로 바뀝니다.
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

void Binarization(Mat source){
	int threshold = 128;
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

void EdgeDetection(Mat source, int threshold_, int colorType){	//채널수에 상관없이 엣지 디텍션을 해드립니다!
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
			}
		}
	}
}

Mat Resizing(Mat source, int size){
	int width = source.cols;
	int height = source.rows;
	int blockSizeX = source.cols / (size - 1);		//블럭의 사이즈
	int blockSizeY = source.rows / (size - 1);		//블럭의 사이즈
	int channel = source.channels();
	if (source.channels() != 1){
		cout << "Fail to Resizing - channel must be 1" << endl;
		return source;
	}

	int rowSum = 0;
	int colSum = 0;
	int rowCount = 0;
	int colCount = 0;

	Mat result(size, size, CV_8UC1);

	for (int y = 0; y < size; y++){
		for (int x = 0; x < size; x++){
			colSum = 0;
			colCount = 0;
			for (int blockY = 0; blockY < blockSizeY && y*blockSizeY + blockY < height; blockY++){
				rowSum = 0;
				rowCount = 0;
				for (int blockX = 0; blockX < blockSizeX && x*blockSizeX + blockX < width; blockX++){
					rowSum += source.data[((y*blockSizeY) + blockY)*width + ((x*blockSizeX) + blockX)];
					rowCount++;
				}
				colSum += rowSum / rowCount;
				colCount++;
			}
			result.data[y*size + x] = colSum / colCount;	
		}
	}
	return result;
}

Mat Simplification(Mat source, int size, int colorType){		//Edge Detection 되었거나 GrayScale된 이미지를 단순화시킴
	int width = source.cols;
	int height = source.rows;
	int blockSizeX = source.cols / size;		//블럭의 사이즈
	int blockSizeY = source.rows / size;		//블럭의 사이즈
	if (blockSizeX == 0 || blockSizeY == 0){
		cout << "source image is too small" << endl;
		return source;
	}
	int channel = source.channels();
	if (channel == 1){
		colorType = COLOR_WHITE;
	}
	Mat result(size,size,CV_8UC1);

	int sum = 0;
	int channelCounter = 0;

	for (int y = 0; y < size; y++){
		for (int x = 0; x < size; x++){
			sum = 0;
			for (int blockY = 0; blockY < blockSizeY; blockY++){
				for (int blockX = 0; blockX < blockSizeX; blockX++){
					channelCounter = 0;
					switch (colorType){
					case COLOR_WHITE:
						for (int c = 0; c < channel; c++){
							if (source.data[((y*blockSizeY) + blockY)*width*channel + ((x*blockSizeX) + blockX)*channel + c] == 255){
								channelCounter++;
							}
						}
						break;
					case COLOR_RED:
						if (source.data[((y*blockSizeY) + blockY)*width*channel + ((x*blockSizeX) + blockX)*channel + 0] == 0){
							channelCounter++;
						}
						if (source.data[((y*blockSizeY) + blockY)*width*channel + ((x*blockSizeX) + blockX)*channel + 1] == 0){
							channelCounter++;
						}
						if (source.data[((y*blockSizeY) + blockY)*width*channel + ((x*blockSizeX) + blockX)*channel + 2] == 255){
							channelCounter++;
						}
						break;
					}
					if (channelCounter == channel){
						sum++;
					}
				}
			}
			result.data[y*size + x] = sum;
		}
	}
	return result;
}

void Equalization(Mat source){		//GrayScale에 대해서만 적용할 수 있음.
	int width = source.cols;
	int height = source.rows;
	int channel = source.channels();
	int lowest = 255;
	int highest = 0;
	int LUT[256];
	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			if (source.data[y*width + x] < lowest){
				lowest = source.data[y*width + x];
			}
			if (source.data[y*width + x] > highest){
				highest = source.data[y*width + x];
			}
		}
	}

	float rate = 255 / ((float)(highest - lowest));
	
	for (int i = 0; i < 256; i++){
		if (i >= lowest && i <= highest){
			LUT[i] = (i - lowest)*rate;
		}
	}

	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			source.data[y*width + x] = LUT[source.data[y*width + x]];
		}
	}
}

Mat FeatureCatch(Mat source){
	source = GrayScale(source);
	//EdgeDetection(source, 128, COLOR_RED);
	//source = Simplification(source, MATCH_SIZE, COLOR_RED);
	source = Resizing(source, MATCH_SIZE);
	Equalization(source);
	return source;
}

int GetRMSE9(Mat source1, Mat source2){		//root mean square error
	if (source1.cols != source2.cols){
		cout << "Image not matched" << endl;
		return -1;
	}
	if (source1.cols != source2.cols){
		cout << "Image not matched" << endl;
		return -1;
	}
	int width = source1.cols;
	int height = source1.rows;
	int difference = 0;
	int rmse = 0;
	for (int y = 0; y < height-2; y++){
		for (int x = 0; x < width-2; x++){
			for (int i = 0; i < 3; i++){
				for (int j = 0; j < 3; j++){
					if (source1.data[(y + i)*width + (x + j)] != 0 && source2.data[y*width + x] != 0){
						difference = source1.data[(y + i)*width + (x + j)] - source2.data[y*width + x];
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
	if (source1.cols != source2.cols){
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
					if (source1.data[(y + i)*width + (x + j)] != 0 && source2.data[y*width + x] != 0){
						difference = source1.data[(y + i)*width + (x + j)] - source2.data[y*width + x];
						rmse += sqrt((double)(difference*difference));
					}
				}
			}
		}
	}
	return rmse;
}

int GetRMSE(Mat source1, Mat source2){		//root mean square error
	if (source1.cols != source2.cols){
		cout << "Image not matched" << endl;
		return -1;
	}
	if (source1.cols != source2.cols){
		cout << "Image not matched" << endl;
		return -1;
	}
	int width = source1.cols;
	int height = source1.rows;
	int difference = 0;
	int rmse = 0;
	for (int y = 0; y < height - 1; y++){
		for (int x = 0; x < width - 1; x++){
			if (source1.data[y * width + x] != 0 && source2.data[y * width + x] != 0){
				difference = source1.data[y * width + x] - source2.data[y*width + x];
				rmse += sqrt((double)(difference*difference));
			}
		}
	}
	return rmse;
}

void AddFeature(Mat featureImage, Mat antiFeatureImage, Mat sample, int* sampleCount){
	int intensitySum = 0;
	int reverseSum = 0;
	sample = Resizing(sample, MATCH_SIZE);
	//Equalization(sample);
	//Binarization(sample);
	for (int y = 0; y < MATCH_SIZE; y++){
		for (int x = 0; x < MATCH_SIZE; x++){
			intensitySum = featureImage.data[y*MATCH_SIZE + x] * (*sampleCount) + sample.data[y*MATCH_SIZE + x];
			reverseSum = antiFeatureImage.data[y*MATCH_SIZE + x] * (*sampleCount) + 255 - sample.data[y*MATCH_SIZE + x];
			featureImage.data[y*MATCH_SIZE + x] = intensitySum / ((*sampleCount) + 1);
			antiFeatureImage.data[y*MATCH_SIZE + x] = reverseSum / ((*sampleCount) + 1);
		}
	}
	(*sampleCount)++;
}

int main() {
	Mat featureOfCar(MATCH_SIZE, MATCH_SIZE, CV_8UC1);
	Mat antiFeatureOfCar(MATCH_SIZE, MATCH_SIZE, CV_8UC1);
	int sampleCount = 0;
	int* scPtr = &sampleCount;
	//featureOfCar = Resizing(GrayScale(imread("C:\\Users\\CIEN\\Desktop\\carData\\car.jpg", CV_LOAD_IMAGE_COLOR)),MATCH_SIZE);
	AddFeature(featureOfCar, antiFeatureOfCar, GrayScale(imread("C:\\Users\\CIEN\\Desktop\\carData\\car.jpg", CV_LOAD_IMAGE_COLOR)), scPtr);
	AddFeature(featureOfCar, antiFeatureOfCar, GrayScale(imread("C:\\Users\\CIEN\\Desktop\\carData\\car2.jpg", CV_LOAD_IMAGE_COLOR)), scPtr);
	AddFeature(featureOfCar, antiFeatureOfCar, GrayScale(imread("C:\\Users\\CIEN\\Desktop\\carData\\car3.jpg", CV_LOAD_IMAGE_COLOR)), scPtr);
	AddFeature(featureOfCar, antiFeatureOfCar, GrayScale(imread("C:\\Users\\CIEN\\Desktop\\carData\\car4.jpg", CV_LOAD_IMAGE_COLOR)), scPtr);
	AddFeature(featureOfCar, antiFeatureOfCar, GrayScale(imread("C:\\Users\\CIEN\\Desktop\\carData\\car5.jpg", CV_LOAD_IMAGE_COLOR)), scPtr);
	AddFeature(featureOfCar, antiFeatureOfCar, GrayScale(imread("C:\\Users\\CIEN\\Desktop\\carData\\car6.jpg", CV_LOAD_IMAGE_COLOR)), scPtr);
	AddFeature(featureOfCar, antiFeatureOfCar, GrayScale(imread("C:\\Users\\CIEN\\Desktop\\carData\\car7.jpg", CV_LOAD_IMAGE_COLOR)), scPtr);
	AddFeature(featureOfCar, antiFeatureOfCar, GrayScale(imread("C:\\Users\\CIEN\\Desktop\\carData\\car8.jpg", CV_LOAD_IMAGE_COLOR)), scPtr);

	compareImage = Resizing(GrayScale(imread("C:\\Users\\CIEN\\Desktop\\not02.jpg", CV_LOAD_IMAGE_COLOR)), MATCH_SIZE);
	Equalization(compareImage);
	//cout << "수치가 낮을수록 유사 : " << GetRMSE(featureOfCar, compareImage) << endl;
	//cout << "수치가 높을수록 유사 : " << GetRMSE(antiFeatureOfCar, compareImage) << endl;
	int similarity = GetRMSE4(antiFeatureOfCar, compareImage) - GetRMSE4(featureOfCar, compareImage);
	cout << "유사도 : " << similarity << endl;
	if (similarity > 1000000){
		cout << "자동차 입니다!" << endl;
	}
	cvNamedWindow("feature",CV_WINDOW_AUTOSIZE);
	cvNamedWindow("compare",CV_WINDOW_AUTOSIZE);
	imshow("feature", featureOfCar);
	imshow("compare", compareImage);

	char ch = waitKey();	// 무한 대기


	return 0;
}