#include <iostream>
#include <string>
#include <Windows.h>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include<opencv\cv.h>
#include<opencv\highgui.h>
#include<math.h>
#include <time.h> 
#define COLOR_NON -1
#define COLOR_WHITE 5
#define COLOR_RED 0
#define COLOR_BLUE 1
#define COLOR_GREEN 2
#define COLOR_MAGENTA 3

#define MATCH_SIZE_X 100
#define MATCH_SIZE_Y 60

#define PI 3.14159

using namespace cv;
using namespace std;

Mat compareImage, src, dst,temp;

int* grad_xr, *grad_yu;
float* angles, *magnit, *angles2, *magnit2, *H;
int hogX, hogY;

int *hist;

int nwin_x = 3, nwin_y = 3;
int B = 9;

void hog(Mat source)
{
	if (grad_xr != NULL) {
		delete(grad_xr);
	}
	if (grad_yu != NULL) {
		delete(grad_yu);
	}
	if (angles != NULL) {
		delete(angles);
	}
	if (magnit != NULL) {
		delete(magnit);
	}
	if (angles2 != NULL) {
		delete(angles2);
	}
	if (magnit2 != NULL) {
		delete(magnit2);
	}


	int ch = source.channels();
	int row = source.rows;
	int col = source.cols;
	hogX = col;
	hogY = row;
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
	H = new float[nwin_x * nwin_y * B];
	float H2[9] {0, };
	float ang_lim = 0;
	grad_xr = new int[row * col * 3];	grad_yu = new int[row * col * 3];
	angles = new float[row * col * 3];	magnit = new float[row * col * 3];
	//
	for (int y = 0; y < row; y++){
		for (int x = 0; x < col; x++){
			for (int z = 0; z < 3; z++){
				if (x == 0){
					grad_xr[y*source.cols*ch + x*ch + z] = source.data[y*source.cols*ch + (x + 1)*ch + z];
				}
				else if (x == col - 1){
					grad_xr[y*source.cols*ch + x*ch + z] = -source.data[y*source.cols*ch + (x - 1)*ch + z];
				}
				else{
					grad_xr[y*source.cols*ch + x*ch + z] = source.data[y*source.cols*ch + (x + 1)*ch + z] - source.data[y*source.cols*ch + (x - 1)*ch + z];
				}
				if (y == 0) {
					grad_yu[y*source.cols*ch + x*ch + z] = -source.data[(y + 1)*source.cols*ch + x*ch + z];
				}
				else if (y == row - 1) {
					grad_yu[y*source.cols*ch + x*ch + z] = source.data[(y - 1)*source.cols*ch + x*ch + z];
				}
				else {
					grad_yu[y*source.cols*ch + x*ch + z] = -source.data[(y + 1)*source.cols*ch + x*ch + z] + source.data[(y - 1)*source.cols*ch + x*ch + z];
				}
			}
		}
	}


	for (int y = 0; y < row; y++){
		for (int x = 0; x < col; x++){
			for (int z = 0; z < ch; z++){
				angles[y*source.cols*ch + x*ch + z] = atan2(grad_xr[y*source.cols*ch + x*ch + z], grad_yu[y*source.cols*ch + x*ch + z]);
				magnit[y*source.cols*ch + x*ch + z] = grad_xr[y*source.cols*ch + x*ch + z] * grad_xr[y*source.cols*ch + x*ch + z] +
				grad_yu[y*source.cols*ch + x*ch + z] * grad_yu[y*source.cols*ch + x*ch + z];
				magnit[y*source.cols*ch + x*ch + z] = sqrt(magnit[y*source.cols*ch + x*ch + z]);
			}
		}
	}
	
	angles2 = new float[step_y * step_x * 12];
	magnit2 = new float[step_y * step_x * 12];
	int K = 0;
	for (int y = 0; y < nwin_y; y++){
		for (int x = 0; x < nwin_x; x++){

			for (int i = y*step_y; i < (y + 2)*step_y; i++) {
				for (int j = x*step_x; j < (x + 2)*step_x; j++) {
					for (int c = 0; c < ch; c++) {
						angles2[(i - (y * step_y)) * 2 * step_x * ch + (j - (x * step_x)) * ch + c] = angles[i * hogX * ch + j * ch + c];
						magnit2[(i - (y * step_y)) * 2 * step_x * ch + (j - (x * step_x)) * ch + c] = magnit[i * hogX * ch + j * ch + c];
					}
				}
			}
			K = step_y * step_x * 12;


			// H2 = zeros(b,1);
			for (int x = 0; x < B; x++)
				H2[x] = 0;
			// 2 for loop
			for (bin = 0, ang_lim = -PI + 2 * PI / B; bin < B; ang_lim += 2 * PI / B, bin++) {
				for (int x = 0; x < K; x++){
					if (angles2[x] < ang_lim){
						angles2[x] = 100;
						H2[bin] += magnit2[x];
					}
				}
			}

			norm_ = 0;
			// H2 = H2/(norm(H2) +0.01);
			for (int i = 0; i < B; i++) {
				norm_ += H2[i] * H2[i];
			}
			norm_ = sqrt(norm_);
			for (int i = 0; i < B; i++){
				H2[i] /= (norm_ + 0.01);
			}
			// H((cont-1)*B+1:cont*B,1)=H2;
			for (int b = 0; b < B; b++) {
				H[y*nwin_x*B + x*B + b] = H2[b];
			}
		}
	}
}

void hogClip(int sizeX, int sizeY, int startX, int startY) {
	int bin = 0;
	int ch = 1;
	int step_x = floor(sizeX / (nwin_x + 1));
	int step_y = floor(sizeY / (nwin_y + 1));
	angles2 = new float[step_y * step_x * 4];
	magnit2 = new float[step_y * step_x * 4];
	float norm_ = 0;
	H = new float[nwin_x * nwin_y * B];
	float H2[9]{ 0, };
	float ang_lim = 0;
	int K = 0;
	for (int y = 0; y < nwin_y; y++) {
		for (int x = 0; x < nwin_x; x++) {
			for (int i = y*step_y; i < (y + 2)*step_y; i++) {
				for (int j = x*step_x; j < (x + 2)*step_x; j++) {
					angles2[(i - (y * step_y)) * 2 * step_x + (j - (x * step_x))] = angles[(i + startX) * hogX + (j + startY)];
					magnit2[(i - (y * step_y)) * 2 * step_x + (j - (x * step_x))] = magnit[(i + startX) * hogX + (j + startY)];
				}
			}
			K = step_y * step_x * 4;


			// H2 = zeros(b,1);
			for (int x = 0; x < B; x++)
				H2[x] = 0;
			// 2 for loop
			for (bin = 0, ang_lim = -PI + 2 * PI / B; bin < B; ang_lim += 2 * PI / B, bin++) {
				for (int x = 0; x < K; x++) {
					if (angles2[x] < ang_lim) {
						angles2[x] = 100;
						H2[bin] += magnit2[x];
					}
				}
			}

			norm_ = 0;
			// H2 = H2/(norm(H2) +0.01);
			for (int i = 0; i < B; i++) {
				norm_ += H2[i] * H2[i];
			}
			norm_ = sqrt(norm_);
			for (int i = 0; i < B; i++) {
				H2[i] /= (norm_ + 0.01);
			}
			// H((cont-1)*B+1:cont*B,1)=H2;
			for (int b = 0; b < B; b++) {
				H[y*nwin_x*B + x*B + b] = H2[b];
				cout << H2[b] << endl;
			}
		}
	}
}

Mat ReverseLR(Mat source) {
	int ch = source.channels();
	int width = source.cols;
	int height = source.rows;
	Mat result;
	if (ch == 1) {
		result = Mat(height, width, CV_8UC1);
	}
	else if (ch == 3){
		result = Mat(height, width, CV_8UC3);
	}
	else {
		cout << "Reverse Error" << endl;
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int c = 0; c < ch; c++) {
				result.data[(height - y - 1)*width*ch + (width - x - 1)*ch + c] = source.data[y*width*ch + x*ch + c];
			}
		}
	}

	return result;
}

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
	int width = source.cols;
	Mat result;
	if (source.channels() == 1) {
		result = Mat(sizeY, sizeX, CV_8UC1);
	}
	else if (source.channels() == 3) {
		result = Mat(sizeY, sizeX, CV_8UC3);
	}
	else {
		cout << "Clipping Error - channel must be 1 or 3" << endl;
		return source;
	}
	int ch = source.channels();
	for (int i = y; i < y + sizeY; i++){
		for (int j = x; j < x + sizeX; j++) {
			for (int c = 0; c < ch; c++) {
				result.data[(i - y)*sizeX*ch + (j - x)*ch + c] = source.data[i*width*ch + j*ch + c];
			}
		}
	}
	return result;
}

float* Clipping(float source[], int width, int height, int ch, int sizeX, int sizeY, int x, int y) {
	float* result = new float[sizeX * sizeY * ch];
	for (int i = y; i < y + sizeY; i++) {
		for (int j = x; j < x + sizeX; j++) {
			for (int c = 0; c < ch; c++) {
				result[(i - y)*sizeX*ch + (j - x)*ch + c] = source[i*width*ch + j*ch + c];
			}
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

float* Resizing(float source[], int width, int height, int channel, int resultX, int resultY) {
	float blockSizeX = width / (float)resultX;		//블럭의 사이즈
	float blockSizeY = height / (float)resultY;		//블럭의 사이즈
	int ch = channel;

	float rowSum[3] = { 0,0,0 };
	float colSum[3] = { 0,0,0 };
	int rowCount = 0;
	int colCount = 0;
	int limitX = 0;
	int limitY = 0;
	int cumulateX = 0;
	int cumulateY = 0;

	float* result;
	result = new float[resultY * resultX *ch];

	for (int y = 0; y < resultY; y++) {
		limitY = (blockSizeY * (y + 1));
		for (int x = 0; x < resultX; x++) {
			limitX = (blockSizeX * (x + 1));
			for (int c = 0; c < ch; c++) {
				colSum[c] = 0;
			}
			colCount = 0;
			for (cumulateY = (int)(blockSizeY * y); cumulateY < limitY; cumulateY++) {
				for (int c = 0; c < ch; c++) {
					rowSum[c] = 0;
				}
				rowCount = 0;
				for (cumulateX = (int)(blockSizeX * x); cumulateX < limitX; cumulateX++) {
					for (int c = 0; c < ch; c++) {
						rowSum[c] += source[cumulateY * width * ch + cumulateX *ch + c];
					}
					rowCount++;
				}
				for (int c = 0; c < ch; c++) {
					colSum[c] += rowSum[c] / rowCount;
				}
				colCount++;
			}
			for (int c = 0; c < ch; c++) {
				result[y*resultX*ch + x*ch + c] = colSum[c] / colCount;
			}
		}
	}
	return result;
}

Mat Resizing(Mat source, int sizeX , int sizeY){
	int width = source.cols;
	int height = source.rows;
	float blockSizeX = source.cols / (float)sizeX;		//블럭의 사이즈
	float blockSizeY = source.rows / (float)sizeY;		//블럭의 사이즈
	int ch = source.channels();

	int rowSum[3] = { 0,0,0 };
	int colSum[3] = { 0,0,0 };
	int rowCount = 0;
	int colCount = 0;
	int limitX = 0;
	int limitY = 0;
	int cumulateX = 0;
	int cumulateY = 0;

	Mat result;
	if (ch == 1) {
		result = Mat(sizeY, sizeX, CV_8UC1);
	}
	else if (ch == 3){
		result = Mat(sizeY, sizeX, CV_8UC3);
	}
	else {
		cout << "Resizing Error - channel is ????" << endl;
		return source;
	}

	for (int y = 0; y < sizeY; y++){
		limitY = (blockSizeY * (y + 1));
		for (int x = 0; x < sizeX; x++){
			limitX = (blockSizeX * (x + 1));
			for (int c = 0; c < ch; c++) {
				colSum[c] = 0;
			}
			colCount = 0;
			for (cumulateY = (int)(blockSizeY * y); cumulateY < limitY ; cumulateY++){
				for (int c = 0; c < ch; c++) {
					rowSum[c] = 0;
				}
				rowCount = 0;
				for (cumulateX = (int)(blockSizeX * x); cumulateX < limitX; cumulateX++){
					for (int c = 0; c < ch; c++) {
						rowSum[c] += source.data[cumulateY * width * ch + cumulateX *ch + c];
					}
					rowCount++;
				}
				for (int c = 0; c < ch; c++) {
					colSum[c] += rowSum[c] / rowCount;
				}
				colCount++;
			}
			for (int c = 0; c < ch; c++) {
				result.data[y*sizeX*ch + x*ch + c] = colSum[c] / colCount;
			}
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

float GetSimilarity(float sample[], float feature[], int width, int height,  int ch) {

	float difference = 0;
	float rmse = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int c = 0; c < ch; c++) {
				difference = feature[y*width*ch + x*ch + c] - sample[y*width*ch + x*ch + c];
				rmse -= sqrt(difference*difference);
			}
		}
	}
	return rmse;
}

float GetSimilarityReverse(float sample[], float feature[], int width, int height, int ch) {

	float difference = 0;
	float rmse = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int c = 0; c < ch; c++) {
				difference = feature[(height - y - 1)*width*ch + (width - x - 1)*ch + c] - sample[y*width*ch + x*ch + c];
				rmse -= sqrt(difference*difference);
			}
		}
	}
	return rmse;
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
	int ch = sample.channels();

	int width = sample.cols;
	int height = sample.rows;
	int difference = 0;
	int rmse = 0;
	for (int y = 0; y < height - 1; y++){
		for (int x = 0; x < width - 1; x++){
			for (int i = 0; i < 2; i++){
				for (int j = 0; j < 2; j++){
					for (int c = 0; c < ch; c++) {
						if (feature.data[(y + i)*width*ch + (x + j)*ch + c] != 0) {
							difference = feature.data[(y + i)*width*ch + (x + j)*ch + c] - sample.data[y*width*ch + x*ch + c];
							rmse -= sqrt((double)(difference*difference));
						}
					}
				}
			}
		}
	}
	return rmse / ch;
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
	int ch = sample.channels();

	int width = sample.cols;
	int height = sample.rows;
	int difference = 0;
	int rmse = 0;
	for (int y = 0; y < height - 1; y++){
		for (int x = 0; x < width - 1; x++){
			for (int i = 0; i < 2; i++){
				for (int j = 0; j < 2; j++){
					for (int c = 0; c < ch; c++) {
						if (feature.data[(y + i)*width + (x + j)] != 255) {
							difference = feature.data[(y + i)*width*ch + (x + j)*ch + c] - sample.data[y*width*ch + x*ch + c];
							rmse -= sqrt((double)(difference*difference));
							difference = antiFeature.data[(y + i)*width*ch + (x + j)*ch + c] - sample.data[y*width*ch + x*ch + c];
							rmse += sqrt((double)(difference*difference));
						}
					}
				}
			}
		}
	}
	return rmse / ch;


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

void AddFeature(Mat intFeatureImage, Mat antiIntFeatureImage, Mat sample, int* sampleCount) { //3 Channel 전용
	int intensitySum = 0;
	int reverseSum = 0;
	int ch = sample.channels();

	sample = Resizing(sample, MATCH_SIZE_X, MATCH_SIZE_Y);
	for (int y = 0; y < MATCH_SIZE_Y; y++) {
		for (int x = 0; x < MATCH_SIZE_X; x++) {
			for (int c = 0; c < ch; c++) {
				intensitySum = intFeatureImage.data[y*MATCH_SIZE_X*ch + x*ch + c] * (*sampleCount) + sample.data[y*MATCH_SIZE_X*ch + x*ch + c];
				reverseSum = antiIntFeatureImage.data[y*MATCH_SIZE_X*ch + x*ch + c] * (*sampleCount) + 255 - sample.data[y*MATCH_SIZE_X*ch + x*ch + c];
				intFeatureImage.data[y*MATCH_SIZE_X*ch + x*ch + c] = intensitySum / ((*sampleCount) + 1);
				antiIntFeatureImage.data[y*MATCH_SIZE_X*ch + x*ch + c] = reverseSum / ((*sampleCount) + 1);
			}
		}
	}
	(*sampleCount)++;
}

void AddFeature(float hogFeat[], float source[], int* sampleCount) {
	float sum = 0;

	for (int y = 0; y < nwin_y; y++) {
		for (int x = 0; x < nwin_x; x++) {
			for (int b = 0; b < B; b++) {
				sum = hogFeat[y*nwin_x*B + x*B + b] * (*sampleCount) + source[y*nwin_x*B + x*B + b];
				hogFeat[y*nwin_x*B + x*B + b] = sum / ((*sampleCount) + 1);
			}
		}
	}
	(*sampleCount)++;
}

void AddFeature(float angleFeat[], float source[],int width, int height, int ch, int* sampleCount){
	float sum = 0;

	source = Resizing(source, width, height, ch, MATCH_SIZE_X, MATCH_SIZE_Y);
	for (int y = 0; y < MATCH_SIZE_Y; y++) {
		for (int x = 0; x < MATCH_SIZE_X; x++) {
			for (int c = 0; c < ch; c++) {
				sum = angleFeat[y*MATCH_SIZE_X*ch + x*ch + c] * (*sampleCount) + source[y*MATCH_SIZE_X*ch + x*ch + c];
				angleFeat[y*MATCH_SIZE_X*ch + x*ch + c] = sum / ((*sampleCount) + 1);
			}
		}
	}
	(*sampleCount)++;
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

void DrawRect(Mat source, int startX, int startY, int sizeX, int sizeY , int color) {
	int ch = source.channels();
	int width = source.cols;
	int height = source.rows;

	for (int y = startY; y < startY + sizeY; y++) {
		for (int x = startX; x < startX + sizeX; x++) {
			if (x == startX || x == startX + sizeX - 1 || y == startY || y == startY + sizeY - 1) {
				switch (color) {
				case COLOR_WHITE:
					source.data[y * width * ch + x * ch + 0] = 255;
					source.data[y * width * ch + x * ch + 1] = 255;
					source.data[y * width * ch + x * ch + 2] = 255;
					break;
				case COLOR_RED:
					source.data[y * width * ch + x * ch + 0] = 0;
					source.data[y * width * ch + x * ch + 1] = 0;
					source.data[y * width * ch + x * ch + 2] = 255;
					break;
				case COLOR_BLUE:
					source.data[y * width * ch + x * ch + 0] = 255;
					source.data[y * width * ch + x * ch + 1] = 0;
					source.data[y * width * ch + x * ch + 2] = 0;
					break;
				case COLOR_GREEN:
					source.data[y * width * ch + x * ch + 0] = 0;
					source.data[y * width * ch + x * ch + 1] = 255;
					source.data[y * width * ch + x * ch + 2] = 0;
					break;
				case COLOR_MAGENTA:
					source.data[y * width * ch + x * ch + 0] = 255;
					source.data[y * width * ch + x * ch + 1] = 0;
					source.data[y * width * ch + x * ch + 2] = 255;
					break;
				default:
					source.data[y * width * ch + x * ch + 0] = 0;
					source.data[y * width * ch + x * ch + 1] = 0;
					source.data[y * width * ch + x * ch + 2] = 0;
				}
			}
		}
	}
}

int main() {
	//Mat intFeatureOfCar3(MATCH_SIZE_Y, MATCH_SIZE_X, CV_8UC3);
	//Mat antiIntFeatureOfCar3(MATCH_SIZE_Y, MATCH_SIZE_X, CV_8UC3);
	clock_t start_time, end_time;
	start_time = clock();
	float* hogFeatureOfCarUZL = new float[nwin_x * nwin_y * B];
	for (int i = 0; i < nwin_x * nwin_y * B; i++) {
		hogFeatureOfCarUZL[i] = 0;
	}
	float* hogFeatureOfCarUZR = new float[nwin_x * nwin_y * B];
	for (int i = 0; i < nwin_x * nwin_y * B; i++) {
		hogFeatureOfCarUZR[i] = 0;
	}
	float* hogFeatureOfCarB = new float[nwin_x * nwin_y * B];
	for (int i = 0; i < nwin_x * nwin_y * B; i++) {
		hogFeatureOfCarB[i] = 0;
	}
	float* hogFeatureOfCarF = new float[nwin_x * nwin_y * B];
	for (int i = 0; i < nwin_x * nwin_y * B; i++) {
		hogFeatureOfCarF[i] = 0;
	}

	float* hogFeatureOfMan = new float[nwin_x * nwin_y * B];
	for (int i = 0; i < nwin_x * nwin_y * B; i++) {
		hogFeatureOfMan[i] = 0;
	}

	Mat intFeatureOfCar(MATCH_SIZE_Y, MATCH_SIZE_X, CV_8UC1);
	Mat antiIntFeatureOfCar(MATCH_SIZE_Y, MATCH_SIZE_X, CV_8UC1);
	Mat edgeFeatureOfCar(MATCH_SIZE_Y, MATCH_SIZE_X, CV_8UC1);
	int sampleCount = 0;
	int* scPtr = &sampleCount;
	//featureOfCar = Resizing(GrayScale(imread("C:\\Users\\CIEN\\Desktop\\carData\\car.jpg", CV_LOAD_IMAGE_COLOR)),MATCH_SIZE);

	cout << "DataBase 로드중.." << endl;
	string filePath = "..\\database\\carData\\uzLeft\\";
	string fileName = "uzLeft01.jpg";
	for (int i = 1; i < 20; i++) {
		hog(GrayScale(imread(filePath + fileName, CV_LOAD_IMAGE_COLOR)));
		AddFeature(hogFeatureOfCarUZL, H, scPtr);
		hog(GrayScale(ReverseLR(imread(filePath + fileName, CV_LOAD_IMAGE_COLOR))));
		AddFeature(hogFeatureOfCarUZR, H, scPtr);
		fileName[7]++;
		if (fileName[7] > '9') {
			fileName[7] -= 10;
			fileName[6] ++;
		}
	}

	sampleCount = 0;
	filePath = "..\\database\\carData\\back\\";
	fileName = "carB01.jpg";
	for (int i = 1; i < 11; i++) {
		hog(GrayScale(imread(filePath + fileName, CV_LOAD_IMAGE_COLOR)));
		AddFeature(hogFeatureOfCarB, H, scPtr);
		fileName[5]++;
		if (fileName[5] > '9') {
			fileName[5] -= 10;
			fileName[4] ++;
		}
	}

	sampleCount = 0;
	filePath = "..\\database\\carData\\front\\";
	fileName = "f01.jpg";
	for (int i = 1; i < 6; i++) {
		hog(GrayScale(imread(filePath + fileName, CV_LOAD_IMAGE_COLOR)));
		AddFeature(hogFeatureOfCarF, H, scPtr);
		fileName[2]++;
		if (fileName[2] > '9') {
			fileName[2] -= 10;
			fileName[1] ++;
		}
	}

	sampleCount = 0;
	filePath = "..\\database\\manData\\";
	fileName = "m01.jpg";
	for (int i = 1; i < 8; i++) {
		hog(GrayScale(imread(filePath + fileName, CV_LOAD_IMAGE_COLOR)));
		AddFeature(hogFeatureOfMan, H, scPtr);
		fileName[2]++;
		if (fileName[2] > '9') {
			fileName[2] -= 10;
			fileName[1] ++;
		}
	}

	cout << "학습 데이터 로드 완료" << endl;
	end_time = clock();                   // End_Time
	printf("Time : %f\n", ((double)(end_time - start_time)) / CLOCKS_PER_SEC);
	system("pause");


	start_time = clock();
	/********************************************************************************
	************************
	************************  이 곳에서 비교할 이미지를 불러옵니다.
	************************
	*********************************************************************************/
	//compareImage = imread("..\\samples\\Car\\11.jpg", CV_LOAD_IMAGE_COLOR);
	compareImage = imread("..\\samples\\car+people\\21m_31s_189008u.jpg", CV_LOAD_IMAGE_COLOR);
	//compareImage = imread("..\\samples\\human\\20120204__05dcastrw_400.jpg", CV_LOAD_IMAGE_COLOR);
	if (compareImage.cols > 500) {
		float rate = compareImage.rows / (float)compareImage.cols;
		compareImage = Resizing(compareImage, 500, (int)(500 * rate));
	}
	if (compareImage.rows > 500) {
		float rate = compareImage.cols / (float)compareImage.rows;
		compareImage = Resizing(compareImage, (int)(500 * rate), 500);
	}

	Mat compareOriginal = compareImage.clone();
	compareImage = GrayScale(compareImage);
	
	int matchCount = 0;
	int clipX = 100;
	int clipY = 60;
	int incX = clipX / 10;
	int incY = clipY / 10;
	Mat compareClip;
	float similarity = 0;
	for (int i = 1; i < 6; i++) {
		clipX = i * 120;
		clipY = i * 70;
		incX = clipX / 10;
		incY = clipY / 10;
		for (int y = 0; y < compareImage.rows - clipY; y += incY) {
			for (int x = 0; x < compareImage.cols - clipX; x += incX) {
				compareClip = Clipping(compareImage, clipX, clipY, x, y);
				hog(compareClip);
				//hogClip(clipX, clipY, x, y);

				similarity = GetSimilarity(H, hogFeatureOfCarUZL, nwin_x, nwin_y, B);
				similarity *= 1 + (i - 1) * 0.3;
				cout << x << "," << y << " 유사도 : " << similarity << endl;
				if (similarity > -6.2 ) {
					//cvNamedWindow(x + "a" + y, CV_WINDOW_AUTOSIZE);
					//imshow(x + "a" + y, compareClip);
					cout << "자동차 입니다!" << endl;
					matchCount++;
					DrawRect(compareOriginal, x, y, clipX, clipY, COLOR_RED);
				}

				similarity = GetSimilarity(H, hogFeatureOfCarUZR, nwin_x, nwin_y, B);
				similarity *= 1 + (i - 1) * 0.3;
				cout << x << "," << y << " 유사도 : " << similarity << endl;
				if (similarity > -6.2) {
					//cvNamedWindow(x + "a" + y, CV_WINDOW_AUTOSIZE);
					//imshow(x + "a" + y, compareClip);
					cout << "자동차 입니다!" << endl;
					matchCount++;

					DrawRect(compareOriginal, x, y, clipX, clipY, -1);
				}
			}
			if (matchCount > 10) {
				//break;
			}
		}
	}

	for (int i = 1; i < 6; i++) {
		clipX = i * 80;
		clipY = i * 60;
		incX = clipX / 10;
		incY = clipY / 10;
		for (int y = 0; y < compareImage.rows - clipY; y += incY) {
			for (int x = 0; x < compareImage.cols - clipX; x += incX) {
				compareClip = Clipping(compareImage, clipX, clipY, x, y);
				hog(compareClip);
				//hogClip(clipX, clipY, x, y);

				similarity = GetSimilarity(H, hogFeatureOfCarB, nwin_x, nwin_y, B);
				similarity *= 1 + (i - 1) * 0.1;
				cout << x << "," << y << " 유사도 : " << similarity << endl;
				if (similarity > -5) {
					//cvNamedWindow(x + "a" + y, CV_WINDOW_AUTOSIZE);
					//imshow(x + "a" + y, compareClip);
					cout << "자동차 입니다!" << endl;
					matchCount++;

					DrawRect(compareOriginal, x, y, clipX, clipY, COLOR_GREEN);

				}

				similarity = GetSimilarity(H, hogFeatureOfCarF, nwin_x, nwin_y, B);
				similarity *= 1 + (i - 1) * 0.1;
				cout << x << "," << y << " 유사도 : " << similarity << endl;
				if (similarity > -5) {
					//cvNamedWindow(x + "a" + y, CV_WINDOW_AUTOSIZE);
					//imshow(x + "a" + y, compareClip);
					cout << "자동차 입니다!" << endl;
					matchCount++;

					DrawRect(compareOriginal, x, y, clipX, clipY, COLOR_BLUE);

				}
			}
		}
	}

	for (int i = 1; i < 8; i++) {
		clipX = i * 30;
		clipY = i * 80;
		incX = clipX / 10;
		incY = clipY / 10;
		for (int y = 0; y < compareImage.rows - clipY; y += incY) {
			for (int x = 0; x < compareImage.cols - clipX; x += incX) {
				compareClip = Clipping(compareImage, clipX, clipY, x, y);
				hog(compareClip);
				//hogClip(clipX, clipY, x, y);

				similarity = GetSimilarity(H, hogFeatureOfMan, nwin_x, nwin_y, B);
				similarity *= 1 + (i - 1) * 0.1;
				cout << x << "," << y << " 유사도 : " << similarity << endl;
				if (similarity > -6.5) {
					//cvNamedWindow(x + "a" + y, CV_WINDOW_AUTOSIZE);
					//imshow(x + "a" + y, compareClip);
					cout << "사람 입니다!" << endl;
					matchCount++;

					DrawRect(compareOriginal, x, y, clipX, clipY, COLOR_MAGENTA);

				}
			}
		}
	}

	
	cvNamedWindow("compare",CV_WINDOW_AUTOSIZE);
	cvNamedWindow("compareO", CV_WINDOW_AUTOSIZE);
	//cvNamedWindow("edge", CV_WINDOW_AUTOSIZE);

	imshow("compare", compareImage);
	imshow("compareO", compareOriginal);
	//imshow("edge", edgeFeatureOfCar);
	end_time = clock();                   // End_Time
	printf("Time : %f\n", ((double)(end_time - start_time)) / CLOCKS_PER_SEC);
	char ch = waitKey();	// 무한 대기


	return 0;
}