
// videoDlg.cpp : ʵ���ļ�
//

#include "stdafx.h"
#include "video.h"
#include "videoDlg.h"
#include "afxdialogex.h"

#include <vector>
#include "cv.h"  
#include "highgui.h"  
#include <math.h>  
#include <stdio.h>  
#include "opencv2/legacy/legacy.hpp"  
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "CvvImage.h"

#define ROW_SUM 9 
#define ROW_SUM_ratio 11 
#define UNKNOWN_FLOW_THRESH 1e9  

#define uchar unsigned char
using namespace std;
using namespace cv;

// ����ȫ��ͼ��
IplImage *src_global; // ����IplImageָ�����src     



#ifdef _DEBUG
#define new DEBUG_NEW
#endif


//��Ÿ�������ֵ  
uchar Median(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5,
	uchar n6, uchar n7, uchar n8, uchar n9) {
	uchar arr[9];
	arr[0] = n1;
	arr[1] = n2;
	arr[2] = n3;
	arr[3] = n4;
	arr[4] = n5;
	arr[5] = n6;
	arr[6] = n7;
	arr[7] = n8;
	arr[8] = n9;
	for (int gap = 9 / 2; gap > 0; gap /= 2)//ϣ������  
		for (int i = gap; i < 9; ++i)
			for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
				swap(arr[j], arr[j + gap]);
	return arr[4];//������ֵ  
}

// //��ֵ�˲���(3*3)
//void AverFiltering(const Mat &src, Mat &dst)
//{
//	if (!src.data) return;
//	//at�������ص�  
//	for (int i = 1; i<src.rows; ++i)
//		for (int j = 1; j < src.cols; ++j) {
//			if ((i - 1 >= 0) && (j - 1) >= 0 && (i + 1)<src.rows && (j + 1)<src.cols) {//��Ե�����д���  
//				dst.at<Vec3b>(i, j)[0] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i - 1, j - 1)[0] + src.at<Vec3b>(i - 1, j)[0] + src.at<Vec3b>(i, j - 1)[0] +
//					src.at<Vec3b>(i - 1, j + 1)[0] + src.at<Vec3b>(i + 1, j - 1)[0] + src.at<Vec3b>(i + 1, j + 1)[0] + src.at<Vec3b>(i, j + 1)[0] +
//					src.at<Vec3b>(i + 1, j)[0]) / 9;
//				dst.at<Vec3b>(i, j)[1] = (src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i - 1, j - 1)[1] + src.at<Vec3b>(i - 1, j)[1] + src.at<Vec3b>(i, j - 1)[1] +
//					src.at<Vec3b>(i - 1, j + 1)[1] + src.at<Vec3b>(i + 1, j - 1)[1] + src.at<Vec3b>(i + 1, j + 1)[1] + src.at<Vec3b>(i, j + 1)[1] +
//					src.at<Vec3b>(i + 1, j)[1]) / 9;
//				dst.at<Vec3b>(i, j)[2] = (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i - 1, j - 1)[2] + src.at<Vec3b>(i - 1, j)[2] + src.at<Vec3b>(i, j - 1)[2] +
//					src.at<Vec3b>(i - 1, j + 1)[2] + src.at<Vec3b>(i + 1, j - 1)[2] + src.at<Vec3b>(i + 1, j + 1)[2] + src.at<Vec3b>(i, j + 1)[2] +
//					src.at<Vec3b>(i + 1, j)[2]) / 9;
//			}
//			else {//��Ե��ֵ  
//				dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
//				dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
//				dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
//			}
//		}
//}
//
//��ֵ�˲�������3*3��
void MedianFlitering(const Mat &src, Mat &dst) {
	if (!src.data)return;
	Mat _dst(src.size(), src.type());
	for (int i = 0; i<src.rows; ++i)
		for (int j = 0; j < src.cols; ++j) {
			if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
				_dst.at<Vec3b>(i, j)[0] = Median(src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i + 1, j + 1)[0],
					src.at<Vec3b>(i + 1, j)[0], src.at<Vec3b>(i, j + 1)[0], src.at<Vec3b>(i + 1, j - 1)[0],
					src.at<Vec3b>(i - 1, j + 1)[0], src.at<Vec3b>(i - 1, j)[0], src.at<Vec3b>(i, j - 1)[0],
					src.at<Vec3b>(i - 1, j - 1)[0]);
				_dst.at<Vec3b>(i, j)[1] = Median(src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i + 1, j + 1)[1],
					src.at<Vec3b>(i + 1, j)[1], src.at<Vec3b>(i, j + 1)[1], src.at<Vec3b>(i + 1, j - 1)[1],
					src.at<Vec3b>(i - 1, j + 1)[1], src.at<Vec3b>(i - 1, j)[1], src.at<Vec3b>(i, j - 1)[1],
					src.at<Vec3b>(i - 1, j - 1)[1]);
				_dst.at<Vec3b>(i, j)[2] = Median(src.at<Vec3b>(i, j)[2], src.at<Vec3b>(i + 1, j + 1)[2],
					src.at<Vec3b>(i + 1, j)[2], src.at<Vec3b>(i, j + 1)[2], src.at<Vec3b>(i + 1, j - 1)[2],
					src.at<Vec3b>(i - 1, j + 1)[2], src.at<Vec3b>(i - 1, j)[2], src.at<Vec3b>(i, j - 1)[2],
					src.at<Vec3b>(i - 1, j - 1)[2]);
			}
			else
				_dst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
		}
	_dst.copyTo(dst);//����  
}

// �ص�����
void thresh_callback(int, void*)
{
	Mat src, src_gray;
	int thresh = 100;
	int max_thresh = 255;
	RNG rng(12345);

	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// canny ��Ե���
	Canny(src_gray, canny_output, thresh, thresh * 2, 3);

	// Ѱ������
	findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

	// ��������
	for (size_t i = 0; i< contours.size(); i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, (int)i, color, 2, 8, hierarchy, 0, Point());
	}

	namedWindow("Contours", WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
}

int imageAdjust(Mat &src, Mat &dst,
	double low_in, double high_in,
	double low_out, double high_out,
	double gamma) {
	if (low_in < 0 && low_in >1 && high_in <0 && high_in >1
		&& low_out < 0 && low_out > 1 && high_out < 0 && high_out > 1
		&& low_out > high_out)
		return -1;

	double low2 = low_in * 255;
	double high2 = high_in * 255;
	double bottom2 = low_out * 255;
	double top2 = high_out * 255;
	double err_in = high2 - low2;
	double err_out = top2 - bottom2;

	int x, y;
	double val;
	uchar* p;

	//���ȱ任    
	for (y = 0; y < src.rows; y++)
	{
		p = src.ptr<uchar>(y);  //��ָ��    
		for (x = 0; x < src.cols; x++)
		{
			val = pow((p[x] - low2) / err_in, gamma)*err_out + bottom2;
			if (val > 255)  val = 255;
			if (val < 0)    val = 0;

			dst.at<uchar>(y, x) = (uchar)val;
		}
	}
}

int flag_str2_int = 0;  //ѡȡ��Ƶ��־λ

int DECTOR_temp_globvar = 0;
double sum_dector = 0;
double sum_sum_dection = 0;
Mat prevgray, gray, flow, cflow, frame_mat; 
Mat motion2color; 

Mat hist_equalization_BGR_dlphay(Mat input)
{
	Mat output;
	uchar *dataIn = (uchar *)input.ptr<uchar>(0);//input��ͷָ�룬ָ���0�е�0�����أ���Ϊ������
	uchar *dataOut = (uchar *)output.ptr<uchar>(0);
	const int ncols = input.cols;//��ʾ����ͼ���ж�����
	const int nrows = input.rows;//��ʾ����ͼ���ж�����
	int nchannel = input.channels();//ͨ������һ����3
	int pixnum = ncols*nrows;
	int vData[765] = { 0 };//����R+G+B��ʱ��255+255+255������Ϊ765�����ȼ�
	double vRate[765] = { 0 };
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++)
		{
			vData[dataIn[i*ncols*nchannel + j*nchannel + 0]
				+ dataIn[i*ncols*nchannel + j*nchannel + 1]
				+ dataIn[i*ncols*nchannel + j*nchannel + 2]]++;//��Ӧ�����ȼ�ͳ��
		}
	}
	for (int i = 0; i < 764; i++)
	{
		for (int j = 0; j < i; j++)
		{
			vRate[i] += (double)vData[j] / (double)pixnum;//���
		}
	}
	for (int i = 0; i < 764; i++)
	{
		vData[i] = (int)(vRate[i] * 764 + 0.5);//���й�һ������
	}
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++)
		{
			int amplification = vData[dataIn[i*ncols*nchannel + j*nchannel + 0]
				+ dataIn[i*ncols*nchannel + j*nchannel + 1]
				+ dataIn[i*ncols*nchannel + j*nchannel + 2]] -
				(dataIn[i*ncols*nchannel + j*nchannel + 0]
					+ dataIn[i*ncols*nchannel + j*nchannel + 1]
					+ dataIn[i*ncols*nchannel + j*nchannel + 2]);//�ñ任���ֵ��ȥԭֵ�ĵ����ȼ��Ĳ�ֵ����3�����ÿ��ͨ��Ӧ���仯��ֵ
			int b = dataIn[i*ncols*nchannel + j*nchannel + 0] + amplification / 3 + 0.5;
			int g = dataIn[i*ncols*nchannel + j*nchannel + 1] + amplification / 3 + 0.5;
			int r = dataIn[i*ncols*nchannel + j*nchannel + 2] + amplification / 3 + 0.5;
			if (b > 255) b = 255;//����Խλ�ж�
			if (g > 255) g = 255;
			if (r > 255) r = 255;
			if (r < 0) r = 0;//����Խλ�ж�
			if (g < 0) g = 0;
			if (b < 0) b = 0;
			dataOut[i*ncols*nchannel + j*nchannel + 0] = b;
			dataOut[i*ncols*nchannel + j*nchannel + 1] = g;
			dataOut[i*ncols*nchannel + j*nchannel + 2] = r;
		}
	}
	return output;
}
// ��ʱ����
void delay_msec(int msec)  
{   
    clock_t now = clock();  
    while(clock()-now < msec);  
}  


void makecolorwheel(vector<Scalar> &colorwheel) 
{ 
    int RY = 15;  
    int YG = 6;  
    int GC = 4;  
    int CB = 11;  
    int BM = 13;  
    int MR = 6;  
  
    int i;  
  
    for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255,       255*i/RY,     0));  
    for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255-255*i/YG, 255,       0));  
    for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0,         255,      255*i/GC));  
    for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0,         255-255*i/CB, 255));  
    for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255*i/BM,      0,        255));  
    for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255,       0,        255-255*i/MR));  
} 

void motionToColor(Mat flow, Mat &color)  
{  
    if (color.empty())  
        color.create(flow.rows, flow.cols, CV_8UC3);  
  
    static vector<Scalar> colorwheel; //Scalar r,g,b  
    if (colorwheel.empty())  
        makecolorwheel(colorwheel);  
  
    // determine motion range:  
    float maxrad = -1;  
  
    // Find max flow to normalize fx and fy  
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
            float fx = flow_at_point[0];  
            float fy = flow_at_point[1];  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
                continue;  
            float rad = sqrt(fx * fx + fy * fy);  
            maxrad = maxrad > rad ? maxrad : rad;  
        }  
    }  
  
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            uchar *data = color.data + color.step[0] * i + color.step[1] * j;  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
  
            float fx = flow_at_point[0] / maxrad;  
            float fy = flow_at_point[1] / maxrad;  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
            {  
                data[0] = data[1] = data[2] = 0;  
                continue;  
            }  
            float rad = sqrt(fx * fx + fy * fy);  
  
            float angle = atan2(-fy, -fx) / CV_PI;  
            float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);  
            int k0 = (int)fk;  
            int k1 = (k0 + 1) % colorwheel.size();  
            float f = fk - k0;  
            //f = 0; // uncomment to see original color wheel  
  
            for (int b = 0; b < 3; b++)   
            {  
                float col0 = colorwheel[k0][b] / 255.0;  
                float col1 = colorwheel[k1][b] / 255.0;  
                float col = (1 - f) * col0 + f * col1;  
                if (rad <= 1)  
                    col = 1 - rad * (1 - col); // increase saturation with radius  
                else  
                    col *= .75; // out of range  
                data[2 - b] = (int)(255.0 * col);  
            }  
        }  
    }  
} 
void AverFiltering(const Mat &src, Mat &dst) 
{
	if (!src.data) return;
	//at�������ص�  
	for (int i = 1; i<src.rows; ++i)
		for (int j = 1; j < src.cols; ++j) {
			if ((i - 1 >= 0) && (j - 1) >= 0 && (i + 1)<src.rows && (j + 1)<src.cols) {//��Ե�����д���  
				dst.at<Vec3b>(i, j)[0] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i - 1, j - 1)[0] + src.at<Vec3b>(i - 1, j)[0] + src.at<Vec3b>(i, j - 1)[0] +
					src.at<Vec3b>(i - 1, j + 1)[0] + src.at<Vec3b>(i + 1, j - 1)[0] + src.at<Vec3b>(i + 1, j + 1)[0] + src.at<Vec3b>(i, j + 1)[0] +
					src.at<Vec3b>(i + 1, j)[0]) / 9;
				dst.at<Vec3b>(i, j)[1] = (src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i - 1, j - 1)[1] + src.at<Vec3b>(i - 1, j)[1] + src.at<Vec3b>(i, j - 1)[1] +
					src.at<Vec3b>(i - 1, j + 1)[1] + src.at<Vec3b>(i + 1, j - 1)[1] + src.at<Vec3b>(i + 1, j + 1)[1] + src.at<Vec3b>(i, j + 1)[1] +
					src.at<Vec3b>(i + 1, j)[1]) / 9;
				dst.at<Vec3b>(i, j)[2] = (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i - 1, j - 1)[2] + src.at<Vec3b>(i - 1, j)[2] + src.at<Vec3b>(i, j - 1)[2] +
					src.at<Vec3b>(i - 1, j + 1)[2] + src.at<Vec3b>(i + 1, j - 1)[2] + src.at<Vec3b>(i + 1, j + 1)[2] + src.at<Vec3b>(i, j + 1)[2] +
					src.at<Vec3b>(i + 1, j)[2]) / 9;
			}
			else {//��Ե��ֵ  
				dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
				dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
				dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
			}
		}
}

////��Ÿ�������ֵ  
//uchar Median(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5,
//	uchar n6, uchar n7, uchar n8, uchar n9) {
//	uchar arr[9];
//	arr[0] = n1;
//	arr[1] = n2;
//	arr[2] = n3;
//	arr[3] = n4;
//	arr[4] = n5;
//	arr[5] = n6;
//	arr[6] = n7;
//	arr[7] = n8;
//	arr[8] = n9;
//	for (int gap = 9 / 2; gap > 0; gap /= 2)//ϣ������  
//		for (int i = gap; i < 9; ++i)
//			for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
//				swap(arr[j], arr[j + gap]);
//	return arr[4];//������ֵ  
//}
////��ֵ�˲�������3*3��
//void MedianFlitering(const Mat &src, Mat &dst) {
//	if (!src.data)return;
//	Mat _dst(src.size(), src.type());
//	for (int i = 0; i<src.rows; ++i)
//		for (int j = 0; j < src.cols; ++j) {
//			if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
//				_dst.at<Vec3b>(i, j)[0] = Median(src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i + 1, j + 1)[0],
//					src.at<Vec3b>(i + 1, j)[0], src.at<Vec3b>(i, j + 1)[0], src.at<Vec3b>(i + 1, j - 1)[0],
//					src.at<Vec3b>(i - 1, j + 1)[0], src.at<Vec3b>(i - 1, j)[0], src.at<Vec3b>(i, j - 1)[0],
//					src.at<Vec3b>(i - 1, j - 1)[0]);
//				_dst.at<Vec3b>(i, j)[1] = Median(src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i + 1, j + 1)[1],
//					src.at<Vec3b>(i + 1, j)[1], src.at<Vec3b>(i, j + 1)[1], src.at<Vec3b>(i + 1, j - 1)[1],
//					src.at<Vec3b>(i - 1, j + 1)[1], src.at<Vec3b>(i - 1, j)[1], src.at<Vec3b>(i, j - 1)[1],
//					src.at<Vec3b>(i - 1, j - 1)[1]);
//				_dst.at<Vec3b>(i, j)[2] = Median(src.at<Vec3b>(i, j)[2], src.at<Vec3b>(i + 1, j + 1)[2],
//					src.at<Vec3b>(i + 1, j)[2], src.at<Vec3b>(i, j + 1)[2], src.at<Vec3b>(i + 1, j - 1)[2],
//					src.at<Vec3b>(i - 1, j + 1)[2], src.at<Vec3b>(i - 1, j)[2], src.at<Vec3b>(i, j - 1)[2],
//					src.at<Vec3b>(i - 1, j - 1)[2]);
//			}
//			else
//				_dst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
//		}
//	_dst.copyTo(dst);//����  
//}
Mat hist_equalization_GRAY_dlphay_test(Mat input_image)
{
	const int grayMax = 255;
	vector<vector<int>> graylevel(grayMax + 1);

	cout << graylevel.size() << endl;
	Mat output_image;
	input_image.copyTo(output_image);

	if (!input_image.data)
	{
		return output_image;
	}
	for (int i = 0; i < input_image.rows - 1; i++)
	{
		uchar* ptr = input_image.ptr<uchar>(i);  // �����Ϊһ��һ�е�����  ������ptr
		for (int j = 0; j < input_image.cols - 1; j++)
		{
			int x = ptr[j];
			graylevel[x].push_back(0);//����ط�д�Ĳ��ã������ά����ֻ��Ϊ�˼�¼ÿһ���Ҷ�ֵ�����ظ���  
		}
	}
	for (int i = 0; i < output_image.rows - 1; i++)
	{
		uchar* imgptr = output_image.ptr<uchar>(i);
		uchar* imageptr = input_image.ptr<uchar>(i);
		for (int j = 0; j < output_image.cols - 1; j++)
		{
			int sumpiexl = 0;
			for (int k = 0; k < imageptr[j]; k++)
			{
				sumpiexl = graylevel[k].size() + sumpiexl;
			}
			imgptr[j] = (grayMax*sumpiexl / (input_image.rows*input_image.cols));
		}
	}
	return output_image;
}


unsigned char GetThreshold_part(Mat  Img, int h, int w){
	int i, j;
	double histgram[256] = { 0 };
	double sum = 0;
	double omiga[256] = { 0 };
	double max = 0;
	unsigned char max_seq = 0;
	unsigned char max_seq_i;
	for (i = 0; i<h; i++){	//��ȡֱ��ͼ
		for (j = 0; j<w; j++){
			histgram[Img.at<uchar>(i, j)]++;
		}
	}
	for (i = 0; i<256; i++){ //omiga��Ϊ�ۼ�ֱ��ͼ
		sum = sum + histgram[i];
		omiga[i] = (double)sum / ((double)h*(double)w);
	}
	sum = 0;
	for (i = 0; i<256; i++){  //p�ۼ�ֱ��ͼǶ��任
		sum = (double)sum + (double)histgram[i] * i / ((double)h*(double)w);
		histgram[i] = sum;
	}
	for (i = 0; i<256; i++){
		if (omiga[i] != 0 && (1 - omiga[i]) != 0){ //��ֹ��ĸΪ0
			omiga[i] = ((double)histgram[255] * omiga[i] - (double)histgram[i])*((double)histgram[255] * omiga[i] - (double)histgram[i]) / (omiga[i] * (1 - omiga[i]));
		}
		else{
			omiga[i] = 0;
		}
	}
	for (max_seq_i = 0; max_seq_i<255; max_seq_i++){
		if (omiga[max_seq_i]>max){
			max = omiga[max_seq_i];
			max_seq = max_seq_i;
		}
	}
	return (unsigned char)((double)(max_seq + 1)*0.9);
}
void MedFilterBin(Mat *img_input, Mat *img_output ,int h,int w)
{
	int i, j;
	unsigned char num;
	for (i = 1; i<(h-1); i++)
	{
		for (j = 1; j < (w-1); j++){
			num = 0;
			if (img_input->at<uchar>(i - 1, j - 1) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i - 1, j ) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i - 1, j + 1) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i, j - 1) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i, j) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i, j + 1) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i + 1, j - 1) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i + 1, j ) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i + 1, j+1) == 255)
			{
				num++;
			}
			if (num > 4)
			{
				img_output->at<uchar>(i, j) = 255;
			}
			else{
				img_output->at<uchar>(i, j) = 0;
			}
		}
	}	
}

CString AAAAA = NULL;
class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// �Ի�������
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

// ʵ��
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CvideoDlg �Ի���



CvideoDlg::CvideoDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CvideoDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CvideoDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_EDIT4, m_folderDir);
	DDX_Control(pDX, IDC_EDIT5, m_edit5);
	DDX_Control(pDX, IDC_STATIC_FLOW, m_flow);
}

BEGIN_MESSAGE_MAP(CvideoDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CvideoDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CvideoDlg::OnBnClickedButton2)
	//ON_BN_CLICKED(IDC_BUTTON5, &CvideoDlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CvideoDlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON3, &CvideoDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CvideoDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON7, &CvideoDlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON9, &CvideoDlg::OnBnClickedButton9)
	ON_BN_CLICKED(IDC_BUTTON10, &CvideoDlg::OnBnClickedButton10)
	ON_BN_CLICKED(IDC_BUTTON8, &CvideoDlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON11, &CvideoDlg::OnBnClickedButton11)
	ON_EN_CHANGE(IDC_EDIT5, &CvideoDlg::OnEnChangeEdit5)
	ON_EN_CHANGE(IDC_EDIT4, &CvideoDlg::OnEnChangeEdit4)
END_MESSAGE_MAP()


// CvideoDlg ��Ϣ�������

BOOL CvideoDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// ��������...���˵�����ӵ�ϵͳ�˵��С�

	// IDM_ABOUTBOX ������ϵͳ���Χ�ڡ�
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// ���ô˶Ի����ͼ�ꡣ��Ӧ�ó��������ڲ��ǶԻ���ʱ����ܽ��Զ�
	//  ִ�д˲���
	SetIcon(m_hIcon, TRUE);			// ���ô�ͼ��
	SetIcon(m_hIcon, FALSE);		// ����Сͼ��

	// TODO: �ڴ���Ӷ���ĳ�ʼ������

	return TRUE;  // ���ǽ��������õ��ؼ������򷵻� TRUE
}

void CvideoDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// �����Ի��������С����ť������Ҫ����Ĵ���
//  �����Ƹ�ͼ�ꡣ����ʹ���ĵ�/��ͼģ�͵� MFC Ӧ�ó���
//  �⽫�ɿ���Զ���ɡ�

void CvideoDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // ���ڻ��Ƶ��豸������

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// ʹͼ���ڹ����������о���
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// ����ͼ��
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//���û��϶���С������ʱϵͳ���ô˺���ȡ�ù��
//��ʾ��
HCURSOR CvideoDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}





void CvideoDlg::OnBnClickedButton1()
{
 
	CDC *pDC = GetDlgItem(IDC_STATIC)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	 HDC hdc= pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
   CRect rect;
   // ������
   GetDlgItem(IDC_STATIC)->GetClientRect(&rect); //��ȡbox1�ͻ���
   CvvImage cimg;
   IplImage *src = NULL; // ����IplImageָ�����src   

   CEdit* MESSAGE_XXX;
   CString str_xxx;
   MESSAGE_XXX = (CEdit*)GetDlgItem(IDC_EDIT5);
   MESSAGE_XXX->GetWindowTextW(str_xxx);

   int CCCC = _ttoi(str_xxx);
   if(CCCC == 1)    src = cvLoadImage("tidddmg.jpg",-1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    
   if(CCCC == 2)    src = cvLoadImage("tiddmg.jpg", -1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp 
   src_global = src;
   cimg.CopyOf(src,src->nChannels);

   cimg.DrawToHDC(hdc,&rect);
   //���ͼ��
   ReleaseDC( pDC );
   cimg.Destroy();
   //����
}


void CvideoDlg::OnBnClickedButton2()
{
	CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	HDC hdc = pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	CRect rect;
	// ������
	GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //��ȡbox1�ͻ���
	CvvImage cimg;
	IplImage *src; // ����IplImageָ�����src     
	//src = cvLoadImage("tidddmg.jpg", -1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    
	
	src = src_global;
	Mat src_mat;
	Mat dst_mat;

	src_mat = cvarrToMat(src);
	//dst_mat = cvarrToMat(src);

	//imageAdjust(src_mat, dst_mat, 0,0.8, 0.8, 0.8, 0.8);
	
	int height = src_mat.rows;
	int width = src_mat.cols;
	dst_mat = Mat::zeros(src_mat.size(), src_mat.type());
	float alpha = 1.8;
	float beta = 20;

	//Mat m1;  
	//src.convertTo(m1,CV_32F);  
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if (src_mat.channels() == 3)
			{
				float b = src_mat.at<Vec3b>(row, col)[0];//blue  
				float g = src_mat.at<Vec3b>(row, col)[1];//green  
				float r = src_mat.at<Vec3b>(row, col)[2];//red  

													 //output  
				dst_mat.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b*alpha + beta);
				dst_mat.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g*alpha + beta);
				dst_mat.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r*alpha + beta);
			}
			else if (src_mat.channels() == 1)
			{
				float gray = src_mat.at<uchar>(row, col);
				dst_mat.at<uchar>(row, col) = saturate_cast<uchar>(gray*alpha + beta);
			}
		}
	}
	IplImage* pBinary = &IplImage(dst_mat);
	
	cimg.CopyOf(pBinary, pBinary->nChannels);

	cimg.DrawToHDC(hdc, &rect);
	//���ͼ��
	ReleaseDC(pDC);
	cimg.Destroy();
	//����
}


void CvideoDlg::OnBnClickedButton6()
{
	CString strFolderPath;  
    BROWSEINFO broInfo = {0};  
    TCHAR szDisplayName[1000] = {0};  
  
    broInfo.hwndOwner = this->m_hWnd;  
    broInfo.pidlRoot = NULL;  
    broInfo.pszDisplayName = szDisplayName;  
    broInfo.lpszTitle = _T("��ѡ�񱣴�·��");  
    broInfo.ulFlags = BIF_USENEWUI | BIF_RETURNONLYFSDIRS;;  
    broInfo.lpfn = NULL;  
    broInfo.lParam = NULL;  
    broInfo.iImage = IDR_MAINFRAME;  
    LPITEMIDLIST pIDList = SHBrowseForFolder(&broInfo);  
    if (pIDList != NULL)    
    {    
        memset(szDisplayName, 0, sizeof(szDisplayName));    
        SHGetPathFromIDList(pIDList, szDisplayName);    
        strFolderPath = szDisplayName;  
		AAAAA = strFolderPath;
        m_folderDir.SetWindowText(strFolderPath);  
		
    } 
	//m_shijirenshu.SetWindowText(AAAAA); 
	   CEdit* pMessage;
       CString str;
       pMessage = (CEdit*) GetDlgItem(IDC_EDIT4);     
       pMessage->GetWindowTextW(str);
       MessageBox(str,_T("��ѡ����ļ�·��"), MB_OK);
}


void CvideoDlg::OnBnClickedButton3()
{
	CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	HDC hdc = pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	CRect rect;
	// ������
	GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //��ȡbox1�ͻ���
	CvvImage cimg;
	IplImage *src; // ����IplImageָ�����src     
	// src = cvLoadImage("C:\\Users\\Administrator\\Desktop\\ҽѧͼ����\\video\\image\\tidddmg.jpg", -1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    

	//src = cvLoadImage("tidddmg.jpg", -1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    
	src = src_global;
	Mat src_mat;
	Mat dst_mat;

	src_mat = cvarrToMat(src);
	//dst_mat = cvarrToMat(src);

	//imageAdjust(src_mat, dst_mat, 0,0.8, 0.8, 0.8, 0.8);

	int height = src_mat.rows;
	int width = src_mat.cols;
	dst_mat = Mat::zeros(src_mat.size(), src_mat.type());
	float alpha = 0.5;
	float beta = 20;

	//Mat m1;  
	//src.convertTo(m1,CV_32F);  
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if (src_mat.channels() == 3)
			{
				float b = src_mat.at<Vec3b>(row, col)[0];//blue  
				float g = src_mat.at<Vec3b>(row, col)[1];//green  
				float r = src_mat.at<Vec3b>(row, col)[2];//red  

														 //output  
				dst_mat.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b*alpha + beta);
				dst_mat.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g*alpha + beta);
				dst_mat.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r*alpha + beta);
			}
			else if (src_mat.channels() == 1)
			{
				float gray = src_mat.at<uchar>(row, col);
				dst_mat.at<uchar>(row, col) = saturate_cast<uchar>(gray*alpha + beta);
			}
		}
	}
	IplImage* pBinary = &IplImage(dst_mat);

	cimg.CopyOf(pBinary, pBinary->nChannels);

	cimg.DrawToHDC(hdc, &rect);
	//���ͼ��
	ReleaseDC(pDC);
	cimg.Destroy();
	//����
}


void CvideoDlg::OnBnClickedButton4()
{
	CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	HDC hdc = pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	CRect rect;
	// ������
	GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //��ȡbox1�ͻ���
	CvvImage cimg;
	IplImage *src; // ����IplImageָ�����src     
	src = src_global;

	Mat src_mat;
	Mat src_gray;
	Mat dst_mat;

	src_mat = cvarrToMat(src);

	cvtColor(src_mat, src_gray, COLOR_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));


	//������Ҫ��ʾ��ͼƬ������������˹�
	IplImage* pBinary = &IplImage(src_gray);

	cimg.CopyOf(pBinary, pBinary->nChannels);

	cimg.DrawToHDC(hdc, &rect);
	//���ͼ��
	ReleaseDC(pDC);
	cimg.Destroy();
	//����
}


void CvideoDlg::OnBnClickedButton7()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	HDC hdc = pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	CRect rect;
	// ������
	GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //��ȡbox1�ͻ���
	CvvImage cimg;
	IplImage *src; // ����IplImageָ�����src     
	src = src_global;

	Mat src_mat;
	Mat src_gray;
	Mat dst_mat;

	src_mat = cvarrToMat(src);

	cvtColor(src_mat, src_gray, COLOR_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));
	cv::threshold(src_gray, dst_mat, 60, 255, cv::THRESH_BINARY_INV);


	//������Ҫ��ʾ��ͼƬ������������˹�
	IplImage* pBinary = &IplImage(dst_mat);

	cimg.CopyOf(pBinary, pBinary->nChannels);

	cimg.DrawToHDC(hdc, &rect);
	//���ͼ��
	ReleaseDC(pDC);
	cimg.Destroy();
	//����
}


void CvideoDlg::OnBnClickedButton9()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	HDC hdc = pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	CRect rect;
	// ������
	GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //��ȡbox1�ͻ���
	CvvImage cimg;
	IplImage *src; // ����IplImageָ�����src     
	src = src_global;

	Mat src_mat;
	Mat src_gray;
	Mat dst_mat;

	src_mat = cvarrToMat(src);

	cvtColor(src_mat, src_gray, COLOR_BGR2GRAY);

	
	MedianFlitering(src_mat, dst_mat);

	//������Ҫ��ʾ��ͼƬ������������˹�
	IplImage* pBinary = &IplImage(dst_mat);

	cimg.CopyOf(pBinary, pBinary->nChannels);

	cimg.DrawToHDC(hdc, &rect);
	//���ͼ��
	ReleaseDC(pDC);
	cimg.Destroy();
}


void CvideoDlg::OnBnClickedButton10()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	HDC hdc = pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	CRect rect;
	// ������
	GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //��ȡbox1�ͻ���
	CvvImage cimg;
	IplImage *src; // ����IplImageָ�����src     
	src = src_global;

	Mat src_mat;
	Mat src_gray;
	Mat dst_mat;

	src_mat = cvarrToMat(src);

	cvtColor(src_mat, src_gray, COLOR_BGR2GRAY);


	MedianFlitering(src_mat, dst_mat);

	//������Ҫ��ʾ��ͼƬ������������˹�
	IplImage* pBinary = &IplImage(dst_mat);

	cimg.CopyOf(pBinary, pBinary->nChannels);

	cimg.DrawToHDC(hdc, &rect);
	//���ͼ��
	ReleaseDC(pDC);
	cimg.Destroy();
}


void CvideoDlg::OnBnClickedButton8()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������



	// ��ά����ͼ��Ķ�ȡ

	char *rawFileName = "Angio.raw";
	//char *rawFileName = "VisMale.raw";
	FILE *fp = NULL;
	int ret = 0, width = 384, height = 512;

	unsigned short *pRawData = (unsigned short *)calloc(width*height, sizeof(unsigned short));
	if (NULL == pRawData)
	{
		printf("Fail to calloc buf\r\n");
		//return -1;
	}

	if (NULL == (fp = fopen(rawFileName, "rb")))
	{
		printf("Fail to read %s.\r\n", rawFileName);
		//return -2;
	}

	ret = fread(pRawData, sizeof(unsigned short)*width*height, 1, fp);
	if (ret != 1)
	{
		printf("Fail to read raw data\r\n");
		//return -3;
	}
	IplImage* pBinary; //�����ʾ���ڴ��
	double ratio_fps = 0;
	double temp_value = 0.009;
	for (int i = 0; i < 1; i++)
	{
		//������ʾ��صı�����
		CDC *pDC = GetDlgItem(IDC_STATIC)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
		HDC hdc = pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
		CRect rect;
		// ������
		GetDlgItem(IDC_STATIC)->GetClientRect(&rect); //��ȡbox1�ͻ���
		CvvImage cimg;

		IplImage *pBayerData = cvCreateImage(cvSize(width, height), 16, 1);
		IplImage *pRgbDataInt16 = cvCreateImage(cvSize(width, height), 16, 3);
		IplImage *pRgbDataInt8 = cvCreateImage(cvSize(width, height), 8, 3);
		memcpy(pBayerData->imageData, (char *)pRawData, width*height * sizeof(unsigned short));
		cvCvtColor(pBayerData, pRgbDataInt16, CV_BayerRG2BGR);

		/*��14bit����ת��Ϊ8bit*/
		cvConvertScale(pRgbDataInt16, pRgbDataInt8, temp_value, 0);

		pBinary = pRgbDataInt8;
		//cvNamedWindow("rgb", 1);
		//cvShowImage("rgb", pRgbDataInt8);
		//cvWaitKey(0);

		//free(pRawData);
		//fclose(fp);
		//cvDestroyWindow("rgb");


		//IplImage *src; // ����IplImageָ�����src     
		//src = cvLoadImage("tidddmg.jpg", -1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp  
		//pBinary = src;
		cimg.CopyOf(pBinary, pBinary->nChannels);
		cimg.DrawToHDC(hdc, &rect);

		//���ͼ��
		ReleaseDC(pDC);
		cimg.Destroy();

		cvReleaseImage(&pBayerData);
		cvReleaseImage(&pRgbDataInt8);
		cvReleaseImage(&pRgbDataInt16);
		temp_value += ratio_fps;
	}
	//������ʾ��صı�����
	CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	HDC hdc = pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	CRect rect;
	// ������
	GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //��ȡbox1�ͻ���
	CvvImage cimg;
	

	IplImage *src; // ����IplImageָ�����src     
	src = cvLoadImage("Angio.jpg", 0); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp  
	pBinary = src;


	cimg.CopyOf(pBinary, pBinary->nChannels);
	cimg.DrawToHDC(hdc, &rect);

	//���ͼ��
	ReleaseDC(pDC);
	cimg.Destroy();

	//����

}


void CvideoDlg::OnBnClickedButton11()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	HDC hdc = pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	CRect rect;
	// ������
	GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //��ȡbox1�ͻ���
	CvvImage cimg;
	int temp[9];
	Mat src, dst1, dst2, dst3; // ����IplImageָ�����src     
	//src = cvLoadImage("C:\\Users\\Administrator\\Desktop\\ҽѧͼ����\\video\\image\\tidddmg.jpg", -1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    
	//Mat srcImage = imread("C:\\Users\\Administrator\\Desktop\\ҽѧͼ����\\video\\image\\tidddmg.jpg");
	
	src = src_global;
	//Mat 
	//cvtColor(srcImage, src, CV_RGB2GRAY);
	int col = src.cols, row = src.rows;

	//for (int i = 1; i < col - 1; i++)
	//{
	//	for (int j = 1; j < row - 1; j++)
	//	{
	//		temp[0] = src.at<uchar>(j, i);
	//		temp[1] = src.at<uchar>(j, i + 1);
	//		temp[2] = src.at<uchar>(j + 1, i);
	//		temp[3] = src.at<uchar>(j + 1, i + 1);
	//		dst1.at<uchar>(j, i) = (int)sqrt((temp[0] - temp[3])*(temp[0] - temp[3])
	//			+ (temp[1] - temp[2])*(temp[1] - temp[2]));
	//	}
	//}

	Canny(src, dst1, 250, 150, 3);

	IplImage temp_image = dst1;
	IplImage *qqqq;
	qqqq = cvCloneImage(&temp_image);
	
	cimg.CopyOf(qqqq, qqqq->nChannels);

	cimg.DrawToHDC(hdc, &rect);
	//���ͼ��
	ReleaseDC(pDC);
	cimg.Destroy();
	//����
}


void CvideoDlg::OnEnChangeEdit5()
{
	// TODO:  ����ÿؼ��� RICHEDIT �ؼ���������
	// ���ʹ�֪ͨ��������д CDialogEx::OnInitDialog()
	// ���������� CRichEditCtrl().SetEventMask()��
	// ͬʱ�� ENM_CHANGE ��־�������㵽�����С�

	// TODO:  �ڴ���ӿؼ�֪ͨ����������
}


void CvideoDlg::OnEnChangeEdit4()
{
	// TODO:  ����ÿؼ��� RICHEDIT �ؼ���������
	// ���ʹ�֪ͨ��������д CDialogEx::OnInitDialog()
	// ���������� CRichEditCtrl().SetEventMask()��
	// ͬʱ�� ENM_CHANGE ��־�������㵽�����С�

	// TODO:  �ڴ���ӿؼ�֪ͨ����������
}
