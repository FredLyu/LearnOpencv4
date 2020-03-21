#include<opencv2\opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

//创建图像
void CreateImage(Mat source)
{
	//克隆
	Mat clone = source.clone();
	imshow("Clone", clone);

	//拷贝
	Mat copy;
	source.copyTo(copy);
	imshow("copy", copy);

	//赋值 指向同一个地址
	Mat source1 = source;
	imshow("Source", source1);

	//使用特殊函数
	Mat black = Mat::zeros(Size(512, 512), CV_8UC3);
	Mat white = Mat::ones(Size(512, 512), CV_8UC3);

	imshow("Black", black);
	imshow("White", white);

	//创建矩阵
	Mat karnel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	imshow("Karnel", karnel);
}

//直接读取图像像素值
void ReadImagePixel(Mat source)
{
	Mat result = Mat::zeros(source.size(), source.type());
	int width = source.cols;
	int height = source.rows;
	int chanel = source.channels();

	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width; w++)
		{
			if (chanel == 3)
			{
				Vec3b bgr = source.at<Vec3b>(h, w);
				bgr[0] = 255 - bgr[0];
				bgr[1] = 255 - bgr[1];
				bgr[2] = 255 - bgr[2];
				result.at<Vec3b>(h, w) = bgr;
			}
			else if (chanel == 1)
			{
				int gray = source.at<uchar>(h, w);
				result.at<uchar>(h, w) = 255 - gray;
			}
		}
	}

	imshow("Result", result);
}

//i通过指针读取图像像素
void ReadImagePixelbyPtr(Mat source)
{
	Mat result = Mat::zeros(source.size(), source.type());
	int width = source.cols;
	int height = source.rows;
	int chanels = source.channels();

	int blue = 0, green = 0, red = 0;
	int gray = 0;

	for (int row = 0; row < height; row++)
	{
		uchar* currRow = source.ptr<uchar>(row);
		uchar* resultRow = result.ptr<uchar>(row);
		for (int col = 0; col < width; col++)
		{
			if (chanels == 3)
			{
				blue = *currRow++;
				green = *currRow++;
				red = *currRow++;

				*resultRow++ = blue;
				*resultRow++ = green;
				*resultRow++ = red;
			}
			else if (chanels == 1)
			{
				gray = *currRow++;
				*resultRow++ = gray;
			}
		}
	}

	imshow("ResultPtr", result);
}

//图像加减乘除操作
void ImageOption(Mat source1, Mat source2)
{
	Mat result = Mat(source1.size(), source1.type());

	add(source1, source2, result);
	imshow("Add", result);

	subtract(source1, source2, result);
	imshow("sub", result);

	multiply(source1, source2, result);
	imshow("mul", result);

	divide(source1, source2, result);
	imshow("divide", result);

	//限制像素值大小
	int b1 = 0, g1 = 0, r1 = 0;
	int b2 = 0, g2 = 0, r2 = 0;
	for (int row = 0; row < source1.rows; row++)
	{
		for (int col = 0; col < source1.cols; col++)
		{
			b1 = source1.at<Vec3b>(row, col)[0];
			g1 = source1.at<Vec3b>(row, col)[1];
			r1 = source1.at<Vec3b>(row, col)[2];

			b2 = source2.at<Vec3b>(row, col)[0];
			g2 = source2.at<Vec3b>(row, col)[1];
			r2 = source2.at<Vec3b>(row, col)[2];

			result.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b1 + b2);
			result.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g1 + g2);
			result.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r1 + r2);
		}
	}
	imshow("Saturate", result);
}

//自定义查找表
void CustomColorMap(Mat source)
{
	int lut[256];
	for (int i = 0; i < 256; i++)
	{
		if (i < 85)
			lut[i] = 0;
		else if (i < 170)
			lut[i] = 128;
		else
			lut[i] = 255;
	}

	Mat result = Mat(source.size(), source.type());
	for (int row = 0; row < source.rows; row++)
	{
		for (int col = 0; col < source.cols; col++)
		{
			Vec3b bgr = source.at<Vec3b>(row, col);
			result.at<Vec3b>(row, col)[0] = lut[bgr[0]];
			result.at<Vec3b>(row, col)[1] = lut[bgr[1]];
			result.at<Vec3b>(row, col)[2] = lut[bgr[2]];
		}
	}
	imshow("Custom", result);
}

int main()
{
	//加载图像,加载3通道的BGR彩色图
	Mat input1 = imread("D:\\Photos\\DSC_7570.JPG", IMREAD_COLOR);

	//默认读取彩色图像
	Mat input2 = imread("D:\\Photos\\DSC_7574.JPG");

	if (input1.empty())
	{
		cerr << "Can'n load image...pls check url of image!" << endl;
		return -1;
	}

	//命名一个窗口，大小自动根据图片确定
	namedWindow("Input", WINDOW_AUTOSIZE);

	//显示图像,窗口名称和图片
	imshow("Input", input1);

	//将图片转换为灰度图
	//Mat gray;
	//cvtColor(input, gray, COLOR_BGR2GRAY);
	//imshow("Gray", gray);

	//将图片转换为HSV
	//Mat hsv;
	//cvtColor(input, hsv, COLOR_BGR2HSV);
	//imshow("HSV", hsv);

	//图像保存
	//imwrite("D:/HSV.jpg", hsv);

	//创建图像
	//CreateImage(input);

	//图像像素读取
	//ReadImagePixel(input1);
	//ReadImagePixelbyPtr(input1);

	//图像加减乘除操作
	//ImageOption(input1, input2);

	//自定义查找表
	CustomColorMap(input1);
	//查找表API
	Mat map = Mat(input1.size(), input1.type());
	applyColorMap(input1, map, COLORMAP_HOT);
	imshow("ColorMap", map);

	//等待任意按键按下
	waitKey(0);

	return 0;
}