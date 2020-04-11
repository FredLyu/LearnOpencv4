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

//图像位操作
void ImageOptionbyBit(Mat source)
{
	//创建图像
	Rect rect = Rect(0, 0, 200, 200);
	Mat src1 = Mat::zeros(Size(400, 400), CV_8UC3);
	rect.x = 100;
	rect.y = 100;
	src1(rect) = Scalar(0, 255, 255);
	imshow("src1", src1);

	Mat src2 = Mat::zeros(Size(400, 400), CV_8UC3);
	rect.x = 150;
	rect.y = 150;
	src2(rect) = Scalar(0, 0, 255);
	imshow("src2", src2);

	//逻辑操作
	Mat dst1, dst2, dst3;
	//与
	bitwise_and(src1, src2, dst1);
	//或
	bitwise_or(src1, src2, dst2);
	//异或
	bitwise_xor(src1, src2, dst3);
	imshow("and", dst1);
	imshow("or", dst2);
	imshow("xor", dst3);

	//取反操作，多用于二值图像分析
	Mat dst;
	bitwise_not(source, dst);
	imshow("Not", dst);
}

//图像通道分离与合并，用于直方图
void ImageSplitAndMerge(Mat source)
{
	vector<Mat> mats;
	Mat dst;

	//分离
	split(source, mats);

	//单通道就是灰色
	imshow("B", mats[0]);
	imshow("G", mats[1]);
	imshow("R", mats[2]);

	//合并
	mats[0] = Scalar(0);
	merge(mats, dst);
	imshow("merge", dst);
}

//图像色彩空间转换
void ExchangImageColorSpace(Mat source)
{
	Mat hsv, yuv, ycrcb;
	cvtColor(source, hsv, COLOR_BGR2HSV);
	cvtColor(source, yuv, COLOR_BGR2YUV);
	cvtColor(source, ycrcb, COLOR_BGR2YCrCb);

	imshow("HSV", hsv);
	imshow("YUV", yuv);
	imshow("YCRCB", ycrcb);

	//去指定区域的颜色
	Mat mask;
	//第二、第三个参数表示想要得到的颜色的取值最大和最小，第四个为输出的想要的区域
	inRange(hsv, Scalar(35, 43, 46), Scalar(99, 255, 255), mask);
	imshow("mask", mask);

	Mat dst;
	bitwise_and(source, source, dst, mask);
	imshow("dst", dst);
}

//图像像素值统计
void CountImagePixels(Mat source)
{
	double minVal, maxVal;
	Point minLoc, maxLoc;
	Mat gray;
	cvtColor(source, gray, COLOR_BGR2GRAY);
	//只可用于单通道的图像
	minMaxLoc(gray, &minVal, &maxVal, &minLoc, &maxLoc);
	printf("min：%.2f, max: %.2f\n", minVal, maxVal);
	printf("min loc:(%d, %d)\n", minLoc.x, minLoc.y);
	printf("max loc:(%d, %d)\n", maxLoc.x, maxLoc.y);

	//均值与方差
	Mat means, stddev;
	meanStdDev(source, means, stddev);
	printf("blue channel->> mean: %.2f, stddev: %.2f\n", means.at<double>(0, 0), stddev.at<double>(0, 0));
	printf("green channel->> mean: %.2f, stddev: %.2f\n", means.at<double>(1, 0), stddev.at<double>(1, 0));
	printf("red channel->> mean: %.2f, stddev: %.2f\n", means.at<double>(2, 0), stddev.at<double>(2, 0));
}

//图像插值
void ImageInterpolation(Mat source)
{
	//常用于几何变换、透视变换、插值计算新像素resize

	int w = source.cols;
	int h = source.rows;
	float fx = 0, fy = 0;
	Mat dst = Mat(source.size(), source.type());
	resize(source, dst, Size(w / 2, h / 2), fx, fy, INTER_NEAREST);
	imshow("INTER_NEAREST", dst);

	resize(source, dst, Size(w / 2, h / 2), fx, fy, INTER_LINEAR);
	imshow("INTER_LINEAR", dst);

	resize(source, dst, Size(w / 2, h / 2), fx, fy, INTER_CUBIC);
	imshow("INTER_CUBIC", dst);

	resize(source, dst, Size(w / 2, h / 2), fx, fy, INTER_LANCZOS4);
	imshow("INTER_LANCZOS4", dst);
}

//像素归一化
void NormalizeImage(Mat source)
{
	Mat gray;
	cvtColor(source, gray, COLOR_BGR2GRAY);

	//转换为32为浮点数
	gray.convertTo(gray, CV_32F);

	//NORM_MINMAX
	Mat dst = Mat::zeros(gray.size(), CV_32FC1);
	normalize(gray, dst, 1, 0, NORM_MINMAX);
	Mat result = dst * 255;
	result.convertTo(dst, CV_8UC1);
	imshow("MINMAX", dst);

	//NORM_INF
	normalize(gray, dst, 1, 0, NORM_INF);
	result = dst * 255;
	result.convertTo(dst, CV_8UC1);
	imshow("INF", dst);
}

//视频读取
void VideoReadAndWrite(string videoUrl)
{
	VideoCapture capture;
	capture.open(videoUrl);
	if (!capture.isOpened())
	{
		cerr << "could not read this video... video url :" << videoUrl << endl;
	}

	//获取视频大小
	Size s = Size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));
	int fps = capture.get(CAP_PROP_FPS);
	cout << "this video fps is " << fps << endl;

	//创建视频写入流
	VideoWriter writer("D:/test.mp4", VideoWriter::fourcc('D', 'I', 'V', 'X'), fps, s, true);

	//显示视频并保存
	Mat frame;
	namedWindow("camera-demo", WINDOW_AUTOSIZE);
	while (capture.read(frame))
	{
		imshow("camera-demo", frame);
		writer.write(frame);

		char c = waitKey(1000 / fps);
		if (c == 27)
			break;
	}

	capture.release();
	writer.release();
}

//图像翻转
void FlipImage(Mat source)
{
	//x轴翻转 flipcode=0
	//y轴翻转 flipcode=1
	//xy翻转  flipcode=-1

	Mat dst;
	//x轴，上下翻转
	flip(source, dst, 0);
	imshow("X", dst);

	//y轴，左右翻转
	flip(source, dst, 1);
	imshow("Y", dst);

	//xy轴，对角翻转
	flip(source, dst, -1);
	imshow("XY", dst);
}

int main()
{
	//加载图像,加载3通道的BGR彩色图
	Mat input1 = imread("D:\\Photos\\DSC_7570.JPG", IMREAD_COLOR);

	//默认读取彩色图像
	//Mat input2 = imread("D:\\Photos\\DSC_7574.JPG");

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
	//CustomColorMap(input1);
	//查找表API
	//Mat map = Mat(input1.size(), input1.type());
	//applyColorMap(input1, map, COLORMAP_HSV);
	//imshow("ColorMap", map);

	//图像位操作
	//ImageOptionbyBit(input1);

	//图像通道分离与合并
	//ImageSplitAndMerge(input1);

	//图像色彩空间转换
	//ExchangImageColorSpace(input1);

	//图像像素值统计
	//CountImagePixels(input1);

	//像素归一化
	//NormalizeImage(input1);

	//视频读写
	//VideoReadAndWrite("G:\\1.MP4");

	//图像翻转
	//FlipImage(input1);

	//图像插值
	//ImageInterpolation(input1);

	//等待任意按键按下
	waitKey(0);

	return 0;
}