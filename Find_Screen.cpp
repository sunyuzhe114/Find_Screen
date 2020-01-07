

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2\calib3d.hpp>

using namespace cv;
using namespace std;


bool bDebug = true;

//int thresh = 50, N = 11;
bool bUseCamer = false;
int thresh = 200, N = 7;
const char* wndname = "水印检测_v2019_12_27";
double area_min_limit = 1000;
double area_max_limit = 10000;
string currentfilename;
Mat perspective_transformation(const vector<Point2f>& final_points, Mat& src);
vector<vector<Point2f>> divide_points_into_4_parts(const vector<Point2f>& line_nodes, int height, int width);




// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
static void findSquares(const Mat& image, vector<vector<Point> >& squares)
{
	squares.clear();

	Mat pyr, timg, gray0(image.size(), CV_8U), gray;

	// down-scale and upscale the image to filter out the noise
	pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
	pyrUp(pyr, timg, image.size());
	vector<vector<Point> > contours;

	// find squares in every color plane of the image
	//for( int c = 0; c < 3; c++ )
	//{
		//int ch[] = {c, 0};
		//mixChannels(&timg, 1, &gray0, 1, ch, 1);
	cvtColor(timg, gray0, COLOR_BGR2GRAY);
	// try several threshold levels
	for (int l = 0; l < N; l++)
	{
		// hack: use Canny instead of zero threshold level.
		// Canny helps to catch squares with gradient shading
		if (l == 0)
		{
			// apply Canny. Take the upper threshold from slider
			// and set the lower to 0 (which forces edges merging)
			Canny(gray0, gray, 0, thresh, 5);
			// dilate canny output to remove potential
			// holes between edge segments

			//namedWindow("gray0", 0);
			//namedWindow("gray00", 0);
			//resizeWindow("gray0", 540, 720);
			//resizeWindow("gray00", 540, 720);
			//imshow("gray0", gray);


			dilate(gray, gray, Mat(), Point(-1, -1));


			// imshow("gray00", gray);

		}
		else
		{
			// apply threshold if l!=0:
			//     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
			gray = gray0 >= (l + 1) * 255 / N;
			//string strname=std::to_string( l);
			//namedWindow(strname, 0); 
			//resizeWindow(strname, 540, 720);
			//imshow(strname, gray);
		}

		// find contours and store them all as a list
		findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);


		vector<Point> approx;

		// test each contour
		double minCos = 1;
		for (size_t i = 0; i < contours.size(); i++)
		{
			// approximate contour with accuracy proportional
			// to the contour perimeter
			approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);

			// square contours should have 4 vertices after approximation
			// relatively large area (to filter out noisy contours)
			// and be convex.
			// Note: absolute value of an area is used because
			// area may be positive or negative - in accordance with the
			// contour orientation
			if (approx.size() == 4 &&
				fabs(contourArea(approx)) > area_min_limit&&
				fabs(contourArea(approx)) < area_max_limit&&
				isContourConvex(approx))
			{
				double maxCosine = 0;

				for (int j = 2; j < 5; j++)
				{
					// find the maximum cosine of the angle between joint edges
					double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
					maxCosine = MAX(maxCosine, cosine);
				}

				// if cosines of all angles are small
				// (all angles are ~90 degree) then write quandrange
				// vertices to resultant sequence
				if (maxCosine < 0.3)
				{
					//cout << "L=" << l << " maxCosine " << maxCosine << endl;
					squares.push_back(approx);
				}
			}
		}
	}
	// }
}
 

vector<vector<Point2f>> divide_points_into_4_parts(const vector<Point2f>& line_nodes, int img_height, int img_width)
{
	vector<Point2f> left_top_line_nodes, left_down_line_nodes, right_top_line_nodes, right_down_line_nodes;
	vector<vector<Point2f>> res;
	int height = img_height / 2, width = img_width / 2;
	float p1 = 1, p2 = 1; //tiao can
	/*for (auto node : line_nodes)
	{
		if (node.x < height * p1)
		{
			if (node.y < width * p1)
				left_top_line_nodes.emplace_back(node);
			if (node.y > width* p2)
				left_down_line_nodes.emplace_back(node);
		}
		if (node.x > height* p2)
		{
			if (node.y < width * p1)
				right_top_line_nodes.emplace_back(node);
			if (node.y > width* p2)
				right_down_line_nodes.emplace_back(node);
		}
	}*/

	for (auto node : line_nodes)
	{
		if (node.x < width * p1)
		{
			if (node.y < height * p1)
				left_top_line_nodes.emplace_back(node);
			if (node.y > height* p2)
				left_down_line_nodes.emplace_back(node);
		}
		if (node.x > width* p2)
		{
			if (node.y < height * p1)
				right_top_line_nodes.emplace_back(node);
			if (node.y > height* p2)
				right_down_line_nodes.emplace_back(node);
		}
	}
	res.emplace_back(left_top_line_nodes);
	res.emplace_back(left_down_line_nodes);
	res.emplace_back(right_top_line_nodes);
	res.emplace_back(right_down_line_nodes);
	return res;
}
void ColorSalt(Mat& image, int n)//本函数加入彩色盐噪声
{
	srand((unsigned)time(NULL));
	for (int k = 0; k < n; k++)//将图像中n个像素随机置零
	{
		int i = rand() % image.cols;
		int j = rand() % image.rows;
		//将图像颜色随机改变
		image.at<Vec3b>(j, i)[0] = 0;
		image.at<Vec3b>(j, i)[1] = 0;
		image.at<Vec3b>(j, i)[2] = 0;
	}
}

void FindLines(Mat& image)
{
	double image_width = image.cols;
	double image_height = image.rows;
	int findtimes = 0;
	int MAXCHECKTIME = 3;
	for (int m = 0; m < MAXCHECKTIME; m++)
	{
		int beginPOS_X = 200 + m * 256;
		int beginPOS_Y = 200 + m * 256;
		if (beginPOS_X  + 256 >= image_width || beginPOS_Y  + 256 >= image_height)
		{
			break;
		}
		Rect rect(beginPOS_X, beginPOS_Y , 256, 256); 
		Mat image_roi = image(rect);

		if (bDebug)
		{
			namedWindow("image_roi", 0);
			imshow("image_roi", image_roi);
			namedWindow("myimage", 0);
		}
		Mat gray, thresh, mat_canny, mat_dilate;
		cvtColor(image_roi, gray, COLOR_BGR2GRAY);
		medianBlur(gray, gray, 3);
		if (bDebug)
		{
			imshow("gray", gray);
		}

		//////////////////////////////////////

		Mat dstImage;        //初始化自适应阈值参数
		const int maxVal = 255;
		int blockSize = 63;    //取值3、5、7....等
		int constValue = 10;
		int adaptiveMethod = 0;
		int thresholdType = 0;
		/*
			自适应阈值算法
			0:ADAPTIVE_THRESH_MEAN_C
			1:ADAPTIVE_THRESH_GAUSSIAN_C
			--------------------------------------
			阈值类型
			0:THRESH_BINARY
			1:THRESH_BINARY_INV
		*/
		//cv2.adaptiveThreshold(img,           255, cv2.ADAPTIVE_THRESH_MEAN_C,	cv2.THRESH_BINARY, 63, 2)
		adaptiveThreshold(gray, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 63, 3);
		//imshow("thresh", thresh);
		//////////////////////////////////////
		Canny(thresh, mat_canny, 40, 200, 3);
		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
		dilate(mat_canny, mat_dilate, kernel, Point(-1, -1));
		if (bDebug)
		{
			imshow("mat_canny", mat_dilate);
		}
		vector<Vec4f> plines;
		HoughLinesP(mat_dilate, plines, 1, CV_PI / 180.0, 100, 90, 7);
		Scalar color = Scalar(0, 0, 255);
		int contoursNum = 0;
		for (size_t i = 0; i < plines.size(); i++) {
			//这里要计算一下line的角度


			Vec4f hline = plines[i];


			double k = (double)(hline[3] - hline[1]) / (double)(hline[2] - hline[0]); //求出直线的斜率
			double tha = atan(k) * 180.0 / 3.1415926;
			if (tha > 40 && tha < 50)
			{
				if (bDebug)
				{
					line(image_roi, Point(hline[0], hline[1]), Point(hline[2], hline[3]), color, 3, LINE_AA);
					cout << "检测到的直线角度为：" << tha << endl;
				}

				contoursNum++;
			}
			if (tha > 30 && tha < 40)
			{
				// line(image_roi, Point(hline[0], hline[1]), Point(hline[2], hline[3]), color, 3, LINE_AA);
				 //cout << "检测到的直线角度为：" << tha << endl; 
				// contoursNum++;
			}

			if (bDebug)
			{
				cout << "检测到的直线角度为：" << tha << endl;
			}

		}
		if (contoursNum >= 2 && contoursNum <= 60)
		{
			findtimes++;
			//cout << currentfilename << " **************此图片存在企业水印 **************"<< contoursNum << endl;
		}
		else
		{

			//cout << currentfilename << " 未检出企业水印 " << "" << endl;
		}
		if (bDebug)
		{
			cout << "检测第：" << m << endl;
		}
	}
	if (findtimes >= MAXCHECKTIME - 1)
	{
		cout << currentfilename << " **************此图片存在企业水印 **************" << "内容为   zte20191228" << endl;
	}
	else
	{

		cout << currentfilename << " 未检出企业水印 " << "" << endl;
	}
}
void FindPoints(Mat& image)
{

	double image_width = image.cols;
	double image_height = image.rows;
	int findtimes = 0;
	int MAXCHECKTIME = 6;
	for (int m = 0; m < MAXCHECKTIME; m++)
	{
		int beginPOS_X = 200 + m * 256;
		int beginPOS_Y = 200 + m * 256;
		if (beginPOS_X   + 256 >= image_width || beginPOS_Y   + 256 >= image_height)
		{
			break;
		}
		Rect rect(beginPOS_X , beginPOS_Y, 256, 256);
		Mat image_roi = image(rect);

		if (bDebug)
		{
			namedWindow("image", 0);
			resizeWindow("image", 640, 960);
			rectangle(image, rect, cv::Scalar(0, 0, 255), 1, 0, 0);
			imwrite("d://sub_" + std::to_string(m) + ".jpg",image_roi );
			imshow("image", image);
			namedWindow("image_roi", 0);
			imshow("image_roi", image_roi);
			namedWindow("myimage", 0);
		}
		Mat gray, thresh, mat_canny, mat_dilate;
		cvtColor(image_roi, gray, COLOR_BGR2GRAY);
		//medianBlur(gray, gray, 3);
		threshold(gray, thresh, 128, 255, THRESH_BINARY);
		//threshold(gray, thresh, 140, 255, THRESH_BINARY);
		//////////////////////////////////////

				//////////////////////////////////////

		//Mat dstImage;        //初始化自适应阈值参数
		//const int maxVal = 255;
		//int blockSize = 11;    //取值3、5、7....等
		//int constValue = 10; 
		///*
		//	自适应阈值算法
		//	0:ADAPTIVE_THRESH_MEAN_C
		//	1:ADAPTIVE_THRESH_GAUSSIAN_C
		//	--------------------------------------
		//	阈值类型
		//	0:THRESH_BINARY
		//	1:THRESH_BINARY_INV
		//*/
		////cv2.adaptiveThreshold(img,           255, cv2.ADAPTIVE_THRESH_MEAN_C,	cv2.THRESH_BINARY, 63, 2)
		//adaptiveThreshold(gray, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, 3);
		////imshow("thresh", thresh);
		//Canny(thresh, mat_canny, 40, 200, 3);
		//////////////////////////////////////
		Canny(thresh, mat_canny, 55, 116, 3);
		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
		dilate(mat_canny, mat_dilate, kernel, Point(-1, -1));
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		if (bDebug)
		{
			namedWindow("mat_canny", 0);
			imshow("mat_canny", mat_canny);
		namedWindow("dilate", 0);
		imshow("dilate", mat_dilate);

		}

		findContours(mat_dilate, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());

		double  threshold_min_area = 4;
		double threshold_max_area = 15.9;

		int contoursNum = 0;
		for (int i = 0; i < contours.size(); i++)
		{
			//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数 
			double area = contourArea(contours[i]);
			if (area > threshold_min_area&& area < threshold_max_area)
			{
				contoursNum++;

				const Point* p = &contours[i][0];
				int n = (int)contours[i].size();
				polylines(image_roi, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
				//cout << area << " size " <<contours[i].size() << endl;
			}
		}
		//cout << "total find " << contoursNum << endl;
		if (contoursNum >= 4)
		{
			findtimes++;
			cout << currentfilename << " **************区域存在企业水印 **************" << endl;
		}
		else
		{
			cout << currentfilename << " 未检出企业水印 " << "" << endl;
		}
		if (bDebug)
		{
			cout << "检测第：" << m << endl;
			imshow("myimage", image_roi);
			waitKey(0);
		}
	}
	if (findtimes >= MAXCHECKTIME - 1)
	{
		cout << currentfilename << " **************此图片存在企业水印 **************" << "内容为   zte20191228" << endl;
	}
	else
	{

		cout << currentfilename << " 未检出企业水印 " << "" << endl;
	}
}
// the function draws all the squares in the image
static void drawSquares(Mat& image, const vector<vector<Point> >& squares, string filename)
{

	/*  for( size_t i = 0; i < squares.size(); i++ )
	  {
		  const Point* p = &squares[i][0];
		  int n = (int)squares[i].size();
		  polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
	  }

	  imshow(wndname, image);
	  */
	Mat imageoutput = image.clone();
	double mincos = 1;
	int rect_index = -1;
	for (size_t i = 0; i < squares.size(); i++)
	{
		const Point* p = &squares[i][0];
		int n = (int)squares[i].size();
		double maxCosine = 0;
		for (int j = 2; j < 5; j++)
		{
			// find the maximum cosine of the angle between joint edges
			double cosine = fabs(angle(squares[i][j % 4], squares[i][j - 2], squares[i][j - 1]));
			maxCosine = MAX(maxCosine, cosine);
		}
		if (maxCosine < mincos)
		{
			mincos = maxCosine;
			rect_index = i;
		}

	}
	if (rect_index != -1)
	{
		const Point* p = &squares[rect_index][0];
		int n = (int)squares[rect_index].size();



		polylines(imageoutput, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
		//cout << "draw " << filename << squares[rect_index] << " maxcos " << mincos << endl;
		//这里保存一下
		//获取文本框的长宽
		std::string text = "Hello World!";
		int font_face = cv::FONT_HERSHEY_COMPLEX;
		double font_scale = 1.5;
		int thickness = 3;
		if (bUseCamer)
		{
			thickness = 1;
			font_scale = 0.5;
		}
		int baseline;
		cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

		//将文本框居中绘制
		cv::Point origin;
		origin.x = image.cols / 2 - text_size.width / 2;
		origin.y = image.rows / 2 + text_size.height / 2;

		vector<Point2f> pfs;
		for (int i = 0; i < 4; i++)
		{
			pfs.push_back(Point2f(squares[rect_index][i].x, squares[rect_index][i].y));
			putText(imageoutput, to_string(i) + "(" + to_string(squares[rect_index][i].x) + "," + to_string(squares[rect_index][i].y) + ")", squares[rect_index][i], font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 8, 0);
		}

		vector<vector<Point2f>>  final_fps = divide_points_into_4_parts(pfs, image.rows, image.cols);
		pfs.clear();

		bool bOK = true;
		for (int i = 0; i < 4; i++)
		{
			if (final_fps[i].size() == 0)
			{
				bOK = false;
				break;
			}

		}
		if (bOK == true)
		{

			pfs.push_back(Point2f(final_fps[0][0].x, final_fps[0][0].y));
			pfs.push_back(Point2f(final_fps[1][0].x, final_fps[1][0].y));
			pfs.push_back(Point2f(final_fps[2][0].x, final_fps[2][0].y));
			pfs.push_back(Point2f(final_fps[3][0].x, final_fps[3][0].y));


			Mat finalresult = perspective_transformation(pfs, image);
			double FORCE_HEIGHT = 640;
			double rate = finalresult.rows / FORCE_HEIGHT;
			resizeWindow("result", finalresult.cols / rate, FORCE_HEIGHT);
			imshow("result", finalresult);
			if (bUseCamer == false)
			{
				size_t a = filename.find_last_of('\\');

				string newfilename = "_" + filename.substr(a + 1);
				string filepath = filename.substr(0, a + 1);
				imwrite(filepath + newfilename, finalresult);
				//这里可以直接处理提取出来的文件
				FindPoints(finalresult);
				//FindLines(finalresult);

			}
		}
		else
		{
			if (bUseCamer == false)
			{

				size_t a = filename.find_last_of('\\');

				string newfilename = "_" + filename.substr(a + 1);
				string filepath = filename.substr(0, a + 1);
				imwrite(filepath + newfilename, image);
				putText(imageoutput, "DETECT ERROR", Point2i(100, 100), font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 8, 0);
				cout << "检测失败" << endl;
			}
		}

		imshow(wndname, imageoutput);
	}
	else
	{
		if (bUseCamer == false)
		{

			size_t a = filename.find_last_of('\\');
			putText(image, "DETECT ERROR", Point2i(100, 100), 3, 3, cv::Scalar(0, 0, 255), 8, 8, 0);
			imshow(wndname, image);
			cout << "检测失败" << endl;
		}

	}

	imageoutput.release();
}

Mat perspective_transformation(const vector<Point2f>& final_points, Mat& src)
{
	//debug->print(final_points);
	Point2f _srcTriangle[4];
	Point2f _dstTriangle[4];
	vector<Point2f>srcTriangle(_srcTriangle, _srcTriangle + 4);
	vector<Point2f>dstTriangle(_dstTriangle, _dstTriangle + 4);
	Mat after_transform;

	const int leftTopX = final_points[0].x;
	const int leftTopY = final_points[0].y;
	const int leftDownX = final_points[1].x;
	const int leftDownY = final_points[1].y;
	const int rightTopX = final_points[2].x;
	const int rightTopY = final_points[2].y;
	const int rightDownX = final_points[3].x;
	const int rightDownY = final_points[3].y;



	int newWidth = 0;
	int newHeight = 0;

	newWidth = sqrt((leftTopX - rightTopX) * (leftTopX - rightTopX) + (leftTopY - rightTopY) * (leftTopY - rightTopY));
	newHeight = sqrt((leftTopX - leftDownX) * (leftTopX - leftDownX) + (leftTopY - leftDownY) * (leftTopY - leftDownY));
	//cout << newWidth << " " << newHeight << endl;
	after_transform = Mat::zeros(newHeight, newWidth, src.type());

	srcTriangle[0] = Point2f(leftTopX, leftTopY);
	srcTriangle[1] = Point2f(rightTopX, rightTopY);
	srcTriangle[2] = Point2f(leftDownX, leftDownY);
	srcTriangle[3] = Point2f(rightDownX, rightDownY);

	dstTriangle[0] = Point2f(0, 0);
	dstTriangle[1] = Point2f(newWidth, 0);
	dstTriangle[2] = Point2f(0, newHeight);
	dstTriangle[3] = Point2f(newWidth, newHeight);


	Mat m1 = Mat(srcTriangle);
	Mat m2 = Mat(dstTriangle);
	Mat status;
	Mat h = findHomography(m1, m2, status, 0, 3);
	perspectiveTransform(srcTriangle, dstTriangle, h);
	warpPerspective(src, after_transform, h, after_transform.size());
	//debug->show_img("after_transform", after_transform);
	return after_transform;
}

int main(int argc, char** argv)
{
	namedWindow(wndname, 0);
	namedWindow("result", 0);
	/*   static const char* names[] = { "pic1.png", "pic2.png", "pic3.png",
		   "pic4.png", "pic5.png", "pic6.png", 0 };"22.jpg",  */
		   //char* names[] = {"new2.jpg", "new1.jpg","2.jpg","15.jpg",  "3.jpg",   "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg",   "12.jpg", "13.jpg",
		   //     "15.jpg", "16.jpg",   "17.jpg",  "20.jpg","new.jpg",   0 };
	char* names[] = { "new4.jpg","15.jpg","new2.jpg", "new1.jpg","2.jpg","15.jpg",  "3.jpg",   "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg",   "12.jpg", "13.jpg",
		 "15.jpg", "16.jpg",   "17.jpg",  "20.jpg","new.jpg",   0 };

	if (argc > 1)
	{
		names[0] = argv[1];
		names[1] = 0;
		if (strcmp(argv[1], "-Camera") == 0)
			bUseCamer = true;
	}

	VideoCapture cap;


	if (bUseCamer)
	{
		cap.open(0);

		if (!cap.isOpened())
		{
			cout << "Could not initialize capturing...\n";
			return 0;
		}
	}

	Mat gray, prevGray, image, frame;
	vector<Point2f> points[2];



	for (int i = 0; ; i++)
	{
		if (!bUseCamer)
		{
			if (names[i] == 0)
				break;
		}
		vector<vector<Point> > squares;

		if (bUseCamer)
		{
			cap >> frame;
			if (frame.empty())
				break;

			frame.copyTo(image);
		}
		else
		{

			// string filename = samples::findFile(names[i]);
			string filename = names[i];
			if (argc > 1)filename = names[i];
			currentfilename = names[i];
			image = imread(filename, IMREAD_COLOR);
		}



		if (image.empty())
		{
			cout << "载入演示 " << currentfilename << " 失败 !!!" << endl;
			continue;
		}

		double width = image.cols;
		double height = image.rows;
		//面积至少占五分之一以上
		area_min_limit = width * height / 5;
		//area_min_limit = width * height  ;
		area_max_limit = width * height * 9 / 10;
		double FORCE_HEIGHT = 640;
		double rate = height / FORCE_HEIGHT;
		//cout << names[i] << " size " << width <<"x" <<height << endl;

		//cout << "re size " << width / rate << "x" << FORCE_HEIGHT << endl;

		resizeWindow(wndname, width / rate, FORCE_HEIGHT);
		//resizeWindow("result", width / rate, FORCE_HEIGHT);
		findSquares(image, squares);
		if (bUseCamer)
			drawSquares(image, squares, "");
		else
			drawSquares(image, squares, names[i]);

		int c = waitKey(10);
		if (bUseCamer)
			c = (char)waitKey(10);
		else
			c = waitKey(0);

		if (c == 27)
			break;
	}
	waitKey(0);
	destroyAllWindows();
	return 0;
}
