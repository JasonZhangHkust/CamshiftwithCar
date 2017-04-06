#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <ctype.h>
#include "Car.h"

using namespace cv;
using namespace std;

Mat image;

bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
int count1 = 0;
bool showHist = true;
Point origin;
Rect selection, selection2;
int vmin = 10, vmax = 256, smin = 30;
Car* car;

// User draws box around object to track. This triggers CAMShift to start tracking
static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)
	{
		if (count1 == 1) {
			selection.x = MIN(x, origin.x);
			selection.y = MIN(y, origin.y);
			selection.width = std::abs(x - origin.x);
			selection.height = std::abs(y - origin.y);

			//selection &= Rect(0, 0, image.cols, image.rows);
		}
		if (count1 == 2) {
			selection2.x = MIN(x, origin.x);
			selection2.y = MIN(y, origin.y);
			selection2.width = std::abs(x - origin.x);
			selection2.height = std::abs(y - origin.y);

			//selection2 &= Rect(0, 0, image.cols, image.rows);
		}
	}

	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		if (count1 == 0)
			selection = Rect(x, y, 0, 0);
		if (count1 == 1)
			selection2 = Rect(x, y, 0, 0);
		selectObject = true;
		count1++;
		break;
	case EVENT_LBUTTONUP:
		selectObject = false;

		if (selection.width > 0 && selection.height > 0 && count1 == 2)
			trackObject = -1;   // Set up CAMShift properties in main() loop
		break;
	}
}

string hot_keys =
"\n\nHot keys: \n"
"\tESC - quit the program\n"
"\tc - stop the tracking\n"
"\tb - switch to/from backprojection view\n"
"\th - show/hide object histogram\n"
"\tp - pause video\n"
"To initialize tracking, select the object with mouse\n";

static void help()
{
	cout << "a little demo for the car detection";
	cout << hot_keys;
}

const char* keys =
{
	"{help h | | show help message}{@camera_number| 0 | camera number}"
};

int main(int argc, const char** argv)
{
	VideoCapture cap;
	VideoWriter video;
	Rect trackWindow;
	int hsize = 16;
	float hranges[] = { 0,180 };
	const float* phranges = hranges;
	CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		help();
		return 0;
	}
	int camNum = parser.get<int>(0);
	cap.open(camNum);

	if (!cap.isOpened())
	{
		help();
		cout << "***Could not initialize capturing...***\n";
		cout << "Current parameter's value: \n";
		parser.printMessage();
		return -1;
	}
	Size S = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));
	video.open("test.avi", CV_FOURCC('P', 'I', 'M', '1'), 25, S);
	cout << cap.get(CAP_PROP_FRAME_HEIGHT);
	cout << hot_keys;
	namedWindow("Histogram", 0);
	namedWindow("CamShift Demo", 0);
	setMouseCallback("CamShift Demo", onMouse, 0);
	createTrackbar("Vmin", "CamShift Demo", &vmin, 256, 0);
	createTrackbar("Vmax", "CamShift Demo", &vmax, 256, 0);
	createTrackbar("Smin", "CamShift Demo", &smin, 256, 0);

	Mat frame;
	bool paused = false;

	for (;;)
	{
		if (!paused)
		{
			cap >> frame;
			if (frame.empty())
				break;
		}

		frame.copyTo(image);

		if (!paused)
		{

			if (trackObject)
			{

				if (trackObject < 0)
				{
					// Object has been selected by user, set up CAMShift search properties once

					car = new Car(1, image, selection, selection2);
					trackObject = 1; // Don't set up again, unless user selects new ROI
					imshow("Histogram", car->getHistimg());

				}
				cv::RotatedRect trackBox;
				if (car != nullptr) {
					trackBox = car->update(image);
					//cout << "update" << endl;
				}
				if (backprojMode)
					cvtColor(car->backproj2, image, COLOR_GRAY2BGR);
				//ellipse(image, trackBox, Scalar(0, 0, 255), 3, LINE_AA);
				Point2f rect_points[4]; trackBox.points(rect_points);
				for (int j = 0; j < 4; j++)
					line(image, rect_points[j], rect_points[(j + 1) % 4], CV_RGB(20, 150, 20), 1, 8);
				cv::circle(image, car->center, 2, CV_RGB(20, 150, 20), -1);
				Point2f rect_points2[4]; car->track_box2.points(rect_points2);
				for (int j = 0; j < 4; j++)
					line(image, rect_points2[j], rect_points2[(j + 1) % 4], CV_RGB(20, 150, 20), 1, 8);
				//cv::circle(image, car->track_box2.center, 2, CV_RGB(255, 0, 0), -1);
				cv::circle(image, car->center2, 2, CV_RGB(20, 150, 20), -1);

			}

		}
		else if (trackObject < 0)
			paused = false;

		if (selectObject && selection.width > 0 && selection.height > 0)
		{
			Mat roi(image, selection);
			bitwise_not(roi, roi);
		}
		if (selectObject && selection2.width > 0 && selection2.height > 0)
		{
			Mat roi2(image, selection2);
			bitwise_not(roi2, roi2);
		}

		imshow("CamShift Demo", image);
		video << image;

		char c = (char)waitKey(10);
		if (c == 27)
			break;
		switch (c)
		{
		case 'b':
			backprojMode = !backprojMode;
			break;
		case 'c':
			trackObject = 0;
			//histimg = Scalar::all(0);
			break;
		case 'h':
			showHist = !showHist;
			if (!showHist)
				destroyWindow("Histogram");
			else
				namedWindow("Histogram", 1);
			break;
		case 'p':
			paused = !paused;
			break;
		default:
			;
		}
	}
	cap.release();
	video.release();
	return 0;
}
