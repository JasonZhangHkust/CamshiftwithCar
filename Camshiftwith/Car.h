#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <deque>
#include <utility>

class Car {
public:
	int id;
	int x, y, w, h;
	int hsize = 16;
	float hranges[2] = { 0,180 };
	float moving_direction();
	std::deque<cv::Point2f>  coordations;
	cv::RotatedRect track_box, track_box2;
	cv::Rect track_window, track_window2;
	cv::Mat roi_hist, roi_hist2, mask, mask2, backproj, backproj2, histimg = cv::Mat::zeros(200, 320, CV_8UC3), hsv, hue, hsv2, hue2;
	const float* phranges = hranges;
	cv::Point center,center2;
public:
	Car(int id, cv::Mat frame, cv::Rect track_window, cv::Rect track_window2);
	cv::RotatedRect update(cv::Mat frame);
	cv::Mat getHistimg();
private:
	cv::KalmanFilter kf;
	cv::Mat state;
	cv::Mat meas;
	cv::Rect predRect;
	
	double ticks = 0;
	int stateSize = 6;
	int measSize = 4;
	int contrSize = 0;
	unsigned int type = CV_32F;
private:
	void kalmanInit();
	void kalmanUpdate();
	void kalCorrect();
	

};
