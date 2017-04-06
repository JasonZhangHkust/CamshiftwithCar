#include "Car.h"

Car::Car(int id, cv::Mat frame, cv::Rect track_window, cv::Rect track_window2) {
	int _vmin = 10, _vmax = 256;
	//cv::FileStorage fs("test2.yml", cv::FileStorage::WRITE);
	cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
	cvtColor(frame, hsv2, cv::COLOR_BGR2HSV);
	cv::inRange(hsv, cv::Scalar(100, 30, MIN(_vmin, _vmax)),
		cv::Scalar(130, 256, MAX(_vmin, _vmax)), mask);
	cv::inRange(hsv2, cv::Scalar(26, 80, 136),
		cv::Scalar(55, 239, 255), mask2);
	int ch[] = { 0, 0 };
	hue.create(hsv.size(), hsv.depth());
	hue2.create(hsv2.size(), hsv2.depth());
	mixChannels(&hsv, 1, &hue, 1, ch, 1);
	mixChannels(&hsv2, 1, &hue2, 1, ch, 1);
	this->id = id;
	this->x = track_window.x;
	this->y = track_window.y;
	this->w = track_window.width;
	this->h = track_window.height;

	this->track_window = track_window;
	this->track_window2 = track_window2;
	erode(hsv, hsv, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	dilate(hsv, hsv, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

	//morphological closing (removes small holes from the foreground)
	dilate(hsv, hsv, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	erode(hsv, hsv, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	cv::Mat maskroi(mask, track_window);
	cv::Mat maskroi2(mask2, track_window2);
	cv::Mat roi(hue, track_window);
	cv::Mat roi2(hue2, track_window2);
	cv::calcHist(&roi, 1, 0, maskroi, roi_hist, 1, &hsize, &phranges);
	cv::calcHist(&roi2, 1, 0, maskroi2, roi_hist2, 1, &hsize, &phranges);
	//fs << "right" << roi_hist;
	this->histimg = cv::Scalar::all(0);
	int binW = histimg.cols / hsize;
	cv::Mat buf(1, hsize, CV_8UC3);
	for (int i = 0; i < hsize; i++)
		buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180. / hsize), 255, 255);
	cv::cvtColor(buf, buf, cv::COLOR_HSV2BGR);

	for (int i = 0; i < hsize; i++)
	{
		int val = cv::saturate_cast<int>(roi_hist.at<float>(i)*histimg.rows / 255);
		cv::rectangle(histimg, cv::Point(i*binW, histimg.rows),
			cv::Point((i + 1)*binW, histimg.rows - val),
			cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8);
	}
	kalmanInit();
}
cv::RotatedRect Car::update(cv::Mat frame) {
	kalmanUpdate();
	int _vmin = 10, _vmax = 256;
	cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
	cvtColor(frame, hsv2, cv::COLOR_BGR2HSV);
	cv::inRange(hsv, cv::Scalar(100, 30, MIN(_vmin, _vmax)),
		cv::Scalar(130, 256, MAX(_vmin, _vmax)), mask);
	cv::inRange(hsv2, cv::Scalar(26, 80, 136),
		cv::Scalar(55, 239, 255), mask2);
	int ch[] = { 0, 0 };
	hue.create(hsv.size(), hsv.depth());
	hue2.create(hsv2.size(), hsv2.depth());
	mixChannels(&hsv, 1, &hue, 1, ch, 1);
	mixChannels(&hsv2, 1, &hue2, 1, ch, 1);

	cv::calcBackProject(&hue, 1, 0, roi_hist, backproj, &phranges);
	cv::calcBackProject(&hue2, 1, 0, roi_hist2, backproj2, &phranges);

	erode(backproj, backproj, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	dilate(backproj, backproj, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

	//morphological closing (removes small holes from the foreground)
	//dilate(backproj, backproj, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	//erode(backproj, backproj, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

	erode(backproj2, backproj2, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	dilate(backproj2, backproj2, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

	//morphological closing (removes small holes from the foreground)
	//dilate(backproj2, backproj2, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	//erode(backproj2, backproj2, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	backproj &= mask;
	backproj2 &= mask2;
	track_box2 = cv::CamShift(backproj2, track_window2,
		cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
	track_box = cv::CamShift(backproj, track_window,
		cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
	kalCorrect();
	if (track_window.area() <= 1)
	{
		int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
		track_window = cv::Rect(track_window.x - r, track_window.y - r,
			track_window.x + r, track_window.y + r) &
			cv::Rect(0, 0, cols, rows);
	}
	if (coordations.size() < 30)
		coordations.push_back(track_box.center);
	else {
		coordations.pop_front();
		coordations.push_back(track_box.center);
	}
	return track_box;
}
cv::Mat Car::getHistimg() {
	return histimg;
};
float Car::moving_direction() {
	if (coordations.size() > 2) {
		auto dx = coordations.back().x - coordations.front().x;
		auto dy = coordations.back().y - coordations.front().y;

	}
	return 0;
}

void Car::kalmanInit() {
	int stateSize = 12;
	int measSize = 8;
	int contrSize = 0;

	unsigned int type = CV_32F;
	kf = cv::KalmanFilter(stateSize, measSize, contrSize, type);

	state = cv::Mat(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
	meas = cv::Mat(measSize, 1, type);    // [z_x,z_y,z_w,z_h]
										  //cv::Mat procNoise(stateSize, 1, type)
										  // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

										  // Transition State Matrix A
										  // Note: set dT at each processing step!
										  // [ 1 0 dT 0  0 0 ]
										  // [ 0 1 0  dT 0 0 ]
										  // [ 0 0 1  0  0 0 ]
										  // [ 0 0 0  1  0 0 ]
										  // [ 0 0 0  0  1 0 ]
										  // [ 0 0 0  0  0 1 ]
	cv::setIdentity(kf.transitionMatrix);

	// Measure Matrix H
	// [ 1 0 0 0 0 0 ]
	// [ 0 1 0 0 0 0 ]
	// [ 0 0 0 0 1 0 ]
	// [ 0 0 0 0 0 1 ]
	kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
	kf.measurementMatrix.at<float>(0) = 1.0f;
	kf.measurementMatrix.at<float>(13) = 1.0f;
	kf.measurementMatrix.at<float>(26) = 1.0f;
	kf.measurementMatrix.at<float>(39) = 1.0f;
	kf.measurementMatrix.at<float>(56) = 1.0f;
	kf.measurementMatrix.at<float>(69) = 1.0f;
	kf.measurementMatrix.at<float>(82) = 1.0f;
	kf.measurementMatrix.at<float>(95) = 1.0f;



	// Process Noise Covariance Matrix Q
	// [ Ex   0   0     0     0    0  ]
	// [ 0    Ey  0     0     0    0  ]
	// [ 0    0   Ev_x  0     0    0  ]
	// [ 0    0   0     Ev_y  0    0  ]
	// [ 0    0   0     0     Ew   0  ]
	// [ 0    0   0     0     0    Eh ]
	//cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
	kf.errorCovPre.at<float>(0) = 1; // px
	kf.errorCovPre.at<float>(13) = 1; // px
	kf.errorCovPre.at<float>(26) = 1;
	kf.errorCovPre.at<float>(39) = 1;
	kf.errorCovPre.at<float>(52) = 1; // px
	kf.errorCovPre.at<float>(65) = 1; // px
	kf.errorCovPre.at<float>(78) = 1; // px
	kf.errorCovPre.at<float>(91) = 1; // px
	kf.errorCovPre.at<float>(104) = 1; // px
	kf.errorCovPre.at<float>(117) = 1; // px
	kf.errorCovPre.at<float>(130) = 1; // px
	kf.errorCovPre.at<float>(143) = 1; // px

	state.at<float>(0) = track_window.x + track_window.width / 2;
	state.at<float>(1) = track_window.y + track_window.height / 2;
	state.at<float>(2) = track_window2.x + track_window2.width / 2;
	state.at<float>(3) = track_window2.y + track_window2.height / 2;
	state.at<float>(4) = 0;
	state.at<float>(5) = 0;
	state.at<float>(6) = 0;
	state.at<float>(7) = 0;
	state.at<float>(8) = track_window.width;
	state.at<float>(9) = track_window.height;
	state.at<float>(10) = track_window2.width;
	state.at<float>(11) = track_window2.height;
	kf.statePost = state;
	// Measures Noise Covariance Matrix R
	cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));

}
void Car::kalmanUpdate() {
	double preTick = ticks;
	ticks = (double)cv::getTickCount();
	double dT = (ticks - preTick) / cv::getTickFrequency();
	kf.transitionMatrix.at<float>(4) = dT;
	kf.transitionMatrix.at<float>(18) = dT;
	kf.transitionMatrix.at<float>(29) = dT;
	kf.transitionMatrix.at<float>(43) = dT;

	state = kf.predict();

	center.x = state.at<float>(0);
	center.y = state.at<float>(1);
	center2.x = state.at<float>(2);
	center2.y = state.at<float>(3);


}
void Car::kalCorrect() {
	meas.at<float>(0) = track_box.center.x;
	meas.at<float>(1) = track_box.center.y;
	meas.at<float>(2) = track_box2.center.x;
	meas.at<float>(3) = track_box2.center.y;
	meas.at<float>(4) = (float)track_window.width;
	meas.at<float>(5) = (float)track_window.height;
	meas.at<float>(6) = (float)track_window2.width;
	meas.at<float>(7) = (float)track_window2.height;

	kf.correct(meas);
}