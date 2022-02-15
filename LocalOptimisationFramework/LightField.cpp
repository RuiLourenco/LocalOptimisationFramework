#include "LightField.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include "DepthMap.h"


using namespace cv;
using namespace std;
using namespace lf;
using namespace col3D;

const uint ANG_SIZE = 9;
const uint SPC_SIZE = 512;
const uint NDIM_COLOR = 5;
const uint NDIM_GRAY = 4;
const char* IMAGE_NAME = "input_Cam";


/*GETTERS*/
cv::Mat LightField::getEpi(uint angular, uint spacial, bool isHorizontal) {

	int shiftSpacialSize;
	if (isHorizontal) {
		shiftSpacialSize = this->size[3];
	}
	else {
		shiftSpacialSize = this->size[2];
	}
	Mat EPI = cv::Mat(ANG_SIZE, shiftSpacialSize, CV_8UC3, (uint)0);
	for (int shiftAngular = 0; shiftAngular < ANG_SIZE; shiftAngular++) {
		for (int shiftSpacial = 0; shiftSpacial < shiftSpacialSize; shiftSpacial++) {
			if (isHorizontal) {
				EPI.at<Vec3b>(shiftAngular, shiftSpacial) = (*this)(angular, shiftAngular, spacial, shiftSpacial);
			}
			else {
				EPI.at<Vec3b>(shiftAngular, shiftSpacial) = (*this)(shiftAngular, angular, shiftSpacial, spacial);
			}

		}
	}
	return EPI;
}
array<uint, 2> LightField::getAngSize() {
	array<uint, 2> angSize = { this->size[0],this->size[1] };
	return angSize;
}
array<uint, 2> LightField::getSpacialSize() {
	array<uint, 2> spacialSize = { this->size[2],this->size[3] };
	return spacialSize;
}
cv::Mat* LightField::getView(uint t, uint s) {
	return &this->data[t][s];
}
cv::Mat* LightField::operator()(uint t, uint s) {
	return LightField::getView(t, s);
}
uint LightField::channelNumber() {
	return this->size[4];
}

inline uint8_t get_value(cv::Mat const& img, int32_t row, int32_t col, int32_t channel)
{
	CV_DbgAssert(channel < img.channels());
	uint8_t const* pixel_ptr(img.ptr(row, col));
	uint8_t const* value_ptr(pixel_ptr + channel);
	return *value_ptr;
}



/*Construct a Light Field Object from the path to the folder containing all the Light Field Images.*/


LightField::LightField(String path, bool isColor) {
	//Initialise the size of the light field with the default sizes.
	if (isColor) {
		this->size = { ANG_SIZE, ANG_SIZE, SPC_SIZE, SPC_SIZE, 3 };
	}
	else {
		this->size = { ANG_SIZE, ANG_SIZE, SPC_SIZE, SPC_SIZE, 1 };
	}

	char imageNumber[4];
	//Set the name of the current scene based on the path.
	int idx = path.find_last_of('\\');
	if (std::string::npos != idx)
	{
		this->name = path.substr(idx + 1);
	}

	for (uint i = 0; i < ANG_SIZE * ANG_SIZE; i++) {
		snprintf(imageNumber, sizeof(imageNumber), "%03d", i);
		uint s = i % ANG_SIZE;
		uint t = i / ANG_SIZE;
		Mat image = imread(path + "\\" + IMAGE_NAME + imageNumber + ".png", IMREAD_ANYCOLOR);
		if (image.empty()) {
			cout << path + "\\" + IMAGE_NAME + imageNumber + ".png" << endl;
			//cout << "Could not open or find the image" << endl;
			////wait for any key press
			//cin.get(); 
		}
		this->size[2] = image.size().height;
		this->size[3] = image.size().width;
		if (!isColor) {
			cvtColor(image, data[t][s], COLOR_BGR2GRAY);
		}
		else {
			data[t][s] = image;
		}
	}
	this->gtPointer = shared_ptr<depthMap::DepthMap>(new depthMap::DepthMap(this, (String)path + "\\" + "gt_disp_lowres.pfm"));
}

void LightField::showView(uint t, uint s) {
	Mat view = this->data[t][s];
	if (view.empty()) {
		cout << "Could not open or find the image" << endl;
		//wait for any key press
		cin.get();
	}
	String windowName = "View"; //Name of the window
	namedWindow(windowName);
	imshow(windowName, data[s][t]);
	waitKey(0);
}

void LightField::showHorizontalEPI(uint t, uint v) {
	Mat epi = Mat(this->size[1], this->size[3], CV_8U, 1);
	for (int i = 0; i < this->size[1]; i++) {
		Mat view = this->data[i][t];
		Mat temp = view.row(v) + 0;
		epi.row(i) = temp + 0;
	}
	//cout << epi << endl;
	if (epi.empty()) {
		cout << "Could not display EPI" << endl;
		//wait for any key press
		cin.get();
	}
	uchar a = epi.at<uchar>(4, 100);
	Mat view = this->data[4][t];
	cout << view.row(100) << endl;
	cout << epi.row(4) << endl;

	String windowName = "EPI"; //Name of the window
	namedWindow(windowName);
	imshow(windowName, epi);

	waitKey(0);
}

std::shared_ptr<depthMap::DepthMap>& LightField::getGT() {
	return this->gtPointer;
}

Vec3b LightField::operator()(uint t, uint s, uint v, uint u) {
	Mat view = this->data[t][s];
	Vec3b integer = view.at<Vec3b>(v, u);
	return integer;
}

LightField::~LightField() {

}