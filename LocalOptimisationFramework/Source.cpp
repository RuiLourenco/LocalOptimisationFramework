#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
//#include "LightField.h"
//#include "DepthMap.h"
#include "TestDepthAlg.h"
#include <filesystem>
#include <fstream>
//#include "matplotlibcpp.h"

//namespace plt = matplotlibcpp;

using namespace cv;
using namespace std;
using namespace depthMap;
using namespace testDepth;
using namespace lf;

namespace fs = std::filesystem;

int main(int argc, char** argv)
{


	vector<cv::String> paths;
	paths.push_back("c:\\users\\ruilo\\documents\\work\\images\\hci_dataset\\training\\boxes");
	paths.push_back("c:\\users\\ruilo\\documents\\work\\images\\hci_dataset\\training\\cotton");
	paths.push_back("c:\\users\\ruilo\\documents\\work\\images\\hci_dataset\\training\\dino");
	paths.push_back("c:\\users\\ruilo\\documents\\work\\images\\hci_dataset\\training\\sideboard");

	vector<double> t0{ 1 };
	vector<double> alpha{ 0.8 };
	vector<double> sigma{ 0.02 };
	vector<uint> nIter{ 2 };
	vector<uint> nTemps{ 6 };
	vector<DepthMethod> depthMethods{DepthMethod::costVolume,DepthMethod::iterative};
	vector<CorrespondenceCost> correspondenceCosts{ CorrespondenceCost::Variance,CorrespondenceCost::CAE,CorrespondenceCost::OAVCost };
	MethodsTest methodsTest{ correspondenceCosts,depthMethods };
	IterativeSettingsTest iterSettingsTest{ nTemps,nIter,t0,alpha,sigma };
	vector<uint> nLabels = { 30,60,90 };
	CostVolumeSettingsTest costVolumeSettingsTest{ nLabels };

	
	
	cv::String testDirectory = ".\\testDirectory";
	TestDepthAlg testBattery = TestDepthAlg(paths, testDirectory, methodsTest, costVolumeSettingsTest,iterSettingsTest);
	testBattery.testIterative();


	return 0;
}