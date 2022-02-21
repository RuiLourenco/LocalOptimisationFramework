#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
//#include "LightField.h"
//#include "DepthMap.h"
#include "json.h"
#include "TestDepthAlg.h"
#include <filesystem>
#include <fstream>
//#include "matplotlibcpp.h"

//namespace plt = matplotlibcpp;

using namespace cv;
using json = nlohmann::json;
using namespace std;
using namespace depthMap;
using namespace testDepth;
using namespace lf;

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
	// read a JSON file
	std::ifstream parameters(".\\parameters.json");
	json j;
	j = json::parse(parameters,nullptr,true,true);
	
	auto lightFields = j["lightFields"];
	vector<cv::String> paths = lightFields["paths"];
	auto iterSettings = j["LocalOptimisationFramework"];
	vector<double> t0 = iterSettings["t0"];
	vector<double> alpha = iterSettings["alpha"];
	vector<double> sigma = iterSettings["sigma"];
	vector<uint> nIter = iterSettings["nIter"];
	vector<uint> nTemps = iterSettings["nTemps"];

	auto cvSettings = j["CostVolumeFramework"];
	vector<uint> nLabels = cvSettings["nLabels"];

	auto methodSettings = j["Methods"];
	vector<cv::String> rawDepthMethods = methodSettings["Framework"];
	vector<DepthMethod> depthMethods = Methods::stringToDepthMethod(rawDepthMethods);
	vector<cv::String> rawCosts = methodSettings["correspondenceCosts"];
	vector<CorrespondenceCost> correspondenceCosts = Methods::stringToCorrespondence(rawCosts);


	MethodsTest methodsTest{ correspondenceCosts,depthMethods };
	IterativeSettingsTest iterSettingsTest{ nTemps,nIter,t0,alpha,sigma };
	
	CostVolumeSettingsTest costVolumeSettingsTest{ nLabels };

	cv::String testDirectory = ".\\testDirectory";
	TestDepthAlg testBattery = TestDepthAlg(paths, testDirectory, methodsTest, costVolumeSettingsTest,iterSettingsTest);
	testBattery.testIterative();


	return 0;
}