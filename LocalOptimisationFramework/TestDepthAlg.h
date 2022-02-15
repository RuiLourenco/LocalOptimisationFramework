
#pragma once
#include <iostream>
#include "LightField.h"
#include "DepthMap.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>


namespace testDepth {
	using namespace depthMap;
	using namespace std;

	struct IterativeSettingsTest {
		vector<uint> nTemps = { 4 };
		vector < uint> nIters = { 10 };
		vector<double> T0 = { 1 };
		vector<double> alpha = { 0.2 };
		vector<double> sigma = { 0.02 };
	};
	struct CostVolumeSettingsTest {
		vector < uint> nLabels = { 90 };
	};
	struct MethodsTest {
		vector<CorrespondenceCost> correspondenceCost = { CorrespondenceCost::Variance };
		vector<DepthMethod> depthMethods = { DepthMethod::iterative };
	};



	struct TestingParameters {
		IterativeSettingsTest iterSettings = {};
		CostVolumeSettingsTest basicSettings = {};
		MethodsTest methodsSettings = {};

		TestingParameters(MethodsTest methodsSettings,IterativeSettingsTest iterSettings) :
			methodsSettings(methodsSettings),
			iterSettings(iterSettings) {}
		TestingParameters(MethodsTest methodsSettings,CostVolumeSettingsTest basicSettings) :
			methodsSettings(methodsSettings),
			basicSettings(basicSettings) {}
		TestingParameters(MethodsTest methodsSettings, CostVolumeSettingsTest basicSettings, IterativeSettingsTest iterSettings) :
			methodsSettings(methodsSettings),
			iterSettings(iterSettings),
			basicSettings(basicSettings) {}
	};

	class TestDepthAlg
	{
		std::vector<lf::LightField> lfs;
		cv::String testDirectory;
		TestingParameters testingParameters;



	public:
		TestDepthAlg(std::vector<cv::String> LFpath,
			cv::String testDirectory,
			MethodsTest methodsSettings,
			IterativeSettingsTest iterSettings);

		TestDepthAlg(std::vector<cv::String> LFpath,
			cv::String testDirectory,
			MethodsTest methodsSettings,
			CostVolumeSettingsTest basicSettings);

		TestDepthAlg(std::vector<cv::String> LFpath,
			cv::String testDirectory,
			MethodsTest methodsSettings,
			CostVolumeSettingsTest basicSettings,
			IterativeSettingsTest iterSettings);

		void testIterative();
	};

}

