#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>
#include "TestDepthAlg.h"
#include <chrono>




using namespace testDepth;
using namespace depthMap;
using namespace std;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
namespace fs = std::filesystem;
//namespace plt = matplotlibcpp;

TestDepthAlg::TestDepthAlg(vector<cv::String> lightFieldPaths,
	cv::String testDirectory,
	MethodsTest methodsSettings,
	IterativeSettingsTest iterSettings) : testingParameters(TestingParameters(methodsSettings, iterSettings)) {
	this->testDirectory = testDirectory;
	for (cv::String path : lightFieldPaths) {
		lf::LightField lightField = lf::LightField(path, true);
		this->lfs.push_back(lightField);
	}
}

TestDepthAlg::TestDepthAlg(vector<cv::String> lightFieldPaths,
	cv::String testDirectory,
	MethodsTest methodsSettings,
	CostVolumeSettingsTest costVolumeSettings) : testingParameters(TestingParameters(methodsSettings, costVolumeSettings)) {
	this->testDirectory = testDirectory;
	for (cv::String path : lightFieldPaths) {
		lf::LightField lightField = lf::LightField(path, true);
		this->lfs.push_back(lightField);
	}
}
TestDepthAlg::TestDepthAlg(vector<cv::String> lightFieldPaths,
	cv::String testDirectory,
	MethodsTest methodsSettings,
	CostVolumeSettingsTest costVolumeSettings,
	IterativeSettingsTest iterSettings) : testingParameters(TestingParameters(methodsSettings, costVolumeSettings, iterSettings )) {
	this->testDirectory = testDirectory;
	for (cv::String path : lightFieldPaths) {
		lf::LightField lightField = lf::LightField(path, true);
		this->lfs.push_back(lightField);
	}
}




void TestDepthAlg::testIterative() {
	std::vector<DepthMap> dms;

	fs::create_directory(this->testDirectory);

	cv::String resultFilePath = this->testDirectory + "\\ResultTable.csv";
	std::ofstream resultsFile(resultFilePath, fstream::app);
	CostVolumeSettingsTest basicSettingsTest = this->testingParameters.basicSettings;
	IterativeSettingsTest iterSettingsTest = this->testingParameters.iterSettings;
	MethodsTest methodsTest = this->testingParameters.methodsSettings;

	

	for (auto depthMethod : methodsTest.depthMethods) {
		if (depthMethod == DepthMethod::costVolume) {
			resultsFile << "LF,cost,nLabels,Time,MSE,Badpix" << endl;
			for (auto lightField : this->lfs) {
				for (auto cost : methodsTest.correspondenceCost) {
					Methods methods{ cost,depthMethod };
					for (auto nLabels : basicSettingsTest.nLabels) {
						BasicSettings basicSettings{ nLabels,{0,0} };
						ostringstream pngPath;
						ostringstream namePath;
						IterativeSettings iterSettings{};
						DepthSettings settings = DepthSettings(methods, basicSettings);
						auto timeBefore = high_resolution_clock::now();
						DepthMap depth = DepthMap(lightField, settings);
						auto timeAfter = high_resolution_clock::now();
						duration<double, std::milli> durationDispEstim = timeAfter - timeBefore;
						double mse = depth.mse(15);
						double badpix = depth.badpix(0.07, 15);
						namePath << this->testDirectory << "\\" << lightField.name << "_" << methods.correspondenceCostString() << "_" << nLabels;
						resultsFile << lightField.name << "," << methods.correspondenceCostString() << "," << nLabels << "," << (durationDispEstim.count() / 1000) << "," << mse << "," << badpix << endl;
						pngPath << namePath.str() << ".png";
						depth.saveImage(pngPath.str());

					}
				}
			}
			
		}
		else {
			resultsFile << "LF,cost,nIters,nTemps,T0,alpha,sigma,OcclusionAware,Time,MSE,Badpix" << endl;
			for (auto lightField : this->lfs) {
				for (auto cost : methodsTest.correspondenceCost) {
					Methods methods{ cost,depthMethod };
					for (auto nTemps : iterSettingsTest.nTemps) {
						for (auto nIters : iterSettingsTest.nIters) {
							for (auto T0 : iterSettingsTest.T0) {
								for (auto alpha : iterSettingsTest.alpha) {
									for (auto sigma : iterSettingsTest.sigma) {
										ostringstream pngPath;
										ostringstream pngNormalPath;
										ostringstream namePath;
										IterativeSettings iterSettings{ nTemps,nIters,T0,alpha,sigma };
										DepthSettings settings = DepthSettings(methods, iterSettings);
										auto timeBefore = high_resolution_clock::now();
										DepthMap depth = DepthMap(lightField, settings);
										auto timeAfter = high_resolution_clock::now();
										duration<double, std::milli> durationDispEstim = timeAfter - timeBefore;
										double mse = depth.mse(15);
										double badpix = depth.badpix(0.07, 15);
										namePath << this->testDirectory << "\\" << lightField.name << "_" << methods.correspondenceCostString() << "_" << nIters << "_" << nTemps << "_" << T0 << "_" << alpha << "_" << sigma;
										resultsFile << lightField.name << "," << methods.correspondenceCostString() << "," << nIters << "," << nTemps << "," << T0 << "," << alpha << "," << sigma  << "," << (durationDispEstim.count() / 1000) << "," << mse << "," << badpix << endl;
										pngPath << namePath.str() << ".png";
										pngNormalPath << namePath.str() << "_NORMALS.png";
										depth.saveImage(pngPath.str());
									}
								}
							}
						}
					}
				}
			}
		}
	}		
	resultsFile.close();
}