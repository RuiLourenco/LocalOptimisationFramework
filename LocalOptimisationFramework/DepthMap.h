#pragma once
#include <opencv2/core.hpp>
#include "LightField.h"
#include <vector>
#include "Collection.h"

namespace depthMap {
	class depthMap;
	enum class CorrespondenceCost {

		Variance,
		entropy,
		CAE,
		OAVCost,
		none

	};
	enum class DepthMethod {
		iterative,
		costVolume,
		groundTruth
	};
	struct IterativeSettings {
		uint nTemps = 4;
		uint nIters = 10;
		double T0 = 1;
		double alpha = 0.2;
		double sigma = 0.02;
	};
	struct BasicSettings {
		inline double dispResolution() { return (this->dispRange[1] - this->dispRange[0]) / (this->nLabels - 1); };


		uint nLabels = 10;
		std::array<double, 2> dispRange;
	};
	struct Methods {
		std::string correspondenceCostString() {
			switch (correspondenceCost) {
			case CorrespondenceCost::CAE:
				return "CAE";
			case CorrespondenceCost::Variance:
				return "Variance";
			case CorrespondenceCost::OAVCost:
				return "OAVC";
			case CorrespondenceCost::entropy:
				return "Entropy";
			}
		}

		static std::vector<DepthMethod> stringToDepthMethod(std::vector<cv::String> rawMethod) {
			std::vector<DepthMethod> depthMethods{};
			for (auto method : rawMethod) {
				if (!method.compare("LocalOptimisation")) {
					depthMethods.push_back(DepthMethod::iterative);
				}
				if (!method.compare("CostVolume")) {
					depthMethods.push_back(DepthMethod::costVolume);
				}
			}
			return depthMethods;
		}
		static std::vector<CorrespondenceCost> stringToCorrespondence(std::vector<cv::String> rawCorrespondence) {
			std::vector<CorrespondenceCost> correspondenceCosts{};
			for (auto cost : rawCorrespondence) {
				if (!cost.compare("Variance")) {
					correspondenceCosts.push_back(CorrespondenceCost::Variance);
				}
				if (!cost.compare("CAE")) {
					correspondenceCosts.push_back(CorrespondenceCost::CAE);
				}
				if (!cost.compare("OAVCost")) {
					correspondenceCosts.push_back(CorrespondenceCost::OAVCost);
				}
			}
			return correspondenceCosts;
		}
		CorrespondenceCost correspondenceCost = CorrespondenceCost::CAE;
		DepthMethod depthMethod = DepthMethod::iterative;
	};
	struct DepthSettings {
		//Optimization Default Settings Constructor

		DepthSettings() {
			methods = Methods{};
			basicSettings = BasicSettings{};
			iterativeSettings = IterativeSettings{};
			
		}
		DepthSettings(Methods methods, BasicSettings basicSettings):
			methods(methods),
			basicSettings(basicSettings){
			iterativeSettings = IterativeSettings{};
		}

		DepthSettings(Methods methods,IterativeSettings iterativeSettings) :
			methods(methods),
			iterativeSettings(iterativeSettings) {
			basicSettings = BasicSettings{};
		}


		Methods methods;
		BasicSettings basicSettings;
		IterativeSettings iterativeSettings;
	};


	class DepthMap {
	public:
		//constructors

		DepthMap(lf::LightField&, DepthSettings);
		cv::Mat data;
		//Build Ground Truth
		DepthMap(lf::LightField*, cv::String path);
		DepthMap() = default;
		//metrics
		double mse(int borderOffset = 0);
		double mseLabels();
		double badpix(double limit = 0.07, int borderSize = 0);
		//utility
		void show(cv::String = "DepthMap", bool showCost = false);
		void saveImage(cv::String path);
		void saveDiff(cv::String path);
		void writeCSV(cv::String filename);
		DepthSettings settings;

		~DepthMap();
	private:
		cv::Mat cost;

		lf::LightField* lightField;

		void calcCurrentCost();
		void iterativeDepthImprovement();
		void costVolumeMinimization();
		void depthFromStructureTensor();



		//CORRESPONDENCE COST FUNCTIONS
		double correspondenceCost(const std::array<vector<double>, 3>& angularPatch, std::array<uchar, 3> imageValue, std::array<double, 3> thresholdValue, CorrespondenceCost costType = CorrespondenceCost::Variance);
		
		//OAVC Cost
		static double OAVCost(const std::array<vector<double>, 3>& angularPatch, std::array<uchar, 3> imageValue, std::array<double, 3> thresholdValue);
		//OAVC COST HELPERS
		std::array<double, 3> getAdaptiveThreshold(cv::Mat* centralViewImage, uchar v, uchar u);

		//CAE COST
		static double constrainedAdaptiveEntropyCost(const std::vector<double>&, uchar);
		//CAE Helpers
		static double shannonEntropy(std::array<double, 256>, double);
		static double constrainedWeight(double value, double centralValue);
		static std::array<double, 256> buildHistogram(const vector<double>& angPatch);

		static double differenceCost(const std::array<vector<double>, 3>& angularPatch, std::array<uchar, 3> imageValue);
		
		static double getVarianceFromVector(const std::vector<double>&);
		static double shannonEntropyCost(const std::vector<double>&);


		//iterativeHelperFunctions

		
		void disparityFromLabels(col3D::Collection<int>& labelMap, double minDisp, double dispResolution, std::vector<double> labelSet);
		std::array<double, 2> getViewIntersection(std::array<int, 2>, std::array<int, 2>, std::array<int, 2>, double);
		std::array<double, 3> valueFromFractionalCoordinates(cv::Mat*, std::array<double, 2>);

		void showDiff();
		
		void filterCost(col3D::Collection<double>& correspondenceData, cv::Mat& view);
	
		static std::pair<uint, double>  getLabelMinCost(double* cost, uint nLab);
		void getLabelsFromCost(col3D::Collection<int>& labelMap, col3D::Collection<double>& costCorr);
		
		void setLightFieldReference(lf::LightField*);
	};

}
