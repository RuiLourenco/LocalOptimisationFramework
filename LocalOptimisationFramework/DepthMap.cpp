#include "DepthMap.h"


//#include "Collection3D.h"
#include <iostream>
#include <cmath>
#include <math.h> 
#include <numeric> 
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <fstream>
#include <valarray>   
#include <random>
#include <chrono>


#define CONST_GAMMA 0.07
#define CONST_BETA 0.9






using namespace std;
using namespace cv;
using namespace depthMap;

using col3D::Collection;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

bool hasMadeFirstIteration = false;

inline double arrayEuclidean(array<double, 3>vec1, array<double, 3>vec2) {
	double accum = 0;
	for (uint i = 0; i < 3; i++) {
		accum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
	}
	return accum;
}

double label2Disp(uint label, array<double, 2> range, double resolution) {
	return range[0] + label * resolution;
}








inline uint8_t get_value(cv::Mat const& img, int32_t row, int32_t col, int32_t channel)
{
	CV_DbgAssert(channel < img.channels());
	uint8_t const* pixel_ptr(img.ptr(row, col));
	uint8_t const* value_ptr(pixel_ptr + channel);
	return *value_ptr;
}

/********************************************************************************** HELPERS *********************************************************************/
double label2Disp(int label, DepthSettings settings) {
	return settings.basicSettings.dispRange[0] + label * settings.basicSettings.dispResolution();
}




void calcDepthFromStructureTensor(Mat& inputEPI, Mat& disparityOut) {
	Mat img;
	int w = 7;
	inputEPI.convertTo(img, CV_32F);
	inputEPI /= 255;
	// GST components calculation (start)
	// J =  (J11 J12; J12 J22) - GST
	Mat imgDiffX, imgDiffY, imgDiffXY;
	Sobel(img, imgDiffX, CV_32F, 1, 0, 3);
	Sobel(img, imgDiffY, CV_32F, 0, 1, 3);
	multiply(imgDiffX, imgDiffY, imgDiffXY);
	Mat imgDiffXX, imgDiffYY;
	multiply(imgDiffX, imgDiffX, imgDiffXX);
	multiply(imgDiffY, imgDiffY, imgDiffYY);
	Mat J11, J22, J12;      // J11, J22 and J12 are GST components
	boxFilter(imgDiffXX, J11, CV_32F, Size(w, w));
	boxFilter(imgDiffYY, J22, CV_32F, Size(w, w));
	boxFilter(imgDiffXY, J12, CV_32F, Size(w, w));
	// GST components calculation (stop)
	// eigenvalue calculation (start)
	// lambda1 = 0.5*(J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2))
	// lambda2 = 0.5*(J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2))
	Mat tmp = J22 - J11;
	Mat tmp2;
	sqrt(tmp.mul(tmp) + 4 * J12.mul(J12), tmp2);
	Mat deltaX = tmp + tmp2;
	Mat deltaS = 2 * J12;
	deltaS = deltaS;
	deltaX = deltaX;
	disparityOut = deltaX / deltaS;
	cv::patchNaNs(disparityOut, 0.0);
	disparityOut.convertTo(disparityOut, CV_64F);
	// orientation angle calculation (stop)
}
void DepthMap::depthFromStructureTensor() {
	array<uint, 2> spacialSize = this->lightField->getSpacialSize();
	uint width = spacialSize[1];
	uint height = spacialSize[0];
	this->data = cv::Mat(height, width, CV_64F, 0.0);

	for (uint v = 0; v < height; v++) {

		Mat EPI = this->lightField->getEpi(4, v, true);
		Mat EPI_GS;

		try {
			cv::cvtColor(EPI, EPI_GS, cv::COLOR_BGR2GRAY);
		}
		catch (cv::Exception& e) {
			cout << e.what() << endl;
		}
		Mat disparity = cv::Mat(9, width, CV_32F);
		calcDepthFromStructureTensor(EPI, disparity);

		for (uint u = 0; u < width; u++) {
			double d = *disparity.ptr<double>(4, u);
			if (!isfinite(d)) d = 0;
			this->data.at<double>(v, u) = d;

			//cout << "(" << v << "," << u << ") " << *disparity.ptr<double>(4, u)<<endl;
		}
		cout << "line: " << v << "		\r";
	}
	this->data = max(min(this->data, this->settings.basicSettings.dispRange[1]), this->settings.basicSettings.dispRange[0]);
	//this->show();
	//
}



/********************************************************************************* CONSTRUCTORS ****************************************************************/
void DepthMap::setLightFieldReference(lf::LightField* lfReference) {
	this->lightField = lfReference;
}

DepthMap::DepthMap(lf::LightField* lightField, cv::String path) : lightField(lightField) {
	this->settings = DepthSettings{};
	this->settings.methods.depthMethod = DepthMethod::groundTruth;

	Mat gt = cv::imread(path, cv::IMREAD_UNCHANGED);
	gt.convertTo(this->data, CV_64F);
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(gt, &minVal, &maxVal, &minLoc, &maxLoc);
	this->settings.basicSettings.dispRange = { minVal,maxVal };
}

DepthMap::DepthMap(lf::LightField& lightField, DepthSettings settings) :lightField(&lightField), settings(settings) {

	settings.basicSettings.dispRange = this->lightField->getGT()->settings.basicSettings.dispRange;
	this->settings = settings;
	if (settings.methods.depthMethod == DepthMethod::iterative) {
		depthFromStructureTensor();
		calcCurrentCost();
		hasMadeFirstIteration = true;
		iterativeDepthImprovement();
	}
	else if (settings.methods.depthMethod == DepthMethod::costVolume) {
		costVolumeMinimization();
	}
}






void DepthMap::calcCurrentCost() {
	IterativeSettings iterSettings = this->settings.iterativeSettings;
	const int channelNumber = 3;
	array<uint, 2> angSize = this->lightField->getAngSize();
	array<uint, 2> spacialSize = this->lightField->getSpacialSize();
	cv::Mat* centralViewImage = this->lightField->getView(angSize[0] / 2, angSize[1] / 2);
	double minDisp = this->settings.basicSettings.dispRange[0];
	double maxDisp = this->settings.basicSettings.dispRange[1];
	//cv::Mat* centralViewImage = this->lightField->getView(4, 4);
	uint width = spacialSize[1];
	uint height = spacialSize[0];
	array<int, 2> centralView = { angSize[0] / 2 , angSize[1] / 2 };

	int vStart = 0;
	int uStart = 0;
	int vEnd = height;
	int uEnd = width;
	int increment = 1;
	if (this->cost.rows == 0) {
		this->cost = cv::Mat(height, width, CV_64F, 0.0);
	}
	for (int v = vStart; v != vEnd; v += increment) {
		double* d = this->data.ptr<double>(v) + uStart;
		double* c = this->cost.ptr<double>(v) + uStart;
		for (int u = uStart; u != uEnd; u += increment) {
			double currentDisp = *d;
			//set image Value
			array<uchar, 3> currentPixel;
			if (channelNumber == 3) {
				for (uint i = 0; i < channelNumber; i++) {
					currentPixel[i] = get_value(*centralViewImage, v, u, i);
				}
			}
			array<double, 3> thresholdValues = { 0 };
			if (this->settings.methods.correspondenceCost == CorrespondenceCost::OAVCost) {
				thresholdValues = DepthMap::getAdaptiveThreshold(centralViewImage, v, u);
			}


			
			array<vector<double>, 3> angularPatch;
			for (uint i = 0; i < channelNumber; i++) {
				angularPatch[i].reserve((long long int)angSize[0] * (long long int)angSize[1]);
			}
			
			for (int t = 0; t < angSize[0]; t++) {
				for (int s = 0; s < angSize[1]; s++) {

					array<double, 2> intersection = DepthMap::getViewIntersection({ t,s }, { v,u }, centralView, *d);
					if (intersection[0] >= 0 && intersection[1] >= 0 && intersection[1] <= spacialSize[1] - 1 && intersection[0] <= spacialSize[0] - 1) {
						array<double, 3> value;
						value = DepthMap::valueFromFractionalCoordinates(this->lightField->getView(t, s), intersection);
						for (uint i = 0; i < 3; i++) {
							angularPatch[i].push_back(value[i]);
						}
					}
				}
			}
			double cost = DepthMap::correspondenceCost(angularPatch, currentPixel, thresholdValues, settings.methods.correspondenceCost);

			*c = cost;

			d += increment;
			c += increment;

		}
		cout << "line: " << v << "\r";
	}
}
void DepthMap::iterativeDepthImprovement() {
	IterativeSettings iterSettings = this->settings.iterativeSettings;
	double currMse = this->mse(15);
	cout << "iter: " << -1 << " mse = " << currMse << endl << endl;

	array<uint, 2> angSize = this->lightField->getAngSize();
	array<uint, 2> spacialSize = this->lightField->getSpacialSize();
	cv::Mat* centralViewImage = this->lightField->getView(angSize[0] / 2, angSize[1] / 2);
	double minDisp = this->settings.basicSettings.dispRange[0];
	double maxDisp = this->settings.basicSettings.dispRange[1];
	uint width = spacialSize[1];
	uint height = spacialSize[0];
	array<int, 2> centralView = { angSize[0] / 2 , angSize[1] / 2 };

	double sigmaWeight = iterSettings.sigma;
	double sigma = sigmaWeight * abs(maxDisp - minDisp);

	std::normal_distribution<double> norm(0, sigma);
	std::uniform_real_distribution<double> lottery(0, 1);
	std::default_random_engine re;
	const int channelNumber = 3;

	int tempIterations = iterSettings.nIters;
	int nTemps = iterSettings.nTemps;
	double T0 = iterSettings.T0;
	double T = T0;
	double alpha = iterSettings.alpha;

	double iterCost = 0;
	//main iteration loop
	for (double t_iter = 0; t_iter < nTemps; t_iter++) {
		cout << T << endl;

		for (int i = 0; i < tempIterations; ++i) {

			iterCost = 0;

			int trueIterCount = (i + tempIterations * t_iter);

			

			int vStart = 0;
			int uStart = 0;
			int vEnd = height;
			int uEnd = width;
			int increment = 1;
			if (i % 2 == 1) {
				vStart = height - 1;
				vEnd = 0 - 1;
				uStart = width - 1;
				uEnd = 0 - 1;
				increment = -1;
			}
			

			for (int v = vStart; v != vEnd; v += increment) {
				double* d = this->data.ptr<double>(v) + uStart;
				double* c = this->cost.ptr<double>(v) + uStart;
				for (int u = uStart; u != uEnd; u += increment) {
					double currentDisp = *d;
					double currentCost = *c;
					
					//set image Value
					array<uchar, 3> currentPixel;
					for (uint i = 0; i < channelNumber; i++) {
						currentPixel[i] = get_value(*centralViewImage, v, u, i);
					}
					vector<double> disparityCandidates;
					vector<double> candidateCosts;

					double r = norm(re);
					double perturbedDisp = currentDisp + r;

					//handle the case when the perturbed disparity is out of bounds
					if (perturbedDisp > maxDisp) {
						int overflowTimes = (perturbedDisp / maxDisp);
						perturbedDisp = maxDisp - (perturbedDisp - (overflowTimes)*maxDisp);
					}
					else if (perturbedDisp < minDisp) {
						int overflowTimes = (perturbedDisp / minDisp);
						perturbedDisp = minDisp + (minDisp - (overflowTimes)*perturbedDisp);
					}
					double diff = abs(perturbedDisp - currentDisp);

					//store in candidate disparities vector
					disparityCandidates.push_back(perturbedDisp);
					//Store Already Estimated Neighbors in the disparity candidates vector
					if (v > 0) {
						if (i % 2 == 0) {
							double dispUp = *(d - width);
							disparityCandidates.push_back(dispUp);
						}
					}
					if (v < height - 1) {
						if (i % 2 == 1) {
							double dispDown = *(d + width);
							disparityCandidates.push_back(dispDown);
						}
					}
					if (u > 0) {
						if (i % 2 == 0) {
							double dispLeft = *(d - 1);
							disparityCandidates.push_back(dispLeft);
						}
					}
					if (u < width - 1) {
						if (i % 2 == 1) {
							double dispRight = *(d + 1);
							disparityCandidates.push_back(dispRight);
						}
					}

					array<double, 3> thresholdValues = { 0 };
					if (this->settings.methods.correspondenceCost == CorrespondenceCost::OAVCost) {
						thresholdValues = DepthMap::getAdaptiveThreshold(centralViewImage, v, u);
					}


					for (auto candidateDisparity : disparityCandidates) {
						array<vector<double>, 3> occAwareAngularPatch;
						array<vector<double>, 3> angularPatch;
						for (uint i = 0; i < channelNumber; i++) {
							angularPatch[i].reserve((long long int)angSize[0] * (long long int)angSize[1]);
						}

						for (int t = 0; t < angSize[0]; t++) {
							for (int s = 0; s < angSize[1]; s++) {
								array<double, 2> intersection = DepthMap::getViewIntersection({ t,s }, { v,u }, centralView, candidateDisparity);
								if (intersection[0] >= 0 && intersection[1] >= 0 && intersection[1] <= spacialSize[1] - 1 && intersection[0] <= spacialSize[0] - 1) {
									array<double, 3> value;
									value = DepthMap::valueFromFractionalCoordinates(this->lightField->getView(t, s), intersection);
									for (uint i = 0; i < 3; i++) {
										angularPatch[i].push_back(value[i]);
									}
								}
							}
						}
						double cost = DepthMap::correspondenceCost(angularPatch, currentPixel, thresholdValues, settings.methods.correspondenceCost);
						candidateCosts.push_back(cost);
					}

					double averageDataDelta = 0;
					double averageNormalDelta = 0;


					int minElementIndex = std::min_element(candidateCosts.begin(), candidateCosts.end()) - candidateCosts.begin();
					double newCost = candidateCosts[minElementIndex];
					double newDisp = disparityCandidates[minElementIndex];


					double delta = exp(-(newCost - currentCost) / T);
					double randomChance = lottery(re);
					if (delta > randomChance) {
						iterCost += newCost;
						*d = newDisp;
						*c = newCost;
					}
					else {
						iterCost += currentCost;
					}
					d += increment;
					c += increment;
				}
				cout << "line: " << v << "   " << "\r";
			}

			double minVal;
			double maxVal;
			Point minLoc;
			Point maxLoc;

			double currMse = this->mse(15);
			double currBP = this->badpix(0.07, 15);
			cout << "iter: " << i << " mse = " << currMse << " badpix = " << currBP << " average cost = " << iterCost / (width * height) << endl << endl;
		}
		T = T0 * pow(alpha, t_iter + 1);
	}
}

/*************************************************************Metrics************************************************************/
double DepthMap::mse(int borderOffset) {
	array<uint, 2> size = this->lightField->getSpacialSize();
	double borderSize = borderOffset;
	std::shared_ptr<DepthMap> groundTruth = this->lightField->getGT();
	Rect roi = Rect(borderSize, borderSize, size[1] - borderSize * 2, size[0] - borderSize * 2);
	Mat gtRoi(groundTruth->data, roi);
	Mat dmRoi(this->data, roi);
	Mat tmp;
	absdiff(dmRoi, gtRoi, tmp);
	tmp = tmp.mul(tmp);
	Scalar s = sum(tmp);
	double sse = s.val[0];
	double mse = sse / dmRoi.total();
	return mse * 100;
}

double DepthMap::badpix(double limit, int borderSize) {
	array<uint, 2> size = this->lightField->getSpacialSize();
	std::shared_ptr<DepthMap> groundTruth = this->lightField->getGT();
	Rect roi = Rect(borderSize, borderSize, size[1] - borderSize * 2, size[0] - borderSize * 2);
	Mat gtRoi(groundTruth->data, roi);
	Mat dmRoi(this->data, roi);
	Mat tmp;
	absdiff(dmRoi, gtRoi, tmp);
	int total = dmRoi.total();
	int accum = 0;
	for (int i = 0; i < size[0] - 2 * borderSize; i++) {
		for (int j = 0; j < size[1] - 2 * borderSize; j++) {
			double gtValue = gtRoi.at<double>(i, j);
			double dispValue = dmRoi.at<double>(i, j);
			double diff = tmp.at<double>(i, j);
			if (diff > limit) {
				tmp.at<double>(i, j) = 1.0;
				accum++;
			}
			else {
				tmp.at<double>(i, j) = 0.0;
			}
		}
	}
	return accum / (double)total;
}


/********************************************************Disparity Estimation Code*************************************************/
//GraphCut Optimized Disparity Estimation
void DepthMap::costVolumeMinimization() {

	//Initializations

	cout << "Initializing Variables" << endl;


	double minDisp = settings.basicSettings.dispRange[0];
	double maxDisp = settings.basicSettings.dispRange[1];
	vector<double> labelVector;
	int nLab = this->settings.basicSettings.nLabels;
	cout << nLab << endl;
	double dispResolution = (maxDisp - minDisp) / ((double)nLab - 1);
	uint channelNumber = this->lightField->channelNumber();
	array<uint, 2> angSize = this->lightField->getAngSize();
	cv::Mat* centralViewImage = this->lightField->getView(angSize[0] / 2, angSize[1] / 2);

	array<uint, 2> spacialSize = this->lightField->getSpacialSize();


	labelVector.reserve(nLab);
	for (int i = 0; i < nLab; i++) {
		labelVector.push_back(minDisp + i * dispResolution);
	}

	vector<double> fullLabelVector = labelVector;
	settings = this->settings;
	

	array<uint, 2> aux = { angSize[0] / 2,angSize[1] / 2 };
	array<int, 2> centralView = { aux[0] + 1, aux[1] + 1 };
 
	int numPixels = (spacialSize[0] * spacialSize[1]);
	uint width = spacialSize[1];
	uint height = spacialSize[0];
	this->data = cv::Mat(height, width, CV_64F, 0.0);
	this->cost = cv::Mat(height, width, CV_64F, 0.0);


	Collection<double> correspondenceCostVolume(height, width, nLab);;
	Collection<int> labelMap(height, width, 1);
	

	int l_current = 0;
	int l_initial = 0;
	uint iterMax = 1;
	int l = 0;
	Collection<double> unbiasedIntersection(labelVector.size(), angSize[0], angSize[1], 2);

	cout << "Starting PRE-CALC View Intersections" << endl;
	for (auto d : labelVector) {
		//double d = minDisp + l * dispResolution;
		for (int t = 0; t < angSize[0]; t++) {
			for (int s = 0; s < angSize[1]; s++) {


				array<int, 2> view = { t + 1,s + 1 };
				array<double, 2> intersection = DepthMap::getViewIntersection(view, { 0,0 }, centralView, d);

				unbiasedIntersection(l, t, s, 0) = intersection[0];
				unbiasedIntersection(l, t, s, 1) = intersection[1];
				//cout << intersection[0] << " " << intersection[1] << endl;
			}
		}
		l++;
	}

	cout << "Starting Cost Estimation Loop" << endl;
	for (int v = 0; v < height; v++) {
		for (int u = 0; u < width; u++) {
			uchar imageValue;

			array<uchar, 3> imageValues;
			if (channelNumber == 3) {
				for (uint i = 0; i < channelNumber; i++) {
					imageValues[i] = get_value(*centralViewImage, v, u, i);
				}
			}
			array<double, 3> thresholdValues;
			if (settings.methods.correspondenceCost == CorrespondenceCost::OAVCost) {
				thresholdValues = DepthMap::getAdaptiveThreshold(centralViewImage, v, u);
			}
			l = 0;

			for (l_current = l_initial; l_current < nLab; l_current++) {
					
				array<vector<double>, 3> angularPatch;

				for (uint i = 0; i < channelNumber; i++) {
					angularPatch[i].reserve((long long int)angSize[0] * (long long int)angSize[1]);
				}
				
				for (int t = 0; t < angSize[0]; t++) {
					for (int s = 0; s < angSize[1]; s++) {
						//cout << s << " " << t << " ";
						array<double, 2> intersection;
						intersection[0] = v + unbiasedIntersection(l, t, s, 0);
						intersection[1] = u + unbiasedIntersection(l, t, s, 1);

						//cout << t<< " "<< s<< " "<<intersection[0] << " " << intersection[1] << endl;
						if (intersection[0] >= 0 && intersection[1] >= 0 && intersection[1] <= spacialSize[1] - 1 && intersection[0] <= spacialSize[0] - 1) {

							array<double, 3> value;
							value = DepthMap::valueFromFractionalCoordinates(this->lightField->getView(t, s), intersection);
							for (uint i = 0; i < 3; i++) {
								angularPatch[i].push_back(value[i]);
							}
						}
					}
				}
				double correspondenceCost = 0;
				correspondenceCost = DepthMap::correspondenceCost(angularPatch, imageValues, thresholdValues, settings.methods.correspondenceCost);		
				correspondenceCostVolume(v, u, l_current) = correspondenceCost;
				l++;
			}
		}

		cout << "line: " << v << "\r";
	}

	DepthMap::getLabelsFromCost(labelMap, correspondenceCostVolume);
	DepthMap::disparityFromLabels(labelMap, minDisp, dispResolution, labelVector);


}

void DepthMap::filterCost(Collection<double>& correspondenceData, Mat& centralView) {
	for (int l = 0; l < correspondenceData.size[2]; l++) {
		Mat src = cv::Mat(correspondenceData.size[0], correspondenceData.size[1], CV_64F, 0.0);
		Mat dst = cv::Mat(correspondenceData.size[0], correspondenceData.size[1], CV_32F, 0.0);
		Mat joint = cv::Mat(correspondenceData.size[0], correspondenceData.size[1], CV_32F, 0.0);
		correspondenceData.toOpenCVMAT(l, src);
		double minVal;
		double maxVal;
		Point minLoc;
		Point maxLoc;
		src.convertTo(src, CV_32F, 1);
		minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);
		try {
			centralView.convertTo(joint, CV_32F, 1);
			joint = joint / 255;
			minMaxLoc(joint, &minVal, &maxVal);

			cout << joint.size() << " " << src.size() << endl;
			cv::ximgproc::jointBilateralFilter(joint, src, dst, 7, 0.1, 3);
		}
		catch (cv::Exception& e) {
			cout << e.what() << endl;
		}
		dst.convertTo(dst, CV_64F, 1);
		correspondenceData.fromOpenCVMAT(l, dst);
	}
}

void DepthMap::disparityFromLabels(Collection<int>& labelMap, double minDisp, double dispResolution, vector<double>labelSet) {

	for (int v = 0; v < labelMap.size[0]; v++) {
		for (int u = 0; u < labelMap.size[1]; u++) {
			int label = labelMap(v , u , 0);
			double disparity = 0;
			disparity = labelSet[label];
			*this->data.ptr<double>(v, u) = disparity;
			cout << label<<" "<<disparity << endl;
		}
	}
}

double DepthMap::correspondenceCost(const array<vector<double>, 3>& angularPatch, array<uchar, 3> imageValue, array<double, 3> thresholdValue, CorrespondenceCost costType) {
	double correspondenceCost = 0;
	switch (costType) {
	case CorrespondenceCost::Variance:
		for (int c = 0; c < 3; c++) {
			correspondenceCost += DepthMap::getVarianceFromVector(angularPatch[c]);
			correspondenceCost /= 3;
		}
		break;
	case CorrespondenceCost::entropy:
		for (int c = 0; c < 3; c++) {
			correspondenceCost = DepthMap::shannonEntropyCost(angularPatch[c]);
			correspondenceCost /= 3;
		}
		break;
	case CorrespondenceCost::CAE:
		for (int c = 0; c < 3; c++) {
			correspondenceCost = DepthMap::constrainedAdaptiveEntropyCost(angularPatch[c], imageValue[c]);
			correspondenceCost /= 3;
		}
		break;
	case CorrespondenceCost::OAVCost:
		correspondenceCost = DepthMap::OAVCost(angularPatch, imageValue, thresholdValue);
		break;
	default:
		correspondenceCost = 1.0;
	}
	return correspondenceCost;
}
double DepthMap::differenceCost(const array < vector<double>, 3>& angularPatch, array<uchar, 3> imageValue) {
	double accum = 0;
	int count = 0;
	for (int a = 0; a < angularPatch[0].size(); a++) {
		count++;
		double error = 0;
		for (int c = 0; c < 3; c++) {
			double pixel = angularPatch[c][a];
			error += abs(pixel - imageValue[c]);
		}
		error = error / 3;
		accum += error;
	}
	double cost = accum / (count - 1);
	return cost;
}
array<double, 3> DepthMap::getAdaptiveThreshold(cv::Mat* centralViewImage, uchar v, uchar u) {
	double deltaEpsilon = 0.1;
	double vStart = max(v - (4 * deltaEpsilon), 0.01);
	double uStart = max(u - (4 * deltaEpsilon), 0.01);
	double uEnd = min(u + (4 * deltaEpsilon), centralViewImage->size[1] - 1.0);
	double vEnd = min(v + (4 * deltaEpsilon), centralViewImage->size[0] - 1.0);

	int count = 0;
	array<double, 3> threshold = { 0 };
	array<double, 3> accum = { 0 };
	array<double, 3> refValue = DepthMap::valueFromFractionalCoordinates(centralViewImage, { (double)v,(double)u });

	for (double vIter = vStart; vIter <= vEnd; vIter += deltaEpsilon) {
		for (double uIter = uStart; uIter <= uEnd; uIter += deltaEpsilon) {
			array<double, 3> currValue = DepthMap::valueFromFractionalCoordinates(centralViewImage, { vIter,uIter });
			for (int c = 0; c < 3; c++) {
				accum[c] += abs(refValue[c] - currValue[c]);
			}
			count++;
		}
	}
	for (int c = 0; c < 3; c++) {
		threshold[c] = max(min(1.0 / (count - 1) * accum[c], 1.275), 0.51);
	}

	return threshold;
}

double DepthMap::OAVCost(const array < vector<double>, 3>& angularPatch, array<uchar, 3> imageValue, array<double, 3> thresholdValue) {
	
	double threshold = std::accumulate(thresholdValue.begin(), thresholdValue.end(), 0.0) / 3.0;
	int vote = 81 - angularPatch[0].size();
	double accum = 0;
	int count = 0;
	for (int a = 0; a < angularPatch[0].size(); a++) {
		count++;
		double error = 0;
		for (int c = 0; c < 3; c++) {
			double pixel = angularPatch[c][a];
			error += abs(pixel - imageValue[c]);
		}
		error = error / 3;
		if (error > threshold) {
			vote++;
		}
		else {
			accum += error / 255;
		}
	}

	double drawCost = accum / (count - 1);
	double cost = vote + drawCost;
	return cost;
}

array<double, 3> DepthMap::valueFromFractionalCoordinates(cv::Mat* view, array<double, 2> fractionalCoordinates) {

	//Correct Way

	uint threshold = 255;

	double uLow = floor(fractionalCoordinates[1]);
	double uHigh = uLow + 1;
	double vLow = floor(fractionalCoordinates[0]);
	double vHigh = vLow + 1;
	double b = 1;

	double wHor = (1 - 2 * b) * (fractionalCoordinates[1] - uLow) + b;
	double wVer = (1 - 2 * b) * (fractionalCoordinates[0] - vLow) + b;
	array<double, 3> value;
	for (uint i = 0; i < view->channels(); i++) {
		double valueLL = get_value(*view, uint(vLow), uint(uLow), i);
		double valueHL = get_value(*view, uint(vLow), uint(uHigh), i);
		double valueLH = get_value(*view, uint(vHigh), uint(uLow), i);
		double valueHH = get_value(*view, uint(vHigh), uint(uHigh), i);

		double value_1;
		double value_2;
		if (abs(valueLL - valueHL) > threshold) {
			if (wHor > 0.5) {
				value_1 = valueLL;
			}
			else {
				value_1 = valueHL;
			}
		}
		else {
			value_1 = wHor * valueLL + (1 - wHor) * valueHL;
		}
		if (abs(valueLH - valueHH) > threshold) {
			if (wHor > 0.5) {
				value_2 = valueLH;
			}
			else {
				value_2 = valueHH;
			}
		}
		else {
			value_2 = wHor * valueLH + (1 - wHor) * valueHH;
		}
		if (abs(value_1 - value_2) > threshold) {
			if (wVer >= 0.5) {
				value[i] = value_1;
			}
			else {
				value[i] = value_2;
			}
		}
		else {
			value[i] = wVer * value_1 + (1 - wVer) * value_2;
		}

	}
	return value;
}


double DepthMap::getVarianceFromVector(const vector<double>& vector) {
	double sum = std::accumulate(vector.begin(), vector.end(), 0.0);

	double m = sum / vector.size();
	double accum = 0;
	std::for_each(vector.begin(), vector.end(), [&](const double d) {
		accum += (d - m) * (d - m);
		});
	double variance = accum / (vector.size() - 1);
	return variance;
}

array<double, 2> DepthMap::getViewIntersection(array<int, 2> view, array<int, 2> referenceViewIntersection, array<int, 2> referenceView, double disparity) {
	double u = -disparity * (double(view[1] - referenceView[1])) + double(referenceViewIntersection[1]);
	double v = -disparity * (double(view[0] - referenceView[0])) + double(referenceViewIntersection[0]);

	array<double, 2> viewIntersection = { v,u };
	return viewIntersection;
}

void DepthMap::getLabelsFromCost(Collection<int>& labelMap, Collection<double>& costCorr) {
	uint nLab = settings.basicSettings.nLabels;
	uint	width = costCorr.size[1];
	uint	height = costCorr.size[0];

	
	uint numPixels = width * height;
	double* costs = costCorr.data;
	
	for (int k = 0; k < numPixels; k++) {

		double* costsLabel = costs + k * nLab;
		pair<uint, double> labelAndCost = DepthMap::getLabelMinCost(costsLabel, nLab);
		this->cost.at<double>(k) = labelAndCost.second;
		labelMap(k) = labelAndCost.first;
	}
}
pair<uint, double> DepthMap::getLabelMinCost(double* cost, uint nLab) {
	uint minIndex = 0;
	double minValue = 999999;
	double maxValue = 0;
	std::vector<uint> lowCosts;
	for (uint l = 0; l < nLab; l++) {
		double currentValue = *(cost + l);
		if (currentValue < minValue) {
			minValue = currentValue;
			minIndex = l;
			
		}
		if (currentValue > maxValue) {
			maxValue = currentValue;
		}

	}

	return std::make_pair(minIndex, minValue);
}

void DepthMap::show(String windowName, bool showCost) {
	Mat depthMap = this->data;
	if (depthMap.empty()) {
		cout << "Attempting to show empty DepthMap" << endl;
		//wait for any key press
		cin.get();
	}
	cv::normalize(depthMap, depthMap, 0, 255, NORM_MINMAX, CV_8UC1);
	namedWindow(windowName);
	imshow(windowName, depthMap);
	waitKey(0);
	if (showCost) {
		Mat cost = this->cost;
		if (cost.empty()) {
			cout << "Attempting to show empty Cost" << endl;
			//wait for any key press
			cin.get();
		}
		cv::normalize(cost, cost, 0, 255, NORM_MINMAX, CV_8UC1);
		namedWindow(windowName);
		imshow(windowName, cost);
		waitKey(0);
	}
}

void DepthMap::showDiff() {
	Mat depthMap = this->data;
	Mat gt = this->lightField->getGT()->data;
	Mat diff;
	absdiff(depthMap, gt, diff);
	for (int i = 0; i < diff.rows; i++) {
		for (int j = 0; j < diff.cols; j++) {
			double dif = diff.at<double>(i, j);
			if (dif > 0.2) {
				diff.at<double>(i, j) = 0.2;
			}

		}
	}
	if (diff.empty()) {
		cout << "Attempting to show empty difference image" << endl;
		//wait for any key press
		cin.get();
	}
	cv::normalize(diff, diff, 0, 255, NORM_MINMAX, CV_8UC1);
	namedWindow("Difference");
	imshow("Difference", diff);
	waitKey(0);
}

void DepthMap::saveDiff(String path) {
	Mat depthMap = this->data;
	Mat gt = this->lightField->getGT()->data;
	Mat diff;
	absdiff(depthMap, gt, diff);
	for (int i = 0; i < diff.rows; i++) {
		for (int j = 0; j < diff.cols; j++) {
			double dif = diff.at<double>(i, j);
			if (dif > 0.2) {
				diff.at<double>(i, j) = 0.2;
			}

		}
	}
	if (diff.empty()) {
		cout << "Attempting to show empty DepthMap" << endl;
		//wait for any key press
		cin.get();
	}
	cv::normalize(diff, diff, 0, 255, NORM_MINMAX, CV_8UC1);
	imwrite(path, diff); 
}

void DepthMap::saveImage(String path) {
	Mat depthMap = this->data;
	if (depthMap.empty()) {
		cout << "Attempting to Save empty Depth Map" << endl;
		//wait for any key press
		cin.get();
	}
	cv::normalize(depthMap, depthMap, 0, 255, NORM_MINMAX, CV_8UC1);
	//DepthMap::show("TEST!");
	imwrite(path, depthMap); //
}

void DepthMap::writeCSV(String filename)
{
	ofstream myfile;
	myfile.open(filename.c_str());
	myfile << cv::format(this->data, cv::Formatter::FMT_CSV) << std::endl;
	myfile.close();
}


array<double, 256> DepthMap::buildHistogram(const vector<double>& angPatch) {
	vector<int> integerAngPatch(angPatch.size());
	array<int, 256> count = { 0 };
	array<double, 256> histogram = { 0.0 };
	for (double a : angPatch) {
		count[(int)round(a)]++;
	}
	for (int i = 0; i < 256; i++) {
		histogram[i] = (double)count[i] / (double)angPatch.size();
		//cout << "Count = "<<count[i] << " Size = " << angPatch.size()<< "Probability =  " << histogram[i]<<endl;
	}
	return histogram;
}
double DepthMap::shannonEntropyCost(const vector<double>& angPatch) {
	array<double, 256> histogram = DepthMap::buildHistogram(angPatch);

	int nlen = angPatch.size();

	double shannonEntropy = DepthMap::shannonEntropy((array<double, 256>)histogram, 1);

	return shannonEntropy;
}

double DepthMap::shannonEntropy(array<double, 256> histogram, double normWeight) {

	float accum = 0;
	double frequency;
	for (double freq : histogram) {
		if (freq > 0) {
			frequency = (freq) / normWeight;
			accum += frequency * log2(freq);
		}
	}
	return -accum;
}

double DepthMap::constrainedWeight(double intensity, double centralIntensity) {
	double sigma = 10;
	double weight = exp(-((intensity - centralIntensity) * (intensity - centralIntensity)) / (2 * (sigma * sigma)));
	return weight;
}

double DepthMap::constrainedAdaptiveEntropyCost(const vector<double>& angPatch, uchar currentValue) {
	array<double, 256> histogram = DepthMap::buildHistogram(angPatch);
	array<double, 256> constrainedHistogram;
	double accum = 0;
	double weight;
	for (int i = 0; i < 256; i++) {
		weight = DepthMap::constrainedWeight(i, currentValue);

		constrainedHistogram[i] = weight * histogram[i];
		accum += constrainedHistogram[i];
	}
	return DepthMap::shannonEntropy(constrainedHistogram, accum);
}








double log2(double number) {
	return log(number) / log(2);
}


DepthMap::~DepthMap() {

}