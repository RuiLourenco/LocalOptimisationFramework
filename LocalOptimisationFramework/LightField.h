#pragma once
#include <opencv2/core.hpp>
#include <vector>

using std::vector;

namespace depthMap {
	class DepthMap;
}

namespace lf {

	class LightField
	{
		cv::Mat data[9][9];
		vector<uint> size;
		std::shared_ptr<depthMap::DepthMap> gtPointer;

	public:
		std::string name;
		LightField(cv::String path_to_folder, bool isColor = true);
		void showView(uint, uint);
		void showHorizontalEPI(uint, uint);

		cv::Vec3b operator()(uint, uint, uint, uint);
		uint channelNumber();
		std::array<uint, 2> getAngSize();
		std::array<uint, 2> getSpacialSize();
		cv::Mat* getView(uint, uint);
		cv::Mat getEpi(uint angular, uint spacial, bool isHorizontal);
		cv::Mat* operator()(uint, uint);
		std::shared_ptr<depthMap::DepthMap>& getGT();
		~LightField();
	};
}

