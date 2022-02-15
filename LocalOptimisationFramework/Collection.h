#pragma once

#ifndef COL3D
#define COL3D
#include <iostream>
#include <opencv2/core.hpp>
#include <vector>
typedef unsigned int uint;
namespace col3D {
	template <class NumberType> class Collection {



	public:
		NumberType* data = NULL;
		uint size[4];

		Collection<NumberType>() = default;
		Collection<NumberType>(uint height, uint width, uint levels, uint channelNumber = 1);
		int cardinality();
		void reserve(uint height, uint width, uint levels, uint channelNumber = 1);
		void expand(uint h, uint w, uint l);
		NumberType& operator()(uint n, uint m, uint l, uint c = 0);
		NumberType& operator()(uint index);
		NumberType& index(uint n, uint m, uint l = 0, uint c = 0);
		NumberType& index(uint index);
		std::vector<NumberType> toVec();
		void toOpenCVMAT(uint level, cv::Mat& dst);
		void fromOpenCVMAT(uint level, cv::Mat& dst);
		void crop(std::array<uint, 3> start, std::array<uint, 3> end);
		~Collection();
	};

	template<class NumberType> __forceinline void Collection<NumberType>::expand(uint h, uint w, uint l) {



		uint height = this->size[0] + h;
		uint width = this->size[1] + w;
		uint labels = this->size[2] + l;
		uint channelNumber = this->size[3];
		uint oldCardinality = this->size[0] * this->size[1] * this->size[2] * this->size[3];
		NumberType* newData = new double[(long long)height * width * labels * channelNumber];
		for (uint v = 0; v < this->size[0]; v++) {
			for (uint u = 0; u < this->size[1]; u++) {
				for (uint l = 0; l < this->size[2]; l++) {
					for (uint c = 0; c < this->size[3]; c++) {
						uint index_new = ((((v * width + u) * labels) + l) * channelNumber) + c;
						newData[index_new] = this->index(v, u, l, c);
					}
				}
			}
		}
		this->size[0] = height; this->size[1] = width; this->size[2] = labels; this->size[3] = channelNumber;
		NumberType* cleanUp = this->data;
		this->data = newData;

		delete[] cleanUp;
	}
	template <class NumberType> __forceinline int Collection<NumberType>::cardinality() {
		return size[0] * size[1] * size[2] * size[3];
	}
	template<class NumberType> __forceinline std::vector<NumberType> Collection<NumberType>::toVec() {
		std::vector<NumberType> vecData(this->data, this->data + this->cardinality());
		return vecData;
	}
	template<class NumberType> __forceinline void Collection<NumberType>::toOpenCVMAT(uint level, cv::Mat& dst) {
		for (int n = 0; n < this->size[0]; n++) {
			for (int m = 0; m < this->size[1]; m++) {
				dst.at<NumberType>(n, m) = this->index(n, m, level);
			}
		}
	}
	template<class NumberType> __forceinline void Collection<NumberType>::fromOpenCVMAT(uint level, cv::Mat& src) {
		for (int n = 0; n < this->size[0]; n++) {
			for (int m = 0; m < this->size[1]; m++) {
				this->index(n, m, level) = src.at<NumberType>(n, m);
			}
		}
	}


	template<class NumberType> __forceinline  void  Collection<NumberType>::crop(std::array<uint, 3> start, std::array<uint, 3> end) {
		uint height = this->size[0] - start[0] - (this->size[0] - end[0]);
		uint width = this->size[1] - start[1] - (this->size[1] - end[1]);
		uint labels = this->size[2] - start[2] - (this->size[2] - end[2]);
		uint channelNumber = this->size[3];

		for (uint v = 0; v < height; v++) {
			for (uint u = 0; u < width; u++) {
				for (uint l = 0; l < labels; l++) {
					for (uint c = 0; c < channelNumber; c++) {
						uint index_new = ((((v * width + u) * labels) + l) * this->size[3]) + c;
						this->index(index_new) = this->index(v + start[0], u + start[1], l + start[2], c);
					}
				}
			}

		}
		this->size[0] = height; this->size[1] = width; this->size[2] = labels;
	}


	template<class NumberType>__forceinline NumberType& Collection<NumberType>::operator()(uint n, uint m, uint l, uint c) {
		return index(n, m, l, c);
	}
	template<class NumberType> __forceinline NumberType& Collection<NumberType>::operator()(uint ind) {
		return index(ind);
	}
	template<class NumberType> __forceinline NumberType& Collection<NumberType>::index(uint n, uint m, uint l, uint c) {
		uint index = ((((n * this->size[1] + m) * this->size[2]) + l) * this->size[3]) + c;
		return this->data[index];
	}
	template<class NumberType> __forceinline NumberType& Collection<NumberType>::index(uint index) {
		return this->data[index];
	}
	template<class NumberType> __forceinline Collection<NumberType>::~Collection<NumberType>() {
		if (this->size[0] || this->size[1] || this->size[2] || this->size[3]) {

			delete[] this->data;
		}


	}
}
#endif

