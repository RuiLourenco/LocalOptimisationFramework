#include "Collection.h"

using namespace col3D;
inline Collection<int>::Collection(uint height, uint width, uint levels, uint channelNumber) {
	this->size[0] = height;
	this->size[1] = width;
	this->size[2] = levels;
	this->size[3] = channelNumber;

	this->data = new int[(long long)height * width * levels * channelNumber];
}

inline void Collection<int>::reserve(uint height, uint width, uint levels, uint channelNumber) {
	this->size[0] = height;
	this->size[1] = width;
	this->size[2] = levels;
	this->size[3] = channelNumber;

	this->data = new int[(long long)height * width * levels * channelNumber];
}
inline col3D::Collection<double>::Collection(uint height, uint width, uint levels, uint channelNumber) {
	this->size[0] = height;
	this->size[1] = width;
	this->size[2] = levels;
	this->size[3] = channelNumber;
	//std::cout <<height<<" "<<width<<" "<<levels<<" "<< (long long) height * width * levels * channelNumber<<" THIS IS A MATRIXXXX"<<std::endl;
	this->data = new double[(long long)height * width * levels * channelNumber];
	this->data = new double[(long long)height * width * levels * channelNumber];
}
inline void Collection<double>::reserve(uint height, uint width, uint levels, uint channelNumber) {
	this->size[0] = height;
	this->size[1] = width;
	this->size[2] = levels;
	this->size[3] = channelNumber;

	this->data = new double[(long long)height * width * levels * channelNumber];
}