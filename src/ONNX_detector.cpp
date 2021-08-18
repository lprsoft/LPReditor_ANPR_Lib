/*
************************************************************************
// Copyright (C) 2021, Raphael Poulenard.
************************************************************************
// Line.h: interface for the C_Line class.
//
This program is free software : you can redistribute itand /or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
GNU General Public License for more details.
//////////////////////////////////////////////////////////////////////
*/
#include "ONNX_detector.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <filesystem>
OnnxDetector::OnnxDetector(Ort::Env& env_, const ORTCHAR_T* model_path, const Ort::SessionOptions& options) : env(env_), session(env_, model_path, options) {
	dump();
}
OnnxDetector::OnnxDetector(Ort::Env& env_, const void* model_data, size_t model_data_length, const Ort::SessionOptions& options) : env(env_),
session(env_, model_data, model_data_length, options)
{
	dump();
}
void OnnxDetector::dump() const {
	std::cout << "Available execution providers:\n";
	for (const auto& s : Ort::GetAvailableProviders()) std::cout << '\t' << s << '\n';
	Ort::AllocatorWithDefaultOptions allocator;
	size_t numInputNodes = session.GetInputCount();
	size_t numOutputNodes = session.GetOutputCount();
	std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
	std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
	const char* inputName = session.GetInputName(0, allocator);
	std::cout << "Input Name: " << inputName << std::endl;
	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
	std::cout << "Input Type: " << inputType << std::endl;
	std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
	//std::cout << "Input Dimensions: " << inputDims << std::endl;
	for (size_t i = 0; i < inputDims.size() - 1; i++)
		std::cout << inputDims[i] << std::endl;
	std::cout << std::endl;
	const char* outputName = session.GetOutputName(0, allocator);
	std::cout << "Output Name: " << outputName << std::endl;
	Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
	auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
	std::cout << "Output Type: " << outputType << std::endl;
	std::vector<int64_t> outputDims = outputTensorInfo.GetShape();//1 25200 41
#ifdef _DEBUG
	assert(outputDims.size() == 3);
	assert(outputDims[0] == 1);
	assert(outputDims[2] == 103);// 0,1,2,3 ->box,4->confidence，1 -> output classes = 36 characters+pays 61 + 1 vehicle= 98 classes =4+1+36+61+1=103
#endif //_DEBUG
	std::cout << "Output Dimensions: " << std::endl;
	for (size_t i = 0; i < outputDims.size(); i++)
		std::cout << outputDims[i] << std::endl;
	std::cout << std::endl;
}
//returns the maximum size of input image (ie width or height of dnn input layer)
int64_t OnnxDetector::max_image_size() const {
	std::vector<std::vector<Detection>> result;
	cv::Mat resizedImageRGB, resizedImage, preprocessedImage;
	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
	std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
	int64_t max_size = inputDims.at(2);// width;
	if (max_size < inputDims.at(3))//height
		max_size = inputDims.at(3);
	return max_size;
}
std::vector<std::vector<Detection>>
OnnxDetector::Run(const cv::Mat& img, float conf_threshold, float iou_threshold, bool preserve_aspect_ratio) {
	std::vector<std::vector<Detection>> result;
	cv::Mat resizedImageRGB, resizedImage, preprocessedImage;
	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
	std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
	//cv::Mat img = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
	int channels_ = img.channels();
	if (
		img.size().width &&
		img.size().height && ((channels_ == 1) || (channels_ == 3) || (channels_ == 4))) {
		if (channels_ == 1) {
			cv::cvtColor(img, resizedImageRGB,
				cv::ColorConversionCodes::COLOR_GRAY2RGB);
		}
		else if (channels_ == 4) {
			cv::cvtColor(img, resizedImageRGB,
				cv::ColorConversionCodes::COLOR_BGRA2RGB);
		}
		else if (channels_ == 3) {
			int type = img.type();
			cv::cvtColor(img, resizedImageRGB,
				cv::ColorConversionCodes::COLOR_BGR2RGB);
		}
		float pad_w = -1.0f, pad_h = -1.0f, scale = -1.0f;
		if (preserve_aspect_ratio) {
			// keep the original image for visualization purpose
			std::vector<float> pad_info = LetterboxImage(resizedImageRGB, resizedImageRGB, cv::Size(int(inputDims.at(2)), int(inputDims.at(3))));
			//pad_w is the left (and also right) border width in the square image feeded to the model
			pad_w = pad_info[0];
			pad_h = pad_info[1];
			scale = pad_info[2];
		}
		else {
			cv::resize(resizedImageRGB, resizedImageRGB,
				 cv::Size(int(inputDims.at(2)), int(inputDims.at(3))),
				cv::InterpolationFlags::INTER_CUBIC);
		}
		resizedImageRGB.convertTo(resizedImage, CV_32FC3, 1.0f / 255.0f);
		// HWC to CHW
		cv::dnn::blobFromImage(resizedImage, preprocessedImage);
		int64_t inputTensorSize = vectorProduct(inputDims);
		std::vector<float> inputTensorValues(inputTensorSize);
		inputTensorValues.assign(preprocessedImage.begin<float>(),
			preprocessedImage.end<float>());
		Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
		auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();//1
#ifdef _DEBUG
		assert(outputType == 1);
#endif //_DEBUG
		std::vector<int64_t> outputDims = outputTensorInfo.GetShape();//1 25200 41
#ifdef _DEBUG
		assert(outputDims.size() == 3);
		assert(outputDims[0] == 1);
		//assert(outputDims[1] == 25200);
		assert(outputDims[2] == 103);// 0,1,2,3 ->box,4->confidence，1 -> output classes = 36 characters+pays 61 + 1 vehicle= 98 classes =4+1+36+61+1=103
#endif //_DEBUG
		int64_t outputTensorSize = vectorProduct(outputDims);
		std::vector<float> outputTensorValues(outputTensorSize);
		Ort::AllocatorWithDefaultOptions allocator;
		const char* inputName = session.GetInputName(0, allocator);
		const char* outputName = session.GetOutputName(0, allocator);
		std::vector<const char*> inputNames{ inputName };
		std::vector<const char*> outputNames{ outputName };
		std::vector<Ort::Value> inputTensors;
		std::vector<Ort::Value> outputTensors;
		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
			OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
		inputTensors.push_back(Ort::Value::CreateTensor<float>(
			memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
			inputDims.size()));
		outputTensors.push_back(Ort::Value::CreateTensor<float>(
			memoryInfo, outputTensorValues.data(), outputTensorSize,
			outputDims.data(), outputDims.size()));
		// https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L353
		session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), inputTensors.data(), 1, outputNames.data(), outputTensors.data(), 1);
		size_t dimensionsCount = outputTensorInfo.GetDimensionsCount();//3
#ifdef _DEBUG
		assert(dimensionsCount == 3);
#endif //_DEBUG
		float* output = outputTensors[0].GetTensorMutableData<float>(); // output of onnx runtime ->>> 1,25200,85
		size_t size = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount(); // 1x25200x85=2142000
		int64_t dimensions = outputDims[2]; // 0,1,2,3 ->box,4->confidence，5-85 -> output classes = 35 characters+pays 60 = 95 classes confidence 
#ifdef _DEBUG
		assert(dimensions >= 41);// 0,1,2,3 ->box,4->confidence，5-85 -> output classes = 35 characters+pays 60 = 95 classes confidence 
#endif //_DEBUG
		const cv::Size& out_size = cv::Size(int(inputDims[3]), int(inputDims[3]));
		std::vector<std::vector<Detection>> detections;
		if (preserve_aspect_ratio) {
			// keep the original image for visualization purpose
			detections = (
				PostProcessing(
					output, // output of onnx runtime ->>> 1,25200,85
					dimensionsCount,
					size, // 1x25200x85=2142000
					int(dimensions),
					//pad_w is the left (and also right) border width in the square image feeded to the model
					pad_w, pad_h, scale, img.size(),
					conf_threshold, iou_threshold));
		}
		else {
			detections = (PostProcessing(
				output, // output of onnx runtime ->>> 1,25200,85
				dimensionsCount,
				size, // 1x25200x85=2142000
				int(dimensions),
				float(out_size.width), float(out_size.height), img.size(),
				conf_threshold, iou_threshold));
		}
#ifdef _DEBUG
		std::list<cv::Rect> true_boxes; std::list<int> classesId;
		std::vector<Detection>::const_iterator it(detections[0].begin());
		while (it != detections[0].end()) {
			true_boxes.push_back(it->bbox);
			classesId.push_back(it->class_idx);
			it++;
		}
		//show_boxes(img, true_boxes, classesId);
#endif //_DEBUG
		return detections;
	}
	return result;
}
std::list<std::vector<std::vector<Detection>>>
OnnxDetector::Run(const cv::Mat& img
	//, float conf_threshold
	, float iou_threshold
) {
	std::list<std::vector<std::vector<Detection>>> result;
	cv::Mat resizedImageRGB, resizedImage, preprocessedImage;
	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
	std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
	//cv::Mat img = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
	int channels_ = img.channels();
	if (
		img.size().width &&
		img.size().height && ((channels_ == 1) || (channels_ == 3) || (channels_ == 4))) {
		if (channels_ == 1) {
			cv::cvtColor(img, resizedImageRGB,
				cv::ColorConversionCodes::COLOR_GRAY2RGB);
		}
		else if (channels_ == 4) {
			cv::cvtColor(img, resizedImageRGB,
				cv::ColorConversionCodes::COLOR_BGRA2RGB);
		}
		else if (channels_ == 3) {
			int type = img.type();
			cv::cvtColor(img, resizedImageRGB,
				cv::ColorConversionCodes::COLOR_BGR2RGB);
		}
		bool preserve_aspect_ratio = true;
		float pad_w = -1.0f, pad_h = -1.0f, scale = -1.0f;
		// keep the original image for visualization purpose
		std::vector<float> pad_info = LetterboxImage(resizedImageRGB, resizedImageRGB,  cv::Size(int(inputDims.at(2)), int(inputDims.at(3))));
		//pad_w is the left (and also right) border width in the square image feeded to the model
		pad_w = pad_info[0];
		pad_h = pad_info[1];
		scale = pad_info[2];
		resizedImageRGB.convertTo(resizedImage, CV_32FC3, 1.0f / 255.0f);
		// HWC to CHW
		cv::dnn::blobFromImage(resizedImage, preprocessedImage);
		int64_t inputTensorSize = vectorProduct(inputDims);
		std::vector<float> inputTensorValues(inputTensorSize);
		inputTensorValues.assign(preprocessedImage.begin<float>(),
			preprocessedImage.end<float>());
		Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
		auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();//1
#ifdef _DEBUG
		assert(outputType == 1);
#endif //_DEBUG
		std::vector<int64_t> outputDims = outputTensorInfo.GetShape();//1 25200 41
#ifdef _DEBUG
		assert(outputDims.size() == 3);
		assert(outputDims[0] == 1);
		//assert(outputDims[1] == 25200);
		assert(outputDims[2] == 103);// 0,1,2,3 ->box,4->confidence，1 -> output classes = 36 characters+pays 61 + 1 vehicle= 98 classes =4+1+36+61+1=103
#endif //_DEBUG
		int64_t outputTensorSize = vectorProduct(outputDims);
		std::vector<float> outputTensorValues(outputTensorSize);
		Ort::AllocatorWithDefaultOptions allocator;
		const char* inputName = session.GetInputName(0, allocator);
		const char* outputName = session.GetOutputName(0, allocator);
		std::vector<const char*> inputNames{ inputName };
		std::vector<const char*> outputNames{ outputName };
		std::vector<Ort::Value> inputTensors;
		std::vector<Ort::Value> outputTensors;
		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
			OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
		inputTensors.push_back(Ort::Value::CreateTensor<float>(
			memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
			inputDims.size()));
		outputTensors.push_back(Ort::Value::CreateTensor<float>(
			memoryInfo, outputTensorValues.data(), outputTensorSize,
			outputDims.data(), outputDims.size()));
		// https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L353
		session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), inputTensors.data(), 1, outputNames.data(), outputTensors.data(), 1);
		size_t dimensionsCount = outputTensorInfo.GetDimensionsCount();//3
#ifdef _DEBUG
		assert(dimensionsCount == 3);
#endif //_DEBUG
		float* output = outputTensors[0].GetTensorMutableData<float>(); // output of onnx runtime ->>> 1,25200,85
		size_t size = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount(); // 1x25200x85=2142000
		int64_t dimensions = outputDims[2]; // 0,1,2,3 ->box,4->confidence，5-85 -> output classes = 35 characters+pays 60 = 95 classes confidence 
#ifdef _DEBUG
		assert(dimensions >= 41);// 0,1,2,3 ->box,4->confidence，5-85 -> output classes = 35 characters+pays 60 = 95 classes confidence 
#endif //_DEBUG
		const cv::Size& out_size = cv::Size(int(inputDims[3]), int(inputDims[3]));
		const int nb_passes = 10;
		const float conf_threshold_min = 0.1f;
		for (int conf_threshold_step = 0; conf_threshold_step < nb_passes; conf_threshold_step++)
		{
			float conf_threshold = conf_threshold_min + float(conf_threshold_step) * (1.0f - 2.0f * conf_threshold_min) / (float(nb_passes));
			// keep the original image for visualization purpose
			std::vector<std::vector<Detection>> current_detections = (
				PostProcessing(
					output, // output of onnx runtime ->>> 1,25200,85
					dimensionsCount,
					size, // 1x25200x85=2142000
					int(dimensions),
					//pad_w is the left (and also right) border width in the square image feeded to the model
					pad_w, pad_h, scale, img.size(),
					conf_threshold, iou_threshold));
			result.push_back(current_detections);
		}
	}
	return result;
}
void nms(const std::vector<cv::Rect>& srcRects, std::vector<cv::Rect>& resRects, std::vector<int>& resIndexs, float thresh) {
	resRects.clear();
	const size_t size = srcRects.size();
	if (!size) return;
	// Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
	std::multimap<int, size_t> idxs;
	for (size_t i = 0; i < size; ++i) {
		idxs.insert(std::pair<int, size_t>(srcRects[i].br().y, i));
	}
	// keep looping while some indexes still remain in the indexes list
	while (idxs.size() > 0) {
		// grab the last rectangle
		auto lastElem = --std::end(idxs);
		const cv::Rect& last = srcRects[lastElem->second];
		resIndexs.push_back(int(lastElem->second));
		resRects.push_back(last);
		idxs.erase(lastElem);
		for (auto pos = std::begin(idxs); pos != std::end(idxs); ) {
			// grab the current rectangle
			const cv::Rect& current = srcRects[pos->second];
			float intArea =float( (last & current).area());
			float unionArea = last.area() + current.area() - intArea;
			float overlap = intArea / unionArea;
			// if there is sufficient overlap, suppress the current bounding box
			if (overlap > thresh) pos = idxs.erase(pos);
			else ++pos;
		}
	}
}