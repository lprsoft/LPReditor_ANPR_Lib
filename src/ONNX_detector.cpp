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
	//dump();
}
OnnxDetector::OnnxDetector(Ort::Env& env_, const void* model_data, size_t model_data_length, const Ort::SessionOptions& options) : env(env_),
session(env_, model_data, model_data_length, options)
{
	//dump();
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
	assert(outputDims[1] == 25200);
	assert(outputDims[2] == 41);
#endif //_DEBUG
	std::cout << "Output Dimensions: " << std::endl;
	for (size_t i = 0; i < outputDims.size(); i++)
		std::cout << outputDims[i] << std::endl;
	std::cout << std::endl;
}

std::vector<std::vector<Detection>>
OnnxDetector::Run(const cv::Mat& img, float conf_threshold, float iou_threshold, bool preserve_aspect_ratio) {


	std::vector<std::vector<Detection>> result;
	cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
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
			cv::cvtColor(img, resizedImageRGB,
				cv::ColorConversionCodes::COLOR_BGR2RGB);
		}


		float pad_w = -1.0f, pad_h = -1.0f, scale = -1.0f;
		if (preserve_aspect_ratio) {
			// keep the original image for visualization purpose
			std::vector<float> pad_info = LetterboxImage(resizedImageRGB, resizedImageRGB, cv::Size(inputDims.at(2), inputDims.at(3)));
			//pad_w is the left (and also right) border width in the square image feeded to the model
			pad_w = pad_info[0];
			pad_h = pad_info[1];
			scale = pad_info[2];



#ifdef _DEBUG
			/*
			bool show_image = true;
			const int time_delay = 4000;
			if (show_image && time_delay >= 0) {
				cv::imshow("PADDING", resizedImageRGB);
			}
			if (time_delay >= 0) {
				if (show_image && time_delay == 0) {
					char c = cv::waitKey(time_delay);
				}
				else if (time_delay > 0) {
					char c = cv::waitKey(time_delay);
				}
			}
			if (show_image && time_delay >= 0) {
				cv::destroyAllWindows();
			}
			*/
#endif //_DEBUG
		}







		/*
	cv::resize(img, resizedImageBGR,
		cv::Size(inputDims.at(2), inputDims.at(3)),
		cv::InterpolationFlags::INTER_CUBIC);
	cv::cvtColor(resizedImageBGR, resizedImageRGB,
		cv::ColorConversionCodes::COLOR_BGR2RGB);
	*/
		else {
			cv::resize(resizedImageRGB, resizedImageRGB,
				cv::Size(inputDims.at(2), inputDims.at(3)),
				cv::InterpolationFlags::INTER_CUBIC);
		}
		resizedImageRGB.convertTo(resizedImage, CV_32FC3, 1.0f / 255.0f);
		/*//this doesnot work for lpreditor
		cv::Mat channels[3];
		cv::split(resizedImage, channels);
		// Normalization per channel
		// Normalization parameters obtained from
		// https://github.com/onnx/models/tree/master/vision/classification/squeezenet
		channels[0] = (channels[0] - 0.485) / 0.229;
		channels[1] = (channels[1] - 0.456) / 0.224;
		channels[2] = (channels[2] - 0.406) / 0.225;
		cv::merge(channels, 3, resizedImage);
		*/
		// HWC to CHW
		cv::dnn::blobFromImage(resizedImage, preprocessedImage);
		size_t inputTensorSize = vectorProduct(inputDims);
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
		assert(outputDims[1] == 25200);
		assert(outputDims[2] == 41);
#endif //_DEBUG



		size_t outputTensorSize = vectorProduct(outputDims);
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
		session.Run(Ort::RunOptions{ nullptr }
			, inputNames.data(),
			inputTensors.data(), 1, outputNames.data(),
			outputTensors.data(), 1);
		size_t dimensionsCount = outputTensorInfo.GetDimensionsCount();//3
#ifdef _DEBUG
		assert(dimensionsCount == 3);
#endif //_DEBUG
		float* output = outputTensors[0].GetTensorMutableData<float>(); // output of onnx runtime ->>> 1,25200,85
		size_t size = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount(); // 1x25200x85=2142000
		int dimensions = outputDims[2]; // 0,1,2,3 ->box,4->confidence，5-85 -> coco classes confidence 
#ifdef _DEBUG
		assert(dimensions == 41);
#endif //_DEBUG

		const cv::Size& out_size = cv::Size(inputDims[3], inputDims[3]);
		std::vector<std::vector<Detection>> detections;
		if (preserve_aspect_ratio) {
			// keep the original image for visualization purpose
			detections=(
				PostProcessing(
					output, // output of onnx runtime ->>> 1,25200,85
					dimensionsCount,
					size, // 1x25200x85=2142000
					dimensions,
					//pad_w is the left (and also right) border width in the square image feeded to the model
					pad_w, pad_h, scale, img.size(),
					conf_threshold, iou_threshold));


		}
		else {
			detections=(PostProcessing(
				output, // output of onnx runtime ->>> 1,25200,85
				dimensionsCount,
				size, // 1x25200x85=2142000
				dimensions,
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



std::vector<std::tuple<cv::Rect, float, int>>
OnnxDetector::Run_v1(const cv::Mat& img, float conf_threshold, float iou_threshold) {
	std::vector<std::tuple<cv::Rect, float, int>> demo_data_vec;
	std::vector<std::vector<Detection>> result =
		Run(img, conf_threshold, iou_threshold, true);
	std::vector<std::vector<Detection>>::const_iterator it1(result.begin());
	while (it1 != result.end()) {
		std::vector<Detection>::const_iterator it2(it1->begin());
		while (it2 != it1->end()) {
			cv::Rect rect(it2->bbox);
			std::tuple<cv::Rect, float, int> t = std::make_tuple(it2->bbox, it2->score, it2->class_idx);
			demo_data_vec.emplace_back(t);
			it2++;
		}
		it1++;
	}
	return demo_data_vec;
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
		resIndexs.push_back(lastElem->second);
		resRects.push_back(last);
		idxs.erase(lastElem);
		for (auto pos = std::begin(idxs); pos != std::end(idxs); ) {
			// grab the current rectangle
			const cv::Rect& current = srcRects[pos->second];
			float intArea = (last & current).area();
			float unionArea = last.area() + current.area() - intArea;
			float overlap = intArea / unionArea;
			// if there is sufficient overlap, suppress the current bounding box
			if (overlap > thresh) pos = idxs.erase(pos);
			else ++pos;
		}
	}
}