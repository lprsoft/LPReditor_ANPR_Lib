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
third party software
c++ inference source code
OpenCV 4.5.0 and higher
Copyright © 2021 , OpenCV team
Apache 2 License
ONNXRUNTIME
Copyright © 2020 Microsoft. All rights reserved.
MIT License
model production
YOLOv5
by Glenn Jocher (Ultralytics.com)
GPL-3.0 License
onnx
Copyright (c) Facebook, Inc. and Microsoft Corporation. All rights reserved.
MIT License
*/
// Open_LPReditor.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include <assert.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "yolov5_anpr_onnx_detector.h"
#include "ONNX_detector.h"
#include <thread>         // std::thread
#include <mutex>          // std::mutex, std::unique_lock, std::defer_lock
std::mutex mtx;           // mutex for critical section
//extern std::unique_ptr<Ort::Env> ort_env;
//step 1 declare a global instance of ONNX Runtime api
const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
std::list<Ort::Env*> envs;
std::list<Ort::SessionOptions*> lsessionOptions;
std::list<Yolov5_anpr_onxx_detector*> detectors;
//detectors_ids : a list that contains the ids of all the detectors that are currently allocated by the library
std::list<size_t> detectors_ids;
//*****************************************************************************
// helper function to check for status
void CheckStatus(OrtStatus* status)
{
	if (status != NULL) {
		const char* msg = g_ort->GetErrorMessage(status);
		fprintf(stderr, "%s\n", msg);
		g_ort->ReleaseStatus(status);
		exit(1);
	}
}
/**
	@brief this func gives a reference to a detector that is (uniquely) identified by its id.
	@param detectors_ids : a list that contains the ids of all the detectors that are currently allocated by the library
	@return an iterator in the list of detectors
	@see
	*/
std::list<Yolov5_anpr_onxx_detector*>::const_iterator get_detector(size_t id, const std::list<Yolov5_anpr_onxx_detector*>& detectors,
	const std::list<size_t>& detectors_ids
	//, const Yolov5_anpr_onxx_detector * * ref
) {
	assert(detectors_ids.size() == detectors.size());
	std::list<Yolov5_anpr_onxx_detector*>::const_iterator it(detectors.begin());
	std::list<size_t>::const_iterator it_id(detectors_ids.begin());
	while (it != detectors.end() && it_id != detectors_ids.end()) {
		if (*it_id == id) {
			//ref= *(*it);
			//(*it)->dump();
			return it;
		}
		else {
			it_id++;
			it++;
		}
	}
	return detectors.end();
}
/**
	@brief this func gives a reference to a detector that is (uniquely) identified by its id.
	@param detectors_ids : a list that contains the ids of all the detectors that are currently allocated by the library
	@return an iterator in the list of detectors
	@see
	*/
Yolov5_anpr_onxx_detector* get_detector_ptr(size_t id, const std::list<Yolov5_anpr_onxx_detector*>& detectors,
	const std::list<size_t>& detectors_ids
) {
	assert(detectors_ids.size() == detectors.size());
	std::list<Yolov5_anpr_onxx_detector*>::const_iterator it(detectors.begin());
	std::list<size_t>::const_iterator it_id(detectors_ids.begin());
	while (it != detectors.end() && it_id != detectors_ids.end()) {
		if (*it_id == id) {
			//ref= *(*it);
			//(*it)->dump();
			return *it;
		}
		else {
			it_id++;
			it++;
		}
	}
	return nullptr;
}
/**
	@brief this func is used internally --> to get an unique interger to identify a new detector to be constructed
	@param detectors_ids : a list that contains the ids of all the detectors that are currently allocated by the library
	@return a new id
	@see
	*/
size_t get_new_id(const std::list<size_t>& detectors_ids) {
	if (detectors_ids.size()) {
		auto result = std::minmax_element(detectors_ids.begin(), detectors_ids.end());
		return *result.second + 1;
	}
	else return 1;
}
/**
	@brief this func is used internally --> to free heap allocated memeory
@param detectors_ids : a list that contains the ids of all the detectors that are currently allocated by the library
	@param id : unique interger to identify the detector to be freed
	@return true upon success
	@see
	*/
bool close_detector(size_t id, std::list<Ort::Env*>& _envs, std::list<Ort::SessionOptions*>& _lsessionOptions, std::list<Yolov5_anpr_onxx_detector*>& _detectors,
	std::list<size_t>& _detectors_ids) {
	assert(_detectors_ids.size() == _detectors.size()
		&& _detectors_ids.size() == _envs.size()
		&& _detectors_ids.size() == _lsessionOptions.size());
	std::list<Yolov5_anpr_onxx_detector*>::iterator it(_detectors.begin());
	std::list<size_t>::iterator it_id(_detectors_ids.begin());
	std::list<Ort::SessionOptions*>::iterator it_sessionOptions(_lsessionOptions.begin());
	std::list<Ort::Env*>::iterator it_envs(_envs.begin());
	while (it != _detectors.end() && it_id != _detectors_ids.end()
		&& it_envs != _envs.end() && it_sessionOptions != _lsessionOptions.end()
		) {
		if (*it_id == id) {
			if (*it_envs != nullptr) delete* it_envs;
			if (*it_sessionOptions != nullptr) delete* it_sessionOptions;
			if (*it != nullptr) delete* it;
			it_envs = _envs.erase(it_envs);
			it_sessionOptions = _lsessionOptions.erase(it_sessionOptions);
			it = _detectors.erase(it);
			it_id = _detectors_ids.erase(it_id);
			return true;
		}
		else {
			it_sessionOptions++;
			it_envs++;
			it_id++;
			it++;
		}
	}
	return false;
}
/**
	@brief initializes a new detector by loading its model file and returns its unique id
	@param model_file : c string model filename (must be allocated by the calling program)
	@param len : length of the model filename
	@return the id of the new detector
	@see
	*/
extern "C"
#ifdef _WINDOWS
__declspec(dllexport)
#endif //_WINDOWS
size_t init_detector(size_t len, const char* model_file)
{
	assert(detectors_ids.size() == detectors.size());
	const std::string model_filename(model_file, len);
	if (!model_filename.size() || !std::filesystem::exists(model_filename)
		|| !std::filesystem::is_regular_file(model_filename)
		)
	{
		std::cout << "model_filename " << model_filename << " is not a regular file " << std::endl;
		return 0;
	}
	//step 2 declare an onnx runtime environment
	std::string instanceName{ "image-classification-inference" };
	// https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L123
	Ort::Env* penv = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
	if (penv != nullptr) {
		//step 3 declare options for the runtime environment
		Ort::SessionOptions* psessionOptions = new Ort::SessionOptions();
		if (psessionOptions != nullptr) {
			psessionOptions->SetIntraOpNumThreads(1);
			// Sets graph optimization level
			// Available levels are
			// ORT_DISABLE_ALL -> To disable all optimizations
			// ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
			// removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
			// (Includes level 1 + more complex optimizations like node fusions)
			// ORT_ENABLE_ALL -> To Enable All possible optimizations
			psessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#ifdef LPR_EDITOR_USE_CUDA
			// Optionally add more execution providers via session_options
			// E.g. for CUDA include cuda_provider_factory.h and uncomment the following line:
			// nullptr for Status* indicates success
			OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(*psessionOptions, 0);
			//or status =nullptr; //if you don t have CUDA
			if (status == nullptr) {
#endif //LPR_EDITOR_USE_CUDA
				Yolov5_anpr_onxx_detector* onnx_net = nullptr;
#ifdef _WIN32
				//step 4 declare an onnx session (ie model), by giving references to the runtime environment, session options and file path to the model
				std::wstring widestr = std::wstring(model_filename.begin(), model_filename.end());
				onnx_net = new Yolov5_anpr_onxx_detector(*penv, widestr.c_str(), *psessionOptions);
#else
				onnx_net = new Yolov5_anpr_onxx_detector(*penv, model_filename.c_str(), *psessionOptions);
#endif
				if (onnx_net != nullptr && penv != nullptr && psessionOptions != nullptr) {
					std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
					lck.lock();
					envs.push_back(penv);
					lsessionOptions.push_back(psessionOptions);
					detectors.push_back(onnx_net);
					size_t id = get_new_id(detectors_ids);
					detectors_ids.push_back(id);
					lck.unlock();
					return id;
				}
				else {
					std::cout << "error while creating onnxruntime session with file : " << model_filename.c_str() << std::endl;
					return 0;
				}
#ifdef LPR_EDITOR_USE_CUDA
			}
			else {
				CheckStatus(status);
				std::cout << "cuda error " << std::endl;
				return 0;
			}
#endif //LPR_EDITOR_USE_CUDA
		}
		else {
			std::cout << "error while creating SessionOptions" << std::endl;
			return 0;
		}
	}
	else {
		std::cout << "error while creating session environment (Ort::Env)" << std::endl;
		return 0;
	}
}

#ifdef LPREDITOR_USE_ONE_STAGE_DETECTION

/**
	@brief detect_without_lpn_detection lpn in frame
	@param width : width of source image
	@param height : height of source image
	@param pixOpt : pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
	@param *pbData : source image bytes buffer
	param step Number of bytes each matrix row occupies.The value should include the padding bytes at
		the end of each row, if any.If the parameter is missing(set to AUTO_STEP), no padding is assumed
		and the actual step is calculated as cols* elemSize().See Mat::elemSize.
		@param id : unique interger to identify the detector to be used
@param ilpn: a c string allocated by the calling program
	@return true upon success
	@see
	*/
extern "C"
#ifdef _WINDOWS
__declspec(dllexport)
#endif //_WINDOWS
bool detect_without_lpn_detection
(const int width,//width of image
	const int height,//height of image i.e. the specified dimensions of the image
	const int pixOpt,// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
	void* pbData,
	size_t step// source image bytes buffer
	, size_t id, size_t lpn_len, char* lpn)
{
	if ((pixOpt != 1) && (pixOpt != 3) && (pixOpt != 4) || height <= 0 || width <= 0 || pbData == nullptr) {
		std::cerr << "condition on image (pixOpt != 1) && (pixOpt != 3) && (pixOpt != 4) || height <= 0 || width <= 0 || pbData == nullptr not met" << std::endl;
		return false;
	}
	else {
		cv::Mat destMat;
		if (pixOpt == 1)
		{
			destMat = cv::Mat(height, width, CV_8UC1, pbData, step);
		}
		if (pixOpt == 3)
		{
			destMat = cv::Mat(height, width, CV_8UC3, pbData, step);
		}
		if (pixOpt == 4)
		{
			destMat = cv::Mat(height, width, CV_8UC4, pbData, step);
		}
		std::list<Yolov5_anpr_onxx_detector*>::const_iterator it = get_detector(id, detectors, detectors_ids);
		if (it != detectors.end()) {
			std::string lpn_str;
			std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
			lck.lock();
			//for normal plates
			(*it)->evaluate_without_lpn_detection(destMat, lpn_str);
			//for small plates
			lck.unlock();
			std::string::const_iterator it_lpn(lpn_str.begin());
			int i = 0;
			while (it_lpn != lpn_str.end() && i < lpn_len - 1) {
				lpn[i] = *it_lpn;
				i++; it_lpn++;
			}
			while (i < lpn_len) {
				lpn[i] = '\0';
				i++;
			}
			return (lpn_str.length() > 0);
		}
		else {
			std::cerr << "id " << id << " doesnot point to a valid detector" << std::endl;
			return false;
		}
	}
}
#endif //LPREDITOR_USE_ONE_STAGE_DETECTION
extern "C"
#ifdef _WINDOWS
__declspec(dllexport)
#endif //_WINDOWS
bool detect_with_lpn_detection
(const int width,//width of image
	const int height,//height of image i.e. the specified dimensions of the image
	const int pixOpt,// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
	void* pbData, size_t step// source image bytes buffer
	, size_t id_global_view, size_t id_focused_on_lp, size_t lpn_len, char* lpn)
{
	if ((pixOpt != 1) && (pixOpt != 3) && (pixOpt != 4) || height <= 0 || width <= 0 || pbData == nullptr) {
		std::cerr << "condition on image (pixOpt != 1) && (pixOpt != 3) && (pixOpt != 4) || height <= 0 || width <= 0 || pbData == nullptr not met" << std::endl;
		return false;
	}
	else {
		cv::Mat destMat;
		if (pixOpt == 1)
		{
			destMat = cv::Mat(height, width, CV_8UC1, pbData, step);
		}
		if (pixOpt == 3)
		{
			destMat = cv::Mat(height, width, CV_8UC3, pbData, step);
		}
		if (pixOpt == 4)
		{
			destMat = cv::Mat(height, width, CV_8UC4, pbData, step);
		}
		std::list<Yolov5_anpr_onxx_detector*>::const_iterator it_global_view = get_detector(id_global_view, detectors, detectors_ids);
		if (it_global_view != detectors.end()) {
			std::list<Yolov5_anpr_onxx_detector*>::const_iterator it_focused_on_lp = get_detector(id_focused_on_lp, detectors, detectors_ids);
			std::string lpn_str;
			std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
			if (it_focused_on_lp != detectors.end()) {
				lck.lock();
				//for normal plates
				(*it_global_view)->evaluate_lpn_with_lpn_detection(*(*it_focused_on_lp), destMat, lpn_str);
				//for small plates
				lck.unlock();
			}
			else {
				std::cerr << "id_focused_on_lp " << id_focused_on_lp << " doesnot point to a valid detector" << std::endl;
				lck.lock();
				//for normal plates
				(*it_global_view)->evaluate_lpn_with_lpn_detection(*(*it_global_view), destMat, lpn_str);
				//for small plates
				lck.unlock();
			}
			std::string::const_iterator it_lpn(lpn_str.begin());
			int i = 0;
			while (it_lpn != lpn_str.end() && i < lpn_len - 1) {
				lpn[i] = *it_lpn;
				i++; it_lpn++;
			}
			while (i < lpn_len) {
				lpn[i] = '\0';
				i++;
			}
			return (lpn_str.length() > 0);
		}
		else {
			std::cerr << "id_global_view " << id_global_view << " doesnot point to a valid detector" << std::endl;
			return false;
		}
	}
}
/**
	@brief call this func once you have finished with the detector --> to free heap allocated memory
	@param id : unique interger to identify the detector to be freed
	@return true upon success
	@see
	*/
extern "C"
#ifdef _WINDOWS
__declspec(dllexport)
#endif //_WINDOWS
bool close_detector(size_t id)
{
	assert(detectors_ids.size() == detectors.size());
	std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
	lck.lock();
	bool session_closed = close_detector(id, envs, lsessionOptions, detectors, detectors_ids);
	lck.unlock();
	return session_closed;
}
