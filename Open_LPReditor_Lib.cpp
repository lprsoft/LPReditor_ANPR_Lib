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
//#include Python.h   
std::mutex mtx;           // mutex for critical section
//extern std::unique_ptr<Ort::Env> ort_env;
//step 1 declare a global instance of ONNX Runtime api
const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
std::list<Ort::Env*> detectors_envs;
std::list<Ort::SessionOptions*> l_detectors_sessionOptions;
std::list<Yolov5_anpr_onxx_detector*> detectors;
//detectors_ids : a list that contains the ids of all the detectors that are currently allocated by the library
std::list<size_t> detectors_ids;
std::list<Ort::Env*> plates_types_envs;
std::list<Ort::SessionOptions*> l_plates_types_classifier_sessionOptions;
std::list<Plates_types_classifier*> plates_types_classifiers;
//detectors_ids : a list that contains the ids of all the detectors that are currently allocated by the library
std::list<size_t> plates_types_classifier_ids;
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
	@brief this func gives a reference to a plates_types_classifier that is (uniquely) identified by its id.
	@param plates_types_classifiers_ids : a list that contains the ids of all the plates_types_classifiers that are currently allocated by the library
	@return an iterator in the list of plates_types_classifiers
	@see
	*/
std::list<Plates_types_classifier*>::const_iterator get_plates_types_classifier(size_t id, const std::list<Plates_types_classifier*>& plates_types_classifiers,
	const std::list<size_t>& plates_types_classifiers_ids
	//, const Yolov5_anpr_onxx_plates_types_classifier * * ref
) {
	assert(plates_types_classifiers_ids.size() == plates_types_classifiers.size());
	std::list<Plates_types_classifier*>::const_iterator it(plates_types_classifiers.begin());
	std::list<size_t>::const_iterator it_id(plates_types_classifiers_ids.begin());
	while (it != plates_types_classifiers.end() && it_id != plates_types_classifiers_ids.end()) {
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
	return plates_types_classifiers.end();
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
			if (*it != nullptr) delete* it;
			if (*it_sessionOptions != nullptr) delete* it_sessionOptions;
			if (*it_envs != nullptr) delete* it_envs;
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
bool close_detector(size_t id, std::list<Ort::Env*>& _envs, std::list<Ort::SessionOptions*>& _lsessionOptions, std::list<Plates_types_classifier*>& _detectors,
	std::list<size_t>& _detectors_ids) {
	assert(_detectors_ids.size() == _detectors.size()
		&& _detectors_ids.size() == _envs.size()
		&& _detectors_ids.size() == _lsessionOptions.size());
	std::list<Plates_types_classifier*>::iterator it(_detectors.begin());
	std::list<size_t>::iterator it_id(_detectors_ids.begin());
	std::list<Ort::SessionOptions*>::iterator it_sessionOptions(_lsessionOptions.begin());
	std::list<Ort::Env*>::iterator it_envs(_envs.begin());
	while (it != _detectors.end() && it_id != _detectors_ids.end()
		&& it_envs != _envs.end() && it_sessionOptions != _lsessionOptions.end()
		) {
		if (*it_id == id) {
			if (*it != nullptr) delete* it;
			if (*it_sessionOptions != nullptr) delete* it_sessionOptions;
			if (*it_envs != nullptr) delete* it_envs;
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
size_t init_yolo_detector(size_t len, const char* model_file)
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
					detectors_envs.push_back(penv);
					l_detectors_sessionOptions.push_back(psessionOptions);
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
/**
	@brief initializes a new plates type classifier by loading its model file and returns its unique id
	@param model_file : c string model filename (must be allocated by the calling program)
	@param len : length of the model filename
	@return the id of the new detector
	@see
	*/
extern "C"
#ifdef _WINDOWS
__declspec(dllexport)
#endif //_WINDOWS
size_t init_plates_classifer(size_t len_models_filename, const char* model_file, size_t len_labels_filename, const char* labels_file)
{
	assert(plates_types_classifier_ids.size() == plates_types_classifiers.size());
	const std::string model_filename(model_file, len_models_filename);
	const std::string labels_filename(labels_file, len_labels_filename);
	if (!model_filename.size() || !std::filesystem::exists(model_filename)
		|| !std::filesystem::is_regular_file(model_filename)
		|| !labels_filename.size() || !std::filesystem::exists(labels_filename)
		|| !std::filesystem::is_regular_file(labels_filename)
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
				Plates_types_classifier* onnx_net = nullptr;
#ifdef _WIN32
				//step 4 declare an onnx session (ie model), by giving references to the runtime environment, session options and file path to the model
				std::wstring widestr = std::wstring(model_filename.begin(), model_filename.end());
				onnx_net = new Plates_types_classifier(*penv, widestr.c_str(), *psessionOptions, labels_filename);
#else
				onnx_net = new Plates_types_classifier(*penv, model_filename.c_str(), *psessionOptions, labels_filename);
#endif
				if (onnx_net != nullptr && penv != nullptr && psessionOptions != nullptr) {
					std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
					lck.lock();
					plates_types_envs.push_back(penv);
					l_plates_types_classifier_sessionOptions.push_back(psessionOptions);
					plates_types_classifiers.push_back(onnx_net);
					size_t id = get_new_id(plates_types_classifier_ids);
					plates_types_classifier_ids.push_back(id);
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
extern "C"
#ifdef _WINDOWS
__declspec(dllexport)
#endif //_WINDOWS
bool two_stage_lpr
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
				(*it_global_view)->two_stage_lpr(*(*it_focused_on_lp), destMat, lpn_str);
				//for small plates
				lck.unlock();
			}
			else {
				std::cerr << "id_focused_on_lp " << id_focused_on_lp << " doesnot point to a valid detector" << std::endl;
				lck.lock();
				//for normal plates
				(*it_global_view)->two_stage_lpr(*(*it_global_view), destMat, lpn_str);
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
extern "C"
#ifdef _WINDOWS
__declspec(dllexport)
#endif //_WINDOWS
bool two_stage_lpr_plates_type_detection
(const int width,//width of image
	const int height,//height of image i.e. the specified dimensions of the image
	const int pixOpt,// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
	void* pbData, size_t step// source image bytes buffer
	, size_t id_global_view, size_t id_focused_on_lp, size_t id_plates_types_classifier, size_t lpn_len, char* lpn)
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
		std::list<Plates_types_classifier*>::const_iterator it_plates_types_classifier = get_plates_types_classifier(id_plates_types_classifier,
			plates_types_classifiers,plates_types_classifier_ids);
		if (it_plates_types_classifier != plates_types_classifiers.end()) {
			std::list<Yolov5_anpr_onxx_detector*>::const_iterator it_global_view = get_detector(id_global_view, detectors, detectors_ids);
			if (it_global_view != detectors.end()) {
				std::list<Yolov5_anpr_onxx_detector*>::const_iterator it_focused_on_lp = get_detector(id_focused_on_lp, detectors, detectors_ids);
				std::string lpn_str;
				std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
				if (it_focused_on_lp != detectors.end()) {
					lck.lock();
					//for normal plates
					(*it_global_view)->two_stage_lpr(*(*it_focused_on_lp), *(*it_plates_types_classifier), destMat, lpn_str);
					//for small plates
					lck.unlock();
				}
				else {
					std::cerr << "id_focused_on_lp " << id_focused_on_lp << " doesnot point to a valid detector" << std::endl;
					lck.lock();
					//for normal plates
					(*it_global_view)->two_stage_lpr(*(*it_global_view), *(*it_plates_types_classifier), destMat, lpn_str);
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
		else {
			std::cerr << "id_plates_types_classifier " << id_plates_types_classifier << " doesnot point to a valid detector" << std::endl;
			return  two_stage_lpr
			(width,//width of image
				height,//height of image i.e. the specified dimensions of the image
				pixOpt,// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
				pbData, step// source image bytes buffer
				, id_global_view, id_focused_on_lp, lpn_len, lpn);
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
	bool session_closed = close_detector(id, detectors_envs, l_detectors_sessionOptions, detectors, detectors_ids);
	lck.unlock();
	return session_closed;
}
/**
	@brief call this func once you have finished with the plates_types_classifier --> to free heap allocated memory
	@param id : unique interger to identify the plates_types_classifier to be freed
	@return true upon success
	@see
	*/
extern "C"
#ifdef _WINDOWS
__declspec(dllexport)
#endif //_WINDOWS
bool close_plates_types_classifier(size_t id)
{
	assert(plates_types_classifier_ids.size() == plates_types_classifiers.size());
	std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
	lck.lock();
	bool session_closed = close_detector(id, plates_types_envs, l_plates_types_classifier_sessionOptions, plates_types_classifiers, plates_types_classifier_ids);
	lck.unlock();
	return session_closed;
}