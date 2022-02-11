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
#include "yolov5_anpr_onnx_detector.h"
#include <assert.h> 
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "../include/utils_image_file.h"
#include"utils_opencv.h"
#define NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE 36
#define NUMBER_OF_COUNTRIES 61
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
Yolov5_anpr_onxx_detector::Yolov5_anpr_onxx_detector(Ort::Env& env, const ORTCHAR_T* model_path, const Ort::SessionOptions& options)
	: OnnxDetector(env, model_path, options)
{
}
Yolov5_anpr_onxx_detector::Yolov5_anpr_onxx_detector(Ort::Env& env, const void* model_data, size_t model_data_length, const Ort::SessionOptions& options)
	: OnnxDetector(env, model_data, model_data_length, options)
{
}
Yolov5_anpr_onxx_detector::~Yolov5_anpr_onxx_detector()
{
}
//Given the @p input frame, create input blob, run net then, from result detections, assembly license plates present in the input image.
//Produces double linked lists : inside list is for characters and outside list is for plates.
void Yolov5_anpr_onxx_detector::detect_with_different_confidences_then_separate_plates(const cv::Mat& frame, std::list<std::vector<int>>& classIds,
	std::list < std::vector<float>>& confidences, std::list < std::vector<cv::Rect>>& boxes,
	float nmsThreshold)
{
	//unlike the current engine that operates only on 35 characters (all the latin ones minus the O)
	bool preserve_aspect_ratio = true;
	std::list<std::vector<std::vector<Detection>>> detections_with_different_confidences = OnnxDetector::Run(frame, nmsThreshold);
	std::list<std::vector<std::vector<Detection>>>::const_iterator it_detections_with_different_confidences(detections_with_different_confidences.begin());
	while (it_detections_with_different_confidences != detections_with_different_confidences.end())
	{
		if (it_detections_with_different_confidences->size() == 1) {//batch 1 one image at a time
			classIds.push_back(std::vector<int>()); confidences.push_back(std::vector<float>()); boxes.push_back(std::vector<cv::Rect>());
			std::vector<Detection>::const_iterator it(it_detections_with_different_confidences->front().begin());
			while (it != it_detections_with_different_confidences->front().end())
			{
				classIds.back().push_back(it->class_idx);
				confidences.back().push_back(it->score);
				boxes.back().push_back(it->bbox);
				it++;
			}
		}
		it_detections_with_different_confidences++;
	}
}
// Given the @p input frame, create input blob, run net and return result detections.
void Yolov5_anpr_onxx_detector::raw_detections_with_different_confidences(const cv::Mat& frame, std::list<std::list<int>>& classIds,
	std::list < std::list<float>>& confidences, std::list < std::list<cv::Rect>>& boxes, float nmsThreshold)
{
	bool engine_dont_distinguish_O_and_0 = false;//from 2021 06 we want to switch to a new engine that operates, from scratch, on the 36 latin digits characters 
	//unlike the current engine that operates only on 35 characters (all the latin ones minus the O)
	bool preserve_aspect_ratio = true;
	std::list<std::vector<std::vector<Detection>>> detections_with_different_confidences = OnnxDetector::Run(frame, nmsThreshold);
	std::list<std::vector<std::vector<Detection>>>::const_iterator it_detections_with_different_confidences(detections_with_different_confidences.begin());
	while (it_detections_with_different_confidences != detections_with_different_confidences.end())
	{
		if (it_detections_with_different_confidences->size() == 1) {//batch 1 one image at a time
			classIds.push_back(std::list<int>()); confidences.push_back(std::list<float>()); boxes.push_back(std::list<cv::Rect>());
			std::vector<Detection>::const_iterator it(it_detections_with_different_confidences->front().begin());
			while (it != it_detections_with_different_confidences->front().end())
			{
				classIds.back().push_back(it->class_idx);
				confidences.back().push_back(it->score);
				boxes.back().push_back(it->bbox);
				it++;
			}
		}
		it_detections_with_different_confidences++;
	}
}//Given the @p input frame, create input blob, run net then, from result detections, assembly license plates present in the input image.
//Produces double linked lists : inside list is for characters and outside list is for plates.
void Yolov5_anpr_onxx_detector::detect_with_different_confidences_then_separate_plates(const cv::Mat& frame, std::list<std::vector<int>>& classIds,
	std::list < std::vector<float>>& confidences, std::list < std::vector<cv::Rect>>& boxes,
	std::list <std::list<std::string>>& lpns,
	float nmsThreshold, const int classId_last_country//classId_last_country : is the class index of the last country in the list of detected classes.
)
{
	std::list<std::list<int>> raw_classIds;
	std::list<std::list<float>> raw_confidences; std::list < std::list<cv::Rect>> raw_boxes;
	raw_detections_with_different_confidences(frame, raw_classIds,
		raw_confidences, raw_boxes,
		nmsThreshold);
	std::list<std::list<int>>::const_iterator it_classIds(raw_classIds.begin());
	std::list<std::list<float>>::const_iterator it_confidences(raw_confidences.begin());
	std::list < std::list<cv::Rect>>::const_iterator it_boxes(raw_boxes.begin());
	while (it_classIds != raw_classIds.end() && it_confidences != raw_confidences.end() && it_boxes != raw_boxes.end()) {
		std::list<std::string> current_lpns;
		std::list < std::vector<cv::Rect>>  l_vect_of_boxes_in_a_license_plate;
		std::list < std::vector<float>>  l_vect_of_confidences_in_a_license_plate; std::list <std::vector<int>>  l_vect_of_classIds_in_a_license_plate;
		std::list < std::vector<cv::Rect>>  l_vect_of_boxes_in_a_license_plate_tri_left;
		std::list < std::vector<float>>  l_vect_of_confidences_in_a_license_plate_tri_left; std::list <std::vector<int>>  l_vect_of_classIds_in_a_license_plate_tri_left;
		//the dnn has detected boxes that represent characters of the license plate, this function now groups characters in the same license plate and then rearranged from left to right.
		//it can deal with license pates that have two lines of charcaters
		separate_license_plates_if_necessary_add_blank_vehicles(*it_boxes, *it_confidences, *it_classIds,
			current_lpns,
			//l_vect_of_boxes_in_a_license_plate, l_vect_of_confidences_in_a_license_plate, l_vect_of_classIds_in_a_license_plate,
			boxes, confidences, classIds,
			l_vect_of_boxes_in_a_license_plate_tri_left, l_vect_of_confidences_in_a_license_plate_tri_left, l_vect_of_classIds_in_a_license_plate_tri_left,
			classId_last_country//classId_last_country : is the class index of the last country in the list of detected classes.
			, nmsThreshold);
		lpns.push_back(current_lpns);
		it_classIds++;
		it_confidences++;
		it_boxes++;
	}
}
//Given the @p input frame, create input blob, run net then, from result detections, assembly license plates present in the input image.
void Yolov5_anpr_onxx_detector::detect_with_different_confidences_then_separate_plates(const cv::Mat& frame, std::list<std::list<int>>& classIds,
	std::list<std::list<float>>& confidences, std::list < std::list<cv::Rect>>& boxes,
	std::list <std::list<std::string>>& lpns,
	float nmsThreshold, const int classId_last_country//classId_last_country : is the class index of the last country in the list of detected classes.
)
{
	std::list<std::list<int>> raw_classIds;
	std::list<std::list<float>> raw_confidences; std::list < std::list<cv::Rect>> raw_boxes;
	raw_detections_with_different_confidences(frame, raw_classIds,
		raw_confidences, raw_boxes,
		nmsThreshold);
	std::list<std::list<int>>::const_iterator it_classIds(raw_classIds.begin());
	std::list<std::list<float>>::const_iterator it_confidences(raw_confidences.begin());
	std::list < std::list<cv::Rect>>::const_iterator it_boxes(raw_boxes.begin());
	while (it_classIds != raw_classIds.end() && it_confidences != raw_confidences.end() && it_boxes != raw_boxes.end()) {
		std::list<std::list<int>> current_classIds; std::list<std::list<float>> current_confidences;
		std::list < std::list<cv::Rect>> current_boxes;
		std::list<std::string > current_lpns;
		std::list < std::vector<cv::Rect>>  l_vect_of_boxes_in_a_license_plate_tri_left;
		std::list < std::vector<float>>  l_vect_of_confidences_in_a_license_plate_tri_left; std::list <std::vector<int>>  l_vect_of_classIds_in_a_license_plate_tri_left;
		//the dnn has detected boxes that represent characters of the license plate, this function now groups characters in the same license plate and then rearranged from left to right.
		//it can deal with license pates that have two lines of charcaters
		separate_license_plates_if_necessary_add_blank_vehicles(*it_boxes, *it_confidences, *it_classIds,
			current_lpns,
			//l_vect_of_boxes_in_a_license_plate, l_vect_of_confidences_in_a_license_plate, l_vect_of_classIds_in_a_license_plate,
			current_boxes, current_confidences, current_classIds,
			l_vect_of_boxes_in_a_license_plate_tri_left, l_vect_of_confidences_in_a_license_plate_tri_left, l_vect_of_classIds_in_a_license_plate_tri_left,
			classId_last_country, nmsThreshold);
		classIds.splice(classIds.end(), current_classIds);
		boxes.splice(boxes.end(), current_boxes);
		confidences.splice(confidences.end(), current_confidences);
		lpns.push_back(current_lpns);
		it_classIds++;
		it_confidences++;
		it_boxes++;
	}
}
// Given the @p input frame, create input blob, run net and return result detections.
//this func can manage list of boxes of characters that dont have an englobing lp box (gloabal rect)
//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
void Yolov5_anpr_onxx_detector::detect_and_add_lp_and_vehicle_if_necessary(const cv::Mat& frame, std::list<std::list<int>>& classIds,
	std::list < std::list<float>>& confidences, std::list < std::list<cv::Rect>>& boxes,
	std::list<std::string>& lpns,
	const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.
	float confThreshold, float nmsThreshold)
{
	std::vector<int> one_conf_thresh_classIds;
	std::vector<float> one_conf_thresh_confidences;
	std::vector<cv::Rect> one_conf_thresh_boxes;
	detect(frame, one_conf_thresh_classIds,
		one_conf_thresh_confidences, one_conf_thresh_boxes,
		confThreshold, nmsThreshold);
	std::list < std::vector<cv::Rect>>  l_vect_of_boxes_in_a_license_plate_tri_left;
	std::list < std::vector<float>>  l_vect_of_confidences_in_a_license_plate_tri_left; std::list <std::vector<int>>  l_vect_of_classIds_in_a_license_plate_tri_left;
	//the dnn has detected boxes that represent characters of the license plate, this function now groups characters in the same license plate and then rearranged from left to right.
	//it can deal with license pates that have two lines of charcaters
	separate_license_plates_if_necessary_add_blank_vehicles(one_conf_thresh_boxes, one_conf_thresh_confidences, one_conf_thresh_classIds,
		lpns,
		//l_vect_of_boxes_in_a_license_plate, l_vect_of_confidences_in_a_license_plate, l_vect_of_classIds_in_a_license_plate,
		boxes, confidences, classIds,
		l_vect_of_boxes_in_a_license_plate_tri_left, l_vect_of_confidences_in_a_license_plate_tri_left, l_vect_of_classIds_in_a_license_plate_tri_left,
		classId_last_country, nmsThreshold);
	//check if we have a lpn
	bool no_lpn = true;
	std::list <std::string>::iterator it_lpns(lpns.begin());
	while (it_lpns != lpns.end())
	{
		if (it_lpns->size()) {
			no_lpn = false;
			break;
		}
		else it_lpns++;
	}
	if (no_lpn) {
		classIds.clear();
		confidences.clear(); boxes.clear();
		lpns.clear();
	}
	if (lpns.empty()) {
		std::vector<cv::Rect> tri_left_vect_of_detected_boxes; std::vector<float> tri_left_confidences; std::vector<int> tri_left_classIds;
		std::string lpn = get_single_lpn(
			one_conf_thresh_boxes, one_conf_thresh_confidences, one_conf_thresh_classIds,
			//characters inside lp
			tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds,
			nmsThreshold
		);
		if (!lpn.empty()) {
			std::list<cv::Rect> l_tri_left_vect_of_detected_boxes; std::list<float> l_tri_left_confidences; std::list<int> l_tri_left_classIds;
			std::copy(tri_left_vect_of_detected_boxes.begin(), tri_left_vect_of_detected_boxes.end(), std::back_inserter(l_tri_left_vect_of_detected_boxes));
			std::copy(tri_left_confidences.begin(), tri_left_confidences.end(), std::back_inserter(l_tri_left_confidences));
			std::copy(tri_left_classIds.begin(), tri_left_classIds.end(), std::back_inserter(l_tri_left_classIds));
			//here we must at the beginning of lists roi as global rect for lp and then a blank vehicle
			//this function adds if they dont already exist, a roi for the licene plate (equal to the global rect englobing the boxes) and a blank rect for the vehicle box
			add_lp_and_vehicle(one_conf_thresh_boxes, one_conf_thresh_confidences, one_conf_thresh_classIds,
				l_tri_left_vect_of_detected_boxes, l_tri_left_confidences, l_tri_left_classIds
				, classId_last_country);
			classIds.push_back(l_tri_left_classIds);
			confidences.push_back(l_tri_left_confidences); boxes.push_back(l_tri_left_vect_of_detected_boxes);
			lpns.push_back(lpn);
		}
	}
}
// Given the @p input frame, create input blob, run net and return result detections.
//this func can manage list of boxes of characters that dont have an englobing lp box (gloabal rect)
//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters.
//Produces double linked lists : inside list is for characters and outside list is for plates.
void Yolov5_anpr_onxx_detector::detect_and_add_lp_and_vehicle_if_necessary(const cv::Mat& frame,
	//setections when they are separated in double linked list (one list for one lp)
	std::list<std::vector<int>>& classIds,
	std::list < std::vector<float>>& confidences, std::list < std::vector<cv::Rect>>& boxes,
	std::list<std::string>& lpns,
	const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.
	float confThreshold, float nmsThreshold)
{
	//raw detection
	std::vector<int> one_conf_thresh_classIds;
	std::vector<float> one_conf_thresh_confidences;
	std::vector<cv::Rect> one_conf_thresh_boxes;
	detect(frame, //raw detection
		one_conf_thresh_classIds, one_conf_thresh_confidences, one_conf_thresh_boxes,
		confThreshold, nmsThreshold);
	std::list < std::vector<cv::Rect>>  l_vect_of_boxes_in_a_license_plate_tri_left;
	std::list < std::vector<float>>  l_vect_of_confidences_in_a_license_plate_tri_left; std::list <std::vector<int>>  l_vect_of_classIds_in_a_license_plate_tri_left;
	//the dnn has detected boxes that represent characters of the license plate, this function now groups characters in the same license plate and then rearranged from left to right.
	//it can deal with license pates that have two lines of charcaters
	separate_license_plates_if_necessary_add_blank_vehicles(one_conf_thresh_boxes, one_conf_thresh_confidences, one_conf_thresh_classIds,
		lpns,
		//l_vect_of_boxes_in_a_license_plate, l_vect_of_confidences_in_a_license_plate, l_vect_of_classIds_in_a_license_plate,
		boxes, confidences, classIds,
		//characters inside lp
		l_vect_of_boxes_in_a_license_plate_tri_left, l_vect_of_confidences_in_a_license_plate_tri_left, l_vect_of_classIds_in_a_license_plate_tri_left,
		classId_last_country, nmsThreshold);
	//check if we have a lpn
	bool no_lpn = true;
	std::list <std::string>::iterator it_lpns(lpns.begin());
	while (it_lpns != lpns.end())
	{
		if (it_lpns->size()) {
			no_lpn = false;
			break;
		}
		else it_lpns++;
	}
	if (no_lpn) {
		classIds.clear();
		confidences.clear(); boxes.clear();
		lpns.clear();
	}
	if (lpns.empty()) {
		std::vector<cv::Rect> tri_left_vect_of_detected_boxes; std::vector<float> tri_left_confidences; std::vector<int> tri_left_classIds;
		std::string lpn = get_single_lpn(
			one_conf_thresh_boxes, one_conf_thresh_confidences, one_conf_thresh_classIds,
			//characters inside lp
			tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds,
			nmsThreshold
		);
		if (!lpn.empty()) {
			//this function adds if they dont already exist, a roi for the licene plate (equal to the global rect englobing the boxes) and a blank rect for the vehicle box
			add_lp_and_vehicle(one_conf_thresh_boxes, one_conf_thresh_confidences, one_conf_thresh_classIds,
				tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds
				, classId_last_country);
			classIds.push_back(tri_left_classIds);
			confidences.push_back(tri_left_confidences); boxes.push_back(tri_left_vect_of_detected_boxes);
			lpns.push_back(lpn);
		}
	}
}
// Given the @p input frame, create input blob, run net and return result detections.
void Yolov5_anpr_onxx_detector::detect(const cv::Mat& frame, std::vector<int>& classIds,
	std::vector<float>& confidences, std::vector<cv::Rect>& boxes,
	float confThreshold, float nmsThreshold)
{
	//unlike the current engine that operates only on 35 characters (all the latin ones minus the O)
	bool preserve_aspect_ratio = true;
	std::vector<std::vector<Detection>>	detected = OnnxDetector::Run(frame, confThreshold, nmsThreshold, preserve_aspect_ratio);
	if (detected.size() == 1) {//batch 1 one image at a time
		std::vector<Detection>::const_iterator it(detected.front().begin());
		while (it != detected.front().end())
		{
			classIds.push_back(it->class_idx);
			confidences.push_back(it->score);
			boxes.push_back(it->bbox);
			it++;
		}
	}
}
// Given the @p input frame, create input blob, run net and return result detections.
void Yolov5_anpr_onxx_detector::detect(const cv::Mat& frame, std::list<int>& classIds,
	std::list<float>& confidences, std::list<cv::Rect>& boxes,
	float confThreshold, float nmsThreshold)
{
	//unlike the current engine that operates only on 35 characters (all the latin ones minus the O)
	bool preserve_aspect_ratio = true;
	std::vector<std::vector<Detection>>	detected = OnnxDetector::Run(frame, confThreshold, nmsThreshold, preserve_aspect_ratio);
	if (detected.size() == 1) {//batch 1 one image at a time
		std::vector<Detection>::const_iterator it(detected.front().begin());
		while (it != detected.front().end())
		{
			classIds.push_back(it->class_idx);
			confidences.push_back(it->score);
			boxes.push_back(it->bbox);
			it++;
		}
	}
}
//process an image file
void Yolov5_anpr_onxx_detector::detect(const std::string& image_filename, std::list <std::string>& lpns, const int classId_last_country//classId_last_country : is the class index of the last country in the list of detected classes.
)
{
	std::filesystem::path p_(image_filename);
	if (exists(p_) && std::filesystem::is_regular_file(p_))    // does p actually exist?
	{
		int flags = -1;//as is
		cv::Mat frame = cv::imread(image_filename, flags);
		if (frame.rows > 0 && frame.cols > 0) {
			if (frame.rows && frame.cols) {
				cv::Scalar mean_ = cv::mean(frame);
				if (mean_[0] > 2.0f && mean_[0] < 250.0f) {
					const float confThreshold = 0.7f;
					const float nmsThreshold = 0.5f;
					std::vector<int> classIds;
					std::vector<float> confidences;
					std::vector<cv::Rect> boxes;
					// Given the @p input frame, create input blob, run net and return result detections.
					detect(frame, classIds,
						confidences, boxes,
						confThreshold, nmsThreshold
					);
					std::list < std::vector<cv::Rect>>  l_vect_of_boxes_in_a_license_plate;
					std::list < std::vector<float>>  l_vect_of_confidences_in_a_license_plate; std::list <std::vector<int>>  l_vect_of_classIds_in_a_license_plate;
					std::list < std::vector<cv::Rect>>  l_vect_of_boxes_in_a_license_plate_tri_left;
					std::list < std::vector<float>>  l_vect_of_confidences_in_a_license_plate_tri_left; std::list <std::vector<int>>  l_vect_of_classIds_in_a_license_plate_tri_left;
					//the dnn has detected boxes that represent characters of the license plate, this function now groups characters in the same license plate and then rearranged from left to right.
					//it can deal with license pates that have two lines of charcaters
					separate_license_plates_if_necessary_add_blank_vehicles(boxes, confidences, classIds,
						lpns,
						l_vect_of_boxes_in_a_license_plate, l_vect_of_confidences_in_a_license_plate, l_vect_of_classIds_in_a_license_plate,
						l_vect_of_boxes_in_a_license_plate_tri_left, l_vect_of_confidences_in_a_license_plate_tri_left, l_vect_of_classIds_in_a_license_plate_tri_left,
						classId_last_country, nmsThreshold);
				}
			}
		}
	}
}
Yolov5_anpr_onxx_detector* get_detector_with_smallest_size_bigger_than_image(const std::list<Yolov5_anpr_onxx_detector*>& detectors, const int max_size)
{
	Yolov5_anpr_onxx_detector* detector_with_neartest_size = nullptr;
	Yolov5_anpr_onxx_detector* detector_with_greatest_size = nullptr;
	std::list<Yolov5_anpr_onxx_detector*>::const_iterator it(detectors.begin());
	int64_t detector_greatest_size = -1;
	int64_t detector_neartest_size = -1;
	while (it != detectors.end())
	{
		int64_t current_detector_size = (*it)->max_image_size();
		if (current_detector_size > detector_greatest_size) {
			detector_greatest_size = current_detector_size;
			detector_with_greatest_size = (*it);
		}
		if (current_detector_size > max_size) {
			if (detector_neartest_size > 0) {
				if (current_detector_size - max_size < detector_neartest_size - max_size) {
					detector_neartest_size = current_detector_size;
					detector_with_neartest_size = (*it);
				}
			}
			else {
				detector_neartest_size = current_detector_size;
				detector_with_neartest_size = (*it);
			}
		}
		it++;
	}
	if (detector_neartest_size > 0) return detector_with_neartest_size;
	else return detector_with_greatest_size;
}
Yolov5_anpr_onxx_detector* get_detector_with_smallest_size_bigger_than_image(const std::list<Yolov5_anpr_onxx_detector*>& detectors, const int width, const int height)
{
	if (width > height)
		return get_detector_with_smallest_size_bigger_than_image(detectors, width);
	else
		return get_detector_with_smallest_size_bigger_than_image(detectors, height);
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
float Yolov5_anpr_onxx_detector::two_stage_lpr(const std::string& dir)
{
	return two_stage_lpr(*this, dir);
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
float Yolov5_anpr_onxx_detector::two_stage_lpr(Yolov5_anpr_onxx_detector& parking_detector, const std::string& dir)
{
	std::string filename = "D:\\Programmation\\LPReditor\\ocr_dataset\\test_svm.txt";
	std::ofstream O(filename.c_str(), std::ios::app);
	O << "Yolov5_anpr_onxx_detector::two_stage_lpr " << std::endl;
	//O.flush(); O.close();
	std::list<std::string> image_filenames;
	//extracts, from a test directory, all images files that come with an xml file containing the bb coordinates in this image
	load_images_filenames(dir, image_filenames);
	std::list<std::string>::const_iterator it_image_filenames(image_filenames.begin());
	int good_nb_caracs = 0;
	int too_much_caracs = 0;
	int misses_carcs = 0;
	int less_1_editdistance_reads = 0;
	int c = 0;
	int good_reads = 0;
	while (it_image_filenames != image_filenames.end())
	{
		std::list <std::string> lpns; std::list <int> lp_country_class; std::list < cv::Rect> lp_rois;
		std::list < std::list<float>>  confidences; std::list < std::list<int>>  classes; std::list < std::list<cv::Rect>>  boxes;
		//detection inside the chosen lp
		std::list<int> chosen_lp_classIds; std::list<float> chosen_lp_confidences; std::list<cv::Rect> chosen_lp_boxes;
		two_stage_lpr(parking_detector, *it_image_filenames,
			//double linked lists to separate lps
			confidences, classes, boxes,
			//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
			lpns, lp_country_class, lp_rois,
			//detection inside the chosen lp
			chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
		);
		std::filesystem::path p_(*it_image_filenames);
		bool vrai_lpn_after_underscore = true;
		//C_OCROutputs availableAlpha(LATIN_LETTERS_LATIN_DIGITS);
		std::string ExactLPN(getTrueLPN(p_.stem().string(), vrai_lpn_after_underscore));
		std::string  lpn;
		int best_country_class;
		cv::Rect  best_lpn_roi;
		std::list<float>  best_confidences;
		std::list<int>  best_classes;
		std::list<cv::Rect>  best_boxes;
		//we know the true license plate number that come from a training image and we want to find the detections boxes to aautomatically annotate the image.
//We also have run the nn that produces detections, the goal of this func is to find the detections that are closest to the true lpn
		std::list < std::list<float>>  just_one_lp_confidences; std::list < std::list<int>>  just_one_lp_classes; std::list < std::list<cv::Rect>>  just_one_lp_boxes;
		just_one_lp_confidences.push_back(chosen_lp_confidences); just_one_lp_classes.push_back(chosen_lp_classIds); just_one_lp_boxes.push_back(chosen_lp_boxes);
		int editdistance = find_nearest_plate_substitutions_allowed(ExactLPN,
			lpns, lp_country_class, lp_rois,
			//confidences, classes, boxes,
			just_one_lp_confidences, just_one_lp_classes, just_one_lp_boxes,
			lpn,
			best_country_class,
			best_lpn_roi,
			best_confidences,
			best_classes,
			best_boxes);
		std::cout << "ExactLPN : " << ExactLPN << " read LPN : " << lpn << std::endl;
#ifdef _DEBUG
		int index = 0;
		while (index < ExactLPN.size()) {
#ifdef _DEBUG		
			//assert(ExactLPN[index] != '0');
#endif //_DEBUG
			index++;
		}
#endif //_DEBUG
		//if (editdistance > 0) miss_reads++;
		if (editdistance > 0) {}
		else good_reads++;
		if (ExactLPN.size() > lpn.size()) misses_carcs++;
		else if (ExactLPN.size() == lpn.size() - 1) too_much_caracs++;
		else if (ExactLPN.size() == lpn.size()) good_nb_caracs++;
		if (editdistance <= 1) less_1_editdistance_reads++;
		it_image_filenames++; c++;
		if ((c % 1000) == 0) {
			O << c << " perc reads less 1 edit distance:" << (float)(less_1_editdistance_reads) / (float)(c) << std::endl;
			O << c << " misses carac(s):" << (float)(misses_carcs) / (float)(c)
				<< "  too_much_caracs:" << (float)(too_much_caracs) / (float)(c) << " good_nb_caracs " << (float)(good_nb_caracs) / (float)(c) << " perc good reads:" << (float)(good_reads) / (float)(c) << std::endl;
			std::cout << c << " misses carac(s):" << (float)(misses_carcs) / (float)(c)
				<< "  too_much_caracs:" << (float)(too_much_caracs) / (float)(c) << " good_nb_caracs " << (float)(good_nb_caracs) / (float)(c) << std::endl;
			std::cout << c << " misses carac(s):" << (float)(misses_carcs) / (float)(c)
				<< "  too_much_caracs:" << (float)(too_much_caracs) / (float)(c) << " good_nb_caracs " << (float)(good_nb_caracs) / (float)(c) << std::endl;
			std::cout << c << " perc good reads:" << (float)(good_reads) / (float)(c) << std::endl;
			std::cout << c << " perc reads less 1 edit distance:" << (float)(less_1_editdistance_reads) / (float)(c) << std::endl;
		}
	}
	float good_nb_caracs_percentage = (float)(good_nb_caracs) / (float)(image_filenames.size());
	float good_reads_percentage = (float)(good_reads) / (float)(c);
	float less_1_editdistance_reads_percentage = (float)(less_1_editdistance_reads) / (float)(c);
	O << "good_nb_caracs : " << good_nb_caracs_percentage << "good_reads : " << good_reads_percentage << "less_1_editdistance : " << less_1_editdistance_reads_percentage << std::endl;
	O.flush(); O.close();
	return good_reads_percentage;
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void Yolov5_anpr_onxx_detector::two_stage_lpr(const std::string& image_filename,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
)
{
	return two_stage_lpr(*this, image_filename,
		//double linked lists to separate lps
		confidences, classes, boxes,
		//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
		lpns, lp_country_class, lp_rois,
		//detection inside the chosen lp
		chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
	);
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void Yolov5_anpr_onxx_detector::two_stage_lpr(Yolov5_anpr_onxx_detector& parking_detector, const std::string& image_filename,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
)
{
	if (parking_detector.max_image_size() > max_image_size())
		return parking_detector.two_stage_lpr(*this, image_filename,
			//double linked lists to separate lps
			confidences, classes, boxes,
			//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
			lpns, lp_country_class, lp_rois,
			//detection inside the chosen lp
			chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
		);
	else {
		std::filesystem::path p_(image_filename);
		if (exists(p_) && std::filesystem::is_regular_file(p_))    // does p actually exist?
		{
			int flags = -1;//as is
			cv::Mat frame = cv::imread(image_filename, flags);
			return two_stage_lpr(parking_detector, frame,
				//double linked lists to separate lps
				confidences, classes, boxes,
				//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
				lpns, lp_country_class, lp_rois,
				//detection inside the chosen lp
				chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
			);
		}
	}
}
void get_larger_roi(cv::Rect& lpn_roi, const int width, const int height
)
{
	int w = lpn_roi.width;
	int h = lpn_roi.height;
	lpn_roi.x -= w;
	lpn_roi.width += int(2.0f * (float)w);
	lpn_roi.y -= int(1.5f * (float)h);
	lpn_roi.height += int(3.0f * (float)h);
	lpn_roi = get_inter(lpn_roi, cv::Rect(0, 0, width, height));
}
void get_larger_roi(cv::Rect& lpn_roi, const int width, const int height, const float& scale_x, const float& scale_y
)
{
	int w = lpn_roi.width;
	int h = lpn_roi.height;
	/*
	lpn_roi.x -= w;
	lpn_roi.width += int(2.0f * (float)w);
	lpn_roi.y -= int(1.5f * (float)h);
	lpn_roi.height += int(3.0f * (float)h);
	*/
	lpn_roi.width = int((float)lpn_roi.width * scale_x);
	lpn_roi.x -= (lpn_roi.width - w) / 2;
	lpn_roi.height = int((float)lpn_roi.height * scale_y);
	lpn_roi.y -= (lpn_roi.height - h) / 2;
	lpn_roi = get_inter(lpn_roi, cv::Rect(0, 0, width, height));
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void Yolov5_anpr_onxx_detector::two_stage_lpr(Yolov5_anpr_onxx_detector& parking_detector, const cv::Mat& frame,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
)
{
	if (parking_detector.max_image_size() > max_image_size())
		return parking_detector.two_stage_lpr(*this, frame,
			//double linked lists to separate lps
			confidences, classes, boxes,
			//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
			lpns, lp_country_class, lp_rois,
			//detection inside the chosen lp
			chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
			//, availableAlpha//, must_convert_from_NO_I_O_2_NO_O
		);
	else {
		std::list<Yolov5_anpr_onxx_detector*> freeflow_detectors;
		freeflow_detectors.push_back(&(*this));
		std::list<Yolov5_anpr_onxx_detector*> parking_detectors;
		parking_detectors.push_back(&(parking_detector));
		return ::two_stage_lpr(freeflow_detectors, parking_detectors,
			frame,
			//double linked lists to separate lps
			confidences, classes, boxes,
			//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
			lpns, lp_country_class, lp_rois,
			//detection inside the chosen lp
			chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
		);
	}
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void Yolov5_anpr_onxx_detector::two_stage_lpr(Yolov5_anpr_onxx_detector& parking_detector, Plates_types_classifier& plates_types_classifier, const cv::Mat& frame,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
)
{
	if (parking_detector.max_image_size() > max_image_size())
		return parking_detector.two_stage_lpr(*this, plates_types_classifier, frame,
			//double linked lists to separate lps
			confidences, classes, boxes,
			//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
			lpns, lp_country_class, lp_rois,
			//detection inside the chosen lp
			chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
			//, availableAlpha//, must_convert_from_NO_I_O_2_NO_O
		);
	else {
		std::list<Yolov5_anpr_onxx_detector*> freeflow_detectors;
		freeflow_detectors.push_back(&(*this));
		std::list<Yolov5_anpr_onxx_detector*> parking_detectors;
		parking_detectors.push_back(&(parking_detector));
		return ::two_stage_lpr(freeflow_detectors, parking_detectors, plates_types_classifier,
			frame,
			//double linked lists to separate lps
			confidences, classes, boxes,
			//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
			lpns, lp_country_class, lp_rois,
			//detection inside the chosen lp
			chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
		);
	}
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void two_stage_lpr(const std::list<Yolov5_anpr_onxx_detector*>& detectors, const std::string& image_filename,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
)
{
	std::filesystem::path p_(image_filename);
	if (exists(p_) && std::filesystem::is_regular_file(p_))    // does p actually exist?
	{
		int flags = -1;//as is
		cv::Mat frame = cv::imread(image_filename, flags);
		two_stage_lpr(detectors, detectors,
			frame,
			//double linked lists to separate lps
			confidences, classes, boxes,
			//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
			lpns, lp_country_class, lp_rois,
			//detection inside the chosen lp
			chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
		);
	}
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void two_stage_lpr(const std::list<Yolov5_anpr_onxx_detector*>& freeflow_detectors, const std::list<Yolov5_anpr_onxx_detector*>& parking_detectors,
	const std::string& image_filename,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
)
{
	std::filesystem::path p_(image_filename);
	if (exists(p_) && std::filesystem::is_regular_file(p_))    // does p actually exist?
	{
		int flags = -1;//as is
		cv::Mat frame = cv::imread(image_filename, flags);
		two_stage_lpr(freeflow_detectors, parking_detectors,
			frame,
			//double linked lists to separate lps
			confidences, classes, boxes,
			//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
			lpns, lp_country_class, lp_rois,
			//detection inside the chosen lp
			chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
		);
	}
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void two_stage_lpr(const std::list<Yolov5_anpr_onxx_detector*>& freeflow_detectors, const std::list<Yolov5_anpr_onxx_detector*>& parking_detectors,
	const cv::Mat& frame,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
)
{
	if (frame.rows > 0 && frame.cols > 0) {
		//global image
		cv::Scalar mean_ = cv::mean(frame);
		if (mean_[0] > 2.0f && mean_[0] < 250.0f) {
			std::vector<int> classIds_lpn;
			std::vector<float> confidences_lpn;
			std::vector<cv::Rect> vect_of_detected_boxes_lpn;
			//Given the @p input frame, create input blob, run net and return result detections.
			//float confidence_lpn_roi;
			//
			Yolov5_anpr_onxx_detector* pnet_lpn_detector = get_detector_with_smallest_size_bigger_than_image(freeflow_detectors, frame.cols, frame.rows);
			if (pnet_lpn_detector != nullptr) {
				std::list<float> confidence_one_lp;
				std::list < cv::Rect> one_lp;
				std::list<int> classIds_one_lp;
				const int classId_last_country = NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE + NUMBER_OF_COUNTRIES - 1;
				//it selects just one lpn although all lps have been detected and stored in double linked lists, then from these lists, selects the one that is the best 
//(with best confidences of its characters and with greateast size)
				//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
				float nmsThreshold = 0.3f;
				pnet_lpn_detector->detect_with_different_confidences_then_separate_plates(frame, classes,
					confidences, boxes, confidence_one_lp, one_lp, classIds_one_lp,
					classId_last_country,
					nmsThreshold);
				if (one_lp.size()) {
					//one_lp is the list of the boundinx boxes of a lp. one_lp is the best lps in the image (see func detect_with_different_confidences_then_separate_plates). And we are sure that the front element of the boxes is the lp.
					cv::Rect lpn_roi = one_lp.front();
#ifdef _DEBUG
					//given the index of a bounding box, we can predict if this box is a single character or if it represents the license plate area or if it is the roi of an entire vehicle
					//single character--> returns 1
					//license plate--> returns 2
					//vehicle--> returns 3
					//negative index--> returns 0 must be an error
					assert(is_this_box_a_character_a_license_plate_or_a_vehicle(classIds_one_lp.front(), classId_last_country) == 2);
#endif //_DEBUG
					float confidence_lpn_roi = confidence_one_lp.front();
					if (lpn_roi.width > 0 && lpn_roi.height > 0 && lpn_roi.x >= 0 && lpn_roi.y >= 0 &&
						confidence_lpn_roi > 0.0f && lpn_roi.x + lpn_roi.width <= frame.cols && lpn_roi.y <= frame.rows) {
						get_larger_roi(lpn_roi, frame.cols, frame.rows);
#ifdef _DEBUG
						if (lpn_roi.area() < ((frame.rows * frame.cols) / 10)) {
#else // _DEBUG		
						if (lpn_roi.area() < ((frame.rows * frame.cols) / 10)) {
#endif //_DEBUG
							//image of lpn
							cv::Mat subimage_ = frame(lpn_roi);
							std::vector<int> vet_of_classIds;
							std::vector<float> vect_of_confidences;
							std::vector<cv::Rect> vect_of_detected_boxes;
							//Given the @p input frame, create input blob, run net and return result detections.
							Yolov5_anpr_onxx_detector* pnet_inside_lpn_detector = get_detector_with_smallest_size_bigger_than_image(parking_detectors, subimage_.cols, subimage_.rows);
							std::list < std::list<float>> confidences_inside_lp; std::list < std::list<int>> classes_inside_lp; std::list < std::list<cv::Rect>> boxes_inside_lp;
							std::list<float> inside_lp_confidence_one_lp;
							std::list < cv::Rect> inside_lp_one_lp;
							std::list<int> inside_lp_classIds_one_lp;
							if (pnet_inside_lpn_detector != nullptr) {
								pnet_inside_lpn_detector->detect_with_different_confidences_then_separate_plates(subimage_, classes_inside_lp,
									confidences_inside_lp, boxes_inside_lp, inside_lp_confidence_one_lp, inside_lp_one_lp, inside_lp_classIds_one_lp,
									classId_last_country, nmsThreshold);
							}
							else pnet_lpn_detector->detect_with_different_confidences_then_separate_plates(subimage_, classes_inside_lp,
								confidences_inside_lp, boxes_inside_lp, inside_lp_confidence_one_lp, inside_lp_one_lp, inside_lp_classIds_one_lp,
								classId_last_country, nmsThreshold);
							//the nnet has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number. 
							//it can deal with license pates that have two lines of charcaters
							std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
							std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
							const float nmsThreshold_lpn = 0.5;
							std::string lpn = get_best_lpn(inside_lp_one_lp, inside_lp_confidence_one_lp, inside_lp_classIds_one_lp,
								tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold_lpn);
							if (inside_lp_one_lp.size() > 2) {
								std::vector<float>::iterator it_confidences(tri_left_confidences.begin());
								std::vector<cv::Rect>::iterator it_boxes(tri_left_vect_of_detected_boxes.begin());
								std::vector<int>::iterator it_out_classes_(tri_left_classIds.begin());
								while (it_out_classes_ != tri_left_classIds.end()
									&& it_confidences != tri_left_confidences.end() && it_boxes != tri_left_vect_of_detected_boxes.end()) {
									if (
										(*it_out_classes_ < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE) ||
										//(must_convert_from_NO_I_O_2_NO_O && *it_out_classes_ < availableAlpha_NO_O.nb_carac()) ||
										(lpn.empty())
										) {
										//now we must change the coordinates of the box to fit global image
										cv::Rect box_in_global_image(it_boxes->x, it_boxes->y, it_boxes->width, it_boxes->height);
										chosen_lp_boxes.push_back(box_in_global_image);
										chosen_lp_classIds.push_back(*it_out_classes_);
										chosen_lp_confidences.push_back(*it_confidences);
									}
									it_out_classes_++;
									it_confidences++;
									it_boxes++;
								}
#ifdef _DEBUG
								assert(chosen_lp_confidences.size() == chosen_lp_classIds.size() && chosen_lp_boxes.size() == chosen_lp_classIds.size());
								assert(lpn.length() == chosen_lp_classIds.size() || lpn.length() == 0);
#endif //_DEBUG
								lpns.push_back(lpn);
								lp_country_class.push_back(classIds_one_lp.front());
								lp_rois.push_back(one_lp.front());
							}
							else {
								//the nnet has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number. 
								//it can deal with license pates that have two lines of charcaters
								std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
								std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
								const float nmsThreshold_lpn = 0.5;
								//rearange boxes from left to right
								std::string lpn = get_best_lpn(one_lp, confidence_one_lp, classIds_one_lp, tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold_lpn);
								std::vector<float>::iterator it_confidences(tri_left_confidences.begin());
								std::vector<cv::Rect>::iterator it_boxes(tri_left_vect_of_detected_boxes.begin());
								std::vector<int>::iterator it_out_classes_(tri_left_classIds.begin());
								while (it_out_classes_ != tri_left_classIds.end()
									&& it_confidences != tri_left_confidences.end() && it_boxes != tri_left_vect_of_detected_boxes.end()) {
									if (
										(*it_out_classes_ < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE) ||
										//(must_convert_from_NO_I_O_2_NO_O && *it_out_classes_ < availableAlpha_NO_O.nb_carac()) ||
										(lpn.empty())
										) {
										//now we must change the coordinates of the box to fit global image
										cv::Rect box_in_global_image(it_boxes->x, it_boxes->y, it_boxes->width, it_boxes->height);
										chosen_lp_boxes.push_back(box_in_global_image);
										chosen_lp_classIds.push_back(*it_out_classes_);
										chosen_lp_confidences.push_back(*it_confidences);
									}
									it_out_classes_++;
									it_confidences++;
									it_boxes++;
								}
#ifdef _DEBUG
								assert(chosen_lp_confidences.size() == chosen_lp_classIds.size() && chosen_lp_boxes.size() == chosen_lp_classIds.size());
								assert(lpn.length() == chosen_lp_classIds.size() || lpn.length() == 0);
#endif //_DEBUG
								lpns.push_back(lpn);
								lp_country_class.push_back(classIds_one_lp.front());
								lp_rois.push_back(one_lp.front());
							}
						}
						else {
							//the nnet has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number. 
							//it can deal with license pates that have two lines of charcaters
							std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
							std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
							const float nmsThreshold_lpn = 0.5;
							//rearange boxes from left to right
							std::string lpn = get_best_lpn(one_lp, confidence_one_lp, classIds_one_lp, tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold_lpn);
							std::vector<float>::iterator it_confidences(tri_left_confidences.begin());
							std::vector<cv::Rect>::iterator it_boxes(tri_left_vect_of_detected_boxes.begin());
							std::vector<int>::iterator it_out_classes_(tri_left_classIds.begin());
							while (it_out_classes_ != tri_left_classIds.end()
								&& it_confidences != tri_left_confidences.end() && it_boxes != tri_left_vect_of_detected_boxes.end()) {
								if (
									(*it_out_classes_ < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE) ||
									//(must_convert_from_NO_I_O_2_NO_O && *it_out_classes_ < availableAlpha_NO_O.nb_carac()) ||
									(lpn.empty())
									) {
									//now we must change the coordinates of the box to fit global image
									cv::Rect box_in_global_image(it_boxes->x, it_boxes->y, it_boxes->width, it_boxes->height);
									chosen_lp_boxes.push_back(box_in_global_image);
									chosen_lp_classIds.push_back(*it_out_classes_);
									chosen_lp_confidences.push_back(*it_confidences);
								}
								it_out_classes_++;
								it_confidences++;
								it_boxes++;
							}
#ifdef _DEBUG
							assert(chosen_lp_confidences.size() == chosen_lp_classIds.size() && chosen_lp_boxes.size() == chosen_lp_classIds.size());
							assert(lpn.length() == chosen_lp_classIds.size() || lpn.length() == 0);
#endif //_DEBUG
							lpns.push_back(lpn);
							lp_country_class.push_back(classIds_one_lp.front());
							lp_rois.push_back(one_lp.front());
						}
					}
					else {
						//the nnet has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number. 
						//it can deal with license pates that have two lines of charcaters
						std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
						std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
						const float nmsThreshold_lpn = 0.5;
						//rearange boxes from left to right
						std::string lpn = get_best_lpn(one_lp, confidence_one_lp, classIds_one_lp, tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold_lpn);
						std::vector<float>::iterator it_confidences(tri_left_confidences.begin());
						std::vector<cv::Rect>::iterator it_boxes(tri_left_vect_of_detected_boxes.begin());
						std::vector<int>::iterator it_out_classes_(tri_left_classIds.begin());
						while (it_out_classes_ != tri_left_classIds.end()
							&& it_confidences != tri_left_confidences.end() && it_boxes != tri_left_vect_of_detected_boxes.end()) {
							if (
								(*it_out_classes_ < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE) ||
								//(must_convert_from_NO_I_O_2_NO_O && *it_out_classes_ < availableAlpha_NO_O.nb_carac()) ||
								(lpn.empty())
								) {
								//now we must change the coordinates of the box to fit global image
								cv::Rect box_in_global_image(it_boxes->x, it_boxes->y, it_boxes->width, it_boxes->height);
								chosen_lp_boxes.push_back(box_in_global_image);
								chosen_lp_classIds.push_back(*it_out_classes_);
								chosen_lp_confidences.push_back(*it_confidences);
							}
							it_out_classes_++;
							it_confidences++;
							it_boxes++;
						}
#ifdef _DEBUG
						assert(chosen_lp_confidences.size() == chosen_lp_classIds.size() && chosen_lp_boxes.size() == chosen_lp_classIds.size());
						assert(lpn.length() == chosen_lp_classIds.size() || lpn.length() == 0);
#endif //_DEBUG
						lpns.push_back(lpn);
						lp_country_class.push_back(classIds_one_lp.front());
						lp_rois.push_back(one_lp.front());
					}
				}
			}
		}
	}
}
//
//Given the @p input frame, create input blob, run net and return result detections.
//it selects just one lpn although all lps are detected in double linked lists
////this func can manage list of boxes of characters that dont have an englobing lp box (gloabal rect)
////output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
////and remaining elements are characters
//			  
void Yolov5_anpr_onxx_detector::detect_lpn_and_add_lp_and_vehicle_if_necessary(const cv::Mat & frame, std::list < std::vector<int>> & classIds,
	std::list < std::vector<float>> & confidences, std::list < std::vector<cv::Rect>> & boxes
	, std::list<float> & confidence_one_lp, std::list < cv::Rect> & one_lp, std::list<int> & classIds_one_lp,
	const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.
	//const C_OCROutputs & availableAlpha,
	const float confThreshold, float nmsThreshold)
{
	std::list<std::string> lpns;
	//this func can manage list of boxes of characters that dont have an englobing lp box (gloabal rect)
	//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
	detect_and_add_lp_and_vehicle_if_necessary(frame, classIds,
		confidences, boxes, lpns,
		classId_last_country,
		confThreshold, nmsThreshold);
	//now choose best lpn
	//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
	get_best_plate(classIds,
		confidences, boxes, confidence_one_lp, one_lp, classIds_one_lp);
}
//Given the @p input frame, create input blob, run net then, from result detections, assembly license plates present in the input image.
void Yolov5_anpr_onxx_detector::detect_with_different_confidences_then_separate_plates(const cv::Mat & frame, std::list < std::list<int>> & classIds,
	std::list < std::list<float>> & confidences, std::list < std::list<cv::Rect>> & boxes
	, std::list<float> & confidence_one_lp, std::list < cv::Rect> & one_lp, std::list<int> & classIds_one_lp,
	const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.
	//const C_OCROutputs & availableAlpha,
	float nmsThreshold)
{
	std::list <std::list<std::string>> lpns;
	detect_with_different_confidences_then_separate_plates(frame, classIds,
		confidences, boxes, lpns
		, nmsThreshold, classId_last_country);
	//now choose best lpn
	//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
	get_best_plate(classIds,
		confidences, boxes, confidence_one_lp, one_lp, classIds_one_lp);
}
//Given the @p input frame, create input blob, run net then, from result detections, assembly license plates present in the input image.
void Yolov5_anpr_onxx_detector::detect_with_different_confidences_then_separate_plates(const cv::Mat & frame, const std::string & ExactLPN, std::list < std::list<int>> & classIds,
	std::list < std::list<float>> & confidences, std::list < std::list<cv::Rect>> & boxes
	, std::list<float> & confidence_one_lp, std::list < cv::Rect> & one_lp, std::list<int> & classIds_one_lp,
	const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.
	//const C_OCROutputs & availableAlpha,
	float nmsThreshold)
{
	std::list <std::list<std::string>> lpns;
	detect_with_different_confidences_then_separate_plates(frame, classIds,
		confidences, boxes, lpns
		, nmsThreshold, classId_last_country);
	//now choose best lpn
	//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
	get_best_plate(ExactLPN, classIds,
		confidences, boxes, confidence_one_lp, one_lp, classIds_one_lp);
}
//Given the @p input frame, create input blob, run net then, from result detections, assembly license plates present in the input image.
void Yolov5_anpr_onxx_detector::detect_with_different_confidences_then_separate_plates(const cv::Mat & frame, const std::string & ExactLPN, std::list < std::list<int>> & classIds,
	std::list < std::list<float>> & confidences, std::list < std::list<cv::Rect>> & boxes
	, std::vector<float> & confidence_one_lp, std::vector < cv::Rect> & one_lp, std::vector<int> & classIds_one_lp,
	const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.
	//const C_OCROutputs & availableAlpha,
	float nmsThreshold)
{
	std::list <std::list<std::string>> lpns;
	detect_with_different_confidences_then_separate_plates(frame, classIds,
		confidences, boxes, lpns
		, nmsThreshold, classId_last_country);
	//now choose best lpn
	//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
	get_best_plate(ExactLPN, classIds,
		confidences, boxes, confidence_one_lp, one_lp, classIds_one_lp);
}
// Given the @p input frame, create input blob, run net and return result detections.
//this func can manage list of boxes of characters that dont have an englobing lp box (gloabal rect)
//output lists looks like : first box = license plate (either a detected box either the global rect englobing other boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
void Yolov5_anpr_onxx_detector::two_stage_lpr(Yolov5_anpr_onxx_detector & parking_detector, const cv::Mat & frame, std::string & lpn)
{
	std::list<std::list<int>> classIds;
	std::list < std::list<float>> confidences;
	std::list < std::list<cv::Rect>> boxes;
	std::list<std::string> lpns;
	std::list <int> lp_country_class;
	std::list < cv::Rect> lp_rois;
	std::string best_lpn;
	std::list<float>  best_confidences; std::list<int> best_classes; std::list<cv::Rect> best_boxes;
	two_stage_lpr(parking_detector, frame,
		//double linked lists to separate lps
		confidences, classIds, boxes,
		//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
		lpns,
		lp_country_class, lp_rois,
		//detection inside the chosen lp
		best_classes, best_confidences, best_boxes
	);
	std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
	std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
	//the nnet has detected boxes that represant characters of the license plate, this function now etracts from these boxes the license plate number. 
	//it can deal with license pates that have two lines of charcaters
	lpn = get_best_lpn(best_boxes, best_confidences, best_classes,
		tri_left_vect_of_detected_boxes,
		tri_left_confidences, tri_left_classIds
	);
}
// Given the @p input frame, create input blob, run net and return result detections.
//this func can manage list of boxes of characters that dont have an englobing lp box (gloabal rect)
//output lists looks like : first box = license plate (either a detected box either the global rect englobing other boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
void Yolov5_anpr_onxx_detector::two_stage_lpr(Yolov5_anpr_onxx_detector & parking_detector, Plates_types_classifier& plates_types_classifier, const cv::Mat & frame, std::string & lpn)
{
	std::list<std::list<int>> classIds;
	std::list < std::list<float>> confidences;
	std::list < std::list<cv::Rect>> boxes;
	std::list<std::string> lpns;
	std::list <int> lp_country_class;
	std::list < cv::Rect> lp_rois;
	std::string best_lpn;
	std::list<float>  best_confidences; std::list<int> best_classes; std::list<cv::Rect> best_boxes;
	two_stage_lpr(parking_detector, plates_types_classifier, frame,
		//double linked lists to separate lps
		confidences, classIds, boxes,
		//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
		lpns,
		lp_country_class, lp_rois,
		//detection inside the chosen lp
		best_classes, best_confidences, best_boxes
	);
	std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
	std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
	//the nnet has detected boxes that represant characters of the license plate, this function now etracts from these boxes the license plate number. 
	//it can deal with license pates that have two lines of charcaters
	lpn = get_best_lpn(best_boxes, best_confidences, best_classes,
		tri_left_vect_of_detected_boxes,
		tri_left_confidences, tri_left_classIds
	);
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void two_stage_lpr(const std::list<Yolov5_anpr_onxx_detector*>& freeflow_detectors, const std::list<Yolov5_anpr_onxx_detector*>& parking_detectors
	, Plates_types_classifier& plates_types_classifier,
	const std::string& image_filename,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
)
{
	std::filesystem::path p_(image_filename);
	if (exists(p_) && std::filesystem::is_regular_file(p_))    // does p actually exist?
	{
		int flags = -1;//as is
		cv::Mat frame = cv::imread(image_filename, flags);
		std::cout << std::endl;
		std::cout << p_.filename().string() << std::endl;
		two_stage_lpr(freeflow_detectors, parking_detectors, plates_types_classifier,
			frame,
			//double linked lists to separate lps
			confidences, classes, boxes,
			//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
			lpns, lp_country_class, lp_rois,
			//detection inside the chosen lp
			chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
		);
	}
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void two_stage_lpr(Yolov5_anpr_onxx_detector & freeflow_detector, Yolov5_anpr_onxx_detector & parking_detector
	, Plates_types_classifier& plates_types_classifier,
	const std::string& image_filename,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
)
{
	std::filesystem::path p_(image_filename);
	if (exists(p_) && std::filesystem::is_regular_file(p_))    // does p actually exist?
	{
		int flags = -1;//as is
		cv::Mat frame = cv::imread(image_filename, flags);
		std::cout << std::endl;
		std::cout << p_.filename().string() << std::endl;
		two_stage_lpr(freeflow_detector, parking_detector, plates_types_classifier,
			frame,
			//double linked lists to separate lps
			confidences, classes, boxes,
			//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
			lpns, lp_country_class, lp_rois,
			//detection inside the chosen lp
			chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
		);
	}
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void two_stage_lpr(const std::list<Yolov5_anpr_onxx_detector*>& freeflow_detectors, const std::list<Yolov5_anpr_onxx_detector*>& parking_detectors
	, Plates_types_classifier& plates_types_classifier,
	const cv::Mat& frame,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
)
{
	if (frame.rows > 0 && frame.cols > 0) {
		//global image
		cv::Scalar mean_ = cv::mean(frame);
		if (mean_[0] > 2.0f && mean_[0] < 250.0f) {
			std::vector<int> classIds_lpn;
			std::vector<float> confidences_lpn;
			std::vector<cv::Rect> vect_of_detected_boxes_lpn;
			//Given the @p input frame, create input blob, run net and return result detections.
			//
			Yolov5_anpr_onxx_detector* pnet_lpn_detector = get_detector_with_smallest_size_bigger_than_image(freeflow_detectors, frame.cols, frame.rows);
			if (pnet_lpn_detector != nullptr) {
				std::list<float> confidence_one_lp;
				std::list < cv::Rect> one_lp;
				std::list<int> classIds_one_lp;
				const int classId_last_country = NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE + NUMBER_OF_COUNTRIES - 1;
				float nmsThreshold = 0.3f;
				//it selects just one lpn although all lps have been detected and stored in double linked lists, then from these lists, selects the one that is the best 
//(with best confidences of its characters and with greateast size)
				//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
				pnet_lpn_detector->detect_with_different_confidences_then_separate_plates(frame, classes,
					confidences, boxes, confidence_one_lp, one_lp, classIds_one_lp,
					classId_last_country,
					nmsThreshold);
				if (one_lp.size()) {
					//one_lp is the list of the boundinx boxes of a lp. one_lp is the best lps in the image (see func detect_with_different_confidences_then_separate_plates). And we are sure that the front element of the boxes is the lp.
					cv::Rect lpn_roi = one_lp.front();
#ifdef _DEBUG
					//given the index of a bounding box, we can predict if this box is a single character or if it represents the license plate area or if it is the roi of an entire vehicle
					//single character--> returns 1
					//license plate--> returns 2
					//vehicle--> returns 3
					//negative index--> returns 0 must be an error
					assert(is_this_box_a_character_a_license_plate_or_a_vehicle(classIds_one_lp.front(), classId_last_country) == 2);
#endif //_DEBUG
					float confidence_lpn_roi = confidence_one_lp.front();
					if (lpn_roi.width > 0 && lpn_roi.height > 0 && lpn_roi.x >= 0 && lpn_roi.y >= 0 &&
						confidence_lpn_roi > 0.0f && lpn_roi.x + lpn_roi.width <= frame.cols && lpn_roi.y <= frame.rows) {
						get_larger_roi(lpn_roi, frame.cols, frame.rows, 2.f, 3.f);
#ifdef _DEBUG
						if (lpn_roi.area() < ((frame.rows * frame.cols) / //40
							1
							)) {
#else // _DEBUG		
						if (lpn_roi.area() < ((frame.rows * frame.cols) / 10)) {
#endif //_DEBUG
							//image of lpn
							cv::Mat subimage_;
							cv::Point  top_left_;
							cv::Point  top_right_;
							cv::Point  bottom_right_;
							cv::Point  bottom_left_;
							cv::Rect rect_OpenLP_;
							bool plaque_trouvee = trouve_la_plaque(frame,
								classIds_one_lp, one_lp,
								top_left_,
								top_right_,
								bottom_right_,
								bottom_left_, rect_OpenLP_);
							if (plaque_trouvee) {
								lpn_roi = rect_OpenLP_;
								get_larger_roi(lpn_roi, frame.cols, frame.rows, 1.2f, 1.2f);
							}
								subimage_ = frame(lpn_roi);
							std::vector<int> vet_of_classIds;
							std::vector<float> vect_of_confidences;
							std::vector<cv::Rect> vect_of_detected_boxes;
							//Given the @p input frame, create input blob, run net and return result detections.
							Yolov5_anpr_onxx_detector* pnet_inside_lpn_detector = get_detector_with_smallest_size_bigger_than_image(parking_detectors, subimage_.cols, subimage_.rows);
							std::list < std::list<float>> confidences_inside_lp; std::list < std::list<int>> classes_inside_lp; std::list < std::list<cv::Rect>> boxes_inside_lp;
							std::list<float> inside_lp_confidence_one_lp;
							std::list < cv::Rect> inside_lp_one_lp;
							std::list<int> inside_lp_classIds_one_lp;
							if (pnet_inside_lpn_detector != nullptr) {
								pnet_inside_lpn_detector->detect_with_different_confidences_then_separate_plates(subimage_, plates_types_classifier, classes_inside_lp,
									confidences_inside_lp, boxes_inside_lp, inside_lp_confidence_one_lp, inside_lp_one_lp, inside_lp_classIds_one_lp,
									classId_last_country,
									nmsThreshold);
							}
							else pnet_lpn_detector->detect_with_different_confidences_then_separate_plates(subimage_, plates_types_classifier, classes_inside_lp,
								confidences_inside_lp, boxes_inside_lp, inside_lp_confidence_one_lp, inside_lp_one_lp, inside_lp_classIds_one_lp,
								classId_last_country,
								nmsThreshold);
							//the nnet has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number. 
							//it can deal with license pates that have two lines of charcaters
							std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
							std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
							const float nmsThreshold_lpn = 0.4f;
							std::string lpn = get_best_lpn(inside_lp_one_lp, inside_lp_confidence_one_lp, inside_lp_classIds_one_lp,
								tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold_lpn);
							if (inside_lp_one_lp.size() > 2) {
								std::vector<float>::iterator it_confidences(tri_left_confidences.begin());
								std::vector<cv::Rect>::iterator it_boxes(tri_left_vect_of_detected_boxes.begin());
								std::vector<int>::iterator it_out_classes_(tri_left_classIds.begin());
								while (it_out_classes_ != tri_left_classIds.end()
									&& it_confidences != tri_left_confidences.end() && it_boxes != tri_left_vect_of_detected_boxes.end()) {
									if (
										(*it_out_classes_ < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE) ||
										(lpn.empty())
										) {
										//now we must change the coordinates of the box to fit global image
										cv::Rect box_in_global_image(it_boxes->x, it_boxes->y, it_boxes->width, it_boxes->height);
										chosen_lp_boxes.push_back(box_in_global_image);
										chosen_lp_classIds.push_back(*it_out_classes_);
										chosen_lp_confidences.push_back(*it_confidences);
									}
									it_out_classes_++;
									it_confidences++;
									it_boxes++;
								}
#ifdef _DEBUG
								assert(chosen_lp_confidences.size() == chosen_lp_classIds.size() && chosen_lp_boxes.size() == chosen_lp_classIds.size());
								assert(lpn.length() == chosen_lp_classIds.size() || lpn.length() == 0);
#endif //_DEBUG
								lpns.push_back(lpn);
								lp_country_class.push_back(classIds_one_lp.front());
								lp_rois.push_back(one_lp.front());
							}
							else {
								get_best_lpn(one_lp, confidence_one_lp, classIds_one_lp,
									//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
									lpns, lp_country_class, lp_rois,
									//detection inside the chosen lp
									chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes);
							}
						}
						else {
							get_best_lpn(one_lp, confidence_one_lp, classIds_one_lp,
								//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
								lpns, lp_country_class, lp_rois,
								//detection inside the chosen lp
								chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes);
						}
						}
					else {
						get_best_lpn(one_lp, confidence_one_lp, classIds_one_lp,
							//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
							lpns, lp_country_class, lp_rois,
							//detection inside the chosen lp
							chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes);
					}
					}
				else {
					//image of lpn
					cv::Mat subimage_;
					subimage_ = frame;
					std::vector<int> vet_of_classIds;
					std::vector<float> vect_of_confidences;
					std::vector<cv::Rect> vect_of_detected_boxes;
					//Given the @p input frame, create input blob, run net and return result detections.
					Yolov5_anpr_onxx_detector* pnet_inside_lpn_detector = get_detector_with_smallest_size_bigger_than_image(parking_detectors, subimage_.cols, subimage_.rows);
					std::list < std::list<float>> confidences_inside_lp; std::list < std::list<int>> classes_inside_lp; std::list < std::list<cv::Rect>> boxes_inside_lp;
					std::list<float> inside_lp_confidence_one_lp;
					std::list < cv::Rect> inside_lp_one_lp;
					std::list<int> inside_lp_classIds_one_lp;
					if (pnet_inside_lpn_detector != nullptr) {
						pnet_inside_lpn_detector->detect_with_different_confidences_then_separate_plates(subimage_, plates_types_classifier, classes_inside_lp,
							confidences_inside_lp, boxes_inside_lp, inside_lp_confidence_one_lp, inside_lp_one_lp, inside_lp_classIds_one_lp,
							classId_last_country,
							nmsThreshold);
					}
					//the nnet has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number. 
					//it can deal with license pates that have two lines of charcaters
					std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
					std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
					const float nmsThreshold_lpn = 0.4f;
					std::string lpn = get_best_lpn(inside_lp_one_lp, inside_lp_confidence_one_lp, inside_lp_classIds_one_lp,
						tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold_lpn);
					if (inside_lp_one_lp.size() > 2) {
						std::vector<float>::iterator it_confidences(tri_left_confidences.begin());
						std::vector<cv::Rect>::iterator it_boxes(tri_left_vect_of_detected_boxes.begin());
						std::vector<int>::iterator it_out_classes_(tri_left_classIds.begin());
						while (it_out_classes_ != tri_left_classIds.end()
							&& it_confidences != tri_left_confidences.end() && it_boxes != tri_left_vect_of_detected_boxes.end()) {
							if (
								(*it_out_classes_ < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE) ||
								(lpn.empty())
								) {
								//now we must change the coordinates of the box to fit global image
								cv::Rect box_in_global_image(it_boxes->x, it_boxes->y, it_boxes->width, it_boxes->height);
								chosen_lp_boxes.push_back(box_in_global_image);
								chosen_lp_classIds.push_back(*it_out_classes_);
								chosen_lp_confidences.push_back(*it_confidences);
							}
							it_out_classes_++;
							it_confidences++;
							it_boxes++;
						}
#ifdef _DEBUG
						assert(chosen_lp_confidences.size() == chosen_lp_classIds.size() && chosen_lp_boxes.size() == chosen_lp_classIds.size());
						assert(lpn.length() == chosen_lp_classIds.size() || lpn.length() == 0);
#endif //_DEBUG
						lpns.push_back(lpn);
						if (inside_lp_classIds_one_lp.front() >= 36)
							lp_country_class.push_back(inside_lp_classIds_one_lp.front());
						else lp_country_class.push_back(36);
						lp_rois.push_back(cv::Rect(0, 0, frame.cols, frame.rows));
					}
				}
				}
			}
		}
	}
	//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
	void two_stage_lpr(Yolov5_anpr_onxx_detector & freeflow_detector, Yolov5_anpr_onxx_detector & parking_detector
		, Plates_types_classifier& plates_types_classifier,
		const cv::Mat& frame,
		//double linked lists to separate lps
		std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
		//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
		std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
		//detection inside the chosen lp
		std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
	)	
	{
		if (frame.rows > 0 && frame.cols > 0) {
			//global image
			cv::Scalar mean_ = cv::mean(frame);
			if (mean_[0] > 2.0f && mean_[0] < 250.0f) {
				std::vector<int> classIds_lpn;
				std::vector<float> confidences_lpn;
				std::vector<cv::Rect> vect_of_detected_boxes_lpn;
				//Given the @p input frame, create input blob, run net and return result detections.
				//
				if (freeflow_detector.is_valid()) {
					std::list<float> confidence_one_lp;
					std::list < cv::Rect> one_lp;
					std::list<int> classIds_one_lp;
					const int classId_last_country = NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE + NUMBER_OF_COUNTRIES - 1;
					float nmsThreshold = 0.3f;
					//it selects just one lpn although all lps have been detected and stored in double linked lists, then from these lists, selects the one that is the best 
	//(with best confidences of its characters and with greateast size)
					//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
	//and remaining elements are characters
					freeflow_detector.detect_with_different_confidences_then_separate_plates(frame, classes,
						confidences, boxes, confidence_one_lp, one_lp, classIds_one_lp,
						classId_last_country,
						nmsThreshold);
					if (one_lp.size()) {
						//one_lp is the list of the boundinx boxes of a lp. one_lp is the best lps in the image (see func detect_with_different_confidences_then_separate_plates). And we are sure that the front element of the boxes is the lp.
						cv::Rect lpn_roi = one_lp.front();
#ifdef _DEBUG
						//given the index of a bounding box, we can predict if this box is a single character or if it represents the license plate area or if it is the roi of an entire vehicle
						//single character--> returns 1
						//license plate--> returns 2
						//vehicle--> returns 3
						//negative index--> returns 0 must be an error
						assert(is_this_box_a_character_a_license_plate_or_a_vehicle(classIds_one_lp.front(), classId_last_country) == 2);
#endif //_DEBUG
						float confidence_lpn_roi = confidence_one_lp.front();
						if (lpn_roi.width > 0 && lpn_roi.height > 0 && lpn_roi.x >= 0 && lpn_roi.y >= 0 &&
							confidence_lpn_roi > 0.0f && lpn_roi.x + lpn_roi.width <= frame.cols && lpn_roi.y <= frame.rows) {
							get_larger_roi(lpn_roi, frame.cols, frame.rows, 2.f, 3.f);
#ifdef _DEBUG
							if (lpn_roi.area() < ((frame.rows * frame.cols) / //40
								1
								)) {
#else // _DEBUG		
							if (lpn_roi.area() < ((frame.rows * frame.cols) / 10)) {
#endif //_DEBUG
								//image of lpn
								cv::Mat subimage_;
								cv::Point  top_left_;
								cv::Point  top_right_;
								cv::Point  bottom_right_;
								cv::Point  bottom_left_;
								cv::Rect rect_OpenLP_;
								bool plaque_trouvee = trouve_la_plaque(frame,
									classIds_one_lp, one_lp,
									top_left_,
									top_right_,
									bottom_right_,
									bottom_left_, rect_OpenLP_);
								if (plaque_trouvee) {
									lpn_roi = rect_OpenLP_;
									get_larger_roi(lpn_roi, frame.cols, frame.rows, 1.2f, 1.2f);
								}
								subimage_ = frame(lpn_roi);
								std::vector<int> vet_of_classIds;
								std::vector<float> vect_of_confidences;
								std::vector<cv::Rect> vect_of_detected_boxes;
								//Given the @p input frame, create input blob, run net and return result detections.
								std::list < std::list<float>> confidences_inside_lp; std::list < std::list<int>> classes_inside_lp; std::list < std::list<cv::Rect>> boxes_inside_lp;
								std::list<float> inside_lp_confidence_one_lp;
								std::list < cv::Rect> inside_lp_one_lp;
								std::list<int> inside_lp_classIds_one_lp;
								if (parking_detector.is_valid()) {
									parking_detector.detect_with_different_confidences_then_separate_plates(subimage_, plates_types_classifier, classes_inside_lp,
										confidences_inside_lp, boxes_inside_lp, inside_lp_confidence_one_lp, inside_lp_one_lp, inside_lp_classIds_one_lp,
										classId_last_country,
										nmsThreshold);
								}
								else freeflow_detector.detect_with_different_confidences_then_separate_plates(subimage_, plates_types_classifier, classes_inside_lp,
									confidences_inside_lp, boxes_inside_lp, inside_lp_confidence_one_lp, inside_lp_one_lp, inside_lp_classIds_one_lp,
									classId_last_country,
									nmsThreshold);
								//the nnet has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number. 
								//it can deal with license pates that have two lines of charcaters
								std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
								std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
								const float nmsThreshold_lpn = 0.4f;
								std::string lpn = get_best_lpn(inside_lp_one_lp, inside_lp_confidence_one_lp, inside_lp_classIds_one_lp,
									tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold_lpn);
								if (inside_lp_one_lp.size() > 2) {
									std::vector<float>::iterator it_confidences(tri_left_confidences.begin());
									std::vector<cv::Rect>::iterator it_boxes(tri_left_vect_of_detected_boxes.begin());
									std::vector<int>::iterator it_out_classes_(tri_left_classIds.begin());
									while (it_out_classes_ != tri_left_classIds.end()
										&& it_confidences != tri_left_confidences.end() && it_boxes != tri_left_vect_of_detected_boxes.end()) {
										if (
											(*it_out_classes_ < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE) ||
											(lpn.empty())
											) {
											//now we must change the coordinates of the box to fit global image
											cv::Rect box_in_global_image(it_boxes->x, it_boxes->y, it_boxes->width, it_boxes->height);
											chosen_lp_boxes.push_back(box_in_global_image);
											chosen_lp_classIds.push_back(*it_out_classes_);
											chosen_lp_confidences.push_back(*it_confidences);
										}
										it_out_classes_++;
										it_confidences++;
										it_boxes++;
									}
#ifdef _DEBUG
									assert(chosen_lp_confidences.size() == chosen_lp_classIds.size() && chosen_lp_boxes.size() == chosen_lp_classIds.size());
									assert(lpn.length() == chosen_lp_classIds.size() || lpn.length() == 0);
#endif //_DEBUG
									lpns.push_back(lpn);
									lp_country_class.push_back(classIds_one_lp.front());
									lp_rois.push_back(one_lp.front());
								}
								else {
									get_best_lpn(one_lp, confidence_one_lp, classIds_one_lp,
										//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
										lpns, lp_country_class, lp_rois,
										//detection inside the chosen lp
										chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes);
								}
							}
							else {
								get_best_lpn(one_lp, confidence_one_lp, classIds_one_lp,
									//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
									lpns, lp_country_class, lp_rois,
									//detection inside the chosen lp
									chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes);
							}
							}
						else {
							get_best_lpn(one_lp, confidence_one_lp, classIds_one_lp,
								//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
								lpns, lp_country_class, lp_rois,
								//detection inside the chosen lp
								chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes);
						}
						}
					else {
						//image of lpn
						cv::Mat subimage_;
						subimage_ = frame;
						std::vector<int> vet_of_classIds;
						std::vector<float> vect_of_confidences;
						std::vector<cv::Rect> vect_of_detected_boxes;
						//Given the @p input frame, create input blob, run net and return result detections.
						std::list < std::list<float>> confidences_inside_lp; std::list < std::list<int>> classes_inside_lp; std::list < std::list<cv::Rect>> boxes_inside_lp;
						std::list<float> inside_lp_confidence_one_lp;
						std::list < cv::Rect> inside_lp_one_lp;
						std::list<int> inside_lp_classIds_one_lp;
						if (parking_detector.is_valid()) {
							parking_detector.detect_with_different_confidences_then_separate_plates(subimage_, plates_types_classifier, classes_inside_lp,
								confidences_inside_lp, boxes_inside_lp, inside_lp_confidence_one_lp, inside_lp_one_lp, inside_lp_classIds_one_lp,
								classId_last_country,
								nmsThreshold);
						}
						//the nnet has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number. 
						//it can deal with license pates that have two lines of charcaters
						std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
						std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
						const float nmsThreshold_lpn = 0.4f;
						std::string lpn = get_best_lpn(inside_lp_one_lp, inside_lp_confidence_one_lp, inside_lp_classIds_one_lp,
							tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold_lpn);
						if (inside_lp_one_lp.size() > 2) {
							std::vector<float>::iterator it_confidences(tri_left_confidences.begin());
							std::vector<cv::Rect>::iterator it_boxes(tri_left_vect_of_detected_boxes.begin());
							std::vector<int>::iterator it_out_classes_(tri_left_classIds.begin());
							while (it_out_classes_ != tri_left_classIds.end()
								&& it_confidences != tri_left_confidences.end() && it_boxes != tri_left_vect_of_detected_boxes.end()) {
								if (
									(*it_out_classes_ < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE) ||
									(lpn.empty())
									) {
									//now we must change the coordinates of the box to fit global image
									cv::Rect box_in_global_image(it_boxes->x, it_boxes->y, it_boxes->width, it_boxes->height);
									chosen_lp_boxes.push_back(box_in_global_image);
									chosen_lp_classIds.push_back(*it_out_classes_);
									chosen_lp_confidences.push_back(*it_confidences);
								}
								it_out_classes_++;
								it_confidences++;
								it_boxes++;
							}
#ifdef _DEBUG
							assert(chosen_lp_confidences.size() == chosen_lp_classIds.size() && chosen_lp_boxes.size() == chosen_lp_classIds.size());
							assert(lpn.length() == chosen_lp_classIds.size() || lpn.length() == 0);
#endif //_DEBUG
							lpns.push_back(lpn);
							if (inside_lp_classIds_one_lp.front() >= 36)
								lp_country_class.push_back(inside_lp_classIds_one_lp.front());
							else lp_country_class.push_back(36);
							lp_rois.push_back(cv::Rect(0, 0, frame.cols, frame.rows));
						}
					}
					}
				}
			}
		}
	//
	//Given the @p input frame, create input blob, run net and return result detections.
	//it selects just one lpn although all lps are detected in double linked lists
	////this func can manage list of boxes of characters that dont have an englobing lp box (gloabal rect)
	////output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
	////and remaining elements are characters.
//Produces double linked lists : inside list is for characters and outside list is for plates.
void Yolov5_anpr_onxx_detector::detect_lpn_and_add_lp_and_vehicle_if_necessary(const cv::Mat& frame, Plates_types_classifier& plates_types_classifier, std::list < std::vector<int>>& classIds,
		std::list < std::vector<float>>& confidences, std::list < std::vector<cv::Rect>>& boxes
		, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp,
		const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.We remember that ascii(latin) characters come fist(36 classes) then come the license plates countries(another 60 classses) then come a long list of vehicles classes
		//const C_OCROutputs & availableAlpha,
		const float confThreshold, float nmsThreshold)
	{
		std::list<std::string> lpns;
		//this func can manage list of boxes of characters that dont have an englobing lp box (gloabal rect)
		//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
	//and remaining elements are characters
		detect_and_add_lp_and_vehicle_if_necessary(frame, classIds,
			confidences, boxes, lpns,
			classId_last_country,
			confThreshold, nmsThreshold);
		//now choose best lpn
		//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
		plates_types_classifier.get_best_plate(frame, classIds,
			confidences, boxes, confidence_one_lp, one_lp, classIds_one_lp);
	}
	//Given the @p input frame, create input blob, run net then, from result detections, assembly license plates present in the input image.
	void Yolov5_anpr_onxx_detector::detect_with_different_confidences_then_separate_plates(const cv::Mat& frame, Plates_types_classifier& plates_types_classifier, std::list < std::list<int>>& classIds,
		std::list < std::list<float>>& confidences, std::list < std::list<cv::Rect>>& boxes
		, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp,
		const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.We remember that ascii(latin) characters come fist(36 classes) then come the license plates countries(another 60 classses) then come a long list of vehicles classes
		//const C_OCROutputs & availableAlpha,
		float nmsThreshold)
	{
		std::list <std::list<std::string>> lpns;
		detect_with_different_confidences_then_separate_plates(frame, classIds,
			confidences, boxes, lpns
			, nmsThreshold, classId_last_country);
		//now choose best lpn
		//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
		plates_types_classifier.get_best_plate(frame, classIds,
			confidences, boxes, confidence_one_lp, one_lp, classIds_one_lp);
	}
	//////////////////////////////////////////////////////////////////////
	// Construction/Destruction
	//////////////////////////////////////////////////////////////////////
	Plates_types_classifier::Plates_types_classifier(Ort::Env& env, const ORTCHAR_T* model_path, const Ort::SessionOptions& options
		, const std::vector<std::string>& labels_)
		: OnnxDetector(env, model_path, options), labels(labels_)
	{
	}
	Plates_types_classifier::Plates_types_classifier(Ort::Env& env, const void* model_data, size_t model_data_length, const Ort::SessionOptions& options
		, const std::vector<std::string>& labels_)
		: OnnxDetector(env, model_data, model_data_length, options), labels(labels_)
	{
	}
	Plates_types_classifier::Plates_types_classifier(Ort::Env& env, const ORTCHAR_T* model_path, const Ort::SessionOptions& options
		, const std::string& labels_filename)
		: OnnxDetector(env, model_path, options)
	{
		labels = get_plates_types_labels(labels_filename);
	}
	Plates_types_classifier::Plates_types_classifier(Ort::Env& env, const void* model_data, size_t model_data_length, const Ort::SessionOptions& options
		, const std::string& labels_filename)
		: OnnxDetector(env, model_data, model_data_length, options)
	{
		labels = get_plates_types_labels(labels_filename);
	}
	Plates_types_classifier::~Plates_types_classifier()
	{
	}
	int Plates_types_classifier::GetPlatesClasse(const cv::Mat& img, float& uncalibrated_confidence) {
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
				cv::cvtColor(img, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
				//resizedImageRGB = img.clone();
			}
			float pad_w = -1.0f, pad_h = -1.0f, scale = -1.0f;
			cv::resize(resizedImageRGB, resizedImageRGB,
				cv::Size(int(inputDims.at(3)), int(inputDims.at(2))),
				cv::InterpolationFlags::INTER_CUBIC);
			resizedImageRGB.convertTo(resizedImage, CV_32FC3, 1.0f / 255.0f);
			/**/
			resizedImage -= 0.1307;
			resizedImage /= 0.3081;
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
			assert(outputDims.size() == 2);
			assert(outputDims[0] == 1);
			assert(outputDims[1] == 502);//502 types of lps
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
			size_t dimensionsCount = outputTensorInfo.GetDimensionsCount();//2
#ifdef _DEBUG
			assert(dimensionsCount == 2);
#endif //_DEBUG
			int predId = 0;
			float activation = 0;
			float maxActivation = std::numeric_limits<float>::lowest();
			float expSum = 0;
			for (int i = 0; i < outputDims[1]; i++)
			{
				activation = outputTensorValues.at(i);
				expSum += std::exp(activation);
				if (activation > maxActivation)
				{
					predId = i;
					maxActivation = activation;
				}
			}
			//std::cout << "Predicted Label ID: " << predId << std::endl;
			//std::cout << "Predicted Label: " << predId << std::endl;
			uncalibrated_confidence = std::exp(maxActivation) / expSum;
			//std::cout << "Uncalibrated Confidence: " << uncalibrated_confidence	<< std::endl;
			int classe = predId;
			return classe;
		}
		uncalibrated_confidence = 0.0f;
		return -1;
	}
	std::string Plates_types_classifier::GetPlatesType(const cv::Mat& img, float& uncalibrated_confidence) {
		int classe = GetPlatesClasse(img, uncalibrated_confidence);
		if (classe >= 0 && classe < labels.size()) {
			return labels[classe];
		}
		else {
			uncalibrated_confidence = 0.0f;
			return std::string();
		}
	}
	//nov 21 update this func with plates_types_classifier classifier
	//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
//Uses double linked lists : inside list is for characters and outside list is for plates.
	void Plates_types_classifier::get_best_plate(const cv::Mat& frame,
		//detections when they are separated license plates by license plates
		const std::list < std::vector<int>>& classIds, const std::list < std::vector<float>>& confidences, const std::list < std::vector<cv::Rect>>& boxes
		//output the list of the best (most probable/readable) lp
		, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp)
	{
		std::list < std::vector<cv::Rect>>::const_iterator it_boxes(boxes.begin());
		std::list < std::vector<float>>::const_iterator it_confidences(confidences.begin());
		std::list < std::vector<int>>::const_iterator it_classIds(classIds.begin());
		float best_score = 0.0f;
		float uncalibrated_confidence;
		std::string plate_type = GetPlatesType(frame, uncalibrated_confidence);
		bool bi_level_plate_type = (plate_type.find('_') != std::string::npos);
		int distance_to_plates_type = 1000;
		std::string found_plates_type;
		std::list <std::string > l_plates_type;
		const int number_of_characters_latin_numberplate = 36;
		while (it_boxes != boxes.end()
			&& it_confidences != confidences.end() && it_classIds != classIds.end()) {
#ifdef _DEBUG		
			assert(it_classIds->size() == it_confidences->size() && it_classIds->size() == it_boxes->size() && it_classIds->size() >= 2);
			//1;->ok
		//2;->size too small
		//4;->second detection is not a vehicle
		//6;->detection after first two ones, is not a character
			int ids_are_ok = is_detections_of_a_unique_license_plate(*it_classIds);
			assert(ids_are_ok == 1);
#endif //_DEBUG
			float current_score = //(float)(it_boxes->size() - 2) * 
				//From confidences of detections of all boxes of a plate, we get the average confidence.
				//(it_boxes->front().width * it_boxes->front().height) * 
				get_average_confidence_of_license_plate(*it_classIds,
					*it_confidences);
			std::string current_plate_type = get_plate_type(
				*it_boxes,
				*it_classIds, number_of_characters_latin_numberplate
			);
			l_plates_type.push_back(current_plate_type);
			Levenshtein lev;
			int editdistance = lev.Get(current_plate_type.c_str(), current_plate_type.length(), plate_type.c_str(), plate_type.length());
			if (distance_to_plates_type > editdistance) {
				distance_to_plates_type = editdistance;
				found_plates_type = current_plate_type;
			}
			if (best_score < current_score)
			{
				best_score = current_score;
				confidence_one_lp.clear();
				one_lp.clear();
				classIds_one_lp.clear();
				std::copy(it_confidences->begin(), it_confidences->end(), std::back_inserter(confidence_one_lp));
				std::copy(it_boxes->begin(), it_boxes->end(), std::back_inserter(one_lp));
				std::copy(it_classIds->begin(), it_classIds->end(), std::back_inserter(classIds_one_lp));
			}
			it_boxes++;
			it_confidences++;
			it_classIds++;
		}
		if (uncalibrated_confidence > 0.1f) {
			best_score = 0.0f;
			it_boxes = (boxes.begin());
			it_confidences = (confidences.begin());
			it_classIds = (classIds.begin());
			std::list < std::string>::const_iterator it_plates_types(l_plates_type.begin());
			while (it_plates_types != l_plates_type.end()
				&& it_boxes != boxes.end()
				&& it_confidences != confidences.end() && it_classIds != classIds.end()) {
				Levenshtein lev;
				int editdistance = lev.Get(it_plates_types->c_str(), it_plates_types->length(), plate_type.c_str(), plate_type.length());
				if (editdistance == distance_to_plates_type) {
					float current_score = //(float)(it_boxes->size() - 2) * 
						//From confidences of detections of all boxes of a plate, we get the average confidence.
						//(it_boxes->front().width * it_boxes->front().height) * 
						get_average_confidence_of_license_plate(*it_classIds,
							*it_confidences);
					if (best_score < current_score)
					{
						best_score = current_score;
						confidence_one_lp.clear();
						one_lp.clear();
						classIds_one_lp.clear();
						std::copy(it_confidences->begin(), it_confidences->end(), std::back_inserter(confidence_one_lp));
						std::copy(it_boxes->begin(), it_boxes->end(), std::back_inserter(one_lp));
						std::copy(it_classIds->begin(), it_classIds->end(), std::back_inserter(classIds_one_lp));
					}
				}
				it_boxes++;
				it_confidences++;
				it_classIds++;
				it_plates_types++;
			}
		}
	}
	//nov 21 update this func with plates_types_classifier classifier
	//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
	void Plates_types_classifier::get_best_plate(const cv::Mat& frame,
		//detections when they are separated license plates by license plates
		const std::list < std::list<int>>& classIds, const std::list < std::list<float>>& confidences, const std::list < std::list<cv::Rect>>& boxes
		//output the list of the best (most probable/readable) lp
		, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp)
	{
		std::list < std::list<cv::Rect>>::const_iterator it_boxes(boxes.begin());
		std::list < std::list<float>>::const_iterator it_confidences(confidences.begin());
		std::list < std::list<int>>::const_iterator it_classIds(classIds.begin());
		float best_score = 0.0f;
		float uncalibrated_confidence;
		std::string plate_type = GetPlatesType(frame, uncalibrated_confidence);
		bool bi_level_plate_type = (plate_type.find('_') != std::string::npos);
		int distance_to_plates_type = 1000;
		std::string found_plates_type;
		std::list <std::string > l_plates_type;
		const int number_of_characters_latin_numberplate = 36;
		while (it_boxes != boxes.end()
			&& it_confidences != confidences.end() && it_classIds != classIds.end()) {
#ifdef _DEBUG		
			assert(it_classIds->size() == it_confidences->size() && it_classIds->size() == it_boxes->size() && it_classIds->size() >= 2);
			//1;->ok
		//2;->size too small
		//4;->second detection is not a vehicle
		//6;->detection after first two ones, is not a character
			int ids_are_ok = is_detections_of_a_unique_license_plate(*it_classIds);
			assert(ids_are_ok == 1);
#endif //_DEBUG
			float current_score = //(float)(it_boxes->size() - 2) * 
				//From confidences of detections of all boxes of a plate, we get the average confidence.
				//(it_boxes->front().width * it_boxes->front().height) * 
				get_average_confidence_of_license_plate(*it_classIds,
					*it_confidences);
			std::string current_plate_type = get_plate_type(
				*it_boxes,
				*it_classIds, number_of_characters_latin_numberplate
			);
			l_plates_type.push_back(current_plate_type);
			Levenshtein lev;
			int editdistance = lev.Get(current_plate_type.c_str(), current_plate_type.length(), plate_type.c_str(), plate_type.length());
			if (distance_to_plates_type > editdistance) {
				distance_to_plates_type = editdistance;
				found_plates_type = current_plate_type;
			}
			if (best_score < current_score)
			{
				best_score = current_score;
				confidence_one_lp.clear();
				one_lp.clear();
				classIds_one_lp.clear();
				std::copy(it_confidences->begin(), it_confidences->end(), std::back_inserter(confidence_one_lp));
				std::copy(it_boxes->begin(), it_boxes->end(), std::back_inserter(one_lp));
				std::copy(it_classIds->begin(), it_classIds->end(), std::back_inserter(classIds_one_lp));
			}
			it_boxes++;
			it_confidences++;
			it_classIds++;
		}
#ifdef _DEBUG		
		if (distance_to_plates_type == 1) {
			std::cout << "distance_to_plates_type : " << distance_to_plates_type << std::endl;
			std::cout << "plates_type from yolov4 detector: " << found_plates_type << std::endl;
			std::cout << "plate_type from image classifier: " << plate_type << std::endl;
		}
#endif //_DEBUG
		if (uncalibrated_confidence > 1.4f) {
			best_score = 0.0f;
			found_plates_type = "";
			it_boxes = (boxes.begin());
			it_confidences = (confidences.begin());
			it_classIds = (classIds.begin());
			std::list < std::string>::const_iterator it_plates_types(l_plates_type.begin());
			while (it_plates_types != l_plates_type.end()
				&& it_boxes != boxes.end()
				&& it_confidences != confidences.end() && it_classIds != classIds.end()) {
				Levenshtein lev;
				int editdistance = lev.Get(it_plates_types->c_str(), it_plates_types->length(), plate_type.c_str(), plate_type.length());
				if (editdistance == distance_to_plates_type) {
					float current_score = //(float)(it_boxes->size() - 2) * 
						//From confidences of detections of all boxes of a plate, we get the average confidence.
						//(it_boxes->front().width * it_boxes->front().height) * 
						get_average_confidence_of_license_plate(*it_classIds,
							*it_confidences);
					if (best_score < current_score)
					{
						best_score = current_score;
						found_plates_type = *it_plates_types;
						confidence_one_lp.clear();
						one_lp.clear();
						classIds_one_lp.clear();
						std::copy(it_confidences->begin(), it_confidences->end(), std::back_inserter(confidence_one_lp));
						std::copy(it_boxes->begin(), it_boxes->end(), std::back_inserter(one_lp));
						std::copy(it_classIds->begin(), it_classIds->end(), std::back_inserter(classIds_one_lp));
					}
				}
				it_boxes++;
				it_confidences++;
				it_classIds++;
				it_plates_types++;
			}
		}
		if (uncalibrated_confidence > 1.8f) {
			std::list<cv::Rect> boxes_tmp;
			std::copy(one_lp.begin(), one_lp.end(), std::back_inserter(boxes_tmp));
			std::list<int> classIds_tmp;
			std::copy(classIds_one_lp.begin(), classIds_one_lp.end(), std::back_inserter(classIds_tmp));
			//filter out lpn box
			//***************************************************
			//                  FILTER
			//***************************************************
			filter_out_everything_but_characters(boxes_tmp,
				classIds_tmp);
			sort_from_left_to_right(boxes_tmp, classIds_tmp);
			std::string lpn = get_lpn(classIds_tmp);
			if (plates_types_differ_with_one_character(found_plates_type, lpn, plate_type)) {
#ifdef _DEBUG		
				assert(one_lp.size() == confidence_one_lp.size() && one_lp.size() == classIds_one_lp.size() && one_lp.size());
#endif //_DEBUG
				std::list<cv::Rect>::const_iterator it_boxes_tmp(boxes_tmp.begin());
				std::list<int>::iterator it_classIds_tmp(classIds_tmp.begin());
				std::string::const_iterator it1(lpn.begin());
				while (it_boxes_tmp != boxes_tmp.end()
					&& it_classIds_tmp != classIds_tmp.end() && it1 != lpn.end()) {
					std::list<cv::Rect>::const_iterator it_boxes(one_lp.begin());
					std::list<int>::iterator it_classIds(classIds_one_lp.begin());
					int index = 0;
					while (it_boxes != one_lp.end()
						&& it_classIds != classIds_one_lp.end()) {
						if (*it_classIds >= 0 && *it_classIds < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE && *it_boxes == *it_boxes_tmp) {
							assert(*it_classIds >= 0 && *it_classIds < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE);
							int classId = get_index(*it1);
							*it_classIds = classId;
						}
						it_boxes++;
						it_classIds++;
					}
					it1++;
					it_boxes_tmp++;
					it_classIds_tmp++;
				}
			}
		}
	}