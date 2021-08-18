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
//from image frame, extract directly license plate number.
int Yolov5_anpr_onxx_detector::evaluate_without_lpn_detection(const cv::Mat& frame, const std::string& ExactLPN, std::list<std::list<int>>& classIds,
	std::list < std::list<float>>& confidences, std::list < std::list<cv::Rect>>& boxes,
	std::list<std::string>& lpns,
	const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.
	std::string& best_lpn,
	//output = characters in nearest lpn 
	std::list<float>& best_confidences, std::list<int>& best_classes, std::list<cv::Rect>& best_boxes)
{
	detect_and_add_lp_and_vehicle_if_necessary(frame,
		//setections when they are separated in double linked list (one list for one lp)
		classIds,
		confidences, boxes,
		lpns,
		classId_last_country
		//classId_last_country : is the class index of the last country in the list of detected classes.
	);
	int min_editdistance = SHRT_MAX;
	Levenshtein lev;
	std::list <std::string>::iterator it_lpns(lpns.begin());
	std::list < std::list<float>>::const_iterator it_confidences(confidences.begin());
	std::list < std::list<int>>::iterator it_classes(classIds.begin());
	std::list < std::list<cv::Rect>>::const_iterator it_boxes(boxes.begin());
	while (it_lpns != lpns.end() &&
		it_confidences != confidences.end() && it_classes != classIds.end() && it_boxes != boxes.end())
	{
		//int editdistance = lev.Get(ExactLPN, *it_lpns);
		int editdistance = lev.Get(ExactLPN.c_str(), ExactLPN.length(), it_lpns->c_str(), it_lpns->length());
		if (min_editdistance > editdistance) {
			min_editdistance = editdistance;
			best_lpn = *it_lpns;
			best_confidences = *it_confidences;
			best_classes = *it_classes;
			best_boxes = *it_boxes;
		}
		it_lpns++;
		it_confidences++;
		it_classes++;
		it_boxes++;
	}
	return min_editdistance;
}
// Given the @p input frame, create input blob, run net and return result detections.
//this func can manage list of boxes of characters that dont have an englobing lp box (gloabal rect)
//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
//from image frame, extract directly license plate number.
int Yolov5_anpr_onxx_detector::evaluate_without_lpn_detection(const std::string& image_filename, const std::string& ExactLPN, std::list<std::list<int>>& classIds,
	std::list < std::list<float>>& confidences, std::list < std::list<cv::Rect>>& boxes,
	std::list<std::string>& lpns,
	const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.
	std::string& best_lpn,
	//output = characters in nearest lpn 
	std::list<float>& best_confidences, std::list<int>& best_classes, std::list<cv::Rect>& best_boxes)
{
	std::filesystem::path p_(image_filename);
	if (exists(p_) && std::filesystem::is_regular_file(p_))    // does p actually exist?
	{
		int flags = -1;//as is
		cv::Mat frame = cv::imread(image_filename, flags);
		//from image frame, extract directly license plate number.
		return evaluate_without_lpn_detection(frame, ExactLPN,
			//double linked lists to separate lps
			classIds, confidences, boxes, lpns, classId_last_country, best_lpn,
			//output = characters in nearest lpn 
			best_confidences, best_classes, best_boxes
			//, confThreshold, nmsThreshold
		);
	}
	else return SHRT_MAX;
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
//and remaining elements are characters
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
float Yolov5_anpr_onxx_detector::evaluate_lpn_with_lpn_detection(const std::string& dir)
{
	return evaluate_lpn_with_lpn_detection(*this, dir);
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
float Yolov5_anpr_onxx_detector::evaluate_lpn_with_lpn_detection(Yolov5_anpr_onxx_detector& parking_detector, const std::string& dir)
{
	std::string filename = "D:\\Programmation\\LPReditor\\ocr_dataset\\test_svm.txt";
	std::ofstream O(filename.c_str(), std::ios::app);
	O << "Yolov5_anpr_onxx_detector::evaluate_lpn_with_lpn_detection " << std::endl;
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
		evaluate_lpn_with_lpn_detection(parking_detector, *it_image_filenames,
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
void Yolov5_anpr_onxx_detector::evaluate_lpn_with_lpn_detection(const std::string& image_filename,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
)
{
	return evaluate_lpn_with_lpn_detection(*this, image_filename,
		//double linked lists to separate lps
		confidences, classes, boxes,
		//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
		lpns, lp_country_class, lp_rois,
		//detection inside the chosen lp
		chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
		//,	availableAlpha//, must_convert_from_NO_I_O_2_NO_O
	);
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void Yolov5_anpr_onxx_detector::evaluate_lpn_with_lpn_detection(Yolov5_anpr_onxx_detector& parking_detector, const std::string& image_filename,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
)
{
	if (parking_detector.max_image_size() > max_image_size())
		return parking_detector.evaluate_lpn_with_lpn_detection(*this, image_filename,
			//double linked lists to separate lps
			confidences, classes, boxes,
			//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
			lpns, lp_country_class, lp_rois,
			//detection inside the chosen lp
			chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
			//,	availableAlpha//, must_convert_from_NO_I_O_2_NO_O
		);
	else {
		std::filesystem::path p_(image_filename);
		if (exists(p_) && std::filesystem::is_regular_file(p_))    // does p actually exist?
		{
			int flags = -1;//as is
			cv::Mat frame = cv::imread(image_filename, flags);
			return evaluate_lpn_with_lpn_detection(parking_detector, frame,
				//double linked lists to separate lps
				confidences, classes, boxes,
				//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
				lpns, lp_country_class, lp_rois,
				//detection inside the chosen lp
				chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
				//,	availableAlpha//, must_convert_from_NO_I_O_2_NO_O
			);
		}
	}
}
void get_larger_roi(cv::Rect& lpn_roi, const int width, const int height
	//, int & left_translation, int & top_translation
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
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void Yolov5_anpr_onxx_detector::evaluate_lpn_with_lpn_detection(Yolov5_anpr_onxx_detector& parking_detector, const cv::Mat& frame,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
)
{
	if (parking_detector.max_image_size() > max_image_size())
		return parking_detector.evaluate_lpn_with_lpn_detection(*this, frame,
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
		return ::evaluate_lpn_with_lpn_detection(freeflow_detectors, parking_detectors,
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
void evaluate_lpn_with_lpn_detection(const std::list<Yolov5_anpr_onxx_detector*>& detectors, const std::string& image_filename,
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
		evaluate_lpn_with_lpn_detection(detectors, detectors,
			frame,
			//double linked lists to separate lps
			confidences, classes, boxes,
			//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
			lpns, lp_country_class, lp_rois,
			//detection inside the chosen lp
			chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
			//,	availableAlpha//, must_convert_from_NO_I_O_2_NO_O
		);
	}
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void evaluate_lpn_with_lpn_detection(const std::list<Yolov5_anpr_onxx_detector*>& freeflow_detectors, const std::list<Yolov5_anpr_onxx_detector*>& parking_detectors,
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
		evaluate_lpn_with_lpn_detection(freeflow_detectors, parking_detectors,
			frame,
			//double linked lists to separate lps
			confidences, classes, boxes,
			//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
			lpns, lp_country_class, lp_rois,
			//detection inside the chosen lp
			chosen_lp_classIds, chosen_lp_confidences, chosen_lp_boxes
			//,	availableAlpha//, must_convert_from_NO_I_O_2_NO_O
		);
	}
}
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void evaluate_lpn_with_lpn_detection(const std::list<Yolov5_anpr_onxx_detector*>& freeflow_detectors, const std::list<Yolov5_anpr_onxx_detector*>& parking_detectors,
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
				pnet_lpn_detector->detect_with_different_confidences_then_separate_plates(frame, classes,
					confidences, boxes, confidence_one_lp, one_lp, classIds_one_lp,
					classId_last_country,
					//availableAlpha, 
					0.5f);
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
						if (lpn_roi.area() < ((frame.rows * frame.cols) / 40)) {
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
							if (pnet_inside_lpn_detector != nullptr) {
								pnet_inside_lpn_detector->detect(subimage_, vet_of_classIds, vect_of_confidences, vect_of_detected_boxes);
							}
							else pnet_lpn_detector->detect(subimage_, vet_of_classIds, vect_of_confidences, vect_of_detected_boxes);
							//now filter boxes with big iou
			//if two boxes have an iou (intersection over union) that is too large, then they cannot represent two adjacent characters of the license plate 
			//so we discard the one with the lowest confidence rate
							filter_iou(vet_of_classIds,
								vect_of_confidences,
								vect_of_detected_boxes);
							if (vect_of_detected_boxes.size() > 2) {
								std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
								std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
								//the nnet has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number. 
								//it can deal with license pates that have two lines of charcaters
								std::string lpn;
								lpn =
									get_best_lpn(vect_of_detected_boxes, vect_of_confidences, vet_of_classIds,
										tri_left_vect_of_detected_boxes,
										tri_left_confidences, tri_left_classIds
									);
								std::vector<float>::iterator it_confidences(tri_left_confidences.begin());
								std::vector<cv::Rect>::iterator it_boxes(tri_left_vect_of_detected_boxes.begin());
								std::vector<int>::iterator it_out_classes_(tri_left_classIds.begin());
								//C_OCROutputs availableAlpha_LATIN(LATIN_LETTERS_LATIN_DIGITS);
								while (it_out_classes_ != tri_left_classIds.end()
									&& it_confidences != tri_left_confidences.end() && it_boxes != tri_left_vect_of_detected_boxes.end()) {
									if (
										(*it_out_classes_ < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE) ||
										(lpn.empty())
										) {
										//now we must change the coordinates of the box to fit global image
										cv::Rect box_in_global_image(it_boxes->x + lpn_roi.x, it_boxes->y + lpn_roi.y, it_boxes->width, it_boxes->height);
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
								std::string lpn = get_best_lpn(one_lp, confidence_one_lp, classIds_one_lp, tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold_lpn);
								std::vector<float>::iterator it_confidences(tri_left_confidences.begin());
								std::vector<cv::Rect>::iterator it_boxes(tri_left_vect_of_detected_boxes.begin());
								std::vector<int>::iterator it_out_classes_(tri_left_classIds.begin());
								//C_OCROutputs availableAlpha_LATIN(LATIN_LETTERS_LATIN_DIGITS);
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
							std::string lpn = get_best_lpn(one_lp, confidence_one_lp, classIds_one_lp, tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold_lpn);
							std::vector<float>::iterator it_confidences(tri_left_confidences.begin());
							std::vector<cv::Rect>::iterator it_boxes(tri_left_vect_of_detected_boxes.begin());
							std::vector<int>::iterator it_out_classes_(tri_left_classIds.begin());
							//C_OCROutputs availableAlpha_LATIN(LATIN_LETTERS_LATIN_DIGITS);
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
						std::string lpn = get_best_lpn(one_lp, confidence_one_lp, classIds_one_lp, tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold_lpn);
						std::vector<float>::iterator it_confidences(tri_left_confidences.begin());
						std::vector<cv::Rect>::iterator it_boxes(tri_left_vect_of_detected_boxes.begin());
						std::vector<int>::iterator it_out_classes_(tri_left_classIds.begin());
						//C_OCROutputs availableAlpha_LATIN(LATIN_LETTERS_LATIN_DIGITS);
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
void Yolov5_anpr_onxx_detector::evaluate_without_lpn_detection(const cv::Mat & frame, std::list<std::list<int>> & classIds,
	std::list < std::list<float>> & confidences, std::list < std::list<cv::Rect>> & boxes,
	std::list<std::string> & lpns,
	const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.
	std::string & best_lpn,
	//output = characters in nearest lpn 
	std::list<float> & best_confidences, std::list<int> & best_classes, std::list<cv::Rect> & best_boxes)
{
	detect_and_add_lp_and_vehicle_if_necessary(frame,
		//setections when they are separated in double linked list (one list for one lp)
		classIds,
		confidences, boxes,
		lpns,
		classId_last_country
		//classId_last_country : is the class index of the last country in the list of detected classes.
		//,confThreshold, nmsThreshold
	);
#ifdef _DEBUG
	assert(classIds.size() == confidences.size() && classIds.size() == boxes.size() && classIds.size() == lpns.size());
#endif //_DEBUG
	if (lpns.size() && classIds.size() == confidences.size() && classIds.size() == boxes.size() && classIds.size() == lpns.size()) {
		best_lpn = lpns.front();
		best_confidences = confidences.front();
		best_classes = classIds.front();
		best_boxes = boxes.front();
	}
}
// Given the @p input frame, create input blob, run net and return result detections.
//this func can manage list of boxes of characters that dont have an englobing lp box (gloabal rect)
//output lists looks like : first box = license plate (either a detected box either the global rect englobing other boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
void Yolov5_anpr_onxx_detector::evaluate_without_lpn_detection(const cv::Mat & frame, std::string & lpn)
{
	std::list<std::list<int>> classIds;
	std::list < std::list<float>> confidences;
	std::list < std::list<cv::Rect>> boxes;
	std::list<std::string> lpns;
	std::string best_lpn;
	std::list<float>  best_confidences; std::list<int> best_classes; std::list<cv::Rect> best_boxes;
	const int classId_last_country = NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE + NUMBER_OF_COUNTRIES - 1;
	evaluate_without_lpn_detection(frame, classIds,
		confidences, boxes,
		lpns,
		classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.
		best_lpn,
		best_confidences, best_classes, best_boxes
		//, confThreshold, nmsThreshold
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
void Yolov5_anpr_onxx_detector::evaluate_lpn_with_lpn_detection(Yolov5_anpr_onxx_detector & parking_detector, const cv::Mat & frame, std::string & lpn)
{
	std::list<std::list<int>> classIds;
	std::list < std::list<float>> confidences;
	std::list < std::list<cv::Rect>> boxes;
	std::list<std::string> lpns;
	std::list <int> lp_country_class;
	std::list < cv::Rect> lp_rois;
	std::string best_lpn;
	std::list<float>  best_confidences; std::list<int> best_classes; std::list<cv::Rect> best_boxes;
	/*
	const int classId_last_country = NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE +NUMBER_OF_COUNTRIES - 1;
	evaluate_without_lpn_detection(frame, classIds,
		confidences, boxes,
		lpns,
		classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.
		best_lpn,
		best_confidences, best_classes, best_boxes
		//, confThreshold, nmsThreshold
	);
	*/
	evaluate_lpn_with_lpn_detection(parking_detector, frame,
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
void Yolov5_anpr_onxx_detector::evaluate_without_lpn_detection(const std::string & image_filename, std::list<std::list<int>> & classIds,
	std::list < std::list<float>> & confidences, std::list < std::list<cv::Rect>> & boxes,
	std::list<std::string> & lpns,
	const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.
	std::string & best_lpn,
	//output = characters in nearest lpn 
	std::list<float> & best_confidences, std::list<int> & best_classes, std::list<cv::Rect> & best_boxes)
{
	std::filesystem::path p_(image_filename);
	if (exists(p_) && std::filesystem::is_regular_file(p_))    // does p actually exist?
	{
		int flags = -1;//as is
		cv::Mat frame = cv::imread(image_filename, flags);
		evaluate_without_lpn_detection(frame,
			//double linked lists to separate lps
			classIds, confidences, boxes, lpns, classId_last_country, best_lpn,
			//output = characters in nearest lpn 
			best_confidences, best_classes, best_boxes
			//, confThreshold, nmsThreshold
		);
	}
}