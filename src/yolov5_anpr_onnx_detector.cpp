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
// Given the @p input frame, create input blob, run net and return result detections.
void Yolov5_anpr_onxx_detector::detect(const cv::Mat& frame, std::string& lpn)
{
	/*
	bool show_image = true;
	int time_delay = 10000;
	std::string image_filename = "D:/Programmation/LPReditor-engine/LPReditor_ANPR/data/images/0000000001_3065WWA34.jpg";
	std::string lpn1;
	detect(image_filename, lpn1, show_image, time_delay);
	*/



	const float confThreshold = 0.7f;
	const float nmsThreshold = 0.5f;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> vect_of_detected_boxes;
	detect(frame, classIds, confidences, vect_of_detected_boxes, confThreshold, nmsThreshold);
	std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
	std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
	//the nnet has detected boxes that represant characters of the license plate, this function now etracts from these boxes the license plate number. 
	//it can deal with license pates that have two lines of charcaters
	lpn = get_lpn(vect_of_detected_boxes, confidences, classIds, tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold);
}
// Given the @p input frame, create input blob, run net and return result detections.
void Yolov5_anpr_onxx_detector::detect(const cv::Mat& frame, std::vector<int>& classIds,
	std::vector<float>& confidences, std::vector<cv::Rect>& boxes,
	float confThreshold, float nmsThreshold)
{
	bool preserve_aspect_ratio = true;
	std::vector<std::vector<Detection>>	 detected = OnnxDetector::Run(frame, confThreshold, nmsThreshold, preserve_aspect_ratio);
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
/** @brief given a directory, finds into it the model file.
	@param dir_path : dir path of the directory containing necessary files
	@param yolo_carac_detect_model_file : file path of the model
	 */
bool load_onnx(const std::string& dir_path, std::string& yolo_carac_detect_model_file) {
	std::size_t index = dir_path.find("deprecated", 0);
	if (index != std::string::npos) return false;
	std::filesystem::path p(dir_path);
	if (std::filesystem::exists(p) && std::filesystem::is_directory(p))    // does p actually exist?
	{
		std::filesystem::directory_iterator end_itr; // default construction yields past-the-end
		for (std::filesystem::directory_iterator itr(p); itr != end_itr; ++itr)
		{
			if (std::filesystem::is_regular_file(itr->status()))
			{
				try {
					const std::filesystem::path copie(itr->path());
					std::string filename(copie.filename().string());
					if (filename.find("yolo_carac_detect.onnx", 0) != std::string::npos && yolo_carac_detect_model_file.length() == 0)
					{
						yolo_carac_detect_model_file = copie.string();
					}
					else if (filename.find("best.onnx", 0) != std::string::npos && yolo_carac_detect_model_file.length() == 0)
					{
						yolo_carac_detect_model_file = copie.string();
					}
				}
#ifdef LPR_MFC
				catch (CMemoryException e)
				{
					e.ReportError(MB_OK, IDS_STRING57604);
#ifdef LPR_LOGGING
					TCHAR   szCause[255];
					e.GetErrorMessage(szCause, 255);
					write_debugging_info(); writeLog(nsCLog::fatal, szCause);
#endif
					return false;
				}
#else	//LPR_MFC
				catch (std::exception & e)
				{
#ifdef LPR_ENGINE_IO
					std::cout << "std exception: " << e.what() << std::endl;
#endif // LPR_ENGINE_IO
					return false;
				}
#endif //LPR_MFC
			}
		}
		return (yolo_carac_detect_model_file.size());
	}
	return false;
}
//process all images files of a directory
float Yolov5_anpr_onxx_detector::detect(const std::string& dir, const bool show_image, const int time_delay)
{
#ifdef LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
	std::filesystem::path p(std::filesystem::current_path());
	//std::string model_path = p.parent_path().string();


	std::string filename = "D:\\Programmation\\LPReditor\\ocr_dataset\\test_svm.txt";
	//std::string filename = p.string();


	std::ofstream O(filename.c_str(), std::ios::app);
	O << "Yolov5_anpr_onxx_detector::detect " << std::endl;
	int c = 0;
#endif //LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
	std::list<std::string> image_filenames;
	//extracts from a test directory all images files that come with an xml file containing the bb coordinates in this image
	load_images_filenames(dir, image_filenames);
	std::list<std::string>::const_iterator it_image_filenames(image_filenames.begin());

	int less_1_editdistance_reads = 0;
	int miss_reads = 0;
	int good_reads = 0;
	while (it_image_filenames != image_filenames.end())
	{
		std::string lpn;
		detect(*it_image_filenames, lpn, show_image, time_delay);
		std::filesystem::path p_(*it_image_filenames);
		bool vrai_lpn_after_underscore = true;
		//returns the true license plate number out of a filename
			//you must place the true license plate number in the image filename this way : number + underscore + license plate number,
			//for instance filename 0000000001_3065WWA34.jpg will be interpreted as an image with the license plate 3065WWA34 in it.
		std::string ExactLPN(getTrueLPN(p_.stem().string(), vrai_lpn_after_underscore));
		std::cout << "ExactLPN : " << ExactLPN << " read LPN : " << lpn << std::endl;
		Levenshtein lev;
		int editdistance = lev.Get(ExactLPN.c_str(), ExactLPN.length(), lpn.c_str(), lpn.length());
		if (editdistance > 0) miss_reads++;
		else good_reads++;
		if (editdistance <= 1) less_1_editdistance_reads++;
		it_image_filenames++;
		c++;
		if ((c % 100) == 0) {
			std::cout << c << " perc good reads:" << (float)(good_reads) / (float)(c) << std::endl;
			std::cout << c << " perc reads less 1 edit distance:" << (float)(less_1_editdistance_reads) / (float)(c) << std::endl;
#ifdef LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
			O << c << " perc good reads:" << (float)(good_reads) / (float)(c) << std::endl;
			O << c << " perc reads less 1 edit distance:" << (float)(less_1_editdistance_reads) / (float)(c) << std::endl;
#endif //LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
		}
	}
	float good_reads_percentage = (float)(good_reads) / (float)(c);
	float less_1_editdistance_reads_percentage = (float)(less_1_editdistance_reads) / (float)(c);
#ifdef LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
	O << "good_reads : " << good_reads_percentage << "less_1_editdistance : " << less_1_editdistance_reads_percentage << std::endl;
	O.flush(); O.close();
#endif //LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
	return good_reads_percentage;
}
//process an image file
void Yolov5_anpr_onxx_detector::detect(const std::string& image_filename, std::string& lpn, const bool show_image, const int time_delay)
{
	std::filesystem::path p_(image_filename);
	if (exists(p_) && std::filesystem::is_regular_file(p_))    // does p actually exist?
	{
		int flags = -1;//as is
		cv::Mat frame = cv::imread(image_filename, flags);
		if (frame.rows > 0 && frame.cols > 0) {
			cv::Scalar mean_ = cv::mean(frame);
			if (mean_[0] > 2.0f && mean_[0] < 250.0f) {
				const float confThreshold = 0.7f;
				const float nmsThreshold = 0.5f;
				std::vector<int> classIds;
				std::vector<float> confidences;
				std::vector<cv::Rect> vect_of_detected_boxes;
				detect(frame, classIds, confidences, vect_of_detected_boxes, confThreshold, nmsThreshold);
				std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
				std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
				//the nnet has detected boxes that represant characters of the license plate, this function now etracts from these boxes the license plate number. 
				//it can deal with license pates that have two lines of charcaters
				lpn = get_lpn(vect_of_detected_boxes, confidences, classIds, tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold);
				if (show_image && time_delay >= 0) {
					cv::imshow(lpn, frame);
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
				}	cv::destroyAllWindows();
			}
		}
	}
}

/** @brief Given the @p input frame, create input blob, run net and return result detections.
	 *  @param[in]  frame : The input image.
	 *  @param[out] classIds Class indexes in result detection.
	 *  @param[out] confidences A set of corresponding confidences.
	 *  @param[out] boxes A set of bounding boxes.
	 *  @param[in] confThreshold A threshold used to filter boxes by confidences.
	 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
	 */
void Yolov5_anpr_onxx_detector::detect_lpn(const cv::Mat& frame, std::vector<int>& classIds,
	std::vector<float>& confidences, std::vector<cv::Rect>& boxes, float& confidence_lpn_roi, cv::Rect& lpn_roi,
	float confThreshold, float nmsThreshold)
{
	detect(frame, classIds,
		confidences, boxes,
		confThreshold, nmsThreshold);

	std::vector<int>::const_iterator it_out_classes(classIds.begin());
	std::vector<float>::const_iterator it_confidences(confidences.begin());
	std::vector<cv::Rect>::const_iterator it_boxes(boxes.begin());
	const int class_lpn = 35;//35 letters or digits in latin lpns
	confidence_lpn_roi = -1.0f;
	lpn_roi = cv::Rect(0, 0, 0, 0);
	while (it_out_classes != classIds.end() && it_confidences != confidences.end() && it_boxes != boxes.end()) {
		if (*it_out_classes == class_lpn) {
			if (confidence_lpn_roi < *it_confidences) {
				confidence_lpn_roi = *it_confidences;
				lpn_roi = *it_boxes;
			}
		}
		it_out_classes++;
		it_confidences++;
		it_boxes++;
	}

}
/** @brief Given the @p input frame, create input blob, run net and return result detections and from detections extract lpn. Difference between this func and detect func is that
	evaluate_lpn_with_lpn_detection first focus on license plate detection and secondly on its characters. This is supposed to get best results
		 *  @param[in]  frame  The input image.
		 *  @param[out] lpn : the license plate number found in the image by the dnn detector.
		 *  @param[out] classIds : classes indeces in resulting detection.
		 *  @param[out] confidences A set of corresponding confidences.
		 *  @param[out] boxes : A set of bounding boxes.
		 */
void Yolov5_anpr_onxx_detector::evaluate_lpn_with_lpn_detection(const cv::Mat& frame, std::string& lpn,
	 std::list<float>& confidences, std::list<int>& classes,std::list<cv::Rect>& boxes)
{
	if (frame.rows > 0 && frame.cols > 0) {
		//global image
		cv::Scalar mean_ = cv::mean(frame);
		if (mean_[0] > 2.0f && mean_[0] < 250.0f) {
			//float w_ratio = 1.0f;
			//float h_ratio = 1.0f;
			//const float confThreshold_lpn = 0.2f;
			//const float nmsThreshold_lpn = 0.15f;
			std::vector<int> classIds_lpn;
			std::vector<float> confidences_lpn;
			std::vector<cv::Rect> vect_of_detected_boxes_lpn;

			/** @brief Given the @p input frame, create input blob, run net and return result detections.
				 *  @param[in]  frame : The input image.
				 *  @param[out] classIds Class indexes in result detection.
				 *  @param[out] confidences A set of corresponding confidences.
				 *  @param[out] boxes A set of bounding boxes.
				 *  @param[in] confThreshold A threshold used to filter boxes by confidences.
				 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
				 */
			float confidence_lpn_roi;
			cv::Rect lpn_roi;
			const float confThreshold = 0.5f; float nmsThreshold = 0.5f;
			detect_lpn(frame, classIds_lpn, confidences_lpn, vect_of_detected_boxes_lpn, confidence_lpn_roi, lpn_roi,
				confThreshold , nmsThreshold );
			if (lpn_roi.width > 0 && lpn_roi.height > 0 && lpn_roi.x >= 0 && lpn_roi.y >= 0 &&
				confidence_lpn_roi > 0.0f && lpn_roi.x + lpn_roi.width <= frame.cols && lpn_roi.y <= frame.rows) {
				int w = lpn_roi.width;
				int h = lpn_roi.height;
				lpn_roi.x -= w ;
				lpn_roi.width += 2.0f* (float)w;
				lpn_roi.y -= 1.5f*(float)h ;
				lpn_roi.height += 3.0f*(float)h;
				lpn_roi = get_inter(lpn_roi, cv::Rect(0, 0, frame.cols, frame.rows));
				const int class_lpn = 35;//35 letters or digits in latin lpns	
				if (lpn_roi.area() < ((frame.rows * frame.cols) / 10)) {
					//image of lpn
					cv::Mat subimage_ = frame(lpn_roi);
					std::vector<int> classIds;
					std::vector<float> vect_of_confidences;
					std::vector<cv::Rect> vect_of_detected_boxes;
					/** @brief Given the @p input frame, create input blob, run net and return result detections.
						 *  @param[in]  frame : The input image.
						 *  @param[out] classIds Class indexes in result detection.
						 *  @param[out] confidences A set of corresponding confidences.
						 *  @param[out] boxes A set of bounding boxes.
						 *  @param[in] confThreshold A threshold used to filter boxes by confidences.
						 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
						 */
					detect(subimage_, classIds, vect_of_confidences, vect_of_detected_boxes,confThreshold, nmsThreshold);
					//now filter boxes with big iou
	//if two boxes have an iou (intersection over union) that is two large, then they cannot represent two adjacent characters of the license plate 
	//so we discard the one with the lowest confidence rate
					filter_iou(classIds,vect_of_confidences,vect_of_detected_boxes);
					if (vect_of_detected_boxes.size() > 2) {
						std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
						std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
						//the nnet has detected boxes that represant characters of the license plate, this function now etracts from these boxes the license plate number. 
						//it can deal with license pates that have two lines of charcaters
						lpn = get_lpn(vect_of_detected_boxes, vect_of_confidences, classIds, tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold);
						std::vector<float>::iterator it_confidences(tri_left_confidences.begin());
						std::vector<cv::Rect>::iterator it_boxes(tri_left_vect_of_detected_boxes.begin());
						std::vector<int>::iterator it_out_classes_(tri_left_classIds.begin());
						while (it_out_classes_ != tri_left_classIds.end()
							&& it_confidences != tri_left_confidences.end() && it_boxes != tri_left_vect_of_detected_boxes.end()) {
							if ((*it_out_classes_ < class_lpn) || (lpn.empty())) {
								//now we must change the coordinates of the box to fit global image
								cv::Rect box_in_global_image(it_boxes->x + lpn_roi.x, it_boxes->y + lpn_roi.y, it_boxes->width, it_boxes->height);
								boxes.push_back(box_in_global_image);
								classes.push_back(*it_out_classes_);
								confidences.push_back(*it_confidences);
							}
							it_out_classes_++;
							it_confidences++;
							it_boxes++;
						}
					}
					else lpn = std::string();
				}
				else {
					//the nnet has detected boxes that represant characters of the license plate, this function now etracts from these boxes the license plate number. 
					//it can deal with license pates that have two lines of charcaters
					std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
					std::vector<float> tri_left_confidences;  std::vector<int> tri_left_classIds;
					const float nmsThreshold_lpn = 0.5f;





					lpn = get_lpn(vect_of_detected_boxes_lpn, confidences_lpn, classIds_lpn, tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds, nmsThreshold_lpn);
					std::vector<float>::iterator it_confidences(tri_left_confidences.begin());
					std::vector<cv::Rect>::iterator it_boxes(tri_left_vect_of_detected_boxes.begin());
					std::vector<int>::iterator it_out_classes_(tri_left_classIds.begin());
					while (it_out_classes_ != tri_left_classIds.end()
						&& it_confidences != tri_left_confidences.end() && it_boxes != tri_left_vect_of_detected_boxes.end()) {
						if ((*it_out_classes_ < class_lpn) || (lpn.empty())) {
							//now we must change the coordinates of the box to fit global image
							cv::Rect box_in_global_image(it_boxes->x, it_boxes->y, it_boxes->width, it_boxes->height);
							boxes.push_back(box_in_global_image);
							classes.push_back(*it_out_classes_);
							confidences.push_back(*it_confidences);
						}
						it_out_classes_++;
						it_confidences++;
						it_boxes++;
					}
				}
			}
			else lpn = std::string();
		}
	}
}
/** @brief Given the @p input frame, create input blob, run net and return result detections and from detections extract lpn. Difference between this func and detect func is that
	evaluate_lpn_with_lpn_detection first focus on license plate detection and secondly on its characters. This is supposed to get best results
		 *  @param[in]  frame  The input image.
		 *  @param[out] lpn : the license plate number found in the image by the dnn detector.
		 */
void Yolov5_anpr_onxx_detector::evaluate_lpn_with_lpn_detection(const cv::Mat& frame, std::string& lpn)
{
	std::list<cv::Rect> boxes; std::list<float> confidences; std::list<int> classes;
	evaluate_lpn_with_lpn_detection(frame, lpn, confidences, classes, boxes);
}
