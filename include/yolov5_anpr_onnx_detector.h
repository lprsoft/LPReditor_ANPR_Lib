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
#if !defined(YOLOV5_ANPR_ONNX_DETECTOR)
#define YOLOV5_ANPR_ONNX_DETECTOR
#include "ONNX_detector.h"
class Plates_types_classifier : public OnnxDetector
{
public:
	//**************************
  //   construct/destruct
  //**************************
	Plates_types_classifier(Ort::Env& env, const void* model_data, size_t model_data_length, const Ort::SessionOptions& options
		, const std::vector<std::string>& labels_);
	Plates_types_classifier(Ort::Env& env, const ORTCHAR_T* model_path, const Ort::SessionOptions& options
		, const std::vector<std::string>& labels_);
	Plates_types_classifier(Ort::Env& env, const ORTCHAR_T* model_path, const Ort::SessionOptions& options
		, const std::string& labels_filename);
	Plates_types_classifier(Ort::Env& env, const void* model_data, size_t model_data_length, const Ort::SessionOptions& options
		, const std::string& labels_filename);
	OnnxDetector& get_ref() {
		return *this;
	};
	const Plates_types_classifier& get_const_ref() {
		return *this;
	};
	virtual ~Plates_types_classifier();
	std::string GetPlatesType(const cv::Mat& img, float& uncalibrated_confidence);
	void get_best_plate(const cv::Mat& frame,
		//detections when they are separated license plates by license plates
		const std::list < std::list<int>>& classIds, const std::list < std::list<float>>& confidences, const std::list < std::list<cv::Rect>>& boxes
		//output the list of the best (most probable/readable) lp
		, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp);
	//nov 21 update this func with plates_types_classifier classifier
	//uses double linked lists : inside list is for characters and outside list is for plates.
	//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
	void get_best_plate(const cv::Mat& frame,
		//detections when they are separated license plates by license plates
		const std::list < std::vector<int>>& classIds, const std::list < std::vector<float>>& confidences, const std::list < std::vector<cv::Rect>>& boxes
		//output the list of the best (most probable/readable) lp
		, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp);
private:
	int GetPlatesClasse(const cv::Mat& img, float& uncalibrated_confidence);
protected:
	std::vector<std::string> labels;
};
class Yolov5_anpr_onxx_detector : public OnnxDetector
{
public:
	//**************************
  //   construct/destruct
  //**************************
	Yolov5_anpr_onxx_detector(Ort::Env& env, const void* model_data, size_t model_data_length, const Ort::SessionOptions& options);
	Yolov5_anpr_onxx_detector(Ort::Env& env, const ORTCHAR_T* model_path, const Ort::SessionOptions& options);
	OnnxDetector& get_ref() {
		return *this;
	};
	const OnnxDetector& get_const_ref() {
		return *this;
	};
	virtual ~Yolov5_anpr_onxx_detector();
	/** @brief Given the @p input frame, create input blob, run net then, from result detections, assembies license plates present in the input image.
	* uses double linked lists : inside list is for characters and outside list is for plates.
	 *  @param[in]  frame : input image.
	 *  @param[out] classIds : classes indeces in resulting detected bounding boxes.
	 *  @param[out] confidences : detection confidences of detected bounding boxes
	 *  @param[out] boxes : set of detected bounding boxes.
	 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
	 */
	void detect_with_different_confidences_then_separate_plates(const cv::Mat& frame, std::list<std::vector<int>>& classIds,
		std::list < std::vector<float>>& confidences, std::list < std::vector<cv::Rect>>& boxes,
		float nmsThreshold);
	/** @brief Given the @p input frame, create input blob, run net and return result detections.
	 *  @param[in]  frame : input image.
	 *  @param[out] classIds : classes indeces in resulting detected bounding boxes.
	 *  @param[out] confidences : detection confidences of detected bounding boxes
	 *  @param[out] boxes : set of detected bounding boxes.
	 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
	 */
	void raw_detections_with_different_confidences(const cv::Mat& frame, std::list<std::list<int>>& classIds,
		std::list < std::list<float>>& confidences, std::list < std::list<cv::Rect>>& boxes,
		float nmsThreshold);
	/** @brief Given the @p input frame, create input blob, run net then, from result detections, assembies license plates present in the input image.
	* uses double linked lists : inside list is for characters and outside list is for plates.
	 *  @param[in]  frame : input image.
	 *  @param[out] classIds : classes indeces in resulting detected bounding boxes.
	 *  @param[out] confidences : detection confidences of detected bounding boxes
	 *  @param[out] boxes : set of detected bounding boxes.
	 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
	 @param[in] classId_last_country : is the class index of the last country in the list of detected classes.
	 */
	void detect_with_different_confidences_then_separate_plates(const cv::Mat& frame, std::list<std::vector<int>>& classIds,
		std::list < std::vector<float>>& confidences, std::list < std::vector<cv::Rect>>& boxes, std::list <std::list<std::string>>& lpns,
		float nmsThreshold, const int classId_last_country//classId_last_country : is the class index of the last country in the list of detected classes.
	);
	/** @brief Given the @p input frame, create input blob, run net then, from result detections, assembies license plates present in the input image.
	*  @param[in]  frame : input image.
 *  @param[out] classIds : classes indeces in resulting detected bounding boxes.
 *  @param[out] confidences : detection confidences of detected bounding boxes
 *  @param[out] boxes : set of detected bounding boxes.
 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
 @param[in] classId_last_country : is the class index of the last country in the list of detected classes.
 */
	void detect_with_different_confidences_then_separate_plates(const cv::Mat& frame, std::list<std::list<int>>& classIds,
		std::list < std::list<float>>& confidences, std::list < std::list<cv::Rect>>& boxes,
		std::list <std::list<std::string>>& lpns,
		float nmsThreshold, const int classId_last_country//classId_last_country : is the class index of the last country in the list of detected classes.
	);
	/** @brief Given the @p input frame, create input blob, run net and return result detections.
	 *  @param[in]  frame : input image.
	 *  @param[out] classIds : classes indeces in resulting detected bounding boxes.
	 *  @param[out] confidences : detection confidences of detected bounding boxes
	 *  @param[out] boxes : set of detected bounding boxes.
	 *  @param[in] confThreshold A threshold used to filter boxes by confidences.
	 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
	 */
	void detect(const cv::Mat& frame, std::vector<int>& classIds, std::vector<float>& confidences, std::vector<cv::Rect>& boxes,
		const float confThreshold = 0.7f, float nmsThreshold = 0.5f);
	// Given the @p input frame, create input blob, run net and return result detections.
	//this func can manage list of boxes of characters that dont have an englobing lp box (gloabal rect)
	//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
	//Produces double linked lists : inside list is for characters and outside list is for plates.
	/** @brief
	*  @param[in]  frame : input image.
	 *  @param[out] classIds : classes indeces in resulting detected bounding boxes.
	 *  @param[out] confidences : detection confidences of detected bounding boxes
	 *  @param[out] boxes : set of detected bounding boxes.
	 @param[in] classId_last_country : is the class index of the last country in the list of detected classes.
	 *  @param[in] confThreshold A threshold used to filter boxes by confidences.
	 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
	 */
	 //and remaining elements are characters
	void detect_and_add_lp_and_vehicle_if_necessary(const cv::Mat& frame, std::list<std::vector<int>>& classIds,
		std::list < std::vector<float>>& confidences, std::list < std::vector<cv::Rect>>& boxes,
		std::list<std::string>& lpns,
		const int classId_last_country,
		const float confThreshold = 0.7f, float nmsThreshold = 0.5f);
	// Given the @p input frame, create input blob, run net and return result detections.
	//this func can manage list of boxes of characters that dont have an englobing lp box (gloabal rect)
	//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
	/** @brief
	*  @param[in]  frame : input image.
	 *  @param[out] classIds : classes indeces in resulting detected bounding boxes.
	 *  @param[out] confidences : detection confidences of detected bounding boxes
	 *  @param[out] boxes : set of detected bounding boxes.
	 @param[in] classId_last_country : is the class index of the last country in the list of detected classes.
	 *  @param[in] confThreshold A threshold used to filter boxes by confidences.
	 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
	 */
	void detect_and_add_lp_and_vehicle_if_necessary(const cv::Mat& frame, std::list<std::list<int>>& classIds,
		std::list < std::list<float>>& confidences, std::list < std::list<cv::Rect>>& boxes,
		std::list<std::string>& lpns,
		const int classId_last_country,
		const float confThreshold = 0.7f, float nmsThreshold = 0.5f);
	void two_stage_lpr(Yolov5_anpr_onxx_detector& parking_detector, const cv::Mat& frame, std::string& lpn);
	void two_stage_lpr(Yolov5_anpr_onxx_detector& parking_detector, Plates_types_classifier& plates_types_classifier, const cv::Mat& frame, std::string& lpn);
	/** @brief Given the @p input frame, create input blob, run net and return result detections.
		 *  @param[in]  frame : input image.
		 *  @param[out] classIds : classes indeces in resulting detected bounding boxes.
		 *  @param[out] confidences : detection confidences of detected bounding boxes
		 *  @param[out] boxes : set of detected bounding boxes.
		 *  @param[in] confThreshold A threshold used to filter boxes by confidences.
		 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
		 */
	void detect(const cv::Mat& frame, std::list<int>& classIds, std::list<float>& confidences, std::list<cv::Rect>& boxes,
		const float confThreshold = 0.7f, float nmsThreshold = 0.5f);
	/** @brief process an image file.
	 *  @param[in]  image_filename : filename of the he input image.
	 *  @param[out] lpn : the license plate number found in the image by the dnn detector.
	 *  @param[in]  show_image : boolean if true the image will be displayed in a window with license plate in image banner
	 *  @param[in]  time_delay : time delay in ms after which the image is destroyed
	@param[in] classId_last_country : is the class index of the last country in the list of detected classes.
	* @return void
		* @see
	 */
	void detect(const std::string& image_filename, std::list<std::string>& lpns, const int classId_last_country
		//, const bool show_image=false, const int time_delay=0
	);
	/** @brief from image dir, extract license plate numbers.
	this performs a two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
	Difference between this func and detect func is that
	two_stage_lpr first focus on license plate detection and secondly on its characters. This is supposed to get best results.
		 *  @param[in]  dir : directory path that contains all images files that will be proceeded.
		* @return void
			* @see
		 */
	float two_stage_lpr(const std::string& dir);
	/** @brief from image dir, extract license plate numbers.
	this performs a two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
	Difference between this func and detect func is that
	two_stage_lpr first focus on license plate detection and secondly on its characters. This is supposed to get best results.
		 *  @param[in]  dir : directory path that contains all images files that will be proceeded.
		* @return void
			* @see
		 */
	float two_stage_lpr(Yolov5_anpr_onxx_detector& parking_detector, const std::string& dir);
	/** @brief from image dir, extract license plate numbers.
	this performs a two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
	Difference between this func and detect func is that
	two_stage_lpr first focus on license plate detection and secondly on its characters. This is supposed to get best results.
		 *  @param[in]  frame : input image.
		 *  @param[out] lpn : the license plate number found in the image by the dnn detector.
		 @param[in] classes : set of indeces that indicate the classes of each of these detected boxes
		 *  @param[out] confidences : detection confidences of detected bounding boxes
		 *  @param[out] boxes : A set of bounding boxes.
		 *  @param[out] chosen_lp_classIds : set of indeces that indicate the classes of each box of the license plate that has been chosen by engine (ie the one with the highest confidence)
		 *  @param[out] chosen_lp_confidences : detection confidences of the corresponding boxes
		 *  @param[out] chosen_lp_boxes : A set of bounding boxes of the license plate that has been chosen by engine (ie the one with the highest confidence)
		 */
	void two_stage_lpr(Yolov5_anpr_onxx_detector& parking_detector, const cv::Mat& frame,
		//double linked lists to separate lps
		std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
		//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
		std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
		//detection inside the chosen lp
		std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
	);
	/** @brief from image dir, extract license plate numbers.
	this performs a two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
	Difference between this func and detect func is that
	two_stage_lpr first focus on license plate detection and secondly on its characters. This is supposed to get best results.
		 *  @param[in]  frame : input image.
		 *  @param[out] lpn : the license plate number found in the image by the dnn detector.
		 @param[in] classes : set of indeces that indicate the classes of each of these detected boxes
		 *  @param[out] confidences : detection confidences of detected bounding boxes
		 *  @param[out] boxes : A set of bounding boxes.
		 *  @param[out] chosen_lp_classIds : set of indeces that indicate the classes of each box of the license plate that has been chosen by engine (ie the one with the highest confidence)
		 *  @param[out] chosen_lp_confidences : detection confidences of the corresponding boxes
		 *  @param[out] chosen_lp_boxes : A set of bounding boxes of the license plate that has been chosen by engine (ie the one with the highest confidence)
		 */
		void two_stage_lpr(Yolov5_anpr_onxx_detector& parking_detector, Plates_types_classifier& plates_types_classifier, const cv::Mat& frame,
		//double linked lists to separate lps
		std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
		//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
		std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
		//detection inside the chosen lp
		std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
	);
	/** @brief from image frame, extract license plate number.
	this performs a two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
	Difference between this func and detect func is that
	two_stage_lpr first focus on license plate detection and secondly on its characters. This is supposed to get best results.
			 *  @param[in]  image_filename : filename of the he input image.
			 *  @param[out] lpn : the license plate number found in the image by the dnn detector.
			 @param[in] classes : set of indeces that indicate the classes of each of these detected boxes
			 *  @param[out] confidences : detection confidences of detected bounding boxes
			 *  @param[out] boxes : A set of bounding boxes.
		 *  @param[out] chosen_lp_classIds : set of indeces that indicate the classes of each box of the license plate that has been chosen by engine (ie the one with the highest confidence)
		 *  @param[out] chosen_lp_confidences : detection confidences of the corresponding boxes
		 *  @param[out] chosen_lp_boxes : A set of bounding boxes of the license plate that has been chosen by engine (ie the one with the highest confidence)
			 */
	void two_stage_lpr(Yolov5_anpr_onxx_detector& parking_detector, const std::string& image_filename,
		//double linked lists to separate lps
		std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
		//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
		std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
		//detection inside the chosen lp
		std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
	);
	/** @brief from image filename, extract license plate number.
	this performs a two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
	Difference between this func and detect func is that
	two_stage_lpr first focus on license plate detection and secondly on its characters. This is supposed to get best results.
	*  @param[in]  image_filename : filename of the he input image.
			 *  @param[out] lpn : the license plate number found in the image by the dnn detector.
			 @param[in] classes : set of indeces that indicate the classes of each of these detected boxes
			 *  @param[out] confidences : detection confidences of detected bounding boxes
			 *  @param[out] boxes : A set of bounding boxes.
		 *  @param[out] chosen_lp_classIds : set of indeces that indicate the classes of each box of the license plate that has been chosen by engine (ie the one with the highest confidence)
		 *  @param[out] chosen_lp_confidences : detection confidences of the corresponding boxes
		 *  @param[out] chosen_lp_boxes : A set of bounding boxes of the license plate that has been chosen by engine (ie the one with the highest confidence)
			 */
	void two_stage_lpr(const std::string& image_filename,
		//double linked lists to separate lps
		std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
		//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
		std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
		//detection inside the chosen lp
		std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
	);
	/** @brief Given the @p input frame, create input blob, run net and return result detections.
	//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters. 
//Produces double linked lists : inside list is for characters and outside list is for plates.
	 *  @param[in]  frame : The input image.
	 *  @param[out] classIds Class indexes in result detection.
	 *  @param[out] confidences : detection confidences of detected bounding boxes
	 *  @param[out] boxes : set of detected bounding boxes.
@param[out] one_lp :set of detected boxes when they rearranged from left to right
			@param[out] confidence_one_lp : confidences corresponding detected boxes
			@param[out] classIds_one_lp : set of indeces of the above detected boxes
	 @param[in] classId_last_country : is the class index of the last country in the list of detected classes.
	 *  @param[in] confThreshold A threshold used to filter boxes by confidences.
	 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
	 */
	void detect_lpn_and_add_lp_and_vehicle_if_necessary(const cv::Mat& frame, std::list < std::vector<int>>& classIds,
		std::list < std::vector<float>>& confidences, std::list < std::vector<cv::Rect>>& boxes
		, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp,
		const int classId_last_country,
		//const C_OCROutputs& availableAlpha,
		const float confThreshold = 0.7f, float nmsThreshold = 0.5f);
	/** @brief Given the @p input frame, create input blob, run net then, from result detections, assembies license plates present in the input image.
						//it selects just one lpn although all lps have been detected and stored in double linked lists, then from these lists, selects the one that is the best
//(with best confidences of its characters and with greateast size)
				//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
		 *  @param[in]  frame : The input image.
		 *  @param[out] classIds Class indexes in result detection.
		 *  @param[out] confidences : detection confidences of detected bounding boxes
		 *  @param[out] boxes : set of detected bounding boxes.
@param[out] one_lp :set of detected boxes when they rearranged from left to right
			@param[out] confidence_one_lp : confidences corresponding detected boxes
			@param[out] classIds_one_lp : set of indeces of the above detected boxes
		 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
		 @param[in] classId_last_country : is the class index of the last country in the list of detected classes.
		 */
	void detect_with_different_confidences_then_separate_plates(const cv::Mat& frame, std::list < std::list<int>>& classIds,
		std::list < std::list<float>>& confidences, std::list < std::list<cv::Rect>>& boxes
		, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp,
		const int classId_last_country,
		//const C_OCROutputs& availableAlpha,
		float nmsThreshold);
	//
	//Given the @p input frame, create input blob, run net and return result detections.
	//it selects just one lpn although all lps are detected in double linked lists
	////this func can manage list of boxes of characters that dont have an englobing lp box (gloabal rect)
	////output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
	////and remaining elements are characters
	//Produces double linked lists : inside list is for characters and outside list is for plates.
	//			  
	void detect_lpn_and_add_lp_and_vehicle_if_necessary(const cv::Mat& frame, Plates_types_classifier& plates_types_classifier, std::list < std::vector<int>>& classIds,
		std::list < std::vector<float>>& confidences, std::list < std::vector<cv::Rect>>& boxes
		, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp,
		const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.We remember that ascii(latin) characters come fist(36 classes) then come the license plates countries(another 60 classses) then come a long list of vehicles classes
		//const C_OCROutputs & availableAlpha,
		const float confThreshold, float nmsThreshold);
	//Given the @p input frame, create input blob, run net then, from result detections, assembly license plates present in the input image.
	void detect_with_different_confidences_then_separate_plates(const cv::Mat& frame, Plates_types_classifier& plates_types_classifier, std::list < std::list<int>>& classIds,
		std::list < std::list<float>>& confidences, std::list < std::list<cv::Rect>>& boxes
		, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp,
		const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.We remember that ascii(latin) characters come fist(36 classes) then come the license plates countries(another 60 classses) then come a long list of vehicles classes
		//const C_OCROutputs & availableAlpha,
		float nmsThreshold);
	/** @brief Given the @p input frame, create input blob, run net then, from result detections, assembies license plates present in the input image.
						//it selects just one lpn although all lps have been detected and stored in double linked lists, then from these lists, selects the one that is the best
//(with best confidences of its characters and with greateast size)
				//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
		 *  @param[in]  frame : The input image.
		 @param[in] ExactLPN : the actual license plate number in the image
		 *  @param[out] classIds Class indexes in result detection.
		 *  @param[out] confidences : detection confidences of detected bounding boxes
		 *  @param[out] boxes : set of detected bounding boxes.
@param[out] one_lp :set of detected boxes when they rearranged from left to right
			@param[out] confidence_one_lp : confidences corresponding detected boxes
			@param[out] classIds_one_lp : set of indeces of the above detected boxes
		 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
		 @param[in] classId_last_country : is the class index of the last country in the list of detected classes.
		 */
	void detect_with_different_confidences_then_separate_plates(const cv::Mat& frame, const std::string& ExactLPN, std::list < std::list<int>>& classIds,
		std::list < std::list<float>>& confidences, std::list < std::list<cv::Rect>>& boxes
		, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp,
		const int classId_last_country,
		//const C_OCROutputs& availableAlpha,
		float nmsThreshold);
	/** @brief Given the @p input frame, create input blob, run net then, from result detections, assembies license plates present in the input image.
						//it selects just one lpn although all lps have been detected and stored in double linked lists, then from these lists, selects the one that is the best
//(with best confidences of its characters and with greateast size)
				//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
		 *  @param[in]  frame : The input image.
		 @param[in] ExactLPN : the actual license plate number in the image
		 *  @param[out] classIds Class indexes in result detection.
		 *  @param[out] confidences : detection confidences of detected bounding boxes
		 *  @param[out] boxes : set of detected bounding boxes.
@param[out] one_lp :set of detected boxes when they rearranged from left to right
			@param[out] confidence_one_lp : confidences corresponding detected boxes
			@param[out] classIds_one_lp : set of indeces of the above detected boxes
		 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
		 @param[in] classId_last_country : is the class index of the last country in the list of detected classes.
		 */
	void detect_with_different_confidences_then_separate_plates(const cv::Mat& frame, const std::string& ExactLPN, std::list < std::list<int>>& classIds,
		std::list < std::list<float>>& confidences, std::list < std::list<cv::Rect>>& boxes
		, std::vector<float>& confidence_one_lp, std::vector < cv::Rect>& one_lp, std::vector<int>& classIds_one_lp,
		const int classId_last_country,
		//const C_OCROutputs& availableAlpha,
		float nmsThreshold);
};
/** @brief from image filename, extract license plate number.
this performs a two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
Difference between this func and detect func is that
two_stage_lpr first focus on license plate detection and secondly on its characters. This is supposed to get best results.
		 *  @param[in]  freeflow_detectors : yolo detectors that detect license plates in images that are not focused and that can contain multiple license plates.
		*  @param[in]  parking_detectors : yolo detectors that read characters in a localized and focused license plate in image
	  *  @param[in]  image_filename : filename of the he input image.
		 *  @param[out] lpn : the license plate number found in the image by the dnn detector.
		 @param[in] classes : set of indeces that indicate the classes of each of these detected boxes
		 *  @param[out] confidences : detection confidences of detected bounding boxes
		 *  @param[out] boxes : A set of bounding boxes.
	 *  @param[out] chosen_lp_classIds : set of indeces that indicate the classes of each box of the license plate that has been chosen by engine (ie the one with the highest confidence)
	 *  @param[out] chosen_lp_confidences : detection confidences of the corresponding boxes
	 *  @param[out] chosen_lp_boxes : A set of bounding boxes of the license plate that has been chosen by engine (ie the one with the highest confidence)
		 */
void two_stage_lpr(const std::list<Yolov5_anpr_onxx_detector*>& freeflow_detectors, const std::list<Yolov5_anpr_onxx_detector*>& parking_detectors,
	const std::string& image_filename,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
);
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
);
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
);
//make a bounding box greater (just to be sure we don t miss something in detections)
void get_larger_roi(cv::Rect& lpn_roi, const int width, const int height
);
//make a bounding box greater (just to be sure we don t miss something in detections)
void get_larger_roi(cv::Rect& lpn_roi, const int width, const int height, const float& scale_x, const float& scale_y
);
Yolov5_anpr_onxx_detector* get_detector_with_smallest_size_bigger_than_image(const std::list<Yolov5_anpr_onxx_detector*>& detectors, const int max_size);
Yolov5_anpr_onxx_detector* get_detector_with_smallest_size_bigger_than_image(const std::list<Yolov5_anpr_onxx_detector*>& detectors, const int width, const int height);
/** @brief from image frame, extract license plate number.
this performs a two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
Difference between this func and detect func is that
two_stage_lpr first focus on license plate detection and secondly on its characters. This is supposed to get best results.
		 *  @param[in]  freeflow_detectors : yolo detectors that detect license plates in images that are not focused and that can contain multiple license plates.
		*  @param[in]  parking_detectors : yolo detectors that read characters in a localized and focused license plate in image
	  *  @param[in]  frame : input image.
		 *  @param[out] lpn : the license plate number found in the image by the dnn detector.
		 @param[in] classes : set of indeces that indicate the classes of each of these detected boxes
		 *  @param[out] confidences : detection confidences of detected bounding boxes
		 *  @param[out] boxes : A set of bounding boxes.
	 *  @param[out] chosen_lp_classIds : set of indeces that indicate the classes of each box of the license plate that has been chosen by engine (ie the one with the highest confidence)
	 *  @param[out] chosen_lp_confidences : detection confidences of the corresponding boxes
	 *  @param[out] chosen_lp_boxes : A set of bounding boxes of the license plate that has been chosen by engine (ie the one with the highest confidence)
		 */
void two_stage_lpr(const std::list<Yolov5_anpr_onxx_detector*>& freeflow_detectors, const std::list<Yolov5_anpr_onxx_detector*>& parking_detectors,
	const cv::Mat& frame,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
);
/** @brief from image filename, extract license plate number.
this performs a two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
Difference between this func and detect func is that
two_stage_lpr first focus on license plate detection and secondly on its characters. This is supposed to get best results.
	 *  @param[in]  image_filename : filename of the he input image.
	 *  @param[out] lpn : the license plate number found in the image by the dnn detector.
	 @param[in] classes : set of indeces that indicate the classes of each of these detected boxes
	 *  @param[out] confidences : detection confidences of detected bounding boxes
	 *  @param[out] boxes : A set of bounding boxes.
	 *  @param[out] chosen_lp_classIds : set of indeces that indicate the classes of each box of the license plate that has been chosen by engine (ie the one with the highest confidence)
	 *  @param[out] chosen_lp_confidences : detection confidences of the corresponding boxes
	 *  @param[out] chosen_lp_boxes : A set of bounding boxes of the license plate that has been chosen by engine (ie the one with the highest confidence)
	 */
void two_stage_lpr(const std::list<Yolov5_anpr_onxx_detector*>& detectors, const std::string& image_filename,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
);
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
);
void two_stage_lpr(Yolov5_anpr_onxx_detector& freeflow_detector, Yolov5_anpr_onxx_detector& parking_detector
	, Plates_types_classifier& plates_types_classifier,
	const std::string& image_filename,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
);
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
);
//two stage lpn detection : first a global nn detects lpn of a free flow vehicle, then a second nn focuses and reads the lpn of the previously detected lpn.
void two_stage_lpr(Yolov5_anpr_onxx_detector& freeflow_detector, Yolov5_anpr_onxx_detector& parking_detector
	, Plates_types_classifier& plates_types_classifier,
	const cv::Mat& frame,
	//double linked lists to separate lps
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, std::list < std::list<cv::Rect>>& boxes,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, std::list <int>& lp_country_class, std::list < cv::Rect>& lp_rois,
	//detection inside the chosen lp
	std::list<int>& chosen_lp_classIds, std::list<float>& chosen_lp_confidences, std::list<cv::Rect>& chosen_lp_boxes
);
#endif // !defined(YOLOV5_ANPR_ONNX_DETECTOR)