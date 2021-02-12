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
	/** @brief Given the @p input frame, create input blob, run net and return result detections.
	 *  @param[in]  frame  The input image.
	 *  @param[out] classIds : classes indeces in resulting detection.
	 *  @param[out] confidences A set of corresponding confidences.
	 *  @param[out] boxes A set of bounding boxes.
	 *  @param[in] confThreshold A threshold used to filter boxes by confidences.
	 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
	 */
	void detect(const cv::Mat& frame, std::vector<int>& classIds, std::vector<float>& confidences, std::vector<cv::Rect>& boxes,
		const float confThreshold = 0.7f, float nmsThreshold = 0.5f);
	/** @brief Given the @p input frame, create input blob, run net and return result detections.
		 *  @param[in]  frame  The input image.
		 *  @param[out] lpn : the license plate number
		 *  @param[in] confThreshold A threshold used to filter boxes by confidences.
		 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
		 */
	void detect(const cv::Mat& frame, std::string& lpn);
		/** @brief Given the @p input frame, create input blob, run net and return result detections.
		 *  @param[in]  frame  The input image.
		 *  @param[out] classIds : classes indeces in resulting detection.
		 *  @param[out] confidences A set of corresponding confidences.
		 *  @param[out] boxes A set of bounding boxes.
		 *  @param[in] confThreshold A threshold used to filter boxes by confidences.
		 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
		 */
	void detect(const cv::Mat& frame, std::list<int>& classIds, std::list<float>& confidences, std::list<cv::Rect>& boxes,
		const float confThreshold = 0.7f, float nmsThreshold = 0.5f);
	/** @brief  extract license plate number of all images files of a directory.
		 *  @param[in]  dir : directory path that contains all images files that will be proceeded.
	 *  @param[in]  show_image : boolean if true the image will be displayed in a window with license plate in image banner
	 *  @param[in]  time_delay : time delay in ms after which the image is destroyed
		* @return void
			* @see
		 */
	float detect(const std::string& dir, const bool show_image = false, const int time_delay = 0);
	/** @brief process an image file.
	 *  @param[in]  filename : filename of the he input image.
	 *  @param[out] lpn : the license plate number found in the image by the dnn detector.
	 *  @param[in]  show_image : boolean if true the image will be displayed in a window with license plate in image banner
	 *  @param[in]  time_delay : time delay in ms after which the image is destroyed
	* @return void
		* @see
	 */
	void detect(const std::string& image_filename, std::string& lpn, const bool show_image = false, const int time_delay = 0);
};
/** @brief given a directory, finds into it the model file.
	@param dir_path : dir path of the directory containing necessary files
	@param yolo_carac_detect_model_file : file path of the model
	 */
bool load_onnx(const std::string& dir_path, std::string& model_file);
#endif // !defined(YOLOV5_ANPR_ONNX_DETECTOR)