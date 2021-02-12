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


#if !defined(UTILS_ANPR_ONNX_H)
#define UTILS_ANPR_ONNX_H

#pragma once


#include <opencv2/core.hpp>
//#include "Levenshtein.h"
#include "Line.h"

enum Det {
    tl_x = 0,
    tl_y = 1,
    br_x = 2,
    br_y = 3,
    score = 4,
    class_idx = 5
};

struct Detection {
    cv::Rect bbox;
    float score;
    int class_idx;
};

/**
		@brief
//rearrange detected bounding boxes from left to right
			@param[out] boxes : std list of detected boxes
			@param[out] confidences : confidences detected boxes
			@param[out] classIds : std::list of indeces that indicate the classes of each of the above detected boxes
			@return void
			@see
			*/
void sort_from_left_to_right(std::list<cv::Rect>& boxes, std::list<float>& confidences, std::list<int>& classIds);
/**
	@brief
	@param r1 : first rectangle
	@param r2 : second rectangle
	@return the intersection rect of the rectangles
	@see
	*/
cv::Rect get_inter(const cv::Rect& r1, const cv::Rect& r2);
/**
	@brief
	//return true if the intersection of the first argument box and the second has an interect area that is at least 90% of the box (which means box is nearly entirely in the second argument)
	@param box : box a bounding box
	@param rect_im : ROi or second bounding box
	@return true if intersection is at least 90% of the box (which means box is nearly entirely in the second argument)
	@see
	*/
bool is_in_rect_if(const cv::Rect& box, const cv::Rect& rect_im);
/**
		@brief
///from all heights of the boxes, get the median value
			@param[in] boxes :std vect of detected boxes
			@return the median height of the boxes
			@see
			*/
int get_median_height(const std::list<cv::Rect>& boxes);
/**
	@brief
	//returns the iou (intersection over union) of two boxes
	@param r1 : first rectangle
	@param r2 : second rectangle
	@return the iou (a float value between 0 and 1)
	@see
	*/
float iou(const cv::Rect& r1, const cv::Rect& r2);
/**
		@brief
//if two boxes have an iou (intersection over union) that is two large, then they cannot represent two adjacent characters of the license plate
//so we discard the one with the lowest confidence rate
			@param[out] boxes : std list of detected boxes
			@param[out] confidences : confidences detected boxes
			@param[out] classIds : std::list of indeces that indicate the classes of each of the above detected boxes

			@return void
			@see
			*/
void filter_iou2(
	std::list<cv::Rect>& boxes,
	std::list<float>& confidences,
	std::list<int>& classIds, float nmsThreshold);
/** @brief get barycenters of a list of bounding boxes.
	 *  @param[in]  boxes : bounding boxes.
	 *  @param[out]  les_points : the barycenters of the above boxes.
	* @return void
		* @see
	 */
void get_barycenters(const std::list<cv::Rect>& boxes, std::list<cv::Point2f>& les_points);
/**
		@brief
//checks if the bounding boxes are ardered from left to right
			@param[in] boxes : std list of detected boxes
			@return true if  the bounding boxes are ardered from left to right
			@see
			*/
#ifdef _DEBUG
			//cette fonction verifie que la liste est trie de gauche a droite 
bool debug_left(const std::list<cv::Rect>& boxes);
#endif //_DEBUG
/**
		@brief
		//examines how boxes are disposed and filter  out boxes with a position that are incompatible with the positions of other boxes
			@param[in] boxes : std list of detected boxes
			@param[out] angles_with_horizontal_line : angles determined by segment joining centroids of two consecutives boxes and the horizontal line
			@param[out] mean_angles_par_rapport_horiz : mean of the above angles - roughly the tilt of the license plate
			@param[out] standard_deviation_consecutives_angles : standard deviation of the above angles (normally a small value, since boxes should be aligned)
			@param[out] interdistances : distances between centroids of two consecutives boxes
			@param[out] mean_interdistances : mean of the above distances
			@param[out] standard_deviation_interdistances : standard deviation of the above distances
			@param[out] mean_produit_interdistance_avec_angle : parameter produced by the algorithm
			@param[out] standard_deviation_produit_interdistance_avec_angle : parameter produced by the algorithm
			@return void
			@see
			*/
void filtre_grubbs_sides(const std::list<cv::Rect>& boxes, std::list<float>& angles_with_horizontal_line,
	float& mean_angles_par_rapport_horiz,
	float& standard_deviation_consecutives_angles,
	std::list<int>& interdistances,
	float& mean_interdistances,
	float& standard_deviation_interdistances,
	float& mean_produit_interdistance_avec_angle, float& standard_deviation_produit_interdistance_avec_angle);
/**
		@brief
//caracters on a license plate  can be disposed on two lines (bi level) or on only just one line (single level).
//anycase the ascii caracters and there bouding boxes must nbe ordered in the inthe same way of the registration ascii chain.
			@param[in] boxes : std list of detected boxes
			@param[in] l_confidences : confidences of detected boxes
			@param[in] l_classIds : classes indeces of detected boxes.
@param[out] l_reordered : std list of the detected boxes when they are rearranged in the order of the registration string
			@param[out] l_reordered_confidences : confidences of the above boxes
			@param[out] l_reordered_classIds : classes indeces of the above boxes.
@param[out] levels : levels are int that indicates on which line of the license plate the box lies : -1 = upper line, +1 = lower line, 0 if the license plate have just one line
			@return void
			@see
			*/
void is_bi_level_plate(const std::list<cv::Rect>& boxes, const std::list<float>& l_confidences, const std::list<int>& l_classIds,
	std::list<cv::Rect>& l_reordered, std::list<float>& l_reordered_confidences, std::list<int>& l_reordered_classIds, std::list<int>& levels);
/**
		@brief
		//checks if the caracter of the lpn is a digit or a letter or something else (by ex examples misread carac or license plate bounding box, ...)
		*  @param[in]  input : a caracter (of the license plate).
			@return 1 if the carac is a digit (0...9), 0 if the carac is a letter (A,B,...,Z), -1 else
			@see
			*/
int is_digit(const char input);
/**
		@brief
//the dnn has detected boxes that represant characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
			@param[in] vect_of_detected_boxes :std vect of detected boxes
			@param[in] confidences confidences of each bb detected by the dnn
			@param[in] classIds : std::vector of indeces that indicate the classes of each of these detected boxes
			@param[out] tri_left_vect_of_detected_boxes :std vect of detected boxes when they rearranged from left to right
			@param[out] confidences confidences detected boxes when they rearranged from left to right
			@param[out] classIds : std::vector of indeces that indicate the classes of each of the above detected boxes
			@param[in] nmsThreshold A threshold used in non maximum suppression.
			@return the lpn extracted out of detections
			@see
			*/
std::string get_lpn(
	const std::vector<cv::Rect>& vect_of_detected_boxes,
	const std::vector<float>& confidences, const std::vector<int>& classIds,
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes,
	std::vector<float>& tri_left_confidences, std::vector<int>& tri_left_classIds, float nmsThreshold);



/***
	 * @brief Padded resize
	 * @param src - input image
	 * @param dst - output image
	 * @param out_size - desired output size
	 * @return padding information - pad width, pad height and zoom scale
	 */
std::vector<float> LetterboxImage(const cv::Mat & src, cv::Mat & dst, const cv::Size & out_size = cv::Size(640, 640)); 
/***
  * @brief Rescale coordinates to original input image
  * @param data - detection result after inference and nms
  * @param pad_w - width padding
  * @param pad_h - height padding
  * @param scale - zoom scale
  * @param img_shape - original input image shape
  */

void ScaleCoordinates(std::vector<Detection>& data, float pad_w, float pad_h,
							   float scale, const cv::Size& img_shape);


/***
 * @brief Performs Non-Maximum Suppression (NMS) on inference results
 * @note For 640x640 image, 640 / 32(max stride) = 20, sum up boxes from each yolo layer with stride (8, 16, 32) and
 *       3 scales at each layer, we can get total number of boxes - (20x20 + 40x40 + 80x80) x 3 = 25200
 * @param detections - inference results from the network, example [1, 25200, 85], 85 = 4(xywh) + 1(obj conf) + 80(class score)
   * @param modelWidth - width of model input 640
  * @param modelHeight - height of model input 640
 * @param conf_threshold - object confidence(objectness) threshold
 * @param iou_threshold - IoU threshold for NMS algorithm
 * @return detections with shape: nx7 (batch_index, x1, y1, x2, y2, score, classification)
 */
std::vector<std::vector<Detection>> PostProcessing(
	
	float* output, // output of onnx runtime ->>> 1,25200,85
	size_t dimensionsCount,
	size_t size, // 1x25200x85=2142000
	int dimensions,
	float modelWidth, float modelHeight, const cv::Size& img_shape,
	float conf_threshold, float iou_threshold);

/***
 * @brief Performs Non-Maximum Suppression (NMS) on inference results
 * @note For 640x640 image, 640 / 32(max stride) = 20, sum up boxes from each yolo layer with stride (8, 16, 32) and
 *       3 scales at each layer, we can get total number of boxes - (20x20 + 40x40 + 80x80) x 3 = 25200
 * @param detections - inference results from the network, example [1, 25200, 85], 85 = 4(xywh) + 1(obj conf) + 80(class score)
   * @param pad_w - width padding
  * @param pad_h - height padding
  * @param scale - zoom scale
 * @param conf_threshold - object confidence(objectness) threshold
 * @param iou_threshold - IoU threshold for NMS algorithm
 * @return detections with shape: nx7 (batch_index, x1, y1, x2, y2, score, classification)
 */
std::vector<std::vector<Detection>> PostProcessing(
	float* output, // output of onnx runtime ->>> 1,25200,85
	size_t dimensionsCount,
	size_t size, // 1x25200x85=2142000
	int dimensions,
	//pad_w is the left (and also right) border width in the square image feeded to the model
	float pad_w, float pad_h, float scale, const cv::Size& img_shape,
	float conf_threshold, float iou_threshold);
#endif // !defined(UTILS_ANPR_ONNX_H)
