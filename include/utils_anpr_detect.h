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
#include "Levenshtein.h"
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
struct LPN_signature {
	int nb_caracs_on_first_line;
	int nb_caracs_on_second_line;
	int nb_caracs_on_third_line;
};
/**
		@brief
		//return the ascii character that corresponds to index class output by the dnn
			@param classe : integer index = class identifier, output by the object detection dnn
			@return an ascii character
			@see
			*/
			//char get_char(const int classe);
			/**
					@brief
					//checks if the characters contained in lpn are compatible with the alphabet
						@param lpn: the registration of the vehicle as a string
						@return
						@see
						*/
						//bool could_be_lpn(const std::string& lpn);
						/**
								@brief
								returns the true license plate number out of a filename
								you must place the true license plate number in the image filename this way : number+underscore+license plate number,
								for instance filename 0000000001_3065WWA34.jpg will be interpreted as an image with the license plate 3065WWA34 in it.
									@param filename: the image filename that contains in it the true registration number
									@return the lpn contained in the image filename
									@see
									*/
									//std::string getTrueLPN(const std::string& filename, const bool& vrai_lpn_after_underscore);
									/**
											@brief
									//rearrange detected bounding boxes from left to right
												@param[out] boxes : set of detected boxes
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
	//return true if the intersection of the first argument box and the second has an interect area that is at least 90% of the box (which means box is nearly entirely in the second argument)
	@param box : box a bounding box
	@param rect_im : ROi or second bounding box
	@return true if intersection is at least 90% of the box (which means box is nearly entirely in the second argument)
	@see
	*/
bool is_in_rect_if(const cv::Rect& box, const cv::Rect& rect_im, const float min_area_ratio);
/**
		@brief
///from all heights of the boxes, get the median value
			@param[in] boxes : set of detected boxes
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
//if two boxes have an iou (intersection over union) that is too large, then they cannot represent two adjacent characters of the license plate
//so we discard the one with the lowest confidence rate
			@param[out] boxes : set of detected boxes
			@param[out] confidences : confidences detected boxes
			@param[out] classIds : std::list of indeces that indicate the classes of each of the above detected boxes
	 *  @param[in] nmsThreshold A threshold used in non maximum suppression.
			@return void
			@see
			*/
void filter_iou2(
	std::list<cv::Rect>& boxes,
	std::list<float>& confidences,
	std::list<int>& classIds, float nmsThreshold = 0.5f);
/** @brief get barycenters of a list of bounding boxes.
	 *  @param[in]  boxes : bounding boxes.
	 *  @param[out]  les_points : the barycenters of the above boxes.
	* @return void
		* @see
	 */
void get_barycenters(const std::list<cv::Rect>& boxes, std::list<cv::Point2f>& les_points);
/**
		@brief
//checks if the bounding boxes are ordered from left to right
			@param[in] boxes : set of detected boxes
			@return true if the bounding boxes are ordered from left to right
			@see
			*/
#ifdef _DEBUG
			//cette fonction verifie que la liste est triee de gauche a droite 
bool debug_left(const std::list<cv::Rect>& boxes);
#endif //_DEBUG
/**
		@brief
		//examines how boxes are disposed and filter out boxes with a position that are incompatible with the positions of other boxes
			@param[in] boxes : set of detected boxes
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
//characters on a license plate can be disposed on two lines (bi level) or on just one unique line (single level).
//anycase the ascii characters and there bouding boxes must nbe ordered in the inthe same way of the registration ascii chain.
			@param[in] boxes : set of detected boxes
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
		//checks if the character of the lpn is a digit or a letter or something else (by ex examples misread carac or license plate bounding box, ...)
		*  @param[in]  input : a character (of the license plate).
			@return 1 if the carac is a digit (0...9), 0 if the carac is a letter (A,B,...,Z), -1 else
			@see
			*/
int is_digit(const char input);
/**
		@brief
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
			@param[out] tri_left_vect_of_detected_boxes :set of detected boxes when they rearranged from left to right
			@param[out] tri_left_confidences : confidences corresponding detected boxes
			@param[out] tri_left_classIds : set of indeces of the above detected boxes
			@param[in] nmsThreshold A threshold used in non maximum suppression.
			@return the lpn extracted out of detections
			@see
			*/
std::string get_single_lpn(
	const std::vector<cv::Rect>& boxes,
	const std::vector<float>& confidences, const std::vector<int>& classIds,
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes,
	std::vector<float>& tri_left_confidences, std::vector<int>& tri_left_classIds, float nmsThreshold = 0.5f);
/**
		@brief
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
			@param[out] tri_left_vect_of_detected_boxes :set of detected boxes when they rearranged from left to right
			@param[out] tri_left_confidences : confidences corresponding detected boxes
			@param[out] tri_left_classIds : set of indeces of the above detected boxes
			@param[in] nmsThreshold A threshold used in non maximum suppression.
			@return the lpn extracted out of detections
			@see
			*/
std::string get_single_lpn(
	const std::vector<cv::Rect>& boxes,
	const std::vector<int>& classIds,
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes,
	std::vector<int>& tri_left_classIds, float nmsThreshold = 0.5f);
//
/**
		@brief : given the index of a bounding box, we can predict if this box is a single character or if it represents the license plate area or if it is the roi of an entire vehicle
		@param[in] classId: current box class index
					@param[in] classId_last_country : is the class index of the last country in the list of detected classes.
			@return single character--> returns 1 or license plate--> returns 2 or vehicle--> returns 3 or negative index--> returns 0 must be an error
			@see
			*/
int is_this_box_a_character_a_license_plate_or_a_vehicle(const int classId, const int classId_last_country);
//
//classId_last_country : is the class index of the last country in the list of detected classes.
/***
 * @brief
@param[in] classId: current box class index
*  @param[in]  number_of_characters_latin_numberplate : number of characters in a latin alphabet (usually 36 = 26 letters + 10 digits)
 * @return //single character--> returns 1
//license plate--> returns 2
//negative index--> returns 0 must be an error
 */
int is_this_box_a_character(const int classId, const int number_of_characters_latin_numberplate);
/**
		@brief
//groups detected boxes that correspond to the same vehicle. The separation is based on raw detections of license plates from the dnn
//it can deal with license pates that have two lines of charcaters
//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
			@param[out] l_vect_of_boxes_in_a_license_plate : double list of detected boxes when they are grouped. A single list contains all the boxes of a single vehicle present in image.
			@param[out] l_vect_of_confidences_in_a_license_plate : double list of corresponding confidences.
			@param[out] l_vect_of_classIds_in_a_license_plate : double list of corresponding indeces.
			@param[in] classId_last_country : is the class index of the last country in the list of detected classes.
			@return void
			@see
			*/
void group_characters_in_the_same_license_plate(
	//raw detections
	const std::list<cv::Rect>& boxes,
	const std::list<float>& confidences, const std::list<int>& classIds,
	//detections of same lp are regrouped in a vector
	std::list < std::list<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::list<float>>& l_vect_of_confidences_in_a_license_plate, std::list < std::list<int>>& l_vect_of_classIds_in_a_license_plate
	, const int classId_last_country
);
/**
		@brief
//groups detected boxes that correspond to the same vehicle. The separation is based on raw detections of license plates from the dnn
//it can deal with license pates that have two lines of charcaters
//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
			@param[out] l_vect_of_boxes_in_a_license_plate : double list of detected boxes when they are grouped. A single list contains all the boxes of a single vehicle present in image.
			@param[out] l_vect_of_confidences_in_a_license_plate : double list of corresponding confidences.
			@param[out] l_vect_of_classIds_in_a_license_plate : double list of corresponding indeces.
@param[in] classId_last_country : is the class index of the last country in the list of detected classes.
			@return void
			@see
			*/
void group_characters_in_the_same_license_plate(
	//raw detections
	const std::vector<cv::Rect>& boxes,
	const std::vector<float>& confidences, const std::vector<int>& classIds,
	//detections of same lp are regrouped in a vector
	std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate, std::list < std::vector<int>>& l_vect_of_classIds_in_a_license_plate
	, const int classId_last_country
);
/**
		@brief
//groups detected boxes that correspond to the same vehicle. The separation is based on raw detections of license plates from the dnn
//it can deal with license pates that have two lines of charcaters
//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
			@param[out] l_vect_of_boxes_in_a_license_plate : double list of detected boxes when they are grouped. A single list contains all the boxes of a single vehicle present in image.
			@param[out] l_vect_of_confidences_in_a_license_plate : double list of corresponding confidences.
			@param[out] l_vect_of_classIds_in_a_license_plate : double list of corresponding indeces.
			@param[in] classId_last_country : is the class index of the last country in the list of detected classes.
			@return void
			@see
			*/
void group_characters_in_the_same_license_plate(
	//raw detections
	const std::vector<cv::Rect>& boxes,
	const std::vector<float>& confidences, const std::vector<int>& classIds,
	//detections of same lp are regrouped in a vector
	std::list < std::list<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::list<float>>& l_vect_of_confidences_in_a_license_plate, std::list < std::list<int>>& l_vect_of_classIds_in_a_license_plate
	, const int classId_last_country
);
/**
		@brief
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
			@param[out] l_vect_of_boxes_in_a_license_plate : double list of detected boxes when they are grouped. A single list contains all the boxes of a single vehicle present in image.
			@param[out] l_vect_of_confidences_in_a_license_plate : double list of corresponding confidences.
			@param[out] l_vect_of_classIds_in_a_license_plate : double list of corresponding indeces.
			@param[in] classId_last_country : is the class index of the last country in the list of detected classes.
			@return void
			@see
			*/
void group_characters_in_the_same_license_plate(
	//raw detections
	const std::list<cv::Rect>& boxes, const std::list<float>& confidences, const std::list<int>& classIds,
	//detections of same lp are regrouped in a vector
	std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate, std::list < std::vector<int>>& l_vect_of_classIds_in_a_license_plate, const int classId_last_country
	//classId_last_country : is the class index of the last country in the list of detected classes.
);
//1;->ok
//2;->size too small
//4;->second detection is not a vehicle
//6;->detection after first two ones, is not a character
int is_detections_of_a_unique_license_plate(const std::vector<int>& classIds);
//1;->ok
//2;->size too small
//4;->second detection is not a vehicle
//6;->detection after first two ones, is not a character
int is_detections_of_a_unique_license_plate(const std::list<int>& classIds);
//From confidences of detections of all boxes of a plate, we get the average confidence.
float get_average_confidence_of_license_plate(const std::vector<int>& classIds,
	const std::vector<float>& confidences);
//From confidences of detections of all boxes of a plate, we get the average confidence.
float get_average_confidence_of_license_plate(const std::list<int>& classIds,
	const std::list<float>& confidences);
/**
		@brief
		For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
@param[out] one_lp :set of detected boxes when they rearranged from left to right
			@param[out] confidence_one_lp : confidences corresponding detected boxes
			@param[out] classIds_one_lp : set of indeces of the above detected boxes
			@return void
			@see
			*/
void get_best_plate(
	//detections when they are separated license plates by license plates
	const std::list < std::vector<int>>& classIds, const std::list < std::vector<float>>& confidences, const std::list < std::vector<cv::Rect>>& boxes
	//output the list of the best (most probable/readable) lp
	, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp);
/**
		@brief
		For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
@param[out] one_lp :set of detected boxes when they rearranged from left to right
			@param[out] confidence_one_lp : confidences corresponding detected boxes
			@param[out] classIds_one_lp : set of indeces of the above detected boxes
			@return void
			@see
			*/
void get_best_plate(
	//detections when they are separated license plates by license plates
	const std::list < std::list<int>>& classIds, const std::list < std::list<float>>& confidences, const std::list < std::list<cv::Rect>>& boxes
	//output the list of the best (most probable/readable) lp
	, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp);//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
/**
		@brief
		given the boxes+confidences+classIds and given the actual lpn string in the image (ExactLPN), outputs the lpn that is closest to ExactLPN
		For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
			@param[in] ExactLPN : the actual license plate number in the image
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
@param[out] one_lp :set of detected boxes when they rearranged from left to right
			@param[out] confidence_one_lp : confidences corresponding detected boxes
			@param[out] classIds_one_lp : set of indeces of the above detected boxes
			@return void
			@see
			*/
void get_best_plate(const std::string& ExactLPN,
	//detections when they are separated license plates by license plates
	const std::list < std::list<int>>& classIds, const std::list < std::list<float>>& confidences, const std::list < std::list<cv::Rect>>& boxes
	//output the list of the best (most probable/readable) lp
	, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp);
/**
		@brief
		given the boxes+confidences+classIds and given the actual lpn string in the image (ExactLPN), outputs the lpn that is closest to ExactLPN
		For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
			@param[in] ExactLPN : the actual license plate number in the image
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
@param[out] one_lp :set of detected boxes when they rearranged from left to right
			@param[out] confidence_one_lp : confidences corresponding detected boxes
			@param[out] classIds_one_lp : set of indeces of the above detected boxes
	 @param[in] classId_last_country : is the class index of the last country in the list of detected classes.
	 @return void
	 */
void get_best_plate(const std::string& ExactLPN,
	//detections when they are separated license plates by license plates
	const std::list < std::list<int>>& classIds, const std::list < std::list<float>>& confidences, const std::list < std::list<cv::Rect>>& boxes
	//output the list of the best (most probable/readable) lp
	, std::vector<float>& confidence_one_lp, std::vector < cv::Rect>& one_lp, std::vector<int>& classIds_one_lp);
//this function adds if they dont already exist, a roi for the licene plate (equal to the global rect englobing the boxes) and a blank rect for the vehicle box
void add_lp_and_vehicle(const std::list<cv::Rect>& boxes, const std::list<float>& confidences, const std::list<int>& classIds,
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes, std::vector<float>& tri_left_confidences, std::vector<int>& tri_left_classIds
	, const int classId_last_country
);
//
/** @brief
this function adds if they dont already exist, a roi for the licene plate (equal to the global rect englobing the boxes) and a blank rect for the vehicle box
@param[out] tri_left_vect_of_detected_boxes :set of detected boxes when they rearranged from left to right
			@param[out] tri_left_confidences : confidences corresponding detected boxes
			@param[out] tri_left_classIds : set of indeces of the above detected boxes
	 @param[in] classId_last_country : is the class index of the last country in the list of detected classes.
	 */
void add_lp_and_vehicle(const std::list<cv::Rect>& boxes, const std::list<float>& confidences, const std::list<int>& classIds,
	std::list<cv::Rect>& tri_left_vect_of_detected_boxes, std::list<float>& tri_left_confidences, std::list<int>& tri_left_classIds
	, const int classId_last_country
);
//
/** @brief
this function adds if they dont already exist, a roi for the licene plate (equal to the global rect englobing the boxes) and a blank rect for the vehicle box
@param[out] tri_left_vect_of_detected_boxes :set of detected boxes when they rearranged from left to right
			@param[out] tri_left_confidences : confidences corresponding detected boxes
			@param[out] tri_left_classIds : set of indeces of the above detected boxes
	 @param[in] classId_last_country : is the class index of the last country in the list of detected classes.
	 */
void add_lp_and_vehicle(const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, const std::vector<int>& classIds,
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes, std::vector<float>& tri_left_confidences, std::vector<int>& tri_left_classIds
	, const int classId_last_country
);
//
/** @brief
this function adds if they dont already exist, a roi for the licene plate (equal to the global rect englobing the boxes) and a blank rect for the vehicle box
@param[out] tri_left_vect_of_detected_boxes :set of detected boxes when they rearranged from left to right
			@param[out] tri_left_confidences : confidences corresponding detected boxes
			@param[out] tri_left_classIds : set of indeces of the above detected boxes
	 @param[in] classId_last_country : is the class index of the last country in the list of detected classes.
	 */
void add_lp_and_vehicle(const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, const std::vector<int>& classIds,
	std::list<cv::Rect>& tri_left_vect_of_detected_boxes, std::list<float>& tri_left_confidences, std::list<int>& tri_left_classIds
	, const int classId_last_country
);
/**
		@brief
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
			@param[out] tri_left_vect_of_detected_boxes :set of detected boxes when they rearranged from left to right
			@param[out] tri_left_confidences : confidences corresponding detected boxes
			@param[out] tri_left_classIds : set of indeces of the above detected boxes
			@param[in] nmsThreshold A threshold used in non maximum suppression.
			@param[in] classId_last_country : is the class index of the last country in the list of detected classes.
			@return the lpn extracted out of detections
			@see
			*/
std::string get_best_lpn(
	//raw detections
	const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, const std::vector<int>& classIds,
	//characters inside the best lpn that have been chosen from the above double linked list
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes,
	std::vector<float>& tri_left_confidences, std::vector<int>& tri_left_classIds, const float nmsThreshold = 0.5f, const int classId_last_country = 96
);
/**
		@brief
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
			@param[out] tri_left_vect_of_detected_boxes :set of detected boxes when they rearranged from left to right
			@param[out] tri_left_confidences : confidences corresponding detected boxes
			@param[out] tri_left_classIds : set of indeces of the above detected boxes
			@param[in] nmsThreshold A threshold used in non maximum suppression.
@param[in] classId_last_country : is the class index of the last country in the list of detected classes.
@return the lpn extracted out of detections
			@see
			*/
std::string get_best_lpn(
	//raw detections
	const std::list<cv::Rect>& boxes, const std::list<float>& confidences, const std::list<int>& classIds,
	//characters inside the best lpn that have been chosen from the above double linked list
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes,
	std::vector<float>& tri_left_confidences, std::vector<int>& tri_left_classIds, const float nmsThreshold = 0.5f, const int classId_last_country =//95
	96
);
//we know the true license plate number that come from a training image and we want to find the detections boxes to aautomatically annotate the image.
//We also have run the nn that produces detections, the goal of this func is to find the detections that are closest to the true lpn
/**
		@brief
		given the boxes+confidences+classIds and given the actual lpn string in the image (ExactLPN), outputs the lpn that is closest to ExactLPN
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classes : set of indeces that indicate the classes of each of these detected boxes
			@param[in] ExactLPN : the actual license plate number in the image
@param[out] best_boxes :set of detected boxes that compose the lp that has the best confidence
			@param[out] best_confidences : confidences corresponding detected boxes
			@param[out] best_classes : set of indeces of the above detected boxes
	 @param[in] classId_last_country : is the class index of the last country in the list of detected classes.
	 @return edit distance to the actual lpn
	 */
int find_nearest_plate_substitutions_allowed(const std::string& ExactLPN,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, const std::list <int>& lp_country_class, const std::list < cv::Rect>& lp_rois, const
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, const std::list < std::list<cv::Rect>>& boxes,
	//output = nearest lpn + its class + its bounding box
	std::string& best_lpn, int& best_country_class, cv::Rect& best_lpn_roi,
	//output = characters in nearest lpn 
	std::list<float>& best_confidences, std::list<int>& best_classes, std::list<cv::Rect>& best_boxes);
/**
		@brief
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
			@param[in] boxes : set of detected boxes
			@param[in] l_confidences : confidences of detected boxes
			@param[in] l_classIds : classes indeces of detected boxes.
			@param[out] tri_left_vect_of_detected_boxes :set of detected boxes when they rearranged from left to right
			@param[out] tri_left_confidences : confidences corresponding detected boxes
			@param[out] tri_left_classIds : set of indeces of the above detected boxes
			@param[in] nmsThreshold A threshold used in non maximum suppression.
			@return the lpn extracted out of detections
			@see
			*/
std::string get_lpn(
	const std::list<cv::Rect>& l_of_detected_boxes,
	const std::list<float>& l_confidences, const std::list<int>& l_classIds,
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes,
	std::vector<float>& tri_left_confidences, std::vector<int>& tri_left_classIds,
	float nmsThreshold = 0.5f
);
/**
		@brief
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
			@param[in] boxes : set of detected boxes
			@param[in] l_confidences : confidences of detected boxes
			@param[in] l_classIds : classes indeces of detected boxes.
			@param[out] tri_left_vect_of_detected_boxes :set of detected boxes when they rearranged from left to right
			@param[out] tri_left_confidences : confidences corresponding detected boxes
			@param[out] tri_left_classIds : set of indeces of the above detected boxes
			@param[in] nmsThreshold A threshold used in non maximum suppression.
			@return the lpn extracted out of detections
			@see
			*/
std::string get_lpn(
	const std::list<cv::Rect>& l_of_detected_boxes,
	const std::list<int>& l_classIds,
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes,
	std::vector<int>& tri_left_classIds,
	float nmsThreshold = 0.5f
);
/**
		@brief
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
			@param[in] l_classIds : classes indeces of detected boxes.
			@return the lpn extracted out of detections
			@see
			*/
std::string get_lpn(const std::list<int>& l_classIds);
/**
		@brief
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
			@param[in] l_classIds : classes indeces of detected boxes.
			@return the lpn extracted out of detections
			@see
			*/
std::string get_lpn(const std::vector<int>& l_classIds);
/**
		@brief
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
			@param[out] l_vect_of_boxes_in_a_license_plate : double list of detected boxes when they are grouped. A single list contains all the boxes of a single vehicle present in image.
			@param[out] l_vect_of_confidences_in_a_license_plate : double list of corresponding confidences.
			@param[out] l_vect_of_classIds_in_a_license_plate : double list of corresponding indeces.
			@param[out] l_vect_of_boxes_in_a_license_plate_tri_left : double list of detected boxes when they rearranged in license plates from left to right
@param[out] l_vect_of_confidences_in_a_license_plate_tri_left : double list of corresponding confidences.
			@param[out] l_vect_of_classIds_in_a_license_plate_tri_left : double list of corresponding indeces.
			@param[in] nmsThreshold A threshold used in non maximum suppression.
			@param[in] classId_last_country : is the class index of the last country in the list of detected classes.
			@return void
			@see
			*/
void separate_license_plates_if_necessary_add_blank_vehicles(
	//raw detections
	const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, const std::vector<int>& classIds,
	//detections when they are separated license plates by license plates
	std::list<std::string>& lpns, std::list < std::list<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::list<float>>& l_vect_of_confidences_in_a_license_plate, std::list <std::list<int>>& l_vect_of_classIds_in_a_license_plate,
	//double lists (one element list for each lp detected) of detected characters inside a lp
	std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate_tri_left,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate_tri_left, std::list <std::vector<int>>& l_vect_of_classIds_in_a_license_plate_tri_left,
	const int classId_last_country, //classId_last_country : is the class index of the last country in the list of detected classes.
	const float nmsThreshold = 0.5f
);
/**
		@brief
//the dnn has detected boxes that represent characters of the license plate, this function now groups characters in the same license plate and then rearranged from left to right.
//it can deal with license pates that have two lines of charcaters
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
			@param[out] l_vect_of_boxes_in_a_license_plate : double list of detected boxes when they are grouped. A single list contains all the boxes of a single vehicle present in image.
			@param[out] l_vect_of_confidences_in_a_license_plate : double list of corresponding confidences.
			@param[out] l_vect_of_classIds_in_a_license_plate : double list of corresponding indeces.
						@param[out] l_vect_of_boxes_in_a_license_plate_tri_left : double list of detected boxes when they rearranged in license plates from left to right
@param[out] l_vect_of_confidences_in_a_license_plate_tri_left : double list of corresponding confidences.
			@param[out] l_vect_of_classIds_in_a_license_plate_tri_left : double list of corresponding indeces.
			@param[in] nmsThreshold A threshold used in non maximum suppression.
@param[in] classId_last_country : is the class index of the last country in the list of detected classes.
@return void
			@see
			*/
			//groups detected boxes that correspond to the same vehicle. The separation is based on detected license plates by the dnn
void separate_license_plates_if_necessary_add_blank_vehicles(
	//raw detections
	const std::list<cv::Rect>& boxes, const std::list<float>& confidences, const std::list<int>& classIds,
	//detections when they are separated license plates by license plates
	std::list<std::string>& lpns, std::list < std::list<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::list<float>>& l_vect_of_confidences_in_a_license_plate, std::list <std::list<int>>& l_vect_of_classIds_in_a_license_plate,
	//double lists (one element list for each lp detected) of detected characters inside a lp
	std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate_tri_left,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate_tri_left, std::list <std::vector<int>>& l_vect_of_classIds_in_a_license_plate_tri_left,
	const int classId_last_country, const float nmsThreshold
);
/**
		@brief
//the dnn has detected boxes that represent characters of the license plate, this function now groups characters in the same license plate and then rearranged from left to right.
//it can deal with license pates that have two lines of charcaters
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
			@param[out] l_vect_of_boxes_in_a_license_plate : double list of detected boxes when they are grouped. A single list contains all the boxes of a single vehicle present in image.
			@param[out] l_vect_of_confidences_in_a_license_plate : double list of corresponding confidences.
			@param[out] l_vect_of_classIds_in_a_license_plate : double list of corresponding indeces.
						@param[out] l_vect_of_boxes_in_a_license_plate_tri_left : double list of detected boxes when they rearranged in license plates from left to right
@param[out] l_vect_of_confidences_in_a_license_plate_tri_left : double list of corresponding confidences.
			@param[out] l_vect_of_classIds_in_a_license_plate_tri_left : double list of corresponding indeces.
			@param[in] nmsThreshold A threshold used in non maximum suppression.
@param[in] classId_last_country : is the class index of the last country in the list of detected classes.
@return void
			@see
			*/
void separate_license_plates_if_necessary_add_blank_vehicles(
	//raw detections
	const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, const std::vector<int>& classIds,
	//detections when they are separated license plates by license plates
	std::list<std::string>& lpns, std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate, std::list <std::vector<int>>& l_vect_of_classIds_in_a_license_plate,
	//double lists (one element list for each lp detected) of detected characters inside a lp
	std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate_tri_left,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate_tri_left, std::list <std::vector<int>>& l_vect_of_classIds_in_a_license_plate_tri_left,
	const int classId_last_country, const float nmsThreshold = 0.5f
);
/**
		@brief
//the dnn has detected boxes that represent characters of the license plate, this function now groups characters in the same license plate and then rearranged from left to right.
//it can deal with license pates that have two lines of charcaters
			@param[in] boxes : set of detected boxes
			@param[in] confidences : confidences of each bb detected by the dnn
			@param[in] classIds : set of indeces that indicate the classes of each of these detected boxes
			@param[out] l_vect_of_boxes_in_a_license_plate : double list of detected boxes when they are grouped. A single list contains all the boxes of a single vehicle present in image.
			@param[out] l_vect_of_confidences_in_a_license_plate : double list of corresponding confidences.
			@param[out] l_vect_of_classIds_in_a_license_plate : double list of corresponding indeces.
						@param[out] l_vect_of_boxes_in_a_license_plate_tri_left : double list of detected boxes when they rearranged in license plates from left to right
@param[out] l_vect_of_confidences_in_a_license_plate_tri_left : double list of corresponding confidences.
			@param[out] l_vect_of_classIds_in_a_license_plate_tri_left : double list of corresponding indeces.
			@param[in] nmsThreshold A threshold used in non maximum suppression.
			@param[in] classId_last_country : is the class index of the last country in the list of detected classes.
			@return void
			@see
			*/
void separate_license_plates_if_necessary_add_blank_vehicles(
	//raw detections
	const std::list<cv::Rect>& boxes, const std::list<float>& confidences, const std::list<int>& classIds,
	//detections when they are separated license plates by license plates
	std::list<std::string>& lpns, std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate, std::list <std::vector<int>>& l_vect_of_classIds_in_a_license_plate,
	//double lists (one element list for each lp detected) of detected characters inside a lp
	std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate_tri_left,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate_tri_left, std::list <std::vector<int>>& l_vect_of_classIds_in_a_license_plate_tri_left,
	const int classId_last_country, const float nmsThreshold = 0.5f
);
//extracts, from a test directory, all images files 
//void load_images_filenames(const std::string& dir, std::list<std::string>& image_filenames);
/***
	 * @brief Padded resize
	 * @param src - input image
	 * @param dst - output image
	 * @param out_size - desired output size
	 * @return padding information - pad width, pad height and zoom scale
	 */
std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size = cv::Size(640, 640));
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
//if two boxes have an iou (intersection over union) that is too large, then they cannot represent two adjacent characters of the license plate 
//so we discard the one with the lowest confidence rate
void filter_iou(std::vector<int>& classIds,
	std::vector<float>& confidences,
	std::vector<cv::Rect>& vect_of_detected_boxes, const float& nmsThreshold = 0.5f);
#endif // !defined(UTILS_ANPR_ONNX_H)
