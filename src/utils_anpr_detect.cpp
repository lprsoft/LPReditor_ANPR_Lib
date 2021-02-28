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

#include "../include/utils_anpr_detect.h"

#include "../include/utils_image_file.h"
#include <filesystem>

/**
		@brief
//rearrange detected bounding boxes from left to right
			@param[out] boxes : std list of detected boxes
			@param[out] confidences : confidences detected boxes
			@param[out] classIds : std::list of indeces that indicate the classes of each of the above detected boxes
			@return void
			@see
			*/
void sort_from_left_to_right(std::list<cv::Rect>& boxes, std::list<float>& confidences, std::list<int>& classIds)
{
	std::list<cv::Rect> l_tri_left;
	std::list<float> confidences_tri_left;
	std::list<int> classIds_tri_left;
	while (!boxes.empty() && !confidences.empty() && !classIds.empty()) {
		int left_courant(boxes.front().x);
		std::list<cv::Rect>::iterator it = l_tri_left.begin();
		std::list<int>::iterator it_classIds = classIds_tri_left.begin();
		std::list<float>::iterator it_confidences = confidences_tri_left.begin();
		while (it != l_tri_left.end() && it_confidences != confidences_tri_left.end() &&
			it_classIds != classIds_tri_left.end()) {
			if (left_courant <= it->x) break;
			else {
				it++; it_confidences++; it_classIds++;
			}
		}
		l_tri_left.splice(it, boxes, boxes.begin());
		confidences_tri_left.splice(it_confidences, confidences, confidences.begin());
		classIds_tri_left.splice(it_classIds, classIds, classIds.begin());
	}
#ifdef _DEBUG
	assert(boxes.empty());
	//VERIFICATION DU TRI CROISSANT
	if (!l_tri_left.empty()) {
		std::list<cv::Rect>::const_iterator it_verif = l_tri_left.begin();
		int pred = it_verif->x;
		it_verif++;
		while (it_verif != l_tri_left.end()) {
#ifdef _DEBUG
			assert(it_verif->x >= pred);
#endif //_DEBUG
			pred = it_verif->x;
			it_verif++;
		}
	}
#endif	  //_DEBUG
	l_tri_left.swap(boxes);
	confidences_tri_left.swap(confidences);
	classIds_tri_left.swap(classIds);
}
/**
	@brief
	@param r1 : first rectangle
	@param r2 : second rectangle
	@return the intersection rect of the rectangles
	@see
	*/
cv::Rect get_inter(const cv::Rect& r1, const cv::Rect& r2)
{
	int width_inter = 0;
	int x_inter = r1.x;
	if (x_inter < r2.x) {
		x_inter = r2.x;
	}
	if (r1.x + r1.width < r2.x + r2.width)
		width_inter = r1.x + r1.width - x_inter;
	else width_inter = r2.x + r2.width - x_inter;
	int height_inter = 0;
	int y_inter = r1.y;
	if (y_inter < r2.y) {
		y_inter = r2.y;
	}
	if (r1.y + r1.height < r2.y + r2.height)
		height_inter = r1.y + r1.height - y_inter;
	else height_inter = r2.y + r2.height - y_inter;
	return cv::Rect(x_inter, y_inter, width_inter, height_inter);
}
/**
	@brief
	//return true if the intersection of the first argument box and the second has an interect area that is at least 90% of the box (which means box is nearly entirely in the second argument)
	@param box : box a bounding box
	@param rect_im : ROi or second bounding box
	@return true if intersection is at least 90% of the box (which means box is nearly entirely in the second argument)
	@see
	*/
bool is_in_rect_if(const cv::Rect& box, const cv::Rect& rect_im)
{
	cv::Rect inter(get_inter(box, rect_im));
	if (inter.width > 0 && inter.height > 0 && box.width > 0 && box.height > 0 && rect_im.width > 0 && rect_im.height > 0) {
		return ((float)(inter.area()) / (float)(box.area()) > 0.9f);
	}
	else return false;
}
/**
		@brief
///from all heights of the boxes, get the median value
			@param[in] boxes :std vect of detected boxes
			@return the median height of the boxes
			@see
			*/
int get_median_height(const std::list<cv::Rect>& boxes)
{
	if (boxes.empty()) return -1;
	else if (boxes.size() == 1) return boxes.front().height;
	else {
		std::list<int> list_tri;
		std::list<cv::Rect>::const_iterator it(boxes.begin());
		while (it != boxes.end()) {
			list_tri.push_back(it->y + it->height - it->y);
			it++;
		}
		list_tri.sort();
		std::list<int>::const_iterator it_median = list_tri.begin();
		int index = 0;
		while (it_median != list_tri.end()
			&& index < (list_tri.size() >> 1)) {
			index++;
			it_median++;
		}
		if (it_median == list_tri.end())
			return list_tri.back();
		else return *it_median;
	}
}
/**
	@brief
	//returns the iou (intersection over union) of two boxes
	@param r1 : first rectangle
	@param r2 : second rectangle
	@return the iou (a float value between 0 and 1)
	@see
	*/
float iou(const cv::Rect& r1, const cv::Rect& r2)
{
	cv::Rect inter(get_inter(r1, r2));
	if (inter.width > 0 && inter.height > 0 && r1.width > 0 && r1.height > 0 && r2.width > 0 && r2.height > 0) {
		return (float)(inter.area()) / (float)(r1.area() + r2.area() - inter.area());
	}
	else return 0.0f;
}
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
	std::list<int>& classIds, float nmsThreshold) {
	//filter out adjacent boxes with iou>nmsThreshold
		//***************************************************
		//                  FILTER
		//***************************************************
	std::list<float>::iterator it_confidences = (confidences.begin());
	std::list<cv::Rect>::iterator it_boxes = (boxes.begin());
	std::list<int>::iterator it_out_classes_ = (classIds.begin());
	///from all heights of the boxes, get the median value
	int median_height = get_median_height(boxes);
	while (it_out_classes_ != classIds.end()
		&& it_confidences != confidences.end() && it_boxes != boxes.end()) {
		std::list<float>::iterator it_confidences_next(it_confidences);
		std::list<cv::Rect>::iterator it_boxes_next(it_boxes);
		std::list<int>::iterator it_out_classes_next(it_out_classes_);
		it_out_classes_next++;
		it_confidences_next++;
		it_boxes_next++;
		bool must_erase_current_element = false;
		while (it_out_classes_next != classIds.end()
			&& it_confidences_next != confidences.end() && it_boxes_next != boxes.end()) {
			bool next_element_erased = false;
			float current_iou = iou(*it_boxes, *it_boxes_next);
			if (current_iou > nmsThreshold) {
				if (*it_confidences_next > * it_confidences) {
					//eliminate it_boxes
					must_erase_current_element = true;
					break;
				}
				else {//eliminate it_boxes_next
					//we dont move current iterator
					it_out_classes_next = classIds.erase(it_out_classes_next);
					it_confidences_next = confidences.erase(it_confidences_next);
					it_boxes_next = boxes.erase(it_boxes_next);
					next_element_erased = true;
				}
			}
			else {
				if (is_in_rect_if(*it_boxes, *it_boxes_next) || is_in_rect_if(*it_boxes_next, *it_boxes)) {
					if (abs(it_boxes->height - median_height) > abs(it_boxes_next->height - median_height)) {
						//eliminate it_boxes
						must_erase_current_element = true;
						break;
					}
					else {//eliminate it_boxes_next
					//we dont move current iterator
						it_out_classes_next = classIds.erase(it_out_classes_next);
						it_confidences_next = confidences.erase(it_confidences_next);
						it_boxes_next = boxes.erase(it_boxes_next);
						next_element_erased = true;
					}
				}
			}
			if (!next_element_erased) {
				it_out_classes_next++;
				it_confidences_next++;
				it_boxes_next++;
			}
		}
		if (//eliminate it_boxes
			must_erase_current_element
			) {
			it_out_classes_ = classIds.erase(it_out_classes_);
			it_confidences = confidences.erase(it_confidences);
			it_boxes = boxes.erase(it_boxes);
		}
		else {
			it_out_classes_++;
			it_confidences++;
			it_boxes++;
		}
	}
}
/** @brief get barycenters of a list of bounding boxes.
	 *  @param[in]  boxes : bounding boxes.
	 *  @param[out]  les_points : the barycenters of the above boxes.
	* @return void
		* @see
	 */
void get_barycenters(const std::list<cv::Rect>& boxes, std::list<cv::Point2f>& les_points)
{
#ifdef _DEBUG
	assert(les_points.empty());
#endif //_DEBUG
	std::list<cv::Rect>::const_iterator it(boxes.begin());
	while (it != boxes.end()) {
		float x_d;
		//update des sommes
		int somme_x(it->x + it->x + it->width);
		if (((somme_x >> 1) << 1) < somme_x) {
#ifdef _DEBUG
			assert(somme_x - ((somme_x >> 1) << 1) == 1);
#endif //_DEBUG
			somme_x = (somme_x >> 1);
			x_d = float(somme_x);
			x_d += 0.5f;
		}
		else {
			somme_x = (somme_x >> 1);
			x_d = float(somme_x);
		}
		float y_d;
		//update des sommes
		int somme_y(it->y + it->y + it->height);
		if (((somme_y >> 1) << 1) < somme_y) {
#ifdef _DEBUG
			assert(somme_y - ((somme_y >> 1) << 1) == 1);
#endif //_DEBUG
			somme_y = (somme_y >> 1);
			y_d = float(somme_y);
			y_d += 0.5f;
		}
		else {
			somme_y = (somme_y >> 1);
			y_d = float(somme_y);
		}
#ifdef _DEBUG
		assert(y_d + FLT_EPSILON > it->y&& y_d - FLT_EPSILON<it->y + it->height && x_d + FLT_EPSILON>it->x && x_d - FLT_EPSILON < it->x + it->width);
#endif //_DEBUG
		les_points.push_back(cv::Point2f(x_d, y_d));
		it++;
	}
}

/**
		@brief
//checks if the bounding boxes are ardered from left to right
			@param[in] boxes : std list of detected boxes
			@return true if  the bounding boxes are ardered from left to right
			@see
			*/
#ifdef _DEBUG
			//cette fonction verifie que la liste est trie de gauche a droite 
bool debug_left(const std::list<cv::Rect>& boxes) {
	if (boxes.size() < 2) return true;
	else {
		//VERIFICATION DU TRI CROISSANT
		std::list<cv::Rect>::const_iterator it_verif = boxes.begin();
		int pred = it_verif->x;
		it_verif++;
		while (it_verif != boxes.end()) {
#ifdef _DEBUG
			assert(it_verif->x >= pred);
#endif //_DEBUG
			if (it_verif->x < pred) return false;
			pred = it_verif->x;
			it_verif++;
		}
		return true;
	}
}
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
	float& mean_produit_interdistance_avec_angle, float& standard_deviation_produit_interdistance_avec_angle)
{
	//cette fonction verifie que la liste est trie de gauche a droite 
#ifdef _DEBUG
	assert(debug_left(boxes));
#endif //_DEBUG
	if (boxes.size() < 3) {
		standard_deviation_interdistances = -1.0f; standard_deviation_consecutives_angles = -1.0f;
	}
	else {
		std::list<cv::Point2f> les_points;
		//retourne la liste forme de centres des ROI
		get_barycenters(boxes, les_points);
#ifdef _DEBUG
		assert(boxes.size() == les_points.size());
#endif //_DEBUG
		//********************************************************************
		//              standard_deviation_consecutives_angles
		//********************************************************************
		float somme_produit_interdistance_avec_angle(.0f);
		float somme_carre_produit_interdistance_avec_angle(.0f);
		float somme_interdistances(.0f);
		float somme_carre_interdistances(.0f);
		float somme_angles_par_rapport_horiz(.0f);
		float somme_carre_angles_par_rapport_horiz(.0f);
		std::list<cv::Point2f>::const_iterator it_centre(les_points.begin());
		//calcul du premier angle 
		cv::Point2f pt_pred(*it_centre);
		it_centre++;
		while (it_centre != les_points.end()) {
			cv::Point2f pt_suivant(*it_centre);
			//constructeur a partir de deux pts
			C_Line current_line(pt_pred, pt_suivant);
			//retour l'angle dee la droite par rapport a l'horizontale
			float angle_pred = current_line.get_skew_angle();
			somme_angles_par_rapport_horiz += angle_pred;
			somme_carre_angles_par_rapport_horiz += (angle_pred * angle_pred);
			//retour l'angle dee la droite par rapport a l'horizontale
			angles_with_horizontal_line.push_back(angle_pred);
			somme_interdistances += (pt_suivant.x - pt_pred.x);
			somme_carre_interdistances +=
				((pt_suivant.x - pt_pred.x) * (pt_suivant.x - pt_pred.x));
			interdistances.push_back(int(pt_suivant.x - pt_pred.x));
			float produit_courant = (pt_suivant.x - pt_pred.x) * angle_pred;
			somme_produit_interdistance_avec_angle += produit_courant;
			somme_carre_produit_interdistance_avec_angle += produit_courant * produit_courant;
			pt_pred = pt_suivant;
			it_centre++;
		}
#ifdef _DEBUG
		assert(angles_with_horizontal_line.size() + 1 == les_points.size());
#endif //_DEBUG
		standard_deviation_consecutives_angles = (somme_carre_angles_par_rapport_horiz)
			*angles_with_horizontal_line.size();
		standard_deviation_consecutives_angles -= (somme_angles_par_rapport_horiz * somme_angles_par_rapport_horiz);
#ifdef _DEBUG
		assert(standard_deviation_consecutives_angles > -FLT_EPSILON);
#endif //_DEBUG
		standard_deviation_consecutives_angles /= (angles_with_horizontal_line.size() * angles_with_horizontal_line.size());
		standard_deviation_consecutives_angles = sqrtf(standard_deviation_consecutives_angles);
		mean_angles_par_rapport_horiz = somme_angles_par_rapport_horiz / (angles_with_horizontal_line.size());
#ifdef _DEBUG
		assert(interdistances.size() + 1 == les_points.size());
#endif //_DEBUG
		standard_deviation_interdistances = (somme_carre_interdistances)*interdistances.size();
		standard_deviation_interdistances -= (somme_interdistances * somme_interdistances);
#ifdef _DEBUG
		assert(standard_deviation_interdistances > -FLT_EPSILON);
#endif //_DEBUG
		standard_deviation_interdistances /= (interdistances.size() * interdistances.size());
		standard_deviation_interdistances = sqrtf(standard_deviation_interdistances);
		mean_interdistances = somme_interdistances / (interdistances.size());
		standard_deviation_produit_interdistance_avec_angle = (somme_carre_produit_interdistance_avec_angle)*interdistances.size();
		standard_deviation_produit_interdistance_avec_angle -= (somme_produit_interdistance_avec_angle * somme_produit_interdistance_avec_angle);
#ifdef _DEBUG
		//assert(standard_deviation_produit_interdistance_avec_angle > -FLT_EPSILON);
#endif //_DEBUG
		standard_deviation_produit_interdistance_avec_angle /= (interdistances.size() * interdistances.size());
		standard_deviation_produit_interdistance_avec_angle = sqrtf(standard_deviation_produit_interdistance_avec_angle);
		mean_produit_interdistance_avec_angle = somme_produit_interdistance_avec_angle / (interdistances.size());
	}
}
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
	std::list<cv::Rect>& l_reordered, std::list<float>& l_reordered_confidences, std::list<int>& l_reordered_classIds, std::list<int>& levels)
{
	if (boxes.size() > 1) {
		//cette fonction trie la liste de gauche a droite 
		std::copy(boxes.begin(), boxes.end(), std::back_inserter(l_reordered));
		std::copy(l_confidences.begin(), l_confidences.end(), std::back_inserter(l_reordered_confidences));
		std::copy(l_classIds.begin(), l_classIds.end(), std::back_inserter(l_reordered_classIds));
		sort_from_left_to_right(l_reordered, l_reordered_confidences, l_reordered_classIds);
		//la difference des angles des n-1 premiers caracteres par rapport l'horizontale
							//l'interdistance de chaque caractere avec son suivant pour les n-1 premiers caracteres
		std::list<float> angles_par_rapport_horiz;
		float moy_angles_par_rapport_horiz;
		float ecart_type_angles_consecutifs;
		std::list<int> interdistances;
		float moy_interdistances;
		float ecart_type_interdistances;
		float moy_produit_des_2; float ecart_type_produit_des_2;
		filtre_grubbs_sides(l_reordered, angles_par_rapport_horiz,
			moy_angles_par_rapport_horiz, ecart_type_angles_consecutifs, interdistances,
			moy_interdistances, ecart_type_interdistances, moy_produit_des_2,
			ecart_type_produit_des_2);
		std::list<cv::Rect>::const_iterator it = l_reordered.begin();
		std::list<float>::const_iterator it_angles(angles_par_rapport_horiz.begin());
		while (it_angles != angles_par_rapport_horiz.end() && it != l_reordered.end()) {
			if (fabs(*it_angles) > 0.785398f) {
				//3.141593/4 pi/4
				if (levels.size()) {
					std::list<cv::Rect>::const_iterator it_next = it;
					it_next++;
					cv::Rect box_left(*it);
					cv::Rect box_right(*it_next);
#ifdef _DEBUG		
					assert(iou(box_left, box_right) < 0.25f);
#endif //_DEBUG
					if (2 * box_left.y + box_left.height < 2 * box_right.y + box_right.height) {
						//in this case box_left is on first line and box_right is on second line
						std::list<int>::reverse_iterator it_levels(levels.rbegin());
						while (it_levels != levels.rend()) {
							if (*it_levels == 0) {
								*it_levels = -1;
								it_levels++;
							}
							else break;
						}
						if ((levels.back() == 1 || levels.back() == 0) && it_angles != angles_par_rapport_horiz.begin()) {
							std::list<float>::const_iterator it_angles_pred(it_angles); it_angles_pred--;
							if (fabs(*it_angles_pred) < fabs(*it_angles)) {
								levels.back() = -1;
							}
						}
						levels.push_back(1);
					}
					else {
						std::list<int>::reverse_iterator it_levels(levels.rbegin());
						while (it_levels != levels.rend()) {
							if (*it_levels == 0) {
								*it_levels = 1;
								it_levels++;
							}
							else break;
						}
						if ((levels.back() == -1 || levels.back() == 0) && it_angles != angles_par_rapport_horiz.begin()) {
							std::list<float>::const_iterator it_angles_pred(it_angles); it_angles_pred--;
							if (fabs(*it_angles_pred) < fabs(*it_angles)) {
								levels.back() = 1;
							}
						}
						//in this case box_right is on first line and box_left is on second line
						levels.push_back(-1);
					}
				}
				else {
					std::list<cv::Rect>::const_iterator it_next = it;
					it_next++;
					cv::Rect box_left(*it);
					cv::Rect box_right(*it_next);
#ifdef _DEBUG		
					assert(iou(box_left, box_right) < 0.25f);
#endif //_DEBUG
					if (2 * box_left.y + box_left.height < 2 * box_right.y + box_right.height) {
						//in this case box_left is on first line and box_right is on second line
						levels.push_back(-1);
						levels.push_back(1);
					}
					else {
						//in this case box_right is on first line and box_left is on second line
						levels.push_back(1);
						levels.push_back(-1);
					}
				}
			}
			else {
				//3.141593/4 pi/4
				if (levels.size()) {
					levels.push_back(levels.back());
				}
				else {
					//in this case box_right is on first line and box_left is on second line
					levels.push_back(0);
					levels.push_back(levels.back());
				}
			}
			it++;
			it_angles++;
		}
#ifdef _DEBUG		
		assert(levels.size() == angles_par_rapport_horiz.size() + 1 && levels.size() == l_reordered.size() && levels.size() == boxes.size());
#endif //_DEBUG
	}
}
/**
		@brief
		//checks if the caracter of the lpn is a digit or a letter or something else (by ex examples misread carac or license plate bounding box, ...)
		*  @param[in]  input : a caracter (of the license plate).
			@return 1 if the carac is a digit (0...9), 0 if the carac is a letter (A,B,...,Z), -1 else
			@see
			*/
int is_digit(const char input)
{
	if (
		input == 'A' ||
		input == 'B' ||
		input == 'C' ||
		input == 'D' ||
		input == 'E' ||
		input == 'F' ||
		input == 'G' ||
		input == 'H' ||
		input == 'I' ||
		input == 'J' ||
		input == 'K' ||
		input == 'L' ||
		input == 'M' ||
		input == 'N' ||
		/*input==   'O'|| */
		input == 'P' ||
		input == 'Q' ||
		input == 'R' ||
		input == 'S' ||
		input == 'T' ||
		input == 'U' ||
		input == 'V' ||
		input == 'W' ||
		input == 'X' ||
		input == 'Y' ||
		input == 'Z') return 0;
	else if (
		/*input==   '0'||*/
		input == '1' ||
		input == '2' ||
		input == '3' ||
		input == '4' ||
		input == '5' ||
		input == '6' ||
		input == '7' ||
		input == '8' ||
		input == '9') return 1;
	else if (input == '0' ||
		input == 'O') return 2;
	else return -1;
}
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
	std::vector<float>& tri_left_confidences, std::vector<int>& tri_left_classIds, float nmsThreshold
) {
	//first from left to right
	//cette fonction trie la liste de gauche a droite 
	//change from vect to list 
	std::list<cv::Rect> boxes;
	std::copy(vect_of_detected_boxes.begin(), vect_of_detected_boxes.end(), std::back_inserter(boxes));
	std::list<float> l_confidences;
	std::copy(confidences.begin(), confidences.end(), std::back_inserter(l_confidences));
	std::list<int> l_classIds;
	std::copy(classIds.begin(), classIds.end(), std::back_inserter(l_classIds));
	sort_from_left_to_right(boxes, l_confidences, l_classIds);
	//filter out lpn box
	//***************************************************
	//                  FILTER
	//***************************************************
	std::list<float>::iterator it_confidences(l_confidences.begin());
	std::list<cv::Rect>::iterator it_boxes(boxes.begin());
	std::list<int>::iterator it_out_classes_(l_classIds.begin());
	const int number_of_caracters = 36;
	while (it_out_classes_ != l_classIds.end()
		&& it_confidences != l_confidences.end() && it_boxes != boxes.end()) {
		if (!(*it_out_classes_ < number_of_caracters - 1)) {
			it_out_classes_ = l_classIds.erase(it_out_classes_);
			it_confidences = l_confidences.erase(it_confidences);
			it_boxes = boxes.erase(it_boxes);
		}
		else {
			it_out_classes_++;
			it_confidences++;
			it_boxes++;
		}
	}
	//filter out adjacent boxes with iou>nmsThreshold
		//***************************************************
		//                  FILTER
		//***************************************************
//if two boxes have an iou (intersection over union) that is two large, then they cannot represent two adjacent characters of the license plate 
//so we discard the one with the lowest confidence rate
	filter_iou2(boxes, l_confidences, l_classIds, nmsThreshold);
	std::list<int> levels;//levels of each character box of l_tri_left
	std::list<char> lpn;
	if (boxes.size() > 3) {
		std::list<cv::Rect> l_tri_left;//list of characters boxes ranged from left to right
		std::list<float> l_tri_left_confidences;
		std::list<int> l_tri_left_classIds;
		is_bi_level_plate(boxes, l_confidences, l_classIds, l_tri_left, l_tri_left_confidences, l_tri_left_classIds, levels);
		//now
		std::list<char> lpn_minus_1;
		std::list<char> lpn_0;
		std::list<char> lpn_plus_1;
		//C_OCROutputs availableAlpha(LATIN_LETTERS_NO_I_O_LATIN_DIGITS);
		std::list<int>::const_iterator it_out_classes(l_tri_left_classIds.begin());
		std::list<int>::const_iterator it_levels(levels.begin());
		std::list<cv::Rect>::const_iterator it_box(l_tri_left.begin());//list of characters boxes ranged from left to right
		std::list<float>::const_iterator it_confidence(l_tri_left_confidences.begin());
		std::list<cv::Rect> l_minus_1;//list of characters boxes ranged from left to right
		std::list<float> l_confidences_minus_1;
		std::list<int> l_classIds_minus_1;
		std::list<cv::Rect> l_0;//list of characters boxes ranged from left to right
		std::list<float> l_confidences_0;
		std::list<int> l_classIds_0;
		std::list<cv::Rect> l_plus_1;//list of characters boxes ranged from left to right
		std::list<float> l_confidences_plus_1;
		std::list<int> l_classIds_plus_1;
		while (it_out_classes != l_tri_left_classIds.end() && it_levels != levels.end() &&
			it_box != l_tri_left.end() && it_confidence != l_tri_left_confidences.end()) {
			if (*it_out_classes < number_of_caracters - 1) {
				if (*it_levels == -1) {
					lpn_minus_1.push_back(get_char(*it_out_classes));
					l_minus_1.push_back(*it_box);//list of characters boxes ranged from left to right
					l_confidences_minus_1.push_back(*it_confidence);
					l_classIds_minus_1.push_back(*it_out_classes);
				}
				else {
					if (*it_levels == 0) {
						lpn_0.push_back(get_char(*it_out_classes));
						l_0.push_back(*it_box);//list of characters boxes ranged from left to right
						l_confidences_0.push_back(*it_confidence);
						l_classIds_0.push_back(*it_out_classes);
					}
					else if (*it_levels == 1) {
						lpn_plus_1.push_back(get_char(*it_out_classes));
						l_plus_1.push_back(*it_box);//list of characters boxes ranged from left to right
						l_confidences_plus_1.push_back(*it_confidence);
						l_classIds_plus_1.push_back(*it_out_classes);
					}
				}
			}
			it_out_classes++; it_levels++; it_box++; it_confidence++;
		}
		lpn.splice(lpn.end(), lpn_minus_1);
		lpn.splice(lpn.end(), lpn_0);
		lpn.splice(lpn.end(), lpn_plus_1);
		l_tri_left.clear();//list of characters boxes ranged from left to right
		l_tri_left_confidences.clear();
		l_tri_left_classIds.clear();
		l_tri_left.splice(l_tri_left.end(), l_minus_1);
		l_tri_left.splice(l_tri_left.end(), l_0);
		l_tri_left.splice(l_tri_left.end(), l_plus_1);
		l_tri_left_confidences.splice(l_tri_left_confidences.end(), l_confidences_minus_1);
		l_tri_left_confidences.splice(l_tri_left_confidences.end(), l_confidences_0);
		l_tri_left_confidences.splice(l_tri_left_confidences.end(), l_confidences_plus_1);
		l_tri_left_classIds.splice(l_tri_left_classIds.end(), l_classIds_minus_1);
		l_tri_left_classIds.splice(l_tri_left_classIds.end(), l_classIds_0);
		l_tri_left_classIds.splice(l_tri_left_classIds.end(), l_classIds_plus_1);
#ifdef _DEBUG		
		assert(lpn.size() > 3 && lpn.size() == l_tri_left.size()
			&& lpn.size() == l_tri_left_confidences.size()
			&& lpn.size() == l_tri_left_classIds.size());
#endif //_DEBUG
		bool european_plate = true;
		if (lpn.size() > 3 && lpn.size() == l_tri_left.size()
			&& lpn.size() == l_tri_left_confidences.size()
			&& lpn.size() == l_tri_left_classIds.size()) {
			std::list<char>::iterator it_char(lpn.begin());
			std::list<int>::const_iterator it_out_classes(l_tri_left_classIds.begin());
			std::list<int>::const_iterator it_levels(levels.begin());
			std::list<cv::Rect>::const_iterator it_box(l_tri_left.begin());//list of characters boxes ranged from left to right
			std::list<float>::const_iterator it_confidence(l_tri_left_confidences.begin());
			while (it_char != lpn.end() && it_out_classes != l_tri_left_classIds.end() &&
				it_box != l_tri_left.end() && it_confidence != l_tri_left_confidences.end()
				) {
#ifdef _DEBUG		
				assert(*it_char != '0');
#endif //_DEBUG
				if (*it_char == 'O') {
					if (european_plate) {
						if (it_char == lpn.begin()) {
							std::list<char>::iterator it_next(it_char); it_next++;
							int digit(is_digit(*it_next));
							if (digit != 0 && digit != 1)
							{

							}
							else if (digit == 1)
							{
								*it_char = '0';
							}
						}
						else {
							std::list<char>::iterator it_next(it_char); it_next--;
							int digit(is_digit(*it_next));
							if (digit == 1)
							{
								*it_char = '0';
							}
						}
					}
					else *it_char = '0';
				}
				if (*it_char == '1') {
					if (european_plate) {
						if (it_char == lpn.begin()) {
							std::list<char>::iterator it_next(it_char); it_next++;
							int digit(is_digit(*it_next));
							if (digit != 0 && digit != 1)
							{
								*it_char = 'I';
							}
						}
						else {
							std::list<char>::iterator it_pred(it_char); it_pred--;
							int digit(is_digit(*it_pred));
							if (digit != 0 && digit != 1)
							{
								std::list<char>::iterator it_next(it_char); it_next++;
								if (it_next != lpn.end())
									int digit(is_digit(*it_next));
								if (digit != 0 && digit != 1)
								{
									*it_char = 'I';
									//*it_out_classes = ;
								}

							}
						}
					}
				}
				it_char++;
				it_out_classes++; it_box++; it_confidence++;
			}
		}
		std::string lpn_corrected;
		std::list<char>::iterator it_char(lpn.begin());
		while (it_char != lpn.end()) {
			lpn_corrected.push_back(*it_char);
			it_char++;
		}
		std::copy(l_tri_left.begin(), l_tri_left.end(), std::back_inserter(tri_left_vect_of_detected_boxes));
		std::copy(l_tri_left_confidences.begin(), l_tri_left_confidences.end(), std::back_inserter(tri_left_confidences));
		std::copy(l_tri_left_classIds.begin(), l_tri_left_classIds.end(), std::back_inserter(tri_left_classIds));
		//std::cout << "read lpn by engine : " << lpn_corrected << std::endl;
		return lpn_corrected;
	}
	else {
		std::string lpn_corrected;
		//C_OCROutputs availableAlpha(LATIN_LETTERS_NO_I_O_LATIN_DIGITS);
		std::list<int>::const_iterator it_out_classes(l_classIds.begin());
		std::list<int>::const_iterator it_levels(levels.begin());
		while (it_out_classes != l_classIds.end() && it_levels != levels.end()) {
			if (*it_out_classes < number_of_caracters - 1) {
				lpn.push_back(get_char(*it_out_classes));
				lpn_corrected.push_back(get_char(*it_out_classes));
			}
			it_out_classes++; it_levels++;
		}
		std::copy(vect_of_detected_boxes.begin(), vect_of_detected_boxes.end(), std::back_inserter(tri_left_vect_of_detected_boxes));
		std::copy(confidences.begin(), confidences.end(), std::back_inserter(tri_left_confidences));
		std::copy(classIds.begin(), classIds.end(), std::back_inserter(tri_left_classIds));
		//std::cout << "read lpn by engine : " << lpn_corrected << std::endl;
		return lpn_corrected;
	}
}


/***
 * @brief Padded resize
 * @param src - input image
 * @param dst - output image
 * @param out_size - desired output size
 * @return padding information - pad width, pad height and zoom scale
 */
std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size) {
	auto in_h = static_cast<float>(src.rows);
	auto in_w = static_cast<float>(src.cols);
	float out_h = float(out_size.height);
	float out_w = float(out_size.width);

	float scale = std::min(out_w / in_w, out_h / in_h);

	int mid_h = static_cast<int>(in_h * scale);
	int mid_w = static_cast<int>(in_w * scale);

	cv::resize(src, dst, cv::Size(mid_w, mid_h));

	int top = (static_cast<int>(out_h) - mid_h) / 2;
	int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
	int left = (static_cast<int>(out_w) - mid_w) / 2;
	int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

	cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	std::vector<float> pad_info{ static_cast<float>(left), static_cast<float>(top), scale };
	return pad_info;
}
//pad_w is the left (and also right) border width in the square image feeded to the model
void ScaleCoordinates(std::vector<Detection>& data, float pad_w, float pad_h,
	float scale, const cv::Size& img_shape) {
	auto clip = [](float n, float lower, float upper) {
		return std::max(lower, std::min(n, upper));
	};

	std::vector<Detection> detections;
	for (auto& i : data) {
		//pad_w is the left (and also right) border width in the square image feeded to the model
		float x1 = (i.bbox.tl().x - pad_w) / scale;  // x padding
		float y1 = (i.bbox.tl().y - pad_h) / scale;  // y padding
		float x2 = (i.bbox.br().x - pad_w) / scale;  // x padding
		float y2 = (i.bbox.br().y - pad_h) / scale;  // y padding

		x1 = clip(x1, 0, float(img_shape.width));
		y1 = clip(y1, 0, float(img_shape.height));
		x2 = clip(x2, 0, float(img_shape.width));
		y2 = clip(y2, 0, float(img_shape.height));

		i.bbox = cv::Rect(cv::Point(lround(x1), lround(y1)), cv::Point(lround(x2), lround(y2)));
	}
}

std::vector<std::vector<Detection>> PostProcessing(
	float* output, // output of onnx runtime ->>> 1,25200,85
	size_t dimensionsCount,
	size_t size, // 1x25200x85=2142000
	int dimensions,
	//pad_w is the left (and also right) border width in the square image feeded to the model
	float modelWidth, float modelHeight, const cv::Size& img_shape,
	float conf_threshold, float iou_threshold) {
	constexpr int item_attr_size = 5;
	int batch_size = 1;
	// number of classes, e.g. 80 for coco dataset

	// get candidates which object confidence > threshold
	int rows = int(size) / dimensions; //25200
	int confidenceIndex = 4;
	int labelStartIndex = 5;
	float xGain = modelWidth / float(img_shape.width);
	float yGain = modelHeight / float(img_shape.height);
	std::vector<cv::Vec4f> locations;
	std::vector<int> labels_;
	std::vector<float> confidences;
	std::vector<cv::Rect> src_rects;
	std::vector<Detection> det_vec;//batch 1 one image at a time
	cv::Rect rect;
	cv::Vec4f location;
	for (int i = 0; i < rows; ++i) {
		int index = i * dimensions;
		if (output[index + confidenceIndex] <= conf_threshold) continue;
		//each class score of the current bb is multiplied by the bb confidence
		for (int j = labelStartIndex; j < dimensions; ++j) {
			output[index + j] = output[index + j] * output[index + confidenceIndex];
		}
		for (int k = labelStartIndex; k < dimensions; ++k) {
			if (output[index + k] <= conf_threshold) continue;
			else {
				//xywh2xyxy
				location[0] = (output[index] - output[index + 2] / 2) / xGain;//top left x
				location[1] = (output[index + 1] - output[index + 3] / 2) / yGain;//top left y
				location[2] = (output[index] + output[index + 2] / 2) / xGain;//bottom right x
				location[3] = (output[index + 1] + output[index + 3] / 2) / yGain;//bottom right y
				locations.emplace_back(location);
				rect = cv::Rect(lround(location[0]), lround(location[1]),
					lround(location[2] - location[0]), lround(location[3] - location[1]));
				src_rects.push_back(rect);
				labels_.emplace_back(k - labelStartIndex);
				confidences.emplace_back(output[index + k]);
			}
		}
	}
	// run NMS
	std::vector<int> nms_indices;
	cv::dnn::NMSBoxes(src_rects, confidences, conf_threshold, iou_threshold, nms_indices);

	for (size_t i = 0; i < nms_indices.size(); i++) {
		Detection detection;
		detection.bbox = src_rects[nms_indices[i]];
		detection.score = confidences[nms_indices[i]];
		detection.class_idx = labels_[nms_indices[i]];
		det_vec.push_back(detection);
	}
	std::vector<std::vector<Detection>> detections;
	detections.reserve(batch_size);
	// save final detection for the current image
	detections.emplace_back(det_vec);
	return detections;
}
std::vector<std::vector<Detection>> PostProcessing(
	float* output, // output of onnx runtime ->>> 1,25200,85
	size_t dimensionsCount,
	size_t size, // 1x25200x85=2142000
	int dimensions,
	//pad_w is the left (and also right) border width in the square image feeded to the model
	float pad_w, float pad_h, float scale, const cv::Size& img_shape,
	float conf_threshold, float iou_threshold) {
	constexpr int item_attr_size = 5;
	int batch_size = 1;
	// number of classes, e.g. 80 for coco dataset

	// get candidates which object confidence > threshold
	int rows = int(size) / dimensions; //25200
	int confidenceIndex = 4;
	int labelStartIndex = 5;
	//float modelWidth = 640.0f;
	//float modelHeight = 640.0f;
	//float xGain = modelWidth / float(img_shape.width);
	//float yGain = modelHeight / float(img_shape.height);
	std::vector<cv::Vec4f> locations;
	std::vector<int> labels_;
	std::vector<float> confidences;
	std::vector<cv::Rect> src_rects;
	std::vector<Detection> det_vec;//batch 1 one image at a time
	cv::Rect rect;
	cv::Vec4f location;
	for (int i = 0; i < rows; ++i) {
		int index = i * dimensions;

		if (output[index + confidenceIndex] <= conf_threshold) continue;

		//each class score of the current bb is multiplied by the bb confidence
		for (int j = labelStartIndex; j < dimensions; ++j) {
			output[index + j] = output[index + j] * output[index + confidenceIndex];
		}
		for (int k = labelStartIndex; k < dimensions; ++k) {
			if (output[index + k] <= conf_threshold) continue;
			else {
				//xywh2xyxy
				location[0] = (output[index] - output[index + 2] / 2);//top left x
				location[1] = (output[index + 1] - output[index + 3] / 2);//top left y
				location[2] = (output[index] + output[index + 2] / 2);//bottom right x
				location[3] = (output[index + 1] + output[index + 3] / 2);//bottom right y
				locations.emplace_back(location);
				rect = cv::Rect(lround(location[0]), lround(location[1]),
					lround(location[2] - location[0]), lround(location[3] - location[1]));
				src_rects.push_back(rect);
				labels_.emplace_back(k - labelStartIndex);
				confidences.emplace_back(output[index + k]);
			}
		}
	}
	// run NMS
	std::vector<int> nms_indices;
	cv::dnn::NMSBoxes(src_rects, confidences, conf_threshold, iou_threshold, nms_indices);

	for (size_t i = 0; i < nms_indices.size(); i++) {
		Detection detection;
		detection.bbox = src_rects[nms_indices[i]];
		detection.score = confidences[nms_indices[i]];
		detection.class_idx = labels_[nms_indices[i]];
		det_vec.push_back(detection);
	}
	//pad_w is the left (and also right) border width in the square image feeded to the model
	ScaleCoordinates(det_vec, pad_w, pad_h, scale, img_shape);
	std::vector<std::vector<Detection>> detections;
	detections.reserve(batch_size);
	// save final detection for the current image
	detections.emplace_back(det_vec);
	return detections;
}


//if two boxes have an iou (intersection over union) that is two large, then they cannot represent two adjacent characters of the license plate 
//so we discard the one with the lowest confidence rate
void filter_iou(std::vector<int>& classIds,
	std::vector<float>& confidences,
	std::vector<cv::Rect>& vect_of_detected_boxes, const float& nmsThreshold
)
{
#ifdef _DEBUG
	assert(classIds.size() == confidences.size() && classIds.size() == vect_of_detected_boxes.size());
#endif //_DEBUG
	std::list<int> l_classIds;
	std::list<float> l_confidences;
	std::list<cv::Rect> list_of_detected_boxes;
	//first copy vector to list which are faster when we need to delete element
	std::copy(classIds.begin(), classIds.end(), std::back_inserter(l_classIds));
	std::copy(confidences.begin(), confidences.end(), std::back_inserter(l_confidences));
	std::copy(vect_of_detected_boxes.begin(), vect_of_detected_boxes.end(), std::back_inserter(list_of_detected_boxes));

	//if two boxes have an iou (intersection over union) that is two large, then they cannot represent two adjacent characters of the license plate 
	//so we discard the one with the lowest confidence rate
	filter_iou2(list_of_detected_boxes, l_confidences,
		l_classIds, nmsThreshold);

	if (classIds.size() > l_classIds.size()) {
		classIds.clear(); confidences.clear(); vect_of_detected_boxes.clear();
		std::copy(l_classIds.begin(), l_classIds.end(), std::back_inserter(classIds));
		std::copy(l_confidences.begin(), l_confidences.end(), std::back_inserter(confidences));
		std::copy(list_of_detected_boxes.begin(), list_of_detected_boxes.end(), std::back_inserter(vect_of_detected_boxes));
	}
}