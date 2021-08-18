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
#include"../include/utils_opencv.h"
#include <filesystem>
#define NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE 36
#define NUMBER_OF_COUNTRIES 61
//returns the iou (intersection over union) of two boxes
float iou(const cv::Rect& r1, const cv::Rect& r2)
{
	cv::Rect inter(get_inter(r1, r2));
	if (inter.width > 0 && inter.height > 0 && r1.width > 0 && r1.height > 0 && r2.width > 0 && r2.height > 0) {
		return (float)(inter.area()) / (float)(r1.area() + r2.area() - inter.area());
	}
	else return 0.0f;
}
//rearrange detected bounding boxes from left to right
void sort_from_left_to_right(std::list<cv::Rect>& boxes, std::list<float>& confidences, std::list<int>& classIds)
{
	std::list<cv::Rect> l_tri_left;
	std::list<float> confidences_tri_left;
	std::list<int> classIds_tri_left;
	while (!boxes.empty() && !confidences.empty() && !classIds.empty()) {
		//int left_courant(boxes.front().x);
		std::list<cv::Rect>::iterator it = l_tri_left.begin();
		std::list<int>::iterator it_classIds = classIds_tri_left.begin();
		std::list<float>::iterator it_confidences = confidences_tri_left.begin();
		while (it != l_tri_left.end() && it_confidences != confidences_tri_left.end() &&
			it_classIds != classIds_tri_left.end()) {
			if (is_on_the_left(boxes.front(), *it))
				break;
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
	assert(debug_left(l_tri_left));
#endif	  //_DEBUG
	l_tri_left.swap(boxes);
	confidences_tri_left.swap(confidences);
	classIds_tri_left.swap(classIds);
}//return the intersection rect of the rectangles
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
//return true if the intersection of the first argument box and the second has an interect area that is at least 90% of the box (which means box is nearly entirely in the second argument)
bool is_in_rect_if(const cv::Rect& box, const cv::Rect& rect_im, const float min_area_ratio)
{
	cv::Rect inter(get_inter(box, rect_im));
	if (inter.width > 0 && inter.height > 0 && box.width > 0 && box.height > 0 && rect_im.width > 0 && rect_im.height > 0) {
		return ((float)(inter.area()) / (float)(box.area()) > min_area_ratio);
	}
	else return false;
}
//return true if the intersection of the first argument box and the second has an interect area that is at least 90% of the box (which means box is nearly entirely in the second argument)
bool is_in_rect_if(const cv::Rect& box, const cv::Rect& rect_im)
{
	const float min_area_ratio = 0.9f;
	return is_in_rect_if(box, rect_im, min_area_ratio);
}
///from all heights of the boxes, get the median value
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
//if two boxes have an iou (intersection over union) that is too large, then they cannot represent two adjacent characters of the license plate
//so we discard the one with the lowest confidence rate
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
//get barycenters of a list of bounding boxes.
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
//checks if the bounding boxes are ordered from left to right
#ifdef _DEBUG
			//cette fonction verifie que la liste est triee de gauche a droite 
bool debug_left(const std::list<cv::Rect>& boxes) {
	if (boxes.size() < 2) return true;
	else {
		//VERIFICATION DU TRI CROISSANT
		std::list<cv::Rect>::const_iterator it_verif = boxes.begin();
		//int pred = it_verif->x;
		cv::Rect pred_box = *it_verif;
		it_verif++;
		while (it_verif != boxes.end()) {
#ifdef _DEBUG
			//assert(it_verif->x >= pred);
			assert(is_on_the_left(pred_box, *it_verif));
#endif //_DEBUG
			if (!is_on_the_left(pred_box, *it_verif) && pred_box != *it_verif) return false;
			pred_box = *it_verif;
			it_verif++;
		}
		return true;
	}
}
#endif //_DEBUG
//examines how boxes are disposed and filter out boxes with a position that are incompatible with the positions of other boxes
void filtre_grubbs_sides(const std::list<cv::Rect>& boxes, std::list<float>& angles_with_horizontal_line,
	float& mean_angles_par_rapport_horiz,
	float& standard_deviation_consecutives_angles,
	std::list<int>& interdistances,
	float& mean_interdistances,
	float& standard_deviation_interdistances,
	float& mean_produit_interdistance_avec_angle, float& standard_deviation_produit_interdistance_avec_angle)
{
	//cette fonction verifie que la liste est triee de gauche a droite 
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
//characters on a license plate can be disposed on two lines (bi level) or on just one unique line (single level).
//anycase the ascii characters and there bouding boxes must nbe ordered in the inthe same way of the registration ascii chain.
void is_bi_level_plate(const std::list<cv::Rect>& boxes, const std::list<float>& l_confidences, const std::list<int>& l_classIds,
	std::list<cv::Rect>& l_reordered, std::list<float>& l_reordered_confidences, std::list<int>& l_reordered_classIds, std::list<int>& levels)
{
	if (boxes.size() > 1) {
		//cette fonction trie la liste de gauche a droite 
		std::copy(boxes.begin(), boxes.end(), std::back_inserter(l_reordered));
		std::copy(l_confidences.begin(), l_confidences.end(), std::back_inserter(l_reordered_confidences));
		std::copy(l_classIds.begin(), l_classIds.end(), std::back_inserter(l_reordered_classIds));
		sort_from_left_to_right(l_reordered, l_reordered_confidences, l_reordered_classIds);
		//la difference des angles des n-1 premiers characteres par rapport l'horizontale
							//l'interdistance de chaque charactere avec son suivant pour les n-1 premiers characteres
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
					assert(iou(box_left, box_right) < 0.54);
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
					assert(iou(box_left, box_right) < 0.54);
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
}//checks if the character of the lpn is a digit or a letter or something else (by ex examples misread carac or license plate bounding box, ...)
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
void filter_out_everything_but_characters(std::list<cv::Rect>& boxes,
	std::list<float>& l_confidences, std::list<int>& l_classIds)
{
	//***************************************************
		//                  FILTER
		//***************************************************
	std::list<float>::iterator it_confidences(l_confidences.begin());
	std::list<cv::Rect>::iterator it_boxes(boxes.begin());
	std::list<int>::iterator it_out_classes(l_classIds.begin());
	while (it_out_classes != l_classIds.end()
		&& it_confidences != l_confidences.end() && it_boxes != boxes.end()) {
		if (!(*it_out_classes < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1 
			&& *it_out_classes >= 0)) {
			it_out_classes = l_classIds.erase(it_out_classes);
			it_confidences = l_confidences.erase(it_confidences);
			it_boxes = boxes.erase(it_boxes);
		}
		else {
			it_out_classes++;
			it_confidences++;
			it_boxes++;
		}
	}
}
//given the index of a bounding box, we can predict if this box is a single character or if it represents the license plate area or if it is the roi of an entire vehicle
//single character--> returns 1
//license plate--> returns 2
//vehicle--> returns 3
//negative index--> returns 0 must be an error
//classId_last_country : is the class index of the last country in the list of detected classes.
int is_this_box_a_character_a_license_plate_or_a_vehicle(const int classId, const int classId_last_country)
{
	if (classId == -1) return 3;
	else if (classId < 0) return 0;
	else if (classId < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1
		) return 1;
	else if (classId <= classId_last_country) return 2;
	else return 3;
}
//given the index of a bounding box, we can predict if this box is a single character or if it represents the license plate area or if it is the roi of an entire vehicle
//single character--> returns 1
//license plate--> returns 2
//negative index--> returns 0 must be an error
//classId_last_country : is the class index of the last country in the list of detected classes.
int is_this_box_a_character(const int classId, const int number_of_characters_latin_numberplate)
{
	if (classId == -1) return 3;
	else if (classId < 0) return 0;
	else if (classId < number_of_characters_latin_numberplate //- 1
		) return 1;
	else return 2;
}
//groups detected boxes that correspond to the same vehicle. The separation is based on raw detections of license plates from the dnn
//it can deal with license pates that have two lines of charcaters
//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
			//groups detected boxes that correspond to the same vehicle. The separation is based on raw detections of license plates from the dnn
void group_characters_in_the_same_license_plate(
	//raw detections
	const std::vector<cv::Rect>& boxes,
	const std::vector<float>& confidences, const std::vector<int>& classIds,
	//detections of same lp are regrouped in a vector
	std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate, std::list < std::vector<int>>& l_vect_of_classIds_in_a_license_plate
	, const int classId_last_country//classId_last_country : is the class index of the last country in the list of detected classes.
) {
	//first find license plates
	std::list<cv::Rect> license_plates;
	std::list<float> license_plates_confidences;
	std::list<int> license_plates_classIds;
	std::vector<float>::const_iterator it_confidences(confidences.begin());
	std::vector<cv::Rect>::const_iterator it_boxes(boxes.begin());
	std::vector<int>::const_iterator it_out_classes(classIds.begin());
	while (it_out_classes != classIds.end()
		&& it_confidences != confidences.end() && it_boxes != boxes.end()) {
		//given the index of a bounding box, we can predict if this box is a single character or if it represents the license plate area or if it is the roi of an entire vehicle
		//single character--> returns 1
		//license plate--> returns 2
		//vehicle--> returns 3
		//negative index--> returns 0 must be an error
		//classId_last_country : is the class index of the last country in the list of detected classes.
		if (is_this_box_a_character_a_license_plate_or_a_vehicle(*it_out_classes, classId_last_country) == 2) {
			license_plates_confidences.push_back(*it_confidences);
			license_plates.push_back(*it_boxes);
			license_plates_classIds.push_back(*it_out_classes);
		}
		it_out_classes++;
		it_confidences++;
		it_boxes++;
	}
	//second associate to each license plate a vehicle
	std::list<cv::Rect> vehicles;
	std::list<float> vehicles_confidences;
	std::list<int> vehicles_classIds;
	std::list<cv::Rect>::const_iterator it_license_plates(license_plates.begin());
	std::list<float>::const_iterator it_license_plates_confidence(license_plates_confidences.begin());
	std::list<int>::const_iterator it_license_plates_classIds(license_plates_classIds.begin());
	while (it_license_plates != license_plates.end() && it_license_plates_confidence != license_plates_confidences.end()
		&& it_license_plates_classIds != license_plates_classIds.end()
		) {
		it_confidences = (confidences.begin());
		it_boxes = (boxes.begin());
		it_out_classes = (classIds.begin());
		while (it_out_classes != classIds.end()
			&& it_confidences != confidences.end() && it_boxes != boxes.end()) {
#ifdef _DEBUG		
			assert(vehicles_confidences.size() == vehicles.size() && vehicles_confidences.size() == vehicles_classIds.size());
#endif //_DEBUG
			if (is_in_rect_if(*it_license_plates, *it_boxes) && (*it_license_plates_classIds != *it_out_classes)
				&& *it_out_classes > classId_last_country) {
				if (vehicles_confidences.empty()) {
					vehicles_confidences.push_back(*it_confidences);
					vehicles.push_back(*it_boxes);
					vehicles_classIds.push_back(*it_out_classes);
				}
				else {
					if (*it_confidences > vehicles_confidences.back()) {
						vehicles_confidences.pop_back();
						vehicles.pop_back(); vehicles_classIds.pop_back();
						vehicles_confidences.push_back(*it_confidences);
						vehicles.push_back(*it_boxes); vehicles_classIds.push_back(*it_out_classes);
					}
				}
			}
			it_out_classes++;
			it_confidences++;
			it_boxes++;
		}
		if (vehicles_confidences.empty()) {
#ifdef _DEBUG		
			assert(vehicles_confidences.size() == vehicles.size() && vehicles_confidences.size() == vehicles_classIds.size());
#endif //_DEBUG
			vehicles_confidences.push_back(0.0f);
			vehicles.push_back(cv::Rect(0, 0, 0, 0));
			vehicles_classIds.push_back(-1);//index ==-1 suggests that there is in fact no vehicle for this plate
		}
		it_license_plates_confidence++;
		it_license_plates++;
		it_license_plates_classIds++;
	}
	//now creates a vect of characters for each license plate that have been detected
	it_license_plates = (license_plates.begin());
	it_license_plates_confidence = (license_plates_confidences.begin());
	it_license_plates_classIds = (license_plates_classIds.begin());
	std::list<cv::Rect>::const_iterator it_vehicles(vehicles.begin());
	std::list<float>::const_iterator it_vehicles_confidence(vehicles_confidences.begin());
	std::list<int>::const_iterator it_vehicles_classIds(vehicles_classIds.begin());
	while (it_license_plates != license_plates.end() && it_license_plates_confidence != license_plates_confidences.end()
		&& it_license_plates_classIds != license_plates_classIds.end()
		&& it_vehicles != vehicles.end() && it_vehicles_confidence != vehicles_confidences.end()
		&& it_vehicles_classIds != vehicles_classIds.end()
		) {
		l_vect_of_boxes_in_a_license_plate.push_back(std::vector<cv::Rect>());
		l_vect_of_confidences_in_a_license_plate.push_back(std::vector<float>());
		l_vect_of_classIds_in_a_license_plate.push_back(std::vector<int>());
		l_vect_of_boxes_in_a_license_plate.back().push_back(*it_license_plates);
		l_vect_of_confidences_in_a_license_plate.back().push_back(*it_license_plates_confidence);
		l_vect_of_classIds_in_a_license_plate.back().push_back(*it_license_plates_classIds);
		l_vect_of_boxes_in_a_license_plate.back().push_back(*it_vehicles);
		l_vect_of_confidences_in_a_license_plate.back().push_back(*it_vehicles_confidence);
		l_vect_of_classIds_in_a_license_plate.back().push_back(*it_vehicles_classIds);
		it_license_plates_confidence++;
		it_license_plates++;
		it_license_plates_classIds++;
		it_vehicles_confidence++;
		it_vehicles++;
		it_vehicles_classIds++;
	}
	//now associate each character to license plates
	it_confidences = (confidences.begin());
	it_boxes = (boxes.begin());
	it_out_classes = (classIds.begin());
	while (it_out_classes != classIds.end()
		&& it_confidences != confidences.end() && it_boxes != boxes.end()) {
		if (*it_out_classes < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1
			) {
			std::list < std::vector<cv::Rect>>::iterator it_l_vect_of_boxes_in_a_license_plate(l_vect_of_boxes_in_a_license_plate.begin());
			std::list < std::vector<float>>::iterator it_l_vect_of_confidences_in_a_license_plate(l_vect_of_confidences_in_a_license_plate.begin());
			std::list < std::vector<int>>::iterator it_l_vect_of_classIds_in_a_license_plate(l_vect_of_classIds_in_a_license_plate.begin());
			while (it_l_vect_of_boxes_in_a_license_plate != l_vect_of_boxes_in_a_license_plate.end()
				&& it_l_vect_of_confidences_in_a_license_plate != l_vect_of_confidences_in_a_license_plate.end() && it_l_vect_of_classIds_in_a_license_plate != l_vect_of_classIds_in_a_license_plate.end()) {
#ifdef _DEBUG		
				assert(it_l_vect_of_boxes_in_a_license_plate->size() == it_l_vect_of_confidences_in_a_license_plate->size() && it_l_vect_of_classIds_in_a_license_plate->size() == it_l_vect_of_boxes_in_a_license_plate->size()
					&& it_l_vect_of_boxes_in_a_license_plate->size() >= 2);
#endif //_DEBUG
				//front is license plate
				if (is_in_rect_if(*it_boxes, it_l_vect_of_boxes_in_a_license_plate->front(), 0.5f)) {
					it_l_vect_of_boxes_in_a_license_plate->push_back(*it_boxes);
					it_l_vect_of_confidences_in_a_license_plate->push_back(*it_confidences);
					it_l_vect_of_classIds_in_a_license_plate->push_back(*it_out_classes);
				}
				it_l_vect_of_boxes_in_a_license_plate++;
				it_l_vect_of_confidences_in_a_license_plate++;
				it_l_vect_of_classIds_in_a_license_plate++;
			}
		}
		it_out_classes++;
		it_confidences++;
		it_boxes++;
	}
}
//groups detected boxes that correspond to the same vehicle. The separation is based on raw detections of license plates from the dnn
//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
//it can deal with license pates that have two lines of charcaters
void group_characters_in_the_same_license_plate(
	//raw detections
	const std::vector<cv::Rect>& boxes,
	const std::vector<float>& confidences, const std::vector<int>& classIds,
	//detections of same lp are regrouped in a vector
	std::list < std::list<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::list<float>>& l_vect_of_confidences_in_a_license_plate, std::list < std::list<int>>& l_vect_of_classIds_in_a_license_plate
	, const int classId_last_country//classId_last_country : is the class index of the last country in the list of detected classes.
) {
	//first find license plates
	std::list<cv::Rect> license_plates;
	std::list<float> license_plates_confidences;
	std::list<int> license_plates_classIds;
	std::vector<float>::const_iterator it_confidences(confidences.begin());
	std::vector<cv::Rect>::const_iterator it_boxes(boxes.begin());
	std::vector<int>::const_iterator it_out_classes(classIds.begin());
	while (it_out_classes != classIds.end()
		&& it_confidences != confidences.end() && it_boxes != boxes.end()) {
		//given the index of a bounding box, we can predict if this box is a single character or if it represents the license plate area or if it is the roi of an entire vehicle
		//single character--> returns 1
		//license plate--> returns 2
		//vehicle--> returns 3
		//negative index--> returns 0 must be an error
		if (is_this_box_a_character_a_license_plate_or_a_vehicle(*it_out_classes, classId_last_country) == 2) {
			license_plates_confidences.push_back(*it_confidences);
			license_plates.push_back(*it_boxes);
			license_plates_classIds.push_back(*it_out_classes);
		}
		it_out_classes++;
		it_confidences++;
		it_boxes++;
	}
	//second associate to each license plate a vehicle
	std::list<cv::Rect> vehicles;
	std::list<float> vehicles_confidences;
	std::list<int> vehicles_classIds;
	std::list<cv::Rect>::const_iterator it_license_plates(license_plates.begin());
	std::list<float>::const_iterator it_license_plates_confidence(license_plates_confidences.begin());
	std::list<int>::const_iterator it_license_plates_classIds(license_plates_classIds.begin());
	while (it_license_plates != license_plates.end() && it_license_plates_confidence != license_plates_confidences.end()
		&& it_license_plates_classIds != license_plates_classIds.end()
		) {
		it_confidences = (confidences.begin());
		it_boxes = (boxes.begin());
		it_out_classes = (classIds.begin());
		while (it_out_classes != classIds.end()
			&& it_confidences != confidences.end() && it_boxes != boxes.end()) {
#ifdef _DEBUG		
			assert(vehicles_confidences.size() == vehicles.size() && vehicles_confidences.size() == vehicles_classIds.size());
#endif //_DEBUG
			if (is_in_rect_if(*it_license_plates, *it_boxes) && (*it_license_plates_classIds != *it_out_classes)
				&& *it_out_classes > classId_last_country) {
				if (vehicles_confidences.empty()) {
					vehicles_confidences.push_back(*it_confidences);
					vehicles.push_back(*it_boxes);
					vehicles_classIds.push_back(*it_out_classes);
				}
				else {
					if (*it_confidences > vehicles_confidences.back()) {
						vehicles_confidences.pop_back();
						vehicles.pop_back(); vehicles_classIds.pop_back();
						vehicles_confidences.push_back(*it_confidences);
						vehicles.push_back(*it_boxes); vehicles_classIds.push_back(*it_out_classes);
					}
				}
			}
			it_out_classes++;
			it_confidences++;
			it_boxes++;
		}
		if (vehicles_confidences.empty()) {
#ifdef _DEBUG		
			assert(vehicles_confidences.size() == vehicles.size() && vehicles_confidences.size() == vehicles_classIds.size());
#endif //_DEBUG
			vehicles_confidences.push_back(0.0f);
			vehicles.push_back(cv::Rect(0, 0, 0, 0));
			vehicles_classIds.push_back(-1);//index ==-1 suggests that there is in fact no vehicle for this plate
		}
		it_license_plates_confidence++;
		it_license_plates++;
		it_license_plates_classIds++;
	}
	//now creates a vect of characters for each license plate that have been detected
	it_license_plates = (license_plates.begin());
	it_license_plates_confidence = (license_plates_confidences.begin());
	it_license_plates_classIds = (license_plates_classIds.begin());
	std::list<cv::Rect>::const_iterator it_vehicles(vehicles.begin());
	std::list<float>::const_iterator it_vehicles_confidence(vehicles_confidences.begin());
	std::list<int>::const_iterator it_vehicles_classIds(vehicles_classIds.begin());
	while (it_license_plates != license_plates.end() && it_license_plates_confidence != license_plates_confidences.end()
		&& it_license_plates_classIds != license_plates_classIds.end()
		&& it_vehicles != vehicles.end() && it_vehicles_confidence != vehicles_confidences.end()
		&& it_vehicles_classIds != vehicles_classIds.end()
		) {
		l_vect_of_boxes_in_a_license_plate.push_back(std::list<cv::Rect>());
		l_vect_of_confidences_in_a_license_plate.push_back(std::list<float>());
		l_vect_of_classIds_in_a_license_plate.push_back(std::list<int>());
		l_vect_of_boxes_in_a_license_plate.back().push_back(*it_license_plates);
		l_vect_of_confidences_in_a_license_plate.back().push_back(*it_license_plates_confidence);
		l_vect_of_classIds_in_a_license_plate.back().push_back(*it_license_plates_classIds);
		l_vect_of_boxes_in_a_license_plate.back().push_back(*it_vehicles);
		l_vect_of_confidences_in_a_license_plate.back().push_back(*it_vehicles_confidence);
		l_vect_of_classIds_in_a_license_plate.back().push_back(*it_vehicles_classIds);
		it_license_plates_confidence++;
		it_license_plates++;
		it_license_plates_classIds++;
		it_vehicles_confidence++;
		it_vehicles++;
		it_vehicles_classIds++;
	}
	//now associate each character to license plates
	it_confidences = (confidences.begin());
	it_boxes = (boxes.begin());
	it_out_classes = (classIds.begin());
	while (it_out_classes != classIds.end()
		&& it_confidences != confidences.end() && it_boxes != boxes.end()) {
		if (*it_out_classes < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1
			) {
			std::list < std::list<cv::Rect>>::iterator it_l_vect_of_boxes_in_a_license_plate(l_vect_of_boxes_in_a_license_plate.begin());
			std::list < std::list<float>>::iterator it_l_vect_of_confidences_in_a_license_plate(l_vect_of_confidences_in_a_license_plate.begin());
			std::list < std::list<int>>::iterator it_l_vect_of_classIds_in_a_license_plate(l_vect_of_classIds_in_a_license_plate.begin());
			while (it_l_vect_of_boxes_in_a_license_plate != l_vect_of_boxes_in_a_license_plate.end()
				&& it_l_vect_of_confidences_in_a_license_plate != l_vect_of_confidences_in_a_license_plate.end() && it_l_vect_of_classIds_in_a_license_plate != l_vect_of_classIds_in_a_license_plate.end()) {
#ifdef _DEBUG		
				assert(it_l_vect_of_boxes_in_a_license_plate->size() == it_l_vect_of_confidences_in_a_license_plate->size() && it_l_vect_of_classIds_in_a_license_plate->size() == it_l_vect_of_boxes_in_a_license_plate->size()
					&& it_l_vect_of_boxes_in_a_license_plate->size() >= 2);
#endif //_DEBUG
				//front is license plate
				if (is_in_rect_if(*it_boxes, it_l_vect_of_boxes_in_a_license_plate->front(), 0.5f)) {
					it_l_vect_of_boxes_in_a_license_plate->push_back(*it_boxes);
					it_l_vect_of_confidences_in_a_license_plate->push_back(*it_confidences);
					it_l_vect_of_classIds_in_a_license_plate->push_back(*it_out_classes);
				}
				it_l_vect_of_boxes_in_a_license_plate++;
				it_l_vect_of_confidences_in_a_license_plate++;
				it_l_vect_of_classIds_in_a_license_plate++;
			}
		}
		it_out_classes++;
		it_confidences++;
		it_boxes++;
	}
}
//groups detected boxes that correspond to the same vehicle. The separation is based on raw detections of license plates from the dnn
//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
//it can deal with license pates that have two lines of charcaters
void group_characters_in_the_same_license_plate(
	//raw detections
	const std::list<cv::Rect>& boxes,
	const std::list<float>& confidences, const std::list<int>& classIds,
	//detections of same lp are regrouped in a vector
	std::list < std::list<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::list<float>>& l_vect_of_confidences_in_a_license_plate, std::list < std::list<int>>& l_vect_of_classIds_in_a_license_plate
	, const int classId_last_country//classId_last_country : is the class index of the last country in the list of detected classes.
) {
	//first find license plates
	std::list<cv::Rect> license_plates;
	std::list<float> license_plates_confidences;
	std::list<int> license_plates_classIds;
	std::list<float>::const_iterator it_confidences(confidences.begin());
	std::list<cv::Rect>::const_iterator it_boxes(boxes.begin());
	std::list<int>::const_iterator it_out_classes(classIds.begin());
	while (it_out_classes != classIds.end()
		&& it_confidences != confidences.end() && it_boxes != boxes.end()) {
		//given the index of a bounding box, we can predict if this box is a single character or if it represents the license plate area or if it is the roi of an entire vehicle
		//single character--> returns 1
		//license plate--> returns 2
		//vehicle--> returns 3
		//negative index--> returns 0 must be an error
		if (is_this_box_a_character_a_license_plate_or_a_vehicle(*it_out_classes, classId_last_country) == 2) {
			license_plates_confidences.push_back(*it_confidences);
			license_plates.push_back(*it_boxes);
			license_plates_classIds.push_back(*it_out_classes);
		}
		it_out_classes++;
		it_confidences++;
		it_boxes++;
	}
	//second associate to each license plate a vehicle
	std::list<cv::Rect> vehicles;
	std::list<float> vehicles_confidences;
	std::list<int> vehicles_classIds;
	std::list<cv::Rect>::const_iterator it_license_plates(license_plates.begin());
	std::list<float>::const_iterator it_license_plates_confidence(license_plates_confidences.begin());
	std::list<int>::const_iterator it_license_plates_classIds(license_plates_classIds.begin());
	while (it_license_plates != license_plates.end() && it_license_plates_confidence != license_plates_confidences.end()
		&& it_license_plates_classIds != license_plates_classIds.end()
		) {
		it_confidences = (confidences.begin());
		it_boxes = (boxes.begin());
		it_out_classes = (classIds.begin());
		while (it_out_classes != classIds.end()
			&& it_confidences != confidences.end() && it_boxes != boxes.end()) {
#ifdef _DEBUG		
			assert(vehicles_confidences.size() == vehicles.size() && vehicles_confidences.size() == vehicles_classIds.size());
#endif //_DEBUG
			if (is_in_rect_if(*it_license_plates, *it_boxes) && (*it_license_plates_classIds != *it_out_classes)
				&& *it_out_classes > classId_last_country) {
				if (vehicles_confidences.empty()) {
					vehicles_confidences.push_back(*it_confidences);
					vehicles.push_back(*it_boxes);
					vehicles_classIds.push_back(*it_out_classes);
				}
				else {
					if (*it_confidences > vehicles_confidences.back()) {
						vehicles_confidences.pop_back();
						vehicles.pop_back(); vehicles_classIds.pop_back();
						vehicles_confidences.push_back(*it_confidences);
						vehicles.push_back(*it_boxes); vehicles_classIds.push_back(*it_out_classes);
					}
				}
			}
			it_out_classes++;
			it_confidences++;
			it_boxes++;
		}
		if (vehicles_confidences.empty()) {
#ifdef _DEBUG		
			assert(vehicles_confidences.size() == vehicles.size() && vehicles_confidences.size() == vehicles_classIds.size());
#endif //_DEBUG
			vehicles_confidences.push_back(0.0f);
			vehicles.push_back(cv::Rect(0, 0, 0, 0));
			vehicles_classIds.push_back(-1);//index ==-1 suggests that there is in fact no vehicle for this plate
		}
		it_license_plates_confidence++;
		it_license_plates++;
		it_license_plates_classIds++;
	}
	//now creates a vect of characters for each license plate that have been detected
	it_license_plates = (license_plates.begin());
	it_license_plates_confidence = (license_plates_confidences.begin());
	it_license_plates_classIds = (license_plates_classIds.begin());
	std::list<cv::Rect>::const_iterator it_vehicles(vehicles.begin());
	std::list<float>::const_iterator it_vehicles_confidence(vehicles_confidences.begin());
	std::list<int>::const_iterator it_vehicles_classIds(vehicles_classIds.begin());
	while (it_license_plates != license_plates.end() && it_license_plates_confidence != license_plates_confidences.end()
		&& it_license_plates_classIds != license_plates_classIds.end()
		&& it_vehicles != vehicles.end() && it_vehicles_confidence != vehicles_confidences.end()
		&& it_vehicles_classIds != vehicles_classIds.end()
		) {
		l_vect_of_boxes_in_a_license_plate.push_back(std::list<cv::Rect>());
		l_vect_of_confidences_in_a_license_plate.push_back(std::list<float>());
		l_vect_of_classIds_in_a_license_plate.push_back(std::list<int>());
		l_vect_of_boxes_in_a_license_plate.back().push_back(*it_license_plates);
		l_vect_of_confidences_in_a_license_plate.back().push_back(*it_license_plates_confidence);
		l_vect_of_classIds_in_a_license_plate.back().push_back(*it_license_plates_classIds);
		l_vect_of_boxes_in_a_license_plate.back().push_back(*it_vehicles);
		l_vect_of_confidences_in_a_license_plate.back().push_back(*it_vehicles_confidence);
		l_vect_of_classIds_in_a_license_plate.back().push_back(*it_vehicles_classIds);
		it_license_plates_confidence++;
		it_license_plates++;
		it_license_plates_classIds++;
		it_vehicles_confidence++;
		it_vehicles++;
		it_vehicles_classIds++;
	}
	//now associate each character to license plates
	it_confidences = (confidences.begin());
	it_boxes = (boxes.begin());
	it_out_classes = (classIds.begin());
	while (it_out_classes != classIds.end()
		&& it_confidences != confidences.end() && it_boxes != boxes.end()) {
		if (*it_out_classes < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1
			) {
			std::list < std::list<cv::Rect>>::iterator it_l_vect_of_boxes_in_a_license_plate(l_vect_of_boxes_in_a_license_plate.begin());
			std::list < std::list<float>>::iterator it_l_vect_of_confidences_in_a_license_plate(l_vect_of_confidences_in_a_license_plate.begin());
			std::list < std::list<int>>::iterator it_l_vect_of_classIds_in_a_license_plate(l_vect_of_classIds_in_a_license_plate.begin());
			while (it_l_vect_of_boxes_in_a_license_plate != l_vect_of_boxes_in_a_license_plate.end()
				&& it_l_vect_of_confidences_in_a_license_plate != l_vect_of_confidences_in_a_license_plate.end() && it_l_vect_of_classIds_in_a_license_plate != l_vect_of_classIds_in_a_license_plate.end()) {
#ifdef _DEBUG		
				assert(it_l_vect_of_boxes_in_a_license_plate->size() == it_l_vect_of_confidences_in_a_license_plate->size() && it_l_vect_of_classIds_in_a_license_plate->size() == it_l_vect_of_boxes_in_a_license_plate->size()
					&& it_l_vect_of_boxes_in_a_license_plate->size() >= 2);
#endif //_DEBUG
				//front is license plate
				if (is_in_rect_if(*it_boxes, it_l_vect_of_boxes_in_a_license_plate->front(), 0.5f)) {
					it_l_vect_of_boxes_in_a_license_plate->push_back(*it_boxes);
					it_l_vect_of_confidences_in_a_license_plate->push_back(*it_confidences);
					it_l_vect_of_classIds_in_a_license_plate->push_back(*it_out_classes);
				}
				it_l_vect_of_boxes_in_a_license_plate++;
				it_l_vect_of_confidences_in_a_license_plate++;
				it_l_vect_of_classIds_in_a_license_plate++;
			}
		}
		it_out_classes++;
		it_confidences++;
		it_boxes++;
	}
}
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
void group_characters_in_the_same_license_plate(
	//raw detections
	const std::list<cv::Rect>& boxes, const std::list<float>& confidences, const std::list<int>& classIds,
	//detections of same lp are regrouped in a vector
	std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate, std::list < std::vector<int>>& l_vect_of_classIds_in_a_license_plate, const int classId_last_country
	//classId_last_country : is the class index of the last country in the list of detected classes.
) {
	//first find license plates
	std::list<cv::Rect> license_plates;
	std::list<float> license_plates_confidences;
	std::list<int> license_plates_classIds;
	std::list<float>::const_iterator it_confidences(confidences.begin());
	std::list<cv::Rect>::const_iterator it_boxes(boxes.begin());
	std::list<int>::const_iterator it_out_classes(classIds.begin());
	while (it_out_classes != classIds.end()
		&& it_confidences != confidences.end() && it_boxes != boxes.end()) {
		//given the index of a bounding box, we can predict if this box is a single character or if it represents the license plate area or if it is the roi of an entire vehicle
		//single character--> returns 1
		//license plate--> returns 2
		//vehicle--> returns 3
		//negative index--> returns 0 must be an error
		if (is_this_box_a_character_a_license_plate_or_a_vehicle(*it_out_classes, classId_last_country) == 2) {
			license_plates_confidences.push_back(*it_confidences);
			license_plates.push_back(*it_boxes);
			license_plates_classIds.push_back(*it_out_classes);
		}
		it_out_classes++;
		it_confidences++;
		it_boxes++;
	}
	//second associate to each license plate a vehicle
	std::list<cv::Rect> vehicles;
	std::list<float> vehicles_confidences;
	std::list<int> vehicles_classIds;
	std::list<cv::Rect>::const_iterator it_license_plates(license_plates.begin());
	std::list<float>::const_iterator it_license_plates_confidence(license_plates_confidences.begin());
	std::list<int>::const_iterator it_license_plates_classIds(license_plates_classIds.begin());
	while (it_license_plates != license_plates.end() && it_license_plates_confidence != license_plates_confidences.end()
		&& it_license_plates_classIds != license_plates_classIds.end()
		) {
		it_confidences = (confidences.begin());
		it_boxes = (boxes.begin());
		it_out_classes = (classIds.begin());
		while (it_out_classes != classIds.end()
			&& it_confidences != confidences.end() && it_boxes != boxes.end()) {
#ifdef _DEBUG		
			assert(vehicles_confidences.size() == vehicles.size() && vehicles_confidences.size() == vehicles_classIds.size());
#endif //_DEBUG
			if (is_in_rect_if(*it_license_plates, *it_boxes) && (*it_license_plates_classIds != *it_out_classes)
				&& *it_out_classes > classId_last_country) {
				if (vehicles_confidences.empty()) {
					vehicles_confidences.push_back(*it_confidences);
					vehicles.push_back(*it_boxes);
					vehicles_classIds.push_back(*it_out_classes);
				}
				else {
					if (*it_confidences > vehicles_confidences.back()) {
						vehicles_confidences.pop_back();
						vehicles.pop_back(); vehicles_classIds.pop_back();
						vehicles_confidences.push_back(*it_confidences);
						vehicles.push_back(*it_boxes); vehicles_classIds.push_back(*it_out_classes);
					}
				}
			}
			it_out_classes++;
			it_confidences++;
			it_boxes++;
		}
		if (vehicles_confidences.empty()) {
#ifdef _DEBUG		
			assert(vehicles_confidences.size() == vehicles.size() && vehicles_confidences.size() == vehicles_classIds.size());
#endif //_DEBUG
			vehicles_confidences.push_back(0.0f);
			vehicles.push_back(cv::Rect(0, 0, 0, 0));
			vehicles_classIds.push_back(-1);//index ==-1 suggests that there is in fact no vehicle for this plate
		}
		it_license_plates_confidence++;
		it_license_plates++;
		it_license_plates_classIds++;
	}
	//now creates a vect of characters for each license plate that have been detected
	it_license_plates = (license_plates.begin());
	it_license_plates_confidence = (license_plates_confidences.begin());
	it_license_plates_classIds = (license_plates_classIds.begin());
	std::list<cv::Rect>::const_iterator it_vehicles(vehicles.begin());
	std::list<float>::const_iterator it_vehicles_confidence(vehicles_confidences.begin());
	std::list<int>::const_iterator it_vehicles_classIds(vehicles_classIds.begin());
	while (it_license_plates != license_plates.end() && it_license_plates_confidence != license_plates_confidences.end()
		&& it_license_plates_classIds != license_plates_classIds.end()
		&& it_vehicles != vehicles.end() && it_vehicles_confidence != vehicles_confidences.end()
		&& it_vehicles_classIds != vehicles_classIds.end()
		) {
		l_vect_of_boxes_in_a_license_plate.push_back(std::vector<cv::Rect>());
		l_vect_of_confidences_in_a_license_plate.push_back(std::vector<float>());
		l_vect_of_classIds_in_a_license_plate.push_back(std::vector<int>());
		l_vect_of_boxes_in_a_license_plate.back().push_back(*it_license_plates);
		l_vect_of_confidences_in_a_license_plate.back().push_back(*it_license_plates_confidence);
		l_vect_of_classIds_in_a_license_plate.back().push_back(*it_license_plates_classIds);
		l_vect_of_boxes_in_a_license_plate.back().push_back(*it_vehicles);
		l_vect_of_confidences_in_a_license_plate.back().push_back(*it_vehicles_confidence);
		l_vect_of_classIds_in_a_license_plate.back().push_back(*it_vehicles_classIds);
		it_license_plates_confidence++;
		it_license_plates++;
		it_license_plates_classIds++;
		it_vehicles_confidence++;
		it_vehicles++;
		it_vehicles_classIds++;
	}
	//now associate each character to license plates
	it_confidences = (confidences.begin());
	it_boxes = (boxes.begin());
	it_out_classes = (classIds.begin());
	while (it_out_classes != classIds.end()
		&& it_confidences != confidences.end() && it_boxes != boxes.end()) {
		if (*it_out_classes < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1
			) {
			std::list < std::vector<cv::Rect>>::iterator it_l_vect_of_boxes_in_a_license_plate(l_vect_of_boxes_in_a_license_plate.begin());
			std::list < std::vector<float>>::iterator it_l_vect_of_confidences_in_a_license_plate(l_vect_of_confidences_in_a_license_plate.begin());
			std::list < std::vector<int>>::iterator it_l_vect_of_classIds_in_a_license_plate(l_vect_of_classIds_in_a_license_plate.begin());
			while (it_l_vect_of_boxes_in_a_license_plate != l_vect_of_boxes_in_a_license_plate.end()
				&& it_l_vect_of_confidences_in_a_license_plate != l_vect_of_confidences_in_a_license_plate.end() && it_l_vect_of_classIds_in_a_license_plate != l_vect_of_classIds_in_a_license_plate.end()) {
#ifdef _DEBUG		
				assert(it_l_vect_of_boxes_in_a_license_plate->size() == it_l_vect_of_confidences_in_a_license_plate->size() && it_l_vect_of_classIds_in_a_license_plate->size() == it_l_vect_of_boxes_in_a_license_plate->size()
					&& it_l_vect_of_boxes_in_a_license_plate->size() >= 2);
#endif //_DEBUG
				//front is license plate
				if (is_in_rect_if(*it_boxes, it_l_vect_of_boxes_in_a_license_plate->front(), 0.5f)) {
					it_l_vect_of_boxes_in_a_license_plate->push_back(*it_boxes);
					it_l_vect_of_confidences_in_a_license_plate->push_back(*it_confidences);
					it_l_vect_of_classIds_in_a_license_plate->push_back(*it_out_classes);
				}
				it_l_vect_of_boxes_in_a_license_plate++;
				it_l_vect_of_confidences_in_a_license_plate++;
				it_l_vect_of_classIds_in_a_license_plate++;
			}
		}
		it_out_classes++;
		it_confidences++;
		it_boxes++;
	}
}
void substitute(const int index, const int classId, std::list<int >& classes) {
	int  current_index = 0;
	std::list<int>::iterator it_classe(classes.begin());
	while (it_classe != classes.end()) {
		if (current_index == index) {
			*it_classe = classId;
			break;
		}
		current_index++;
		it_classe++;
	}
}
//compares two characters return true if there is an admissible switch between the two characters (by ex O eand 0). classId is the class index of the character that belongs to the true lpn 
//(later erroneous_character will be changed to classId)
bool substitution_admissible(const char VraiLPN_character, const char erroneous_character, int& classId)
{
	classId = -1;
	if (VraiLPN_character == erroneous_character) return false;
	else {
		//we dont accept any substitution
		return false;
	}
}
int index_character_substitution(const std::string& ExactLPN, const std::string& lpn, char& VraiLPN_character, char& erroneous_character) {
	if (ExactLPN.length() == lpn.length()) {
		std::string::const_iterator it_VraiLPN(ExactLPN.begin());
		std::string::const_iterator it(lpn.begin());
		int index = 0;
		while (it != lpn.end() && it_VraiLPN != ExactLPN.end()) {
			if (*it != *it_VraiLPN) {
				VraiLPN_character = *it_VraiLPN;
				erroneous_character = *it;
				return index;
			}
			else { it++; it_VraiLPN++; index++; }
		}return -1;
	}
	else return -1;
}
//we know the true license plate number that come from a training image and we want to find the detections boxes to aautomatically annotate the image.
//We also have run the nn that produces detections, the goal of this func is to find the detections that are closest to the true lpn
int find_nearest_plate_substitutions_allowed(const std::string& ExactLPN,
	//all lps in the image given by lpn (as string), lp country ppronenace (as class index) and lp area in the image (cv::Rect)
	std::list <std::string>& lpns, const std::list <int>& lp_country_class, const std::list < cv::Rect>& lp_rois, const
	std::list < std::list<float>>& confidences, std::list < std::list<int>>& classes, const std::list < std::list<cv::Rect>>& boxes,
	//output = nearest lpn + its class + its bounding box
	std::string& best_lpn, int& best_country_class, cv::Rect& best_lpn_roi,
	//output = characters in nearest lpn 
	std::list<float>& best_confidences, std::list<int>& best_classes, std::list<cv::Rect>& best_boxes)
{
	int min_editdistance = SHRT_MAX;
	Levenshtein lev;
	std::list <std::string>::iterator it_lpns(lpns.begin());
	std::list <int>::const_iterator it_country_class(lp_country_class.begin());
	std::list < cv::Rect>::const_iterator it_lpn_roi(lp_rois.begin());
	std::list < std::list<float>>::const_iterator it_confidences(confidences.begin());
	std::list < std::list<int>>::iterator it_classes(classes.begin());
	std::list < std::list<cv::Rect>>::const_iterator it_boxes(boxes.begin());
	while (it_lpns != lpns.end() && it_country_class != lp_country_class.end() &&
		it_lpn_roi != lp_rois.end() &&
		it_confidences != confidences.end() && it_classes != classes.end() && it_boxes != boxes.end())
	{
		int editdistance = lev.Get(ExactLPN.c_str(), ExactLPN.length(), it_lpns->c_str(), it_lpns->length());
		if (min_editdistance > editdistance) {
			min_editdistance = editdistance;
			best_lpn = *it_lpns;
			best_country_class = *it_country_class;
			best_lpn_roi = *it_lpn_roi;
			best_confidences = *it_confidences;
			best_classes = *it_classes;
			best_boxes = *it_boxes;
		}
		it_lpns++;
		it_country_class++;
		it_lpn_roi++;
		it_confidences++;
		it_classes++;
		it_boxes++;
	}
	return min_editdistance;
}
//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
void get_best_plate(
	//detections when they are separated license plates by license plates
	const std::list < std::list<int>>& classIds, const std::list < std::list<float>>& confidences, const std::list < std::list<cv::Rect>>& boxes
	//output the list of the best (most probable/readable) lp
	, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp)
{
	std::list < std::list<cv::Rect>>::const_iterator it_boxes(boxes.begin());
	std::list < std::list<float>>::const_iterator it_confidences(confidences.begin());
	std::list < std::list<int>>::const_iterator it_classIds(classIds.begin());
	float best_score = 0.0f;
	while (it_boxes != boxes.end()
		&& it_confidences != confidences.end() && it_classIds != classIds.end()) {
#ifdef _DEBUG		
		assert(it_classIds->size() == it_confidences->size() && it_classIds->size() == it_boxes->size() && it_classIds->size() >= 2);
		//1;->ok
	//2;->size too small
	//4;->second detection is not a vehicle
	//6;->detection after first two ones, is not a character
		assert(is_detections_of_a_unique_license_plate(*it_classIds) == 1);
#endif //_DEBUG
		float current_score = //(float)(it_boxes->size() - 2) * 
			//From confidences of detections of all boxes of a plate, we get the average confidence.
			(it_boxes->front().width * it_boxes->front().height) * get_average_confidence_of_license_plate(*it_classIds,
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
		it_boxes++;
		it_confidences++;
		it_classIds++;
	}
}
//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
void get_best_plate(const std::string& ExactLPN,
	//detections when they are separated license plates by license plates
	const std::list < std::list<int>>& classIds, const std::list < std::list<float>>& confidences, const std::list < std::list<cv::Rect>>& boxes
	//output the list of the best (most probable/readable) lp
	, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp)
{
	std::list < std::list<cv::Rect>>::const_iterator it_boxes(boxes.begin());
	std::list < std::list<float>>::const_iterator it_confidences(confidences.begin());
	std::list < std::list<int>>::const_iterator it_classIds(classIds.begin());
	float best_score = 0.0f;
	Levenshtein lev;
	int best_editdistance = SHRT_MAX;
	size_t best_lpn_length = -1;
	while (it_boxes != boxes.end()
		&& it_confidences != confidences.end() && it_classIds != classIds.end()) {
#ifdef _DEBUG		
		assert(it_classIds->size() == it_confidences->size() && it_classIds->size() == it_boxes->size() && it_classIds->size() >= 2);
		//1;->ok
	//2;->size too small
	//4;->second detection is not a vehicle
	//6;->detection after first two ones, is not a character
		assert(is_detections_of_a_unique_license_plate(*it_classIds) == 1);
#endif //_DEBUG
		float current_score = //(float)(it_boxes->size() - 2) * 
			//From confidences of detections of all boxes of a plate, we get the average confidence.
			(it_boxes->front().width * it_boxes->front().height) * get_average_confidence_of_license_plate(*it_classIds,
				*it_confidences);
		std::vector<cv::Rect> tri_left_vect_of_detected_boxes;			std::vector<float> tri_left_confidences; std::vector<int> tri_left_classIds;
		std::string lpn = get_lpn(*it_boxes, *it_confidences, *it_classIds, tri_left_vect_of_detected_boxes,
			tri_left_confidences, tri_left_classIds);
		int editdistance = lev.Get(ExactLPN.c_str(), ExactLPN.length(), lpn.c_str(), lpn.length());
		if (best_editdistance > editdistance || (best_lpn_length != ExactLPN.length() && lpn.length() == ExactLPN.length() && best_editdistance == editdistance)
			|| (best_score < current_score && best_editdistance == editdistance && !(best_lpn_length == ExactLPN.length() && lpn.length() != ExactLPN.length()))) {
			best_lpn_length = lpn.length();
			best_editdistance = editdistance;
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
}
//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
void get_best_plate(const std::string& ExactLPN,
	//detections when they are separated license plates by license plates
	const std::list < std::list<int>>& classIds, const std::list < std::list<float>>& confidences, const std::list < std::list<cv::Rect>>& boxes
	//output the list of the best (most probable/readable) lp
	, std::vector<float>& confidence_one_lp, std::vector < cv::Rect>& one_lp, std::vector<int>& classIds_one_lp)
{
	std::list < std::list<cv::Rect>>::const_iterator it_boxes(boxes.begin());
	std::list < std::list<float>>::const_iterator it_confidences(confidences.begin());
	std::list < std::list<int>>::const_iterator it_classIds(classIds.begin());
	float best_score = 0.0f;
	Levenshtein lev;
	int best_editdistance = SHRT_MAX;
	size_t best_lpn_length = -1;
	while (it_boxes != boxes.end()
		&& it_confidences != confidences.end() && it_classIds != classIds.end()) {
#ifdef _DEBUG		
		assert(it_classIds->size() == it_confidences->size() && it_classIds->size() == it_boxes->size() && it_classIds->size() >= 2);
		//1;->ok
	//2;->size too small
	//4;->second detection is not a vehicle
	//6;->detection after first two ones, is not a character
		assert(is_detections_of_a_unique_license_plate(*it_classIds) == 1);
#endif //_DEBUG
		float current_score = //(float)(it_boxes->size() - 2) * 
			//From confidences of detections of all boxes of a plate, we get the average confidence.
			(it_boxes->front().width * it_boxes->front().height) * get_average_confidence_of_license_plate(*it_classIds,
				*it_confidences);
		std::vector<cv::Rect> tri_left_vect_of_detected_boxes;			std::vector<float> tri_left_confidences; std::vector<int> tri_left_classIds;
		std::string lpn = get_lpn(*it_boxes, *it_confidences, *it_classIds, tri_left_vect_of_detected_boxes,
			tri_left_confidences, tri_left_classIds);
		int editdistance = lev.Get(ExactLPN.c_str(), ExactLPN.length(), lpn.c_str(), lpn.length());
		if (best_editdistance > editdistance || (best_lpn_length != ExactLPN.length() && lpn.length() == ExactLPN.length() && best_editdistance == editdistance)
			|| (best_score < current_score && best_editdistance == editdistance && !(best_lpn_length == ExactLPN.length() && lpn.length() != ExactLPN.length()))) {
			best_lpn_length = lpn.length();
			best_editdistance = editdistance;
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
}
//the nnet has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number. 
//it can deal with license pates that have two lines of charcaters
std::string get_lpn(
	const std::list<cv::Rect>& l_of_detected_boxes,
	const std::list<float>& l_confidences, const std::list<int>& l_classIds,
	//list of characters inside the lp
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes,
	std::vector<float>& tri_left_confidences, std::vector<int>& tri_left_classIds,
	float nmsThreshold
) {
	std::vector<cv::Rect> vect_of_detected_boxes;
	std::vector<float> confidences;
	std::vector<int> classIds;
	std::copy(l_of_detected_boxes.begin(), l_of_detected_boxes.end(), std::back_inserter(vect_of_detected_boxes));
	std::copy(l_confidences.begin(), l_confidences.end(), std::back_inserter(confidences));
	std::copy(l_classIds.begin(), l_classIds.end(), std::back_inserter(classIds));
	return get_single_lpn(
		vect_of_detected_boxes, confidences, classIds,
		tri_left_vect_of_detected_boxes,
		tri_left_confidences, tri_left_classIds,
		nmsThreshold);
}
//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
void get_best_plate(
	//detections when they are separated license plates by license plates
	const std::list < std::vector<int>>& classIds, const std::list < std::vector<float>>& confidences, const std::list < std::vector<cv::Rect>>& boxes
	//output the list of the best (most probable/readable) lp
	, std::list<float>& confidence_one_lp, std::list < cv::Rect>& one_lp, std::list<int>& classIds_one_lp)
{
	std::list < std::vector<cv::Rect>>::const_iterator it_boxes(boxes.begin());
	std::list < std::vector<float>>::const_iterator it_confidences(confidences.begin());
	std::list < std::vector<int>>::const_iterator it_classIds(classIds.begin());
	float best_score = 0.0f;
	while (it_boxes != boxes.end()
		&& it_confidences != confidences.end() && it_classIds != classIds.end()) {
#ifdef _DEBUG		
		assert(it_classIds->size() == it_confidences->size() && it_classIds->size() == it_boxes->size() && it_classIds->size() >= 2);
		//1;->ok
	//2;->size too small
	//4;->second detection is not a vehicle
	//6;->detection after first two ones, is not a character
		assert(is_detections_of_a_unique_license_plate(*it_classIds) == 1);
#endif //_DEBUG
		float current_score = //(float)(it_boxes->size() - 2) * 
			//From confidences of detections of all boxes of a plate, we get the average confidence.
			(it_boxes->front().width * it_boxes->front().height) * get_average_confidence_of_license_plate(*it_classIds,
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
		it_boxes++;
		it_confidences++;
		it_classIds++;
	}
}
//For each plate in the image, the detections have been separated. From these, we select the detections of the plates that have have the best detection score.
void add_lp_and_vehicle(const std::list<cv::Rect>& boxes, const std::list<float>& confidences, const std::list<int>& classIds,
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes, std::vector<float>& tri_left_confidences, std::vector<int>& tri_left_classIds
	, const int classId_last_country
	//classId_last_country : is the class index of the last country in the list of detected classes.
)
{
	std::list<cv::Rect> l_tri_left_vect_of_detected_boxes;
	std::list<float> l_tri_left_confidences;
	std::list<int> l_tri_left_classIds;
	std::copy(tri_left_vect_of_detected_boxes.begin(), tri_left_vect_of_detected_boxes.end(), std::back_inserter(l_tri_left_vect_of_detected_boxes));
	std::copy(tri_left_confidences.begin(), tri_left_confidences.end(), std::back_inserter(l_tri_left_confidences));
	std::copy(tri_left_classIds.begin(), tri_left_classIds.end(), std::back_inserter(l_tri_left_classIds));
	//this function adds if they dont already exist, a roi for the licene plate (equal to the global rect englobing the boxes) and a blank rect for the vehicle box
	add_lp_and_vehicle(boxes, confidences, classIds,
		l_tri_left_vect_of_detected_boxes, l_tri_left_confidences, l_tri_left_classIds
		, classId_last_country);
	std::copy(l_tri_left_vect_of_detected_boxes.begin(), l_tri_left_vect_of_detected_boxes.end(), std::back_inserter(tri_left_vect_of_detected_boxes));
	std::copy(l_tri_left_confidences.begin(), l_tri_left_confidences.end(), std::back_inserter(tri_left_confidences));
	std::copy(l_tri_left_classIds.begin(), l_tri_left_classIds.end(), std::back_inserter(tri_left_classIds));
}
//this function adds if they dont already exist, a roi for the licene plate (equal to the global rect englobing the boxes) and a blank rect for the vehicle box
void add_lp_and_vehicle(const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, const std::vector<int>& classIds,
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes, std::vector<float>& tri_left_confidences, std::vector<int>& tri_left_classIds
	, const int classId_last_country//classId_last_country : is the class index of the last country in the list of detected classes.
)
{
	std::list<cv::Rect> l_tri_left_vect_of_detected_boxes;
	std::list<float> l_tri_left_confidences;
	std::list<int> l_tri_left_classIds;
	std::copy(tri_left_vect_of_detected_boxes.begin(), tri_left_vect_of_detected_boxes.end(), std::back_inserter(l_tri_left_vect_of_detected_boxes));
	std::copy(tri_left_confidences.begin(), tri_left_confidences.end(), std::back_inserter(l_tri_left_confidences));
	std::copy(tri_left_classIds.begin(), tri_left_classIds.end(), std::back_inserter(l_tri_left_classIds));
	//this function adds if they dont already exist, a roi for the licene plate (equal to the global rect englobing the boxes) and a blank rect for the vehicle box
	add_lp_and_vehicle(boxes, confidences, classIds,
		l_tri_left_vect_of_detected_boxes, l_tri_left_confidences, l_tri_left_classIds
		, classId_last_country);
	std::copy(l_tri_left_vect_of_detected_boxes.begin(), l_tri_left_vect_of_detected_boxes.end(), std::back_inserter(tri_left_vect_of_detected_boxes));
	std::copy(l_tri_left_confidences.begin(), l_tri_left_confidences.end(), std::back_inserter(tri_left_confidences));
	std::copy(l_tri_left_classIds.begin(), l_tri_left_classIds.end(), std::back_inserter(tri_left_classIds));
}
//this function adds if they dont already exist, a roi for the licene plate (equal to the global rect englobing the boxes) and a blank rect for the vehicle box
void add_lp_and_vehicle(const std::list<cv::Rect>& boxes, const std::list<float>& confidences, const std::list<int>& classIds,
	std::list<cv::Rect>& tri_left_vect_of_detected_boxes, std::list<float>& tri_left_confidences, std::list<int>& tri_left_classIds
	, const int classId_last_country//classId_last_country : is the class index of the last country in the list of detected classes.
)
{
	std::list < cv::Rect>::const_iterator it_boxes(boxes.begin());
	std::list < float>::const_iterator it_confidences(confidences.begin());
	std::list < int>::const_iterator it_classIds(classIds.begin());
	cv::Rect lp(0, 0, 0, 0);
	int lp_classIds = -2;
	float lp_confidence = .0f;
	//first scan raw detections to find lp that can encapsulate tri_left_vect_of_detected_boxes
	while (it_boxes != boxes.end() && it_confidences != confidences.end() && it_classIds != classIds.end()) {//single character--> returns 1
//license plate--> returns 2
//vehicle--> returns 3
//negative index--> returns 0 must be an error
		if (is_this_box_a_character_a_license_plate_or_a_vehicle(*it_classIds, classId_last_country) == 2)
		{//for each box in the container, check that it is nearly entirely contained in the second argument
			if (is_in_rect_if(tri_left_vect_of_detected_boxes, *it_boxes)) {
				if (lp.area() > it_boxes->area() || (lp_classIds == -2)) {
					lp_classIds = *it_classIds;
					lp = *it_boxes;
					lp_confidence = *it_confidences;
				}
			}
		}
		it_boxes++;
		it_confidences++;
		it_classIds++;
	}
	if (is_this_box_a_character_a_license_plate_or_a_vehicle(lp_classIds, classId_last_country) == 2) {
#ifdef _DEBUG		
		assert(lp.area() > 0 && lp_classIds >= NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1 
			&& lp_confidence > 0.0f);
#endif //_DEBUG
	}
	else {
#ifdef _DEBUG		
		assert(lp == cv::Rect(0, 0, 0, 0) && lp_classIds == -2 && lp_confidence == 0.0f);
#endif //_DEBUG
		//be carefull there, we cannot just invoke lp = get_global_rect(boxes); since boxes may contains the vehicle box
		std::list<cv::Rect> l_boxes;
		std::copy(boxes.begin(), boxes.end(), std::back_inserter(l_boxes));
		std::list<float> l_confidences;
		std::copy(confidences.begin(), confidences.end(), std::back_inserter(l_confidences));
		std::list<int> l_classIds;
		std::copy(classIds.begin(), classIds.end(), std::back_inserter(l_classIds));
		sort_from_left_to_right(l_boxes, l_confidences, l_classIds);//sorts all the boxes from left to right
		//filter out lpn box
		//***************************************************
		//                  FILTER
		//***************************************************
		filter_out_everything_but_characters(l_boxes,
			l_confidences, l_classIds);
		lp = get_global_rect(l_boxes);
		lp_classIds = NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1
			;
		lp_confidence = 0.8f;
	}
	it_boxes = (boxes.begin());
	it_confidences = (confidences.begin());
	it_classIds = (classIds.begin());
	cv::Rect vehicle(0, 0, 0, 0);
	int vehicle_classIds = -1;
	float vehicle_confidence = .0f;
	//first scan raw detections to find vehicle that can encapsulate tri_left_vect_of_detected_boxes
	while (it_boxes != boxes.end() && it_confidences != confidences.end() && it_classIds != classIds.end()) {//single character--> returns 1
//license plate--> returns 2
//vehicle--> returns 3
//negative index--> returns 0 must be an error
		if (is_this_box_a_character_a_license_plate_or_a_vehicle(*it_classIds, classId_last_country) == 3)
		{//for each box in the container, check that it is nearly entirely contained in the second argument
			if (is_in_rect_if(lp, *it_boxes)) {
				if (vehicle.area() > it_boxes->area() || (vehicle_classIds == -1)) {
					vehicle_classIds = *it_classIds;
					vehicle = *it_boxes;
					vehicle_confidence = *it_confidences;
				}
			}
		}
		it_boxes++;
		it_confidences++;
		it_classIds++;
	}
#ifdef _DEBUG		
	assert((vehicle.area() > 0 && vehicle_classIds > classId_last_country&& vehicle_confidence > 0.0f) ||
		(vehicle == cv::Rect(0, 0, 0, 0) && vehicle_classIds == -1 && vehicle_confidence == 0.0f)
	);
#endif //_DEBUG
#ifdef _DEBUG		
	assert(lp.area() > 0 && lp_classIds >= NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1 
		&& lp_confidence > 0.0f);
#endif //_DEBUG
	//now add vehicle at the beginning of the detections
	tri_left_vect_of_detected_boxes.push_front(vehicle);
	tri_left_confidences.push_front(vehicle_confidence);
	tri_left_classIds.push_front(vehicle_classIds);
	//now add lp at the beginning of the detections
	tri_left_vect_of_detected_boxes.push_front(lp);
	tri_left_confidences.push_front(lp_confidence);
	tri_left_classIds.push_front(lp_classIds);
}
//this function adds if they dont already exist, a roi for the licene plate (equal to the global rect englobing the boxes) and a blank rect for the vehicle box
void add_lp_and_vehicle(const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, const std::vector<int>& classIds,
	std::list<cv::Rect>& tri_left_vect_of_detected_boxes, std::list<float>& tri_left_confidences, std::list<int>& tri_left_classIds
	, const int classId_last_country//classId_last_country : is the class index of the last country in the list of detected classes.
)
{
	std::vector < cv::Rect>::const_iterator it_boxes(boxes.begin());
	std::vector < float>::const_iterator it_confidences(confidences.begin());
	std::vector < int>::const_iterator it_classIds(classIds.begin());
	cv::Rect lp(0, 0, 0, 0);
	int lp_classIds = -2;
	float lp_confidence = .0f;
	//first scan raw detections to find lp that can encapsulate tri_left_vect_of_detected_boxes
	while (it_boxes != boxes.end() && it_confidences != confidences.end() && it_classIds != classIds.end()) {//single character--> returns 1
//license plate--> returns 2
//vehicle--> returns 3
//negative index--> returns 0 must be an error
		if (is_this_box_a_character_a_license_plate_or_a_vehicle(*it_classIds, classId_last_country) == 2)
		{//for each box in the container, check that it is nearly entirely contained in the second argument
			if (is_in_rect_if(tri_left_vect_of_detected_boxes, *it_boxes)) {
				if (lp.area() > it_boxes->area() || (lp_classIds == -2)) {
					lp_classIds = *it_classIds;
					lp = *it_boxes;
					lp_confidence = *it_confidences;
				}
			}
		}
		it_boxes++;
		it_confidences++;
		it_classIds++;
	}
	if (is_this_box_a_character_a_license_plate_or_a_vehicle(lp_classIds, classId_last_country) == 2) {
#ifdef _DEBUG		
		assert(lp.area() > 0 && lp_classIds >= NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1 
			&& lp_confidence > 0.0f);
#endif //_DEBUG
	}
	else {
#ifdef _DEBUG		
		assert(lp == cv::Rect(0, 0, 0, 0) && lp_classIds == -2 && lp_confidence == 0.0f);
#endif //_DEBUG
		std::list<cv::Rect> l_boxes;
		std::copy(boxes.begin(), boxes.end(), std::back_inserter(l_boxes));
		std::list<float> l_confidences;
		std::copy(confidences.begin(), confidences.end(), std::back_inserter(l_confidences));
		std::list<int> l_classIds;
		std::copy(classIds.begin(), classIds.end(), std::back_inserter(l_classIds));
		sort_from_left_to_right(l_boxes, l_confidences, l_classIds);//sorts all the boxes from left to right
		//filter out lpn box
		//***************************************************
		//                  FILTER
		//***************************************************
		filter_out_everything_but_characters(l_boxes,
			l_confidences, l_classIds);
		lp = get_global_rect(l_boxes); lp_classIds = NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1
			;
		lp_confidence = 0.8f;
	}
	it_boxes = (boxes.begin());
	it_confidences = (confidences.begin());
	it_classIds = (classIds.begin());
	cv::Rect vehicle(0, 0, 0, 0);
	int vehicle_classIds = -1;
	float vehicle_confidence = .0f;
	//first scan raw detections to find vehicle that can encapsulate tri_left_vect_of_detected_boxes
	while (it_boxes != boxes.end() && it_confidences != confidences.end() && it_classIds != classIds.end()) {//single character--> returns 1
//license plate--> returns 2
//vehicle--> returns 3
//negative index--> returns 0 must be an error
		if (is_this_box_a_character_a_license_plate_or_a_vehicle(*it_classIds, classId_last_country) == 3)
		{//for each box in the container, check that it is nearly entirely contained in the second argument
			if (is_in_rect_if(lp, *it_boxes)) {
				if (vehicle.area() > it_boxes->area() || (vehicle_classIds == -1)) {
					vehicle_classIds = *it_classIds;
					vehicle = *it_boxes;
					vehicle_confidence = *it_confidences;
				}
			}
		}
		it_boxes++;
		it_confidences++;
		it_classIds++;
	}
#ifdef _DEBUG		
	assert((vehicle.area() > 0 && vehicle_classIds > classId_last_country&& vehicle_confidence > 0.0f) ||
		(vehicle == cv::Rect(0, 0, 0, 0) && vehicle_classIds == -1 && vehicle_confidence == 0.0f)
	);
#endif //_DEBUG
#ifdef _DEBUG		
	assert(lp.area() > 0 && lp_classIds >= NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1 
		&& lp_confidence > 0.0f);
#endif //_DEBUG
	//now add vehicle at the beginning of the detections
	tri_left_vect_of_detected_boxes.push_front(vehicle);
	tri_left_confidences.push_front(vehicle_confidence);
	tri_left_classIds.push_front(vehicle_classIds);
	//now add lp at the beginning of the detections
	tri_left_vect_of_detected_boxes.push_front(lp);
	tri_left_confidences.push_front(lp_confidence);
	tri_left_classIds.push_front(lp_classIds);
}
//1;->ok
//2;->size too small
//4;->second detection is not a vehicle
//6;->detection after first two ones, is not a character
int is_detections_of_a_unique_license_plate(const std::vector<int>& classIds)
{
	if (classIds.size() >= 2) {
		const int classId_last_country = NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE + NUMBER_OF_COUNTRIES - 1;
		std::vector<int>::const_iterator it_classIds(classIds.begin());
		if (*it_classIds < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1 
			|| *it_classIds >= classId_last_country) return 4;//first detection is not a plate
		it_classIds++;
		if (*it_classIds < classId_last_country && *it_classIds>0) return 4;//second detection is not a vehicle
		it_classIds++;
		while (it_classIds != classIds.end()) {
			if (*it_classIds >= NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1 
				|| *it_classIds < 0)
				return 6;//detection after first two ones, is not a character
			it_classIds++;
		} return 1;//ok
	}
	else return 2;//size too small
}
//1;->ok
//2;->size too small
//4;->second detection is not a vehicle
//6;->detection after first two ones, is not a character
int is_detections_of_a_unique_license_plate(const std::list<int>& classIds)
{
	if (classIds.size() >= 2) {
		//classId_last_country : is the class index of the last country in the list of detected classes.
		const int classId_last_country = NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE + NUMBER_OF_COUNTRIES - 1;
		std::list<int>::const_iterator it_classIds(classIds.begin());
		if (*it_classIds < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1 
			|| *it_classIds > classId_last_country) return 4;//first detection is not a plate
		it_classIds++;
		if (*it_classIds < classId_last_country && *it_classIds>0) return 4;//second detection is not a vehicle
		it_classIds++;
		while (it_classIds != classIds.end()) {
			if (*it_classIds >= NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1 
				|| *it_classIds < 0)
				return 6;//detection after first two ones, is not a character
			it_classIds++;
		} return 1;//ok
	}
	else return 2;//size too small
}
//From confidences of detections of all boxes of a plate, we get the average confidence.
float get_average_confidence_of_license_plate(const std::vector<int>& classIds,
	const std::vector<float>& confidences)
{
#ifdef _DEBUG		
	assert(classIds.size() == confidences.size() && confidences.size() >= 2);
	//1;->ok
//2;->size too small
//4;->second detection is not a vehicle
//6;->detection after first two ones, is not a character
	assert(is_detections_of_a_unique_license_plate(classIds) == 1);
#endif //_DEBUG
	std::vector<int>::const_iterator it_classIds(classIds.begin());
	std::vector<float>::const_iterator it_confidences(confidences.begin());
	float score = confidences.front();
	while (it_confidences != confidences.end() && it_classIds != classIds.end()) {
		if (*it_classIds < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1
			)
			score += *it_confidences;
		it_confidences++;
		it_classIds++;
	}
	return (score / (float)(confidences.size() - 1));
}
//From confidences of detections of all boxes of a plate, we get the average confidence.
float get_average_confidence_of_license_plate(const std::list<int>& classIds,
	const std::list<float>& confidences)
{
#ifdef _DEBUG		
	assert(classIds.size() == confidences.size() && confidences.size() >= 2);
	//1;->ok
//2;->size too small
//4;->second detection is not a vehicle
//6;->detection after first two ones, is not a character
	assert(is_detections_of_a_unique_license_plate(classIds) == 1);
#endif //_DEBUG
	std::list<int>::const_iterator it_classIds(classIds.begin());
	std::list<float>::const_iterator it_confidences(confidences.begin());
	float score = confidences.front();
	while (it_confidences != confidences.end() && it_classIds != classIds.end()) {
		if (*it_classIds < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1
			)
			score += *it_confidences;
		it_confidences++;
		it_classIds++;
	}
	return (score / (float)(confidences.size() - 1));
}
//
//the dnn has detected boxes that represent characters of the license plate, this function now groups characters in the same license plate and then rearranged from left to right.
//it can deal with license pates that have two lines of charcaters
void separate_license_plates_if_necessary_add_blank_vehicles(
	//raw detections
	const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, const std::vector<int>& classIds,
	//detections when they are separated license plates by license plates
	std::list<std::string>& lpns, std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate, std::list <std::vector<int>>& l_vect_of_classIds_in_a_license_plate,
	//double lists (one element list for each lp detected) of detected characters inside a lp
	std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate_tri_left,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate_tri_left, std::list <std::vector<int>>& l_vect_of_classIds_in_a_license_plate_tri_left,
	const int classId_last_country,//classId_last_country : is the class index of the last country in the list of detected classes.
	const float nmsThreshold)
{
	//groups detected boxes that correspond to the same vehicle. The separation is based on raw detections of license plates from the dnn
	//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
	group_characters_in_the_same_license_plate(
		boxes,
		confidences, classIds, l_vect_of_boxes_in_a_license_plate, l_vect_of_confidences_in_a_license_plate, l_vect_of_classIds_in_a_license_plate, classId_last_country
	);
	std::list < std::vector<cv::Rect>>::const_iterator it_l_vect_of_boxes_in_a_license_plate(l_vect_of_boxes_in_a_license_plate.begin());
	std::list < std::vector<float>>::const_iterator it_l_vect_of_confidences_in_a_license_plate(l_vect_of_confidences_in_a_license_plate.begin());
	std::list < std::vector<int>>::const_iterator it_l_vect_of_classIds_in_a_license_plate(l_vect_of_classIds_in_a_license_plate.begin());
	while (it_l_vect_of_boxes_in_a_license_plate != l_vect_of_boxes_in_a_license_plate.end()
		&& it_l_vect_of_confidences_in_a_license_plate != l_vect_of_confidences_in_a_license_plate.end() && it_l_vect_of_classIds_in_a_license_plate != l_vect_of_classIds_in_a_license_plate.end()) {
#ifdef _DEBUG		
		assert(it_l_vect_of_classIds_in_a_license_plate->size() == it_l_vect_of_confidences_in_a_license_plate->size());
		assert(it_l_vect_of_classIds_in_a_license_plate->size() == it_l_vect_of_boxes_in_a_license_plate->size());
		assert(it_l_vect_of_classIds_in_a_license_plate->size() >= 2);
		//1;->ok
	//2;->size too small
	//4;->second detection is not a vehicle
	//6;->detection after first two ones, is not a character
		assert(is_detections_of_a_unique_license_plate(*it_l_vect_of_classIds_in_a_license_plate) == 1);
#endif //_DEBUG
		std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
		std::vector<float> tri_left_confidences;
		std::vector<int> tri_left_classIds;
		std::string lpn = get_single_lpn(
			*it_l_vect_of_boxes_in_a_license_plate,
			*it_l_vect_of_confidences_in_a_license_plate, *it_l_vect_of_classIds_in_a_license_plate,
			tri_left_vect_of_detected_boxes,
			tri_left_confidences, tri_left_classIds, nmsThreshold
		);
		l_vect_of_boxes_in_a_license_plate_tri_left.push_back(tri_left_vect_of_detected_boxes);
		l_vect_of_confidences_in_a_license_plate_tri_left.push_back(tri_left_confidences);
		l_vect_of_classIds_in_a_license_plate_tri_left.push_back(tri_left_classIds);
		lpns.push_back(lpn);
		it_l_vect_of_boxes_in_a_license_plate++;
		it_l_vect_of_confidences_in_a_license_plate++;
		it_l_vect_of_classIds_in_a_license_plate++;
	}
#ifdef _DEBUG		
	assert(boxes.size() ==
		confidences.size() && boxes.size() ==
		classIds.size());
	assert(l_vect_of_boxes_in_a_license_plate.size() ==
		l_vect_of_classIds_in_a_license_plate.size() && l_vect_of_boxes_in_a_license_plate.size() ==
		l_vect_of_confidences_in_a_license_plate.size()
		&& l_vect_of_boxes_in_a_license_plate.size() ==
		lpns.size());
	assert(l_vect_of_boxes_in_a_license_plate_tri_left.size() ==
		l_vect_of_confidences_in_a_license_plate_tri_left.size() && l_vect_of_classIds_in_a_license_plate_tri_left.size() ==
		l_vect_of_confidences_in_a_license_plate_tri_left.size()
		&& l_vect_of_classIds_in_a_license_plate_tri_left.size() ==
		lpns.size());
#endif //_DEBUG
}
//the dnn has detected boxes that represent characters of the license plate, this function now groups characters in the same license plate and then rearranged from left to right.
//it can deal with license pates that have two lines of charcaters
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
	const float nmsThreshold
) {
	//groups detected boxes that correspond to the same vehicle. The separation is based on raw detections of license plates from the dnn
	//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
	group_characters_in_the_same_license_plate(
		boxes,
		confidences, classIds, l_vect_of_boxes_in_a_license_plate, l_vect_of_confidences_in_a_license_plate, l_vect_of_classIds_in_a_license_plate, classId_last_country
	);
	std::list < std::list<cv::Rect>>::const_iterator it_l_vect_of_boxes_in_a_license_plate(l_vect_of_boxes_in_a_license_plate.begin());
	std::list < std::list<float>>::const_iterator it_l_vect_of_confidences_in_a_license_plate(l_vect_of_confidences_in_a_license_plate.begin());
	std::list < std::list<int>>::const_iterator it_l_vect_of_classIds_in_a_license_plate(l_vect_of_classIds_in_a_license_plate.begin());
	while (it_l_vect_of_boxes_in_a_license_plate != l_vect_of_boxes_in_a_license_plate.end()
		&& it_l_vect_of_confidences_in_a_license_plate != l_vect_of_confidences_in_a_license_plate.end() && it_l_vect_of_classIds_in_a_license_plate != l_vect_of_classIds_in_a_license_plate.end()) {
#ifdef _DEBUG		
		assert(it_l_vect_of_classIds_in_a_license_plate->size() == it_l_vect_of_confidences_in_a_license_plate->size());
		assert(it_l_vect_of_classIds_in_a_license_plate->size() == it_l_vect_of_boxes_in_a_license_plate->size());
		assert(it_l_vect_of_classIds_in_a_license_plate->size() >= 2);
		//1;->ok
	//2;->size too small
	//4;->second detection is not a vehicle
	//6;->detection after first two ones, is not a character
		assert(is_detections_of_a_unique_license_plate(*it_l_vect_of_classIds_in_a_license_plate) == 1);
#endif //_DEBUG
		std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
		std::vector<float> tri_left_confidences;
		std::vector<int> tri_left_classIds;
		std::string lpn = get_lpn(
			*it_l_vect_of_boxes_in_a_license_plate,
			*it_l_vect_of_confidences_in_a_license_plate, *it_l_vect_of_classIds_in_a_license_plate,
			tri_left_vect_of_detected_boxes,
			tri_left_confidences, tri_left_classIds, nmsThreshold
		);
		l_vect_of_boxes_in_a_license_plate_tri_left.push_back(tri_left_vect_of_detected_boxes);
		l_vect_of_confidences_in_a_license_plate_tri_left.push_back(tri_left_confidences);
		l_vect_of_classIds_in_a_license_plate_tri_left.push_back(tri_left_classIds);
		lpns.push_back(lpn);
		it_l_vect_of_boxes_in_a_license_plate++;
		it_l_vect_of_confidences_in_a_license_plate++;
		it_l_vect_of_classIds_in_a_license_plate++;
	}
#ifdef _DEBUG		
	assert(boxes.size() ==
		confidences.size() && boxes.size() ==
		classIds.size());
	assert(l_vect_of_boxes_in_a_license_plate.size() ==
		l_vect_of_classIds_in_a_license_plate.size() && l_vect_of_boxes_in_a_license_plate.size() ==
		l_vect_of_confidences_in_a_license_plate.size()
		&& l_vect_of_boxes_in_a_license_plate.size() ==
		lpns.size());
	assert(l_vect_of_boxes_in_a_license_plate_tri_left.size() ==
		l_vect_of_confidences_in_a_license_plate_tri_left.size() && l_vect_of_classIds_in_a_license_plate_tri_left.size() ==
		l_vect_of_confidences_in_a_license_plate_tri_left.size()
		&& l_vect_of_classIds_in_a_license_plate_tri_left.size() ==
		lpns.size());
#endif //_DEBUG
}
//the dnn has detected boxes that represent characters of the license plate, this function now groups characters in the same license plate and then rearranged from left to right.
//it can deal with license pates that have two lines of charcaters
void separate_license_plates_if_necessary_add_blank_vehicles(
	//raw detections
	const std::list<cv::Rect>& boxes, const std::list<float>& confidences, const std::list<int>& classIds,
	//detections when they are separated license plates by license plates
	std::list<std::string>& lpns, std::list < std::list<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::list<float>>& l_vect_of_confidences_in_a_license_plate, std::list <std::list<int>>& l_vect_of_classIds_in_a_license_plate,
	//double lists (one element list for each lp detected) of detected characters inside a lp
	std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate_tri_left,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate_tri_left, std::list <std::vector<int>>& l_vect_of_classIds_in_a_license_plate_tri_left,
	const int classId_last_country, //classId_last_country : is the class index of the last country in the list of detected classes.
	const float nmsThreshold
) {
	//groups detected boxes that correspond to the same vehicle. The separation is based on raw detections of license plates from the dnn
	//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
	group_characters_in_the_same_license_plate(
		boxes,
		confidences, classIds, l_vect_of_boxes_in_a_license_plate, l_vect_of_confidences_in_a_license_plate, l_vect_of_classIds_in_a_license_plate, classId_last_country
	);
	std::list < std::list<cv::Rect>>::const_iterator it_l_vect_of_boxes_in_a_license_plate(l_vect_of_boxes_in_a_license_plate.begin());
	std::list < std::list<float>>::const_iterator it_l_vect_of_confidences_in_a_license_plate(l_vect_of_confidences_in_a_license_plate.begin());
	std::list < std::list<int>>::const_iterator it_l_vect_of_classIds_in_a_license_plate(l_vect_of_classIds_in_a_license_plate.begin());
	while (it_l_vect_of_boxes_in_a_license_plate != l_vect_of_boxes_in_a_license_plate.end()
		&& it_l_vect_of_confidences_in_a_license_plate != l_vect_of_confidences_in_a_license_plate.end() && it_l_vect_of_classIds_in_a_license_plate != l_vect_of_classIds_in_a_license_plate.end()) {
#ifdef _DEBUG		
		assert(it_l_vect_of_classIds_in_a_license_plate->size() == it_l_vect_of_confidences_in_a_license_plate->size());
		assert(it_l_vect_of_classIds_in_a_license_plate->size() == it_l_vect_of_boxes_in_a_license_plate->size());
		assert(it_l_vect_of_classIds_in_a_license_plate->size() >= 2);
		//1;->ok
	//2;->size too small
	//4;->second detection is not a vehicle
	//6;->detection after first two ones, is not a character
		assert(is_detections_of_a_unique_license_plate(*it_l_vect_of_classIds_in_a_license_plate) == 1);
#endif //_DEBUG
		std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
		std::vector<float> tri_left_confidences;
		std::vector<int> tri_left_classIds;
		std::string lpn = get_lpn(
			*it_l_vect_of_boxes_in_a_license_plate,
			*it_l_vect_of_confidences_in_a_license_plate, *it_l_vect_of_classIds_in_a_license_plate,
			tri_left_vect_of_detected_boxes,
			tri_left_confidences, tri_left_classIds, nmsThreshold
		);
		l_vect_of_boxes_in_a_license_plate_tri_left.push_back(tri_left_vect_of_detected_boxes);
		l_vect_of_confidences_in_a_license_plate_tri_left.push_back(tri_left_confidences);
		l_vect_of_classIds_in_a_license_plate_tri_left.push_back(tri_left_classIds);
		lpns.push_back(lpn);
		it_l_vect_of_boxes_in_a_license_plate++;
		it_l_vect_of_confidences_in_a_license_plate++;
		it_l_vect_of_classIds_in_a_license_plate++;
	}
#ifdef _DEBUG		
	assert(boxes.size() ==
		confidences.size() && boxes.size() ==
		classIds.size());
	assert(l_vect_of_boxes_in_a_license_plate.size() ==
		l_vect_of_classIds_in_a_license_plate.size() && l_vect_of_boxes_in_a_license_plate.size() ==
		l_vect_of_confidences_in_a_license_plate.size()
		&& l_vect_of_boxes_in_a_license_plate.size() ==
		lpns.size());
	assert(l_vect_of_boxes_in_a_license_plate_tri_left.size() ==
		l_vect_of_confidences_in_a_license_plate_tri_left.size() && l_vect_of_classIds_in_a_license_plate_tri_left.size() ==
		l_vect_of_confidences_in_a_license_plate_tri_left.size()
		&& l_vect_of_classIds_in_a_license_plate_tri_left.size() ==
		lpns.size());
#endif //_DEBUG
}
//the dnn has detected boxes that represent characters of the license plate, this function now groups characters in the same license plate and then rearranged from left to right.
//it can deal with license pates that have two lines of charcaters
void separate_license_plates_if_necessary_add_blank_vehicles(
	//raw detections
	const std::list<cv::Rect>& boxes, const std::list<float>& confidences, const std::list<int>& classIds,
	//detections when they are separated license plates by license plates
	std::list<std::string>& lpns, std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate, std::list <std::vector<int>>& l_vect_of_classIds_in_a_license_plate,
	//double lists (one element list for each lp detected) of detected characters inside a lp
	std::list < std::vector<cv::Rect>>& l_vect_of_boxes_in_a_license_plate_tri_left,
	std::list < std::vector<float>>& l_vect_of_confidences_in_a_license_plate_tri_left, std::list <std::vector<int>>& l_vect_of_classIds_in_a_license_plate_tri_left,
	const int classId_last_country, //classId_last_country : is the class index of the last country in the list of detected classes.
	const float nmsThreshold
) {
	//groups detected boxes that correspond to the same vehicle. The separation is based on raw detections of license plates from the dnn
	//output lists look like : first box = license plate (either a detected box either the global rect englobing characters boxes, second element = vehicle (either a detected vehicle either (0,0,0,0)
//and remaining elements are characters
	group_characters_in_the_same_license_plate(
		boxes,
		confidences, classIds, l_vect_of_boxes_in_a_license_plate, l_vect_of_confidences_in_a_license_plate, l_vect_of_classIds_in_a_license_plate, classId_last_country
	);
	std::list < std::vector<cv::Rect>>::const_iterator it_l_vect_of_boxes_in_a_license_plate(l_vect_of_boxes_in_a_license_plate.begin());
	std::list < std::vector<float>>::const_iterator it_l_vect_of_confidences_in_a_license_plate(l_vect_of_confidences_in_a_license_plate.begin());
	std::list < std::vector<int>>::const_iterator it_l_vect_of_classIds_in_a_license_plate(l_vect_of_classIds_in_a_license_plate.begin());
	while (it_l_vect_of_boxes_in_a_license_plate != l_vect_of_boxes_in_a_license_plate.end()
		&& it_l_vect_of_confidences_in_a_license_plate != l_vect_of_confidences_in_a_license_plate.end() && it_l_vect_of_classIds_in_a_license_plate != l_vect_of_classIds_in_a_license_plate.end()) {
		std::vector<cv::Rect> tri_left_vect_of_detected_boxes;
		std::vector<float> tri_left_confidences;
		std::vector<int> tri_left_classIds;
		std::string lpn = get_single_lpn(
			*it_l_vect_of_boxes_in_a_license_plate,
			*it_l_vect_of_confidences_in_a_license_plate, *it_l_vect_of_classIds_in_a_license_plate,
			tri_left_vect_of_detected_boxes,
			tri_left_confidences, tri_left_classIds, nmsThreshold
		);
		l_vect_of_boxes_in_a_license_plate_tri_left.push_back(tri_left_vect_of_detected_boxes);
		l_vect_of_confidences_in_a_license_plate_tri_left.push_back(tri_left_confidences);
		l_vect_of_classIds_in_a_license_plate_tri_left.push_back(tri_left_classIds);
		lpns.push_back(lpn);
		it_l_vect_of_boxes_in_a_license_plate++;
		it_l_vect_of_confidences_in_a_license_plate++;
		it_l_vect_of_classIds_in_a_license_plate++;
	}
}
/*
//the nnet has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
std::string get_lpn(const std::list<int>& l_classIds) {
	std::string lpn;
	C_OCROutputs availableAlpha(LATIN_LETTERS_LATIN_DIGITS);
	std::list<int>::const_iterator it_out_classes(l_classIds.begin());
	while (it_out_classes != l_classIds.end()) {
		lpn += availableAlpha.get_char(*it_out_classes);
		it_out_classes++;
	}
	return lpn;
}
std::string get_lpn(const std::vector<int>& l_classIds) {
	std::string lpn;
	C_OCROutputs availableAlpha(LATIN_LETTERS_LATIN_DIGITS);
	std::vector<int>::const_iterator it_out_classes(l_classIds.begin());
	while (it_out_classes != l_classIds.end()) {
		lpn += availableAlpha.get_char(*it_out_classes);
		it_out_classes++;
	}
	return lpn;
}*/
//the nnet has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number. 
//it can deal with license pates that have two lines of charcaters
std::string get_lpn(
	const std::list<cv::Rect>& l_of_detected_boxes,
	const std::list<int>& l_classIds,
	//list of characters inside the lp
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes,
	std::vector<int>& tri_left_classIds,
	float nmsThreshold
) {
	std::vector<cv::Rect> vect_of_detected_boxes;
	std::vector<int> classIds;
	std::copy(l_of_detected_boxes.begin(), l_of_detected_boxes.end(), std::back_inserter(vect_of_detected_boxes));
	std::copy(l_classIds.begin(), l_classIds.end(), std::back_inserter(classIds));
	return get_single_lpn(
		vect_of_detected_boxes, classIds,
		tri_left_vect_of_detected_boxes,
		tri_left_classIds,
		nmsThreshold);
}
int horizontal_dist(const cv::Rect& r1, const cv::Rect& r2)
{
	return abs((r1.x - r2.x + r1.x - r2.x + r1.width - r2.width) / 2);
}
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
std::string get_single_lpn(
	const std::vector<cv::Rect>& boxes,
	const std::vector<float>& confidences, const std::vector<int>& classIds,
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes,
	std::vector<float>& tri_left_confidences, std::vector<int>& tri_left_classIds, float nmsThreshold
) {
	//first from left to right
	//cette fonction trie la liste de gauche a droite 
	//change from vect to list 
	std::list<cv::Rect> l_boxes;
	std::copy(boxes.begin(), boxes.end(), std::back_inserter(l_boxes));
	std::list<float> l_confidences;
	std::copy(confidences.begin(), confidences.end(), std::back_inserter(l_confidences));
	std::list<int> l_classIds;
	std::copy(classIds.begin(), classIds.end(), std::back_inserter(l_classIds));
	sort_from_left_to_right(l_boxes, l_confidences, l_classIds);//sorts all the boxes from left to right
	//filter out lpn box
	//***************************************************
	//                  FILTER
	//***************************************************
	filter_out_everything_but_characters(l_boxes,
		l_confidences, l_classIds);
	//filter out adjacent l_boxes with iou>nmsThreshold
		//***************************************************
		//                  FILTER
		//***************************************************
//if two l_boxes have an iou (intersection over union) that is too large, then they cannot represent two adjacent characters of the license plate 
//so we discard the one with the lowest confidence rate
	filter_iou2(l_boxes, l_confidences, l_classIds, nmsThreshold);
	std::list<int> levels;//levels of each character box of l_tri_left
	std::list<char> lpn;
	if (l_boxes.size() > 3) {
		std::list<cv::Rect> l_tri_left;//list of characters l_boxes ranged from left to right
		std::list<float> l_tri_left_confidences;
		std::list<int> l_tri_left_classIds;
		is_bi_level_plate(l_boxes, l_confidences, l_classIds, l_tri_left, l_tri_left_confidences, l_tri_left_classIds, levels);
		//now
		std::list<char> lpn_minus_1;
		std::list<char> lpn_0;
		std::list<char> lpn_plus_1;
		//C_OCROutputs availableAlpha(LATIN_LETTERS_NO_I_O_LATIN_DIGITS);
		std::list<int>::const_iterator it_out_classes(l_tri_left_classIds.begin());
		std::list<int>::const_iterator it_levels(levels.begin());
		std::list<cv::Rect>::const_iterator it_box(l_tri_left.begin());//list of characters l_boxes ranged from left to right
		std::list<float>::const_iterator it_confidence(l_tri_left_confidences.begin());
		std::list<cv::Rect> l_minus_1;//list of characters l_boxes ranged from left to right
		std::list<float> l_confidences_minus_1;
		std::list<int> l_classIds_minus_1;
		std::list<cv::Rect> l_0;//list of characters l_boxes ranged from left to right
		std::list<float> l_confidences_0;
		std::list<int> l_classIds_0;
		std::list<cv::Rect> l_plus_1;//list of characters l_boxes ranged from left to right
		std::list<float> l_confidences_plus_1;
		std::list<int> l_classIds_plus_1;
		while (it_out_classes != l_tri_left_classIds.end() && it_levels != levels.end() &&
			it_box != l_tri_left.end() && it_confidence != l_tri_left_confidences.end()) {
			if (*it_out_classes < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1
				) {
				if (*it_levels == -1) {
					lpn_minus_1.push_back(get_char(*it_out_classes));
					l_minus_1.push_back(*it_box);//list of characters l_boxes ranged from left to right
					l_confidences_minus_1.push_back(*it_confidence);
					l_classIds_minus_1.push_back(*it_out_classes);
				}
				else {
					if (*it_levels == 0) {
						lpn_0.push_back(get_char(*it_out_classes));
						l_0.push_back(*it_box);//list of characters l_boxes ranged from left to right
						l_confidences_0.push_back(*it_confidence);
						l_classIds_0.push_back(*it_out_classes);
					}
					else if (*it_levels == 1) {
						lpn_plus_1.push_back(get_char(*it_out_classes));
						l_plus_1.push_back(*it_box);//list of characters l_boxes ranged from left to right
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
		//std::cout << "get_lpn lpn : " << lpn<< std::endl;
		l_tri_left.clear();//list of characters l_boxes ranged from left to right
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
		//bool european_plate = true;
		if (lpn.size() > 3 && lpn.size() == l_tri_left.size()
			&& lpn.size() == l_tri_left_confidences.size()
			&& lpn.size() == l_tri_left_classIds.size()) {
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
#ifdef _DEBUG		
		//std::cout << "read lpn by engine : " << lpn_corrected << std::endl;
#endif //_DEBUG
		return lpn_corrected;
	}
	else {
		std::string lpn_corrected;
		//C_OCROutputs availableAlpha(LATIN_LETTERS_NO_I_O_LATIN_DIGITS);
		std::list<int>::const_iterator it_out_classes(l_classIds.begin());
		std::list<int>::const_iterator it_levels(levels.begin());
		while (it_out_classes != l_classIds.end() && it_levels != levels.end()) {
			if (*it_out_classes < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE //- 1
				) {
				lpn.push_back(get_char(*it_out_classes));
				lpn_corrected.push_back(get_char(*it_out_classes));
			}
			it_out_classes++; it_levels++;
		}
		std::copy(boxes.begin(), boxes.end(), std::back_inserter(tri_left_vect_of_detected_boxes));
		std::copy(confidences.begin(), confidences.end(), std::back_inserter(tri_left_confidences));
		std::copy(classIds.begin(), classIds.end(), std::back_inserter(tri_left_classIds));
#ifdef _DEBUG		
		//std::cout << "read lpn by engine : " << lpn_corrected << std::endl;
#endif //_DEBUG
		return lpn_corrected;
	}
}
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
std::string get_single_lpn(
	const std::vector<cv::Rect>& boxes,
	const std::vector<int>& classIds,
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes,
	std::vector<int>& tri_left_classIds, float nmsThreshold
) {
	float no_importance = 0.7f;
	std::vector<float> confidences(boxes.size(), no_importance);
	std::vector<float> tri_left_confidences;
	return get_single_lpn(
		boxes,
		confidences, classIds,
		tri_left_vect_of_detected_boxes,
		tri_left_confidences, tri_left_classIds, nmsThreshold);
}
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
std::string get_best_lpn(
	//raw detections
	const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, const std::vector<int>& classIds,
	//characters inside the best lpn that have been chosen from the above double linked list
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes,
	std::vector<float>& tri_left_confidences, std::vector<int>& tri_left_classIds, const float nmsThreshold, const int classId_last_country//classId_last_country : is the class index of the last country in the list of detected classes.
) {
	//detections when they are separated license plates by license plates
	std::list<std::string>  lpns; std::list < std::vector<cv::Rect>>  l_vect_of_boxes_in_a_license_plate;
	std::list < std::vector<float>>  l_vect_of_confidences_in_a_license_plate; std::list <std::vector<int>>  l_vect_of_classIds_in_a_license_plate;
	//double lists (one element list for each lp detected) of detected characters inside a lp
	std::list < std::vector<cv::Rect>>  l_vect_of_boxes_in_a_license_plate_tri_left;
	std::list < std::vector<float>>  l_vect_of_confidences_in_a_license_plate_tri_left; std::list <std::vector<int>>  l_vect_of_classIds_in_a_license_plate_tri_left;
	//the dnn has detected boxes that represent characters of the license plate, this function now groups characters in the same license plate and then rearranged from left to right.
	//it can deal with license pates that have two lines of charcaters
	separate_license_plates_if_necessary_add_blank_vehicles(
		//raw detections
		boxes, confidences, classIds,
		//detections when they are separated license plates by license plates
		lpns, l_vect_of_boxes_in_a_license_plate,
		l_vect_of_confidences_in_a_license_plate, l_vect_of_classIds_in_a_license_plate,
		//double lists (one element list for each lp detected) of detected characters inside a lp
		l_vect_of_boxes_in_a_license_plate_tri_left,
		l_vect_of_confidences_in_a_license_plate_tri_left, l_vect_of_classIds_in_a_license_plate_tri_left,
		classId_last_country);
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
		lpns.clear();
	}
	if (lpns.empty()) {
		//std::vector<cv::Rect> tri_left_vect_of_detected_boxes; std::vector<float> tri_left_confidences; std::vector<int> tri_left_classIds;
		std::string lpn = get_single_lpn(
			boxes, confidences, classIds,
			//characters inside lp
			tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds,
			nmsThreshold
		);
		return lpn;
	}
	else {
		std::list<float> confidence_one_lp; std::list < cv::Rect> one_lp; std::list<int> classIds_one_lp;
		get_best_plate(
			//detections when they are separated license plates by license plates
			l_vect_of_classIds_in_a_license_plate, l_vect_of_confidences_in_a_license_plate, l_vect_of_boxes_in_a_license_plate
			//l_vect_of_classIds_in_a_license_plate_tri_left, l_vect_of_confidences_in_a_license_plate_tri_left,l_vect_of_boxes_in_a_license_plate_tri_left
			//output the list of the best (most probable/readable) lp
			, confidence_one_lp, one_lp, classIds_one_lp);
		return get_lpn(one_lp,
			confidence_one_lp, classIds_one_lp,
			//list of characters inside the lp
			tri_left_vect_of_detected_boxes,
			tri_left_confidences, tri_left_classIds,
			nmsThreshold
		);
	}
}
//the dnn has detected boxes that represent characters of the license plate, this function now etracts from these boxes the license plate number.
//it can deal with license pates that have two lines of charcaters
std::string get_best_lpn(
	//raw detections
	const std::list<cv::Rect>& boxes, const std::list<float>& confidences, const std::list<int>& classIds,
	//characters inside the best lpn that have been chosen from the above double linked list
	std::vector<cv::Rect>& tri_left_vect_of_detected_boxes,
	std::vector<float>& tri_left_confidences, std::vector<int>& tri_left_classIds, const float nmsThreshold, const int classId_last_country//classId_last_country : is the class index of the last country in the list of detected classes.
) {
	//detections when they are separated license plates by license plates
	std::list<std::string>  lpns; std::list < std::vector<cv::Rect>>  l_vect_of_boxes_in_a_license_plate;
	std::list < std::vector<float>>  l_vect_of_confidences_in_a_license_plate; std::list <std::vector<int>>  l_vect_of_classIds_in_a_license_plate;
	//double lists (one element list for each lp detected) of detected characters inside a lp
	std::list < std::vector<cv::Rect>>  l_vect_of_boxes_in_a_license_plate_tri_left;
	std::list < std::vector<float>>  l_vect_of_confidences_in_a_license_plate_tri_left; std::list <std::vector<int>>  l_vect_of_classIds_in_a_license_plate_tri_left;
	//the dnn has detected boxes that represent characters of the license plate, this function now groups characters in the same license plate and then rearranged from left to right.
	//it can deal with license pates that have two lines of charcaters
	separate_license_plates_if_necessary_add_blank_vehicles(
		//raw detections
		boxes, confidences, classIds,
		//detections when they are separated license plates by license plates
		lpns, l_vect_of_boxes_in_a_license_plate,
		l_vect_of_confidences_in_a_license_plate, l_vect_of_classIds_in_a_license_plate,
		//double lists (one element list for each lp detected) of detected characters inside a lp
		l_vect_of_boxes_in_a_license_plate_tri_left,
		l_vect_of_confidences_in_a_license_plate_tri_left, l_vect_of_classIds_in_a_license_plate_tri_left,
		classId_last_country);
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
		lpns.clear();
	}
	if (lpns.empty()) {
		//std::vector<cv::Rect> tri_left_vect_of_detected_boxes; std::vector<float> tri_left_confidences; std::vector<int> tri_left_classIds;
		std::string lpn = get_lpn(
			boxes, confidences, classIds,
			//characters inside lp
			tri_left_vect_of_detected_boxes, tri_left_confidences, tri_left_classIds,
			nmsThreshold
		);
		return lpn;
	}
	else {
		std::list<float> confidence_one_lp; std::list < cv::Rect> one_lp; std::list<int> classIds_one_lp;
		get_best_plate(
			//detections when they are separated license plates by license plates
			l_vect_of_classIds_in_a_license_plate, l_vect_of_confidences_in_a_license_plate, l_vect_of_boxes_in_a_license_plate
			//l_vect_of_classIds_in_a_license_plate_tri_left, l_vect_of_confidences_in_a_license_plate_tri_left,l_vect_of_boxes_in_a_license_plate_tri_left
			//output the list of the best (most probable/readable) lp
			, confidence_one_lp, one_lp, classIds_one_lp);
		return get_lpn(one_lp,
			confidence_one_lp, classIds_one_lp,
			//list of characters inside the lp
			tri_left_vect_of_detected_boxes,
			tri_left_confidences, tri_left_classIds,
			nmsThreshold
		);
	}
}
/*
//extracts, from a test directory, all images files
void load_images_filenames(const std::string& dir, std::list<std::string>& image_filenames)
{
	std::filesystem::path p(dir);
	std::vector<std::filesystem::directory_entry> v; // To save the file names in a vector.
	if (is_directory(p))
	{
		const std::string dir_path = p.string();
		std::filesystem::directory_iterator b(p), e;
		for (auto i = b; i != e; ++i)
		{
			if (std::filesystem::is_regular_file(*i)) {
				std::filesystem::path fe = i->path().extension();
				std::string extension = fe.string();
				if (extension == ".bmp" || extension == ".BMP" || extension == ".jpg" || extension == ".JPG" || extension == ".jpeg")
				{
					std::filesystem::path p_(i->path());
					//if you want to select images that have the true license plate number in the image filename
					const bool select_images_with_lpn = true;
					if (select_images_with_lpn) {
						bool vrai_lpn_after_underscore = true;
						//returns the true license plate number out of a filename
							//you must place the true license plate number in the image filename this way : number + underscore + license plate number,
							//for instance filename 0000000001_3065WWA34.jpg will be interpreted as an image with the license plate 3065WWA34 in it.
						std::string ExactLPN(getTrueLPN(p_.stem().string(), vrai_lpn_after_underscore));
						if (ExactLPN.size() > 3 && ExactLPN.size() < 11) {
							image_filenames.push_back(i->path().string());
						}
					}
					else {//take all images files -- output stats impossible
						image_filenames.push_back(i->path().string());
					}
				}
			}
		}
	}
}*/
//Padded resize
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
/*
void drawPred(int classId, int left, int top, int right, int bottom, cv::Mat& frame, const std::vector<std::string>& classes)
{
	cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 1);
	std::string label;// = cv::format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] //+ ": " + label
			;
	}
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = cv::max(top, labelSize.height);
	cv::rectangle(frame, cv::Point(left, top - labelSize.height), cv::Point(left + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
	cv::rectangle(frame, cv::Point(left, top - labelSize.height), cv::Point(left + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
	cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
}
*/
//if two boxes have an iou (intersection over union) that is too large, then they cannot represent two adjacent characters of the license plate 
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
	//if two boxes have an iou (intersection over union) that is too large, then they cannot represent two adjacent characters of the license plate 
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