#include"utils_opencv.h"
#include <filesystem>
#include "utils_anpr_detect.h"
#include "../include/StatSommesX_Y_H_dbl.h"
#include "../include/utils_image_file.h"
#define NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE 36
#define MIN_WIDTH_FOR_A_LP 60
#define MIN_HEIGHT_FOR_A_LP 10
#define MAX_SLOPE 0.35
#define RACINE_3_SUR_2 0.85
#define RACINE_2_SUR_2 0.707
#define NB_MIN_PIXELS_IN_PLATE	1000
void indeces_tri(std::list<cv::Rect>& boxes, //  the boxes detected by nn detector, stored in a list of rectagles objects
	std::list<float>& ious)
{
	if (boxes.size() == ious.size() && ious.size()) {
		//////////////////////////////////////////////////////////////////
		//TRI DE LA LISTE
		//////////////////////////////////////////////////////////////////
		//des listes pour trier (vides au debut puis se remplissant un par un, progressivement)
		std::list<cv::Rect> liste_tri_boxes;
		std::list<float> liste_tri_iou;
		while (!boxes.empty() && !ious.empty()) {
			float _iou = ious.front();
			std::list<float>::iterator it_tri_iou(liste_tri_iou.begin());
			std::list<cv::Rect>::iterator it_tri(liste_tri_boxes.begin());
			while (it_tri != liste_tri_boxes.end() && it_tri_iou != liste_tri_iou.end()) {
				if (_iou <= (*it_tri_iou)) break;
				else {
					it_tri_iou++;
					it_tri++;
				}
			}
			liste_tri_iou.splice(it_tri_iou, ious,
				ious.begin());
			liste_tri_boxes.splice(it_tri, boxes,
				boxes.begin());
		}
		//VERIF DE LA LISTE
#ifdef _DEBUG
		assert(boxes.empty() && ious.empty());
		std::list<float>::const_iterator it_verif(liste_tri_iou.begin());
		float _iou = -1.0f;
		while (it_verif != liste_tri_iou.end()) {
#ifdef _DEBUG
			assert((*it_verif) > -0.1f);
			assert((*it_verif) >= _iou);
#endif //_DEBUG
			_iou = (*it_verif);
			it_verif++;
		}
#endif //_DEBUG
		liste_tri_iou.swap(ious);
		liste_tri_boxes.swap(boxes);
	}
}
std::list<cv::Rect> fiter_out(
	const std::vector<cv::Rect>& true_boxes,  // the true boxes extracted from pascal voc xml file, as a list
	const std::list<cv::Rect>& detected_boxes, //  the boxes detected by nn detector, stored in a list of rectagles objects
	const std::list<float>& ious,
	std::list<float>& out_ious)
{
	std::list<cv::Rect> lboxes;
	vector_to_list(true_boxes, lboxes);
	return fiter_out(
		lboxes,  // the true boxes extracted from pascal voc xml file, as a list
		detected_boxes, //  the boxes detected by nn detector, stored in a list of rectagles objects
		ious,
		out_ious);
}
std::list<cv::Rect> fiter_out(
	const std::list<cv::Rect>& true_boxes,  // the true boxes extracted from pascal voc xml file, as a list
	const std::list<cv::Rect>& detected_boxes, //  the boxes detected by nn detector, stored in a list of rectagles objects
	const std::list<float>& ious,
	std::list<float>& out_ious)
{
	std::list<cv::Rect> return_list;
	if (detected_boxes.size() == ious.size() && true_boxes.size() && ious.size()) {
		std::list<cv::Rect> detected_boxes_copie = detected_boxes;  //  the boxes detected by nn detector, stored in a list of rectagles objects
		std::list<float>  ious_copie = ious;
		indeces_tri(detected_boxes_copie, //  the boxes detected by nn detector, stored in a list of rectagles objects
			ious_copie);
		const float inv_pow = 1.0f / (float)(true_boxes.size());
		float response = 1.0f;
		std::list<cv::Rect> true_boxes_copy = true_boxes;
		std::list<cv::Rect> detected_boxes_copy = detected_boxes;
		while (!detected_boxes_copy.empty() && !ious_copie.empty()) {
			float carac_iou = 0.0f;
			std::list<cv::Rect>::const_iterator it_true_boxes(true_boxes_copy.begin());
			while (it_true_boxes != true_boxes_copy.end()) {
				float current_iou = iou(detected_boxes_copy.back(), *it_true_boxes);
				if (carac_iou < current_iou) {
					carac_iou = current_iou;
				}
				it_true_boxes++;
			}
			if (carac_iou < 0.1f) {
				out_ious.push_back(ious_copie.back());
				return_list.push_back(detected_boxes_copy.back());
			}
			ious_copie.pop_back();
			detected_boxes_copy.pop_back();
		}
	}
	return return_list;
}
float iou(
	const std::list<cv::Rect>& true_boxes,  // the true boxes extracted from pascal voc xml file, as a list
	const std::list<cv::Rect>& detected_boxes //  the boxes detected by nn detector, stored in a list of rectagles objects
)
{
	if (detected_boxes.size() == true_boxes.size() && true_boxes.size()) {
		const float inv_pow = 1.0f / (float)(true_boxes.size());
		float response = 1.0f;
		std::list<cv::Rect> true_boxes_copy = true_boxes;
		std::list<cv::Rect> detected_boxes_copy = detected_boxes;
		while (!detected_boxes_copy.empty()) {
			float carac_iou = 0.0f;
			std::list<cv::Rect>::const_iterator it_true_boxes(true_boxes_copy.begin());
			std::list<cv::Rect>::const_iterator it_erase(it_true_boxes);
			while (it_true_boxes != true_boxes_copy.end()) {
				float current_iou = iou(detected_boxes_copy.front(), *it_true_boxes);
				if (carac_iou < current_iou) {
					carac_iou = current_iou;
					it_erase = it_true_boxes;
				}
				it_true_boxes++;
			}
			response *= carac_iou;
			true_boxes_copy.erase(it_erase);
			detected_boxes_copy.pop_front();
		}
		return float(pow(response, inv_pow));
	}
	return 0.0f;
}
float iou(
	const std::list<cv::Rect>& boxes,  // the true boxes extracted from pascal voc xml file, as a list
	const cv::Rect& box //  the boxes detected by nn detector, stored in a list of rectagles objects
)
{
	if (boxes.size()) {
		float carac_iou = 0.0f;
		std::list<cv::Rect>::const_iterator it_boxes(boxes.begin());
		while (it_boxes != boxes.end()) {
			float current_iou = iou(box, *it_boxes);
			if (carac_iou < current_iou) {
				carac_iou = current_iou;
			}
			it_boxes++;
		}
		return carac_iou;
	}
	else return 0.0f;
}
//cette fonction trie la liste de gauche a droite 
std::list<cv::Rect> sort_from_left_to_right(const std::list<cv::Rect>& boxes) {
	std::list<cv::Rect> copie = boxes;
	std::list<cv::Rect> l_tri_left;
	while (!copie.empty()) {
		int left_courant(copie.front().x);
		std::list<cv::Rect>::iterator it = l_tri_left.begin();
		while (it != l_tri_left.end()) {
			//if (left_courant <= it->x) break;
			if (is_on_the_left(copie.front(), *it))
				break;
			else it++;
		}
		l_tri_left.splice(it, copie, copie.begin());
	}
#ifdef _DEBUG
	assert(debug_left(l_tri_left));
	assert(copie.empty());
#endif	 //_DEBUG
	return l_tri_left;
}
//cette fonction trie la liste de gauche a droite 
//returns a list of characters of lp sorted from left to right
std::list<cv::Rect> sort_from_left_to_right(const std::list<cv::Rect>& boxes, const std::list<int>& classes, std::list<int>& sorted_classes,
	const int number_of_characters_latin_numberplate//, const int index_first_mark_model
) {
	//const int NUMBER_OF_COUNTRIES = index_first_mark_model - number_of_characters_latin_numberplate;
	//const int classId_last_country = number_of_characters_latin_numberplate + NUMBER_OF_COUNTRIES - 1;
	std::list<cv::Rect>::const_iterator it_box(boxes.begin());
	std::list<int>::const_iterator it_classes(classes.begin());
	std::list<cv::Rect> copie;
	while (it_box != boxes.end() && it_classes != classes.end()
		) {
		//given the index of a bounding box, we can predict if this box is a single character or if it represents the license plate area or if it is the roi of an entire vehicle
		//single character--> returns 1
		//license plate--> returns 2
		//negative index--> returns 0 must be an error
		//classId_last_country : is the class index of the last country in the list of detected classes.
		if (is_this_box_a_character(*it_classes, number_of_characters_latin_numberplate) == 1)//single character--> returns 1
		{
			copie.push_back(*it_box);
			sorted_classes.push_back(*it_classes);
		}
		it_classes++;
		it_box++;
	}
	sort_from_left_to_right(copie, sorted_classes);
	return copie;
}
			//cette fonction trie la liste de gauche a droite 
bool is_on_the_left(const cv::Rect& box1, const cv::Rect& box2)
{
	if (box1.x < box2.x) return true;
	else if (box1.x > box2.x) return false;
	else {
		if (box1.x + box1.width < box2.x + box2.width) return true;
		else if (box1.x + box1.width > box2.x + box2.width) return false;
		else {
			if (box1.y < box2.y) return true;
			else if (box1.y > box2.y) return false;
			else {
				if (box1.y + box1.height < box2.y + box2.height) return true;
				else if (box1.y + box1.height > box2.y + box2.height) return false;
				else return false;
			}
		}
	}
}
void sort_from_left_to_right(std::list<cv::Rect>& boxes, std::list<int>& classIds_)
{
	std::list<cv::Rect> l_tri_left;
	std::list<int> classIds_tri_left;
	while (!boxes.empty() && !classIds_.empty()) {
		//int left_courant(boxes.front().x);
		std::list<cv::Rect>::iterator it = l_tri_left.begin();
		std::list<int>::iterator it_classIds = classIds_tri_left.begin();
		while (it != l_tri_left.end() &&
			it_classIds != classIds_tri_left.end()) {
			if(is_on_the_left(boxes.front(), *it))
			//if (left_courant < it->x || (left_courant == it->x && left_courant + boxes.front().width < it->x + it->width)) 
				break;
			else {
				it++;  it_classIds++;
			}
		}
		l_tri_left.splice(it, boxes, boxes.begin());
		classIds_tri_left.splice(it_classIds, classIds_, classIds_.begin());
	}
#ifdef _DEBUG
	assert(boxes.empty());
	assert(debug_left(l_tri_left));
	//VERIFICATION DU TRI CROISSANT
#endif	  //_DEBUG
	l_tri_left.swap(boxes);
	classIds_tri_left.swap(classIds_);
}
//cette fonction trie la liste de gauche a droite 
void sort_from_left_to_right(std::list<cv::Rect>& boxes, std::list<float>& confidences)
{
	std::list<cv::Rect> l_tri_left;
	std::list<float> confidences_tri_left;
	while (!boxes.empty() && !confidences.empty()) {
		//int left_courant(boxes.front().x);
		std::list<cv::Rect>::iterator it = l_tri_left.begin();
		std::list<float>::iterator it_confidences = confidences_tri_left.begin();
		while (it != l_tri_left.end() && it_confidences != confidences_tri_left.end()) {
			//if (left_courant <= it->x)
			if (is_on_the_left(boxes.front(), *it))
				break;
			else {
				it++; it_confidences++;
			}
		}
		l_tri_left.splice(it, boxes, boxes.begin());
		confidences_tri_left.splice(it_confidences, confidences, confidences.begin());
	}
#ifdef _DEBUG
	assert(boxes.empty());
	//VERIFICATION DU TRI CROISSANT
	assert(debug_left(l_tri_left));
#endif	  //_DEBUG
	l_tri_left.swap(boxes);
	confidences_tri_left.swap(confidences);
}
bool is_in_rect(const cv::Rect& box, const cv::Rect& rect_im)
{
#ifdef _DEBUG
	/*
	assert(!(box.x<rect_im.x || box.y<rect_im.y
		|| box.x + box.width > rect_im.x + rect_im.width
		|| box.y + box.height> rect_im.y + rect_im.height));*/
#endif //_DEBUG
	return (!(box.x<rect_im.x || box.y<rect_im.y
		|| box.x + box.width > rect_im.x + rect_im.width
		|| box.y + box.height> rect_im.y + rect_im.height));
}
bool is_in_rect(const cv::Point& pt, const cv::Rect& rect_im)
{
	return (!(pt.x < rect_im.x || pt.y < rect_im.y || pt.x >= rect_im.x + rect_im.width || pt.y >= rect_im.y + rect_im.height));
}
//for each box in the container, check that it is nearly entirely contained in the second argument
bool is_in_rect_if(const std::list<cv::Rect>& boxes, const cv::Rect& rect_im)
{
	std::list<cv::Rect>::const_iterator it_boxes(boxes.begin());
	while (it_boxes != boxes.end()) {//return true if the intersection of the first argument box and the second has an area that is at least 90% of the first argument box (which means box is nearly entirely in the second argument)
		if (!is_in_rect_if(*it_boxes, rect_im))
			return false;
		else it_boxes++;
	}
	return true;
}
//for each box in the container, check that it is nearly entirely contained in the second argument
bool is_in_rect_if(const std::vector<cv::Rect>& boxes, const cv::Rect& rect_im)
{
	std::vector<cv::Rect>::const_iterator it_boxes(boxes.begin());
	while (it_boxes != boxes.end()) {//return true if the intersection of the first argument box and the second has an area that is at least 90% of the first argument box (which means box is nearly entirely in the second argument)
		if (!is_in_rect_if(*it_boxes, rect_im))
			return false;
		else it_boxes++;
	}
	return true;
}
bool is_in_rect(const std::list<cv::Rect>& boxes, const cv::Rect& rect_im)
{
	std::list<cv::Rect>::const_iterator it_boxes(boxes.begin());
	while (it_boxes != boxes.end()) {
		if (!is_in_rect(*it_boxes, rect_im))
			return false;
		else it_boxes++;
	}
	return true;
}
void get_mean_std(const cv::Mat& frame, const cv::Rect& box, float& mean, float& standard_deviation)
{
	if (box.x >= 0 && box.y >= 0 && box.width >= 0 && box.height >= 0 && box.x + box.width <= frame.cols && box.y + box.height <= frame.rows) {
		cv::Rect roi(box.x + box.width / 2, box.y, 1, box.height);
		cv::Mat tmp(frame(roi));
		cv::Scalar mean_, dev_;
		cv::meanStdDev(tmp, mean_, dev_);
		mean = float( mean_[0]); 
		standard_deviation = float(dev_[0]);
	}
	else {
		mean = 0.0f; standard_deviation = 0.0f;
	}
}
bool get_upperand_lower_lines
(const std::list<cv::Rect>& boxes, C_Line& line_sup, C_Line& line_inf)
{
	if (boxes.size() > 1) {
		C_SumsRegLineXYHDbl return_value;
		//initialisation des sommes partielles
		std::list<cv::Rect>::const_iterator it(boxes.begin());
		while (it != boxes.end()) {
			//update des sommes
			int somme(it->x + it->x + it->width);
			if (((somme >> 1) << 1) < somme) {
#ifdef _DEBUG
				assert(somme - ((somme >> 1) << 1) == 1);
#endif //_DEBUG
				somme = (somme >> 1);
				cv::Point2f Center(somme + 0.5f, intToFloat(it->y));
				return_value += Center;//le centre du bord sup de la region *it.
			}
			else {
				somme = (somme >> 1);
				cv::Point2f Center(intToFloat(somme), intToFloat(it->y));
				return_value += Center;//le centre du bord sup de la region *it.
			}
			it++;
		}
#ifdef _DEBUG
		//initialisation des sommes partielles
		float somme_x = 0.0f;
		float somme_y = 0.0f;
		float produit_xy = 0.0f;
		float somme_carre_x = 0.0f;
		std::list<cv::Rect>::const_iterator it_verif = boxes.begin();
		while (it_verif != boxes.end()) {
			//le centre du bord sup de la region *it.
			cv::Point2f Center;
			Center.y = intToFloat(it_verif->y);
			int somme(it_verif->x + it_verif->x + it_verif->width);
			if (((somme >> 1) << 1) < somme) {
#ifdef _DEBUG
				assert(somme - ((somme >> 1) << 1) == 1);
#endif //_DEBUG
				somme = (somme >> 1);
				Center.x = somme + 0.5f;
			}
			else {
				somme = (somme >> 1);
				Center.x = intToFloat(somme);
			}
			//update des sommes
			somme_x += Center.x;
			somme_y += Center.y;
			produit_xy += Center.x * Center.y;
			somme_carre_x += Center.x * Center.x;
			it_verif++;
		}
		return_value.somme_hauteurs = 1;
#ifdef _DEBUG
		assert(return_value.debug(somme_x, somme_y, produit_xy, somme_carre_x));
#endif //_DEBUG
#endif
		line_sup = return_value.regression_line(boxes.size());
		return_value.clear();
		it = boxes.begin();
		while (it != boxes.end()) {
			//update des sommes
			int somme(it->x + it->x + it->width);
			if (((somme >> 1) << 1) < somme) {
#ifdef _DEBUG
				assert(somme - ((somme >> 1) << 1) == 1);
#endif //_DEBUG
				somme = (somme >> 1);
				cv::Point2f Center(intToFloat(somme) + 0.5f, intToFloat(it->y + it->height));
				return_value += Center;//le centre du bord sup de la region *it.
			}
			else {
				somme = (somme >> 1);
				cv::Point2f Center(intToFloat(somme), intToFloat(it->y + it->height));
				return_value += Center;//le centre du bord sup de la region *it.
			}
			it++;
		}
		line_inf = return_value.regression_line(boxes.size());
		return (line_inf.a > -MAX_SLOPE && line_inf.a<MAX_SLOPE&& line_sup.a>-MAX_SLOPE && line_sup.a < MAX_SLOPE);
	}
	else {
		line_sup.a = 0.0f;
		line_sup.b = 0.0f;
		line_inf.a = 0.0f;
		line_inf.b = 0.0f;
		return false;
	}
}
cv::Rect  get_global_rect(const cv::Point& bottom_right,
	const 	cv::Point& bottom_left, const cv::Point& top_right
	, const cv::Point& top_left)
{
	int left = top_left.x;
	if (left > bottom_left.x)left = bottom_left.x;
	int right = top_right.x;
	if (right < bottom_right.x)right = bottom_right.x;
	int bottom = bottom_right.y;
	if (bottom < bottom_left.y)bottom = bottom_left.y;
	int top = top_right.y;
	if (top > top_left.y)top = top_left.y;
	return cv::Rect(left, top, right - left, bottom - top);
}
void vector_to_list(
	const std::vector<cv::Rect>& lboxes,
	std::list<cv::Rect>& boxes) {
	std::copy(lboxes.begin(), lboxes.end(), std::back_inserter(boxes));
}
//cette fonction retourne le rect englobant la collection
//gets the reunion of all the boxes
cv::Rect get_global_rect(const std::list<cv::Rect>& l)
{
	if (l.empty()) return cv::Rect(0, 0, 0, 0);
	else {
		std::list<cv::Rect>::const_iterator it(l.begin());
		cv::Rect rect(*it);
		it++;
		while (it != l.end()) {
			if (rect.x > it->x) {
				rect.width += rect.x - it->x; rect.x = it->x;
			}
			if (rect.y > it->y) {
				rect.height += rect.y - it->y; rect.y = it->y;
			}
			if (rect.x + rect.width < it->x + it->width) rect.width = it->x + it->width - rect.x;
			if (rect.y + rect.height < it->y + it->height) rect.height = it->y + it->height - rect.y;
			it++;
		}
		return rect;
	}
}
//cette fonction retourne le rect englobant la collection
//gets the reunion of all the boxes
cv::Rect get_global_rect(const std::vector<cv::Rect>& l)
{
	if (l.empty()) return cv::Rect(0, 0, 0, 0);
	else {
		std::vector<cv::Rect>::const_iterator it(l.begin());
		cv::Rect rect(*it);
		it++;
		while (it != l.end()) {
			if (rect.x > it->x) {
				rect.width += rect.x - it->x; rect.x = it->x;
			}
			if (rect.y > it->y) {
				rect.height += rect.y - it->y; rect.y = it->y;
			}
			if (rect.x + rect.width < it->x + it->width) rect.width = it->x + it->width - rect.x;
			if (rect.y + rect.height < it->y + it->height) rect.height = it->y + it->height - rect.y;
			it++;
		}
		return rect;
	}
}
cv::Rect array2CRect(const cv::Point& p0, const cv::Point& p1,
	const cv::Point& p2, const cv::Point& p3,
	cv::Point& top_left,
	cv::Point& top_right,
	cv::Point& bottom_right,
	cv::Point& bottom_left
)
{
	if (p0.x == SHRT_MIN) {
#ifdef _DEBUG
		assert(p0.x == SHRT_MIN && p0.y == SHRT_MIN &&
			p1.x == SHRT_MIN && p1.y == SHRT_MIN &&
			p2.x == SHRT_MIN && p2.y == SHRT_MIN &&
			p3.x == SHRT_MIN && p3.y == SHRT_MIN);
#endif //_DEBUG
		return cv::Rect(-1, -1, -1, -1);
	}
	else {
#ifdef _DEBUG
		assert(p0.x >= 0 && p0.y >= 0 &&
			p1.x >= 0 && p1.y >= 0 &&
			p2.x >= 0 && p2.y >= 0 &&
			p3.x >= 0 && p3.y >= 0);
#endif //_DEBUG
		int resultat[4][2];
		resultat[0][0] = p0.x;
		resultat[0][1] = p0.y;
		resultat[1][0] = p1.x;
		resultat[1][1] = p1.y;
		resultat[2][0] = p2.x;
		resultat[2][1] = p2.y;
		resultat[3][0] = p3.x;
		resultat[3][1] = p3.y;
		//trouver top_left
		//c'est le point dont la somme des corrdonnes est la plus petite
		int somme_coord_min = SHRT_MAX;
		int somme_coord_max = SHRT_MIN;
		int index_top_left(-1);
		int index_bottom_right(-1);
		unsigned int i;
		for (i = 0; i < 4; i++)
		{
			const int somme_coord = resultat[i][0] + resultat[i][1];
			if (somme_coord < somme_coord_min) {
				somme_coord_min = somme_coord;
				index_top_left = i;
			}
			if (somme_coord > somme_coord_max) {
				somme_coord_max = somme_coord;
				index_bottom_right = i;
			}
		}
		top_left.x = resultat[index_top_left][0];
		top_left.y = resultat[index_top_left][1];
		bottom_right.x = resultat[index_bottom_right][0];
		bottom_right.y = resultat[index_bottom_right][1];
		//trouver top_right
		int right(SHRT_MIN);
		int index_top_right(-1);
		for (i = 0; i < 4; i++)
		{
			if (i != index_top_left && i != index_bottom_right) {
				if (resultat[i][0] - resultat[i][1] >= right) {
					right = resultat[i][0] - resultat[i][1];
					top_right.x = resultat[i][0];
					top_right.y = resultat[i][1];
					index_top_right = i;
				}
			}
		}
		//trouver bottom_left
		int left(resultat[index_top_left][0]);
		int index_bottom_left(-1);
		for (i = 0; i < 4; i++)
		{
			if (i != index_top_left && i != index_bottom_right && i != index_top_right) {
				index_bottom_left = i;
				bottom_left.x = resultat[i][0];
				bottom_left.y = resultat[i][1];
				break;
			}
		}
		int gauche = resultat[0][0];
		int droite = resultat[0][0];
		int haut = resultat[0][1];
		int bas = resultat[0][1];
		for (i = 1; i < 4; i++)
		{
#ifdef _DEBUG
			assert(gauche <= droite);
#endif //_DEBUG
			if (resultat[i][0] < gauche)
				gauche = resultat[i][0];
			else if (resultat[i][0] > droite)
				droite = resultat[i][0];
#ifdef _DEBUG
			assert(haut <= bas);
#endif //_DEBUG
			if (resultat[i][1] < haut)
				haut = resultat[i][1];
			else if (resultat[i][1] > bas)
				bas = resultat[i][1];
		}
		return cv::Rect(gauche, haut, droite - gauche + 1, bas - haut + 1);
	}
}
std::vector<cv::Point2f> get_points(const CvSeq* polygone)
{
	std::vector<cv::Point2f> vertVect;
	for (int i = 0; i < polygone->total; i++) {
		CvPoint* pt = CV_GET_SEQ_ELEM(CvPoint, polygone, i);
		vertVect.push_back(cv::Point2f(pt->x, pt->y));
	}
	return vertVect;
}
cv::RotatedRect get_rect(const CvSeq* polygone)
{
	std::vector<cv::Point2f> vertVect(get_points(polygone));
	cv::RotatedRect calculatedRect = cv::minAreaRect(vertVect);
	return calculatedRect;
}
void swap(CvSeq* polygone, const cv::RotatedRect& RotatedRect)
{
	int total = polygone->total;
	if (total >= 4) {
		cv::Point2f vertices[4];
		RotatedRect.points(vertices);
		std::list<cv::Point2f> list_sommets;
		for (int j = 0; j < 4; j++) {
			list_sommets.push_back(vertices[j]);
		}
		std::list<cv::Point2f> liste_sommets_tri;
		list_sommets.swap(liste_sommets_tri);
		while (!liste_sommets_tri.empty()) {
			std::list<cv::Point2f>::iterator it_sommets(liste_sommets_tri.begin());
			float sum = it_sommets->x + it_sommets->y;
			std::list<cv::Point2f>::iterator it = list_sommets.begin();
			while (it != list_sommets.end()) {
				if (sum <= it->x + it->y) break;
				it++;
			}
			list_sommets.splice(it, liste_sommets_tri, it_sommets);
		}
		cv::Point2f top_left;
		cv::Point2f top_right;
		cv::Point2f bottom_left;
		cv::Point2f bottom_right;
		std::list<cv::Point2f>::iterator it = list_sommets.begin();
		if (it != list_sommets.end()) {
			top_left = *it;
			it++;
		}
		if (it != list_sommets.end()) {
			bottom_left = *it;
			it++;
		}
		if (it != list_sommets.end()) {
			top_right = *it;
			it++;
		}
		if (it != list_sommets.end()) {
			bottom_right = *it;
		}
		float top_left_distance = FLT_MAX;
		float top_right_distance = FLT_MAX;
		float bottom_left_distance = FLT_MAX;
		float bottom_right_distance = FLT_MAX;
		cv::Point2f new_top_left;
		cv::Point2f new_top_right;
		cv::Point2f new_bottom_left;
		cv::Point2f new_bottom_right;
		for (int i = 0; i < total; i++) {
			CvPoint* pt = CV_GET_SEQ_ELEM(CvPoint, polygone, i);
			float current_distance_top_left_distance
				= (pt->x - top_left.x) * (pt->x - top_left.x) +
				(pt->y - top_left.y) * (pt->y - top_left.y);
			if (top_left_distance > current_distance_top_left_distance) {
				new_top_left = *pt;
			}
			float current_distance_bottom_left_distance
				= (pt->x - bottom_left.x) * (pt->x - bottom_left.x) +
				(pt->y - bottom_left.y) * (pt->y - bottom_left.y);
			if (bottom_left_distance > current_distance_bottom_left_distance) {
				new_bottom_left = *pt;
			}
			float current_distance_top_right_distance
				= (pt->x - top_right.x) * (pt->x - top_right.x) +
				(pt->y - top_right.y) * (pt->y - top_right.y);
			if (top_right_distance > current_distance_top_right_distance) {
				new_top_right = *pt;
			}
			float current_distance_bottom_right_distance
				= (pt->x - bottom_right.x) * (pt->x - bottom_right.x) +
				(pt->y - bottom_right.y) * (pt->y - bottom_right.y);
			if (bottom_right_distance > current_distance_bottom_right_distance) {
				new_bottom_right = *pt;
			}
		}
		for (int i = 0; i < total - 4; i++) {
			cvSeqRemove(polygone, 0);
		}
		int i = 0;
		CvPoint* pt = CV_GET_SEQ_ELEM(CvPoint, polygone, i);
		pt->x = new_top_left.x;
		pt->y = new_top_left.y;
		i++;
		pt = CV_GET_SEQ_ELEM(CvPoint, polygone, i);
		pt->x = new_top_right.x;
		pt->y = new_top_right.y;
		i++;
		pt = CV_GET_SEQ_ELEM(CvPoint, polygone, i);
		pt->x = new_bottom_right.x;
		pt->y = new_bottom_right.y;
		i++;
		pt = CV_GET_SEQ_ELEM(CvPoint, polygone, i);
		pt->x = new_bottom_left.x;
		pt->y = new_bottom_left.y;
	}
}
void init_contours_weight(const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min,
	const int widthimage, const int heightimage,
	CvMemStorage* storage, CvSeq* contours,
	CvSeq* squares,
	// diffeentes seuences contentant les notes correspondant aux diffeents critees 
	CvSeq* note_plus_grand_cosinus,
	CvSeq* note_rapport_long_cotes_opposes,//means largeur_sur_hauteur
	CvSeq* note_plus_petit_sinus,
	CvSeq* note_plus_grand_sinus,
	CvSeq* notesym,
	float cvApproxPoly_coeff)
{
	int i, index_element_a_zapper;
	//cvApproxPoly_coeff regulation
	if (cvApproxPoly_coeff < .0) cvApproxPoly_coeff = -cvApproxPoly_coeff;
	if (cvApproxPoly_coeff < .0 || cvApproxPoly_coeff>1.0f) cvApproxPoly_coeff = 0.05f;
	CvPoint pt[4];
	const int nb_max_contours = 20;//11
	// teste chaque contour
	while (contours)
	{
		// approximate contour with accuracy proportional
		// to the contour perimeter
		CvSeq* polygone = cvApproxPoly(contours, sizeof(CvContour), storage,
			CV_POLY_APPROX_DP, (cvContourPerimeter(contours
			)) * cvApproxPoly_coeff, 0);
		// le 0.02 est un indice d'approximation. il n'est pas etre judicieux 
		// de le modifier, mieux vaut utiliser la methode transformant toutes
		// sortes de polygones (pentagones ou +) en quadrilatees
		// si il y a max 10 cee (cette valeur est arbitraire)
		// et que le polygone est fermeet a une aire suffisante pour ere inteessant
		if (polygone->total < nb_max_contours && polygone->total>3)
		{
			//The function cvCheckContourConvexity tests whether the input 
			//contour is convex or not. The contour must be simple, i.e. 
			//without self-intersections. 
			double area = fabsf(doubleToFloat(cvContourArea(polygone, CV_WHOLE_SEQ)));
			if (area > NB_MIN_PIXELS_IN_PLATE)
			{
				if (
					cvCheckContourConvexity(polygone)
					) {//a remttre
					//tant qu'il y a plus de 4 cee
					while (polygone->total > 4)
					{
						index_element_a_zapper = zappeUnPoint(polygone, storage);//on regarde si un point est inutile
						if (index_element_a_zapper != -1)//si c'est non on arree
						{
							cvSeqRemove(polygone, index_element_a_zapper);
						}
						else//si c'est oui on retire ce point inutile
						{
							cv::RotatedRect RotatedRect(get_rect(polygone));
							double rotated_area = RotatedRect.size.area();
							int angle_degree = abs(int(RotatedRect.angle));
							div_t divresult;
							divresult = div(angle_degree, 90);
							cv::Rect boundingrect(RotatedRect.boundingRect());
							if (rotated_area * 0.85 < area && divresult.rem < 25
								&& boundingrect.width>2.5 * boundingrect.height)
							{
								swap(polygone, RotatedRect);
							}
							else//si c'est oui on retire ce point inutile
							{
								cvClearSeq(polygone);
							}
						}
					}
				}
				else//si c'est oui on retire ce point inutile
				{
					cv::RotatedRect RotatedRect(get_rect(polygone));
					double rotated_area = RotatedRect.size.area();
					int angle_degree = abs(int(RotatedRect.angle));
					div_t divresult;
					divresult = div(angle_degree, 90);
					cv::Rect boundingrect(RotatedRect.boundingRect());
					if (rotated_area * 0.85 < area && divresult.rem < 25
						&& boundingrect.width>2.5 * boundingrect.height)
					{
						swap(polygone, RotatedRect);
					}
					else//si c'est oui on retire ce point inutile
					{
						cvClearSeq(polygone);
					}
					//cvClearSeq(polygone); 
				}
			}
			else//si c'est oui on retire ce point inutile
			{
				cvClearSeq(polygone);
			}
		}
		// square contours should have 4 vertices after approximation
		// relatively large area (to filter out noisy contours)
		// and be convex.
		// Note: absolute value of an area is used because
		// area may be positive or negative - in accordance with the
		// contour orientation
		// si le quadrilatee obtenu a 4 cee
		// et que les conditions preeemment eonces sont veifies
		if (polygone)
		{
			if (polygone->total == 4)
			{
				if ( //The function cvCheckContourConvexity tests whether the input 
					//contour is convex or not. The contour must be simple, i.e. 
						//without self-intersections.
					cvCheckContourConvexity(polygone) &&
					fabsf(doubleToFloat(cvContourArea(polygone, CV_WHOLE_SEQ))) > NB_MIN_PIXELS_IN_PLATE
					)
				{
					float plus_grand_cosinus = -1.0;
					float t, plus_petit_sinus = 1.0, plus_grand_sinus = -1.0, //sin between two opposite sides of the lp //the greatest sin between two opposite sides of the lp
						sym,//means ratio between difference and sum of left and right side or ratio between difference and sum of top and bottom side (in fact max of these two ratio)
							// we want these ratio to be as small as possible
						petit//le min de la somme de deux cote oppose
						;
					float rapport_long_cotes_opposes = -1.0;//means largeur_sur_hauteur
					CvPoint p;
					float dist_plus_grand_cote = 0.0f;
					float cos_plus_grand_cote, anglePetitCote;//=cos between width and horizontal
					CvPoint p1;
					bool colleAuxBords = false;
					//************************************************
					//   CALC DES GRANDEURS GEO POUR UNE PLAQUE
					//************************************************
					for (i = 0; i < 5 && !colleAuxBords; i++)
					{
						// store all the contour vertices in the buffer
						// (pt sert estocker le quadrilatee eudie
						// afin de simplifier les expressions)
						pt[i & 3] = *(CvPoint*)cvGetSeqElemLPR(polygone, i, 0);
						//colleAuxBords&=(pt)&&()
						//epartir de i = 2 on peut faire un angle
						if (i >= 2)
						{
							//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
							t = fabsf(cosine(pt[i & 3], pt[(i - 2) & 3], pt[(i - 1) & 3]));
							if (plus_grand_cosinus < t) plus_grand_cosinus = t;
							//plus_grand_cosinus = MAX( plus_grand_cosinus, t );// on garde le + grand cos d'angle du quadrilatee
						}
						// on regarde si le + grand ceeforme 1 angle
						// infeieur ou eal 45 eseuil'horizontale
						if (i >= 1)
						{
							//  on regarde si on a pas tout simplement detecte toute l'image
							// c a d si tous les points ne seraient pas dans les coins de l'image
							colleAuxBords = colleAuxBords || (pt[i & 3].x < dist_min_bord) || (pt[i & 3].x > widthimage - dist_min_bord) ||
								(pt[i & 3].y < dist_min_bord) || (pt[i & 3].y > heightimage - dist_min_bord);
							if (dist(pt[i & 3], pt[(i - 1) & 3]) > dist_plus_grand_cote)
							{
								p1.y = pt[i & 3].y;
								p1.x = pt[(i - 1) & 3].x;
								//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
								cos_plus_grand_cote = fabsf(cosine(p1, pt[(i - 1) & 3], pt[i & 3]));//pt[(i - 1) & 3]->pt[i & 3]=WIDTH //=cos between width and horizontal
								dist_plus_grand_cote = dist(pt[i & 3], pt[(i - 1) & 3]);
							}
						}
						// infeieur ou eal 45 eseuil'horizontale
						if (i >= 3)
						{
							// on garde le + petit sinus d'angle entre 2 cotes opposes
							p.x = pt[(i - 3) & 3].x - pt[(i - 2) & 3].x + pt[(i - 1) & 3].x;
							p.y = pt[(i - 3) & 3].y - pt[(i - 2) & 3].y + pt[(i - 1) & 3].y;
							//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
							float cosinus_courant(cosine(pt[i & 3], p, pt[(i - 1) & 3]));
#ifdef _DEBUG
							assert(cosinus_courant > -1.0 - FLT_EPSILON &&
								cosinus_courant < 1.0f + FLT_EPSILON);
#endif //_DEBUG
							t = sqrtf(1 - (cosinus_courant * cosinus_courant));
							if (plus_petit_sinus > t)plus_petit_sinus = t;//sin between two opposite sides of the lp
							//plus_petit_sinus le +petit sinus (trapee)
							if (plus_grand_sinus < t) plus_grand_sinus = t;//the greatest sin between two opposite sides of the lp
							//plus_grand_sinus le +grand sinus (paralleogramme)
						}
						if (i == 4)
						{
							float d3 = dist(pt[1], pt[0]);
							float d0 = dist(pt[0], pt[3]);
							float d1 = dist(pt[3], pt[2]);
							float d2 = dist(pt[2], pt[1]);
							///verifie le repport des longueurs du quadrilatee
							if ((d0 + d2) > (d1 + d3)) rapport_long_cotes_opposes = (d0 + d2) / (d1 + d3);//means largeur_sur_hauteur
							else rapport_long_cotes_opposes = (d1 + d3) / (d0 + d2);
							//sym mesure le plus petit eart en proportions entre 2 cotes opposes
							sym = MAX(fabsf(d0 - d2) / (d0 + d2),
								fabsf(d1 - d3) / (d1 + d3));//means ratio between difference and sum of left and right side or ratio between difference and sum of top and bottom side (in fact max of these two ratio)
							// we want these ratio to be as small as possible
							// on utilise petit pour mettre la + petite somme de longueurs de 2 cotes opposes
							// qui doit ere au moins de 36 (soit 2*18)
							petit = MIN((d0 + d2), (d1 + d3));//means hauteur
							//on verifie si les 2 plus petits cote (en fait le plus petit cotedes 2 derniers controles et celui lui faisant face)
							// sont suffisament verticaux
							if (d0 < d1)//d0 and d2 are height and d1 , d3 are width : pt[3]->pt[2]= WIDTH , pt[0]->pt[3]=HEIGHT
							{
								p1.y = pt[0].y;
								p1.x = pt[3].x;
								//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
								anglePetitCote = fabsf(cosine(p1, pt[0], pt[3]));//pt[0]->pt[3]=HEIGHT. p1->pt[3] =  cos of angle betwween vertical and  pt[0]->pt[3]=HEIGHT
								p1.y = pt[2].y;
								p1.x = pt[1].x;
								anglePetitCote = MIN(anglePetitCote, fabsf(cosine(p1, pt[2], pt[1])));
							}
							else
							{
								p1.y = pt[3].y;
								p1.x = pt[2].x;
								//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
								anglePetitCote = fabsf(cosine(p1, pt[3], pt[2]));
								p1.y = pt[1].y;
								p1.x = pt[0].x;
								anglePetitCote = MIN(anglePetitCote, fabsf(cosine(p1, pt[1], pt[0])));
							}
							anglePetitCote = sqrtf(1.0f - anglePetitCote * anglePetitCote);
						}
					}
#ifdef _DEBUG
					assert(rapport_long_cotes_opposes >= 1.0 || rapport_long_cotes_opposes < -0.5f);//means largeur_sur_hauteur
#endif //_DEBUG
					cv::Point top_left, top_right, bottom_right, bottom_left;
					cv::Point P0(pt[0].x, pt[0].y);
					cv::Point P1(pt[1].x, pt[1].y);
					cv::Point P2(pt[2].x, pt[2].y);
					cv::Point P3(pt[3].x, pt[3].y);
					// on n'applique une note que si le quadrilatee semble inteessant
					// (pas de forme aberrante)
					if ((rapport_long_cotes_opposes > rapport_largeur_sur_hauteur_min)//means largeur_sur_hauteur always true
						&& ((rapport_long_cotes_opposes < 10) && (rapport_long_cotes_opposes > 0.164))//(rapport_long_cotes_opposes > 0.164) always true since (rapport_long_cotes_opposes >=1)
						//(rapport_long_cotes_opposes < 10) //largeur = 7* hauteur 
						&& (plus_petit_sinus < 0.5f)//sin between two opposite sides of the lp
						&& (plus_grand_sinus < 0.8f)//the greatest sin between two opposite sides of the lp
						&& (sym < 0.5f)//means ratio between difference and sum of left and right side or ratio between difference and sum of top and bottom side (in fact max of these two ratio)
							// we want these ratio to be as small as possible
						//le min de la somme de deux cote oppose
						&& (petit > (hauteur_plaque_min << 1))//means hauteur
						&& (cos_plus_grand_cote > RACINE_2_SUR_2)//=cos between width and horizontal so this means that the width must not be inclined of more than 45 degrees 
						&& !colleAuxBords
						&& (anglePetitCote < 0.3)
						&&
						//must be convex quadrilatere
						get_corners(P0, P1, P2, P3,
							top_left, top_right, bottom_right, bottom_left)
						)
						//FILTRE
					{
						// on stocke les quadrilatees
						for (i = 0; i < 4; i++)
						{
							cvSeqPush(squares, cvGetSeqElemLPR(polygone, i, 0));
						}
						// on stocke les notes correspondantes
						cvSeqPush(note_plus_grand_cosinus, &plus_grand_cosinus);
#ifdef _DEBUG
						assert(plus_grand_cosinus > -1.0 - FLT_EPSILON
							&& plus_grand_cosinus < 1.0f + FLT_EPSILON);
						assert(rapport_long_cotes_opposes >= 1.0f);//means largeur_sur_hauteur always true
#endif //_DEBUG
						rapport_long_cotes_opposes = fabsf(rapport_long_cotes_opposes - 5.04f);//means largeur_sur_hauteur always true 5.04 ?????????????
						cvSeqPush(note_rapport_long_cotes_opposes, &rapport_long_cotes_opposes);
#ifdef _DEBUG
						assert(plus_petit_sinus > -1.0 - FLT_EPSILON &&//sin between two opposite sides of the lp
							plus_petit_sinus < 1.0f + FLT_EPSILON);
#endif //_DEBUG
						cvSeqPush(note_plus_petit_sinus, &plus_petit_sinus);//sin between two opposite sides of the lp
						cvSeqPush(note_plus_grand_sinus, &plus_grand_sinus);//the greatest sin between two opposite sides of the lp
						cvSeqPush(notesym, &sym);//means ratio between difference and sum of left and right side or ratio between difference and sum of top and bottom side (in fact max of these two ratio)
							// we want these ratio to be as small as possible
						//NOTES
					}
				}
			}
		}
		// on passe au suivant
		contours = contours->h_next;
	}
	/*seqence stock�e ds storage donc m�moire lib�r�e avec storage
	if(contours)
	cvClearSeq(contours);*/
}
void init_contours_weight(const IplImage* im, const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min,
	const int widthimage, const int heightimage,
	CvMemStorage* storage, CvSeq* contours,
	CvSeq* squares,
	// diffeentes seuences contentant les notes correspondant aux diffeents critees 
	CvSeq* note_plus_grand_cosinus,
	CvSeq* note_rapport_long_cotes_opposes,//means largeur_sur_hauteur
	CvSeq* note_plus_petit_sinus,
	CvSeq* note_plus_grand_sinus,
	CvSeq* notesym,
	float cvApproxPoly_coeff)
{
	int i, index_element_a_zapper;
	//cvApproxPoly_coeff regulation
	if (cvApproxPoly_coeff < .0) cvApproxPoly_coeff = -cvApproxPoly_coeff;
	if (cvApproxPoly_coeff < .0 || cvApproxPoly_coeff>1.0f) cvApproxPoly_coeff = 0.05f;
	CvPoint pt[4];
	const int nb_max_contours = 20;//11
	// teste chaque contour
	while (contours)
	{
		// approximate contour with accuracy proportional
		// to the contour perimeter
		CvSeq* polygone = cvApproxPoly(contours, sizeof(CvContour), storage,
			CV_POLY_APPROX_DP, (cvContourPerimeter(contours
			)) * cvApproxPoly_coeff, 0);
		// le 0.02 est un indice d'approximation. il n'est pas etre judicieux 
		// de le modifier, mieux vaut utiliser la methode transformant toutes
		// sortes de polygones (pentagones ou +) en quadrilatees
		// si il y a max 10 cee (cette valeur est arbitraire)
		// et que le polygone est fermeet a une aire suffisante pour ere inteessant
		if (polygone->total < nb_max_contours && polygone->total>3)
		{
			//The function cvCheckContourConvexity tests whether the input 
			//contour is convex or not. The contour must be simple, i.e. 
			//without self-intersections. 
			double area = fabsf(doubleToFloat(cvContourArea(polygone, CV_WHOLE_SEQ)));
			if (area > NB_MIN_PIXELS_IN_PLATE)
			{
				if (
					cvCheckContourConvexity(polygone)
					) {//a remttre
					//tant qu'il y a plus de 4 cee
					while (polygone->total > 4)
					{
						index_element_a_zapper = zappeUnPoint(polygone, storage);//on regarde si un point est inutile
						if (index_element_a_zapper != -1)//si c'est non on arree
						{
							cvSeqRemove(polygone, index_element_a_zapper);
						}
						else//si c'est oui on retire ce point inutile
						{
							cv::RotatedRect RotatedRect(get_rect(polygone));
							double rotated_area = RotatedRect.size.area();
							int angle_degree = abs(int(RotatedRect.angle));
							div_t divresult;
							divresult = div(angle_degree, 90);
							cv::Rect boundingrect(RotatedRect.boundingRect());
							if (rotated_area * 0.85 < area && divresult.rem < 25
								&& boundingrect.width>2.5 * boundingrect.height)
							{
								swap(polygone, RotatedRect);
							}
							else//si c'est oui on retire ce point inutile
							{
								cvClearSeq(polygone);
							}
						}
					}
				}
				else//si c'est oui on retire ce point inutile
				{
					cv::RotatedRect RotatedRect(get_rect(polygone));
					double rotated_area = RotatedRect.size.area();
					int angle_degree = abs(int(RotatedRect.angle));
					div_t divresult;
					divresult = div(angle_degree, 90);
					cv::Rect boundingrect(RotatedRect.boundingRect());
					if (rotated_area * 0.85 < area && divresult.rem < 25
						&& boundingrect.width>2.5 * boundingrect.height)
					{
						swap(polygone, RotatedRect);
					}
					else//si c'est oui on retire ce point inutile
					{
						cvClearSeq(polygone);
					}
					//cvClearSeq(polygone); 
				}
			}
			else//si c'est oui on retire ce point inutile
			{
				cvClearSeq(polygone);
			}
		}
		// square contours should have 4 vertices after approximation
		// relatively large area (to filter out noisy contours)
		// and be convex.
		// Note: absolute value of an area is used because
		// area may be positive or negative - in accordance with the
		// contour orientation
		// si le quadrilatee obtenu a 4 cee
		// et que les conditions preeemment eonces sont veifies
		if (polygone)
		{
			if (polygone->total == 4)
			{
				if ( //The function cvCheckContourConvexity tests whether the input 
					//contour is convex or not. The contour must be simple, i.e. 
						//without self-intersections.
					cvCheckContourConvexity(polygone) &&
					fabsf(doubleToFloat(cvContourArea(polygone, CV_WHOLE_SEQ))) > NB_MIN_PIXELS_IN_PLATE
					)
				{
					float plus_grand_cosinus = -1.0;
					float t, plus_petit_sinus = 1.0f, plus_grand_sinus = -1.0f, //sin between two opposite sides of the lp //the greatest sin between two opposite sides of the lp
						sym = -1.0f,//means ratio between difference and sum of left and right side or ratio between difference and sum of top and bottom side (in fact max of these two ratio)
							// we want these ratio to be as small as possible
						petit = 0.0f//le min de la somme de deux cote oppose
						;
					float rapport_long_cotes_opposes = -1.0;//means largeur_sur_hauteur
					CvPoint p;
					float dist_plus_grand_cote = 0.0f;
					float cos_plus_grand_cote = -2.0f, anglePetitCote = -2.0f;//=cos between width and horizontal
					CvPoint p1;
					bool colleAuxBords = false;
					//************************************************
					//   CALC DES GRANDEURS GEO POUR UNE PLAQUE
					//************************************************
					for (i = 0; i < 5// && !colleAuxBords
						; i++)
					{
						// store all the contour vertices in the buffer
						// (pt sert estocker le quadrilatee eudie
						// afin de simplifier les expressions)
						pt[i & 3] = *(CvPoint*)cvGetSeqElemLPR(polygone, i, 0);
						//colleAuxBords&=(pt)&&()
						//epartir de i = 2 on peut faire un angle
						if (i >= 2)
						{
							//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
							t = fabsf(cosine(pt[i & 3], pt[(i - 2) & 3], pt[(i - 1) & 3]));
							if (plus_grand_cosinus < t) plus_grand_cosinus = t;
							//plus_grand_cosinus = MAX( plus_grand_cosinus, t );// on garde le + grand cos d'angle du quadrilatee
						}
						// on regarde si le + grand ceeforme 1 angle
						// infeieur ou eal 45 eseuil'horizontale
						if (i >= 1)
						{
							//  on regarde si on a pas tout simplement detecte toute l'image
							// c a d si tous les points ne seraient pas dans les coins de l'image
							colleAuxBords = colleAuxBords || (pt[i & 3].x < dist_min_bord) || (pt[i & 3].x > widthimage - dist_min_bord) ||
								(pt[i & 3].y < dist_min_bord) || (pt[i & 3].y > heightimage - dist_min_bord);
							if (dist(pt[i & 3], pt[(i - 1) & 3]) > dist_plus_grand_cote)
							{
								p1.y = pt[i & 3].y;
								p1.x = pt[(i - 1) & 3].x;
								//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
								cos_plus_grand_cote = fabsf(cosine(p1, pt[(i - 1) & 3], pt[i & 3]));//pt[(i - 1) & 3]->pt[i & 3]=WIDTH //=cos between width and horizontal
								dist_plus_grand_cote = dist(pt[i & 3], pt[(i - 1) & 3]);
							}
						}
						// infeieur ou eal 45 eseuil'horizontale
						if (i >= 3)
						{
							// on garde le + petit sinus d'angle entre 2 cotes opposes
							p.x = pt[(i - 3) & 3].x - pt[(i - 2) & 3].x + pt[(i - 1) & 3].x;
							p.y = pt[(i - 3) & 3].y - pt[(i - 2) & 3].y + pt[(i - 1) & 3].y;
							//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
							float cosinus_courant(cosine(pt[i & 3], p, pt[(i - 1) & 3]));
#ifdef _DEBUG
							assert(cosinus_courant > -1.0 - FLT_EPSILON &&
								cosinus_courant < 1.0f + FLT_EPSILON);
#endif //_DEBUG
							t = sqrtf(1 - (cosinus_courant * cosinus_courant));
							if (plus_petit_sinus > t)plus_petit_sinus = t;//sin between two opposite sides of the lp
							//plus_petit_sinus le +petit sinus (trapee)
							if (plus_grand_sinus < t) plus_grand_sinus = t;//the greatest sin between two opposite sides of the lp
							//plus_grand_sinus le +grand sinus (paralleogramme)
						}
						if (i == 4)
						{
							float d3 = dist(pt[1], pt[0]);
							float d0 = dist(pt[0], pt[3]);
							float d1 = dist(pt[3], pt[2]);
							float d2 = dist(pt[2], pt[1]);
							///verifie le repport des longueurs du quadrilatee
							if ((d0 + d2) > (d1 + d3)) rapport_long_cotes_opposes = (d0 + d2) / (d1 + d3);//means largeur_sur_hauteur
							else rapport_long_cotes_opposes = (d1 + d3) / (d0 + d2);
							//sym mesure le plus petit eart en proportions entre 2 cotes opposes
							sym = MAX(fabsf(d0 - d2) / (d0 + d2),
								fabsf(d1 - d3) / (d1 + d3));//means ratio between difference and sum of left and right side or ratio between difference and sum of top and bottom side (in fact max of these two ratio)
							// we want these ratio to be as small as possible
							// on utilise petit pour mettre la + petite somme de longueurs de 2 cotes opposes
							// qui doit ere au moins de 36 (soit 2*18)
							petit = MIN((d0 + d2), (d1 + d3));//means hauteur
							//on verifie si les 2 plus petits cote (en fait le plus petit cotedes 2 derniers controles et celui lui faisant face)
							// sont suffisament verticaux
							if (d0 < d1)//d0 and d2 are height and d1 , d3 are width : pt[3]->pt[2]= WIDTH , pt[0]->pt[3]=HEIGHT
							{
								p1.y = pt[0].y;
								p1.x = pt[3].x;
								//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
								anglePetitCote = fabsf(cosine(p1, pt[0], pt[3]));//pt[0]->pt[3]=HEIGHT. p1->pt[3] =  cos of angle betwween vertical and  pt[0]->pt[3]=HEIGHT
								p1.y = pt[2].y;
								p1.x = pt[1].x;
								anglePetitCote = MIN(anglePetitCote, fabsf(cosine(p1, pt[2], pt[1])));
							}
							else
							{
								p1.y = pt[3].y;
								p1.x = pt[2].x;
								//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
								anglePetitCote = fabsf(cosine(p1, pt[3], pt[2]));
								p1.y = pt[1].y;
								p1.x = pt[0].x;
								anglePetitCote = MIN(anglePetitCote, fabsf(cosine(p1, pt[1], pt[0])));
							}
							anglePetitCote = sqrtf(1.0f - anglePetitCote * anglePetitCote);
						}
					}
#ifdef _DEBUG
					assert(rapport_long_cotes_opposes >= 1.0 || rapport_long_cotes_opposes < -0.5f);//means largeur_sur_hauteur
#endif //_DEBUG
					cv::Point top_left, top_right, bottom_right, bottom_left;
					cv::Point P0(pt[0].x, pt[0].y);
					cv::Point P1(pt[1].x, pt[1].y);
					cv::Point P2(pt[2].x, pt[2].y);
					cv::Point P3(pt[3].x, pt[3].y);
					bool corners = (get_corners(im, pt[0], pt[1], pt[2], pt[3], top_left, top_right, bottom_right, bottom_left));
					// on n'applique une note que si le quadrilatee semble inteessant
					// (pas de forme aberrante)
					if (corners &&
						(rapport_long_cotes_opposes > rapport_largeur_sur_hauteur_min)//means largeur_sur_hauteur always true
						&& (rapport_long_cotes_opposes < 10.0f)
						&& (rapport_long_cotes_opposes > 0.164f)//(rapport_long_cotes_opposes > 0.164) always true since (rapport_long_cotes_opposes >=1)
						//(rapport_long_cotes_opposes < 10) //largeur = 7* hauteur 
						&& (plus_petit_sinus < 0.5f)//sin between two opposite sides of the lp
						&& (plus_grand_sinus < 0.8f)//the greatest sin between two opposite sides of the lp
						&& (sym < 0.5f)//means ratio between difference and sum of left and right side or ratio between difference and sum of top and bottom side (in fact max of these two ratio)
							// we want these ratio to be as small as possible
						//le min de la somme de deux cote oppose
						&& (petit > (hauteur_plaque_min / 2))//means hauteur
						&& (cos_plus_grand_cote > 0.707f)//=cos between width and horizontal so this means that the width must not be inclined of more than 45 degrees 
						&& (!colleAuxBords)
						&& (anglePetitCote < 0.3f)
						//&& (get_corners(im, pt[0], pt[1], pt[2], pt[3],top_left, top_right, bottom_right, bottom_left))
						)
						//FILTRE
					{
						// on stocke les quadrilatees
						for (i = 0; i < 4; i++)
						{
							cvSeqPush(squares, cvGetSeqElemLPR(polygone, i, 0));
						}
						// on stocke les notes correspondantes
						cvSeqPush(note_plus_grand_cosinus, &plus_grand_cosinus);
#ifdef _DEBUG
						assert(plus_grand_cosinus > -1.0 - FLT_EPSILON
							&& plus_grand_cosinus < 1.0f + FLT_EPSILON);
						assert(rapport_long_cotes_opposes >= 1.0f);//means largeur_sur_hauteur always true
#endif //_DEBUG
						rapport_long_cotes_opposes = fabsf(rapport_long_cotes_opposes - 5.04f);//means largeur_sur_hauteur always true 5.04 ?????????????
						cvSeqPush(note_rapport_long_cotes_opposes, &rapport_long_cotes_opposes);
#ifdef _DEBUG
						assert(plus_petit_sinus > -1.0 - FLT_EPSILON &&//sin between two opposite sides of the lp
							plus_petit_sinus < 1.0f + FLT_EPSILON);
#endif //_DEBUG
						cvSeqPush(note_plus_petit_sinus, &plus_petit_sinus);//sin between two opposite sides of the lp
						cvSeqPush(note_plus_grand_sinus, &plus_grand_sinus);//the greatest sin between two opposite sides of the lp
						cvSeqPush(notesym, &sym);//means ratio between difference and sum of left and right side or ratio between difference and sum of top and bottom side (in fact max of these two ratio)
							// we want these ratio to be as small as possible
						//NOTES
					}
				}
			}
		}
		// on passe au suivant
		contours = contours->h_next;
	}
	/*seqence stock�e ds storage donc m�moire lib�r�e avec storage
	if(contours)
	cvClearSeq(contours);*/
}
cv::Rect get_inter_(const cv::Rect& r1, const cv::Rect& r2)
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
float iou_(const cv::Rect& r1, const cv::Rect& r2)
{
	cv::Rect inter(get_inter_(r1, r2));
	if (inter.width > 0 && inter.height > 0 && r1.width > 0 && r1.height > 0 && r2.width > 0 && r2.height > 0) {
		return (float)(inter.area()) / (float)(r1.area() + r2.area() - inter.area());
	}
	else return 0.0f;
}
void init_contours_weight(const IplImage* im, const cv::Rect& global_rect, const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min,
	const int widthimage, const int heightimage,
	CvMemStorage* storage, CvSeq* contours,
	CvSeq* squares,
	// diffeentes seuences contentant les notes correspondant aux diffeents critees 
	CvSeq* note_plus_grand_cosinus,
	CvSeq* note_rapport_long_cotes_opposes,//means largeur_sur_hauteur
	CvSeq* note_plus_petit_sinus,
	CvSeq* note_plus_grand_sinus,
	CvSeq* notesym,
	float cvApproxPoly_coeff)
{
	int i, index_element_a_zapper;
	//cvApproxPoly_coeff regulation
	if (cvApproxPoly_coeff < .0) cvApproxPoly_coeff = -cvApproxPoly_coeff;
	if (cvApproxPoly_coeff < .0 || cvApproxPoly_coeff>1.0f) cvApproxPoly_coeff = 0.05f;
	CvPoint pt[4];
	const int nb_max_contours = 20;//11
	// teste chaque contour
	while (contours)
	{
		// approximate contour with accuracy proportional
		// to the contour perimeter
		CvSeq* polygone = cvApproxPoly(contours, sizeof(CvContour), storage,
			CV_POLY_APPROX_DP, (cvContourPerimeter(contours
			)) * cvApproxPoly_coeff, 0);
		// le 0.02 est un indice d'approximation. il n'est pas etre judicieux 
		// de le modifier, mieux vaut utiliser la methode transformant toutes
		// sortes de polygones (pentagones ou +) en quadrilatees
		// si il y a max 10 cee (cette valeur est arbitraire)
		// et que le polygone est fermeet a une aire suffisante pour ere inteessant
		if (polygone->total < nb_max_contours && polygone->total>3)
		{
			//The function cvCheckContourConvexity tests whether the input 
			//contour is convex or not. The contour must be simple, i.e. 
			//without self-intersections. 
			double area = fabsf(doubleToFloat(cvContourArea(polygone, CV_WHOLE_SEQ)));
			if (area > NB_MIN_PIXELS_IN_PLATE)
			{
				if (
					cvCheckContourConvexity(polygone)
					) {//a remttre
					//tant qu'il y a plus de 4 cee
					while (polygone->total > 4)
					{
						index_element_a_zapper = zappeUnPoint(polygone, storage);//on regarde si un point est inutile
						if (index_element_a_zapper != -1)//si c'est non on arree
						{
							cvSeqRemove(polygone, index_element_a_zapper);
						}
						else//si c'est oui on retire ce point inutile
						{
							cv::RotatedRect RotatedRect(get_rect(polygone));
							double rotated_area = RotatedRect.size.area();
							int angle_degree = abs(int(RotatedRect.angle));
							div_t divresult;
							divresult = div(angle_degree, 90);
							cv::Rect boundingrect(RotatedRect.boundingRect());
							if (rotated_area * 0.85 < area && divresult.rem < 25
								&& boundingrect.width>2.5 * boundingrect.height)
							{
								swap(polygone, RotatedRect);
							}
							else//si c'est oui on retire ce point inutile
							{
								cvClearSeq(polygone);
							}
						}
					}
				}
				else//si c'est oui on retire ce point inutile
				{
					cv::RotatedRect RotatedRect(get_rect(polygone));
					double rotated_area = RotatedRect.size.area();
					int angle_degree = abs(int(RotatedRect.angle));
					div_t divresult;
					divresult = div(angle_degree, 90);
					cv::Rect boundingrect(RotatedRect.boundingRect());
					if (rotated_area * 0.85 < area && divresult.rem < 25
						&& boundingrect.width>2.5 * boundingrect.height)
					{
						swap(polygone, RotatedRect);
					}
					else//si c'est oui on retire ce point inutile
					{
						cvClearSeq(polygone);
					}
					//cvClearSeq(polygone); 
				}
				if (polygone) {
					if (polygone->total >= 4)
					{
						cv::RotatedRect RotatedRect_(get_rect(polygone));
						cv::Rect global_rotated_rect = RotatedRect_.boundingRect();
						if (iou_(global_rect, global_rotated_rect) < 0.3f) {
							cvClearSeq(polygone);
						}
					}
				}
			}
			else//si c'est oui on retire ce point inutile
			{
				cvClearSeq(polygone);
			}
		}
		// square contours should have 4 vertices after approximation
		// relatively large area (to filter out noisy contours)
		// and be convex.
		// Note: absolute value of an area is used because
		// area may be positive or negative - in accordance with the
		// contour orientation
		// si le quadrilatee obtenu a 4 cee
		// et que les conditions preeemment eonces sont veifies
		if (polygone)
		{
			if (polygone->total == 4)
			{
				if ( //The function cvCheckContourConvexity tests whether the input 
					//contour is convex or not. The contour must be simple, i.e. 
						//without self-intersections.
					cvCheckContourConvexity(polygone) &&
					fabsf(doubleToFloat(cvContourArea(polygone, CV_WHOLE_SEQ))) > NB_MIN_PIXELS_IN_PLATE
					)
				{
					float plus_grand_cosinus = -1.0;
					float t, plus_petit_sinus = 1.0f, plus_grand_sinus = -1.0f, //sin between two opposite sides of the lp //the greatest sin between two opposite sides of the lp
						sym = -1.0f,//means ratio between difference and sum of left and right side or ratio between difference and sum of top and bottom side (in fact max of these two ratio)
							// we want these ratio to be as small as possible
						petit = 0.0f//le min de la somme de deux cote oppose
						;
					float rapport_long_cotes_opposes = -1.0;//means largeur_sur_hauteur
					CvPoint p;
					float dist_plus_grand_cote = 0.0f;
					float cos_plus_grand_cote = -2.0f, anglePetitCote = -2.0f;//=cos between width and horizontal
					CvPoint p1;
					bool colleAuxBords = false;
					//************************************************
					//   CALC DES GRANDEURS GEO POUR UNE PLAQUE
					//************************************************
					for (i = 0; i < 5// && !colleAuxBords
						; i++)
					{
						// store all the contour vertices in the buffer
						// (pt sert estocker le quadrilatee eudie
						// afin de simplifier les expressions)
						pt[i & 3] = *(CvPoint*)cvGetSeqElemLPR(polygone, i, 0);
						//colleAuxBords&=(pt)&&()
						//epartir de i = 2 on peut faire un angle
						if (i >= 2)
						{
							//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
							t = fabsf(cosine(pt[i & 3], pt[(i - 2) & 3], pt[(i - 1) & 3]));
							if (plus_grand_cosinus < t) plus_grand_cosinus = t;
							//plus_grand_cosinus = MAX( plus_grand_cosinus, t );// on garde le + grand cos d'angle du quadrilatee
						}
						// on regarde si le + grand ceeforme 1 angle
						// infeieur ou eal 45 eseuil'horizontale
						if (i >= 1)
						{
							//  on regarde si on a pas tout simplement detecte toute l'image
							// c a d si tous les points ne seraient pas dans les coins de l'image
							colleAuxBords = colleAuxBords || (pt[i & 3].x < dist_min_bord) || (pt[i & 3].x > widthimage - dist_min_bord) ||
								(pt[i & 3].y < dist_min_bord) || (pt[i & 3].y > heightimage - dist_min_bord);
							if (dist(pt[i & 3], pt[(i - 1) & 3]) > dist_plus_grand_cote)
							{
								p1.y = pt[i & 3].y;
								p1.x = pt[(i - 1) & 3].x;
								//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
								cos_plus_grand_cote = fabsf(cosine(p1, pt[(i - 1) & 3], pt[i & 3]));//pt[(i - 1) & 3]->pt[i & 3]=WIDTH //=cos between width and horizontal
								dist_plus_grand_cote = dist(pt[i & 3], pt[(i - 1) & 3]);
							}
						}
						// infeieur ou eal 45 eseuil'horizontale
						if (i >= 3)
						{
							// on garde le + petit sinus d'angle entre 2 cotes opposes
							p.x = pt[(i - 3) & 3].x - pt[(i - 2) & 3].x + pt[(i - 1) & 3].x;
							p.y = pt[(i - 3) & 3].y - pt[(i - 2) & 3].y + pt[(i - 1) & 3].y;
							//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
							float cosinus_courant(cosine(pt[i & 3], p, pt[(i - 1) & 3]));
#ifdef _DEBUG
							assert(cosinus_courant > -1.0 - FLT_EPSILON &&
								cosinus_courant < 1.0f + FLT_EPSILON);
#endif //_DEBUG
							t = sqrtf(1 - (cosinus_courant * cosinus_courant));
							if (plus_petit_sinus > t)plus_petit_sinus = t;//sin between two opposite sides of the lp
							//plus_petit_sinus le +petit sinus (trapee)
							if (plus_grand_sinus < t) plus_grand_sinus = t;//the greatest sin between two opposite sides of the lp
							//plus_grand_sinus le +grand sinus (paralleogramme)
						}
						if (i == 4)
						{
							float d3 = dist(pt[1], pt[0]);
							float d0 = dist(pt[0], pt[3]);
							float d1 = dist(pt[3], pt[2]);
							float d2 = dist(pt[2], pt[1]);
							///verifie le repport des longueurs du quadrilatee
							if ((d0 + d2) > (d1 + d3)) rapport_long_cotes_opposes = (d0 + d2) / (d1 + d3);//means largeur_sur_hauteur
							else rapport_long_cotes_opposes = (d1 + d3) / (d0 + d2);
							//sym mesure le plus petit eart en proportions entre 2 cotes opposes
							sym = MAX(fabsf(d0 - d2) / (d0 + d2),
								fabsf(d1 - d3) / (d1 + d3));//means ratio between difference and sum of left and right side or ratio between difference and sum of top and bottom side (in fact max of these two ratio)
							// we want these ratio to be as small as possible
							// on utilise petit pour mettre la + petite somme de longueurs de 2 cotes opposes
							// qui doit ere au moins de 36 (soit 2*18)
							petit = MIN((d0 + d2), (d1 + d3));//means hauteur
							//on verifie si les 2 plus petits cote (en fait le plus petit cotedes 2 derniers controles et celui lui faisant face)
							// sont suffisament verticaux
							if (d0 < d1)//d0 and d2 are height and d1 , d3 are width : pt[3]->pt[2]= WIDTH , pt[0]->pt[3]=HEIGHT
							{
								p1.y = pt[0].y;
								p1.x = pt[3].x;
								//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
								anglePetitCote = fabsf(cosine(p1, pt[0], pt[3]));//pt[0]->pt[3]=HEIGHT. p1->pt[3] =  cos of angle betwween vertical and  pt[0]->pt[3]=HEIGHT
								p1.y = pt[2].y;
								p1.x = pt[1].x;
								anglePetitCote = MIN(anglePetitCote, fabsf(cosine(p1, pt[2], pt[1])));
							}
							else
							{
								p1.y = pt[3].y;
								p1.x = pt[2].x;
								//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
								anglePetitCote = fabsf(cosine(p1, pt[3], pt[2]));
								p1.y = pt[1].y;
								p1.x = pt[0].x;
								anglePetitCote = MIN(anglePetitCote, fabsf(cosine(p1, pt[1], pt[0])));
							}
							anglePetitCote = sqrtf(1.0f - anglePetitCote * anglePetitCote);
						}
					}
#ifdef _DEBUG
					assert(rapport_long_cotes_opposes >= 1.0 || rapport_long_cotes_opposes < -0.5f);//means largeur_sur_hauteur
#endif //_DEBUG
					cv::Point top_left, top_right, bottom_right, bottom_left;
					cv::Point P0(pt[0].x, pt[0].y);
					cv::Point P1(pt[1].x, pt[1].y);
					cv::Point P2(pt[2].x, pt[2].y);
					cv::Point P3(pt[3].x, pt[3].y);
					bool corners = (get_corners(im, pt[0], pt[1], pt[2], pt[3], top_left, top_right, bottom_right, bottom_left));
					// on n'applique une note que si le quadrilatee semble inteessant
					// (pas de forme aberrante)
					if (corners &&
						(rapport_long_cotes_opposes > rapport_largeur_sur_hauteur_min)//means largeur_sur_hauteur always true
						&& (rapport_long_cotes_opposes < 10.0f)
						&& (rapport_long_cotes_opposes > 0.164f)//(rapport_long_cotes_opposes > 0.164) always true since (rapport_long_cotes_opposes >=1)
						//(rapport_long_cotes_opposes < 10) //largeur = 7* hauteur 
						&& (plus_petit_sinus < 0.5f)//sin between two opposite sides of the lp
						&& (plus_grand_sinus < 0.8f)//the greatest sin between two opposite sides of the lp
						&& (sym < 0.5f)//means ratio between difference and sum of left and right side or ratio between difference and sum of top and bottom side (in fact max of these two ratio)
							// we want these ratio to be as small as possible
						//le min de la somme de deux cote oppose
						&& (petit > (hauteur_plaque_min / 2))//means hauteur
						&& (cos_plus_grand_cote > 0.707f)//=cos between width and horizontal so this means that the width must not be inclined of more than 45 degrees 
						&& (!colleAuxBords)
						&& (anglePetitCote < 0.3f)
						//&& (get_corners(im, pt[0], pt[1], pt[2], pt[3],top_left, top_right, bottom_right, bottom_left))
						)
						//FILTRE
					{
						// on stocke les quadrilatees
						for (i = 0; i < 4; i++)
						{
							cvSeqPush(squares, cvGetSeqElemLPR(polygone, i, 0));
						}
						// on stocke les notes correspondantes
						cvSeqPush(note_plus_grand_cosinus, &plus_grand_cosinus);
#ifdef _DEBUG
						assert(plus_grand_cosinus > -1.0 - FLT_EPSILON
							&& plus_grand_cosinus < 1.0f + FLT_EPSILON);
						assert(rapport_long_cotes_opposes >= 1.0f);//means largeur_sur_hauteur always true
#endif //_DEBUG
						rapport_long_cotes_opposes = fabsf(rapport_long_cotes_opposes - 5.04f);//means largeur_sur_hauteur always true 5.04 ?????????????
						cvSeqPush(note_rapport_long_cotes_opposes, &rapport_long_cotes_opposes);
#ifdef _DEBUG
						assert(plus_petit_sinus > -1.0 - FLT_EPSILON &&//sin between two opposite sides of the lp
							plus_petit_sinus < 1.0f + FLT_EPSILON);
#endif //_DEBUG
						cvSeqPush(note_plus_petit_sinus, &plus_petit_sinus);//sin between two opposite sides of the lp
						cvSeqPush(note_plus_grand_sinus, &plus_grand_sinus);//the greatest sin between two opposite sides of the lp
						cvSeqPush(notesym, &sym);//means ratio between difference and sum of left and right side or ratio between difference and sum of top and bottom side (in fact max of these two ratio)
							// we want these ratio to be as small as possible
						//NOTES
					}
				}
			}
		}
		// on passe au suivant
		contours = contours->h_next;
	}
	/*seqence stock�e ds storage donc m�moire lib�r�e avec storage
	if(contours)
	cvClearSeq(contours);*/
}
void LPRThreshold(const IplImage* src, IplImage*& dst,
	const int threshold, int& nb_pts_sup_threshold)
{
#ifdef _DEBUG
	assert(dst->tileInfo == NULL);
	//assert(dst->align == IPL_ALIGN_DWORD);
	assert(dst->depth == IPL_DEPTH_8U && dst->nChannels == 1);
	assert(dst->widthStep >= src->width
		&& dst->widthStep < src->width + 4);
	assert(dst->origin == 0);
#endif //_DEBUG
	// 0 - top-left origin
#ifdef _DEBUG
#ifdef _DEBUG
	assert(src->tileInfo == NULL);
	//assert(src->align == IPL_ALIGN_DWORD);
	assert(src->depth == IPL_DEPTH_8U && src->nChannels == 1);
	//assert(src->widthStep >= src->width && src->widthStep < src->width + 4);
	assert(src->origin == 0);
#endif //_DEBUG
#endif //_DEBUG
	// 0 - top-left origin
	if (dst->width == src->width &&
		dst->height == src->height
		&& dst->depth == IPL_DEPTH_8U
		&& src->depth == IPL_DEPTH_8U
		&& dst->nChannels == 1
		&& src->nChannels == 1) {
		int width = (dst->width % dst->align == 0) ?
			dst->width :
			((dst->width / dst->align) + 1) * dst->align;
#ifdef _DEBUG
		assert(dst->widthStep == width);
#endif //_DEBUG
		int nb_byte_data = width * dst->height;
		int alignement = width - dst->width;
#ifdef _DEBUG
		assert(alignement >= 0 && alignement < 4);
#endif //_DEBUG
		nb_pts_sup_threshold = 0;
		for (int i = 0; i < src->height; ++i) {	//pour chaque ligne
			unsigned char* ptr_cv_dst = (unsigned char*)dst->imageData + i * dst->widthStep;
			unsigned char* ptr_cv_src = (unsigned char*)src->imageData + i * src->widthStep;
			for (int j = 0; j < src->width; ++j)
			{
				if (*ptr_cv_src > threshold) {
					*ptr_cv_dst = 127;
					nb_pts_sup_threshold++;
				}
				else *ptr_cv_dst = 0;
				ptr_cv_dst++;
				ptr_cv_src++;
			}
		}
		// Save the Output Image
		// int cvSaveImage( const char* filename, const CvArr* image );
	}
	else {
		nb_pts_sup_threshold = -1;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
// renvoie une seuence de 4 points encadrant la plaque
// trouveesur l'image im_src
// IplImage* im_src
// CvMemStorage* storage 
//peut traiter les images sources couleur ou en niveaux de gris
CvSeq* findSquares4
(IplImage* im_src, const cv::Rect& global_rect, CvMemStorage* storage,
	const int nTresh, const int mean_carac
	, const int mean_fond, //
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min)
{
	CvSeq* contours = 0;
	CvPoint centre;
	int  i, j, k, l, m,
		un = 1,
		pond, max;
	int thresh_canny = 50;
	CvSize sz = cvSize(im_src->width, im_src->height);
	//quelques images temporaires qui permettront de travailler
	IplImage* gray1 = cvCreateImage(sz, IPL_DEPTH_8U//unsigned 8-bit integers im_src->depth
		, 1);
	IplImage* tgray = cvCreateImage(sz, IPL_DEPTH_8U//im_src->depth
		, 1);
	IplImage* gray2 = cvCreateImage(sz, IPL_DEPTH_8U//unsigned 8-bit integers im_src->depth
		, 1);
	int filterSize = 5;
	IplConvKernel* convKernel = cvCreateStructuringElementEx
	(filterSize, filterSize, (filterSize - 1) / 2, (filterSize - 1) / 2,
		CV_SHAPE_ELLIPSE, NULL);
#ifdef _DEBUG
	assert(im_src->nChannels == im_src->nChannels);
	assert(im_src->depth == im_src->depth);
	assert(im_src->align == im_src->align);
#endif //_DEBUG
	// squares et classement permettront de travailler sur les listes de quadrilatees
	CvSeq* squares = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	CvSeq* classement = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	// centres est une liste de centres de quadrilatees utilisee ela fin
	CvSeq* centres = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	// diffeentes seuences contentant les notes correspondant aux diffeents critees 
	CvSeq* note_plus_grand_cosinus = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* note_rapport_long_cotes_opposes = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* note_plus_petit_sinus = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* note_plus_grand_sinus = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* notesym = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	// diffeentes seuences contenant les places correspondant aux diffeents critees
	CvSeq* place_plus_grand_cosinus = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* place_rapport_long_cotes_opposes = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* place_plus_petit_sinus = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* place_plus_grand_sinus = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* placesym = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	// place contiendra la liste de sommes de places
	CvSeq* place = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	//classementplace contiendra la liste des quadrilatees ordonnee selon les sommes de places
	CvSeq* classementplace = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	// selectionne le + grand rectangle dans l'image
	// dont les dimensions sont divisibles par 2
#ifdef _DEBUG
	assert(im_src->width == sz.width);
	assert(im_src->height == sz.height);
#endif //_DEBUG
	//		cvSetImageROI( im_src, cvRect( 0, 0, sz.width, sz.height ));
	// pt servira eplusieurs reprises estocker des quadrilatees
	// (tableau de 4 points)
	CvPoint pt[4];
	// on teste toutes les couleurs R,G,B
	//const int sz.width=im_src->width;const int sz.height=im_src->height;
	cvCopy(im_src, tgray, 0);
	//si on est pas en NB
	//on recopie ds une image de travail
	bool prochaine_image_gray1 = true;
	// on teste N niveaux de seuillage
	// apply Canny. Take the upper threshold from slider
	// and set the lower to 0 (which forces edges merging) 
	cvCanny(tgray, gray1, thresh_canny, 255 - thresh_canny, 5//Aperture parameter for Sobel operator
	);
	// dilate canny output to remove potential
	// holes between edge segments 
	//cvErode( gray1, gray1, 0, 1 );//a rajouter
	// trouve les contours et en fait une liste
	float cvApproxPoly_coeff = 0.02f;
	cvFindContours(gray1, storage, &contours, sizeof(CvContour),
		CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	// teste chaque contour
	//calcule le poids des contours
	init_contours_weight(tgray, global_rect, dist_min_bord,
		hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
		sz.width, sz.height,
		storage, contours,
		squares,
		// diffeentes seuences contentant les notes correspondant aux diffeents critees 
		note_plus_grand_cosinus,
		note_rapport_long_cotes_opposes,
		note_plus_petit_sinus,
		note_plus_grand_sinus,
		notesym, cvApproxPoly_coeff);
	if (!squares->total) {
		cv::Mat img = cv::cvarrToMat(tgray);
		cv::Mat image_final = cv::cvarrToMat(gray1);
		int blockSize = (sz.width + sz.height) / 2;
		if (blockSize % 2 == 0) blockSize += 1;
		cv::adaptiveThreshold(img, image_final, 127, cv::ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, blockSize, 0);
		cvFindContours(gray1, storage, &contours, sizeof(CvContour),
			CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		// teste chaque contour
		//calcule le poids des contours
		int total_number_of_plates = squares->total;
		init_contours_weight(tgray, global_rect, dist_min_bord,
			hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
			sz.width, sz.height,
			storage, contours,
			squares,
			// diffeentes seuences contentant les notes correspondant aux diffeents critees 
			note_plus_grand_cosinus,
			note_rapport_long_cotes_opposes,
			note_plus_petit_sinus,
			note_plus_grand_sinus,
			notesym, cvApproxPoly_coeff);
		if (total_number_of_plates == squares->total) {
			int nb_pts_sup_threshold = (sz.width * sz.height);
			int new_mean_carac = mean_carac;
			int new_mean_fond = mean_fond;
			if (mean_carac > mean_fond) {
				new_mean_carac = mean_fond;
				new_mean_fond = mean_carac;
				new_mean_carac -= (mean_carac - mean_fond) / 10;
				new_mean_fond += (mean_carac - mean_fond) / 10;
			}
			else {
				new_mean_carac -= (mean_fond - mean_carac) / 10;
				new_mean_fond += (mean_fond - mean_carac) / 10;
			}
#ifdef _DEBUG
			assert(new_mean_carac <= new_mean_fond);
#endif //_DEBUG
			if (new_mean_carac < 0) new_mean_carac = 0;
			if (new_mean_fond > 255) new_mean_fond = 255;
			int new_nTresh = nTresh;
			if (new_nTresh > (new_mean_fond - new_mean_carac) >> 1)
				new_nTresh = (new_mean_fond - new_mean_carac) >> 1;
			int seuil;
			for (seuil = 0; seuil < new_nTresh; seuil++)
			{
				// sinon on applique le seuillage
				///((tgray(x,y) = gray(x,y) < (seuil+1)*255/N ? 255 : 0))
				int current_nb_pts_sup_threshold = 0;
				if (new_mean_carac < new_mean_fond)
					LPRThreshold(im_src, gray1,
						new_mean_carac + ((new_mean_fond - new_mean_carac) * seuil) / new_nTresh,
						current_nb_pts_sup_threshold);// trouve les contours et en fait une liste
				else LPRThreshold(im_src, gray1,
					new_mean_fond + ((new_mean_carac - new_mean_fond) * seuil) / new_nTresh,
					current_nb_pts_sup_threshold);
				if (current_nb_pts_sup_threshold >= 0 &&
					current_nb_pts_sup_threshold <= nb_pts_sup_threshold) {
					///////////////////////////////////////////////////////////////////////////////////////////
					// retourne le nb de points en commun entre deux images
#ifdef _DEBUG
					assert(current_nb_pts_sup_threshold <= nb_pts_sup_threshold);
#endif //_DEBUG
					float percentage_match = 1.0f - ((float)(nb_pts_sup_threshold -
						current_nb_pts_sup_threshold)) / (sz.width * sz.height);
#ifdef _DEBUG
					assert(percentage_match > -FLT_EPSILON &&
						percentage_match < 1.0f + FLT_EPSILON);
#endif //_DEBUG
					if (percentage_match < 0.75) {
						nb_pts_sup_threshold = current_nb_pts_sup_threshold;
						cvFindContours(gray1, storage, &contours, sizeof(CvContour),
							CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
						total_number_of_plates = squares->total;
						init_contours_weight(tgray, global_rect, dist_min_bord,
							hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
							sz.width, sz.height,
							storage, contours,
							squares,
							// diffeentes seuences contentant les notes correspondant aux diffeents critees 
							note_plus_grand_cosinus,
							note_rapport_long_cotes_opposes,
							note_plus_petit_sinus,
							note_plus_grand_sinus,
							notesym, cvApproxPoly_coeff);
						if (seuil == 0) cvApproxPoly_coeff = 0.02f;
						else if (seuil == 1) cvApproxPoly_coeff = 0.03f;
						else if (seuil == 2) cvApproxPoly_coeff = 0.04f;
						else cvApproxPoly_coeff = 0.05f;//if(seuil==3) 
														// teste chaque contour
														//calcule le poids des contours
						if (total_number_of_plates == squares->total) {
							cvMorphologyEx(gray1, gray2, NULL, convKernel, CV_MOP_CLOSE);
							cvFindContours(gray2, storage, &contours, sizeof(CvContour),
								CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
							init_contours_weight(tgray, global_rect, dist_min_bord,
								hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
								sz.width, sz.height,
								storage, contours,
								squares,
								// diffeentes seuences contentant les notes correspondant aux diffeents critees 
								note_plus_grand_cosinus,
								note_rapport_long_cotes_opposes,
								note_plus_petit_sinus,
								note_plus_grand_sinus,
								notesym, cvApproxPoly_coeff);
						}
					}
				}
			}
		}
	}
	// (on libee de la meoire)
	cvReleaseImage(&gray1);
	cvReleaseImage(&tgray);
	cvReleaseImage(&gray2);
	cvReleaseStructuringElement(&convKernel);
#ifdef _DEBUG
	assert(note_plus_grand_cosinus->total == note_rapport_long_cotes_opposes->total &&
		note_plus_grand_cosinus->total == note_plus_petit_sinus->total &&
		note_plus_grand_cosinus->total == note_plus_grand_sinus->total &&
		note_plus_grand_cosinus->total == notesym->total);
#endif //_DEBUG
	//si rien n'a passele filtre, on arree le:
	//on renvoie une seuence de -1
	if (notesym->total == 0)
	{
		cvClearSeq(classement);
		centre = cvPoint(SHRT_MIN, SHRT_MIN);
		cvSeqPush(classement, &centre);
		cvSeqPush(classement, &centre);
		cvSeqPush(classement, &centre);
		cvSeqPush(classement, &centre);
		return classement;
	}
	// CLASSEMENT
	// renvoie la seuence de places epartir de la seuence de notes
	place_plus_grand_cosinus = sort(note_plus_grand_cosinus, storage);
	place_rapport_long_cotes_opposes = sort(note_rapport_long_cotes_opposes, storage);
	place_plus_petit_sinus = sort(note_plus_petit_sinus, storage);
	place_plus_grand_sinus = sort(note_plus_grand_sinus, storage);
	placesym = sort(notesym, storage);
#ifdef _DEBUG
	assert(place_plus_grand_cosinus->total == place_rapport_long_cotes_opposes->total &&
		place_plus_grand_cosinus->total == place_plus_petit_sinus->total &&
		place_plus_grand_cosinus->total == place_plus_grand_sinus->total &&
		place_plus_grand_cosinus->total == placesym->total);
#endif //_DEBUG
	//dans place, on met la somme des places pour chaque quadrilatee
	for (i = 0; i < place_plus_grand_cosinus->total; i++)
	{
		j = (*(int*)cvGetSeqElemLPR(place_plus_grand_cosinus, i, 0) + *(int*)cvGetSeqElemLPR(place_rapport_long_cotes_opposes, i, 0) +
			*(int*)cvGetSeqElemLPR(place_plus_petit_sinus, i, 0) + *(int*)cvGetSeqElemLPR(place_plus_grand_sinus, i, 0) +
			*(int*)cvGetSeqElemLPR(placesym, i, 0));
		cvSeqPush(place, &j);
	}
	// on initialise classementplace et classement
	cvSeqPush(classementplace, cvGetSeqElemLPR(place, 0, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 0, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 1, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 2, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 3, 0));
	//dans classement on met les quadrilatees
	//et dans classmentplace leurs sommes de places
	// (le tout de faen ordonnee)
	for (j = 1; j < place->total; j++)
	{
		for (i = 0; i < classementplace->total; i++)
		{
			k = 0;
			if (*(int*)cvGetSeqElemLPR(classementplace, i, 0) >
				*(int*)cvGetSeqElemLPR(place, j, 0) || (i == classementplace->total))
			{
				cvSeqInsert(classementplace, i, cvGetSeqElemLPR(place, j, 0));
				cvSeqInsert(classement, (i << 2), cvGetSeqElemLPR(squares, (j << 2), 0));
				cvSeqInsert(classement, (i << 2) + 1, cvGetSeqElemLPR(squares, (j << 2) + 1, 0));
				cvSeqInsert(classement, (i << 2) + 2, cvGetSeqElemLPR(squares, (j << 2) + 2, 0));
				cvSeqInsert(classement, (i << 2) + 3, cvGetSeqElemLPR(squares, (j << 2) + 3, 0));
				k = 1;
			}
			if (k == 1)
				break;
		}
	}
	// on va se resservir de squares et note_plus_grand_cosinus (on les vide donc)
	cvClearSeq(squares);
	cvClearSeq(note_plus_grand_cosinus);
	//dans squares on stocke les rectangles moyens
	//dans centres leurs centres
	//dans note_plus_grand_cosinus, leurs pondeations
	//on eudie le quart supeieur des quadrilatees
	//(ou 1 si y en a - de 4)
	max = (classement->total) >> 4;
	//s'il n'y a pas 4 quadrilatees on les prends tous
	if (max < 1) max = ((classement->total) >> 2);
	k = 0;
	for (j = 0; j < max; j++)//j est le no du quadrilatee auquel on cherche un sosie
	{
		//pt[((j<<2))&3] est un des quatre points du quadrilatee courant.
		pt[((j << 2)) & 3] = *(CvPoint*)cvGetSeqElemLPR(classement, (j << 2), 0);
		//centre est le centre du quadrilatee, il nous permet de l'identifier
		centre = moyenne4(pt[((j << 2)) & 3], pt[((j << 2) + 1) & 3], pt[((j << 2) + 2) & 3], pt[((j << 2) + 3) & 3]);
		if (k == 1)
		{
			m = 0;
			for (i = 0; i < centres->total; i++)// i est le no du qudrilatee qui est peut-ere un sosie
			{
				if (dist(centre, *(CvPoint*)cvGetSeqElemLPR(centres, i, 0)) < 10)
					//si les 2 centres sont emoins de 10 pixels d'intervalle
						//on fait la moyenne des centres
							//on fait la moyenne des rectangles 
								//le coef de pond augmente de 1
				{
					m = 1;
					pond = *(int*)cvGetSeqElemLPR(note_plus_grand_cosinus, i, 0) + 1;
					cvSeqRemove(note_plus_grand_cosinus, i);
					cvSeqInsert(note_plus_grand_cosinus, i, &pond);
					//dans centre on met la moyenne pondere des 2 centres
					//puis on stocke ce point dans la seq. centres
					centre = moyennepond(centre, *(CvPoint*)cvGetSeqElemLPR(centres, i, 0), 1, (pond - 1));
					cvSeqRemove(centres, i);
					cvSeqInsert(centres, i, &centre);
					//centre va servir echaque fois estocker les coordonnees de chaque coin du rectangle moyen
					for (l = 0; l < 4; l++)
					{
						//dans centre, on met les nouvelles coordonnees d'un coin du rectangle 
						//moyen c a d la moyenne pondere du point rectangle preedemment stockeet du 
						// point (du nouveau rectangle) le + proche
						centre = moyennepond(*(CvPoint*)cvGetSeqElemLPR(squares, (i << 2) + l, 0),
							trouveleplusproche(*(CvPoint*)cvGetSeqElemLPR(squares, (i << 2) + l, 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2), 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2) + 1, 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2) + 2, 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2) + 3, 0)), (pond - 1), 1);
						// et echaque fois on stocke le point qui vient d'ere calculedans squares
						cvSeqRemove(squares, (i << 2) + l);
						cvSeqInsert(squares, (i << 2) + l, &centre);
					}
				}
				//si on a associeles 2 rect., on break
				if (m == 1)
					break;
				// si on a pas trouveede sosie ej,
				// on le classe eune nouvelle place dans squares
				// avec une pondeation de 1
				if (i == centres->total)
				{
					cvSeqPush(centres, &centre);
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2), 0));
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2) + 1, 0));
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2) + 2, 0));
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2) + 3, 0));
					cvSeqPush(note_plus_grand_cosinus, &un);
				}
			}
		}
		if (k == 0)// la 1 ere fois  qu on fait cette boucle 
			// on compte systematiquement le quadrilatee comme nouveau
				// donc on le stocke eune nouvelle place dans squares
					// et on lui met une pondeation de 1
		{
			cvSeqPush(centres, &centre);
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 0, 0));
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 1, 0));
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 2, 0));
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 3, 0));
			cvSeqPush(note_plus_grand_cosinus, &un);
			k = 1;
		}
	}
	// leon reclasse nos rectangles par ordre de pondeation
	k = 0; l = 0;
	for (j = 1; j < note_plus_grand_cosinus->total; j++)
	{
		i = *cvGetSeqElemLPR(note_plus_grand_cosinus, j, 0);
		if (i > k)
		{
			k = i;
			l = j;
		}
	}
	// et dans ce classement on garde que le meilleur
	cvClearSeq(classement);
	cvSeqPush(classement, cvGetSeqElemLPR(squares, (l << 2), 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 1 + (l << 2), 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 2 + (l << 2), 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 3 + (l << 2), 0));
	//il ne reste qu'un quadrilatee. on le renvoie
	return classement;
}
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
// renvoie une seuence de 4 points encadrant la plaque
// trouveesur l'image im_src
// IplImage* im_src
// CvMemStorage* storage 
//peut traiter les images sources couleur ou en niveaux de gris
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
// renvoie une seuence de 4 points encadrant la plaque
// trouveesur l'image im_src
// IplImage* im_src
// CvMemStorage* storage 
//peut traiter les images sources couleur ou en niveaux de gris
CvSeq* findSquares4
(IplImage* im_src, CvMemStorage* storage,
	const int nTresh, const int mean_carac
	, const int mean_fond, //
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min)
{
	CvSeq* contours = 0;
	CvPoint centre;
	int  i, j, k, l, m,
		un = 1,
		pond, max;
	int thresh_canny = 50;
	CvSize sz = cvSize(im_src->width, im_src->height);
	//quelques images temporaires qui permettront de travailler
	IplImage* gray1 = cvCreateImage(sz, IPL_DEPTH_8U//unsigned 8-bit integers im_src->depth
		, 1);
	IplImage* tgray = cvCreateImage(sz, IPL_DEPTH_8U//im_src->depth
		, 1);
	IplImage* gray2 = cvCreateImage(sz, IPL_DEPTH_8U//unsigned 8-bit integers im_src->depth
		, 1);
	int filterSize = 5;
	IplConvKernel* convKernel = cvCreateStructuringElementEx
	(filterSize, filterSize, (filterSize - 1) / 2, (filterSize - 1) / 2,
		CV_SHAPE_ELLIPSE, NULL);
#ifdef _DEBUG
	assert(im_src->nChannels == im_src->nChannels);
	assert(im_src->depth == im_src->depth);
	assert(im_src->align == im_src->align);
#endif //_DEBUG
	// squares et classement permettront de travailler sur les listes de quadrilatees
	CvSeq* squares = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	CvSeq* classement = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	// centres est une liste de centres de quadrilatees utilisee ela fin
	CvSeq* centres = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	// diffeentes seuences contentant les notes correspondant aux diffeents critees 
	CvSeq* note_plus_grand_cosinus = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* note_rapport_long_cotes_opposes = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* note_plus_petit_sinus = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* note_plus_grand_sinus = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* notesym = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	// diffeentes seuences contenant les places correspondant aux diffeents critees
	CvSeq* place_plus_grand_cosinus = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* place_rapport_long_cotes_opposes = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* place_plus_petit_sinus = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* place_plus_grand_sinus = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* placesym = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	// place contiendra la liste de sommes de places
	CvSeq* place = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	//classementplace contiendra la liste des quadrilatees ordonnee selon les sommes de places
	CvSeq* classementplace = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	// selectionne le + grand rectangle dans l'image
	// dont les dimensions sont divisibles par 2
#ifdef _DEBUG
	assert(im_src->width == sz.width);
	assert(im_src->height == sz.height);
#endif //_DEBUG
	//		cvSetImageROI( im_src, cvRect( 0, 0, sz.width, sz.height ));
	// pt servira eplusieurs reprises estocker des quadrilatees
	// (tableau de 4 points)
	CvPoint pt[4];
	// on teste toutes les couleurs R,G,B
	//const int sz.width=im_src->width;const int sz.height=im_src->height;
	cvCopy(im_src, tgray, 0);
	//si on est pas en NB
	//on recopie ds une image de travail
	bool prochaine_image_gray1 = true;
	// on teste N niveaux de seuillage
	// apply Canny. Take the upper threshold from slider
	// and set the lower to 0 (which forces edges merging) 
	cvCanny(tgray, gray1, thresh_canny, 255 - thresh_canny, 5//Aperture parameter for Sobel operator
	);
	// dilate canny output to remove potential
	// holes between edge segments 
	//cvErode( gray1, gray1, 0, 1 );//a rajouter
	// trouve les contours et en fait une liste
	float cvApproxPoly_coeff = 0.02f;
	cvFindContours(gray1, storage, &contours, sizeof(CvContour),
		CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	// teste chaque contour
	//calcule le poids des contours
	init_contours_weight(tgray, dist_min_bord,
		hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
		sz.width, sz.height,
		storage, contours,
		squares,
		// diffeentes seuences contentant les notes correspondant aux diffeents critees 
		note_plus_grand_cosinus,
		note_rapport_long_cotes_opposes,
		note_plus_petit_sinus,
		note_plus_grand_sinus,
		notesym, cvApproxPoly_coeff);
	if (!squares->total) {
		cv::Mat img = cv::cvarrToMat(tgray);
		cv::Mat image_final = cv::cvarrToMat(gray1);
		int blockSize = (sz.width + sz.height) / 2;
		if (blockSize % 2 == 0) blockSize += 1;
		cv::adaptiveThreshold(img, image_final, 127, cv::ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, blockSize, 0);
		cvFindContours(gray1, storage, &contours, sizeof(CvContour),
			CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		// teste chaque contour
		//calcule le poids des contours
		int total_number_of_plates = squares->total;
		init_contours_weight(tgray, dist_min_bord,
			hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
			sz.width, sz.height,
			storage, contours,
			squares,
			// diffeentes seuences contentant les notes correspondant aux diffeents critees 
			note_plus_grand_cosinus,
			note_rapport_long_cotes_opposes,
			note_plus_petit_sinus,
			note_plus_grand_sinus,
			notesym, cvApproxPoly_coeff);
		if (total_number_of_plates == squares->total) {
			int nb_pts_sup_threshold = (sz.width * sz.height);
			int new_mean_carac = mean_carac;
			int new_mean_fond = mean_fond;
			if (mean_carac > mean_fond) {
				new_mean_carac = mean_fond;
				new_mean_fond = mean_carac;
				new_mean_carac -= (mean_carac - mean_fond) / 10;
				new_mean_fond += (mean_carac - mean_fond) / 10;
			}
			else {
				new_mean_carac -= (mean_fond - mean_carac) / 10;
				new_mean_fond += (mean_fond - mean_carac) / 10;
			}
#ifdef _DEBUG
			assert(new_mean_carac <= new_mean_fond);
#endif //_DEBUG
			if (new_mean_carac < 0) new_mean_carac = 0;
			if (new_mean_fond > 255) new_mean_fond = 255;
			int new_nTresh = nTresh;
			if (new_nTresh > (new_mean_fond - new_mean_carac) >> 1)
				new_nTresh = (new_mean_fond - new_mean_carac) >> 1;
			int seuil;
			for (seuil = 0; seuil < new_nTresh; seuil++)
			{
				// sinon on applique le seuillage
				///((tgray(x,y) = gray(x,y) < (seuil+1)*255/N ? 255 : 0))
				int current_nb_pts_sup_threshold = 0;
				if (new_mean_carac < new_mean_fond)
					LPRThreshold(im_src, gray1,
						new_mean_carac + ((new_mean_fond - new_mean_carac) * seuil) / new_nTresh,
						current_nb_pts_sup_threshold);// trouve les contours et en fait une liste
				else LPRThreshold(im_src, gray1,
					new_mean_fond + ((new_mean_carac - new_mean_fond) * seuil) / new_nTresh,
					current_nb_pts_sup_threshold);
				if (current_nb_pts_sup_threshold >= 0 &&
					current_nb_pts_sup_threshold <= nb_pts_sup_threshold) {
					///////////////////////////////////////////////////////////////////////////////////////////
					// retourne le nb de points en commun entre deux images
#ifdef _DEBUG
					assert(current_nb_pts_sup_threshold <= nb_pts_sup_threshold);
#endif //_DEBUG
					float percentage_match = 1.0f - ((float)(nb_pts_sup_threshold -
						current_nb_pts_sup_threshold)) / (sz.width * sz.height);
#ifdef _DEBUG
					assert(percentage_match > -FLT_EPSILON &&
						percentage_match < 1.0f + FLT_EPSILON);
#endif //_DEBUG
					if (percentage_match < 0.75) {
						nb_pts_sup_threshold = current_nb_pts_sup_threshold;
						cvFindContours(gray1, storage, &contours, sizeof(CvContour),
							CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
						total_number_of_plates = squares->total;
						init_contours_weight(tgray, dist_min_bord,
							hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
							sz.width, sz.height,
							storage, contours,
							squares,
							// diffeentes seuences contentant les notes correspondant aux diffeents critees 
							note_plus_grand_cosinus,
							note_rapport_long_cotes_opposes,
							note_plus_petit_sinus,
							note_plus_grand_sinus,
							notesym, cvApproxPoly_coeff);
						if (seuil == 0) cvApproxPoly_coeff = 0.02f;
						else if (seuil == 1) cvApproxPoly_coeff = 0.03f;
						else if (seuil == 2) cvApproxPoly_coeff = 0.04f;
						else cvApproxPoly_coeff = 0.05f;//if(seuil==3) 
														// teste chaque contour
														//calcule le poids des contours
						if (total_number_of_plates == squares->total) {
							cvMorphologyEx(gray1, gray2, NULL, convKernel, CV_MOP_CLOSE);
							cvFindContours(gray2, storage, &contours, sizeof(CvContour),
								CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
							init_contours_weight(tgray, dist_min_bord,
								hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
								sz.width, sz.height,
								storage, contours,
								squares,
								// diffeentes seuences contentant les notes correspondant aux diffeents critees 
								note_plus_grand_cosinus,
								note_rapport_long_cotes_opposes,
								note_plus_petit_sinus,
								note_plus_grand_sinus,
								notesym, cvApproxPoly_coeff);
						}
					}
				}
			}
		}
	}
	// (on libee de la meoire)
	cvReleaseImage(&gray1);
	cvReleaseImage(&tgray);
	cvReleaseImage(&gray2);
	cvReleaseStructuringElement(&convKernel);
#ifdef _DEBUG
	assert(note_plus_grand_cosinus->total == note_rapport_long_cotes_opposes->total &&
		note_plus_grand_cosinus->total == note_plus_petit_sinus->total &&
		note_plus_grand_cosinus->total == note_plus_grand_sinus->total &&
		note_plus_grand_cosinus->total == notesym->total);
#endif //_DEBUG
	//si rien n'a passele filtre, on arree le:
	//on renvoie une seuence de -1
	if (notesym->total == 0)
	{
		cvClearSeq(classement);
		centre = cvPoint(SHRT_MIN, SHRT_MIN);
		cvSeqPush(classement, &centre);
		cvSeqPush(classement, &centre);
		cvSeqPush(classement, &centre);
		cvSeqPush(classement, &centre);
		return classement;
	}
	// CLASSEMENT
	// renvoie la seuence de places epartir de la seuence de notes
	place_plus_grand_cosinus = sort(note_plus_grand_cosinus, storage);
	place_rapport_long_cotes_opposes = sort(note_rapport_long_cotes_opposes, storage);
	place_plus_petit_sinus = sort(note_plus_petit_sinus, storage);
	place_plus_grand_sinus = sort(note_plus_grand_sinus, storage);
	placesym = sort(notesym, storage);
#ifdef _DEBUG
	assert(place_plus_grand_cosinus->total == place_rapport_long_cotes_opposes->total &&
		place_plus_grand_cosinus->total == place_plus_petit_sinus->total &&
		place_plus_grand_cosinus->total == place_plus_grand_sinus->total &&
		place_plus_grand_cosinus->total == placesym->total);
#endif //_DEBUG
	//dans place, on met la somme des places pour chaque quadrilatee
	for (i = 0; i < place_plus_grand_cosinus->total; i++)
	{
		j = (*(int*)cvGetSeqElemLPR(place_plus_grand_cosinus, i, 0) + *(int*)cvGetSeqElemLPR(place_rapport_long_cotes_opposes, i, 0) +
			*(int*)cvGetSeqElemLPR(place_plus_petit_sinus, i, 0) + *(int*)cvGetSeqElemLPR(place_plus_grand_sinus, i, 0) +
			*(int*)cvGetSeqElemLPR(placesym, i, 0));
		cvSeqPush(place, &j);
	}
	// on initialise classementplace et classement
	cvSeqPush(classementplace, cvGetSeqElemLPR(place, 0, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 0, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 1, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 2, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 3, 0));
	//dans classement on met les quadrilatees
	//et dans classmentplace leurs sommes de places
	// (le tout de faen ordonnee)
	for (j = 1; j < place->total; j++)
	{
		for (i = 0; i < classementplace->total; i++)
		{
			k = 0;
			if (*(int*)cvGetSeqElemLPR(classementplace, i, 0) >
				*(int*)cvGetSeqElemLPR(place, j, 0) || (i == classementplace->total))
			{
				cvSeqInsert(classementplace, i, cvGetSeqElemLPR(place, j, 0));
				cvSeqInsert(classement, (i << 2), cvGetSeqElemLPR(squares, (j << 2), 0));
				cvSeqInsert(classement, (i << 2) + 1, cvGetSeqElemLPR(squares, (j << 2) + 1, 0));
				cvSeqInsert(classement, (i << 2) + 2, cvGetSeqElemLPR(squares, (j << 2) + 2, 0));
				cvSeqInsert(classement, (i << 2) + 3, cvGetSeqElemLPR(squares, (j << 2) + 3, 0));
				k = 1;
			}
			if (k == 1)
				break;
		}
	}
	// on va se resservir de squares et note_plus_grand_cosinus (on les vide donc)
	cvClearSeq(squares);
	cvClearSeq(note_plus_grand_cosinus);
	//dans squares on stocke les rectangles moyens
	//dans centres leurs centres
	//dans note_plus_grand_cosinus, leurs pondeations
	//on eudie le quart supeieur des quadrilatees
	//(ou 1 si y en a - de 4)
	max = (classement->total) >> 4;
	//s'il n'y a pas 4 quadrilatees on les prends tous
	if (max < 1) max = ((classement->total) >> 2);
	k = 0;
	for (j = 0; j < max; j++)//j est le no du quadrilatee auquel on cherche un sosie
	{
		//pt[((j<<2))&3] est un des quatre points du quadrilatee courant.
		pt[((j << 2)) & 3] = *(CvPoint*)cvGetSeqElemLPR(classement, (j << 2), 0);
		//centre est le centre du quadrilatee, il nous permet de l'identifier
		centre = moyenne4(pt[((j << 2)) & 3], pt[((j << 2) + 1) & 3], pt[((j << 2) + 2) & 3], pt[((j << 2) + 3) & 3]);
		if (k == 1)
		{
			m = 0;
			for (i = 0; i < centres->total; i++)// i est le no du qudrilatee qui est peut-ere un sosie
			{
				if (dist(centre, *(CvPoint*)cvGetSeqElemLPR(centres, i, 0)) < 10)
					//si les 2 centres sont emoins de 10 pixels d'intervalle
						//on fait la moyenne des centres
							//on fait la moyenne des rectangles 
								//le coef de pond augmente de 1
				{
					m = 1;
					pond = *(int*)cvGetSeqElemLPR(note_plus_grand_cosinus, i, 0) + 1;
					cvSeqRemove(note_plus_grand_cosinus, i);
					cvSeqInsert(note_plus_grand_cosinus, i, &pond);
					//dans centre on met la moyenne pondere des 2 centres
					//puis on stocke ce point dans la seq. centres
					centre = moyennepond(centre, *(CvPoint*)cvGetSeqElemLPR(centres, i, 0), 1, (pond - 1));
					cvSeqRemove(centres, i);
					cvSeqInsert(centres, i, &centre);
					//centre va servir echaque fois estocker les coordonnees de chaque coin du rectangle moyen
					for (l = 0; l < 4; l++)
					{
						//dans centre, on met les nouvelles coordonnees d'un coin du rectangle 
						//moyen c a d la moyenne pondere du point rectangle preedemment stockeet du 
						// point (du nouveau rectangle) le + proche
						centre = moyennepond(*(CvPoint*)cvGetSeqElemLPR(squares, (i << 2) + l, 0),
							trouveleplusproche(*(CvPoint*)cvGetSeqElemLPR(squares, (i << 2) + l, 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2), 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2) + 1, 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2) + 2, 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2) + 3, 0)), (pond - 1), 1);
						// et echaque fois on stocke le point qui vient d'ere calculedans squares
						cvSeqRemove(squares, (i << 2) + l);
						cvSeqInsert(squares, (i << 2) + l, &centre);
					}
				}
				//si on a associeles 2 rect., on break
				if (m == 1)
					break;
				// si on a pas trouveede sosie ej,
				// on le classe eune nouvelle place dans squares
				// avec une pondeation de 1
				if (i == centres->total)
				{
					cvSeqPush(centres, &centre);
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2), 0));
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2) + 1, 0));
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2) + 2, 0));
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2) + 3, 0));
					cvSeqPush(note_plus_grand_cosinus, &un);
				}
			}
		}
		if (k == 0)// la 1 ere fois  qu on fait cette boucle 
			// on compte systematiquement le quadrilatee comme nouveau
				// donc on le stocke eune nouvelle place dans squares
					// et on lui met une pondeation de 1
		{
			cvSeqPush(centres, &centre);
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 0, 0));
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 1, 0));
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 2, 0));
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 3, 0));
			cvSeqPush(note_plus_grand_cosinus, &un);
			k = 1;
		}
	}
	// leon reclasse nos rectangles par ordre de pondeation
	k = 0; l = 0;
	for (j = 1; j < note_plus_grand_cosinus->total; j++)
	{
		i = *cvGetSeqElemLPR(note_plus_grand_cosinus, j, 0);
		if (i > k)
		{
			k = i;
			l = j;
		}
	}
	// et dans ce classement on garde que le meilleur
	cvClearSeq(classement);
	cvSeqPush(classement, cvGetSeqElemLPR(squares, (l << 2), 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 1 + (l << 2), 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 2 + (l << 2), 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 3 + (l << 2), 0));
	//il ne reste qu'un quadrilatee. on le renvoie
	return classement;
}
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
// renvoie une seuence de 4 points encadrant la plaque
// trouveesur l'image im_src
// IplImage* im_src
// CvMemStorage* storage 
//peut traiter les images sources couleur ou en niveaux de gris
void resize(CvSeq* fin, const float s)
{
	if (fin) {
		if (fin->total == 4) {
			CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
			pt0->x *= s;
			pt0->y *= s;
			CvPoint* pt1 = CV_GET_SEQ_ELEM(CvPoint, fin, 1);
			pt1->x *= s;
			pt1->y *= s;
			CvPoint* pt2 = CV_GET_SEQ_ELEM(CvPoint, fin, 2);
			pt2->x *= s;
			pt2->y *= s;
			CvPoint* pt3 = CV_GET_SEQ_ELEM(CvPoint, fin, 3);
			pt3->x *= s;
			pt3->y *= s;
		}
	}
}
float dist(const CvPoint* pt, const int i, const int j)
{
	return (pt->x - i) * (pt->x - i) + (pt->y - j) * (pt->y - j);
}
void update(const CvPoint* pt0,
	const CvPoint* pt1,
	const CvPoint* pt2,
	const CvPoint* pt3, const int i, const int j,
	cv::Point& p0, cv::Point& p1, cv::Point& p2, cv::Point& p3,
	float& min_d0, float& min_d1, float& min_d2, float& min_d3)
{
	float d0 = dist(pt0, i, j);
	float d1 = dist(pt1, i, j);
	float d2 = dist(pt2, i, j);
	float d3 = dist(pt3, i, j);
	if (d0 < d1) {//d0 and d2 are height and d1 , d3 are width
		if (d0 < d2) {
			if (d0 < d3) {
				if (d0 < min_d0) {
					min_d0 = d0;
					p0 = cv::Point(i, j);
				}
			}
			else {
				if (d3 < min_d3) {
					min_d3 = d3;
					p3 = cv::Point(i, j);
				}
			}
		}
		else {
			if (d2 < d3) {
				if (d2 < min_d2) {
					min_d2 = d2;
					p2 = cv::Point(i, j);
				}
			}
			else {
				if (d3 < min_d3) {
					min_d3 = d3;
					p3 = cv::Point(i, j);
				}
			}
		}
	}
	else {
		if (d1 < d2) {
			if (d1 < d3) {
				if (d1 < min_d1) {
					min_d1 = d1;
					p1 = cv::Point(i, j);
				}
			}
			else {
				if (d2 < min_d2) {
					min_d2 = d2;
					p2 = cv::Point(i, j);
				}
			}
		}
		else {
			if (d2 < d3) {
				if (d2 < min_d2) {
					min_d2 = d2;
					p2 = cv::Point(i, j);
				}
			}
			else {
				if (d3 < min_d3) {
					min_d3 = d3;
					p3 = cv::Point(i, j);
				}
			}
		}
	}
}
/** @function refine_cornerHarris */
void refine_cornerHarris(const IplImage* im_src, const CvSeq* fin, cv::Point& p0, cv::Point& p1, cv::Point& p2, cv::Point& p3)
{
	cv::Mat dst, dst_norm, dst_norm_scaled, src_gray;
	src_gray = cv::cvarrToMat(im_src);
	dst = cv::Mat::zeros(src_gray.size(), CV_32FC1);
	/// Detector parameters
	int blockSize = 5;
	int apertureSize = 7;
	double k = 0.04;
	int thresh = 127;
	int max_thresh = 255;
	/// Detecting corners
	cv::cornerHarris(src_gray, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
	/// Normalizing
	cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
	cv::convertScaleAbs(dst_norm, dst_norm_scaled);
	float min_d0 = FLT_MAX, min_d1 = FLT_MAX, min_d2 = FLT_MAX, min_d3 = FLT_MAX;
	CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
	CvPoint* pt1 = CV_GET_SEQ_ELEM(CvPoint, fin, 1);
	CvPoint* pt2 = CV_GET_SEQ_ELEM(CvPoint, fin, 2);
	CvPoint* pt3 = CV_GET_SEQ_ELEM(CvPoint, fin, 3);
	p0.x = pt0->x; p0.y = pt0->y;
	p1.x = pt1->x; p1.y = pt1->y;
	p2.x = pt2->x; p2.y = pt2->y;
	p3.x = pt3->x; p3.y = pt3->y;
	/// Drawing a circle around corners
	std::list<float> median;
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			median.push_back(dst_norm.at<float>(j, i));
		}
	}
	median.sort();
	int i = 0;
	std::list<float>::const_reverse_iterator it(median.rbegin());
	const int quartille = median.size() / 40;
	while (it != median.rend() && i < quartille) {
		//operator*(const C_Transform_2d& L, const C_Transform_2d& R)
		i++;
		thresh = *it;
		it++;
	}
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > thresh)
			{
				update(pt0, pt1, pt2, pt3, i, j, p0, p1, p2, p3, min_d0, min_d1, min_d2, min_d3);
			}
		}
	}
}
void refine_SIFT(const IplImage* im_src, const CvSeq* fin, cv::Point& p0, cv::Point& p1, cv::Point& p2, cv::Point& p3)
{
	cv::Mat src_gray =
		cv::cvarrToMat(im_src);
	// Create smart pointer for SIFT feature detector.
	cv::Ptr<cv::FeatureDetector> featureDetector = cv::AKAZE::create();//cv::FeatureDetector::create("FAST");
	std::vector<cv::KeyPoint> keypoints;
	keypoints.reserve(10000);
	// Detect the keypoints
	featureDetector->detect(src_gray, keypoints); // NOTE: featureDetector is a pointer hence the '->'.
	float min_d0 = FLT_MAX, min_d1 = FLT_MAX, min_d2 = FLT_MAX, min_d3 = FLT_MAX;
	CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
	CvPoint* pt1 = CV_GET_SEQ_ELEM(CvPoint, fin, 1);
	CvPoint* pt2 = CV_GET_SEQ_ELEM(CvPoint, fin, 2);
	CvPoint* pt3 = CV_GET_SEQ_ELEM(CvPoint, fin, 3);
	p0.x = pt0->x; p0.y = pt0->y;
	p1.x = pt1->x; p1.y = pt1->y;
	p2.x = pt2->x; p2.y = pt2->y;
	p3.x = pt3->x; p3.y = pt3->y;
	std::vector<cv::KeyPoint>::const_iterator it(keypoints.begin());
	while (it != keypoints.end())
	{
		update(pt0, pt1, pt2, pt3, it->pt.x, it->pt.y, p0, p1, p2, p3, min_d0, min_d1, min_d2, min_d3);
		it++;
	}
}
/** @function refine_cornerHarris */
void refine_cornerHarris(const IplImage* im_src, CvSeq* fin)
{
	cv::Point p0, p1, p2, p3;
	refine_SIFT(im_src, fin, p0, p1, p2, p3);
	if (fin) {
		if (fin->total == 4) {
			CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
			pt0->x = p0.x;
			pt0->y = p0.y;
			CvPoint* pt1 = CV_GET_SEQ_ELEM(CvPoint, fin, 1);
			pt1->x = p1.x;
			pt1->y = p1.y;
			CvPoint* pt2 = CV_GET_SEQ_ELEM(CvPoint, fin, 2);
			pt2->x = p2.x;
			pt2->y = p2.y;
			CvPoint* pt3 = CV_GET_SEQ_ELEM(CvPoint, fin, 3);
			pt3->x = p3.x;
			pt3->y = p3.y;
		}
	}
}
CvSeq* findSquares4_multiresolution
(
	IplImage* im_src, CvMemStorage* storage,
	const int nTresh, const int mean_carac
	, const int mean_fond, //
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min
)
{
	CvSeq* fin = findSquares4
	(im_src, storage,
		nTresh, //
		mean_carac, mean_fond, dist_min_bord,
		hauteur_plaque_min,
		rapport_largeur_sur_hauteur_min);
	if (fin) {
		if (fin->total == 4) {
			CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
			if (pt0->x >= 0 && pt0->y >= 0)	return fin;
		}
	}
	if (im_src->width > MIN_WIDTH_FOR_A_LP && im_src->height > MIN_HEIGHT_FOR_A_LP) {
		cv::Point p0, p1, p2, p3;
		IplImage* new_img_2 = cvCreateImage(cvSize(im_src->width * 0.5, im_src->height * 0.5), im_src->depth, im_src->nChannels);
		cvResize(im_src, new_img_2, CV_INTER_CUBIC);
		CvSeq* fin = findSquares4
		(new_img_2, storage,
			nTresh, //
			mean_carac, mean_fond, dist_min_bord,
			hauteur_plaque_min,
			rapport_largeur_sur_hauteur_min);
		if (fin) {
			if (fin->total == 4) {
				CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
				if (pt0->x >= 0 && pt0->y >= 0) {
					/*
					#ifdef _DEBUG
										cv::Mat tmp=new_img_2;
										imshow("new_img_2",tmp);
										char c = cv::waitKey(0);
					#endif //_DEBUG
										*/
					resize(fin, 2.0);
					refine_cornerHarris(im_src, fin);
					return fin;
				}
			}
		}
		if (new_img_2->width > MIN_WIDTH_FOR_A_LP && new_img_2->height > MIN_HEIGHT_FOR_A_LP) {
			IplImage* new_img_4 = cvCreateImage(cvSize(new_img_2->width * 0.5, new_img_2->height * 0.5), new_img_2->depth, new_img_2->nChannels);
			cvResize(new_img_2, new_img_4, CV_INTER_CUBIC);
			CvSeq* fin = findSquares4
			(new_img_4, storage,
				nTresh, //
				mean_carac, mean_fond, dist_min_bord,
				hauteur_plaque_min,
				rapport_largeur_sur_hauteur_min);
			if (fin) {
				if (fin->total == 4) {
					CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
					if (pt0->x >= 0 && pt0->y >= 0) {
						/*
						#ifdef _DEBUG
												cv::Mat tmp=new_img_4;
												imshow("new_img_4",tmp);
												char c = cv::waitKey(0);
						#endif //_DEBUG
												*/
						resize(fin, 4.0);
						refine_cornerHarris(im_src, fin);
					}
				}
			}
			return fin;
		}
		else {
			return fin;
		}
	}
	else {
		return fin;
	}
}
CvSeq* findSquares4_multiresolution
(
	IplImage* im_src, const cv::Rect& global_rect, CvMemStorage* storage,
	const int nTresh, const int mean_carac
	, const int mean_fond, //
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min
)
{
	CvSeq* fin = findSquares4
	(im_src, global_rect, storage,
		nTresh, //
		mean_carac, mean_fond, dist_min_bord,
		hauteur_plaque_min,
		rapport_largeur_sur_hauteur_min);
	if (fin) {
		if (fin->total == 4) {
			CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
			if (pt0->x >= 0 && pt0->y >= 0) {
#ifdef _DEBUG
				cv::RotatedRect RotatedRect_(get_rect(fin));
				cv::Rect global_rotated_rect = RotatedRect_.boundingRect();
				float iou__ = iou_(global_rect, global_rotated_rect);
				//assert(iou__ >= 0.3f);
#endif //_DEBUG
				return fin;
			}
		}
	}
	if (im_src->width > MIN_WIDTH_FOR_A_LP && im_src->height > MIN_HEIGHT_FOR_A_LP) {
		cv::Point p0, p1, p2, p3;
		IplImage* new_img_2 = cvCreateImage(cvSize(im_src->width * 0.5f, im_src->height * 0.5), im_src->depth, im_src->nChannels);
		cvResize(im_src, new_img_2, CV_INTER_CUBIC);
		CvSeq* fin = findSquares4
		(new_img_2, cv::Rect(global_rect.x * 0.5f, global_rect.x * 0.5f, global_rect.width * 0.5f, global_rect.height * 0.5f), storage,
			nTresh, //
			mean_carac, mean_fond, dist_min_bord,
			hauteur_plaque_min,
			rapport_largeur_sur_hauteur_min);
		if (fin) {
			if (fin->total == 4) {
				CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
				if (pt0->x >= 0 && pt0->y >= 0) {
					/*
					#ifdef _DEBUG
										cv::Mat tmp=new_img_2;
										imshow("new_img_2",tmp);
										char c = cv::waitKey(0);
					#endif //_DEBUG
										*/
					resize(fin, 2.0);
					refine_cornerHarris(im_src, fin);
#ifdef _DEBUG
					cv::RotatedRect RotatedRect_(get_rect(fin));
					cv::Rect global_rotated_rect = RotatedRect_.boundingRect();
					float iou__ = iou_(cv::Rect(global_rect.x * 0.5f, global_rect.x * 0.5f, global_rect.width * 0.5f, global_rect.height * 0.5f), global_rotated_rect);
					//assert(iou__ >= 0.3f);
#endif //_DEBUG
					return fin;
				}
			}
		}
		if (new_img_2->width > MIN_WIDTH_FOR_A_LP && new_img_2->height > MIN_HEIGHT_FOR_A_LP) {
			IplImage* new_img_4 = cvCreateImage(cvSize(new_img_2->width * 0.5, new_img_2->height * 0.5), new_img_2->depth, new_img_2->nChannels);
			cvResize(new_img_2, new_img_4, CV_INTER_CUBIC);
			CvSeq* fin = findSquares4
			(new_img_4, cv::Rect(global_rect.x * 0.25f, global_rect.x * 0.25f, global_rect.width * 0.25f, global_rect.height * 0.25f), storage,
				nTresh, //
				mean_carac, mean_fond, dist_min_bord,
				hauteur_plaque_min,
				rapport_largeur_sur_hauteur_min);
			if (fin) {
				if (fin->total == 4) {
					CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
					if (pt0->x >= 0 && pt0->y >= 0) {
						/*
						#ifdef _DEBUG
												cv::Mat tmp=new_img_4;
												imshow("new_img_4",tmp);
												char c = cv::waitKey(0);
						#endif //_DEBUG
												*/
#ifdef _DEBUG
						cv::RotatedRect RotatedRect_(get_rect(fin));
						cv::Rect global_rotated_rect = RotatedRect_.boundingRect();
						float iou__ = iou_(cv::Rect(global_rect.x * 0.25f, global_rect.x * 0.25f, global_rect.width * 0.25f, global_rect.height * 0.25f), global_rotated_rect);
						assert(iou__ >= 0.2f);
#endif //_DEBUG
						resize(fin, 4.0);
						refine_cornerHarris(im_src, fin);
					}
				}
			}
			return fin;
		}
		else {
			return fin;
		}
	}
	else {
		return fin;
	}
}
CvSeq* findSquares4
(IplImage* im_src, CvMemStorage* storage,
	const int nTresh, //
	//le nb de composantes couleurs ex nb_composantes_color=3 pour une image RGB
	const int nb_composantes_color,
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min)
{
	const int mean_carac = 50;
	const int mean_fond = 206;
	if (nb_composantes_color == 1) {
		return findSquares4
		(im_src, storage,
			nTresh, //
			mean_carac
			, mean_fond,
			dist_min_bord,
			hauteur_plaque_min,
			rapport_largeur_sur_hauteur_min);
	}
	else	if (nb_composantes_color != 3) {
		return NULL;
	}
	CvSeq* contours = 0;
	CvPoint centre;
	int  i, j, k, l, m,
		un = 1,
		pond, max;
	/*	*/
	int thresh_canny = 50;
	CvSize sz = cvSize(im_src->width, im_src->height);
	//quelques images temporaires qui permettront de travailler
	//		IplImage* im_copy = cvCloneImage( im_src ); // make a copy of input image
	IplImage* gray1 = cvCreateImage(sz, IPL_DEPTH_8U//unsigned 8-bit integers im_src->depth
		, 1);
	IplImage* tgray = cvCreateImage(sz, IPL_DEPTH_8U//im_src->depth
		, 1);
	IplImage* gray2 = cvCreateImage(sz, IPL_DEPTH_8U//unsigned 8-bit integers im_src->depth
		, 1);
	int filterSize = 5;
	IplConvKernel* convKernel = cvCreateStructuringElementEx
	(filterSize, filterSize, (filterSize - 1) / 2, (filterSize - 1) / 2,
		CV_SHAPE_ELLIPSE, NULL);
#ifdef _DEBUG
	assert(im_src->nChannels == im_src->nChannels);
	assert(im_src->depth == im_src->depth);
	assert(im_src->align == im_src->align);
#endif //_DEBUG
	// squares et classement permettront de travailler sur les listes de quadrilatees
	CvSeq* squares = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	CvSeq* classement = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	// centres est une liste de centres de quadrilatees utilisee ela fin
	CvSeq* centres = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	// diffeentes seuences contentant les notes correspondant aux diffeents critees 
	CvSeq* note_plus_grand_cosinus = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* note_rapport_long_cotes_opposes = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* note_plus_petit_sinus = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* note_plus_grand_sinus = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* notesym = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	// diffeentes seuences contenant les places correspondant aux diffeents critees
	CvSeq* place_plus_grand_cosinus = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* place_rapport_long_cotes_opposes = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* place_plus_petit_sinus = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* place_plus_grand_sinus = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* placesym = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	// place contiendra la liste de sommes de places
	CvSeq* place = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	//classementplace contiendra la liste des quadrilatees ordonnee selon les sommes de places
	CvSeq* classementplace = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	// selectionne le + grand rectangle dans l'image
	// dont les dimensions sont divisibles par 2
	//	cvSetImageROI( im_src, cvRect( 0, 0, sz.width, sz.height ));
	// pt servira eplusieurs reprises estocker des quadrilatees
	// (tableau de 4 points)
	CvPoint pt[4];
	// on teste toutes les couleurs R,G,B
	//const int sz.width=im_src->width;const int sz.height=im_src->height;
	for (int c = 0; c < nb_composantes_color; c++)
	{
		//si on est pas en NB
		if (nb_composantes_color != 1) {
			if (im_src->nChannels == nb_composantes_color && nb_composantes_color != 1)
				//on extrait la composante dans la couleur choisie
				cvSetImageCOI(im_src, c + 1);// evoir s'il ne faut pas mettre c+1 cvSetImageCOI( im_src, c+1 );
			else if (im_src->nChannels == nb_composantes_color + 1)
				cvSetImageCOI(im_src, c + 2);
		}
		//on recopie ds une image de travail
		cvCopy(im_src, tgray, 0);
		// apply Canny. Take the upper threshold from slider
		// and set the lower to 0 (which forces edges merging) 
		cvCanny(tgray, gray1, thresh_canny, 255 - thresh_canny, 5//Aperture parameter for Sobel operator
		);
		// dilate canny output to remove potential
		// holes between edge segments 
		//cvErode( gray1, gray1, 0, 1 );//a rajouter
		// trouve les contours et en fait une liste
		float cvApproxPoly_coeff = 0.02f;
		cvFindContours(gray1, storage, &contours, sizeof(CvContour),
			CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		// teste chaque contour
		//calcule le poids des contours
		init_contours_weight(tgray, dist_min_bord,
			hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
			sz.width, sz.height,
			storage, contours,
			squares,
			// diffeentes seuences contentant les notes correspondant aux diffeents critees 
			note_plus_grand_cosinus,
			note_rapport_long_cotes_opposes,
			note_plus_petit_sinus,
			note_plus_grand_sinus,
			notesym, cvApproxPoly_coeff);
		if (!squares->total) {
			/*
			Mat cv::cvarrToMat 	( 	const CvArr *  	arr,
			bool  	copyData = false,
			bool  	allowND = true,
			int  	coiMode = 0,
			AutoBuffer< double > *  	buf = 0
		)
		*/
			cv::Mat img = cv::cvarrToMat(tgray);
			cv::Mat image_final = cv::cvarrToMat(gray1);
			int blockSize = (sz.width + sz.height) / 2;
			if (blockSize % 2 == 0) blockSize += 1;
			cv::adaptiveThreshold(img, image_final, 127, cv::ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, blockSize, 0);
			cvFindContours(gray1, storage, &contours, sizeof(CvContour),
				CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
			// teste chaque contour
			//calcule le poids des contours
			int total_number_of_plates = squares->total;
			init_contours_weight(tgray, dist_min_bord,
				hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
				sz.width, sz.height,
				storage, contours,
				squares,
				// diffeentes seuences contentant les notes correspondant aux diffeents critees 
				note_plus_grand_cosinus,
				note_rapport_long_cotes_opposes,
				note_plus_petit_sinus,
				note_plus_grand_sinus,
				notesym, cvApproxPoly_coeff);
			if (total_number_of_plates == squares->total) {
				int nb_pts_sup_threshold = (sz.width * sz.height);
				// on teste N niveaux de seuillage
				int seuil;
				for (seuil = 0; seuil < nTresh; seuil++)
				{
					int current_nb_pts_sup_threshold = 0;
					LPRThreshold(im_src, gray1, ((seuil + 1) * 200 / nTresh) + 25,
						current_nb_pts_sup_threshold);// trouve les contours et en fait une liste
					if (current_nb_pts_sup_threshold >= 0 &&
						current_nb_pts_sup_threshold <= nb_pts_sup_threshold) {
						///////////////////////////////////////////////////////////////////////////////////////////
						// retourne le nb de points en commun entre deux images
#ifdef _DEBUG
						assert(current_nb_pts_sup_threshold <= nb_pts_sup_threshold);
#endif //_DEBUG
						float percentage_match = 1.0f - ((float)(nb_pts_sup_threshold -
							current_nb_pts_sup_threshold)) / (sz.width * sz.height);
#ifdef _DEBUG
						assert(percentage_match > -FLT_EPSILON &&
							percentage_match < 1.0f + FLT_EPSILON);
#endif //_DEBUG
						if (percentage_match < 0.75) {
							cvFindContours(gray1, storage, &contours, sizeof(CvContour),
								CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
							nb_pts_sup_threshold = current_nb_pts_sup_threshold;
							if (seuil == 0) cvApproxPoly_coeff = 0.02f;
							else if (seuil == 1) cvApproxPoly_coeff = 0.03f;
							else if (seuil == 2) cvApproxPoly_coeff = 0.04f;
							else cvApproxPoly_coeff = 0.05f;//if(seuil==3) 
							total_number_of_plates = squares->total;
							init_contours_weight(tgray, dist_min_bord,
								hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
								sz.width, sz.height,
								storage, contours,
								squares,
								// diffeentes seuences contentant les notes correspondant aux diffeents critees 
								note_plus_grand_cosinus,
								note_rapport_long_cotes_opposes,
								note_plus_petit_sinus,
								note_plus_grand_sinus,
								notesym, cvApproxPoly_coeff);
							if (total_number_of_plates == squares->total) {
								cvMorphologyEx(gray1, gray2, NULL, convKernel, CV_MOP_CLOSE);
								cvFindContours(gray2, storage, &contours, sizeof(CvContour),
									CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
								init_contours_weight(tgray, dist_min_bord,
									hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
									sz.width, sz.height,
									storage, contours,
									squares,
									// diffeentes seuences contentant les notes correspondant aux diffeents critees 
									note_plus_grand_cosinus,
									note_rapport_long_cotes_opposes,
									note_plus_petit_sinus,
									note_plus_grand_sinus,
									notesym, cvApproxPoly_coeff);
							}
						}
					}
				}
			}
		}
	}
	// (on libee de la meoire)
	cvReleaseImage(&gray1);
	//	cvReleaseImage( &pyr );
	cvReleaseImage(&tgray);
	//		cvReleaseImage( &im_copy );
	cvReleaseImage(&gray2);
	cvReleaseStructuringElement(&convKernel);
#ifdef _DEBUG
	assert(note_plus_grand_cosinus->total == note_rapport_long_cotes_opposes->total &&
		note_plus_grand_cosinus->total == note_plus_petit_sinus->total &&
		note_plus_grand_cosinus->total == note_plus_grand_sinus->total &&
		note_plus_grand_cosinus->total == notesym->total);
#endif //_DEBUG
	//si rien n'a passele filtre, on arree le:
	//on renvoie une seuence de -1
	if (notesym->total == 0)
	{
		cvClearSeq(classement);
		centre = cvPoint(SHRT_MIN, SHRT_MIN);
		cvSeqPush(classement, &centre);
		cvSeqPush(classement, &centre);
		cvSeqPush(classement, &centre);
		cvSeqPush(classement, &centre);
		return classement;
	}
	// CLASSEMENT
	// renvoie la seuence de places epartir de la seuence de notes
	place_plus_grand_cosinus = sort(note_plus_grand_cosinus, storage);
	place_rapport_long_cotes_opposes = sort(note_rapport_long_cotes_opposes, storage);
	place_plus_petit_sinus = sort(note_plus_petit_sinus, storage);
	place_plus_grand_sinus = sort(note_plus_grand_sinus, storage);
	placesym = sort(notesym, storage);
#ifdef _DEBUG
	assert(place_plus_grand_cosinus->total == place_rapport_long_cotes_opposes->total &&
		place_plus_grand_cosinus->total == place_plus_petit_sinus->total &&
		place_plus_grand_cosinus->total == place_plus_grand_sinus->total &&
		place_plus_grand_cosinus->total == placesym->total);
#endif //_DEBUG
	//dans place, on met la somme des places pour chaque quadrilatee
	for (i = 0; i < place_plus_grand_cosinus->total; i++)
	{
		j = (*(int*)cvGetSeqElemLPR(place_plus_grand_cosinus, i, 0) + *(int*)cvGetSeqElemLPR(place_rapport_long_cotes_opposes, i, 0) +
			*(int*)cvGetSeqElemLPR(place_plus_petit_sinus, i, 0) + *(int*)cvGetSeqElemLPR(place_plus_grand_sinus, i, 0) +
			*(int*)cvGetSeqElemLPR(placesym, i, 0));
		cvSeqPush(place, &j);
	}
	// on initialise classementplace et classement
	cvSeqPush(classementplace, cvGetSeqElemLPR(place, 0, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 0, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 1, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 2, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 3, 0));
	//dans classement on met les quadrilatees
	//et dans classmentplace leurs sommes de places
	// (le tout de faen ordonnee)
	for (j = 1; j < place->total; j++)
	{
		for (i = 0; i < classementplace->total; i++)
		{
			k = 0;
			if (*(int*)cvGetSeqElemLPR(classementplace, i, 0) >
				*(int*)cvGetSeqElemLPR(place, j, 0) || (i == classementplace->total))
			{
				cvSeqInsert(classementplace, i, cvGetSeqElemLPR(place, j, 0));
				cvSeqInsert(classement, (i << 2), cvGetSeqElemLPR(squares, (j << 2), 0));
				cvSeqInsert(classement, (i << 2) + 1, cvGetSeqElemLPR(squares, (j << 2) + 1, 0));
				cvSeqInsert(classement, (i << 2) + 2, cvGetSeqElemLPR(squares, (j << 2) + 2, 0));
				cvSeqInsert(classement, (i << 2) + 3, cvGetSeqElemLPR(squares, (j << 2) + 3, 0));
				k = 1;
			}
			if (k == 1)
				break;
		}
	}
	// on va se resservir de squares et note_plus_grand_cosinus (on les vide donc)
	cvClearSeq(squares);
	cvClearSeq(note_plus_grand_cosinus);
	//dans squares on stocke les rectangles moyens
	//dans centres leurs centres
	//dans note_plus_grand_cosinus, leurs pondeations
	//on eudie le quart supeieur des quadrilatees
	//(ou 1 si y en a - de 4)
	max = (classement->total) >> 4;
	//s'il n'y a pas 4 quadrilatees on les prends tous
	if (max < 1) max = ((classement->total) >> 2);
	k = 0;
	for (j = 0; j < max; j++)//j est le no du quadrilatee auquel on cherche un sosie
	{
		//pt[((j<<2))&3] est un des quatre points du quadrilatee courant.
		pt[((j << 2)) & 3] = *(CvPoint*)cvGetSeqElemLPR(classement, (j << 2), 0);
		//centre est le centre du quadrilatee, il nous permet de l'identifier
		centre = moyenne4(pt[((j << 2)) & 3], pt[((j << 2) + 1) & 3], pt[((j << 2) + 2) & 3], pt[((j << 2) + 3) & 3]);
		if (k == 1)
		{
			m = 0;
			for (i = 0; i < centres->total; i++)// i est le no du qudrilatee qui est peut-ere un sosie
			{
				if (dist(centre, *(CvPoint*)cvGetSeqElemLPR(centres, i, 0)) < 10)
					//si les 2 centres sont emoins de 10 pixels d'intervalle
						//on fait la moyenne des centres
							//on fait la moyenne des rectangles 
								//le coef de pond augmente de 1
				{
					m = 1;
					pond = *(int*)cvGetSeqElemLPR(note_plus_grand_cosinus, i, 0) + 1;
					cvSeqRemove(note_plus_grand_cosinus, i);
					cvSeqInsert(note_plus_grand_cosinus, i, &pond);
					//dans centre on met la moyenne pondere des 2 centres
					//puis on stocke ce point dans la seq. centres
					centre = moyennepond(centre, *(CvPoint*)cvGetSeqElemLPR(centres, i, 0), 1, (pond - 1));
					cvSeqRemove(centres, i);
					cvSeqInsert(centres, i, &centre);
					//centre va servir echaque fois estocker les coordonnees de chaque coin du rectangle moyen
					for (l = 0; l < 4; l++)
					{
						//dans centre, on met les nouvelles coordonnees d'un coin du rectangle 
						//moyen c a d la moyenne pondere du point rectangle preedemment stockeet du 
						// point (du nouveau rectangle) le + proche
						centre = moyennepond(*(CvPoint*)cvGetSeqElemLPR(squares, (i << 2) + l, 0),
							trouveleplusproche(*(CvPoint*)cvGetSeqElemLPR(squares, (i << 2) + l, 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2), 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2) + 1, 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2) + 2, 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2) + 3, 0)), (pond - 1), 1);
						// et echaque fois on stocke le point qui vient d'ere calculedans squares
						cvSeqRemove(squares, (i << 2) + l);
						cvSeqInsert(squares, (i << 2) + l, &centre);
					}
				}
				//si on a associeles 2 rect., on break
				if (m == 1)
					break;
				// si on a pas trouveede sosie ej,
				// on le classe eune nouvelle place dans squares
				// avec une pondeation de 1
				if (i == centres->total)
				{
					cvSeqPush(centres, &centre);
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2), 0));
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2) + 1, 0));
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2) + 2, 0));
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2) + 3, 0));
					cvSeqPush(note_plus_grand_cosinus, &un);
				}
			}
		}
		if (k == 0)// la 1 ere fois  qu on fait cette boucle 
			// on compte systematiquement le quadrilatee comme nouveau
				// donc on le stocke eune nouvelle place dans squares
					// et on lui met une pondeation de 1
		{
			cvSeqPush(centres, &centre);
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 0, 0));
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 1, 0));
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 2, 0));
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 3, 0));
			cvSeqPush(note_plus_grand_cosinus, &un);
			k = 1;
		}
	}
	// leon reclasse nos rectangles par ordre de pondeation
	k = 0; l = 0;
	for (j = 1; j < note_plus_grand_cosinus->total; j++)
	{
		i = *cvGetSeqElemLPR(note_plus_grand_cosinus, j, 0);
		if (i > k)
		{
			k = i;
			l = j;
		}
	}
	// et dans ce classement on garde que le meilleur
	cvClearSeq(classement);
	cvSeqPush(classement, cvGetSeqElemLPR(squares, (l << 2), 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 1 + (l << 2), 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 2 + (l << 2), 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 3 + (l << 2), 0));	
	//il ne reste qu'un quadrilatee. on le renvoie
	return classement;
}
CvSeq* findSquares4_multiresolution
(
	IplImage* im_src, CvMemStorage* storage,
	const int nTresh, //
	//le nb de composantes couleurs ex nb_composantes_color=3 pour une image RGB
	const int nb_composantes_color,
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min
)
{
#ifdef _DEBUG	
	//show_cornerHarris(im_src);
#endif //_DEBUG
	CvSeq* fin = findSquares4
	(im_src, storage,
		nTresh, //
		nb_composantes_color, dist_min_bord,
		hauteur_plaque_min,
		rapport_largeur_sur_hauteur_min);
	if (fin) {
		if (fin->total == 4) {
			CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
			if (pt0->x >= 0 && pt0->y >= 0)	return fin;
		}
	}
	if (im_src->width > MIN_WIDTH_FOR_A_LP && im_src->height > MIN_HEIGHT_FOR_A_LP) {
		cv::Point p0, p1, p2, p3;
		IplImage* new_img_2 = cvCreateImage(cvSize(im_src->width * 0.5, im_src->height * 0.5), im_src->depth, im_src->nChannels);
		cvResize(im_src, new_img_2, CV_INTER_CUBIC);
		CvSeq* fin = findSquares4
		(new_img_2, storage,
			nTresh, //
			nb_composantes_color, dist_min_bord,
			hauteur_plaque_min,
			rapport_largeur_sur_hauteur_min);
		if (fin) {
			if (fin->total == 4) {
				CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
				if (pt0->x >= 0 && pt0->y >= 0) {
					resize(fin, 2.0);
					refine_cornerHarris(im_src, fin);
					return fin;
				}
			}
		}
		if (new_img_2->width > MIN_WIDTH_FOR_A_LP && new_img_2->height > MIN_HEIGHT_FOR_A_LP) {
			IplImage* new_img_4 = cvCreateImage(cvSize(new_img_2->width * 0.5, new_img_2->height * 0.5), new_img_2->depth, new_img_2->nChannels);
			cvResize(new_img_2, new_img_4, CV_INTER_CUBIC);
			CvSeq* fin = findSquares4
			(new_img_4, storage,
				nTresh, //
				nb_composantes_color, dist_min_bord,
				hauteur_plaque_min,
				rapport_largeur_sur_hauteur_min);
			if (fin) {
				if (fin->total == 4) {
					CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
					if (pt0->x >= 0 && pt0->y >= 0) {
						resize(fin, 4.0);
						refine_cornerHarris(im_src, fin);
					}
				}
			}
			return fin;
		}
		else {
			return fin;
		}
	}
	else {
		return fin;
	}
}
CvSeq* findSquares4_multiresolution
(IplImage* imgR, IplImage* imgG, IplImage* imgB, CvMemStorage* storage,
	const int nTresh, //
	//le nb de composantes couleurs ex nb_composantes_color=3 pour une image RGB
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min)
{
	CvSeq* fin = findSquares4
	(imgR, imgG, imgB, storage,
		nTresh, //
		dist_min_bord,
		hauteur_plaque_min,
		rapport_largeur_sur_hauteur_min);
	if (fin) {
		if (fin->total == 4) {
			CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
			if (pt0->x >= 0 && pt0->y >= 0)	return fin;
		}
	}
	if (imgR->width > MIN_WIDTH_FOR_A_LP && imgR->height > MIN_HEIGHT_FOR_A_LP) {
		cv::Point p0, p1, p2, p3;
		IplImage* new_img_2R = cvCreateImage(cvSize(imgR->width * 0.5, imgR->height * 0.5), imgR->depth, imgR->nChannels);
		cvResize(imgR, new_img_2R, CV_INTER_CUBIC);
		IplImage* new_img_2G = cvCreateImage(cvSize(imgR->width * 0.5, imgR->height * 0.5), imgR->depth, imgR->nChannels);
		cvResize(imgG, new_img_2G, CV_INTER_CUBIC);
		IplImage* new_img_2B = cvCreateImage(cvSize(imgR->width * 0.5, imgR->height * 0.5), imgR->depth, imgR->nChannels);
		cvResize(imgB, new_img_2B, CV_INTER_CUBIC);
		CvSeq* fin = findSquares4
		(new_img_2R, new_img_2G, new_img_2B
			, storage,
			nTresh, //
			dist_min_bord,
			hauteur_plaque_min,
			rapport_largeur_sur_hauteur_min);
		if (fin) {
			if (fin->total == 4) {
				CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
				if (pt0->x >= 0 && pt0->y >= 0) {
#ifdef _DEBUG	
					/*cv::Mat tmp=new_img_2R;
					imshow("new_img_2R",tmp);
					char c = cv::waitKey(0);
					tmp=new_img_2G;
					imshow("new_img_2G",tmp);
					c = cv::waitKey(0);
					tmp=new_img_2B;
					imshow("new_img_2B",tmp);
					c = cv::waitKey(0);*/
#endif //_DEBUG
					resize(fin, 2.0);
					refine_cornerHarris(imgG, fin);
					return fin;
				}
			}
		}
		if (new_img_2B->width > MIN_WIDTH_FOR_A_LP && new_img_2B->height > MIN_HEIGHT_FOR_A_LP) {
			IplImage* new_img_4R = cvCreateImage(cvSize(new_img_2B->width * 0.5, new_img_2B->height * 0.5), new_img_2B->depth, new_img_2B->nChannels);
			cvResize(new_img_2R, new_img_4R, CV_INTER_CUBIC);
			IplImage* new_img_4G = cvCreateImage(cvSize(new_img_2B->width * 0.5, new_img_2B->height * 0.5), new_img_2B->depth, new_img_2B->nChannels);
			cvResize(new_img_2G, new_img_4G, CV_INTER_CUBIC);
			IplImage* new_img_4B = cvCreateImage(cvSize(new_img_2B->width * 0.5, new_img_2B->height * 0.5), new_img_2B->depth, new_img_2B->nChannels);
			cvResize(new_img_2B, new_img_4B, CV_INTER_CUBIC);
			CvSeq* fin = findSquares4
			(new_img_4R, new_img_4G, new_img_4B, storage,
				nTresh, //
				dist_min_bord,
				hauteur_plaque_min,
				rapport_largeur_sur_hauteur_min);
			if (fin) {
				if (fin->total == 4) {
					CvPoint* pt0 = CV_GET_SEQ_ELEM(CvPoint, fin, 0);
					if (pt0->x >= 0 && pt0->y >= 0) {
#ifdef _DEBUG	
						/*cv::Mat tmp=new_img_4R;
						imshow("new_img_4R",tmp);
						char c = cv::waitKey(0);
						tmp=new_img_4G;
						imshow("new_img_4G",tmp);
						c = cv::waitKey(0);
						tmp=new_img_4B;
						imshow("new_img_4B",tmp);
						c = cv::waitKey(0);*/
#endif //_DEBUG
						resize(fin, 4.0);
						refine_cornerHarris(imgG, fin);
						return fin;
					}
				}
			}
		}
	}
	return fin;
}
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
// renvoie une seuence de 4 points encadrant la plaque
// trouveesur l'image im_src
// IplImage* im_src
// CvMemStorage* storage 
//peut traiter les images sources couleur ou en niveaux de gris
CvSeq* findSquares4
(IplImage* imgR, IplImage* imgG, IplImage* imgB, CvMemStorage* storage,
	const int nTresh, //
	//le nb de composantes couleurs ex nb_composantes_color=3 pour une image RGB
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min)
{
	if (imgR->width != imgG->width || imgR->width != imgB->width
		|| imgG->width != imgB->width
		||
		imgR->height != imgG->height || imgR->height != imgB->height
		|| imgG->height != imgB->height) {
		return NULL;
	}
	CvSeq* contours = 0;
	CvPoint centre;
	int  i, j, k, l, m,
		un = 1,
		pond, max;
	/*	*/
	int thresh_canny = 50;
	CvSize sz = cvSize(imgR->width, imgR->height);
	//quelques images temporaires qui permettront de travailler
	IplImage* gray1 = cvCreateImage(sz, IPL_DEPTH_8U//unsigned 8-bit integers im_src->depth
		, 1);
	IplImage* tgray = cvCreateImage(sz, IPL_DEPTH_8U//im_src->depth
		, 1);
	IplImage* gray2 = cvCreateImage(sz, IPL_DEPTH_8U//unsigned 8-bit integers im_src->depth
		, 1);
	int filterSize = 5;
	IplConvKernel* convKernel = cvCreateStructuringElementEx
	(filterSize, filterSize, (filterSize - 1) / 2, (filterSize - 1) / 2,
		CV_SHAPE_ELLIPSE, NULL);
	// squares et classement permettront de travailler sur les listes de quadrilatees
	CvSeq* squares = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	CvSeq* classement = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	// centres est une liste de centres de quadrilatees utilisee ela fin
	CvSeq* centres = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
	// diffeentes seuences contentant les notes correspondant aux diffeents critees 
	CvSeq* note_plus_grand_cosinus = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* note_rapport_long_cotes_opposes = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* note_plus_petit_sinus = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* note_plus_grand_sinus = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	CvSeq* notesym = cvCreateSeq(CV_32FC1, sizeof(CvSeq), sizeof(float), storage);
	// diffeentes seuences contenant les places correspondant aux diffeents critees
	CvSeq* place_plus_grand_cosinus = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* place_rapport_long_cotes_opposes = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* place_plus_petit_sinus = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* place_plus_grand_sinus = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeq* placesym = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	// place contiendra la liste de sommes de places
	CvSeq* place = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	//classementplace contiendra la liste des quadrilatees ordonnee selon les sommes de places
	CvSeq* classementplace = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	// selectionne le + grand rectangle dans l'image
	// dont les dimensions sont divisibles par 2
	cvSetImageROI(imgR, cvRect(0, 0, sz.width, sz.height));
	cvSetImageROI(imgG, cvRect(0, 0, sz.width, sz.height));
	cvSetImageROI(imgB, cvRect(0, 0, sz.width, sz.height));
	// pt servira eplusieurs reprises estocker des quadrilatees
	// (tableau de 4 points)
	CvPoint pt[4];
	int nb_pts_sup_threshold = (sz.width * sz.height);
	int seuil;
	// on teste toutes les couleurs R,G,B
	//const int sz.width=imgR->width;const int sz.height=imgR->height;
	cvCopy(imgR, tgray, 0);
	// apply Canny. Take the upper threshold from slider
	// and set the lower to 0 (which forces edges merging) 
	/*cvCanny( tgray, gray1, (thresh_canny>>1), 0, 3//Aperture parameter for Sobel operator
	);*/
	cvCanny(tgray, gray1, thresh_canny, 255 - thresh_canny, 5//Aperture parameter for Sobel operator
	);
	// dilate canny output to remove potential
	// holes between edge segments 
	//cvErode( gray1, gray1, 0, 1 );//a rajouter
	float cvApproxPoly_coeff = 0.02f;
	// dilate canny output to remove potential
	// holes between edge segments 
	//cvErode( gray1, gray1, 0, 1 );//a rajouter
	// trouve les contours et en fait une liste
	cvFindContours(gray1, storage, &contours, sizeof(CvContour),
		CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	// teste chaque contour
	//calcule le poids des contours
	init_contours_weight(tgray, dist_min_bord,
		hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
		sz.width, sz.height,
		storage, contours,
		squares,
		// diffeentes seuences contentant les notes correspondant aux diffeents critees 
		note_plus_grand_cosinus,
		note_rapport_long_cotes_opposes,
		note_plus_petit_sinus,
		note_plus_grand_sinus,
		notesym, cvApproxPoly_coeff);
	if (!squares->total) {
		/*
		Mat cv::cvarrToMat 	( 	const CvArr *  	arr,
		bool  	copyData = false,
		bool  	allowND = true,
		int  	coiMode = 0,
		AutoBuffer< double > *  	buf = 0
	)
	*/
		cv::Mat img = cv::cvarrToMat(tgray);
		cv::Mat image_final = cv::cvarrToMat(gray1);
		int blockSize = (sz.width + sz.height) / 2;
		if (blockSize % 2 == 0) blockSize += 1;
		cv::adaptiveThreshold(img, image_final, 127, cv::ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, blockSize, 0);
		cvFindContours(gray1, storage, &contours, sizeof(CvContour),
			CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		int total_number_of_plates = squares->total;
		init_contours_weight(tgray, dist_min_bord,
			hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
			sz.width, sz.height,
			storage, contours,
			squares,
			// diffeentes seuences contentant les notes correspondant aux diffeents critees 
			note_plus_grand_cosinus,
			note_rapport_long_cotes_opposes,
			note_plus_petit_sinus,
			note_plus_grand_sinus,
			notesym, cvApproxPoly_coeff);
		// teste chaque contour
		//calcule le poids des contours
		if (total_number_of_plates == squares->total) {
			// on teste N niveaux de seuillage
			for (seuil = 0; seuil < nTresh; seuil++)
			{
				int current_nb_pts_sup_threshold = 0;
				LPRThreshold(imgR, gray1, ((seuil + 1) * 200 / nTresh) + 25,
					current_nb_pts_sup_threshold);// trouve les contours et en fait une liste
				if (current_nb_pts_sup_threshold >= 0 &&
					current_nb_pts_sup_threshold <= nb_pts_sup_threshold) {
					///////////////////////////////////////////////////////////////////////////////////////////
					// retourne le nb de points en commun entre deux images
#ifdef _DEBUG
					assert(current_nb_pts_sup_threshold <= nb_pts_sup_threshold);
#endif //_DEBUG
					float percentage_match = 1.0f - ((float)(nb_pts_sup_threshold -
						current_nb_pts_sup_threshold)) / (sz.width * sz.height);
#ifdef _DEBUG
					assert(percentage_match > -FLT_EPSILON &&
						percentage_match < 1.0f + FLT_EPSILON);
#endif //_DEBUG
					if (percentage_match < 0.75) {
						cvFindContours(gray1, storage, &contours, sizeof(CvContour),
							CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
						nb_pts_sup_threshold = current_nb_pts_sup_threshold;
						if (seuil == 0) cvApproxPoly_coeff = 0.02f;
						else if (seuil == 1) cvApproxPoly_coeff = 0.03f;
						else if (seuil == 2) cvApproxPoly_coeff = 0.04f;
						else cvApproxPoly_coeff = 0.05f;//if(seuil==3) 
						int total_number_of_plates = squares->total;
						init_contours_weight(tgray, dist_min_bord,
							hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
							sz.width, sz.height,
							storage, contours,
							squares,
							// diffeentes seuences contentant les notes correspondant aux diffeents critees 
							note_plus_grand_cosinus,
							note_rapport_long_cotes_opposes,
							note_plus_petit_sinus,
							note_plus_grand_sinus,
							notesym, cvApproxPoly_coeff);
						// teste chaque contour
						//calcule le poids des contours
						/*init_contours_weight(tgray,dist_min_bord,
							hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
							sz.width, sz.height,
							storage, contours,
							squares,
							// diffeentes seuences contentant les notes correspondant aux diffeents critees
							note_plus_grand_cosinus,
							note_rapport_long_cotes_opposes,
							note_plus_petit_sinus,
							note_plus_grand_sinus,
							notesym, cvApproxPoly_coeff);*/
						if (total_number_of_plates == squares->total) {
							cvMorphologyEx(gray1, gray2, NULL, convKernel, CV_MOP_CLOSE);
							cvFindContours(gray2, storage, &contours, sizeof(CvContour),
								CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
							init_contours_weight(tgray, dist_min_bord,
								hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
								sz.width, sz.height,
								storage, contours,
								squares,
								// diffeentes seuences contentant les notes correspondant aux diffeents critees 
								note_plus_grand_cosinus,
								note_rapport_long_cotes_opposes,
								note_plus_petit_sinus,
								note_plus_grand_sinus,
								notesym, cvApproxPoly_coeff);
						}
					}
				}
			}
		}
	}
	cvCopy(imgG, tgray, 0);
	// apply Canny. Take the upper threshold from slider
	// and set the lower to 0 (which forces edges merging) 
	cvCanny(tgray, gray1, thresh_canny, 255 - thresh_canny, 5//Aperture parameter for Sobel operator
	);
	// dilate canny output to remove potential
	// holes between edge segments 
	//cvErode( gray1, gray1, 0, 1 );//a rajouter
	// trouve les contours et en fait une liste
	cvFindContours(gray1, storage, &contours, sizeof(CvContour),
		CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	int total_number_of_plates = squares->total;
	cvApproxPoly_coeff = 0.02f;
	// teste chaque contour
	//calcule le poids des contours
	init_contours_weight(tgray, dist_min_bord,
		hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
		sz.width, sz.height,
		storage, contours,
		squares,
		// diffeentes seuences contentant les notes correspondant aux diffeents critees 
		note_plus_grand_cosinus,
		note_rapport_long_cotes_opposes,
		note_plus_petit_sinus,
		note_plus_grand_sinus,
		notesym, cvApproxPoly_coeff);
	if (!squares->total) {
		/*
		Mat cv::cvarrToMat 	( 	const CvArr *  	arr,
		bool  	copyData = false,
		bool  	allowND = true,
		int  	coiMode = 0,
		AutoBuffer< double > *  	buf = 0
	)
	*/
		cv::Mat img = cv::cvarrToMat(tgray);
		cv::Mat image_final = cv::cvarrToMat(gray1);
		int blockSize = (sz.width + sz.height) / 2;
		if (blockSize % 2 == 0) blockSize += 1;
		cv::adaptiveThreshold(img, image_final, 127, cv::ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, blockSize, 0);
#ifdef LPR_SAVE_DISPLAY
		cv::Mat debug = cv::cvarrToMat(gray1);
		imshow("adaptiveThreshold", debug);
		char c = cv::waitKey(0);
#endif //_DEBUG
		cvFindContours(gray1, storage, &contours, sizeof(CvContour),
			CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		init_contours_weight(tgray, dist_min_bord,
			hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
			sz.width, sz.height,
			storage, contours,
			squares,
			// diffeentes seuences contentant les notes correspondant aux diffeents critees 
			note_plus_grand_cosinus,
			note_rapport_long_cotes_opposes,
			note_plus_petit_sinus,
			note_plus_grand_sinus,
			notesym, cvApproxPoly_coeff);
		// teste chaque contour
		//calcule le poids des contours
		if (total_number_of_plates == squares->total) {
			// on teste N niveaux de seuillage
			nb_pts_sup_threshold = (sz.width * sz.height);
			for (seuil = 0; seuil < nTresh; seuil++)
			{
				int current_nb_pts_sup_threshold = 0;
				LPRThreshold(imgG, gray1, ((seuil + 1) * 200 / nTresh) + 25,
					current_nb_pts_sup_threshold);// trouve les contours et en fait une liste
				if (current_nb_pts_sup_threshold >= 0 &&
					current_nb_pts_sup_threshold <= nb_pts_sup_threshold) {
					///////////////////////////////////////////////////////////////////////////////////////////
					// retourne le nb de points en commun entre deux images
#ifdef _DEBUG
					assert(current_nb_pts_sup_threshold <= nb_pts_sup_threshold);
#endif //_DEBUG
					float percentage_match = 1.0f - ((float)(nb_pts_sup_threshold -
						current_nb_pts_sup_threshold)) / (sz.width * sz.height);
#ifdef _DEBUG
					assert(percentage_match > -FLT_EPSILON &&
						percentage_match < 1.0f + FLT_EPSILON);
#endif //_DEBUG
					if (percentage_match < 0.75) {
						cvFindContours(gray1, storage, &contours, sizeof(CvContour),
							CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
						nb_pts_sup_threshold = current_nb_pts_sup_threshold;
						if (seuil == 0) cvApproxPoly_coeff = 0.02f;
						else if (seuil == 1) cvApproxPoly_coeff = 0.03f;
						else if (seuil == 2) cvApproxPoly_coeff = 0.04f;
						else cvApproxPoly_coeff = 0.05f;//if(seuil==3) 
						int total_number_of_plates = squares->total;
						init_contours_weight(tgray, dist_min_bord,
							hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
							sz.width, sz.height,
							storage, contours,
							squares,
							// diffeentes seuences contentant les notes correspondant aux diffeents critees 
							note_plus_grand_cosinus,
							note_rapport_long_cotes_opposes,
							note_plus_petit_sinus,
							note_plus_grand_sinus,
							notesym, cvApproxPoly_coeff);
						// teste chaque contour
						//calcule le poids des contours
						/**/
						if (total_number_of_plates == squares->total) {
							cvMorphologyEx(gray1, gray2, NULL, convKernel, CV_MOP_CLOSE);
							cvFindContours(gray2, storage, &contours, sizeof(CvContour),
								CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
							init_contours_weight(tgray, dist_min_bord,
								hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
								sz.width, sz.height,
								storage, contours,
								squares,
								// diffeentes seuences contentant les notes correspondant aux diffeents critees 
								note_plus_grand_cosinus,
								note_rapport_long_cotes_opposes,
								note_plus_petit_sinus,
								note_plus_grand_sinus,
								notesym, cvApproxPoly_coeff);
						}
					}
				}
			}
		}
	}
	cvCopy(imgB, tgray, 0);
	// apply Canny. Take the upper threshold from slider
	// and set the lower to 0 (which forces edges merging) 
	cvCanny(tgray, gray1, thresh_canny, 255 - thresh_canny, 5//Aperture parameter for Sobel operator
	);
	// dilate canny output to remove potential
	// holes between edge segments 
	//cvErode( gray1, gray1, 0, 1 );//a rajouter
#ifdef LPR_SAVE_DISPLAY
	debug = cv::cvarrToMat(gray1);
	imshow("canny", debug);
	c = cv::waitKey(0);
#endif //_DEBUG
	// trouve les contours et en fait une liste
	cvFindContours(gray1, storage, &contours, sizeof(CvContour),
		CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	cvApproxPoly_coeff = 0.02f;
	total_number_of_plates = squares->total;
	// teste chaque contour
	//calcule le poids des contours
	init_contours_weight(tgray, dist_min_bord,
		hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
		sz.width, sz.height,
		storage, contours,
		squares,
		// diffeentes seuences contentant les notes correspondant aux diffeents critees 
		note_plus_grand_cosinus,
		note_rapport_long_cotes_opposes,
		note_plus_petit_sinus,
		note_plus_grand_sinus,
		notesym, cvApproxPoly_coeff);
	if (!squares->total) {
		/*
		Mat cv::cvarrToMat 	( 	const CvArr *  	arr,
		bool  	copyData = false,
		bool  	allowND = true,
		int  	coiMode = 0,
		AutoBuffer< double > *  	buf = 0
	)
	*/
		cv::Mat img = cv::cvarrToMat(tgray);
		cv::Mat image_final = cv::cvarrToMat(gray1);
		//cv::adaptiveThreshold(img, image_final, 127, cv::ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 17, 0);
		int blockSize = (sz.width + sz.height) / 2;
		if (blockSize % 2 == 0) blockSize += 1;
		cv::adaptiveThreshold(img, image_final, 127, cv::ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, blockSize, 0);
#ifdef LPR_SAVE_DISPLAY
		cv::Mat debug = cv::cvarrToMat(gray1);
		imshow("adaptiveThreshold", debug);
		char c = cv::waitKey(0);
#endif //_DEBUG
		cvFindContours(gray1, storage, &contours, sizeof(CvContour),
			CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		init_contours_weight(tgray, dist_min_bord,
			hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
			sz.width, sz.height,
			storage, contours,
			squares,
			// diffeentes seuences contentant les notes correspondant aux diffeents critees 
			note_plus_grand_cosinus,
			note_rapport_long_cotes_opposes,
			note_plus_petit_sinus,
			note_plus_grand_sinus,
			notesym, cvApproxPoly_coeff);
		// teste chaque contour
		//calcule le poids des contours
		if (total_number_of_plates == squares->total) {
			// on teste N niveaux de seuillage
			nb_pts_sup_threshold = (sz.width * sz.height);
			for (seuil = 0; seuil < nTresh; seuil++)
			{
				int current_nb_pts_sup_threshold = 0;
				LPRThreshold(imgB, gray1, ((seuil + 1) * 200 / nTresh) + 25,
					current_nb_pts_sup_threshold);// trouve les contours et en fait une liste
				if (current_nb_pts_sup_threshold >= 0 &&
					current_nb_pts_sup_threshold <= nb_pts_sup_threshold) {
					///////////////////////////////////////////////////////////////////////////////////////////
					// retourne le nb de points en commun entre deux images
#ifdef _DEBUG
					assert(current_nb_pts_sup_threshold <= nb_pts_sup_threshold);
#endif //_DEBUG
					float percentage_match = 1.0f - ((float)(nb_pts_sup_threshold -
						current_nb_pts_sup_threshold)) / (sz.width * sz.height);
#ifdef _DEBUG
					assert(percentage_match > -FLT_EPSILON &&
						percentage_match < 1.0f + FLT_EPSILON);
#endif //_DEBUG
					if (percentage_match < 0.75) {
						cvFindContours(gray1, storage, &contours, sizeof(CvContour),
							CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
						nb_pts_sup_threshold = current_nb_pts_sup_threshold;
						if (seuil == 0) cvApproxPoly_coeff = 0.02f;
						else if (seuil == 1) cvApproxPoly_coeff = 0.03f;
						else if (seuil == 2) cvApproxPoly_coeff = 0.04f;
						else cvApproxPoly_coeff = 0.05f;//if(seuil==3) 
						int total_number_of_plates = squares->total;
						init_contours_weight(tgray, dist_min_bord,
							hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
							sz.width, sz.height,
							storage, contours,
							squares,
							// diffeentes seuences contentant les notes correspondant aux diffeents critees 
							note_plus_grand_cosinus,
							note_rapport_long_cotes_opposes,
							note_plus_petit_sinus,
							note_plus_grand_sinus,
							notesym, cvApproxPoly_coeff);
						// teste chaque contour
						//calcule le poids des contours
						/**/
						if (total_number_of_plates == squares->total) {
							cvMorphologyEx(gray1, gray2, NULL, convKernel, CV_MOP_CLOSE);
							cvFindContours(gray2, storage, &contours, sizeof(CvContour),
								CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
							init_contours_weight(tgray, dist_min_bord,
								hauteur_plaque_min, rapport_largeur_sur_hauteur_min,
								sz.width, sz.height,
								storage, contours,
								squares,
								// diffeentes seuences contentant les notes correspondant aux diffeents critees 
								note_plus_grand_cosinus,
								note_rapport_long_cotes_opposes,
								note_plus_petit_sinus,
								note_plus_grand_sinus,
								notesym, cvApproxPoly_coeff);
						}
					}
				}
			}
		}
	}
	// (on libee de la meoire)
	cvReleaseImage(&gray1);
	cvReleaseImage(&tgray);
	cvReleaseImage(&gray2);
	cvReleaseStructuringElement(&convKernel);
	/*
	cvReleaseImage( &imgR );
	cvReleaseImage( &imgG );
	cvReleaseImage( &imgB );
	*/
#ifdef _DEBUG
	assert(note_plus_grand_cosinus->total == note_rapport_long_cotes_opposes->total &&
		note_plus_grand_cosinus->total == note_plus_petit_sinus->total &&
		note_plus_grand_cosinus->total == note_plus_grand_sinus->total &&
		note_plus_grand_cosinus->total == notesym->total);
#endif //_DEBUG
	//si rien n'a passele filtre, on arree le:
	//on renvoie une seuence de -1
	if (notesym->total == 0)
	{
		cvClearSeq(classement);
		centre = cvPoint(SHRT_MIN, SHRT_MIN);
		cvSeqPush(classement, &centre);
		cvSeqPush(classement, &centre);
		cvSeqPush(classement, &centre);
		cvSeqPush(classement, &centre);
		return classement;
	}
	// CLASSEMENT
	// renvoie la seuence de places epartir de la seuence de notes
	place_plus_grand_cosinus = sort(note_plus_grand_cosinus, storage);
	place_rapport_long_cotes_opposes = sort(note_rapport_long_cotes_opposes, storage);
	place_plus_petit_sinus = sort(note_plus_petit_sinus, storage);
	place_plus_grand_sinus = sort(note_plus_grand_sinus, storage);
	placesym = sort(notesym, storage);
#ifdef _DEBUG
	assert(place_plus_grand_cosinus->total == place_rapport_long_cotes_opposes->total &&
		place_plus_grand_cosinus->total == place_plus_petit_sinus->total &&
		place_plus_grand_cosinus->total == place_plus_grand_sinus->total &&
		place_plus_grand_cosinus->total == placesym->total);
#endif //_DEBUG
	//dans place, on met la somme des places pour chaque quadrilatee
	for (i = 0; i < place_plus_grand_cosinus->total; i++)
	{
		j = (*(int*)cvGetSeqElemLPR(place_plus_grand_cosinus, i, 0) + *(int*)cvGetSeqElemLPR(place_rapport_long_cotes_opposes, i, 0) +
			*(int*)cvGetSeqElemLPR(place_plus_petit_sinus, i, 0) + *(int*)cvGetSeqElemLPR(place_plus_grand_sinus, i, 0) +
			*(int*)cvGetSeqElemLPR(placesym, i, 0));
		cvSeqPush(place, &j);
	}
	// on initialise classementplace et classement
	cvSeqPush(classementplace, cvGetSeqElemLPR(place, 0, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 0, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 1, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 2, 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 3, 0));
	//dans classement on met les quadrilatees
	//et dans classmentplace leurs sommes de places
	// (le tout de faen ordonnee)
	for (j = 1; j < place->total; j++)
	{
		for (i = 0; i < classementplace->total; i++)
		{
			k = 0;
			if (*(int*)cvGetSeqElemLPR(classementplace, i, 0) >
				*(int*)cvGetSeqElemLPR(place, j, 0) || (i == classementplace->total))
			{
				cvSeqInsert(classementplace, i, cvGetSeqElemLPR(place, j, 0));
				cvSeqInsert(classement, (i << 2), cvGetSeqElemLPR(squares, (j << 2), 0));
				cvSeqInsert(classement, (i << 2) + 1, cvGetSeqElemLPR(squares, (j << 2) + 1, 0));
				cvSeqInsert(classement, (i << 2) + 2, cvGetSeqElemLPR(squares, (j << 2) + 2, 0));
				cvSeqInsert(classement, (i << 2) + 3, cvGetSeqElemLPR(squares, (j << 2) + 3, 0));
				k = 1;
			}
			if (k == 1)
				break;
		}
	}
	// on va se resservir de squares et note_plus_grand_cosinus (on les vide donc)
	cvClearSeq(squares);
	cvClearSeq(note_plus_grand_cosinus);
	//dans squares on stocke les rectangles moyens
	//dans centres leurs centres
	//dans note_plus_grand_cosinus, leurs pondeations
	//on eudie le quart supeieur des quadrilatees
	//(ou 1 si y en a - de 4)
	max = (classement->total) >> 4;
	//s'il n'y a pas 4 quadrilatees on les prends tous
	if (max < 1) max = ((classement->total) >> 2);
	k = 0;
	for (j = 0; j < max; j++)//j est le no du quadrilatee auquel on cherche un sosie
	{
		//pt[((j<<2))&3] est un des quatre points du quadrilatee courant.
		pt[((j << 2)) & 3] = *(CvPoint*)cvGetSeqElemLPR(classement, (j << 2), 0);
		//centre est le centre du quadrilatee, il nous permet de l'identifier
		centre = moyenne4(pt[((j << 2)) & 3], pt[((j << 2) + 1) & 3], pt[((j << 2) + 2) & 3], pt[((j << 2) + 3) & 3]);
		if (k == 1)
		{
			m = 0;
			for (i = 0; i < centres->total; i++)// i est le no du qudrilatee qui est peut-ere un sosie
			{
				if (dist(centre, *(CvPoint*)cvGetSeqElemLPR(centres, i, 0)) < 10)
					//si les 2 centres sont emoins de 10 pixels d'intervalle
						//on fait la moyenne des centres
							//on fait la moyenne des rectangles 
								//le coef de pond augmente de 1
				{
					m = 1;
					pond = *(int*)cvGetSeqElemLPR(note_plus_grand_cosinus, i, 0) + 1;
					cvSeqRemove(note_plus_grand_cosinus, i);
					cvSeqInsert(note_plus_grand_cosinus, i, &pond);
					//dans centre on met la moyenne pondere des 2 centres
					//puis on stocke ce point dans la seq. centres
					centre = moyennepond(centre, *(CvPoint*)cvGetSeqElemLPR(centres, i, 0), 1, (pond - 1));
					cvSeqRemove(centres, i);
					cvSeqInsert(centres, i, &centre);
					//centre va servir echaque fois estocker les coordonnees de chaque coin du rectangle moyen
					for (l = 0; l < 4; l++)
					{
						//dans centre, on met les nouvelles coordonnees d'un coin du rectangle 
						//moyen c a d la moyenne pondere du point rectangle preedemment stockeet du 
						// point (du nouveau rectangle) le + proche
						centre = moyennepond(*(CvPoint*)cvGetSeqElemLPR(squares, (i << 2) + l, 0),
							trouveleplusproche(*(CvPoint*)cvGetSeqElemLPR(squares, (i << 2) + l, 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2), 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2) + 1, 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2) + 2, 0),
								*(CvPoint*)cvGetSeqElemLPR(classement, (j << 2) + 3, 0)), (pond - 1), 1);
						// et echaque fois on stocke le point qui vient d'ere calculedans squares
						cvSeqRemove(squares, (i << 2) + l);
						cvSeqInsert(squares, (i << 2) + l, &centre);
					}
				}
				//si on a associeles 2 rect., on break
				if (m == 1)
					break;
				// si on a pas trouveede sosie ej,
				// on le classe eune nouvelle place dans squares
				// avec une pondeation de 1
				if (i == centres->total)
				{
					cvSeqPush(centres, &centre);
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2), 0));
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2) + 1, 0));
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2) + 2, 0));
					cvSeqPush(squares, cvGetSeqElemLPR(classement, (j << 2) + 3, 0));
					cvSeqPush(note_plus_grand_cosinus, &un);
				}
			}
		}
		if (k == 0)// la 1 ere fois  qu on fait cette boucle 
			// on compte systematiquement le quadrilatee comme nouveau
				// donc on le stocke eune nouvelle place dans squares
					// et on lui met une pondeation de 1
		{
			cvSeqPush(centres, &centre);
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 0, 0));
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 1, 0));
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 2, 0));
			cvSeqPush(squares, cvGetSeqElemLPR(classement, 3, 0));
			cvSeqPush(note_plus_grand_cosinus, &un);
			k = 1;
		}
	}
	// leon reclasse nos rectangles par ordre de pondeation
	k = 0; l = 0;
	for (j = 1; j < note_plus_grand_cosinus->total; j++)
	{
		i = *cvGetSeqElemLPR(note_plus_grand_cosinus, j, 0);
		if (i > k)
		{
			k = i;
			l = j;
		}
	}
	// et dans ce classement on garde que le meilleur
	cvClearSeq(classement);
	cvSeqPush(classement, cvGetSeqElemLPR(squares, (l << 2), 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 1 + (l << 2), 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 2 + (l << 2), 0));
	cvSeqPush(classement, cvGetSeqElemLPR(squares, 3 + (l << 2), 0));
	//il ne reste qu'un quadrilatee. on le renvoie
	return classement;
}
bool is_valid(const cv::Point& top_left_,
	const cv::Point& top_right_,
	const cv::Point& bottom_right_,
	const cv::Point& bottom_left_) {
	std::vector<cv::Point> points;
	points.push_back(cv::Point(top_left_.x, top_left_.y));
	points.push_back(cv::Point(top_right_.x, top_right_.y));
	points.push_back(cv::Point(bottom_right_.x, bottom_right_.y));
	points.push_back(cv::Point(bottom_left_.x, bottom_left_.y));
	cv::RotatedRect box = cv::minAreaRect(points);
	float rotated_area = box.size.area();
	float contourArea_ = cv::contourArea(points);
	//from -45 to +45 degrees
	float angle_ = box.angle;
	cv::Rect global_rect = box.boundingRect();
	if (angle_ >= 70.0f || angle_ <= -70.0f || (angle_ >= -20.0f && angle_ <= 20.0f)) {
		cv::Point2f pts[4];
		box.points(pts);
#ifdef _DEBUG
		std::list<int> ordonnees;
		ordonnees.push_back(fabs(pts[0].y - pts[1].y));
		ordonnees.push_back(fabs(pts[1].y - pts[2].y));
		ordonnees.push_back(fabs(pts[2].y - pts[3].y));
		ordonnees.push_back(fabs(pts[3].y - pts[0].y));
		std::list<int> abscisses;
		abscisses.push_back(fabs(pts[0].x - pts[1].x));
		abscisses.push_back(fabs(pts[1].x - pts[2].x));
		abscisses.push_back(fabs(pts[2].x - pts[3].x));
		abscisses.push_back(fabs(pts[3].x - pts[0].x));
#endif //_DEBUG
		float width_ = std::max({ fabs(pts[0].x - pts[1].x), fabs(pts[1].x - pts[2].x),
			fabs(pts[2].x - pts[3].x), fabs(pts[3].x - pts[0].x) });
		float height_ = std::max({ fabs(pts[0].y - pts[1].y), fabs(pts[1].y - pts[2].y),
			fabs(pts[2].y - pts[3].y), fabs(pts[3].y - pts[0].y) });
		return (float(global_rect.width * global_rect.height) < contourArea_ * 5.0f//ok
			&& rotated_area < contourArea_ * 2.5f//ok
			&& 1.5f * global_rect.width > global_rect.height//ok
			&& width_ >= height_
			);
	}
	else {
#ifdef _DEBUG
		//bottom left-top left-top right-bottom right
		assert(angle_ <= 90.0f && angle_ >= -90.0f);
		cv::Point2f pts[4];
		box.points(pts);
		//angle entre l'horizontal et la plus grande dimension
		//width et la dimension la plus proche de l'horizontal
		if (angle_ > 45.0f) {
			assert(box.size.width - 1 <= box.size.height);
		}
		else if (angle_ < -45.0f) {
			assert(box.size.width - 1 <= box.size.height);
		}
		else {
			assert(box.size.width >= box.size.height - 1);
		}
#endif //_DEBUG
		/*
		return (float(global_rect.width * global_rect.height) < contourArea_ * 5.0f//ok
			&& rotated_area < contourArea_ * 2.5f//ok
			&& fabs(angle_) < 30.0f//this has to be fixed
			&& 1.5f*global_rect.width > global_rect.height//ok
			);*/
		if (float(global_rect.width * global_rect.height) < contourArea_ * 5.0f//ok
			&& rotated_area < contourArea_ * 2.5f//ok
			&& 1.5f * global_rect.width > global_rect.height//ok
			&& box.size.width >= box.size.height
			)
		{
			return true;
			//first case
			if (box.size.width <= box.size.height) {
				return (angle_ < -60);
			}
			else return (angle_ > -30);
		}
		//first case
		if (box.size.width <= box.size.height) {
			return (float(global_rect.width * global_rect.height) < contourArea_ * 5.0f//ok
				&& rotated_area < contourArea_ * 2.5f//ok
				&& (angle_ < -60)
				&& 1.5f * global_rect.width > global_rect.height//ok
				);
			//return (angle_ < -60);
		}
		else {
			return (float(global_rect.width * global_rect.height) < contourArea_ * 5.0f//ok
				&& rotated_area < contourArea_ * 2.5f//ok
				&& (angle_ > -30)
				&& 1.5f * global_rect.width > global_rect.height//ok
				);
			//return (angle_ > -30);
		}
	}
}
//peut traiter les images sources couleur ou en niveaux de gris
bool trouve_la_plaque(IplImage* etude, const cv::Rect& global_rect, int nTresh
	, const int mean_carac
	, const int mean_fond,
	cv::Point& p0, cv::Point& p1, cv::Point& p2, cv::Point& p3,
	cv::Point& top_left_,
	cv::Point& top_right_,
	cv::Point& bottom_right_,
	cv::Point& bottom_left_, cv::Rect& rect_OpenLP,
	const int dist_min_bord
	,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min)
{
	CvSeq* fin = NULL;
	CvMemStorage* storage = cvCreateMemStorage(0);
	// traite l'image
	//peut traiter les images sources couleur ou en niveaux de gris	
	fin = findSquares4_multiresolution(etude, global_rect, storage, nTresh, mean_carac
		, mean_fond
		, dist_min_bord, hauteur_plaque_min,
		rapport_largeur_sur_hauteur_min);
	if (fin) {
		CvPoint* pt =
			CV_GET_SEQ_ELEM(CvPoint, fin, 0);
		if (pt != NULL)
		{
			p0.x = pt->x;
			p0.y = pt->y;
			pt = CV_GET_SEQ_ELEM(CvPoint, fin, 1);
			if (pt != NULL)
			{
				p1.x = pt->x;
				p1.y = pt->y;
				pt = CV_GET_SEQ_ELEM(CvPoint, fin, 2);
				if (pt != NULL)
				{
					p2.x = pt->x;
					p2.y = pt->y;
					pt = CV_GET_SEQ_ELEM(CvPoint, fin, 3);
					if (pt != NULL)
					{
						p3.x = pt->x;
						p3.y = pt->y;
					}
					else {
						p0.x = p0.y = SHRT_MIN;
						p1.x = p1.y = SHRT_MIN;
						p2.x = p2.y = SHRT_MIN;
						p3.x = p3.y = SHRT_MIN;
					}
				}
				else {
					p0.x = p0.y = SHRT_MIN;
					p1.x = p1.y = SHRT_MIN;
					p2.x = p2.y = SHRT_MIN;
					p3.x = p3.y = SHRT_MIN;
				}
			}
			else {
				p0.x = p0.y = SHRT_MIN;
				p1.x = p1.y = SHRT_MIN;
				p2.x = p2.y = SHRT_MIN;
				p3.x = p3.y = SHRT_MIN;
			}
		}
		else {
			p0.x = p0.y = SHRT_MIN;
			p1.x = p1.y = SHRT_MIN;
			p2.x = p2.y = SHRT_MIN;
			p3.x = p3.y = SHRT_MIN;
		}
	}
	else {
		p0.x = p0.y = SHRT_MIN;
		p1.x = p1.y = SHRT_MIN;
		p2.x = p2.y = SHRT_MIN;
		p3.x = p3.y = SHRT_MIN;
	}
	// on libee la meoire occupe pas les cvseq
	cvClearMemStorage(storage);
	cvReleaseMemStorage(&storage);
	if (p3.x != SHRT_MIN) {
		cv::Point bottom_right;
		cv::Point top_right;
		cv::Point top_left;
		cv::Point bottom_left;
		cv::Rect r(array2CRect(cv::Point(p0.x, p0.y), cv::Point(p1.x, p1.y),
			cv::Point(p2.x, p2.y), cv::Point(p3.x, p3.y),
			top_left,
			top_right,
			bottom_right,
			bottom_left));
		if (r.x >= 0 && r.width > MIN_WIDTH_FOR_A_LP && r.y >= 0 && r.height >= MIN_HEIGHT_FOR_A_LP
			&& top_left.x < top_right.x && bottom_left.x < bottom_right.x && top_left.y < bottom_left.y && top_right.y < bottom_right.y) {
			bottom_right_.x = bottom_right.x;
			top_right_.x = top_right.x;
			top_left_.x = top_left.x;
			bottom_left_.x = bottom_left.x;
			bottom_right_.y = bottom_right.y;
			top_right_.y = top_right.y;
			top_left_.y = top_left.y;
			bottom_left_.y = bottom_left.y;
			rect_OpenLP = r;
			return is_valid(top_left_,
				top_right_,
				bottom_right_,
				bottom_left_);
			//return true;
		}
		else return false;
	}
	else return false;
}
bool trouve_la_plaque(const cv::Mat& frame, const cv::Rect& global_rect
	, cv::Point& p0, cv::Point& p1, cv::Point& p2, cv::Point& p3,
	cv::Point& top_left_,
	cv::Point& top_right_,
	cv::Point& bottom_right_,
	cv::Point& bottom_left_,
	cv::Rect& rect_OpenLP)
{
	//std::cout << "  starts two_stage_lpr" << std::endl;
	const int dist_min_bord = 3;
	const int hauteur_plaque_min = 9;
	float rapport_largeur_sur_hauteur_min(1.0f);
	int nTresh = 8;
	IplImage iplImage = cvIplImage(frame);
	IplImage* iptemp = &iplImage;
	//convertit une image vissdk 8 bits en une image ipl 8 bits
	if (iptemp) {
		/*
				// convert to binary
				cv::Mat thresh_img;
				int blockSize = 7;
				double C = 0.0;
				adaptiveThreshold(frame, thresh_img, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);
				*/
		const int mean_carac = 50;
		const int  mean_fond = 200;
		cv::Point p0_; cv::Point p1_; cv::Point p2_; cv::Point p3_; cv::Point top_left__;
		cv::Point top_right__;
		cv::Point bottom_right__;
		cv::Point bottom_left__; cv::Rect rect_OpenLP_;
		bool return_value = trouve_la_plaque(iptemp, global_rect, nTresh, mean_carac
			, mean_fond, p0_, p1_, p2_, p3_,
			top_left__,
			top_right__,
			bottom_right__,
			bottom_left__, rect_OpenLP_,
			dist_min_bord, hauteur_plaque_min,
			rapport_largeur_sur_hauteur_min);
		p0.x = p0_.x;
		p0.y = p0_.y;
		p1.x = p1_.x;
		p1.y = p1_.y;
		p2.x = p2_.x;
		p2.y = p2_.y;
		p3.x = p3_.x;
		p3.y = p3_.y;
		top_left_.x = top_left__.x;
		top_left_.y = top_left__.y;
		top_right_.x = top_right__.x;
		top_right_.y = top_right__.y;
		bottom_right_.x = bottom_right__.x;
		bottom_right_.y = bottom_right__.y;
		bottom_left_.x = bottom_left__.x;
		bottom_left_.y = bottom_left__.y;
		rect_OpenLP = rect_OpenLP_;
		//cvReleaseImage(&iptemp);
		return return_value;
	}
	return false;
}
cv::Rect fit_in(const cv::Rect& rect_im, const cv::Rect& roi) {
	cv::Rect fit_in_rect(roi);
	if (fit_in_rect.x < rect_im.x) {
		fit_in_rect.width -= rect_im.x - fit_in_rect.x;
		fit_in_rect.x = rect_im.x;
	}
	if (fit_in_rect.y < rect_im.y) {
		fit_in_rect.height -= rect_im.y - fit_in_rect.y;
		fit_in_rect.y = rect_im.y;
	}
	if (fit_in_rect.y + fit_in_rect.height > rect_im.y + rect_im.height) fit_in_rect.height = rect_im.y + rect_im.height - fit_in_rect.y;
	if (fit_in_rect.x + fit_in_rect.width > rect_im.x + rect_im.width) fit_in_rect.width = rect_im.x + rect_im.width - fit_in_rect.x;
#ifdef _DEBUG
	assert(fit_in_rect == get_inter(rect_im, roi));
#endif //_DEBUG
	return fit_in_rect;
}
cv::Rect get_greater_rect(const std::list<cv::Rect>& boxes)
{
	int area = -1;
	cv::Rect return_rect(0, 0, 0, 0);
	std::list<cv::Rect>::const_iterator it_boxes(boxes.begin());
	while (it_boxes != boxes.end()) {
		if (area < it_boxes->width * it_boxes->height) {
			area = it_boxes->width * it_boxes->height;
			return_rect = *it_boxes;
		}
		it_boxes++;
	}
	return return_rect;
}
//gets the greater rect of the boxes list and then gets the reunion of all the boxes 
//if these two rects are similar means that the reunion is not too much larger than the greater rect, and thus the iou is near 1, then we consider that the greater rect 
//is license plate bb
bool has_global_rect(const std::list<cv::Rect>& boxes, cv::Rect& greater_rect)
{
	greater_rect = cv::Rect(get_greater_rect(boxes));
	//cette fonction retourne le rect englobant la collection
	//gets the reunion of all the boxes
	cv::Rect global_rect(get_global_rect(boxes));
	float iou_ = iou(global_rect, greater_rect);
	return (iou_ > 0.9f);
}
std::list<cv::Rect> translate(const std::list<cv::Rect>& l, const int x_, const int y_) {
	std::list<cv::Rect> return_value;
	std::list<cv::Rect>::const_iterator it(l.begin());
	while (it != l.end()) {
		return_value.push_back(cv::Rect(it->x + x_, it->y + y_, it->width, it->height));
		it++;
	}
	return return_value;
}
void translate(cv::Point& pt, const int tx, const int ty)
{
	pt.x += tx;
	pt.y += ty;
}
cv::Rect translate(const cv::Rect& r, const int x_, const int y_) {
	return cv::Rect(r.x + x_, r.y + y_, r.width, r.height);
}
bool in_quadrilatere(const cv::Rect& box,
	const cv::Point& top_left, const cv::Point& top_right, const cv::Point& bottom_right, const cv::Point& bottom_left, const float& signed_distance)
{
	std::vector<cv::Point > contours;
	contours.push_back(top_left);
	contours.push_back(top_right);
	contours.push_back(bottom_right);
	contours.push_back(bottom_left);
	contours.push_back(top_left);
	return in_quadrilatere(box, contours, signed_distance);
}
bool in_quadrilatere(const std::list<cv::Rect>& boxes,
	const cv::Point& top_left, const cv::Point& top_right, const cv::Point& bottom_right, const cv::Point& bottom_left, const float& signed_distance)
{
	std::vector<cv::Point > contours;
	contours.push_back(top_left);
	contours.push_back(top_right);
	contours.push_back(bottom_right);
	contours.push_back(bottom_left);
	contours.push_back(top_left);
	return in_quadrilatere(boxes, contours, signed_distance);
}
bool in_quadrilatere(const std::vector<cv::Rect>& boxes, const std::vector<cv::Point >& contours, const float& signed_distance)
{
	std::list<cv::Rect> boxes_;
	std::copy(boxes.begin(), boxes.end(), std::back_inserter(boxes_));
	return in_quadrilatere(boxes_, contours, signed_distance);
}
bool in_quadrilatere(const std::vector<cv::Rect>& boxes,
	const cv::Point& top_left, const cv::Point& top_right, const cv::Point& bottom_right, const cv::Point& bottom_left, const float& signed_distance)
{
	std::list<cv::Rect> boxes_;
	std::copy(boxes.begin(), boxes.end(), std::back_inserter(boxes_));
	return in_quadrilatere(boxes_, top_left, top_right, bottom_right, bottom_left, signed_distance);
}
bool trouve_la_plaque(const cv::Mat& frame,
	const std::list<int>& classes, const std::list<cv::Rect>& boxes
	,
	cv::Point& top_left_,
	cv::Point& top_right_,
	cv::Point& bottom_right_,
	cv::Point& bottom_left_, cv::Rect& rect_OpenLP)
{
	cv::Mat frame_grayscale(frame.size(), CV_8U);
	if (frame.type() == CV_8UC3)
		cv::cvtColor(frame, frame_grayscale, cv::COLOR_BGR2GRAY);
	else if (frame.type() == CV_8UC4)
		cv::cvtColor(frame, frame_grayscale, cv::COLOR_BGRA2GRAY);
	else if (frame.type() == CV_8UC1)
		frame_grayscale = frame.clone();
	cv::Scalar mean_ = cv::mean(frame_grayscale);
	if (frame_grayscale.rows > 0 && frame_grayscale.cols > 0 && mean_[0] > 2.0f && mean_[0] < 250.0f) {
		//now print corners
		cv::Point  p0; cv::Point  p1; cv::Point  p2; cv::Point  p3;
		std::list<cv::Rect> l_boxes;
		std::copy(boxes.begin(), boxes.end(), std::back_inserter(l_boxes));
		std::list<int> l_classIds;
		std::copy(classes.begin(), classes.end(), std::back_inserter(l_classIds));
		sort_from_left_to_right(l_boxes, l_classIds);//sorts all the boxes from left to right
		//filter out lpn box
		//***************************************************
		//                  FILTER
		//***************************************************
		filter_out_everything_but_characters(l_boxes, l_classIds);
		//cette fonction retourne le rect englobant la collection
		//gets the reunion of all the l_boxes
		cv::Rect roi(get_global_rect(l_boxes));
		if (roi.width && roi.height) {
			roi.x -= roi.width / 2;
			roi.width += int((float)roi.width);
			roi.y -= roi.height;
			roi.height += 2 * roi.height;
			const cv::Rect rect_im(0, 0, frame_grayscale.cols, frame_grayscale.rows);
			roi = fit_in(rect_im, roi);
#ifdef _DEBUG		
			assert(roi.width && roi.height);
#endif //_DEBUG
			cv::Mat subimage = frame_grayscale(roi);
			cv::Rect greater_rect;
			const int im_width = frame.cols;
			const int im_height = frame.rows;
			const int classe_lpn_rect = NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE;
			if (!has_global_rect(l_boxes, greater_rect) && im_width > 0 && im_height > 0 && classe_lpn_rect >= 0 && l_boxes.size() > 1)
			{
				//now print corners
				cv::Rect global_rect_in_subimage(get_global_rect(translate(l_boxes, -roi.x, -roi.y)));
				if (trouve_la_plaque(subimage, global_rect_in_subimage
					, p0, p1, p2, p3, top_left_, top_right_, bottom_right_, bottom_left_, rect_OpenLP)) {
					translate(top_left_, roi.x, roi.y);
					translate(top_right_, roi.x, roi.y);
					translate(bottom_right_, roi.x, roi.y);
					translate(bottom_left_, roi.x, roi.y);
					rect_OpenLP = translate(rect_OpenLP, roi.x, roi.y);
					if (is_in_rect(top_left_, rect_im) && is_in_rect(top_right_, rect_im) && is_in_rect(bottom_right_, rect_im) && is_in_rect(bottom_left_, rect_im)
						) {
						cv::Rect global_rect = get_global_rect(l_boxes);
						if (in_quadrilatere(l_boxes, top_left_, top_right_, bottom_right_, bottom_left_, (float)(-get_median_height(l_boxes) / 4))) {
							global_rect = global_rect | (::get_global_rect(bottom_right_, bottom_left_, top_right_, top_left_));//rect = rect1 | rect2 (minimum area rectangle containing rect2 and rect3 )
							return true;
						}
						else return false;
					}
					else return false;
				}
				else return false;
			}
			else return false;
		}
		else return false;
	}
	else return false;
}
bool in_quadrilatere(const cv::Point& pt,
	const cv::Point& top_left, const cv::Point& top_right, const cv::Point& bottom_right, const cv::Point& bottom_left, const float& signed_distance)
{
	std::vector<cv::Point > contours;
	contours.push_back(top_left);
	contours.push_back(top_right);
	contours.push_back(bottom_right);
	contours.push_back(bottom_left);
	contours.push_back(top_left);
	return in_quadrilatere(pt, contours, signed_distance);
}
bool in_quadrilatere(const cv::Point& pt, const std::vector<cv::Point >& contours, const float& signed_distance)
{
	const cv::Point2f point(pt.x, pt.y);
	const double result = cv::pointPolygonTest(contours, point, true);
	return (result >= signed_distance);
}
bool in_quadrilatere(const cv::Rect& box, const std::vector<cv::Point >& contours, const float& signed_distance)
{
	return (in_quadrilatere(cv::Point(box.tl()), contours, signed_distance) && in_quadrilatere(cv::Point(box.x + box.width, box.y), contours, signed_distance)
		&& in_quadrilatere(cv::Point(box.x + box.width, box.y + box.height), contours, signed_distance) && in_quadrilatere(cv::Point(box.x, box.y + box.height), contours, signed_distance)
		);
}
bool in_quadrilatere(const std::list<cv::Rect>& boxes, const std::vector<cv::Point >& contours, const float& signed_distance)
{
	std::list<cv::Rect>::const_iterator it(boxes.begin());
	while (it != boxes.end()) {
		if (!in_quadrilatere(*it, contours, signed_distance))
			return false;
		else it++;
	}
	return true;
}
bool quadrilatere_is_convex(const CvPoint& pt0, const CvPoint& pt1, const CvPoint& pt2, const CvPoint& pt3)
{
	std::vector<cv::Point > contours;
	contours.push_back(pt0);
	contours.push_back(pt1);
	contours.push_back(pt2);
	contours.push_back(pt3);
	contours.push_back(pt0);
	/*
	std::vector<cv::Point> hull;
	cv::convexHull(contours, hull);
	*/
	std::vector<cv::Vec4i> cx;
	std::vector<int> hull_idx;
	cv::convexHull(contours, hull_idx, false, false);
	convexityDefects(contours, hull_idx, cx);
	return (cx.size() == 0);
}
bool quadrilatere_is_convex(const cv::Point& pt0, const cv::Point& pt1, const cv::Point& pt2, const cv::Point& pt3)
{
	CvPoint pt0_;  pt0_.x = pt0.x; pt0_.y = pt0.y;
	CvPoint pt1_;  pt1_.x = pt1.x; pt1_.y = pt1.y;
	CvPoint pt2_;  pt2_.x = pt2.x; pt2_.y = pt2.y;
	CvPoint pt3_;  pt3_.x = pt3.x; pt3_.y = pt3.y;
	return quadrilatere_is_convex(pt0_, pt1_, pt2_, pt3_);
}
//must be convex quadrilatere
bool width_is_larger(const CvPoint& pt0, const CvPoint& pt1, const CvPoint& pt2, const CvPoint& pt3)
{
	if (quadrilatere_is_convex(pt0, pt1, pt2, pt3)) {
		std::vector<cv::Point > contours;
		contours.push_back(pt0);
		contours.push_back(pt1);
		contours.push_back(pt2);
		contours.push_back(pt3);
		cv::RotatedRect rrect = cv::minAreaRect(contours);//! returns width and height of the rectangle
				//Size2f size;
		//returns 4 vertices of the rectangle
		cv::Point2f pts[4];
		rrect.points(pts);
		float dx = (pts[2].x - pts[1].x) * (pts[2].x - pts[1].x) + (pts[2].y - pts[1].y) * (pts[2].y - pts[1].y);
		float dy = (pts[0].x - pts[1].x) * (pts[0].x - pts[1].x) + (pts[0].y - pts[1].y) * (pts[0].y - pts[1].y);
		return (dx > dy);
	}
	else return false;
}
//must be convex quadrilatere
bool width_is_larger(const cv::Point& pt0, const cv::Point& pt1, const cv::Point& pt2, const cv::Point& pt3)
{
	CvPoint pt0_;  pt0_.x = pt0.x; pt0_.y = pt0.y;
	CvPoint pt1_;  pt1_.x = pt1.x; pt1_.y = pt1.y;
	CvPoint pt2_;  pt2_.x = pt2.x; pt2_.y = pt2.y;
	CvPoint pt3_;  pt3_.x = pt3.x; pt3_.y = pt3.y;
	return width_is_larger(pt0_, pt1_, pt2_, pt3_);
}
//must be convex quadrilatere
bool get_corners(const CvPoint& pt0, const CvPoint& pt1, const CvPoint& pt2, const CvPoint& pt3,
	cv::Point& top_left, cv::Point& top_right, cv::Point& bottom_right, cv::Point& bottom_left)
{
	if (quadrilatere_is_convex(pt0, pt1, pt2, pt3) && width_is_larger(pt0, pt1, pt2, pt3)) {
		std::vector<cv::Point > contours;
		contours.push_back(pt0);
		contours.push_back(pt1);
		contours.push_back(pt2);
		contours.push_back(pt3);
		cv::RotatedRect rrect = cv::minAreaRect(contours);//! returns width and height of the rectangle
		//returns 4 vertices of the rectangle
		cv::Point2f pts[4];
		rrect.points(pts);
		cv::Point pts_quadrilatere[4];
		pts_quadrilatere[0] = pt0;
		pts_quadrilatere[1] = pt1;
		pts_quadrilatere[2] = pt2;
		pts_quadrilatere[3] = pt3;
		//*****************************
				//TOP_LEFT
				//*****************************
		float d0 = (pts[1].x - pts_quadrilatere[0].x) * (pts[1].x - pts_quadrilatere[0].x) + (pts[1].y - pts_quadrilatere[0].y) * (pts[1].y - pts_quadrilatere[0].y);
		float d1 = (pts[1].x - pts_quadrilatere[1].x) * (pts[1].x - pts_quadrilatere[1].x) + (pts[1].y - pts_quadrilatere[1].y) * (pts[1].y - pts_quadrilatere[1].y);
		float d2 = (pts[1].x - pts_quadrilatere[2].x) * (pts[1].x - pts_quadrilatere[2].x) + (pts[1].y - pts_quadrilatere[2].y) * (pts[1].y - pts_quadrilatere[2].y);
		float d3 = (pts[1].x - pts_quadrilatere[3].x) * (pts[1].x - pts_quadrilatere[3].x) + (pts[1].y - pts_quadrilatere[3].y) * (pts[1].y - pts_quadrilatere[3].y);
		std::list<float> dists; std::list<int> indeces;
		indeces.push_back(0); indeces.push_back(1); indeces.push_back(2); indeces.push_back(3);
		dists.push_back(d0); dists.push_back(d1); dists.push_back(d2); dists.push_back(d3);
		//////////////////////////////////////////////////////////////////
		//TRI DE LA LISTE
		//////////////////////////////////////////////////////////////////
		//des  pour trier (vides au debut puis se remplissant un par un, progressivement)
		std::list<float> dists_tri;
		std::list<int> indeces_tri;
		while (!dists.empty() && !indeces.empty()) {
			float dist(dists.front());
			std::list<int>::iterator it_indeces_tri(indeces_tri.begin());
			std::list<float>::iterator it_dists_tri(dists_tri.begin());
			while (it_dists_tri != dists_tri.end()
				&& it_indeces_tri != indeces_tri.end()) {
				if (dist >= *it_dists_tri) { break; }
				else { it_dists_tri++; it_indeces_tri++; }
			}
			indeces_tri.splice(it_indeces_tri
				, indeces, indeces.begin());
			dists_tri.splice(it_dists_tri, dists, dists.begin());
		}
		//VERIF DE LA LISTE
#ifdef _DEBUG
		assert(indeces.empty());
		assert(dists.empty());
		assert(indeces_tri.size() == dists_tri.size());
		std::list<float>::iterator it_dists_verif(dists_tri.begin());
		float poids_ = FLT_MAX;
		while (it_dists_verif != dists_tri.end()) {
#ifdef _DEBUG
			assert(*it_dists_verif < poids_ + FLT_EPSILON);
#endif //_DEBUG
			poids_ = *it_dists_verif;
			it_dists_verif++;
		}
#endif //_DEBUG
		indeces.swap(indeces_tri);
		dists.swap(dists_tri);
		top_left = pts_quadrilatere[indeces.back()];
		indeces.clear();
		dists.clear();
		//*****************************
				//BOTTOM_RIGHT
				//*****************************
		d0 = (pts[3].x - pts_quadrilatere[0].x) * (pts[3].x - pts_quadrilatere[0].x) + (pts[3].y - pts_quadrilatere[0].y) * (pts[3].y - pts_quadrilatere[0].y);
		d1 = (pts[3].x - pts_quadrilatere[1].x) * (pts[3].x - pts_quadrilatere[1].x) + (pts[3].y - pts_quadrilatere[1].y) * (pts[3].y - pts_quadrilatere[1].y);
		d2 = (pts[3].x - pts_quadrilatere[2].x) * (pts[3].x - pts_quadrilatere[2].x) + (pts[3].y - pts_quadrilatere[2].y) * (pts[3].y - pts_quadrilatere[2].y);
		d3 = (pts[3].x - pts_quadrilatere[3].x) * (pts[3].x - pts_quadrilatere[3].x) + (pts[3].y - pts_quadrilatere[3].y) * (pts[3].y - pts_quadrilatere[3].y);
		indeces.push_back(0); indeces.push_back(1); indeces.push_back(2); indeces.push_back(3);
		dists.push_back(d0); dists.push_back(d1); dists.push_back(d2); dists.push_back(d3);
		//////////////////////////////////////////////////////////////////
		//TRI DE LA LISTE
		//////////////////////////////////////////////////////////////////
		//des  pour trier (vides au debut puis se remplissant un par un, progressivement)
		while (!dists.empty() && !indeces.empty()) {
			float dist(dists.front());
			std::list<int>::iterator it_indeces_tri(indeces_tri.begin());
			std::list<float>::iterator it_dists_tri(dists_tri.begin());
			while (it_dists_tri != dists_tri.end()
				&& it_indeces_tri != indeces_tri.end()) {
				if (dist >= *it_dists_tri) { break; }
				else { it_dists_tri++; it_indeces_tri++; }
			}
			indeces_tri.splice(it_indeces_tri
				, indeces, indeces.begin());
			dists_tri.splice(it_dists_tri, dists, dists.begin());
		}
		//VERIF DE LA LISTE
#ifdef _DEBUG
		assert(indeces.empty());
		assert(dists.empty());
		assert(indeces_tri.size() == dists_tri.size());
		it_dists_verif = (dists_tri.begin());
		poids_ = FLT_MAX;
		while (it_dists_verif != dists_tri.end()) {
#ifdef _DEBUG
			assert(*it_dists_verif < poids_ + FLT_EPSILON);
#endif //_DEBUG
			poids_ = *it_dists_verif;
			it_dists_verif++;
		}
#endif //_DEBUG
		indeces.swap(indeces_tri);
		dists.swap(dists_tri);
		bottom_right = pts_quadrilatere[indeces.back()];
		indeces.clear();
		dists.clear();
		//*****************************
				//TOP_RIGHT
				//*****************************
		d0 = (pts[2].x - pts_quadrilatere[0].x) * (pts[2].x - pts_quadrilatere[0].x) + (pts[2].y - pts_quadrilatere[0].y) * (pts[2].y - pts_quadrilatere[0].y);
		d1 = (pts[2].x - pts_quadrilatere[1].x) * (pts[2].x - pts_quadrilatere[1].x) + (pts[2].y - pts_quadrilatere[1].y) * (pts[2].y - pts_quadrilatere[1].y);
		d2 = (pts[2].x - pts_quadrilatere[2].x) * (pts[2].x - pts_quadrilatere[2].x) + (pts[2].y - pts_quadrilatere[2].y) * (pts[2].y - pts_quadrilatere[2].y);
		d3 = (pts[2].x - pts_quadrilatere[3].x) * (pts[2].x - pts_quadrilatere[3].x) + (pts[2].y - pts_quadrilatere[3].y) * (pts[2].y - pts_quadrilatere[3].y);
		indeces.push_back(0); indeces.push_back(1); indeces.push_back(2); indeces.push_back(3);
		dists.push_back(d0); dists.push_back(d1); dists.push_back(d2); dists.push_back(d3);
		//////////////////////////////////////////////////////////////////
		//TRI DE LA LISTE
		//////////////////////////////////////////////////////////////////
		//des  pour trier (vides au debut puis se remplissant un par un, progressivement)
		while (!dists.empty() && !indeces.empty()) {
			float dist(dists.front());
			std::list<int>::iterator it_indeces_tri(indeces_tri.begin());
			std::list<float>::iterator it_dists_tri(dists_tri.begin());
			while (it_dists_tri != dists_tri.end()
				&& it_indeces_tri != indeces_tri.end()) {
				if (dist >= *it_dists_tri) { break; }
				else { it_dists_tri++; it_indeces_tri++; }
			}
			indeces_tri.splice(it_indeces_tri
				, indeces, indeces.begin());
			dists_tri.splice(it_dists_tri, dists, dists.begin());
		}
		//VERIF DE LA LISTE
#ifdef _DEBUG
		assert(indeces.empty());
		assert(dists.empty());
		assert(indeces_tri.size() == dists_tri.size());
		it_dists_verif = (dists_tri.begin());
		poids_ = FLT_MAX;
		while (it_dists_verif != dists_tri.end()) {
#ifdef _DEBUG
			assert(*it_dists_verif < poids_ + FLT_EPSILON);
#endif //_DEBUG
			poids_ = *it_dists_verif;
			it_dists_verif++;
		}
#endif //_DEBUG
		indeces.swap(indeces_tri);
		dists.swap(dists_tri);
		top_right = pts_quadrilatere[indeces.back()];
		indeces.clear();
		dists.clear();
		//*****************************
				//BOTTOM_LEFT
				//*****************************
		d0 = (pts[0].x - pts_quadrilatere[0].x) * (pts[0].x - pts_quadrilatere[0].x) + (pts[0].y - pts_quadrilatere[0].y) * (pts[0].y - pts_quadrilatere[0].y);
		d1 = (pts[0].x - pts_quadrilatere[1].x) * (pts[0].x - pts_quadrilatere[1].x) + (pts[0].y - pts_quadrilatere[1].y) * (pts[0].y - pts_quadrilatere[1].y);
		d2 = (pts[0].x - pts_quadrilatere[2].x) * (pts[0].x - pts_quadrilatere[2].x) + (pts[0].y - pts_quadrilatere[2].y) * (pts[0].y - pts_quadrilatere[2].y);
		d3 = (pts[0].x - pts_quadrilatere[3].x) * (pts[0].x - pts_quadrilatere[3].x) + (pts[0].y - pts_quadrilatere[3].y) * (pts[0].y - pts_quadrilatere[3].y);
		indeces.push_back(0); indeces.push_back(1); indeces.push_back(2); indeces.push_back(3);
		dists.push_back(d0); dists.push_back(d1); dists.push_back(d2); dists.push_back(d3);
		//////////////////////////////////////////////////////////////////
		//TRI DE LA LISTE
		//////////////////////////////////////////////////////////////////
		//des  pour trier (vides au debut puis se remplissant un par un, progressivement)
		while (!dists.empty() && !indeces.empty()) {
			float dist(dists.front());
			std::list<int>::iterator it_indeces_tri(indeces_tri.begin());
			std::list<float>::iterator it_dists_tri(dists_tri.begin());
			while (it_dists_tri != dists_tri.end()
				&& it_indeces_tri != indeces_tri.end()) {
				if (dist >= *it_dists_tri) { break; }
				else { it_dists_tri++; it_indeces_tri++; }
			}
			indeces_tri.splice(it_indeces_tri
				, indeces, indeces.begin());
			dists_tri.splice(it_dists_tri, dists, dists.begin());
		}
		//VERIF DE LA LISTE
#ifdef _DEBUG
		assert(indeces.empty());
		assert(dists.empty());
		assert(indeces_tri.size() == dists_tri.size());
		it_dists_verif = (dists_tri.begin());
		poids_ = FLT_MAX;
		while (it_dists_verif != dists_tri.end()) {
#ifdef _DEBUG
			assert(*it_dists_verif < poids_ + FLT_EPSILON);
#endif //_DEBUG
			poids_ = *it_dists_verif;
			it_dists_verif++;
		}
#endif //_DEBUG
		indeces.swap(indeces_tri);
		dists.swap(dists_tri);
		bottom_left = pts_quadrilatere[indeces.back()];
		indeces.clear();
		dists.clear();
		return (bottom_left != bottom_right && bottom_left != top_right && bottom_left != top_left && top_left != bottom_right && top_left != top_right && top_right != bottom_right);
	}
	else {
		bottom_right = bottom_left = top_right = top_left = cv::Point(SHRT_MIN, SHRT_MIN);
		return false;
	}
}
//must be convex quadrilatere
bool get_corners(const IplImage* im, const CvPoint& pt0, const CvPoint& pt1, const CvPoint& pt2, const CvPoint& pt3,
	cv::Point& top_left, cv::Point& top_right, cv::Point& bottom_right, cv::Point& bottom_left)
{
	if (quadrilatere_is_convex(pt0, pt1, pt2, pt3) && width_is_larger(pt0, pt1, pt2, pt3)) {
#ifdef _DEBUG
		/*
		cv::Mat frame = cv::cvarrToMat(im);
		cv::line(frame, pt0, pt1, cv::Scalar(0, 255, 0), 1);
		cv::line(frame, pt1, pt2, cv::Scalar(0, 255, 0), 1);
		cv::line(frame, pt2, pt3, cv::Scalar(0, 255, 0), 1);
		cv::line(frame, pt3, pt0, cv::Scalar(0, 255, 0), 1);
		imshow("sqaure", frame);
		char c = cv::waitKey(0);
		*/
#endif //_DEBUG
		std::vector<cv::Point > contours;
		contours.push_back(pt0);
		contours.push_back(pt1);
		contours.push_back(pt2);
		contours.push_back(pt3);
		cv::RotatedRect rrect = cv::minAreaRect(contours);//! returns width and height of the rectangle
		//returns 4 vertices of the rectangle
		cv::Point2f pts[4];
		rrect.points(pts);
		cv::Point pts_quadrilatere[4];
		pts_quadrilatere[0] = pt0;
		pts_quadrilatere[1] = pt1;
		pts_quadrilatere[2] = pt2;
		pts_quadrilatere[3] = pt3;
		//*****************************
				//TOP_LEFT
				//*****************************
		float d0 = (pts[1].x - pts_quadrilatere[0].x) * (pts[1].x - pts_quadrilatere[0].x) + (pts[1].y - pts_quadrilatere[0].y) * (pts[1].y - pts_quadrilatere[0].y);
		float d1 = (pts[1].x - pts_quadrilatere[1].x) * (pts[1].x - pts_quadrilatere[1].x) + (pts[1].y - pts_quadrilatere[1].y) * (pts[1].y - pts_quadrilatere[1].y);
		float d2 = (pts[1].x - pts_quadrilatere[2].x) * (pts[1].x - pts_quadrilatere[2].x) + (pts[1].y - pts_quadrilatere[2].y) * (pts[1].y - pts_quadrilatere[2].y);
		float d3 = (pts[1].x - pts_quadrilatere[3].x) * (pts[1].x - pts_quadrilatere[3].x) + (pts[1].y - pts_quadrilatere[3].y) * (pts[1].y - pts_quadrilatere[3].y);
		std::list<float> dists; std::list<int> indeces;
		indeces.push_back(0); indeces.push_back(1); indeces.push_back(2); indeces.push_back(3);
		dists.push_back(d0); dists.push_back(d1); dists.push_back(d2); dists.push_back(d3);
		//////////////////////////////////////////////////////////////////
		//TRI DE LA LISTE
		//////////////////////////////////////////////////////////////////
		//des  pour trier (vides au debut puis se remplissant un par un, progressivement)
		std::list<float> dists_tri;
		std::list<int> indeces_tri;
		while (!dists.empty() && !indeces.empty()) {
			float dist(dists.front());
			std::list<int>::iterator it_indeces_tri(indeces_tri.begin());
			std::list<float>::iterator it_dists_tri(dists_tri.begin());
			while (it_dists_tri != dists_tri.end()
				&& it_indeces_tri != indeces_tri.end()) {
				if (dist >= *it_dists_tri) { break; }
				else { it_dists_tri++; it_indeces_tri++; }
			}
			indeces_tri.splice(it_indeces_tri
				, indeces, indeces.begin());
			dists_tri.splice(it_dists_tri, dists, dists.begin());
		}
		//VERIF DE LA LISTE
#ifdef _DEBUG
		assert(indeces.empty());
		assert(dists.empty());
		assert(indeces_tri.size() == dists_tri.size());
		std::list<float>::iterator it_dists_verif(dists_tri.begin());
		float poids_ = FLT_MAX;
		while (it_dists_verif != dists_tri.end()) {
#ifdef _DEBUG
			assert(*it_dists_verif < poids_ + FLT_EPSILON);
#endif //_DEBUG
			poids_ = *it_dists_verif;
			it_dists_verif++;
		}
#endif //_DEBUG
		indeces.swap(indeces_tri);
		dists.swap(dists_tri);
		top_left = pts_quadrilatere[indeces.back()];
		indeces.clear();
		dists.clear();
		//*****************************
				//BOTTOM_RIGHT
				//*****************************
		d0 = (pts[3].x - pts_quadrilatere[0].x) * (pts[3].x - pts_quadrilatere[0].x) + (pts[3].y - pts_quadrilatere[0].y) * (pts[3].y - pts_quadrilatere[0].y);
		d1 = (pts[3].x - pts_quadrilatere[1].x) * (pts[3].x - pts_quadrilatere[1].x) + (pts[3].y - pts_quadrilatere[1].y) * (pts[3].y - pts_quadrilatere[1].y);
		d2 = (pts[3].x - pts_quadrilatere[2].x) * (pts[3].x - pts_quadrilatere[2].x) + (pts[3].y - pts_quadrilatere[2].y) * (pts[3].y - pts_quadrilatere[2].y);
		d3 = (pts[3].x - pts_quadrilatere[3].x) * (pts[3].x - pts_quadrilatere[3].x) + (pts[3].y - pts_quadrilatere[3].y) * (pts[3].y - pts_quadrilatere[3].y);
		indeces.push_back(0); indeces.push_back(1); indeces.push_back(2); indeces.push_back(3);
		dists.push_back(d0); dists.push_back(d1); dists.push_back(d2); dists.push_back(d3);
		//////////////////////////////////////////////////////////////////
		//TRI DE LA LISTE
		//////////////////////////////////////////////////////////////////
		//des  pour trier (vides au debut puis se remplissant un par un, progressivement)
		while (!dists.empty() && !indeces.empty()) {
			float dist(dists.front());
			std::list<int>::iterator it_indeces_tri(indeces_tri.begin());
			std::list<float>::iterator it_dists_tri(dists_tri.begin());
			while (it_dists_tri != dists_tri.end()
				&& it_indeces_tri != indeces_tri.end()) {
				if (dist >= *it_dists_tri) { break; }
				else { it_dists_tri++; it_indeces_tri++; }
			}
			indeces_tri.splice(it_indeces_tri
				, indeces, indeces.begin());
			dists_tri.splice(it_dists_tri, dists, dists.begin());
		}
		//VERIF DE LA LISTE
#ifdef _DEBUG
		assert(indeces.empty());
		assert(dists.empty());
		assert(indeces_tri.size() == dists_tri.size());
		it_dists_verif = (dists_tri.begin());
		poids_ = FLT_MAX;
		while (it_dists_verif != dists_tri.end()) {
#ifdef _DEBUG
			assert(*it_dists_verif < poids_ + FLT_EPSILON);
#endif //_DEBUG
			poids_ = *it_dists_verif;
			it_dists_verif++;
		}
#endif //_DEBUG
		indeces.swap(indeces_tri);
		dists.swap(dists_tri);
		bottom_right = pts_quadrilatere[indeces.back()];
		indeces.clear();
		dists.clear();
		//*****************************
				//TOP_RIGHT
				//*****************************
		d0 = (pts[2].x - pts_quadrilatere[0].x) * (pts[2].x - pts_quadrilatere[0].x) + (pts[2].y - pts_quadrilatere[0].y) * (pts[2].y - pts_quadrilatere[0].y);
		d1 = (pts[2].x - pts_quadrilatere[1].x) * (pts[2].x - pts_quadrilatere[1].x) + (pts[2].y - pts_quadrilatere[1].y) * (pts[2].y - pts_quadrilatere[1].y);
		d2 = (pts[2].x - pts_quadrilatere[2].x) * (pts[2].x - pts_quadrilatere[2].x) + (pts[2].y - pts_quadrilatere[2].y) * (pts[2].y - pts_quadrilatere[2].y);
		d3 = (pts[2].x - pts_quadrilatere[3].x) * (pts[2].x - pts_quadrilatere[3].x) + (pts[2].y - pts_quadrilatere[3].y) * (pts[2].y - pts_quadrilatere[3].y);
		indeces.push_back(0); indeces.push_back(1); indeces.push_back(2); indeces.push_back(3);
		dists.push_back(d0); dists.push_back(d1); dists.push_back(d2); dists.push_back(d3);
		//////////////////////////////////////////////////////////////////
		//TRI DE LA LISTE
		//////////////////////////////////////////////////////////////////
		//des  pour trier (vides au debut puis se remplissant un par un, progressivement)
		while (!dists.empty() && !indeces.empty()) {
			float dist(dists.front());
			std::list<int>::iterator it_indeces_tri(indeces_tri.begin());
			std::list<float>::iterator it_dists_tri(dists_tri.begin());
			while (it_dists_tri != dists_tri.end()
				&& it_indeces_tri != indeces_tri.end()) {
				if (dist >= *it_dists_tri) { break; }
				else { it_dists_tri++; it_indeces_tri++; }
			}
			indeces_tri.splice(it_indeces_tri
				, indeces, indeces.begin());
			dists_tri.splice(it_dists_tri, dists, dists.begin());
		}
		//VERIF DE LA LISTE
#ifdef _DEBUG
		assert(indeces.empty());
		assert(dists.empty());
		assert(indeces_tri.size() == dists_tri.size());
		it_dists_verif = (dists_tri.begin());
		poids_ = FLT_MAX;
		while (it_dists_verif != dists_tri.end()) {
#ifdef _DEBUG
			assert(*it_dists_verif < poids_ + FLT_EPSILON);
#endif //_DEBUG
			poids_ = *it_dists_verif;
			it_dists_verif++;
		}
#endif //_DEBUG
		indeces.swap(indeces_tri);
		dists.swap(dists_tri);
		top_right = pts_quadrilatere[indeces.back()];
		indeces.clear();
		dists.clear();
		//*****************************
				//BOTTOM_LEFT
				//*****************************
		d0 = (pts[0].x - pts_quadrilatere[0].x) * (pts[0].x - pts_quadrilatere[0].x) + (pts[0].y - pts_quadrilatere[0].y) * (pts[0].y - pts_quadrilatere[0].y);
		d1 = (pts[0].x - pts_quadrilatere[1].x) * (pts[0].x - pts_quadrilatere[1].x) + (pts[0].y - pts_quadrilatere[1].y) * (pts[0].y - pts_quadrilatere[1].y);
		d2 = (pts[0].x - pts_quadrilatere[2].x) * (pts[0].x - pts_quadrilatere[2].x) + (pts[0].y - pts_quadrilatere[2].y) * (pts[0].y - pts_quadrilatere[2].y);
		d3 = (pts[0].x - pts_quadrilatere[3].x) * (pts[0].x - pts_quadrilatere[3].x) + (pts[0].y - pts_quadrilatere[3].y) * (pts[0].y - pts_quadrilatere[3].y);
		indeces.push_back(0); indeces.push_back(1); indeces.push_back(2); indeces.push_back(3);
		dists.push_back(d0); dists.push_back(d1); dists.push_back(d2); dists.push_back(d3);
		//////////////////////////////////////////////////////////////////
		//TRI DE LA LISTE
		//////////////////////////////////////////////////////////////////
		//des  pour trier (vides au debut puis se remplissant un par un, progressivement)
		while (!dists.empty() && !indeces.empty()) {
			float dist(dists.front());
			std::list<int>::iterator it_indeces_tri(indeces_tri.begin());
			std::list<float>::iterator it_dists_tri(dists_tri.begin());
			while (it_dists_tri != dists_tri.end()
				&& it_indeces_tri != indeces_tri.end()) {
				if (dist >= *it_dists_tri) { break; }
				else { it_dists_tri++; it_indeces_tri++; }
			}
			indeces_tri.splice(it_indeces_tri
				, indeces, indeces.begin());
			dists_tri.splice(it_dists_tri, dists, dists.begin());
		}
		//VERIF DE LA LISTE
#ifdef _DEBUG
		assert(indeces.empty());
		assert(dists.empty());
		assert(indeces_tri.size() == dists_tri.size());
		it_dists_verif = (dists_tri.begin());
		poids_ = FLT_MAX;
		while (it_dists_verif != dists_tri.end()) {
#ifdef _DEBUG
			assert(*it_dists_verif < poids_ + FLT_EPSILON);
#endif //_DEBUG
			poids_ = *it_dists_verif;
			it_dists_verif++;
		}
#endif //_DEBUG
		indeces.swap(indeces_tri);
		dists.swap(dists_tri);
		bottom_left = pts_quadrilatere[indeces.back()];
		indeces.clear();
		dists.clear();
		return (bottom_left != bottom_right && bottom_left != top_right && bottom_left != top_left && top_left != bottom_right && top_left != top_right && top_right != bottom_right);
	}
	else {
		bottom_right = bottom_left = top_right = top_left = cv::Point(SHRT_MIN, SHRT_MIN);
		return false;
	}
}
//must be convex quadrilatere
bool get_corners(const cv::Point& pt0, const cv::Point& pt1, const cv::Point& pt2, const cv::Point& pt3,
	cv::Point& top_left, cv::Point& top_right, cv::Point& bottom_right, cv::Point& bottom_left)
{
	CvPoint pt0_;  pt0_.x = pt0.x; pt0_.y = pt0.y;
	CvPoint pt1_;  pt1_.x = pt1.x; pt1_.y = pt1.y;
	CvPoint pt2_;  pt2_.x = pt2.x; pt2_.y = pt2.y;
	CvPoint pt3_;  pt3_.x = pt3.x; pt3_.y = pt3.y;
	return get_corners(pt0_, pt1_, pt2_, pt3_,
		top_left, top_right, bottom_right, bottom_left);
}
std::string get_plate_type(
	const std::list<cv::Rect>& vect_of_detected_boxes,
	const std::list<int>& classIds, const int number_of_characters_latin_numberplate
)
{
	//first from left to right
	//cette fonction trie la liste de gauche a droite 
	//change from vect to list 
	std::list<cv::Rect> boxes;
	std::copy(vect_of_detected_boxes.begin(), vect_of_detected_boxes.end(), std::back_inserter(boxes));
	std::list<int> l_classIds;
	std::copy(classIds.begin(), classIds.end(), std::back_inserter(l_classIds));
	sort_from_left_to_right(boxes, l_classIds);
	//filter out lpn box
	//***************************************************
	//                  FILTER
	//***************************************************
	filter_out_everything_but_characters(boxes,
		l_classIds);
	//filter out adjacent boxes with iou>nmsThreshold
		//***************************************************
		//                  FILTER
		//***************************************************
//if two boxes have an iou (intersection over union) that is too large, then they cannot represent two adjacent characters of the license plate 
//so we discard the one with the lowest confidence rate
	//filter_iou2(boxes, l_classIds, nmsThreshold);
	//std::list<cv::Rect> l_tri_left;
	std::list<int> levels;
	std::list<char> lpn;
	std::list<cv::Rect> l_tri_left;//list of characters boxes ranged from left to right
	std::list<int> l_tri_left_classIds;
	is_bi_level_plate(boxes,
		l_classIds, l_tri_left,
		l_tri_left_classIds, levels);
	//now
	std::list<char> lpn_minus_1;
	std::list<char> lpn_0;
	std::list<char> lpn_plus_1;
	/*
	C_OCROutputs availableAlpha;
	if (number_of_characters_latin_numberplate == 36)availableAlpha = C_OCROutputs(LATIN_LETTERS_LATIN_DIGITS);
	else if (number_of_characters_latin_numberplate == 35)availableAlpha = C_OCROutputs(LATIN_LETTERS_NO_O_LATIN_DIGITS);
	else availableAlpha = C_OCROutputs(LATIN_LETTERS_NO_I_O_LATIN_DIGITS);
	*/
	std::list<int>::const_iterator it_out_classes(l_tri_left_classIds.begin());
	//std::list<int>::const_iterator it_out_classes(l_classIds.begin());
	std::list<int>::const_iterator it_levels(levels.begin());
	while (
		//it_out_classes != l_classIds.end() 
		it_out_classes != l_tri_left_classIds.end()
		&& it_levels != levels.end()) {
		if (*it_out_classes < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE) {
			if (*it_levels == -1)
				lpn_minus_1.push_back(get_char(*it_out_classes));
			else {
				if (*it_levels == 0)
					lpn_0.push_back(get_char(*it_out_classes));
				else if (*it_levels == 1)
					lpn_plus_1.push_back(get_char(*it_out_classes));
			}
		}
		it_out_classes++; it_levels++;
	}
	int levels_count = 0;
	if (lpn_minus_1.size())levels_count++;
	if (lpn_0.size())levels_count++;
	if (lpn_plus_1.size())levels_count++;
	std::string sub_type_minus_1(get_plate_sub_type(lpn_minus_1));
	std::string sub_type_0(get_plate_sub_type(lpn_0));
	std::string sub_type_plus_1(get_plate_sub_type(lpn_plus_1));
	std::string sub_type;
	if (sub_type_minus_1.size()) {
#ifdef _DEBUG		
		assert(sub_type_plus_1.size() == 0 || sub_type_0.size() == 0);
#endif //_DEBUG
		sub_type += sub_type_minus_1;
	}
	if (sub_type_0.size()) {
#ifdef _DEBUG		
		assert(sub_type_plus_1.size() == 0 || sub_type_minus_1.size() == 0);
#endif //_DEBUG
		if (sub_type_minus_1.size()) {
			sub_type += '_';
		}
		sub_type += sub_type_0;
	}
	if (sub_type_plus_1.size()) {
#ifdef _DEBUG		
		assert(sub_type_0.size() == 0 || sub_type_minus_1.size() == 0);
#endif //_DEBUG
		if (sub_type.size()) {
			sub_type += '_';
		}
		sub_type += sub_type_plus_1;
	}
	return sub_type;
}
std::string get_plate_type(
	const std::vector<cv::Rect>& vect_of_detected_boxes,
	const std::vector<int>& classIds, const int number_of_characters_latin_numberplate
)
{
	std::list<cv::Rect> list_of_detected_boxes;
	std::list<int> list_classIds;
	std::copy(vect_of_detected_boxes.begin(), vect_of_detected_boxes.end(), std::back_inserter(list_of_detected_boxes));
	std::copy(classIds.begin(), classIds.end(), std::back_inserter(list_classIds));
	return get_plate_type(list_of_detected_boxes, list_classIds, number_of_characters_latin_numberplate);
}
float cosine(const CvPoint& pt1, const CvPoint& pt2, const CvPoint& summit)//gives the cos in radians of the angle of summit summit formed by pt1 and pt2 and summit
{// dot servira dans la methode angle pour simplifier les calculs
#define dot(a,b) (a.x*b.x + a.y*b.y)
	cv::Point v1(pt1.x - summit.x, pt1.y - summit.y);
	cv::Point v2(pt2.x - summit.x, pt2.y - summit.y);
	int64_t denomin = (v1.x * v1.x + v1.y * v1.y); denomin *= (v2.x * v2.x + v2.y * v2.y);
	float denomin_d = int64ToFloat(denomin);
	//denomin_d+=FLT_EPSILON;
	denomin_d = sqrtf(denomin_d);
	if (denomin_d < FLT_EPSILON) denomin_d = FLT_EPSILON;
#ifdef _DEBUG
	float a = ((float)dot(v1, v2)) / denomin_d;
#ifdef _DEBUG
	assert(a > -1.0f - FLT_EPSILON && a < 1.0f + FLT_EPSILON);
#endif //_DEBUG
	return a;
#else
	return ((float)dot(v1, v2)) / denomin_d;
#endif
}
//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
float cosine(const CvPoint* pt1, const CvPoint* pt2, const CvPoint* summit)
{/*
 float dx1 = pt1->x - summit->x;
 float dy1 = pt1->y - summit->y;
 float dx2 = pt2->x - summit->x;
 float dy2 = pt2->y - summit->y;
 return (dx1*dx2 + dy1*dy2)/sqrtf((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + FLT_EPSILON);
 */
	int dx1 = pt1->x - summit->x;
	int dy1 = pt1->y - summit->y;
	int dx2 = pt2->x - summit->x;
	int dy2 = pt2->y - summit->y;
	if (dx1 == 0 && dx2 == 0 && dy1 == 0 && dy2 == 0) return 0.0f;
	else return ((float)(dx1 * dx2 + dy1 * dy2)) / sqrtf(((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2)));
}
// la methode zappeUnPoint renvoie l'indice du point qui apparae comme eant le plus inutile
// (c.a.d. dont le sinus de l'angle est le + faible) dans la seuence result.
// cette methode renvoie -1 si aucun point ne semble inutile (aucun sinus <0.5f)
int zappeUnPoint(const CvSeq* polygone, CvMemStorage* storage)
{
	// on cree une seuence contenant les points du polygones, 
	// puis enouveau les 2 premiers points,
	// pour permettre le calcul du sin entre 3 points conseutifs
	if (polygone->total < 3) return -1;
	else {
		int i, j;
		float sinus1, sinusmin = 1.0f;
		//he function finds the element with the given index in the sequence and returns the pointer to
		//it. If the element is not found, the function returns 0. The function supports negative indices, where
		//-1 stands for the last sequence element, -2 stands for the one before last, etc.
		// on cherche l'angle dont le sinus est le + faible, on met l'indice de cet angle dans j
		for (i = -1; i < polygone->total - 1; i++)
		{
			//gives the cos in radians of the angle of summit pt0 formed by pt1 and pt2 and pt0
// angle renvoie le cos de l'angle derit par 2 vecteurs
// pt0->pt1 and pt0->pt2 
			sinus1 = cosine(
				(CvPoint*)cvGetSeqElemLPR(polygone, i - 1, 0),
				(CvPoint*)cvGetSeqElemLPR(polygone, i + 1, 0),
				(CvPoint*)cvGetSeqElemLPR(polygone, i, 0)
			);
			sinus1 = sqrtf(1.0f - (sinus1 * sinus1));
			if (sinus1 <= sinusmin)
			{
				sinusmin = sinus1;
				j = i;
			}
		}
		// si le sinus le + faible est en dessous d'un seuil (ici, 0.5f)
		// on renvoie l'indice du point deigne
		// (si j tombe sur un des 2 points rajoute ela fin, 
		// on le replace sur un des 2 points correspondant au deut)
		if (sinusmin < 0.5f)
		{
#ifdef _DEBUG
			assert(j<polygone->total - 1 && j>-2);
#endif
			if (j < 0) j += polygone->total;
			return j;
		}
		else
			// si le sinus le + faible n'est pas en dessous du seuil, on renvoie -1
		{
			return -1;
		}
	}
}
// renvoie la distance entre 2 points
// point pt1
// point pt2
float dist(const CvPoint& pt1, const CvPoint& pt2)
{
#ifdef _DEBUG
	float a(sqrtf(float((pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y))));
	return a;
#else
	//		return sqrtf((pt1.x-pt2.x)*(pt1.x-pt2.x)+(pt1.y-pt2.y)*(pt1.y-pt2.y));
	return sqrtf(((float)(pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y)));
#endif
}
// renvoie la moyenne pondere de 2 pts
// point pt1, de pondeation a1
// point pt2, de pondeation a2
CvPoint moyennepond(const CvPoint& pt1, const CvPoint& pt2,
	const int a1, const int a2)
{
	CvPoint resultat;
	float inverse(1.0f); inverse /= (a1 + a2);
	resultat.x = floatToInt((a1 * pt1.x + a2 * pt2.x) * inverse);
	resultat.y = floatToInt((a1 * pt1.y + a2 * pt2.y) * inverse);
	return resultat;
}
// renvoie la moyenne de 2 pts
// point pt1
// point pt2
CvPoint moyenne(const CvPoint& pt1, const CvPoint& pt2)
{
	CvPoint resultat;
	resultat.x = ((pt1.x + pt2.x) >> 1);
	resultat.y = ((pt1.y + pt2.y) >> 1);
	return resultat;
}
//renvoie la moyenne de 4 pts
// point pt1
// point pt2
// point pt3
// point pt4
CvPoint moyenne4(const CvPoint& pt1, const CvPoint& pt2,
	const CvPoint& pt3, const CvPoint& pt4)
{
	CvPoint resultat1, resultat2;
	resultat1 = moyenne(pt1, pt2);
	resultat2 = moyenne(pt3, pt4);
	return moyenne(resultat1, resultat2);
}
//trouve le point le + proche d'un point de reference parmis 4 pts
// point de reeence ref
// point pt1
// point pt2
// point pt3
// point pt4
CvPoint trouveleplusproche
(const CvPoint& ref, const CvPoint& pt1,
	const CvPoint& pt2, const CvPoint& pt3, const CvPoint& pt4)
{
	float distance_min = FLT_MAX;
	int index = -1;
	float distance = (dist(ref, pt1));
	if (distance_min > distance) {
		distance_min = distance;
		index = 1;
	}
	distance = (dist(ref, pt2));
	if (distance_min > distance) {
		distance_min = distance;
		index = 2;
	}
	distance = (dist(ref, pt3));
	if (distance_min > distance) {
		distance_min = distance;
		index = 3;
	}
	distance = (dist(ref, pt4));
	if (distance_min > distance) {
		distance_min = distance;
		index = 4;
	}
	if (index == 1) return pt1;
	else if (index == 2) return pt2;
	else if (index == 3) return pt3;
	else if (index == 4) return pt4;
	else return pt1;
	/*
	CvPoint resultat=pt1;
	float distance=dist(ref,pt1);
	if(distance>dist(ref,pt2))
	{resultat=pt2;
	distance=dist(ref,pt2);}
	if(distance>dist(ref,pt3))
	{resultat=pt3;
	distance=dist(ref,pt3);}
	if(distance>dist(ref,pt4))
	{resultat=pt4;}
	return resultat;
	*/
}
// renvoie la seuence de places epartir de la seuence de notes
// seuence de notes not
CvSeq* sort(const CvSeq* notes, CvMemStorage* storage)
{
#ifdef _DEBUG
	assert(notes->elem_size == 4 || notes->elem_size == 8);
#endif //_DEBUG
	CvSeq* cle = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);
	CvSeqReader reader;
	std::list<int> liste_indexs;
	//enrgistrement des notes ds une liste
	int i;
	if (notes->elem_size == 4) {
		std::list<int> liste_notes;
		cvStartReadSeq(notes, &reader, 0);
		for (i = 0; i < notes->total; i++)
		{
			int val;
			CV_READ_SEQ_ELEM(val, reader);
			liste_notes.push_back(val);
			liste_indexs.push_back(i);
		}
		//////////////////////////////////////////////////////////////////
		//TRI DE LA LISTE
		//////////////////////////////////////////////////////////////////
		//des listes pour trier (vides au deut puis se remplissant un par un, progressivement)
		std::list<int> liste_notes_tri;
		std::list<int> liste_indexs_tri;
		while (!liste_notes.empty() && !liste_indexs.empty()) {
#ifdef _DEBUG
#ifdef _DEBUG
			assert(liste_notes.size() == liste_indexs.size());
#endif //_DEBUG
#endif //_DEBUG
			int note_(liste_notes.front());
			std::list<int>::iterator it_index(liste_indexs_tri.begin());
			std::list<int>::iterator it_tri(liste_notes_tri.begin());
			while (it_tri != liste_notes_tri.end()
				&& it_index != liste_indexs_tri.end()) {
				if (note_ <= *it_tri) break;
				it_tri++;
				it_index++;
			}
			liste_notes_tri.splice(it_tri, liste_notes, liste_notes.begin());
			liste_indexs_tri.splice(it_index, liste_indexs, liste_indexs.begin());
		}
		//VERIF DE LA LISTE
#ifdef _DEBUG
		assert(liste_notes.empty() &&
			liste_indexs.empty());
		std::list<int>::const_iterator it_verif(liste_notes_tri.begin());
		int note_(-1);
		while (it_verif != liste_notes_tri.end()) {
#ifdef _DEBUG
			assert(*it_verif >= note_);
#endif //_DEBUG
			note_ = *it_verif;
			it_verif++;
		}
#endif //_DEBUG
		//////////////////////////////////////////////////////////////////
		//INSCRIPTION DES RANGS
		//////////////////////////////////////////////////////////////////
		//Pushes several elements to the either end of sequence
		int zero = 0;
		cvSeqPushMulti(cle, &zero, liste_notes_tri.size(), CV_BACK);
		int rang(1);
		std::list<int>::iterator it_index(liste_indexs_tri.begin());
		while (it_index != liste_indexs_tri.end()) {
#ifdef _DEBUG
			assert(*it_index >= 0 && *it_index < cle->total);
#endif //_DEBUG
			* cvGetSeqElemLPR(cle, *it_index, NULL) = rang;
			it_index++;
			rang++;
		}
	}
	else if (notes->elem_size == 8)
	{
		std::list<float> liste_notes;
		cvStartReadSeq(notes, &reader, 0);
		for (i = 0; i < notes->total; i++)
		{
			float val;
			CV_READ_SEQ_ELEM(val, reader);
			liste_notes.push_back(val);
			liste_indexs.push_back(i);
		}
		//////////////////////////////////////////////////////////////////
		//TRI DE LA LISTE
		//////////////////////////////////////////////////////////////////
		//des listes pour trier (vides au deut puis se remplissant un par un, progressivement)
		std::list<float> liste_notes_tri;
		std::list<int> liste_indexs_tri;
		while (!liste_notes.empty() && !liste_indexs.empty()) {
#ifdef _DEBUG
#ifdef _DEBUG
			assert(liste_notes.size() == liste_indexs.size());
#endif //_DEBUG
#endif //_DEBUG
			float note_(liste_notes.front());
			std::list<int>::iterator it_index(liste_indexs_tri.begin());
			std::list<float>::iterator it_tri(liste_notes_tri.begin());
			while (it_tri != liste_notes_tri.end()
				&& it_index != liste_indexs_tri.end()) {
				if (note_ <= *it_tri) break;
				it_tri++;
				it_index++;
			}
			liste_notes_tri.splice(it_tri, liste_notes, liste_notes.begin());
			liste_indexs_tri.splice(it_index, liste_indexs, liste_indexs.begin());
		}
		//VERIF DE LA LISTE
#ifdef _DEBUG
		assert(liste_notes.empty() &&
			liste_indexs.empty());
		std::list<float>::const_iterator it_verif(liste_notes_tri.begin());
		float note_(-1);
		while (it_verif != liste_notes_tri.end()) {
#ifdef _DEBUG
			assert(*it_verif + FLT_EPSILON > note_);
#endif //_DEBUG
			note_ = *it_verif;
			it_verif++;
		}
#endif //_DEBUG
		//////////////////////////////////////////////////////////////////
		//INSCRIPTION DES RANGS
		//////////////////////////////////////////////////////////////////
		//Pushes several elements to the either end of sequence
		int zero = 0;
		cvSeqPushMulti(cle, &zero, liste_notes_tri.size(), CV_BACK);
		int rang(1);
		std::list<int>::iterator it_index(liste_indexs_tri.begin());
		while (it_index != liste_indexs_tri.end()) {
#ifdef _DEBUG
			assert(*it_index >= 0 && *it_index < cle->total);
#endif //_DEBUG
			* cvGetSeqElemLPR(cle, *it_index, NULL) = rang;
			it_index++;
			rang++;
		}
	}
	return cle;
}