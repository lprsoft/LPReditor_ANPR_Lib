#include"utils_opencv.h"
#include <filesystem>
#include "utils_anpr_detect.h"
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
#ifdef _DEBUG
	/*
	assert(!(box.x<rect_im.x || box.y<rect_im.y
		|| box.x + box.width > rect_im.x + rect_im.width
		|| box.y + box.height> rect_im.y + rect_im.height));*/
#endif //_DEBUG
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