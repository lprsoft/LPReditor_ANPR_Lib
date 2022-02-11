#if !defined(UTILS_OPEN_CV)
#define UTILS_OPEN_CV
#include "Line.h"
#include <opencv2/opencv.hpp>
#include <opencv2/text.hpp>
#define cvGetSeqElemLPR( param1, param2, param3) cvGetSeqElem( param1, param2)
/**
	@brief 
	//for each box in the container, check that it is nearly entirely contained in the second argument
	@param box : box a bounding box
	@param rect_im : ROi or second bounding box
	@return true if intersection is at least 90% of the box (which means box is nearly entirely in the second argument)
	@see
	*/
	//for each box in the container, check that it is nearly entirely contained in the second argument
bool is_in_rect_if(const std::list<cv::Rect>& boxes, const cv::Rect& rect_im);
/**
	@brief 
	//for each box in the container, check that it is nearly entirely contained in the second argument
	@param box : box a bounding box
	@param rect_im : ROi or second bounding box
	@return true if intersection is at least 90% of the box (which means box is nearly entirely in the second argument)
	@see
	*/
	//for each box in the container, check that it is nearly entirely contained in the second argument
bool is_in_rect_if(const std::vector<cv::Rect>& boxes, const cv::Rect& rect_im);
/**
	@brief
	//returns the iou (intersection over union) of two boxes
	@param r1 : first rectangle
	@param r2 : second rectangle
	@return the iou (a float value between 0 and 1)
	@see
	*/
//float iou(const cv::Rect& r1, const cv::Rect& r2);
std::list<cv::Rect> fiter_out(
	const std::vector<cv::Rect>& true_boxes,  // the true boxes extracted from pascal voc xml file, as a list
	const std::list<cv::Rect>& detected_boxes, //  the boxes detected by nn detector, stored in a list of rectagles objects
	const std::list<float>& ious,
	std::list<float>& out_ious);
std::list<cv::Rect> fiter_out(
	const std::list<cv::Rect>& true_boxes,  // the true boxes extracted from pascal voc xml file, as a list
	const std::list<cv::Rect>& detected_boxes, //  the boxes detected by nn detector, stored in a list of rectagles objects
	const std::list<float>& ious,
	std::list<float>& out_ious);
float iou(
	const std::list<cv::Rect>& true_boxes,  // the true boxes extracted from pascal voc xml file, as a list
	const std::list<cv::Rect>& detected_boxes //  the boxes detected by nn detector, stored in a list of rectagles objects
);
float iou(
	const std::list<cv::Rect>& boxes,  // the true boxes extracted from pascal voc xml file, as a list
	const cv::Rect& box //  the boxes detected by nn detector, stored in a list of rectagles objects
);
/**
	@brief
//rearrange bounding boxes from left to right
cette fonction trie la liste de gauche a droite
			@return reordered set of boxes
			@see
			*/
std::list<cv::Rect> sort_from_left_to_right(const std::list<cv::Rect>& boxes);
/**
	@brief
//rearrange bounding boxes from left to right
cette fonction trie la liste de gauche a droite
*@param[in]  number_of_characters_latin_numberplate : number of characters in a latin alphabet(usually 36 = 26 letters + 10 digits)
			@return reordered set of boxes
			@see
			*/
std::list<cv::Rect> sort_from_left_to_right(const std::list<cv::Rect>& boxes, const std::list<int>& classes, std::list<int>& sorted_classes,
	const int number_of_characters_latin_numberplate
	//, const int index_first_mark_model
);
/**
	@brief
//rearrange bounding boxes from left to right
cette fonction trie la liste de gauche a droite
			@return reordered set of boxes
			@see
			*/
void sort_from_left_to_right(
	std::list<cv::Rect>& boxes, std::list<float>& confidences,
	std::list<int>& classIds_);
/**
		@brief
		cette fonction trie la liste de gauche a droite
//rearrange detected bounding boxes from left to right
			@param[out]  :std list of detected boxes
			@param[out] confidences : confidences of detected boxes
			@param[out] classIds : std::list of indeces that indicate the classes of each of the above detected boxes
			@return void
			@see
			*/
void sort_from_left_to_right(
	std::list<cv::Rect>& boxes,
	std::list<int>& classIds_);
bool is_on_the_left(const cv::Rect& box1, const cv::Rect& box2);
// 
/**
		@brief cette fonction trie la liste de gauche a droite
//rearrange detected bounding boxes from left to right
			@param[out]  :std list of detected boxes
			@param[out] confidences : confidences of detected boxes
			@param[out] classIds : std::list of indeces that indicate the classes of each of the above detected boxes
			@return void
			@see
			*/
void sort_from_left_to_right(
	std::list<cv::Rect>& boxes, std::list<float>& confidences);
bool is_in_rect(const std::list<cv::Rect>& boxes, const cv::Rect& rect_im);
bool is_in_rect(const cv::Rect& box, const cv::Rect& rect_im);
bool is_in_rect(const cv::Point& pt, const cv::Rect& rect_im);
//type_of_roi_for_iou==1 large
				//type_of_roi_for_iou ==2 right_side
				//type_of_roi_for_iou == 3 left side
				//type_of_roi_for_iou ==0 no expansion
//float detect(const cv::Mat & frame, const std::list<cv::Rect>& boxes, const cv::Rect& box, const int w, const int type_of_roi_for_iou);
void get_mean_std(const cv::Mat & frame, const cv::Rect& box, float& mean, float& standard_deviation);
bool get_upperand_lower_lines
(const std::list<cv::Rect>& boxes, C_Line& line_sup, C_Line& line_inf);
struct MSERParams
{
	MSERParams(int _delta = 5, int _min_area = 60, int _max_area = 14400,
		double _max_variation = 0.25, double _min_diversity = .2,
		int _max_evolution = 200, double _area_threshold = 1.01,
		double _min_margin = 0.003, int _edge_blur_size = 5)
	{
		delta = _delta;
		minArea = _min_area;
		maxArea = _max_area;
		maxVariation = _max_variation;
		minDiversity = _min_diversity;
		maxEvolution = _max_evolution;
		areaThreshold = _area_threshold;
		minMargin = _min_margin;
		edgeBlurSize = _edge_blur_size;
		pass2Only = false;
	}
	int delta;
	int minArea;
	int maxArea;
	double maxVariation;
	double minDiversity;
	bool pass2Only;
	int maxEvolution;
	double areaThreshold;
	double minMargin;
	int edgeBlurSize;
};
/*
bool trouve_la_plaque(const cv::Mat& frame
	, cv::Point& p0, cv::Point& p1, cv::Point& p2, cv::Point& p3,
	cv::Point& top_left_,
	cv::Point& top_right_,
	cv::Point& bottom_right_,
	cv::Point& bottom_left_,
	cv::Rect& rect_OpenLP);*/
bool trouve_la_plaque(const cv::Mat& frame, const cv::Rect& global_rect
	, cv::Point& p0, cv::Point& p1, cv::Point& p2, cv::Point& p3,
	cv::Point& top_left_,
	cv::Point& top_right_,
	cv::Point& bottom_right_,
	cv::Point& bottom_left_,
	cv::Rect& rect_OpenLP);
cv::Rect  get_global_rect(const cv::Point& bottom_right,
	const 	cv::Point& bottom_left, const cv::Point& top_right
	, const cv::Point& top_left);
//cette fonction retourne le rect englobant la collection
//gets the reunion of all the boxes
cv::Rect get_global_rect(const std::list<cv::Rect>& l);
//cette fonction retourne le rect englobant la collection
//gets the reunion of all the boxes
cv::Rect get_global_rect(const std::vector<cv::Rect>& l);
void vector_to_list(
	const std::vector<cv::Rect>& boxes,
	std::list<cv::Rect>& lboxes);
bool trouve_la_plaque(const cv::Mat& frame, const cv::Rect& global_rect
	, cv::Point& p0, cv::Point& p1, cv::Point& p2, cv::Point& p3,
	cv::Point& top_left_,
	cv::Point& top_right_,
	cv::Point& bottom_right_,
	cv::Point& bottom_left_,
	cv::Rect& rect_OpenLP);
bool trouve_la_plaque(const cv::Mat& frame,
	const std::list<int>& classes, const std::list<cv::Rect>& boxes
	,
	cv::Point& top_left_,
	cv::Point& top_right_,
	cv::Point& bottom_right_,
	cv::Point& bottom_left_, cv::Rect& rect_OpenLP);
bool in_quadrilatere(const cv::Point& pt,
	const cv::Point& top_left, const cv::Point& top_right, const cv::Point& bottom_right, const cv::Point& bottom_left, const float& signed_distance);
bool in_quadrilatere(const cv::Point& pt, const std::vector<cv::Point >& contours, const float& signed_distance);
bool in_quadrilatere(const cv::Rect& box, const std::vector<cv::Point >& contours, const float& signed_distance);
bool in_quadrilatere(const std::list<cv::Rect>& boxes, const std::vector<cv::Point >& contours, const float& signed_distance);
#include "opencv2/imgproc/imgproc_c.h"
float cosine(const CvPoint& pt1, const CvPoint& pt2, const CvPoint& pt0);
float cosine(const CvPoint* pt1, const CvPoint* pt2, const CvPoint* summit);
bool is_2D(const CvSeq* polygone);
int zappeUnPoint(const CvSeq* polygone, CvMemStorage* storage);
float dist(const CvPoint& pt1, const CvPoint& pt2);
CvPoint moyennepond(const CvPoint& pt1, const CvPoint& pt2, const int a1, const int a2);
CvPoint moyenne(const CvPoint& pt1, const CvPoint& pt2);
CvPoint moyenne4(const CvPoint& pt1, const CvPoint& pt2, const CvPoint& pt3, const CvPoint& pt4);
CvPoint trouveleplusproche(const CvPoint& ref, const CvPoint& pt1,
	const CvPoint& pt2, const CvPoint& pt3, const CvPoint& pt4);
CvSeq* sort(const CvSeq* note, CvMemStorage* storage);
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
// renvoie une seuence de 4 points encadrant la plaque
// trouveesur l'image img
// IplImage* img
// CvMemStorage* storage 
//peut traiter les images sources couleur ou en niveaux de gris
CvSeq* findSquares4(IplImage* img, CvMemStorage* storage, const int nTresh, //
					//le nb de composantes couleurs ex nb_composantes_color=3 pour une image RGB
	const int nb_composantes_color, const int dist_min_bord,
	const int hauteur_plaque_min, const float& rapport_largeur_sur_hauteur_min);
CvSeq* findSquares4
(IplImage* im_src, const cv::Rect& global_rect, CvMemStorage* storage,
	const int nTresh, const int mean_carac
	, const int mean_fond, //
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min);
CvSeq* findSquares4
(IplImage* im_src, CvMemStorage* storage,
	const int nTresh, //
	//le nb de composantes couleurs ex nb_composantes_color=3 pour une image RGB
	const int nb_composantes_color,
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min);
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
	const float& rapport_largeur_sur_hauteur_min);
CvSeq* findSquares4_multiresolution
(IplImage* im_src, CvMemStorage* storage,
	const int nTresh, //
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min);
CvSeq* findSquares4_multiresolution(
	IplImage* im_src, CvMemStorage* storage,
	const int nTresh, const int mean_carac
	, const int mean_fond, //
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min);
CvSeq* findSquares4_multiresolution
(
	IplImage* im_src, const cv::Rect& global_rect, CvMemStorage* storage,
	const int nTresh, const int mean_carac
	, const int mean_fond, //
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min
);
CvSeq* findSquares4_multiresolution
(
	IplImage* im_src, CvMemStorage* storage,
	const int nTresh, //
	//le nb de composantes couleurs ex nb_composantes_color=3 pour une image RGB
	const int nb_composantes_color,
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min
);
CvSeq* findSquares4_multiresolution
(IplImage* imgR, IplImage* imgG, IplImage* imgB, CvMemStorage* storage,
	const int nTresh, //
	//le nb de composantes couleurs ex nb_composantes_color=3 pour une image RGB
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min);
/*
CvSeq* findSquares4
(IplImage* imgR, IplImage* imgG, IplImage* imgB, CvMemStorage* storage,
	const int nTresh, //
	//le nb de composantes couleurs ex nb_composantes_color=3 pour une image RGB
	const int dist_min_bord,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min);*/
//peut traiter les images sources couleur ou en niveaux de gris
/*
bool trouve_la_plaque(IplImage* etude, const int nTresh, const int nb_composantes_color,
	int resultat[4][2], const int dist_min_bord,
	const int hauteur_plaque_min, const float& rapport_largeur_sur_hauteur_min);*/
///////////////////////////////////////////////////////////////////////////////////////////
// retourne le nb de points en commun entre deux images
float percentage_match(IplImage* img1, IplImage* img2);
bool quadrilatere_is_convex(const cv::Point& pt0, const cv::Point& pt1, const cv::Point& pt2, const cv::Point& pt3);
//must be convex quadrilatere
bool width_is_larger(const cv::Point& pt0, const cv::Point& pt1, const cv::Point& pt2, const cv::Point& pt3);
//must be convex quadrilatere
bool get_corners(const cv::Point& pt0, const cv::Point& pt1, const cv::Point& pt2, const cv::Point& pt3,
	cv::Point& top_left, cv::Point& top_right, cv::Point& bottom_right, cv::Point& bottom_left);
//must be convex quadrilatere
bool get_corners(const IplImage* im, const CvPoint& pt0, const CvPoint& pt1, const CvPoint& pt2, const CvPoint& pt3,
	cv::Point& top_left, cv::Point& top_right, cv::Point& bottom_right, cv::Point& bottom_left);
/*
bool trouve_la_plaque(IplImage* etude, int nTresh,
	cv::Point& p0, cv::Point& p1, cv::Point& p2, cv::Point& p3,
	cv::Point& top_left_,
	cv::Point& top_right_,
	cv::Point& bottom_right_,
	cv::Point& bottom_left_, cv::Rect& rect_OpenLP,
	const int dist_min_bord
	,
	const int hauteur_plaque_min,
	const float& rapport_largeur_sur_hauteur_min);
//peut traiter les images sources couleur ou en niveaux de gris
bool trouve_la_plaque(IplImage* etude, int nTresh
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
	const float& rapport_largeur_sur_hauteur_min);
*/
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
	const float& rapport_largeur_sur_hauteur_min);
bool trouve_la_plaque(const cv::Mat& frame, const cv::Rect& global_rect
	, cv::Point& p0, cv::Point& p1, cv::Point& p2, cv::Point& p3,
	cv::Point& top_left_,
	cv::Point& top_right_,
	cv::Point& bottom_right_,
	cv::Point& bottom_left_,
	cv::Rect& rect_OpenLP);
bool trouve_la_plaque(const cv::Mat& frame,
	const std::list<int>& classes, const std::list<cv::Rect>& boxes
	,
	cv::Point& top_left_,
	cv::Point& top_right_,
	cv::Point& bottom_right_,
	cv::Point& bottom_left_, cv::Rect& rect_OpenLP);
std::string get_plate_type(
	const std::list<cv::Rect>& vect_of_detected_boxes,
	const std::list<int>& classIds, const int number_of_characters_latin_numberplate
);
std::string get_plate_type(
	const std::vector<cv::Rect>& vect_of_detected_boxes,
	const std::vector<int>& classIds, const int number_of_characters_latin_numberplate
);
#endif // !defined UTILS_OPEN_CV
#pragma once
