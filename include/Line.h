/*
************************************************************************
// Copyright (C) 2003-2006, LPReditor SARL, all rights reserved.
// author : Raphael Poulenard.
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
#if !defined(AFX_LINE_H__E3A0AA20_F151_11D5_B96A_CCFB30288840__INCLUDED_)
#define AFX_LINE_H__E3A0AA20_F151_11D5_B96A_CCFB30288840__INCLUDED_
#include <list>
#include <opencv2/opencv.hpp>
#ifdef _DEBUG
#include <assert.h>
#endif //_DEBUG
//**************************
//   conversion between number types
//**************************
inline  float  doubleToFloat(const double& value)
{
#ifdef LPR_PRINTF
	if ((value > FLT_MAX) || (value < -FLT_MAX))
	{
		printf("Probleme de depassement lors d'un cast de DOUBLE en FLOAT\n");
	}
#endif
	return (float)value;
}
inline float intToFloat(int value)
{
	return float(value);
}
inline  float  int64ToFloat(const int64_t& value)
{
	return float(value);
}
inline  int  floatToInt(float value)
{
#ifdef LPR_PRINTF
	if ((value > INT_MAX) || (value < -INT_MAX))
	{
		printf("Probleme de depassement lors d'un cast de float en int\n");
	}
#endif //LPR_PRINTF
	return int(value);
}
/// \brief une classe qui modelise une droite dans un repere, donn�e par son ordonn�e � l'origine et sa pente
/*! \class C_OCROutput
* \brief a class that models a line in a frame, given by its intercept and its slope
**
*/
class C_Line  
{
public:
	//true if the line doesnot have too large slope and not large ordonnee a l'origine
	inline	bool is_valid() const
	{
		return (fabsf(a)<FLT_MAX && fabsf(b)<FLT_MAX);
	};
	//true if the line is (nearly) a vertical one
	inline	bool is_vertical() const
	{
		return fabsf(a)>FLT_MAX*0.5f;
	};
	void reset();
	//determine si les deux droites sont paralleles
	inline bool is_parallel(const C_Line & line_) const
	{
		const float MAX_FLT_ERROR=0.000001f;
		return (fabsf(a-line_.a)<MAX_FLT_ERROR);
	};
	//retourne l'image de abscisse par la fonction affi,e definie par la droite
	int get_image_entiere(const int abscisse) const;
	//retourne l'image de abscisse par la fonction affi,e definie par la droite
	float get_image(const int abscisse) const;
	//retourne l'image de abscisse par la fonction affi,e definie par la droite
	float get_image(const float& abscisse) const;
	//retourne le coefficient directeur
	inline	float get_skew() const{
		return a;
	};
	//retour l'angle de la droite par rapport a l'horizontale
	inline	float get_skew_angle() const{
		return float(atanf(a));
	};
	//retour l'angle de la droite par rapport a l'horizontale
	inline	float get_skew_angle(const C_Line & l) const{
		return (l.get_skew_angle()- atanf(a));
	};
	/**
	@brief cette fonction donne l'angle de l'inclinaison d'un caractere
	cette fonction sert  effectueer une correction de l'inclinaison des characteres
	cette fonction est utilise apres rotation de l'image de la plaque. Elle doit d'abord dterminer 
	si les characteres ainsi reforms prsentent un angle de slant.
	cette fonction effectuee une correction de l'inclinaison des characteres
	this function uses the vanishing point detection algorithm, based 
	on the concept of clustering line intersections based on the density of line intersections in a local region. 
	@return the slant angle of the line of characters
	@see 
	*/
	inline	float get_slant_angle() const{
		const float pi_sur_2=1.570796327f;
		float slant_angle(atanf(a));
#ifdef _DEBUG
		assert(slant_angle > -pi_sur_2 - .1 && slant_angle < pi_sur_2 + .1);
#endif //_DEBUG
		if(slant_angle>0) {
			//      *       *******
			//      *        *     *
			//      *         *     *
			//      *          *******
			//      *           *
			//      *            *
			//      *             *
			//      *              *
			//In that case the slant is positive
			//correction
			// Y'=y
			// X'= x-y*cos(teta) 
			// X'= x-y*sin (ret)
			slant_angle-=pi_sur_2;
#ifdef _DEBUG
			assert(-slant_angle>-FLT_EPSILON);
#endif //_DEBUG
		}
		else {
			//      *       *******
			//      *      *     *
			//      *     *     *
			//      *    *******
			//      *   *
			//      *  *
			//      * *
			//      **
			//In that case the slant is negative
			//il faut que ret <0
			slant_angle+=pi_sur_2;									 									 
#ifdef _DEBUG
			assert(-slant_angle<FLT_EPSILON);
#endif //_DEBUG
		}
#ifdef _DEBUG
		assert(slant_angle > -pi_sur_2 - .1 && slant_angle < pi_sur_2 + .1);
#endif //_DEBUG
		return -slant_angle;
	};
	inline	bool is_x_axis() const{
		if(fabsf(a)>FLT_EPSILON) return false;
		else return(fabsf(b)<FLT_EPSILON);
	};
	//l'ordonnee a l'origine
	float b;
	//le coefficient directeur
	float a;
	//**************************
	//   construct/destruct
	//**************************
	C_Line();
	C_Line(const C_Line & line_);
	C_Line(const float & a_,const float & b_);
	C_Line(const cv::Point2f& A, const cv::Point2f& B);
	virtual ~C_Line();
	//retourne le pt d'inter de C_Line avec la dte horiz y=ordonnee
	float get_abs(const int ordonnee) const;
	float get_abs(const float ordonnee) const;
	bool is_horizontale() const
	{
		return(fabsf(a)<FLT_EPSILON);
	};
	bool is_nearly_the_same(const C_Line & right_op) const;
	//returns the distance (positive or negative) of the point from its vertical projection on the line
	float difference_y(const cv::Point2f& pt) const;
	//retourne la distance du point a sa projection verticale sur la droite
	float dist_y(const cv::Point& pt) const;
	//returns the distance of the point from its vertical projection on the line
	float dist_y(const cv::Point2f& pt) const;
	//returns sum of distances of a list of points from their vertical projections on the line
	float dist_y(const std::list<cv::Point2f>& points) const;
	//returns the distance of the center point of a box from its vertical projection on the line
	float dist_y_from_center(const cv::Rect& box) const;
	//returns sum of distances of the center points of a list of boxes,from their vertical projections on the line
	float dist_y_from_center(const std::list<cv::Rect>& boxes) const;
	//returns sum of distances (which can be positive or negative) of the center points of a list of boxes,from their vertical projections on the line
	float difference_y_from_center(const std::list<cv::Rect>& boxes) const;
	//returns the distance (which can be positive or negative) of the center point of a box from its vertical projection on the line
	float difference_y_from_center(const cv::Rect& box) const;
	float dist_y(const std::list<cv::Point>& points) const;
	//returns average distance of the center points of a list of boxes,from their vertical projections on the line
	float error(const std::list<cv::Rect>& boxes) const;
	//returns average distance of a list of points from their vertical projections on the line
	float error(const std::list<cv::Point2f>& points) const;
protected:
	//true if the line is nearly horizontal
	bool is_nearly_horiz() const;
};
#endif // !defined(AFX_LINE_H__E3A0AA20_F151_11D5_B96A_CCFB30288840__INCLUDED_)
