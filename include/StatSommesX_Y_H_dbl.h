//************************************************************************
// Copyright (C) 2021, Raphael Poulenard.
// author : Raphael Poulenard.
//************************************************************************
// StatSommesX_Y_H_dbl.h: interface for the C_SumsRegLineXYHDbl class.
//
//////////////////////////////////////////////////////////////////////
#if !defined(AFX_STATSOMMESX_Y_H_DBL_H__EA007980_0205_11D6_B96A_DB3892D34B43__INCLUDED_)
#define AFX_STATSOMMESX_Y_H_DBL_H__EA007980_0205_11D6_B96A_DB3892D34B43__INCLUDED_
#ifndef GITHUB_LPREDITOR
#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
#include "Line.h"
/// class used in segmentation algo filtering method
/**cette classe sert a calculer des sommes moyennes et
 variances necessaires a l'tabissement d'une droite de regresssion*/
 /**cette classe sert a calculer des sommes moyennes et
  variances necessaires a l'tabissement d'une droite de regresssion*/
class C_SumsRegLineXYHDbl  
{
public:
	//! to reset the structure.
    /*!
	this function reassign to the member their default value.
    */
	void clear();
	//gets the barycenter of the points cloud
cv::Point2f barycenter(const int nb_elements);
C_Line regression_line(const int nb_elements);
	float pente(const int nb_elements) const;
#ifdef _DEBUG
	bool debug(const float & somme_x_, 
	const float & somme_y_, 
	const float &  produit_xy_, 
	const float &  somme_carre_x_) const;
	bool debug(const float & somme_x_, 
	const float & somme_y_, 
	const float &  produit_xy_, 
	const float &  somme_carre_x_,
	const int  heigth_) const;
#endif //_DEBUG
	//**************************
  //   construct/destruct
  //**************************
	C_SumsRegLineXYHDbl();
	C_SumsRegLineXYHDbl(const int somme_hauteurs_);
	virtual ~C_SumsRegLineXYHDbl();
	//sum of x coordinates of points cloud
	float somme_x;
	//sum of y coordinates of points cloud
	float somme_y;
	//sum of x by y coordinates product of points cloud
	float produit_xy;
	//sum of squared x coordinates of points cloud
	float somme_carre_x;
	//sum of heights of bouding boxes (not used in that project)
	int somme_hauteurs;
	inline void add(const float& x, const float& y, const int hauteur) {
		somme_x += x;
		somme_y += y;
		produit_xy += x*y;
		somme_carre_x += x*x;
		somme_hauteurs += hauteur;
	}
	inline void operator+=(const cv::Point2f & Center) {
		somme_x += Center.x;
		somme_y += Center.y;
		produit_xy += Center.x*Center.y;
		somme_carre_x += Center.x*Center.x;
	}
	inline void operator-=(const cv::Point2f & Center) {
		somme_x -= Center.x;
		somme_y -= Center.y;
		produit_xy -= Center.x*Center.y;
		somme_carre_x -= Center.x*Center.x;
		#ifdef _DEBUG
assert(somme_x >=0 && somme_y >=0 && 
		produit_xy >=0 && somme_carre_x>=0);
	#endif //_DEBUG
	}
	inline void substract(const float & x, const float & y, const int hauteur) {
		somme_x -= x;
		somme_y -= y;
		produit_xy -= x*y;
		somme_carre_x -=x*x;
		#ifdef _DEBUG
assert(somme_x >=0 && somme_y >=0 && 
		produit_xy >=0 && somme_carre_x>=0);
	#endif //_DEBUG
somme_hauteurs -= hauteur;
	}
	inline void operator+=(const C_SumsRegLineXYHDbl & stat) {
		somme_x += stat.somme_x;
		somme_y += stat.somme_y;
		produit_xy += stat.produit_xy;
		somme_carre_x += stat.somme_carre_x;
		somme_hauteurs+=stat.somme_hauteurs;
	}
 	inline void subtract(const cv::Point2f & Center, const int hauteur) {
 		somme_x -= Center.x;
 		somme_y -= Center.y;
 		produit_xy -= Center.x*Center.y;
 		somme_carre_x-=Center.x*Center.x;
 		somme_hauteurs-=hauteur;
 	}
	bool operator ==(const C_SumsRegLineXYHDbl & right_op) const;
};
#endif //GITHUB_LPREDITOR
#endif // !defined(AFX_STATSOMMESX_Y_H_DBL_H__EA007980_0205_11D6_B96A_DB3892D34B43__INCLUDED_)
