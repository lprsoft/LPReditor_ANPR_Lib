//************************************************************************
// Copyright (C) 2021, Raphael Poulenard.
// author : Raphael Poulenard.
//************************************************************************
// StatSommesX_Y_H_dbl.cpp: implementation of the C_SumsRegLineXYHDbl class.
//
//////////////////////////////////////////////////////////////////////
#ifndef GITHUB_LPREDITOR
#include "StatSommesX_Y_H_dbl.h"
#include <math.h> 
#include <limits.h>
#ifdef _WINDOWS
#endif //_WINDOWS
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#define new DEBUG_NEW
#endif
inline float divise_par_zero(const float& numerateur, // Modif PN passage de float en double
	const bool denominateur_positif = true) {
	if (denominateur_positif) {
		if (numerateur > 0.0f) {
			if (numerateur < FLT_EPSILON) return 0.0f;
			else return FLT_MAX;
		}
		else {
			if (FLT_EPSILON + numerateur > 0.0f) return 0.0f;
			else return -FLT_MAX;
		}
	}
	else {
		if (numerateur > 0.0f) {
			if (numerateur < FLT_EPSILON) return 0.0f;
			else return -FLT_MAX;
		}
		else {
			if (FLT_EPSILON + numerateur > 0.0f) return 0.0f;
			else return FLT_MAX;
		}
	}
};
C_SumsRegLineXYHDbl::C_SumsRegLineXYHDbl() : somme_x(0.0f)
, somme_y(0.0f)
, produit_xy(0.0f)
, somme_carre_x(0.0f)
, somme_hauteurs(0)
{
}
C_SumsRegLineXYHDbl::C_SumsRegLineXYHDbl(const int somme_hauteurs_) : somme_x(0.0f)
, somme_y(0.0f)
, produit_xy(0.0f)
, somme_carre_x(0.0f)
, somme_hauteurs(somme_hauteurs_)
{
}
C_SumsRegLineXYHDbl::~C_SumsRegLineXYHDbl()
{
}
#ifdef _DEBUG
bool C_SumsRegLineXYHDbl::debug(const float& somme_x_,
	const float& somme_y_,
	const float& produit_xy_,
	const float& somme_carre_x_) const
{
#ifdef LPR_DOUBLE_PRECISION
	return (fabsf(somme_x - somme_x_) < FLT_EPSILON
		&& fabsf(somme_y - somme_y_) < FLT_EPSILON
		&& fabsf((produit_xy - produit_xy_)) < FLT_EPSILON
		&& fabsf((somme_carre_x - somme_carre_x_)) < FLT_EPSILON
		);
#else // LPR_DOUBLE_PRECISION
	bool ok = fabsf(somme_x - somme_x_) < FLT_EPSILON
		&& fabsf(somme_y - somme_y_) < FLT_EPSILON;
	if (produit_xy + produit_xy_ > FLT_EPSILON)
		ok = ok && fabsf((produit_xy - produit_xy_)) < FLT_EPSILON * (produit_xy + produit_xy_);
	else ok = ok && fabsf((produit_xy - produit_xy_)) < FLT_EPSILON;
	if (somme_carre_x + somme_carre_x_ > FLT_EPSILON)
		ok = ok && fabsf((somme_carre_x - somme_carre_x_)) < FLT_EPSILON * (somme_carre_x + somme_carre_x_);
	else ok = ok && fabsf((somme_carre_x - somme_carre_x_)) < FLT_EPSILON;
	return ok;
#endif // LPR_DOUBLE_PRECISION
}
bool C_SumsRegLineXYHDbl::debug(const float& somme_x_,
	const float& somme_y_,
	const float& produit_xy_,
	const float& somme_carre_x_,
	const int  heigth_) const
{
	/*
	return (somme_x== somme_x_ && 	somme_y==somme_y_ &&
		produit_xy==produit_xy_ && somme_carre_x==somme_carre_x_ && somme_hauteurs==heigth_);
		*/
#ifdef LPR_DOUBLE_PRECISION
	return (fabsf(somme_x - somme_x_) < FLT_EPSILON
		&& fabsf(somme_y - somme_y_) < FLT_EPSILON
		&& fabsf((produit_xy - produit_xy_)) < FLT_EPSILON
		&& fabsf((somme_carre_x - somme_carre_x_)) < FLT_EPSILON
		&& somme_hauteurs == heigth_);
#else // LPR_DOUBLE_PRECISION
	return (fabsf(somme_x - somme_x_) < FLT_EPSILON
		&& fabsf(somme_y - somme_y_) < FLT_EPSILON
		&& fabsf((produit_xy - produit_xy_)) < FLT_EPSILON * (produit_xy + produit_xy_)
		&& fabsf((somme_carre_x - somme_carre_x_)) < FLT_EPSILON * (somme_carre_x + somme_carre_x_)
		&& somme_hauteurs == heigth_);
#endif // LPR_DOUBLE_PRECISION
}
#endif //_DEBUG
float C_SumsRegLineXYHDbl::pente(const int nb_elements) const
{//calcul de la moyenne des xi 
				//calcul de la moyenne des yi 
#ifdef _DEBUG
	assert(nb_elements > 1);
#endif //_DEBUG
	if (nb_elements == 1) return 0.0f;
	else if (nb_elements > 1) {
		float moyenne_x = somme_x / nb_elements;
		float moyenne_y = somme_y / nb_elements;
		//calcul de la std_deviation(X)
		float variance = (somme_carre_x)-somme_x * moyenne_x;
		if (fabsf(variance) < FLT_EPSILON) return FLT_MAX;
		//calcul de la Covariance(X,Y)
		float Covariance = (produit_xy)-moyenne_x * somme_y;
		//calcul de la pente p=Covariance(X,Y)/variance(X)
#ifdef _DEBUG
		assert(variance > -FLT_EPSILON);
#endif //_DEBUG
		float pente_;
		if (variance != 0.0f) pente_ = Covariance / variance;
		else pente_ = divise_par_zero(Covariance);
#ifdef _DEBUG
		const float pi = 3.1415926535897932384626433832795f;
#ifdef _DEBUG
		assert(atanf(pente_) <= pi / 2 && (atanf(pente_) + pi / 2) > -FLT_EPSILON);
#endif //_DEBUG
#endif
		return pente_;
	}
	else return 0.0f;
}
cv::Point2f C_SumsRegLineXYHDbl::barycenter(const int nb_points)
{
#ifdef _DEBUG
	assert(nb_points > 1);
#endif //_DEBUG
	if (nb_points <= 0) return cv::Point2f(FLT_MAX, FLT_MAX);
	else if (nb_points == 0) {
		float moyenne_x = somme_x;
		float moyenne_y = somme_y;
		return cv::Point2f(moyenne_x, moyenne_y);
	}
	else {
#ifdef _DEBUG
		assert(nb_points > 1);
#endif //_DEBUG
		float moyenne_x = somme_x / nb_points;
		float moyenne_y = somme_y / nb_points;
		return cv::Point2f(moyenne_x, moyenne_y);
	}
}
C_Line C_SumsRegLineXYHDbl::regression_line(const int nb_elements)
{
	if (nb_elements == 0) return C_Line();
	else if (nb_elements == 1) return C_Line(0.0, somme_y);
	else if (nb_elements > 1) {
		float moyenne_x = somme_x / nb_elements;
		float moyenne_y = somme_y / nb_elements;
		//calcul de la std_deviation(X)
		float variance = (somme_carre_x)-somme_x * moyenne_x;
		if (fabsf(variance) < FLT_EPSILON) return C_Line(FLT_MAX, FLT_MAX);
		//calcul de la Covariance(X,Y)
		float Covariance = (produit_xy)-moyenne_x * somme_y;
		//calcul de la pente_ p=Covariance(X,Y)/variance(X)
		float pente_ = Covariance / variance;
		//calcul du coefficient q ( y=px+q )
		float ordonnee_origine = moyenne_y - pente_ * moyenne_x;
#ifdef _DEBUG
		const float pi = 3.1415926535897932384626433832795f;
		assert(atanf(pente_) <= pi / 2 && (atanf(pente_) + pi / 2) > -FLT_EPSILON);
#endif //_DEBUG	
		C_Line regression_line(pente_, ordonnee_origine);
#ifdef _DEBUG
		//calcul de la moyenne des xi 
		//calcul de la moyenne des yi 
		//ce sont les coordonnees du centre de gravit du fond principal
		float moyenne_x_ = somme_x / nb_elements;
		float moyenne_y_ = somme_y / nb_elements;
		//calcul de la std_deviation(X)
		float variance_ = (somme_carre_x - somme_x * moyenne_x_);
		//calcul de la Covariance(X,Y)
		float Covariance_ = (produit_xy - moyenne_x_ * somme_y);
		//calcul de la pente p=Covariance(X,Y)/variance(X)
#ifdef _DEBUG
		assert(variance > -FLT_EPSILON);
#endif //_DEBUG		
		float pente__;
		if (variance_ > FLT_EPSILON) pente__ = Covariance_ / variance_;
		else {
			pente__ = divise_par_zero(Covariance_);//calcul du coefficient q ( y=px+q )
		}
		float ordonnee_origine_ = moyenne_y_ - pente__ * moyenne_x_;
#ifdef _DEBUG
		assert(fabsf(regression_line.a - pente__) < FLT_EPSILON && fabsf(regression_line.b - ordonnee_origine_) < 0.001f);
#endif //_DEBUG
		float diff_ = regression_line.a - pente__;
		diff_ = regression_line.b - ordonnee_origine_;
#endif //_DEBUG
		return regression_line;
	}
	else return C_Line();
}
bool C_SumsRegLineXYHDbl::operator ==(const C_SumsRegLineXYHDbl& right_op) const
{
#ifdef _DEBUG
	double somme_x_r = right_op.somme_x;
	double somme_y_r = right_op.somme_y;
	double produit_xy_r = right_op.produit_xy;
	double somme_carre_x_r = right_op.somme_carre_x;
	double produit_xy_ = produit_xy;
	double somme_carre_x_ = somme_carre_x;
	assert(fabsf(somme_x - right_op.somme_x) < FLT_EPSILON
		&& fabsf(somme_y - right_op.somme_y) < FLT_EPSILON
		&& somme_hauteurs == right_op.somme_hauteurs);
#ifdef LPR_DOUBLE_PRECISION
	assert(fabsf((produit_xy - right_op.produit_xy)) <
		FLT_EPSILON * (produit_xy + right_op.produit_xy)
		&& fabsf((somme_carre_x - right_op.somme_carre_x)) < FLT_EPSILON *
		(somme_carre_x + right_op.somme_carre_x));
#else // LPR_DOUBLE_PRECISION
	assert(fabsf((produit_xy - right_op.produit_xy)) / (produit_xy + right_op.produit_xy) < FLT_EPSILON
		&& fabsf((somme_carre_x - right_op.somme_carre_x)) / (somme_carre_x + right_op.somme_carre_x) < FLT_EPSILON);
#endif // LPR_DOUBLE_PRECISION
	float dif = fabsf(somme_x - right_op.somme_x);
	dif = fabsf(somme_y - right_op.somme_y);
	dif = fabsf((produit_xy - right_op.produit_xy));
	dif = fabsf((somme_carre_x - right_op.somme_carre_x));
#endif
#ifdef LPR_DOUBLE_PRECISION
	return (fabsf(somme_x - right_op.somme_x) < FLT_EPSILON
		&& fabsf(somme_y - right_op.somme_y) < FLT_EPSILON
		&& fabsf((produit_xy - right_op.produit_xy)) < FLT_EPSILON * fabsf((produit_xy + right_op.produit_xy))
		&& fabsf((somme_carre_x - right_op.somme_carre_x)) < FLT_EPSILON * (somme_carre_x + right_op.somme_carre_x)
		&& somme_hauteurs == right_op.somme_hauteurs);
#else // LPR_DOUBLE_PRECISION
	return (fabsf(somme_x - right_op.somme_x) < FLT_EPSILON
		&& fabsf(somme_y - right_op.somme_y) < FLT_EPSILON
		&& fabsf((produit_xy - right_op.produit_xy)) < FLT_EPSILON * (produit_xy + right_op.produit_xy)
		&& fabsf((somme_carre_x - right_op.somme_carre_x)) < FLT_EPSILON * (somme_carre_x + right_op.somme_carre_x)
		&& somme_hauteurs == right_op.somme_hauteurs);
#endif // LPR_DOUBLE_PRECISION
}
void C_SumsRegLineXYHDbl::clear()
{
	somme_x = 0.0f;
	somme_y = 0.0f;
	produit_xy = 0.0f;
	somme_carre_x = 0.0f;
	somme_hauteurs = 0;
}
#endif //GITHUB_LPREDITOR