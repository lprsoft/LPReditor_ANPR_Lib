//************************************************************************
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
// Line.cpp: implementation of the C_Line class.
//
//////////////////////////////////////////////////////////////////////


#include "Line.h"
#ifdef _WINDOWS


#endif //_WINDOWS

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

C_Line::C_Line():a(0.0f),b(0.0f)
{
	
}

//constructeur a partir de deux pts
C_Line::C_Line(const cv::Point2f& A,const cv::Point2f& B)
{
	#ifdef _DEBUG
//assert(A.x!=B.x);
#endif //_DEBUG
	if(fabsf(A.x-B.x)<FLT_EPSILON) {
	a=(A.y-B.y)/(A.x-B.x+FLT_EPSILON);
	b=0.0f;
	}
	else {a=(A.y-B.y)/(A.x-B.x);
	b=A.y-(a*A.x);
	}
}
C_Line::C_Line(const C_Line & line_)
{
	a=line_.a;
	b=line_.b;
}
C_Line::C_Line(const float & a_,const float & b_):a(a_),b(b_)
{
	
}
C_Line::~C_Line()
{
	
}
//retourne le pt d'inter de C_Line avec la dte horiz y=ordonnee
float C_Line::get_abs(const int ordonnee) const
{
	#ifdef _DEBUG
assert(a!=0.0f);
#endif //_DEBUG
	return (ordonnee-b)/a;
}
float C_Line::get_abs(const float ordonnee) const
{
	#ifdef _DEBUG
assert(a!=0.0f);
#endif //_DEBUG
	return (ordonnee-b)/a;
}

//retourne l'image de abscisse par la fonction affi,e definie par la droite
int C_Line::get_image_entiere(const int abscisse) const
{
	float ordonnee(a*abscisse+b);
				int ordonne_floor=int(floorf(ordonnee));
				if (ordonnee-ordonne_floor>0.5f) {
					//le point le plus proche de la droite est situe au dessus.
					ordonne_floor++;
					#ifdef _DEBUG
assert(ordonne_floor-ordonnee<0.5f);
#endif //_DEBUG

				}
				return ordonne_floor;
}
//retourne l'image de abscisse par la fonction affi,e definie par la droite
float C_Line::get_image(const float & abscisse) const
{
	return a*abscisse+b;
}

//retourne l'image de abscisse par la fonction affi,e definie par la droite
float C_Line::get_image(const int abscisse) const
{
	return a*abscisse+b;
}
bool C_Line::is_nearly_the_same(const C_Line & right_op) const
{
	return (fabs(a-right_op.a)<FLT_EPSILON 
		&& fabs(b-right_op.b)<FLT_EPSILON);
}
bool C_Line::is_nearly_horiz() const
{
	return (a+0.000001>0.0 && a<0.000001);
}

void C_Line::reset()
{
a=0.0f;b=0.0f;
}
