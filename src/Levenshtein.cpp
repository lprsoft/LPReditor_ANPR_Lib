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
// Levenshtein.cpp: implementation of the Levenshtein class.
//
// Based on the work by Michael Gilleland
// http://www.merriampark.com/ld.htm
//////////////////////////////////////////////////////////////////////
#include "../include/Levenshtein.h"
#include <string.h>
#include <stdlib.h>
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
Levenshtein::Levenshtein()
{
}
Levenshtein::~Levenshtein()
{
}
//****************************************************
// Compute Levenshtein distance
// This code is based on the work by Michael Gilleland
//
// It uses dynamic arrays which allows any std::string size.
//****************************************************
int Levenshtein::Get2 (char const *s, char const *t)
{
	return Get2(s, strlen(s), t, strlen(t) );
}
int Levenshtein::Get2 (const char *s, size_t n, const char* t, size_t dst)
{
	int *d; // pointer to matrix
	size_t i; // iterates through s
	size_t j; // iterates through t
	//char s_i; // ith character of s
	char t_j; // jth character of t
	int cost; // cost
	int result; // result
	int cell; // contents of target cell
	int above; // contents of cell immediately above
	int left; // contents of cell immediately to left
	int diag; // contents of cell immediately above and to left
	size_t sz; // number of cells in matrix
	// Step 1
	if (n == 0) {
		return int(dst);
	}
	if (dst == 0) {
		return int(n);
	}
	sz = (n+1) * (dst+1) * sizeof (int);
	d = (int *) malloc (sz);
	// Step 2
	for (i = 0; i <= n; i++) {
		PutAt (d, i, 0, n, int(i));
	}
	for (j = 0; j <= dst; j++) {
		PutAt (d, 0, j, n, int(j));
	}
	// Step 3
	for (i = 1; i <= n; i++) {
		char s_i; // ith character of s
		s_i = s[i-1];
		// Step 4
		for (j = 1; j <= dst; j++) {
			t_j = t[j-1];
			// Step 5
			if (s_i == t_j) {
				cost = 0;
			}
			else {
				cost = 1;
			}
			// Step 6
			above = GetAt (d,i-1,j, n);
			left = GetAt (d,i, j-1, n);
			diag = GetAt (d, i-1,j-1, n);
			cell = Minimum (above + 1, left + 1, diag + cost);
			PutAt (d, i, j, n, cell);
		}
	}
	// Step 7
	result = GetAt (d, n, dst, n);
	free (d);
	return result;
}
/*****************************************************
 * Second implementation of the Levenshtein algorithm.
 * A static array is used with length MAXLINE. Make
 * sure that your strings are no longer! Otherwise use
 * the algorithm above.
*/
#define MAXLINE 128
int Levenshtein::Get(const char *a, const char *b)
{
	return Get(a, strlen(a), b, strlen(b));
}
int Levenshtein::Get(const char *a, size_t aLen, const char *b, size_t bLen)
{
	int arr[MAXLINE][MAXLINE];
	int i,j,l,dst,n,add;
	// MAXLINE is the limit! If the strings are longer use the other implementation
	if (aLen > MAXLINE || bLen > MAXLINE)
	{
		return Get2(a, aLen, b, bLen);
	}
	for (i=0;i<=aLen;i++) 
	{
		arr[0][i]=i;
	}
	for (j=0;j<=bLen;j++) 
	{
		arr[j][0]=j;
	}
	for (j=1;j<=bLen;j++) 
	{
		for (i=1;i<=aLen;i++) 
		{
			if (a[i-1] == b[j-1])
			{ 
				add=0; 
			} 
			else 
			{ 
				add=1; 
			}
			dst = 1+arr[j-1][i];
			l = 1+arr[j][i-1];
			n = add+arr[j-1][i-1];
			arr[j][i] = (dst < l ? (dst < n ? dst : n): (l < n ? l : n));
		}
	}
	return arr[bLen][aLen];
}

