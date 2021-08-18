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
#if !defined(UTILS_IMAGE_FILE_H)
#define UTILS_IMAGE_FILE_H
#pragma once
#include <list>
#include "Levenshtein.h"
#include <opencv2/core.hpp>
/**
		@brief
		//return the ascii character that corresponds to index class output by the dnn
			@param classe : integer index = class identifier, output by the object detection dnn
			@return an ascii character
			@see
			*/
char get_char(const int classe);
/**
		@brief
		//checks if the characters contained in lpn are compatible with the alphabet
			@param lpn: the registration of the vehicle as a string
			@return
			@see
			*/
bool could_be_lpn(const std::string& lpn);
/**
		@brief
		returns the true license plate number out of a filename
		you must place the true license plate number in the image filename this way : number+underscore+license plate number,
		for instance filename 0000000001_3065WWA34.jpg will be interpreted as an image with the license plate 3065WWA34 in it.
			@param filename: the image filename that contains in it the true registration number
			@return the lpn contained in the image filename
			@see
			*/
std::string getTrueLPN(const std::string& filename, const bool& vrai_lpn_after_underscore);
//extracts from a test directory all images files 
void load_images_filenames(const std::string& dir, std::list<std::string>& image_filenames);
void show_boxes(const cv::Mat& frame, const std::list<cv::Rect>& true_boxes, const std::list<int>& classesId);
void drawPred(const int classId, const int left, const int top, const int right, const int bottom, cv::Mat& frame, const std::vector<std::string>& classes);
#endif // !defined(UTILS_IMAGE_FILE_H)
