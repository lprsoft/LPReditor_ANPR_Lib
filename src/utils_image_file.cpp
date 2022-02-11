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
#include "../include/utils_image_file.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "../include/utils_anpr_detect.h"
#define NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE 36
/**
		@brief
		returns the true license plate number out of a filename
		you must place the true license plate number in the image filename this way : number+underscore+license plate number,
		for instance filename 0000000001_3065WWA34.jpg will be interpreted as an image with the license plate 3065WWA34 in it.
			@param filename: the image filename that contains in it the true registration number
			@return the lpn contained in the image filename
			@see
			*/
std::string getTrueLPN(const std::string& filename, const bool& vrai_lpn_after_underscore)
{
	std::string analysing_string = filename;
	if (analysing_string == "")
		return std::string();
	char sep_underscore = '_';
	size_t index = 0;
	index = (analysing_string.find(sep_underscore));
	if (index != -1)
	{
		std::string subanalysing_string;//la sous chaine
		if (!vrai_lpn_after_underscore) {
			subanalysing_string = analysing_string.substr(0, index);//la sous chaine
		}
		else {
			subanalysing_string = analysing_string.substr(index + 1, analysing_string.length() - (index + 1));//la sous chaine
		}
		if (could_be_lpn(subanalysing_string))
			return subanalysing_string;
		else return std::string();
	}
	else {
		if (could_be_lpn(filename))
			return filename;
		else return std::string();
	}
}
/**
		@brief
		//checks if the characters contained in lpn are compatible with the alphabet
			@param lpn: the registration of the vehicle as a string
			@return
			@see
			*/
//extracts from a test directory all images files 
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
}
/**
		@brief
		//return the ascii character that corresponds to index class output by the dnn
			@param classe : integer index = class identifier, output by the object detection dnn
			@return an ascii character
			@see
			*/
char get_char(const int classe) {
	char _LATIN_LETTERS_LATIN_DIGITS[NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
		'Y', 'Z','0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
	if (classe >= 0 && classe < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE)
		return _LATIN_LETTERS_LATIN_DIGITS[classe];
	else return '?';
}
//retourne l'index du caractere LPChar
int get_index(const char LPChar)
{
	switch (LPChar) {
	case 'A': {return  0; } break;
	case 'B': {return  1; } break;
	case 'C': {return  2; } break;
	case 'D': {return  3; } break;
	case 'E': {return  4; } break;
	case 'F': {return  5; } break;
	case 'G': {return  6; } break;
	case 'H': {return  7; } break;
	case 'I': {return  8; } break;
	case 'J': {return  9; } break;
	case 'K': {return  10; } break;
	case 'L': {return  11; } break;
	case 'M': {return  12; } break;
	case 'N': {return  13; } break;
	case 'O': {return  14; } break;
	case 'P': {return  15; } break;
	case 'Q': {return  16; } break;
	case 'R': {return  17; } break;
	case 'S': {return  18; } break;
	case 'T': {return  19; } break;
	case 'U': {return  20; } break;
	case 'V': {return  21; } break;
	case 'W': {return  22; } break;
	case 'X': {return  23; } break;
	case 'Y': {return  24; } break;
	case 'Z': {return  25; } break;
	case '0': {return  26; } break;
	case '1': {return  27; } break;
	case '2': {return  28; } break;
	case '3': {return  29; } break;
	case '4': {return  30; } break;
	case '5': {return  31; } break;
	case '6': {return  32; } break;
	case '7': {return  33; } break;
	case '8': {return  34; } break;
	case '9': {return  35; } break;
	}
	return -1;
}
//checks if the characters contained in lpn are compatible with the alphabet
/**
		@brief
		//checks if the characters contained in lpn are compatible with the alphabet
			@param lpn: the registration of the vehicle as a string
			@return
			@see
			*/
bool could_be_lpn(const std::string& lpn) {
	char _LATIN_LETTERS_LATIN_DIGITS[NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
		'Y', 'Z','0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
	std::string::const_iterator it(lpn.begin());
	std::list<char> chars;
	while (it != lpn.end()) {
		int i;
		for (i = 0; i < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE; i++) {
			if (*it == _LATIN_LETTERS_LATIN_DIGITS[i]) break;
		}
		if (i < NUMBER_OF_CARACTERS_LATIN_NUMBERPLATE) {
			it++;
		}
		else return false;
	}
	return true;
}