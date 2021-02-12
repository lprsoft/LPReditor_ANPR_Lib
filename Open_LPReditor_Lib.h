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

third party software

c++ inference source code

OpenCV 4.5.0 and higher
Copyright © 2021 , OpenCV team
Apache 2 License

ONNXRUNTIME
Copyright © 2020 Microsoft. All rights reserved.
MIT License

model production

YOLOv5
by Glenn Jocher (Ultralytics.com)
GPL-3.0 License

onnx
Copyright (c) Facebook, Inc. and Microsoft Corporation. All rights reserved.
MIT License

*/

#if !defined(OPEN_LPREDITOR_LIB_H)
#define OPEN_LPREDITOR_LIB_H
#pragma once

#include <stddef.h>
/**
	@brief displays image in highgui

	@param int width : width of source image
	@param height : height of source image
	@param pixOpt : pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
	@param *pbData : source image bytes buffer
	param step Number of bytes each matrix row occupies.The value should include the padding bytes at
		the end of each row, if any.If the parameter is missing(set to AUTO_STEP), no padding is assumed
		and the actual step is calculated as cols* elemSize().See Mat::elemSize.
	@return
	@see
	*/

#ifndef LPREDITOR_EXPORTS
#ifdef __cplusplus
extern "C"
{
#endif
#ifdef _WINDOWS
	__declspec(dllimport)
#else //_WINDOWS
#endif //_WINDOWS
	/**
		@brief initializes a new detector by loading its model file and returns its unique id
		@param model_file : c string model filename (must be allocated by the calling program)
		@param len : length of the model filename
		@return the id of the new detector
		@see
		*/
	size_t init_session(size_t len, const char* model_file);

#ifdef _WINDOWS
	__declspec(dllimport)
#else //_WINDOWS
#endif //_WINDOWS
	/**
		@brief call this func once you have finished with the detector --> to free heap allocated memeory
		@param id : unique interger to identify the detector to be freed
		@return true upon success
		@see
		*/
	bool close_session(size_t id);


	/**
		@brief detect lpn in frame

		@param int width : width of source image
		@param height : height of source image
		@param pixOpt : pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
		@param *pbData : source image bytes buffer
		param step Number of bytes each matrix row occupies.The value should include the padding bytes at
			the end of each row, if any.If the parameter is missing(set to AUTO_STEP), no padding is assumed
			and the actual step is calculated as cols* elemSize().See Mat::elemSize.
		@param error_code : un code erreur specifique au lecteur de plaques
		@return
		@see
		*/
#ifdef _WINDOWS
	__declspec(dllimport)
#else //_WINDOWS
#endif //_WINDOWS
	bool detect
	(const int width,//width of image
		const int height,//height of image i.e. the specified dimensions of the image
		const int pixOpt,// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
		void* pbData, size_t step// source image bytes buffer
		, size_t id, size_t lpn_len, char* lpn);

#ifdef __cplusplus
}
#endif
#else 
	//__declspec(dllexport)
#endif //LPREDITOR_EXPORTS


#endif // !defined(OPEN_LPREDITOR_LIB_H)
