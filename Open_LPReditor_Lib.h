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
		@brief initializes a new detector, by loading its model file and returns its unique id. The repo comes with two models namely lpreditor_anpr_focused_on_lp and lpreditor_anpr_global_view.
		So you have to call this function twice to initialize both models (note that it is possible, but not a good idea, to initialize just one model).
		@param model_file : c string model filename (must be allocated by the calling program)
		@param len : length of the model filename
		@return the id of the new detector
		@see
		*/
	size_t init_detector(size_t len, const char* model_file);
	/**
		@brief detect lpn in frame. this function is equivalent to detect_without_lpn_detection (same usage and same arguments)
		but it is more accurate (and also longer to run) since it uses two models instead of one.
		please make sure you have initialized the lpreditor_anpr_global_view model in the init_detector function (see sample_cpp for uses examples).
	@param width : width of source image
		@param width : width of source image
		@param height : height of source image
		@param pixOpt : pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
		@param *pbData : source image bytes buffer
		@param step Number of bytes each matrix row occupies. The value should include the padding bytes at
			the end of each row, if any..See sample_cpp for a use case.
@param id_global_view : unique id to a model initialized in init_detector function. See init_detector function.
@param id_focused_on_lp : unique id to a model initialized in init_detector function. See init_detector function.
@param lpn_len : length of the preallocated c string to store the resulting license plate number.
@param lpn : the resulting license plate number as a c string, allocated by the calling program.
		@return
		@see
		*/
#ifdef _WINDOWS
	__declspec(dllimport)
#else //_WINDOWS
#endif //_WINDOWS
	bool detect_with_lpn_detection
	(const int width,//width of image
		const int height,//height of image i.e. the specified dimensions of the image
		const int pixOpt,// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
		void* pbData,
		size_t step// source image bytes buffer
		, size_t id_global_view, size_t id_focused_on_lp, size_t lpn_len, char* lpn);
	/**
		@brief detect lpn in frame. this function is equivalent to detect_with_lpn_detection (same result and same arguments).
		It uses just one model instead of two if you opt for detect_with_lpn_detection.
		The repo comes with two models namely lpreditor_anpr_focused_on_lp and lpreditor_anpr_global_view.
		If you use this function, make sure you have initialized the lpreditor_anpr_global_view model in the init_detector function. (see sample_cpp for uses examples).
		@param width : width of source image
		@param height : height of source image
		@param pixOpt : pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
		@param *pbData : source image bytes buffer
		@param step Number of bytes each matrix row occupies. The value should include the padding bytes at
			the end of each row, if any..See sample_cpp for a use case.
@param id: unique id to a model initialized in init_detector function. See init_detector function.
@param lpn_len : length of the preallocated c string to store the resulting license plate number.
@param lpn : the resulting license plate number as a c string, allocated by the calling program.
		@return
		@see
		*/
#ifdef _WINDOWS
	__declspec(dllimport)
#else //_WINDOWS
#endif //_WINDOWS
	bool detect_without_lpn_detection
	(const int width,//width of image
		const int height,//height of image i.e. the specified dimensions of the image
		const int pixOpt,// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
		void* pbData, size_t step// source image bytes buffer
		, size_t id, size_t lpn_len, char* lpn);
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
	bool close_detector(size_t id);
#ifdef __cplusplus
}
#endif
#else 
//__declspec(dllexport)
#endif //LPREDITOR_EXPORTS
#endif // !defined(OPEN_LPREDITOR_LIB_H)
