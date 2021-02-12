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

#if !defined(ONNX_RUNTIME_DETECTOR_H)
#define ONNX_RUNTIME_DETECTOR_H


# pragma once

//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/dnn/dnn.hpp>


#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <functional>

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>
#include "utils_anpr_detect.h"
class OnnxDetector {
public:
    /***
     * @brief constructor
     * @param model_path - path of the TorchScript weight file
     */
    OnnxDetector(Ort::Env& env, const void* model_data, size_t model_data_length, const Ort::SessionOptions& options);
    OnnxDetector(Ort::Env& env, const ORTCHAR_T* model_path, const Ort::SessionOptions& options);


    void dump() const;
    /***
     * @brief loads moodel file
     * @param model_path - path of the TorchScript weight file
     */
    //bool OnnxDetector::load(const std::string& model_path);



    /***
     * @brief inference module
     * @param img - input image
     * @param conf_threshold - confidence threshold
     * @param iou_threshold - IoU threshold for nms
     * @return detection result - bounding box, score, class index
     */
    std::vector<std::vector<Detection>>
    Run(const cv::Mat& img, float conf_threshold, float iou_threshold, bool preserve_aspect_ratio);




    std::vector<std::tuple<cv::Rect, float, int>>
        Run_v1(const cv::Mat& img, float conf_threshold, float iou_threshold);
private:




protected :
    Ort::SessionOptions sessionOptions;
    Ort::Session session;
    const Ort::Env& env;
#ifdef LPR_EDITOR_USE_CUDA
    bool useCUDA;
#endif //LPR_EDITOR_USE_CUDA
};



void nms(const std::vector<cv::Rect>& srcRects, std::vector<cv::Rect>& resRects, std::vector<int>& resIndexs, float thresh);

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

#endif // !defined(ONNX_RUNTIME_DETECTOR_H)