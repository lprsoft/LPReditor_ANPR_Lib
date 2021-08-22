# LPReditor_ANPR_Lib
C library that performs license plate recognition.
  
*Deep learning number plate recognition engine, based on ![YOLOv5](https://github.com/ultralytics/yolov5) and ![ONNX](https://github.com/onnx/onnx). Operates on any latin license plate.*
- [LPReditor_ANPR_Lib](#lpreditor_anpr_lib)
- [C API](#c-api)
	- [Building the API](#building-the-api)
			- [(Common) Step 1 : Install !OpenCV and CUDA & cuDNN (Optional but recommended if you want to use CUDA Execution Provider)](#common-step-1--install--and-cuda--cudnn-optional-but-recommended-if-you-want-to-use-cuda-execution-provider)
		- [On Windows :](#on-windows-)
			- [Step 2 : !onnxruntime-win-x64-1.x.0](#step-2--)
			- [Step 3 : modify CMakeLists.txt](#step-3--modify-cmakeliststxt)
			- [Step 4 : cmake](#step-4--cmake)
			- [Step 5 : build solution in Visual Studio](#step-5--build-solution-in-visual-studio)
		- [On Linux :](#on-linux-)
			- [Step 2 : !onnxruntime-linux-x64-1.x.0](#step-2---1)
			- [Step 3 : modify CMakeLists.txt](#step-3--modify-cmakeliststxt-1)
			- [Step 4 : cmake](#step-4--cmake-1)
			- [Step 5 : make in the build LPReditor_ANPR/build dir](#step-5--make-in-the-build-lpreditor_anprbuild-dir)
	- [Usage](#usage)
	- [API Documentation](#api-documentation)
		- [*init_detector*](#init_detector)
		- [*detect_with_lpn_detection*](#detect_with_lpn_detection)
		- [*close_detector*](#close_detector)
- [sample_cpp](#sample_cpp)
	- [Building sample_cpp](#building-sample_cpp)
	- [Binary release](#binary-release)
	- [Command line syntax](#command-line-syntax)
- [Deep learning model files](#deep-learning-model-files)
- [Third party software](#third-party-software)
	- [API (present code)](#api-present-code)
		- [!OpenCV](#)
		- [!ONNXRuntime](#-1)
	- [model trained with the use of :](#model-trained-with-the-use-of-)
		- [!YOLOv5](#-2)
		- [!ONNX](#-3)
	- [dataset of images that you can test with sample_cpp :](#dataset-of-images-that-you-can-test-with-sample_cpp-)
		- [!openalpr /benchmarks](#-4)
- [License](#license)
- [Tests](#tests)
# C API
This C library is a C API. It is designed to be used without pain, the number of exported functions is limited to three. It exposes no classes. No need of any tuning also. It is ready to operate, on any latin license plate number image. [Accuracy and speed](#tests) are enough for most commercial applications. [Reproducibly](#tests) tests are available. The library supports multithreading.
## Building the API
The code is standard c++ and relies on ![ONNXRuntime](https://github.com/microsoft/onnxruntime). Based on that, it is possible to build on various platforms, with our without CUDA. Among them, I tested successfully Windows 10 and Linux Ubuntu (20.04). 

#### (Common) Step 1 : Install ![OpenCV](https://github.com/opencv/opencv) and CUDA & cuDNN (Optional but recommended if you want to use CUDA Execution Provider)

The installation process of CUDA is quite straightforward. You can Install CUDA v11.0 from 
![here](https://developer.nvidia.com/cuda-11.0-download-archive). Next, install cuDNN by downloading the installer from ![here](https://developer.nvidia.com/rdp/cudnn-archive)
### On Windows :
#### Step 2 : ![onnxruntime-win-x64-1.x.0](https://github.com/microsoft/onnxruntime/releases)
Download onnxruntime-win-x64-1.x.0.zip and decompress somewhere. 
I used onnxruntime-...-1.8.1, and noticed that I had to change an include directive, in the file cpu_provider_factory.h (line 7), *#include "core/framework/provider_options.h"* to *#include "provider_options.h"* .
#### Step 3 : modify CMakeLists.txt
In LPReditor_ANPR/CMakeLists.txt, change ../onnxruntime-win-x64-.../ to point to the actual path of the onnxruntime-win-x64... directory
#### Step 4 : cmake
From cmake-gui, configure and generate LPReditor_ANPR/CMakeLists.txt 
#### Step 5 : build solution in Visual Studio

### On Linux :
#### Step 2 : ![onnxruntime-linux-x64-1.x.0](https://github.com/microsoft/onnxruntime/releases)
Download onnxruntime-linux-x64-1.x.0.tgz and decompress somewhere
#### Step 3 : modify CMakeLists.txt
In LPReditor_ANPR/CMakeLists.txt, change ../onnxruntime-linux-x64-.../ to point to the actual path of the onnxruntime-linux-x64-... directory
#### Step 4 : cmake
From cmake-gui, configure and generate LPReditor_ANPR/CMakeLists.txt 
#### Step 5 : make in the build LPReditor_ANPR/build dir
## Usage
The use of the library is pretty straighforward and decomposes in three distinct steps. At first, engine initialization, via calling the function *init_detector*. It initializes a new detector, by loading its model file and returns a (unique) id. 
This id must be passed, as a parameter, to the two others functions. Second, call the *detect_with_lpn_detection* function, to recognize license plates in images. Parameters of the *detect_with_lpn_detection* function are :

- 4 parameters, to access the image, (preloaded) in memory.
- the ids of models returned by *init_detector*.
- a pointer to a (preallocated) c string (filled, in return, with the license plate number string)

Third, when reading images is ended, call the *close_detector* to free the memory, consumed by the detector (important : pass, as parameter, the id returnned by *init_detector*). 
```javascript


std::string model_filename_global_view = "The/path/to/the/model/that/is/in/repo/lpreditor_anpr_global_view.onnx";
size_t len = model_filename_global_view.size();

//step 1 : Initializes a new detector by loading its model file. In return, you get a unique id. The repo comes with two models namely lpreditor_anpr_focused_on_lpand lpreditor_anpr_global_view.
//So you have to call this function twice to initialize both models.
size_t id_global_view = init_detector(len, model_filename_global_view.c_str());
if (id_global_view > 0) {
	std::cout << "\n global_view Model loaded succesfully\n" << std::endl;
}
else {
	std::cerr << "\n global_view Model not loaded error\n" << std::endl;
	return;
}
model_filename_global_view = "The/path/to/the/model/that/is/in/repo/lpreditor_anpr_focused_on_lp.onnx";
len = model_filename_global_view.size();
size_t id_focused_on_lp = init_detector(len, model_filename_focused_on_lp.c_str());
if (id_focused_on_lp > 0) {
	std::cout << "\n focused_on_lp Model loaded succesfully\n" << std::endl;
}
else {
	std::cerr << "\n focused_on_lp Model not loaded error\n" << std::endl;
	id_focused_on_lp = id_global_view;
}




```
```javascript
//step 2 : detect lpn in frame    
//allocates a c string to store the read lpn
const size_t lpn_len = 15;
char lpn[lpn_len] = "\0";
//the code below, comes from sample_cpp (in repo) and frame is cv::Mat image instance.
bool detected = detect_with_lpn_detection
(frame.cols,//width of image
frame.rows,//height of image i.e. the specified dimensions of the image
frame.channels(),// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
frame.data, 
frame.step// source image bytes buffer
, id_global_view,id_focused_on_lp,//id : unique interger to identify the detector to be used
lpn_len, lpn//lpn : a c string allocated by the calling program
);
std::cout << lpn;
```
```javascript
//step 3: call this func once you have finished with the detector-- > to free memeory
bool session_closed = close_detector(id//id : unique interger to identify the detector to be freed
);
```
## API Documentation
### *init_detector*
```javascript
/**
	@brief initializes a new detector, by loading its model file and returns its unique id. The repo comes with two models namely lpreditor_anpr_focused_on_lp and lpreditor_anpr_global_view. So you have to call this function twice to initialize both models (note that it is possible, but not a good idea, to initialize just one model).
	@param model_file : c string model filename 
	@param len : length of the model filename
	@return the id of the new detector
	@see
	*/
extern "C"
#ifdef _WINDOWS
__declspec(dllexport)
#endif //_WINDOWS
size_t init_detector(size_t len, const char* model_file)

```


### *detect_with_lpn_detection*
```javascript
/**
	@brief detect lpn in frame. This function uses a two stage detection method that requires two models.
	please make sure you have initialized the lpreditor_anpr_global_view model in the init_detector function (see sample_cpp for uses examples).
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
	@return true upon success
	*/
extern "C"
#ifdef _WINDOWS
__declspec(dllexport)
#endif //_WINDOWS
bool detect_with_lpn_detection
(const int width,//width of image
	const int height,//height of image i.e. the specified dimensions of the image
	const int pixOpt,// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
	void* pbData, size_t step// source image bytes buffer
	, size_t id, size_t lpn_len, char* lpn)
	
```


### *close_detector*
```javascript
/**
	@brief call this func once you have finished with the detector --> to free heap allocated memeory
	@param id : unique interger to identify the detector to be freed
	@return true upon success
	@see
	*/
extern "C"
#ifdef _WINDOWS
__declspec(dllexport)
#endif //_WINDOWS
bool close_detector(size_t id)

```
# sample_cpp
The repo comes with a example, called sample_cpp. It needs ![OpenCV](https://github.com/opencv/opencv) to load images.
## Building sample_cpp
The easiest way is to use cmake (since sample_cpp comes with CMakeLists.txt file), to configure and generate the solution. This way, you make sure that the projrct links to the LPReditor_ANPR_Lib library (on windows files LPReditor_ANPR_Lib.lib + LPReditor_ANPR_Lib.dll or, on linux, file libLPReditor_ANPR_Lib.so). 

Building will produce an executable, with command line options (see them in the sample_cpp.cpp file or [below](#Command_line_syntax)). It can read lpn(s) from a single image file or alternatively, from multiple image files, in a common directory. If the actual license plate number is provided (see func getTrueLPN in the code), in the image filename, then statistics of the correctness of the readings, are available. 

---
&nbsp;

## Binary release
Note that the binary release, sample_cpp, is available in the repo :

## Command line syntax
Below is the syntax : 

```javascript

sample_cpp -global_view_model=path/to/lpreditor_anpr_global_view.onnx -focused_on_lp_model=path/to/lpreditor_anpr_focused_on_lp.onnx [-image=path/to/your/image/file][-dir=path/to/your/image/dir]

```
Example :

```javascript
sample_cpp -focused_on_lp_model=../../data/models/lpreditor_anpr_focused_on_lp.onnx  -global_view_model=../../data/models/lpreditor_anpr_global_view.onnx -image=../../data/images/images_test/0000000001_3065WWA34.jpg
```

# Deep learning model files
Mandatory : to operate, the executable must load the model file(s). Another option, is to train your own model, on your dataset, using ![YOLOv5](https://github.com/ultralytics/yolov5) and then ![export](https://github.com/ultralytics/yolov5/issues/251) it.



# Third party software

## API (present code)

### ![OpenCV](https://github.com/opencv/opencv)
Copyright © 2021 , OpenCV team
Apache 2 License

### ![ONNXRuntime](https://github.com/microsoft/onnxruntime)
Copyright © 2020 Microsoft. All rights reserved.
MIT License

## model trained with the use of :

### ![YOLOv5](https://github.com/ultralytics/yolov5)

by Glenn Jocher (Ultralytics.com)
GPL-3.0 License

### ![ONNX](https://github.com/onnx/onnx)
Copyright (c) Facebook, Inc. and Microsoft Corporation. All rights reserved.
MIT License

## dataset of images that you can test with sample_cpp :
### ![openalpr /benchmarks](https://github.com/openalpr/benchmarks)
Benchmark tests supporting the openalpr , 
Affero GPLv3 http://www.gnu.org/licenses/agpl-3.0.html


# License
All files, including the deep learning model file provided, are subject to GNU General Public License v3.0.

Commercial-friendly licensing available (furthermore commercial version includes more models).



# Tests

The the openalpr benchmark is made of 455 images, originally split in three different sets (brazilian, eu, us). I mergesd theses sets in a single directory, called plate_un. Furthermore I renamed the images, in order that, the license plate number is shown in the filename. 

To be fair, none of these images has been used in training/testing phase of this engine dev (by the way, the engine has not been trained with any brazilian or american lps).

| Origin  | command line          | Score : Exact readings| Score : readings with at most one error on one character | Speed (im/s) with ONNX provided with CUDA Execution (Nvidia GeForce RTX 2060S)|
| :--------------- |:---------------| :-----:| :-----:| :-----:|
| brazil + eu + us |   sample_cpp -focused_on_lp_model=../../data/models/lpreditor_anpr_focused_on_lp.onnx -global_view_model=../../data/models/lpreditor_anpr_global_view.onnx -dir=../../data/images/benchmarks-master/endtoend/plate_un        |  0.79 |  0.94 | 4 im/s |

One error on one character means that the numberplate, read by the engine, differs to the actual numberplate, with at most one character : either the engine misses a character, either it detects a charcater where there is nothing, either it missreads a character, either it reads correctly the numberplate.

You can make the test yourself, using the build/release sample_cpp executable (with command line args shown above). 


