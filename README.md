# LPReditor_ANPR_Lib
C library that performs automatic license plate recognition

- [Build](#build)
    + [(Common) Step 1 : Install ![OpenCV](https://github.com/opencv/opencv)](#-common--step-1---install---opencv--https---githubcom-opencv-opencv-)
  * [On Windows :](#on-windows--)
    + [Step 2 : ![onnxruntime-win-x64-1.4.0](https://github.com/microsoft/onnxruntime/releases)](#step-2-----onnxruntime-win-x64-140--https---githubcom-microsoft-onnxruntime-releases-)
    + [Step 3 : modify CMakeLists.txt](#step-3---modify-cmakeliststxt)
    + [Step 4 : cmake](#step-4---cmake)
    + [Step 4 : build solution in Visual Studio](#step-4---build-solution-in-visual-studio)
  * [On Linux :](#on-linux--)
    + [Step 2 : ![onnxruntime-linux-x64-1.6.0](https://github.com/microsoft/onnxruntime/releases)](#step-2-----onnxruntime-linux-x64-160--https---githubcom-microsoft-onnxruntime-releases-)
    + [Step 3 : modify CMakeLists.txt](#step-3---modify-cmakeliststxt-1)
    + [Step 4 : cmake](#step-4---cmake-1)
    + [Step 4 : make in the build LPReditor_ANPR/build dir](#step-4---make-in-the-build-lpreditor-anpr-build-dir)
- [Integrating to your c++ code](#integrating-to-your-c---code)
- [Deep learning model file](#deep-learning-model-file)
- [More detailed description](#more-detailed-description)
- [Third party software](#third-party-software)
  * [c++ inference (present code)](#c---inference--present-code-)
    + [![OpenCV 4.5.0 and higher](https://github.com/opencv/opencv)](#--opencv-450-and-higher--https---githubcom-opencv-opencv-)
    + [![ONNXRuntime](https://github.com/microsoft/onnxruntime)](#--onnxruntime--https---githubcom-microsoft-onnxruntime-)
  * [model trained with the use of :](#model-trained-with-the-use-of--)
    + [![YOLOv5](https://github.com/ultralytics/yolov5)](#--yolov5--https---githubcom-ultralytics-yolov5-)
    + [![ONNX](https://github.com/onnx/onnx)](#--onnx--https---githubcom-onnx-onnx-)
- [License](#license)


  
*Deep learning number plate recognition engine, based on ![YOLOv5](https://github.com/ultralytics/yolov5) and ![ONNX](https://github.com/onnx/onnx). Operates on latin characters.*

# C API
This C library is a C API, that allows to recognize license plate numbers in images. It is designed to use without pain, since the number of exported functions is strictly limited (only 3 functions). It exposes no structs (and of course no C++ classes). No need of any tuning also. It is ready to operate, on any latin license plate number image. Furthermore, it relies on standard technologies, that make it possible, to (build and) deploy on many platforms. Lastly, the library supports multithreading.
## Building the API
The code is standard c++ and relies on ![OpenCV](https://github.com/opencv/opencv) and ![ONNXRuntime](https://github.com/microsoft/onnxruntime). These two softwares are meant to operate on a vast range of hardwares and os. Based on that, it should be possible to build on various platforms. Among them, I tested successfully Windows 10 and Linux Ubuntu (20.04). Use of CUDA has not (yet) been tested (only CPU). 
#### (Common) Step 1 : Install ![OpenCV](https://github.com/opencv/opencv)
### On Windows :
#### Step 2 : ![onnxruntime-win-x64-1.4.0](https://github.com/microsoft/onnxruntime/releases)
Download onnxruntime-win-x64-1.4.0.zip and decompress somewhere
#### Step 3 : modify CMakeLists.txt
In LPReditor_ANPR/CMakeLists.txt, change ../onnxruntime-win-x64-1.4.0/ to point to the actual path of the onnxruntime-win-x64-1.4.0 directory
#### Step 4 : cmake
From cmake-gui, configure and generate LPReditor_ANPR/CMakeLists.txt 
#### Step 5 : build solution in Visual Studio

### On Linux :
#### Step 2 : ![onnxruntime-linux-x64-1.6.0](https://github.com/microsoft/onnxruntime/releases)
Download onnxruntime-linux-x64-1.6.0.tgz and decompress somewhere
#### Step 3 : modify CMakeLists.txt
In LPReditor_ANPR/CMakeLists.txt, change ../onnxruntime-linux-x64-1.6.0/ to point to the actual path of the onnxruntime-linux-x64-1.6.0 directory
#### Step 4 : cmake
From cmake-gui, configure and generate LPReditor_ANPR/CMakeLists.txt 
#### Step 5 : make in the build LPReditor_ANPR/build dir
## Calling the API in your code
The use of the library is pretty straighforward and decomposes in three distinct steps. At first, engine initialization, via calling the function *init_session*. It initializes a new detector, by loading its model file and returns a (unique) id. 
This id must be passed, as a parameter, to the two others functions. Second, call the *detect* function, to recognize license plates in images. Parameters of the *detect* function are :
- the id returned by *init_session*.
- 4 parameters, to access the image, (preloaded) in memory.
- a pointer to a (preallocated) c string (filled, in return, with the license plate number string)

Third, when reading images is ended, call the *close_session* to free the memory, consumed by the detector (important : pass, as parameter, the id returnned by *init_session*). 
```javascript

//step 1 : initializes a new detector, by loading its model file. In return, you get a unique id.
//file path of the model
//the model file is in the repo under /data/models/lpreditor_anpr.zip, due to its size (Github limits file size). It must be dezipped to lpreditor_anpr.onnx,       after cloning the repo.
std::string model_filename = "The/path/to/the/model/that/is/in/repo/lpreditor_anpr.onnx";
size_t len = model_filename.size();
size_t id = init_session(len, model_filename.c_str());
```
```javascript
//step 2 : detect lpn in frame    
//allocates a c string to store the read lpn
const size_t lpn_len = 15;
char lpn[lpn_len] = "\0";
//the code below, comes from sample_cpp (in repo) and frame is cv::Mat image instance.
bool detected = detect
(frame.cols,//width of image
frame.rows,//height of image i.e. the specified dimensions of the image
frame.channels(),// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
frame.data, step// source image bytes buffer
, id,//id : unique interger to identify the detector to be used
lpn_len, lpn//lpn : a c string allocated by the calling program
);
std::cout << lpn;
```
```javascript
//step 3: call this func once you have finished with the detector-- > to free memeory
bool session_closed = close_session(id//id : unique interger to identify the detector to be freed
);
```
## API Documentation
### *init_session*
```javascript
/**
	@brief initializes a new detector, by loading its model file and returns its unique id
	@param model_file : c string model filename 
	@param len : length of the model filename
	@return the id of the new detector
	@see
	*/
extern "C"
#ifdef _WINDOWS
__declspec(dllexport)
#endif //_WINDOWS
size_t init_session(size_t len, const char* model_file)

```

### *detect*
```javascript
/**
	@brief detect lpn in frame
	@param int width : width of source image
	@param height : height of source image
	@param pixOpt : pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
	@param *pbData : source image bytes buffer
	param step Number of bytes each matrix row occupies.The value should include the padding bytes at
	the end of each row, if any.If the parameter is missing(set to AUTO_STEP), no padding is assumed
	and the actual step is calculated as cols* elemSize().See Mat::elemSize.
	param id : unique interger to identify the detector to be used
	@param lpn: a c string allocated by the calling program
	@return true upon success
	*/
extern "C"
#ifdef _WINDOWS
__declspec(dllexport)
#endif //_WINDOWS
bool detect
(const int width,//width of image
	const int height,//height of image i.e. the specified dimensions of the image
	const int pixOpt,// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
	void* pbData, size_t step// source image bytes buffer
	, size_t id, size_t lpn_len, char* lpn)
	
```


### *close_session*
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
bool close_session(size_t id)

```
# sample_cpp
The repo comes with a example, called sample_cpp. It needs ![OpenCV](https://github.com/opencv/opencv) to load images, so you first need to install it.
## Building sample_cpp
The easiest way is to use cmake (since sample_cpp comes with CMakeLists.txt file), to configure and generate the solution. This way, you make sure that the projrct links to the LPReditor_ANPR_Lib library (on windows files LPReditor_ANPR_Lib.lib + LPReditor_ANPR_Lib.dll or, on linux, file libLPReditor_ANPR_Lib.so). 

Building will produce an executable, with command line options (see them in the sample_cpp.cpp file or [below](#Command_line_syntax)). It can read lpn(s) from a single image file or alternatively, from multiple image files, in a common directory. If the actual license plate number is provided (see func getTrueLPN in the code), in the image filename, then statistics of the correctness of the readings, are available. 

---
&nbsp;
![highgui](https://github.com/lprsoft/LPReditor_ANPR_Lib/data/screenshot.jpg).
&nbsp;

## Binary release
Note that the binary release, sample_cpp, is available in the repo :
<a name="Install_dir_on_windows">
- on windows sample_cpp.exe under LPReditor_ANPR_Lib/build/Debug or LPReditor_ANPR_Lib/build/Release 
<a name="Install_dir_on_linux">
- on Linux sample_cpp LPReditor_ANPR_Lib/sample_cpp

As said, to run the binary, you have to :
- install OpenCV (and, on windows, to copy the opencv*.dlls in [installation dir](#Install_dir_on_windows)
- copy also the onnxruntime library in the [installation dir](#Install_dir_on_linux)

<a name="Command_line_syntax">
	
## Command line syntax
Below is the syntax : 

```javascript

sample_cpp -model path/to/lpreditor_anpr.onnx [-image path/to/your/image/file][-dir path/to/your/image/dir]" << std::endl;
```
Example :

```javascript
sample_cpp -model ../../data/models/lpreditor_anpr.onnx -image ../../data/images/images test/0000000001_3065WWA34.jpg -dir ../../data/images/images test
```

# Deep learning model file
Mandatory : to operate, the executable must load the model file. You can download the model : due to its size, the lpreditor_anpr.onnx file in the repo, as a zipped file (/data/models/lpreditor_anpr.zip). Another option, is to train your own model, on your dataset, using ![YOLOv5](https://github.com/ultralytics/yolov5) and then ![export](https://github.com/ultralytics/yolov5/issues/251) it.


 


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

# License
All files, including the deep learning model file provided, are subject to GNU General Public License v3.0.

Commercial-friendly licensing available.


