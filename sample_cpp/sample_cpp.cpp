// test_LPReditor_ANPR_Lib.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include "Open_LPReditor_Lib.h"

#include <string>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "../include/utils_image_file.h"

void detect_one_image(const std::string& image_filename, const std::string & model_filename) {
	int flags = -1;//as is
	cv::Mat frame = cv::imread(image_filename, flags);
	int channels_ = frame.channels();
	if (frame.size().width &&
		frame.size().height && ((channels_ == 1) || (channels_ == 3) || (channels_ == 4))
		&& (frame.type() == CV_8UC1 || frame.type() == CV_8UC3 || frame.type() == CV_8UC4)) {
		size_t step = 0;
		//file path of the model
		//std::string model_filename = "D:/Programmation/LPReditor-engine/LPReditor_ANPR_Lib/data/models/lpreditor_anpr.onnx";
		size_t len = model_filename.size();
		//allocates a c string to store the read lpn
		const size_t lpn_len = 15;
		char lpn[lpn_len] = "\0";
		//step 1
//initializes a new detector by loading its model file and gets its unique id
		size_t id = init_session(len, model_filename.c_str());
		//step 2 
		//detect lpn in frame
		bool detected = detect
		(frame.cols,//width of image
			frame.rows,//height of image i.e. the specified dimensions of the image
			frame.channels(),// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
			frame.data, step// source image bytes buffer
			, id,//id : unique interger to identify the detector to be used
			lpn_len, lpn//lpn : a c string allocated by the calling program
		);
		std::cout << lpn;
		//step 3
		//call this func once you have finished with the detector-- > to free heap allocated memeory
		bool session_closed = close_session(id//id : unique interger to identify the detector to be freed
		);
	}
}

void detect_one_directory(const std::string& dir, const std::string & model_filename) {

#ifdef LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
	//std::filesystem::path p(std::filesystem::current_path());
	//std::string filename = p.string()+"/test_svm.txt";
	std::string filename = "D:\\Programmation\\LPReditor\\ocr_dataset\\test_svm.txt";
	
	std::ofstream O(filename.c_str(), std::ios::app);
	O << "Yolov5_anpr_onxx_detector::detect " << std::endl;
#endif //LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
	//file path of the model
	//std::string model_filename = "D:/Programmation/LPReditor-engine/LPReditor_ANPR_Lib/data/models/lpreditor_anpr.onnx";
	size_t len = model_filename.size();
	//step 1
//initializes a new detector by loading its model file and gets its unique id
	size_t id = init_session(len, model_filename.c_str());
	std::list<std::string> image_filenames;
	//extracts from a test directory all images files that come with an xml file containing the bb coordinates in this image
	load_images_filenames(dir, image_filenames);
	std::list<std::string>::const_iterator it_image_filenames(image_filenames.begin());
	
	int c = 0;
	int less_1_editdistance_reads = 0;
	int miss_reads = 0;
	int good_reads = 0;
	while (it_image_filenames != image_filenames.end())
	{
		//allocates a c string to store the read lpn
		const size_t lpn_len = 15;
		char lpn[lpn_len] = "\0";
		int flags = -1;//as is
		cv::Mat frame = cv::imread(*it_image_filenames, flags);
		int channels_ = frame.channels();
		if (frame.size().width &&
			frame.size().height && ((channels_ == 1) || (channels_ == 3) || (channels_ == 4))
			&& (frame.type() == CV_8UC1 || frame.type() == CV_8UC3 || frame.type() == CV_8UC4)) {
			size_t step = 0;
			//step 2 
			//detect lpn in frame
			bool detected = detect
			(frame.cols,//width of image
				frame.rows,//height of image i.e. the specified dimensions of the image
				frame.channels(),// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
				frame.data, step// source image bytes buffer
				, id,//id : unique interger to identify the detector to be used
				lpn_len, lpn//lpn : a c string allocated by the calling program
			);
			std::string lpn_str(lpn);


			std::filesystem::path p_(*it_image_filenames);
			bool vrai_lpn_after_underscore = true;
			//returns the true license plate number out of a filename
				//you must place the true license plate number in the image filename this way : number + underscore + license plate number,
				//for instance filename 0000000001_3065WWA34.jpg will be interpreted as an image with the license plate 3065WWA34 in it.
			std::string ExactLPN(getTrueLPN(p_.stem().string(), vrai_lpn_after_underscore));
			Levenshtein lev;
			int editdistance = lev.Get(ExactLPN.c_str(), ExactLPN.length(), lpn_str.c_str(), lpn_str.length());
			std::cout << "ExactLPN : " << ExactLPN << " read LPN : " << lpn << "edit distance: " << editdistance << std::endl;
			if (editdistance > 0) miss_reads++;
			else good_reads++;
			if (editdistance <= 1) less_1_editdistance_reads++;
			it_image_filenames++; c++;
			if ((c % 1
				//00
				) == 0) {
				std::cout << c << " perc good reads:" << (float)(good_reads) / (float)(c) << std::endl;
				std::cout << c << " perc reads less 1 edit distance:" << (float)(less_1_editdistance_reads) / (float)(c) << std::endl;
#ifdef LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
				O << c  << " perc good reads:" << (float)(good_reads) / (float)(c) << std::endl;
				O << c << " perc reads less 1 edit distance:" << (float)(less_1_editdistance_reads) / (float)(c) << std::endl;
#endif //LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
			}
		}
	}
	//step 3
		//call this func once you have finished with the detector-- > to free heap allocated memeory
	bool session_closed = close_session(id//id : unique interger to identify the detector to be freed
	);
}

static void help(char** argv)
{
	std::cout << "\nThis program demonstrates the automatic numberplate recognition software named LPReditor\n"
		"Usage:\n" << argv[0] << "\n--model = path to the model *.pt file\n"
		<< "[--image = path to your image file (if you opt to process just one image) ]\n"
		<< "[--dir = path to a directory containing images files (if you opt to process all the images in the same directory)]\n"
		<< "[--show], whether to show the image in a window with license plate in banner\n"
		<< "[--time_delay= time delay in ms between two consecutive images]\n" << std::endl;
	std::cout << "Note : model.pt file is in the package" << std::endl;
	std::cout << "Note : options [--image ] and [--dir ] are incompatible, model argument is mandatory" << std::endl;
	std::cout << "Note : if you want to see how well the engine performs, you must place the true license plate number in the image filename this way : number+underscore+license plate number\n"
		<< "for instance filename 0000000001_3065WWA34.jpg will be interpreted as an image with the license plate 3065WWA34 in it." << std::endl;
}
int main(int argc, char* argv[])
{
#ifdef LPREDITOR_DEMO_NO_ARGS
	const int argc_ = 4;
	char* argv_[argc_];
	/*
	argv_[0] = argv[0];
	argv_[1] = "--image=D:\\my\\path\\to\\an\\image.jpg";//
	argv_[2] = "--model=D:\\my\\path\\to\\the\\yolo\\model\\yolo_carac_detect.pt ";//the yolo model file is provided in the repo
	argv_[3] = "--dir=D:\\my\\path\\to\\a\\directory\\with\\image\\files";
	
	*/
	argv_[0] = argv[0];
	argv_[1] = "--image=../../data/images/images test/0000000001_3065WWA34.jpg";//
	argv_[2] = "--model=../../data/models/lpreditor_anpr.onnx";
	argv_[3] = "--dir=../../data/images/images test";
	cv::CommandLineParser parser(argc_, argv_, "{help h | | }{model | | }{image | | }{dir | | }");
#else //LPREDITOR_DEMO_NO_ARGS
	cv::CommandLineParser parser(argc, argv, "{help h | | }{model | | }{image | | }{dir | | }");
#endif //LPREDITOR_DEMO_NO_ARGS
	if (parser.has("help"))
	{
		help(argv);
		return 0;
	}
	
	if (!parser.has("model"))
	{
		std::cout << "\nYou must specify the model pathname by using mandatory arg --model=...\n" << std::endl;
		help(argv);
		return 0;
	}
	std::string model_filename = (parser.get<std::string>("model"));
	if (!model_filename.size())
	{
		std::cout << "\nCan't find the model file\n" << std::endl;
		help(argv);
		return 0;
	}
	else {
		
		std::cout << "\nModel load succesfully\n" << std::endl;
		std::string image_filename = (parser.get<std::string>("image"));
		std::string dir = (parser.get<std::string>("dir"));
		if (!parser.has("image") && !parser.has("dir"))
		{
			std::cout << "\nYou must specify either an image filename or a directory (with images in it)\n" << std::endl;
			help(argv);
			return 0;
		}
		if (!dir.size() && image_filename.size())
		{
			detect_one_image(image_filename, model_filename); 
		}
		else {
			//process all images files of a directory
			//step 5 call the detect function of the Yolov5_anpr_onxx_detector object, on a cv::mat object or an image file.
			//This will retieves boxes and classes of the license plate caracters
			detect_one_directory(dir, model_filename);
		}
	}
	/*
	std::string dir = "D:/Programmation/LPReditor-engine/LPReditor_ANPR_Lib/data/images/images test";
	bool _detect_one_image = false;
	std::string image_filename = "D:/Programmation/LPReditor-engine/LPReditor_ANPR_Lib/data/images/0000000001_3065WWA34.jpg";
	
	if (_detect_one_image) {
		detect_one_image(image_filename);
	}
	else {
		detect_one_directory(dir);
	}
	*/
	return 0;
}
