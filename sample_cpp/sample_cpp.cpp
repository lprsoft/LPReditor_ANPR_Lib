// test_LPReditor_ANPR_Lib.cpp : Defines the entry point for the console application.
//
//#include "stdafx.h"
#include "Open_LPReditor_Lib.h"
#include <string>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "../include/utils_image_file.h"
void detect_one_image(const std::string& image_filename, const std::string& model_filename) {
	int flags = -1;//as is
	cv::Mat frame = cv::imread(image_filename, flags);
	int channels_ = frame.channels();
	if (frame.size().width &&
		frame.size().height && ((channels_ == 1) || (channels_ == 3) || (channels_ == 4))
		&& (frame.type() == CV_8UC1 || frame.type() == CV_8UC3 || frame.type() == CV_8UC4)) {
		size_t step = frame.step;
		//file path of the model
		//the model file is in the repo under /data/models/lpreditor_anpr.zip, due to its size (Github limits file size). It must be dezipped to lpreditor_anpr.onnx, after cloning the repo.
		std::string model_filename = "The/path/to/the/model/that/is/in/repo/lpreditor_anpr.onnx";
		size_t len = model_filename.size();
		//allocates a c string to store the read lpn
		const size_t lpn_len = 15;
		char lpn[lpn_len] = "\0";
		//step 1 : Initializes a new detector by loading its model file. In return, you get a unique id.
		size_t id = init_detector(len, model_filename.c_str());
		if (id > 0) {
			std::cout << "\nModel loaded succesfully\n" << std::endl;
		}
		else {
			std::cerr << "\nModel not loaded error\n" << std::endl;
			return;
		}
		//step 2 : detect_without_lpn_detection lpn in frame
		//the code below, comes from sample_cpp (in repo) and frame is cv::Mat image instance.
		bool detected = detect_without_lpn_detection
		(frame.cols,//width of image
			frame.rows,//height of image i.e. the specified dimensions of the image
			frame.channels(),// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
			frame.data, step// source image bytes buffer
			, id,//id : unique interger to identify the detector to be used
			lpn_len, lpn//lpn : a c string allocated by the calling program
		);
		std::cout << lpn;
		//step 3: call this func once you have finished with the detector-- > to free memeory
		bool session_closed = close_detector(id//id : unique interger to identify the detector to be freed
		);
	}
}
void detect_one_image(const std::string& image_filename, const std::string& model_filename_global_view
	, const std::string& model_filename_focused_on_lp) {
	int flags = -1;//as is
	cv::Mat frame = cv::imread(image_filename, flags);
	int channels_ = frame.channels();
	if (frame.size().width &&
		frame.size().height && ((channels_ == 1) || (channels_ == 3) || (channels_ == 4))
		&& (frame.type() == CV_8UC1 || frame.type() == CV_8UC3 || frame.type() == CV_8UC4)) {
		size_t step = frame.step;
		//file path of the model
		//the model file is in the repo under /data/models/lpreditor_anpr.zip, due to its size (Github limits file size). It must be dezipped to lpreditor_anpr.onnx, after cloning the repo.
		std::string model_filename = "The/path/to/the/model/that/is/in/repo/lpreditor_anpr.onnx";
		size_t len = model_filename.size();
		//allocates a c string to store the read lpn
		const size_t lpn_len = 15;
		char lpn[lpn_len] = "\0";
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
		size_t id_focused_on_lp = init_detector(len, model_filename_focused_on_lp.c_str());
		if (id_focused_on_lp > 0) {
			std::cout << "\n focused_on_lp Model loaded succesfully\n" << std::endl;
		}
		else {
			std::cerr << "\n focused_on_lp Model not loaded error\n" << std::endl;
			id_focused_on_lp = id_global_view;
		}
		//step 2 
		//detect_with_lpn_detection lpn in frame
		bool detected = detect_with_lpn_detection
		(frame.cols,//width of image
			frame.rows,//height of image i.e. the specified dimensions of the image
			frame.channels(),// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
			frame.data, step// source image bytes buffer
			, id_global_view, id_focused_on_lp,//id : unique interger to identify the detector to be used
			lpn_len, lpn//lpn : a c string allocated by the calling program
		);
		std::cout << lpn;
		//step 3
		//call this func once you have finished with the detector-- > to free heap allocated memeory
		bool session_closed = close_detector(id_global_view//id : unique interger to identify the detector to be freed
		);
		session_closed = close_detector(id_focused_on_lp//id : unique interger to identify the detector to be freed
		);
	}
}
void detect_one_directory(const std::string& dir, const std::string& model_filename) {
#ifdef LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
	std::filesystem::path p(std::filesystem::current_path());
	//std::string filename = p.string()+"/test_svm.txt";
	std::string filename = "D:\\Programmation\\LPReditor\\ocr_dataset\\test_svm.txt";
	std::ofstream O(filename.c_str(), std::ios::app);
	O << "Yolov5_anpr_onxx_detector::detect_without_lpn_detection " << std::endl;
#endif //LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
	//file path of the model
	//std::string model_filename = "D:/Programmation/LPReditor-engine/LPReditor_ANPR_Lib/data/models/lpreditor_anpr.onnx";
	size_t len = model_filename.size();
	//step 1
//step 1 : Initializes a new detector by loading its model file. In return, you get a unique id.
	size_t id = init_detector(len, model_filename.c_str());
	if (id > 0) {
		std::cout << "\nModel loaded succesfully\n" << std::endl;
	}
	else {
		std::cerr << "\nModel not loaded error\n" << std::endl;
		return;
	}
	std::list<std::string> image_filenames;
	//extracts, from a test directory, all images files that come with an xml file containing the bb coordinates in this image
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
			size_t step = frame.step;
			//step 2 
			//detect_without_lpn_detection lpn in frame
			bool detected = detect_without_lpn_detection
			(frame.cols,//width of image
				frame.rows,//height of image i.e. the specified dimensions of the image
				frame.channels(),// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
				frame.data, step// source image bytes buffer
				, id,//id : unique interger to identify the detector to be used
				lpn_len, lpn//lpn : a c string allocated by the calling program
			);
			std::string lpn_str(lpn);
			bool vrai_lpn_after_underscore = true;
			std::filesystem::path p_(*it_image_filenames);
			//returns the true license plate number out of a filename
				//you must place the true license plate number in the image filename this way : number + underscore + license plate number,
				//for instance filename 0000000001_3065WWA34.jpg will be interpreted as an image with the license plate 3065WWA34 in it.
			std::string ExactLPN(getTrueLPN(p_.stem().string(), vrai_lpn_after_underscore));
			Levenshtein lev;
			int editdistance = lev.Get(ExactLPN.c_str(), ExactLPN.length(), lpn_str.c_str(), lpn_str.length());
			std::cout << c << " true lpn :" << ExactLPN << " detected lpn :" << lpn_str << " editdistance :" << editdistance << std::endl;
			//std::cout << "ExactLPN : " << ExactLPN << " read LPN : " << lpn << "edit distance: " << editdistance << std::endl;
			if (editdistance > 0) miss_reads++;
			else good_reads++;
			if (editdistance <= 1) less_1_editdistance_reads++;
			c++;
			if ((c % 1000
				) == 0) {
				std::cout << c << " perc good reads:" << (float)(good_reads) / (float)(c) << std::endl;
				std::cout << c << " perc reads less 1 edit distance:" << (float)(less_1_editdistance_reads) / (float)(c) << std::endl;
#ifdef LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
				O << c << " perc good reads:" << (float)(good_reads) / (float)(c) << std::endl;
				O << c << " perc reads less 1 edit distance:" << (float)(less_1_editdistance_reads) / (float)(c) << std::endl;
#endif //LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
			}
		}
		it_image_filenames++;
	}
	std::cout << c << " perc good reads:" << (float)(good_reads) / (float)(c) << std::endl;
	std::cout << c << " perc reads less 1 edit distance:" << (float)(less_1_editdistance_reads) / (float)(c) << std::endl;
#ifdef LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
	O << c << " perc good reads:" << (float)(good_reads) / (float)(c) << std::endl;
	O << c << " perc reads less 1 edit distance:" << (float)(less_1_editdistance_reads) / (float)(c) << std::endl;
#endif //LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
	//step 3
		//call this func once you have finished with the detector-- > to free heap allocated memeory
	bool session_closed = close_detector(id//id : unique interger to identify the detector to be freed
	);
}
void detect_one_directory(const std::string& dir, const std::string& model_filename_global_view
	, const std::string& model_filename_focused_on_lp) {
#ifdef LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
	std::filesystem::path p(std::filesystem::current_path());
	//std::string filename = p.string()+"/test_svm.txt";
	std::string filename = "D:\\Programmation\\LPReditor\\ocr_dataset\\test_svm.txt";
	std::ofstream O(filename.c_str(), std::ios::app);
	O << "Yolov5_anpr_onxx_detector::detect_without_lpn_detection " << std::endl;
#endif //LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
	//file path of the model
	//std::string model_filename = "D:/Programmation/LPReditor-engine/LPReditor_ANPR_Lib/data/models/lpreditor_anpr.onnx";
	size_t len = model_filename_global_view.size();
	//step 1
//step 1 : Initializes a new detector by loading its model file. In return, you get a unique id.
	size_t id_global_view = init_detector(len, model_filename_global_view.c_str());
	if (id_global_view > 0) {
		std::cout << "\n global_view Model loaded succesfully\n" << std::endl;
	}
	else {
		std::cerr << "\n global_view Model not loaded error\n" << std::endl;
		return;
	}
	len = model_filename_focused_on_lp.size();
	size_t id_focused_on_lp = init_detector(len, model_filename_focused_on_lp.c_str());
	if (id_focused_on_lp > 0) {
		std::cout << "\n focused_on_lp Model loaded succesfully\n" << std::endl;
	}
	else {
		std::cerr << "\n focused_on_lp Model not loaded error\n" << std::endl;
		id_focused_on_lp = id_global_view;
	}
	std::list<std::string> image_filenames;
	//extracts, from a test directory, all images files that come with an xml file containing the bb coordinates in this image
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
			size_t step = frame.step;
			//step 2 
			//detect_with_lpn_detection lpn in frame
			bool detected = detect_with_lpn_detection
			(frame.cols,//width of image
				frame.rows,//height of image i.e. the specified dimensions of the image
				frame.channels(),// pixel type : 1 (8 bpp greyscale image) 3 (RGB 24 bpp image) or 4 (RGBA 32 bpp image)
				frame.data, step// source image bytes buffer
				, id_global_view, id_focused_on_lp,//id : unique interger to identify the detector to be used
				lpn_len, lpn//lpn : a c string allocated by the calling program
			);
			std::string lpn_str(lpn);
			bool vrai_lpn_after_underscore = true;
			std::filesystem::path p_(*it_image_filenames);
			//returns the true license plate number out of a filename
				//you must place the true license plate number in the image filename this way : number + underscore + license plate number,
				//for instance filename 0000000001_3065WWA34.jpg will be interpreted as an image with the license plate 3065WWA34 in it.
			std::string ExactLPN(getTrueLPN(p_.stem().string(), vrai_lpn_after_underscore));
			Levenshtein lev;
			int editdistance = lev.Get(ExactLPN.c_str(), ExactLPN.length(), lpn_str.c_str(), lpn_str.length());
			std::cout << c << " true lpn :" << ExactLPN << " detected lpn :" << lpn_str << " editdistance :" << editdistance << std::endl;
			//std::cout << "ExactLPN : " << ExactLPN << " read LPN : " << lpn << "edit distance: " << editdistance << std::endl;
			if (editdistance > 0) miss_reads++;
			else good_reads++;
			if (editdistance <= 1) less_1_editdistance_reads++;
			c++;
			if ((c % 1000
				) == 0) {
				std::cout << c << " perc good reads:" << (float)(good_reads) / (float)(c) << std::endl;
				std::cout << c << " perc reads less 1 edit distance:" << (float)(less_1_editdistance_reads) / (float)(c) << std::endl;
#ifdef LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
				O << c << " perc good reads:" << (float)(good_reads) / (float)(c) << std::endl;
				O << c << " perc reads less 1 edit distance:" << (float)(less_1_editdistance_reads) / (float)(c) << std::endl;
#endif //LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
			}
		}
		it_image_filenames++;
	}
	std::cout << c << " perc good reads:" << (float)(good_reads) / (float)(c) << std::endl;
	std::cout << c << " perc reads less 1 edit distance:" << (float)(less_1_editdistance_reads) / (float)(c) << std::endl;
#ifdef LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
	O << c << " perc good reads:" << (float)(good_reads) / (float)(c) << std::endl;
	O << c << " perc reads less 1 edit distance:" << (float)(less_1_editdistance_reads) / (float)(c) << std::endl;
#endif //LPREDITOR_DEMO_PRINT_STATS_IN_TXT_FILE
	//step 3
		//call this func once you have finished with the detector-- > to free heap allocated memeory
	bool session_closed = close_detector(id_global_view//id : unique interger to identify the detector to be freed
	);
	session_closed = close_detector(id_focused_on_lp//id : unique interger to identify the detector to be freed
	);
}
static void help(char** argv)
{
	std::cout << "\nThis program demonstrates the automatic numberplate recognition API named LPReditor_ANPR_Lib\n"
		"Usage:\n" << argv[0] << "\n--global_view_model = file path to the model lpreditor_anpr_global_view.onnx\n"
		<< "[--focused_on_lp_model = file path to the model lpreditor_anpr_focused_on_lp.onnx]\n"
		<< "[--image = file path to your image file (if you opt to process just one image) ]\n"
		<< "[--dir = path to a directory containing images files (if you opt to process all the images in the same directory)]\n"
		<< std::endl;
	std::cout << "Note : lpreditor_anpr_global_view.onnx and lpreditor_anpr_focused_on_lp.onnx files are in the github repo, as zipped files. They must be dezipped first." << std::endl;
	std::cout << "Note : options [--image ] and [--dir ] are incompatible, model argument is mandatory" << std::endl;
	std::cout << "Note : if you want to see how well the engine performs, you must place the true license plate number in the image filename this way : number+underscore+license plate number\n"
		<< "for instance filename 0000000001_3065WWA34.jpg will be interpreted as an image with the license plate 3065WWA34 in it." << std::endl;
	std::cout << "COMMAND LINE SYNTAX" << std::endl;
	std::cout << "sample_cpp -global_view_model=path/to/lpreditor_anpr_global_view.onnx -focused_on_lp_model=path/to/lpreditor_anpr_focused_on_lp.onnx [-image=path/to/your/image/file][-dir=path/to/your/image/dir]" << std::endl;
	std::cout << "EXAMPLE : on windows open command prompt change dir (cd) to the LPReditor_ANPR_Lib/build/Debug directory and prompt :" << std::endl;
	std::cout << "sample_cpp -focused_on_lp_model=../../data/models/lpreditor_anpr_focused_on_lp.onnx  -global_view_model=../../data/models/lpreditor_anpr_global_view.onnx -image=../../data/images/images test/0000000001_3065WWA34.jpg" << std::endl;
}
int main(int argc, char* argv[])
{
#ifdef LPREDITOR_DEMO_NO_ARGS
	const int argc_ = 5;
	char* argv_[argc_];

	argv_[0] = argv[0];
	argv_[1] = "--image=../../data/images/images test/0000000001_3065WWA34.jpg";//
	argv_[2] = "--global_view_model=../../data/models/lpreditor_anpr_global_view.onnx";
	argv_[3] = "--focused_on_lp_model=../../data/models/lpreditor_anpr_focused_on_lp.onnx";
	argv_[4] = "--dir=../../data/images/images test";
	argv_[4] = "--dir=../../data/images/benchmarks-master/endtoend/plate_br";
	argv_[4] = "--dir=../../data/images/benchmarks-master/endtoend/plate_eu";
	argv_[4] = "--dir=../../data/images/benchmarks-master/endtoend/plate_us";


	cv::CommandLineParser parser(argc, argv, "{help h | | }{global_view_model | | }{focused_on_lp_model | | }{image | | }{dir | | }");
#else //LPREDITOR_DEMO_NO_ARGS
	cv::CommandLineParser parser(argc, argv, "{help h | | }{global_view_model | | }{focused_on_lp_model | | }{image | | }{dir | | }");
#endif //LPREDITOR_DEMO_NO_ARGS
	if (parser.has("help"))
	{
		help(argv);
		return 0;
	}
	if (!parser.has("global_view_model"))
	{
		std::cout << "\nYou must specify the model pathname by using mandatory arg --global_view_model=...\n" << std::endl;
		help(argv);
		return 0;
	}
	if (!parser.has("focused_on_lp_model"))
	{
		std::cout << "\nYou must specify the model pathname by using mandatory arg --focused_on_lp_model=...\n" << std::endl;
		help(argv);
		return 0;
	}
	std::string global_view_model_filename = (parser.get<std::string>("global_view_model"));
	std::string focused_on_lp_model_filename = (parser.get<std::string>("focused_on_lp_model"));
	if (!global_view_model_filename.size() || !std::filesystem::exists(global_view_model_filename)
		|| !std::filesystem::is_regular_file(global_view_model_filename)
		)
	{
		std::cout << "\nCan't find the global_view_model file";
		if (global_view_model_filename.size())
		{
			std::cout << " : global_view_model file arg is : " << global_view_model_filename;
		}
		std::cout << std::endl;
		std::cout << std::endl;
		help(argv);
		return 0;
	}
	else {
		if (!focused_on_lp_model_filename.size() || !std::filesystem::exists(focused_on_lp_model_filename)
			|| !std::filesystem::is_regular_file(focused_on_lp_model_filename)
			)
		{
			std::cout << "\nCan't find the focused_on_lp_model file";
			if (focused_on_lp_model_filename.size())
			{
				std::cout << " : focused_on_lp_model file arg is : " << focused_on_lp_model_filename;
			}
			std::cout << std::endl;
			std::cout << std::endl;
			help(argv);
			focused_on_lp_model_filename = global_view_model_filename;
		}
		std::string image_filename = (parser.get<std::string>("image"));
		std::string dir = (parser.get<std::string>("dir"));
		std::cout << "image file =" << image_filename << std::endl;
		std::cout << "images dir =" << dir << std::endl;
		if (!parser.has("image") && !parser.has("dir"))
		{
			std::cout << "\nYou must specify either an image filename or a directory (with images in it)\n" << std::endl;
			help(argv);
			return 0;
		}
		if (!dir.size() && image_filename.size())
		{
			detect_one_image(image_filename, global_view_model_filename, focused_on_lp_model_filename);
		}
		else {
			//process all images files of a directory
			//step 5 call the detect_without_lpn_detection function of the Yolov5_anpr_onxx_detector object, on a cv::mat object or an image file.
			//This will retieves boxes and classes of the license plate characters
			detect_one_directory(dir, global_view_model_filename, focused_on_lp_model_filename);
		}
	}
	return 0;
}
