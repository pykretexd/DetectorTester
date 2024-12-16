#include <iostream>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#include "model.cpp"


void draw_fps(cv::Mat frame, double fps) {
	std::ostringstream oss;
	oss << "FPS: " << fps;

	cv::putText(
		frame,
		oss.str(), 
		cv::Point(10, frame.rows / 20), 
		cv::FONT_HERSHEY_DUPLEX, 
		1.0, 
		CV_RGB(255, 0, 0), 
		2
	);
}

int main(int argc, char* argv[]) {
	// Fetch command-line arguments
	if (argc != 3) {
		std::cout << "Missing command-line arguments. Usage: ./DetectorTester <model_xml_path> <media_path>" << std::endl;
		return -1;
	}
	std::string xml_path = argv[1];
	std::string media_path = argv[2];

	try {
		// Initialize VideoCapture, read media_path and fetch first frame
		cv::VideoCapture cap = cv::VideoCapture(media_path);
		cv::Mat curr_frame;
		cap.read(curr_frame);
		cv::Mat next_frame;

		mdl::Model model(xml_path, curr_frame);

		// Start inference
		std::cout << "Starting inference." << std::endl;
		int64 start = cv::getTickCount();

		// Start inference on curr frame
		model.start_curr_async(curr_frame);
		
		cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
		while (true) {
			// Fetch and start inference on next frame
			cap.read(next_frame);
			model.start_next_async(next_frame);

			// Fetch curr frame results
			const ov::Tensor& result = model.wait_curr_async();

			// Calculate and draw FPS
			double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
			start = cv::getTickCount();
			draw_fps(curr_frame, fps);

			// Show frame and break if ESC key is inputted
			cv::imshow("window", curr_frame);
			int escape_key = 27;
			if (cv::waitKey(1) == escape_key || next_frame.empty())
				break;

			// Swap curr and next objects
			curr_frame = next_frame;
			model.swap_requests();
		}
		std::cout << "Finished inference." << std::endl;
	} catch (const std::exception& ex) {
		std::cerr << ex.what() << std::endl;
		return -1;
	}
    return 0;
}