#include <iostream>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

struct SharedMatAllocator {
    const cv::Mat mat;
    void* allocate(size_t bytes, size_t) {return bytes <= mat.rows * mat.step[0] ? mat.data : nullptr;}
    void deallocate(void*, size_t, size_t) {}
    bool is_equal(const SharedMatAllocator& other) const noexcept {return this == &other;}
};

int main(int argc, char* argv[]) {
	try {
		// Fetch command-line arguments
		if (argc != 3) {
			std::cerr << "Missing command-line arguments. Usage: ./DetectorTester <model_xml_path> <media_path>" << std::endl;
			return -1;
		}
        std::string xml_path = argv[1];
		std::string media_path = argv[2];

		// Initialize VideoCapture, read media_path and fetch first frame
		cv::VideoCapture cap = cv::VideoCapture(media_path);
		cv::Mat curr_frame;
		cap.read(curr_frame);
		cv::Mat next_frame;

		// Read OpenVINO model
		ov::Core core;
		std::cout << "Loading model: " << xml_path << std::endl;
		std::shared_ptr<ov::Model> model = core.read_model(xml_path);
		std::cout << "Model loaded: " << xml_path << std::endl;

		// Configure data augmentation
		unsigned long height = curr_frame.rows;
		unsigned long width = curr_frame.cols;

		ov::element::Type input_type = ov::element::u8;
		ov::Shape input_shape = { 1, height, width, 3};
		ov::Layout input_layout = { "NHWC" };

		ov::preprocess::PrePostProcessor ppp(model);
		ppp.input().tensor()
			.set_shape(input_shape)
			.set_layout(input_layout)
			.set_element_type(input_type);
		ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
		ppp.input().model().set_layout("NCHW");
		ppp.output().tensor().set_element_type(ov::element::f32);

		// Apply data augmentation to model
		model = ppp.build();

		// Compile model and create infer requests
		ov::CompiledModel compiled_model = core.compile_model(model);
		ov::InferRequest curr_request = compiled_model.create_infer_request();
		ov::InferRequest next_request = compiled_model.create_infer_request();
		ov::Tensor curr_tensor;
		ov::Tensor next_tensor;

		// Start inference
		std::cout << "Starting inference." << std::endl;
		int64 start = cv::getTickCount();

		// Fetch and start inference on curr frame
		curr_tensor = ov::Tensor(input_type, input_shape, SharedMatAllocator{ curr_frame });
		curr_request.set_input_tensor(curr_tensor);
		curr_request.infer();
		
		while (true) {
			// Fetch and start inference on next frame
			cap.read(next_frame);
			next_tensor = ov::Tensor(input_type, input_shape, SharedMatAllocator{ next_frame });
			next_request.set_input_tensor(next_tensor);
			next_request.start_async();

			// Fetch curr frame results
			curr_request.wait();
			const ov::Tensor& output_tensor = curr_request.get_output_tensor();

			// Calculate FPS
			double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
			std::cout << "FPS: " << fps << std::endl;
			start = cv::getTickCount();

			// Show frame and break if ESC key is inputted
			cv::imshow("frame", curr_frame);
			int escape_key = 27;
			if (cv::waitKey(1) == escape_key || next_frame.empty())
				break;

			// Swap curr and next objects
			curr_frame = next_frame;
			curr_tensor = next_tensor;
			std::swap(next_request, curr_request);
		}
		std::cout << "Finished inference." << std::endl;
	} catch (const std::exception& ex) {
		std::cerr << ex.what() << std::endl;
		return -1;
	}
    return 0;
}