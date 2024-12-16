#include <iostream>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

// Currently unused

namespace mdl {
    class Model {
        public:
            Model(std::string xml_path, cv::Mat initial_frame);
            void swap_requests();
            void start_curr_async(cv::Mat curr_frame);
            void start_next_async(cv::Mat next_frame);
            ov::Tensor wait_curr_async();
            ov::Tensor wait_next_async();
        private:
            ov::Core core;
            ov::CompiledModel compiled_model;
            ov::InferRequest curr_request;
            ov::InferRequest next_request;
            ov::element::Type input_type;
            ov::Shape input_shape;
            ov::Layout input_layout;
    };
}