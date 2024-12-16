#include <iostream>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

struct SharedMatAllocator {
    const cv::Mat mat;
    void* allocate(size_t bytes, size_t) {return bytes <= mat.rows * mat.step[0] ? mat.data : nullptr;}
    void deallocate(void*, size_t, size_t) {}
    bool is_equal(const SharedMatAllocator& other) const noexcept {return this == &other;}
};

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


    Model::Model(std::string xml_path, cv::Mat initial_frame) {
        // Read OpenVINO model
        std::cout << "Loading model: " << xml_path << std::endl;
        std::shared_ptr<ov::Model> model = core.read_model(xml_path);
        std::cout << "Model loaded: " << xml_path << std::endl;

        // Configure data augmentation
        unsigned long height = initial_frame.rows;
        unsigned long width = initial_frame.cols;
        input_type = ov::element::u8;
        input_shape = { 1, height, width, 3};
        input_layout = { "NHWC" };

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
        compiled_model = core.compile_model(model);
        curr_request = compiled_model.create_infer_request();
        next_request = compiled_model.create_infer_request();
    }

    void Model::swap_requests() {
        std::swap(next_request, curr_request);
    }

    void Model::start_curr_async(cv::Mat curr_frame) {
        ov::Tensor curr_tensor = ov::Tensor(input_type, input_shape, SharedMatAllocator{ curr_frame });
        curr_request.set_input_tensor(curr_tensor);
        curr_request.infer();
    }

    void Model::start_next_async(cv::Mat next_frame) {
        ov::Tensor next_tensor = ov::Tensor(input_type, input_shape, SharedMatAllocator{ next_frame });
        next_request.set_input_tensor(next_tensor);
        next_request.infer();
    }

    ov::Tensor Model::wait_curr_async() {
        curr_request.wait();
        return curr_request.get_output_tensor();
    }

    ov::Tensor Model::wait_next_async() {
        next_request.wait();
        return next_request.get_output_tensor();
    }
}