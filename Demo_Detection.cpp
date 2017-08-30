#include "FasterRCNN.h"
#include <opencv.hpp>

void vis_boxes(cv::Mat &image, int num_out,
		float* boxes, float* scores, float CONF_THRESH) {
	const int BOX_DIMS = 4;
	for (int i = 0; i < num_out; i++) {

		if (scores[i] > CONF_THRESH) {
			cv::rectangle(image,
					cv::Point(boxes[i * BOX_DIMS + 0],
							boxes[i * BOX_DIMS + 1]),
					cv::Point(boxes[i * BOX_DIMS + 2],
							boxes[i * BOX_DIMS + 3]),
					cv::Scalar(255, 0, 0));
			char text[32];
			sprintf(text, "%.3f", scores[i]);
			cv::putText(image, text,
					cv::Point(boxes[i * BOX_DIMS + 0],
							boxes[i * BOX_DIMS + 1]),
					CV_FONT_HERSHEY_DUPLEX, 1.0f, cv::Scalar(255, 0, 0));
		}
	}
}

void setPixel(float* bytes, int width, int height, float value, int w, int h, int c) {

	bytes[c*(width*height) + h*width + w] = value;
}

// void cvMat2bytes(cv::Mat m, float* bytes){

// #pragma omp parallel for collapse(2)
// 	for (int h = 0; h < m.rows; ++h) {
// 		for (int w = 0; w < m.cols; ++w) {
// 			bytes[(0 * m.rows + h) * m.cols + w] = float(
// 					m.at<cv::Vec3b>(cv::Point(w, h))[0]);
// 			bytes[(1 * m.rows + h) * m.cols + w] = float(
// 					m.at<cv::Vec3b>(cv::Point(w, h))[1]);
// 			bytes[(2 * m.rows + h) * m.cols + w] = float(
// 					m.at<cv::Vec3b>(cv::Point(w, h))[2]);
// 		}
// 	}
// }

void cvMat2bytes(cv::Mat m, float* bytes){

#pragma omp parallel for collapse(2)
	for (int h = 0; h < m.rows; ++h) {
		for (int w = 0; w < m.cols; ++w) {
			bytes[(0 * m.rows + h) * m.cols + w] = (float)(m.data[h*(3*m.cols)+w*3+0]);
			bytes[(1 * m.rows + h) * m.cols + w] = (float)(m.data[h*(3*m.cols)+w*3+1]);
			bytes[(2 * m.rows + h) * m.cols + w] = (float)(m.data[h*(3*m.cols)+w*3+2]);
		}
	}
}

int main()
{
	const char * model_file = "./model/proposal_test.ncnnproto";
	const char * weights_file = "./model/proposal_final.ncnnbin";
	const char * detection_model_file = "./model/detection_test.ncnnproto";
	const char * detection_weights_file = "./model/detection_final.ncnnbin";

	FasterRCNN faster_rcnn;
	printf("init rpn net\n");
	faster_rcnn.initRPN_NET(model_file, weights_file);
	printf("init fast net\n");
	faster_rcnn.initFastRCNN_NET(detection_model_file, detection_weights_file);
	printf("start\n");
	const char * images[] = { "./we.png"};
	int width = 1000;
	int height = 595;
	for (int i = 0;i<1;i++){
		cv::Mat src = cv::imread(images[i],3);
		cv::Mat dst;
		cv::resize(src, dst, cv::Size(width, height));
		float* bytes = new float[width*height*3];
		cvMat2bytes(dst, bytes);
		faster_rcnn.proposal_im_detect(bytes, width, height, 3);
		printf("per\n");
		faster_rcnn.fast_rcnn_conv_feat_detect();
		printf("fast\n");
		vis_boxes(dst, faster_rcnn.getDetectionNum(), faster_rcnn.getDetectionBoxes(), faster_rcnn.getDetectionScores(), 0.90);
		printf("show\n");
		faster_rcnn.release();
		cv::imwrite("faster-rcnn.jpg",dst);
		cv::imshow("faster-rcnn",dst);
		cv::waitKey(0);
		delete[] bytes;
	}
	return 0;
}