#ifndef FASTERRCNN_H_
#define FASTERRCNN_H_

#include "ncnn_include/net.h"
#include "config.h"
#include "nms.h"

class FasterRCNN {
public:
	FasterRCNN();
	~FasterRCNN();

	void initRPN_NET(const char* model_file, const char* weights_file);
	void initFastRCNN_NET(const char* model_file, const char* weights_file);

	void proposal_im_detect(float* bytes, int w, int h, int c);
	void fast_rcnn_conv_feat_detect();

	static void print_net(ncnn::Net net);
	void log_proposals(float CONF_THRESH);
	void log_detections(float CONF_THRESH);

	float getFeatureWidth() {
		return feature_width_;
	}
	;
	float getFeatureHeight() {
		return feature_height_;
	}
	;
	float getDetectionNum() {
		return detection_num;
	}
	;
	float* getDetectionBoxes() {
		return detection_boxes;
	}
	;
	float* getDetectionScores() {
		return detection_scores;
	}
	;
	float getProposalNum() {
		return proposal_num;
	}
	;
	float* getProposalBoxes() {
		return proposal_boxes;
	}
	;
	float* getProposalScores() {
		return proposal_scores;
	}
	;
	ncnn::Net getRPN_NET() {
		return rpn_net;
	}
	;
	ncnn::Net getFastRCNN_NET() {
		return fast_rcnn_net;
	}
	;
	void release();
private:
	void proposal_locate_anchors(int feature_width, int feature_height,
			float* anchors);
	void fast_rcnn_bbox_transform_inv(const float* box_deltas, float* anchors,
			float* pred_boxes, int num);
	void clip_boxes(float* pred_boxes, int im_width, int im_height, int num);
	void filter_boxes_sort(float* pred_boxes, float* pred_scores, int num,
			int &valid_num);
	void precess_bbox_pred(const float* data, int num, int channels, int width, int height, float* &pred_boxes);
	void precess_cls_prob(const float* data, int num, int channels, int width, int height, float* &pred_scores);
	ncnn::Net  rpn_net;
	ncnn::Mat conv_feat_blob_mat;
	ncnn::Net  fast_rcnn_net;
	Config cfg;

	float scale;
	float input_width;
	float input_height;

	float feature_width_;
	float feature_height_;

	float *proposal_boxes;
	float *proposal_scores;
	int proposal_num;

	float *detection_boxes;
	float *detection_scores;
	int detection_num;
};
#endif
