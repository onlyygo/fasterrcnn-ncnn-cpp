#include <assert.h>
#include <math.h>
#include "FasterRCNN.h"

FasterRCNN::FasterRCNN() {

	proposal_boxes = NULL;
	proposal_scores = NULL;
	detection_boxes = NULL;
	detection_scores = NULL;
}

FasterRCNN::~FasterRCNN() {
	this->release();
}

void FasterRCNN::release() {

	if (proposal_boxes != NULL) {

		delete[] proposal_boxes;
		proposal_boxes = NULL;
	}

	if (proposal_scores != NULL) {

		delete[] proposal_scores;
		proposal_scores = NULL;
	}

	if (detection_boxes != NULL) {

		delete[] detection_boxes;
		detection_boxes = NULL;
	}

	if (detection_scores != NULL) {

		delete[] detection_scores;
		detection_scores = NULL;
	}

}

void FasterRCNN::initRPN_NET(const char* model_file,
		const char* weights_file) {

	rpn_net.load_param(model_file);
	if (weights_file)
		rpn_net.load_model(weights_file);
}

void FasterRCNN::initFastRCNN_NET(const char* model_file,
		const char* weights_file) {

    fast_rcnn_net.load_param(model_file);
	if (weights_file)
        fast_rcnn_net.load_model(weights_file);
}

void bytes2ncnnMat(ncnn::Mat &m, float* bytes){

#pragma omp parallel for
    for(int i =0;i<m.c;i++){

		float* dst = m.channel(i);
		const float* src = bytes + (m.w*m.h)*i;
		memcpy(dst, src, m.w*m.h * sizeof(float));
    }
}

void ncnnMat2bytes(ncnn::Mat m, float* bytes){

#pragma omp parallel for
    for(int i =0;i<m.c;i++){

		const float* src = m.channel(i);
		float* dst = bytes + (m.w*m.h)*i;
		memcpy(dst, src, m.w*m.h * sizeof(float));
    }
}

void FasterRCNN::proposal_im_detect(float* bytes, int w, int h, int c) {

	ncnn::Mat image0(w,h,c);
	bytes2ncnnMat(image0, bytes);
	const int BOX_DIMS = 4;

	input_width = image0.w;
	input_height = image0.h;

	float *data_buf = NULL;
	float* box_deltas = NULL;
	float* anchors = NULL;
	float* pred_boxes = NULL;
	float* pred_scores = NULL;
	float* boxes_nms = NULL;
	float* scores_nms = NULL;
	float* temp = NULL;

	int num;
	int valid_num;
	int top_num;
	int pick_num;

    const float mean_vals[3] = {103.9390f, 116.7790, 123.6800f};
    image0.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = rpn_net.create_extractor();
    ex.set_light_mode(false);
    ex.input(cfg.data_layer, image0);

    ex.extract(cfg.shared_layer, conv_feat_blob_mat);
    ncnn::Mat proposal_bbox_pred_mat;
    ex.extract(cfg.proposal_bbox_layer, proposal_bbox_pred_mat);
    ncnn::Mat proposal_cls_prob_mat;
    ex.extract(cfg.proposal_cls_layer, proposal_cls_prob_mat);

	int proposal_bbox_pred_width = proposal_bbox_pred_mat.w;
	int proposal_bbox_pred_height = proposal_bbox_pred_mat.h;
	int proposal_bbox_pred_channels = proposal_bbox_pred_mat.c;
	int proposal_bbox_pred_num = 1; //forward num always 1
	float* proposal_bbox_pred_data = new float[proposal_bbox_pred_width*proposal_bbox_pred_height*proposal_bbox_pred_channels];
    ncnnMat2bytes(proposal_bbox_pred_mat, proposal_bbox_pred_data);
	printf("%d,%d,%d\n",proposal_bbox_pred_width,proposal_bbox_pred_height,proposal_bbox_pred_channels);

	int feature_width = proposal_bbox_pred_width;
	int feature_height = proposal_bbox_pred_height;
	feature_width_ = feature_width;
	feature_height_ = feature_height;

	num = ANCHOR_NUM * feature_width * feature_height;

	box_deltas = new float[num * BOX_DIMS];
	int index = 0;
	for (int width_index = 0; width_index < proposal_bbox_pred_width;
			width_index++) {

		for (int height_index = 0; height_index < proposal_bbox_pred_height;
				height_index++) {

			for (int channels_index = 0;
					channels_index < proposal_bbox_pred_channels;
					channels_index++) {

				//w h c
				box_deltas[index++] =
						proposal_bbox_pred_data[channels_index
								* proposal_bbox_pred_width
								* proposal_bbox_pred_height
								+ height_index * proposal_bbox_pred_width
								+ width_index];
			}
		}
	}

	const int SIZE = feature_width * feature_height;
	anchors = new float[ANCHOR_NUM * SIZE * ANCHOR_DIM];	//(9, 2268, 4)
	proposal_locate_anchors(feature_width, feature_height, anchors);
	pred_boxes = new float[num * BOX_DIMS];
	fast_rcnn_bbox_transform_inv(box_deltas, anchors, pred_boxes, num);
	clip_boxes(pred_boxes, image0.w, image0.h, num);

	int proposal_cls_prob_width = proposal_cls_prob_mat.w;
	int proposal_cls_prob_height = proposal_cls_prob_mat.h;
	int proposal_cls_prob_channels = proposal_cls_prob_mat.c;
	int proposal_cls_prob_num = 1;
	float* proposal_cls_prob_data = new float[proposal_cls_prob_width*proposal_cls_prob_height*proposal_cls_prob_channels];
	ncnnMat2bytes(proposal_cls_prob_mat, proposal_cls_prob_data);
	printf("%d,%d,%d\n",proposal_cls_prob_width,proposal_cls_prob_height,proposal_cls_prob_channels);
	temp = new float[proposal_cls_prob_height * proposal_cls_prob_width];
	#pragma omp parallel for
	for (int i = 0; i < proposal_cls_prob_height * proposal_cls_prob_width;
			i++) {

		temp[i] = proposal_cls_prob_data[proposal_cls_prob_height
				* proposal_cls_prob_width + i];
	}
	pred_scores = new float[proposal_cls_prob_height * proposal_cls_prob_width];
	index = 0;
	int width_reshape = proposal_cls_prob_width;
	int height_reshape = feature_height;
	int channels_reshape = proposal_cls_prob_height / height_reshape;
	for (int width_index = 0; width_index < width_reshape; width_index++) {

		for (int height_index = 0; height_index < height_reshape;
				height_index++) {

			for (int channels_index = 0; channels_index < channels_reshape;
					channels_index++) {

				//w h c
				pred_scores[index++] = temp[channels_index * height_reshape
						* width_reshape + height_index * width_reshape
						+ width_index];
			}
		}
	}
	filter_boxes_sort(pred_boxes, pred_scores, num, valid_num);
	if (cfg.per_nms_topN > 0)
		top_num = std::min(valid_num, cfg.per_nms_topN);

	nms<float>(pred_boxes, pred_scores, top_num, cfg.nms_overlap_thresint,
			pick_num, boxes_nms, scores_nms);


	if (cfg.after_nms_topN > 0)
		proposal_num = std::min(pick_num, cfg.after_nms_topN);

	proposal_boxes = new float[proposal_num * BOX_DIMS];
	proposal_scores = new float[proposal_num];
	#pragma omp parallel for
	for (int i = 0; i < proposal_num; i++) {

		proposal_boxes[i * BOX_DIMS + 0] = boxes_nms[i * BOX_DIMS + 0];
		proposal_boxes[i * BOX_DIMS + 1] = boxes_nms[i * BOX_DIMS + 1];
		proposal_boxes[i * BOX_DIMS + 2] = boxes_nms[i * BOX_DIMS + 2];
		proposal_boxes[i * BOX_DIMS + 3] = boxes_nms[i * BOX_DIMS + 3];
		proposal_scores[i] = scores_nms[i];
	}
	// FILE* fp = fopen("./proposal_boxes.bin","wb");
	// fwrite(proposal_boxes, sizeof(float)*BOX_DIMS, proposal_num, fp);
	// fclose(fp);

	// FILE* fp2 = fopen("./conv_feat_blob_mat.bin","wb");
	// fwrite(conv_feat_blob_mat.data, sizeof(float)*conv_feat_blob_mat.total(), 1, fp2);
	// fclose(fp2);
	// printf("conv_feat_blob_mat: c=%d,h=%d,w=%d\n",conv_feat_blob_mat.c,conv_feat_blob_mat.h,conv_feat_blob_mat.w);

	delete[] proposal_bbox_pred_data;
	delete[] proposal_cls_prob_data;
	delete[] temp;
	delete[] boxes_nms;
	delete[] scores_nms;
	delete[] pred_scores;
	delete[] pred_boxes;
	delete[] anchors;
	delete[] box_deltas;
	delete[] data_buf;
}


void FasterRCNN::fast_rcnn_conv_feat_detect() {

	const int BOX_DIMS = 4;

	float* pred_boxes = NULL;
	float* pred_scores = NULL;

	// proposal_num = 300;
	// proposal_boxes = new float[proposal_num*BOX_DIMS];
	// FILE* fp = fopen("./proposal_boxes.bin","rb");
	// fread(proposal_boxes, sizeof(float)*BOX_DIMS, proposal_num, fp);
	// fclose(fp);

	// ncnn::Mat tmat(63,38,512);
	// FILE* fp2 = fopen("./conv_feat_blob_mat.bin","rb");
	// fread(tmat.data, sizeof(float)*tmat.total(), 1, fp2);
	// fclose(fp2);

	// conv_feat_blob_mat = tmat;

    //for layer data
	printf("c=%d,h=%d,w=%d\n",conv_feat_blob_mat.c,conv_feat_blob_mat.h,conv_feat_blob_mat.w);
    //for layer rois
	float* bbox_pred_mat_vector = new float[proposal_num*cfg.bbox_pred_num_output];
	float* cls_prob_mat_vector = new float[proposal_num*cfg.cls_score_num_output];
	
    #pragma omp parallel for
    for (int i = 0; i < proposal_num; i++) {

		ncnn::Mat roi_blob(4,1,1);
        float* roi_ptr = roi_blob.data;
        roi_ptr[3] = proposal_boxes[i * BOX_DIMS + 3] - proposal_boxes[i * BOX_DIMS + 1];
        roi_ptr[2] = proposal_boxes[i * BOX_DIMS + 2] - proposal_boxes[i * BOX_DIMS + 0];
        roi_ptr[1] = proposal_boxes[i * BOX_DIMS + 1];
        roi_ptr[0] = proposal_boxes[i * BOX_DIMS + 0];

		ncnn::Extractor ex = fast_rcnn_net.create_extractor();
    	ex.set_light_mode(false);
		ex.input(cfg.data_layer, conv_feat_blob_mat);
		ex.input(cfg.rois_layer, roi_blob);
		//for output

		ncnn::Mat bbox_pred_mat;
		ex.extract(cfg.detection_bbox_layer, bbox_pred_mat);
		assert(bbox_pred_mat.c == cfg.bbox_pred_num_output);
		assert(bbox_pred_mat.w == 1);
		assert(bbox_pred_mat.h == 1);

		int copyStart = i*cfg.bbox_pred_num_output;
		ncnnMat2bytes(bbox_pred_mat, bbox_pred_mat_vector+copyStart);

		ncnn::Mat cls_prob_mat;
		ex.extract(cfg.detection_cls_layer, cls_prob_mat);
		assert(cls_prob_mat.c == cfg.cls_score_num_output);
		assert(cls_prob_mat.w == 1);
		assert(cls_prob_mat.h == 1);
		int copyStart2 = i*cfg.cls_score_num_output;
		ncnnMat2bytes(cls_prob_mat, cls_prob_mat_vector+copyStart2);		
    }

	const float* bbox_pred_data = bbox_pred_mat_vector;
	int bbox_pred_height = 1;
    int bbox_pred_width = 1;
	int bbox_pred_channels = cfg.bbox_pred_num_output;
	int bbox_pred_num = proposal_num;
	precess_bbox_pred(bbox_pred_data, bbox_pred_num, bbox_pred_channels, bbox_pred_width, bbox_pred_height, pred_boxes);

	const float* cls_prob_data = cls_prob_mat_vector;
	int cls_prob_height = 1;
	int cls_prob_width = 1;
	int cls_prob_channels = cfg.cls_score_num_output;
	int cls_prob_num = proposal_num;
	precess_cls_prob(cls_prob_data, cls_prob_num, cls_prob_channels, cls_prob_width, cls_prob_width, pred_scores);
	nms<float>(pred_boxes, pred_scores, bbox_pred_num,
			cfg.nms_overlap_thresint2, detection_num, detection_boxes,
			detection_scores);
	delete[] bbox_pred_mat_vector;
	delete[] cls_prob_mat_vector;
	delete[] pred_boxes;
	delete[] pred_scores;

}

void FasterRCNN::proposal_locate_anchors(int feature_width, int feature_height,
		float* anchors) {

	const int SIZE = feature_width * feature_height;

	for (int c = 0; c < SIZE; c++) {

		for (int r = 0; r < ANCHOR_NUM; r++) {

			for (int n = 0; n < ANCHOR_DIM; n++) {

				float temp = 0;
				if (n % 2 == 0) {

					temp = (c - 1) / feature_height * cfg.feat_stride;
				} else {

					//c = 0 36 ...=0
					//c = 1 37 ...=16

					//c = 35 71...=35*16=560
					temp = (c % feature_height) * cfg.feat_stride;
				}
				temp += ANCHORS[r][n];
				anchors[c * ANCHOR_NUM * ANCHOR_DIM + r * ANCHOR_DIM + n] = temp;
			}
		}
	}
}

void FasterRCNN::fast_rcnn_bbox_transform_inv(const float* box_deltas,
		float* anchors, float* pred_boxes, int num) {
	const int BOX_DIMS = 4;
	#pragma omp parallel for
	for (int i = 0; i < num; i++) {

		float src_w = anchors[i * BOX_DIMS + 2] - anchors[i * BOX_DIMS + 0] + 1;
		float src_h = anchors[i * BOX_DIMS + 3] - anchors[i * BOX_DIMS + 1] + 1;
		float src_ctr_x = float(anchors[i * BOX_DIMS + 0] + 0.5 * (src_w - 1));
		float src_ctr_y = float(anchors[i * BOX_DIMS + 1] + 0.5 * (src_h - 1));

		float dst_ctr_x = float(box_deltas[i * BOX_DIMS + 0]);
		float dst_ctr_y = float(box_deltas[i * BOX_DIMS + 1]);
		float dst_scl_x = float(box_deltas[i * BOX_DIMS + 2]);
		float dst_scl_y = float(box_deltas[i * BOX_DIMS + 3]);

		float pred_ctr_x = dst_ctr_x * src_w + src_ctr_x;
		float pred_ctr_y = dst_ctr_y * src_h + src_ctr_y;
		float pred_w = exp(dst_scl_x) * src_w;
		float pred_h = exp(dst_scl_y) * src_h;
		pred_boxes[i * BOX_DIMS + 0] = pred_ctr_x - 0.5 * (pred_w - 1);
		pred_boxes[i * BOX_DIMS + 1] = pred_ctr_y - 0.5 * (pred_h - 1);
		pred_boxes[i * BOX_DIMS + 2] = pred_ctr_x + 0.5 * (pred_w - 1);
		pred_boxes[i * BOX_DIMS + 3] = pred_ctr_y + 0.5 * (pred_h - 1);
	}
}

void FasterRCNN::clip_boxes(float* pred_boxes, int im_width, int im_height,
		int num) {

	const int BOX_DIMS = 4;
	#pragma omp parallel for
	for (int i = 0; i < num; i++) {

		pred_boxes[i * BOX_DIMS + 0] = std::max(
				std::min(pred_boxes[i * BOX_DIMS + 0], (float) im_width), 0.f);
		pred_boxes[i * BOX_DIMS + 1] = std::max(
				std::min(pred_boxes[i * BOX_DIMS + 1], (float) im_height), 0.f);
		pred_boxes[i * BOX_DIMS + 2] = std::max(
				std::min(pred_boxes[i * BOX_DIMS + 2], (float) im_width), 0.f);
		pred_boxes[i * BOX_DIMS + 3] = std::max(
				std::min(pred_boxes[i * BOX_DIMS + 3], (float) im_height), 0.f);
	}
}

void FasterRCNN::filter_boxes_sort(float* pred_boxes, float* pred_scores,
		int num, int &valid_num) {

	const int BOX_DIMS = 4;
	valid_num = num;
	for (int i = 0; i < num; i++) {

		int widths = pred_boxes[i * BOX_DIMS + 2] - pred_boxes[i * BOX_DIMS + 0]
				+ 1;
		int heights = pred_boxes[i * BOX_DIMS + 3]
				- pred_boxes[i * BOX_DIMS + 1] + 1;
		
		if (widths < cfg.test_min_box_size
				|| heights < cfg.test_min_box_size
				|| pred_scores[i] == 0) {

			pred_scores[i] = 0;
			valid_num--;
		}
	}
	//sort by pred_scores
	for (int findMin = 0; findMin < num; findMin++) {

		for (int i = 0; i < num - 1 - findMin; i++) {

			float pro = pred_scores[i];
			float next = pred_scores[i + 1];
			if (pro < next) {

				pred_scores[i + 1] = pro;
				pred_scores[i] = next;

				//pred_boxes 
				for (int j = 0; j < BOX_DIMS; j++) {
					int temp = pred_boxes[i * BOX_DIMS + j];
					pred_boxes[i * BOX_DIMS + j] = pred_boxes[(i + 1) * BOX_DIMS
							+ j];
					pred_boxes[(i + 1) * BOX_DIMS + j] = temp;
				}
			}
		}
	}
}

void FasterRCNN::precess_bbox_pred(const float* data, int num, int channels, int width, int height, float* &pred_boxes){

	float* pred_transforms = new float[num*ANCHOR_DIM];//release later
	pred_boxes = new float[num*ANCHOR_DIM];//for return

	int anchor_length = channels * width * height;
	assert(anchor_length == cfg.bbox_pred_num_output);
	#pragma omp parallel for
	for (int row = 0; row < num; row++) {

		pred_transforms[row*ANCHOR_DIM+0] = data[row * anchor_length + ANCHOR_DIM*cfg.bbox_pred_index+0];
		pred_transforms[row*ANCHOR_DIM+1] = data[row * anchor_length + ANCHOR_DIM*cfg.bbox_pred_index+1];
		pred_transforms[row*ANCHOR_DIM+2] = data[row * anchor_length + ANCHOR_DIM*cfg.bbox_pred_index+2];
		pred_transforms[row*ANCHOR_DIM+3] = data[row * anchor_length + ANCHOR_DIM*cfg.bbox_pred_index+3];
	}
	fast_rcnn_bbox_transform_inv(pred_transforms, proposal_boxes,
			pred_boxes, num);
	clip_boxes(pred_boxes, input_width, input_height, num);

	delete[] pred_transforms;
}

void FasterRCNN::precess_cls_prob(const float* data, int num, int channels, int width, int height, float* &pred_scores){

	pred_scores = new float[num];//for return

	int anchor_length = channels * width * height;
	assert(anchor_length == cfg.cls_score_num_output);
	#pragma omp parallel for
	for (int row = 0; row < num; row++) {

		pred_scores[row] = data[row * anchor_length + cfg.cls_score_index];
	}
}


void FasterRCNN::print_net(ncnn::Net net) {

//	cout << "************************************************" << endl;
//	cout << "********************" << net->name() << "**********************"
//			<< endl;
//	cout << "************************************************" << endl;
//	vector<string> blob_names = net->blob_names();
//	for (int i = 0; i < blob_names.size(); i++) {
//
//		int _height = net->blob_by_name(blob_names[i])->height();
//		int _width = net->blob_by_name(blob_names[i])->width();
//		int _channels = net->blob_by_name(blob_names[i])->channels();
//		int _num = net->blob_by_name(blob_names[i])->num();
//		cout << blob_names[i] << " : " << "_height = " << _height
//				<< " _width = " << _width << " _channels = " << _channels
//				<< " _num = " << _num << endl;
//	}
}

void FasterRCNN::log_detections(float CONF_THRESH) {
	const int BOX_DIMS = 4;
	for (int i = 0; i < detection_num; i++) {

		if (detection_scores[i] > CONF_THRESH) {

            printf("%f, %f, %f, %f, %f\n",detection_boxes[i * BOX_DIMS + 0],
                   detection_boxes[i * BOX_DIMS + 1],
                    detection_boxes[i * BOX_DIMS + 2],
                    detection_boxes[i * BOX_DIMS + 3],
                    detection_scores[i]);
		}
	}
}

void FasterRCNN::log_proposals(float CONF_THRESH) {
	const int BOX_DIMS = 4;
	for (int i = 0; i < proposal_num; i++) {

		if (proposal_scores[i] > CONF_THRESH) {

            printf("%f, %f, %f, %f, %f\n",proposal_boxes[i * BOX_DIMS + 0],
                   proposal_boxes[i * BOX_DIMS + 1],
                    proposal_boxes[i * BOX_DIMS + 2],
                    proposal_boxes[i * BOX_DIMS + 3],
                    proposal_scores[i]);
		}
	}
}
