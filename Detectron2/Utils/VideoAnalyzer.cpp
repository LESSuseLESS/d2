#include "Base.h"
#include "VideoAnalyzer.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VideoAnalyzer::VideoAnalyzer() {
}

VideoAnalyzer::~VideoAnalyzer() {
}

void VideoAnalyzer::on_instance_predictions(cv::Mat frame, const InstancesPtr &predictions,
	const std::vector<std::string> &keypoint_names) {
}

void VideoAnalyzer::on_sem_seg(cv::Mat frame, const torch::Tensor &sem_seg) {
}

void VideoAnalyzer::on_panoptic_seg_predictions(cv::Mat frame, const torch::Tensor &panoptic_seg,
	const std::vector<SegmentInfo> &segments_info) {
}
