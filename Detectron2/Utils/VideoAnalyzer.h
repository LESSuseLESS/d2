#pragma once

#include <Detectron2/Structures/Instances.h>
#include <Detectron2/Structures/PanopticSegment.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from utils/video_visualizer.py

	class VideoAnalyzer {
	public:
		VideoAnalyzer();
		virtual ~VideoAnalyzer();

		virtual void on_instance_predictions(cv::Mat frame, const InstancesPtr &predictions,
			const std::vector<std::string> &keypoint_names);
		virtual void on_sem_seg(cv::Mat frame, const torch::Tensor &sem_seg);
		virtual void on_panoptic_seg_predictions(cv::Mat frame, const torch::Tensor &panoptic_seg,
			const std::vector<SegmentInfo> &segments_info);
	};
}
