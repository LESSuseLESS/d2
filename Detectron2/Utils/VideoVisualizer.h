#pragma once

#include <Detectron2/Data/MetadataCatalog.h>
#include <Detectron2/Structures/Instances.h>
#include "VisImage.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from utils/video_visualizer.py

	class VideoVisualizer {
	public:
		// metadata (MetadataCatalog): image metadata.
		VideoVisualizer(Metadata metadata, ColorMode instance_mode = ColorMode::kIMAGE);

		/**
			Draw instance-level prediction results on an image.

			Args:
				frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
				predictions (Instances): the output of an instance detection/segmentation
					model. Following fields will be used to draw:
					"pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

			Returns:
				output (VisImage): image object with visualizations.
		*/
		VisImage draw_instance_predictions(cv::Mat frame, const InstancesPtr &predictions);

		/**
			Args:
				sem_seg (ndarray or Tensor): semantic segmentation of shape (H, W),
					each value is the integer label.
				area_threshold (Optional[int]): only draw segmentations larger than the threshold
		*/
		VisImage draw_sem_seg(cv::Mat frame, const torch::Tensor &sem_seg, int area_threshold = 0);

		VisImage draw_panoptic_seg_predictions(cv::Mat frame, const torch::Tensor &panoptic_seg,
			const std::vector<SegmentInfo> &segments_info, int area_threshold = 0, float alpha = 0.5);

	private:
		Metadata m_metadata;
		ColorMode m_instance_mode;

		/**
			Used to store data about detected objects in video frame,
			in order to transfer color to objects in the future frames.

			Attributes:
				label (int):
				bbox (tuple[float]):
				mask_rle (dict):
				color (tuple[float]): RGB colors in range (0, 1)
				ttl (int): time-to-live for the instance. For example, if ttl=2,
					the instance color can be transferred to objects in the next two frames.
		*/
		struct _DetectedInstance {
			int64_t label;
			torch::Tensor bbox;
			mask_util::MaskObject mask_rle;
			VisColor color;
			int ttl;
		};
		std::vector<std::shared_ptr<_DetectedInstance>> m_old_instances;

		/**
			Naive tracking heuristics to assign same color to the same instance,
			will update the internal state of tracked instances.

			Returns:
				list[tuple[float]]: list of colors.
		*/
		std::vector<VisColor> _assign_colors(const std::vector<std::shared_ptr<_DetectedInstance>> &instances);
	};
}
