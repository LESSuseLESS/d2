#include "Base.h"
#include "VideoVisualizer.h"

#include "Utils.h"
#include "Visualizer.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VideoVisualizer::VideoVisualizer(Metadata metadata, ColorMode instance_mode) :
	m_metadata(metadata), m_instance_mode(instance_mode) {
	// Other mode not supported yet.
	assert(m_instance_mode == ColorMode::kIMAGE || m_instance_mode == ColorMode::kIMAGE_BW);
}

VisImage VideoVisualizer::draw_instance_predictions(cv::Mat frame, const InstancesPtr &predictions) {
	Visualizer frame_visualizer(image_to_tensor(frame), m_metadata);
	int num_instances = predictions->len();
	if (num_instances == 0) {
		return frame_visualizer.get_output();
	}

	Tensor boxes; if (predictions->has("pred_boxes")) boxes = predictions->getTensor("pred_boxes");
	Tensor scores; if (predictions->has("scores")) scores = predictions->getTensor("scores");
	Tensor classes; if (predictions->has("pred_classes")) classes = predictions->getTensor("pred_classes");
	Tensor keypoints; if (predictions->has("pred_keypoints")) keypoints = predictions->getTensor("pred_keypoints");

	vector<shared_ptr<GenericMask>> masks;
	if (predictions->has("pred_masks")) {
		auto t_masks = predictions->getTensor("pred_masks");
		int count = t_masks.size(0);
		masks.reserve(count);
		for (int i = 0; i < count; i++) {
			auto mask = make_shared<GenericMask>(t_masks[i], frame_visualizer.height(), frame_visualizer.width());
			masks.push_back(mask);
		}
		// mask IOU is not yet enabled
		// masks_rles = mask_util.encode(np.asarray(masks.permute(1, 2, 0), order="F"))
		// assert len(masks_rles) == num_instances
	}

	vector<shared_ptr<_DetectedInstance>> detected;
	detected.reserve(num_instances);
	for (int i = 0; i < num_instances; i++) {
		auto di = make_shared <_DetectedInstance>();
		di->label = classes[i].item<int64_t>();
		di->bbox = boxes[i];
		di->ttl = 8;
		detected.push_back(di);
	}
	auto colors = _assign_colors(detected);

	auto labels = Visualizer::_create_text_labels(classes, scores, m_metadata->thing);

	float alpha = 0.5;
	if (m_instance_mode == ColorMode::kIMAGE_BW) {
		// any() returns uint8 tensor
		frame_visualizer.set_grayscale_image(predictions->getTensor("pred_masks").any(0) > 0);
		alpha = 0.3;
	}

	frame_visualizer.overlay_instances(
		masks.empty() ? boxes : Tensor(), // boxes are a bit distracting
		labels, masks, keypoints, colors, alpha
	);
	return frame_visualizer.get_output();
}

VisImage VideoVisualizer::draw_sem_seg(cv::Mat frame, const torch::Tensor &sem_seg, int) {
	// don't need to do anything special
	Visualizer frame_visualizer(image_to_tensor(frame), m_metadata);
	frame_visualizer.draw_sem_seg(sem_seg);
	return frame_visualizer.get_output();
}

VisImage VideoVisualizer::draw_panoptic_seg_predictions(cv::Mat frame, const torch::Tensor &panoptic_seg,
	const std::vector<SegmentInfo> &segments_info, int area_threshold, float alpha) {
	Visualizer frame_visualizer(image_to_tensor(frame), m_metadata);
	_PanopticPrediction pred(panoptic_seg, segments_info);

	if (m_instance_mode == ColorMode::kIMAGE_BW) {
		frame_visualizer.set_grayscale_image(pred.non_empty_mask());
	}

    // draw mask for all semantic segments first i.e. "stuff"
	pred.semantic_masks([&](torch::Tensor mask, const SegmentInfo &sinfo) {
		auto category_idx = sinfo.category_id;
		assert(category_idx >= 0 && category_idx < m_metadata->stuff.size());
		VisColor mask_color = color_normalize(m_metadata->stuff[category_idx].color);
		auto &text = m_metadata->stuff[category_idx].cls;
		frame_visualizer.draw_binary_mask(mask, mask_color, {}, text, alpha, area_threshold);
		});

	// draw mask for all instances second
	vector<shared_ptr<GenericMask>> masks;
	vector<int64_t> category_ids;
	vector<string> labels;
	int num_instances = 0;
	pred.instance_masks([&](torch::Tensor mask, const SegmentInfo &sinfo) {
		++num_instances;
		masks.push_back(make_shared<GenericMask>(mask, frame_visualizer.height(), frame_visualizer.width()));
		auto category_id = sinfo.category_id;
		category_ids.push_back(category_id);
		assert(category_id >= 0 && category_id < m_metadata->thing.size());
		labels.push_back(m_metadata->thing[category_id].cls);
		});

	auto masks_rles = mask_util::encode(GenericMask::toCocoMask(masks).permute({ 1, 2, 0 }).to(torch::kUInt8));
	assert(masks_rles.size() == num_instances);
	vector<shared_ptr<_DetectedInstance>> detected;
	detected.reserve(num_instances);
	for (int i = 0; i < num_instances; i++) {
		auto di = make_shared <_DetectedInstance>();
		di->label = category_ids[i];
		di->mask_rle = masks_rles[i];
		di->ttl = 8;
		detected.push_back(di);
	}
	auto colors = _assign_colors(detected);

    frame_visualizer.overlay_instances(Tensor(), labels, masks, {}, colors, alpha);
	return frame_visualizer.get_output();
}

std::vector<VisColor> VideoVisualizer::_assign_colors(
	const std::vector<std::shared_ptr<_DetectedInstance>> &instances) {
	assert(!instances.empty());

	// Compute iou with either boxes or masks:
	auto is_crowd = torch::zeros({ (int)instances.size() }, torch::kBool);
	auto &bbox = instances[0]->bbox;
	float threshold = 0.6;
	torch::Tensor ious;
	if (bbox.numel() == 0) {
		assert(instances[0]->mask_rle);
		// use mask iou only when box iou is None
		// because box seems good enough
		auto get_rles = [](const std::vector<std::shared_ptr<_DetectedInstance>> &instances) {
			mask_util::MaskObjectVec ret;
			int count = instances.size();
			ret.reserve(count);
			for (int i = 0; i < count; i++) {
				ret.push_back(instances[i]->mask_rle);
			}
			return ret;
		};

		auto rles_old = get_rles(m_old_instances);
		auto rles_new = get_rles(instances);
		ious = mask_util::iou(rles_old, rles_new, is_crowd);
		threshold = 0.5;
	}
	else if (!m_old_instances.empty()) {
		auto get_boxes = [](const std::vector<std::shared_ptr<_DetectedInstance>> &instances) {
			TensorVec ret;
			int count = instances.size();
			ret.reserve(count);
			for (int i = 0; i < count; i++) {
				ret.push_back(instances[i]->bbox);
			}
			return torch::stack(ret);
		};
		auto boxes_old = get_boxes(m_old_instances);
		auto boxes_new = get_boxes(instances);
		ious = mask_util::iou(boxes_old, boxes_new, is_crowd);
	}

	std::vector<std::shared_ptr<_DetectedInstance>> extra_instances;
	if (!m_old_instances.empty()) {
		if (ious.size(0) == 0) {
			ious = torch::zeros({ (int)m_old_instances.size(), (int)instances.size() }, torch::kFloat32);
		}

		// Only allow matching instances of the same label:
		for (int old_idx = 0; old_idx < m_old_instances.size(); old_idx++) {
			auto &old_instance = m_old_instances[old_idx];
			for (int new_idx = 0; new_idx < instances.size(); new_idx++) {
				auto &new_instance = instances[new_idx];
				if (old_instance->label != new_instance->label) {
					ious.index_put_({ old_idx, new_idx }, 0);
				}
			}
		}

		auto matched_new_per_old = ious.argmax(1);
		auto max_iou_per_old = ious.max_values(1);

		// Try to find match for each old instance:
		extra_instances.reserve(m_old_instances.size());
		for (int idx = 0; idx < m_old_instances.size(); idx++) {
			auto &inst = m_old_instances[idx];
			if (max_iou_per_old[idx].item<float>() > threshold) {
				int newidx = matched_new_per_old[idx].item<int64_t>();
				assert(newidx >= 0 && newidx < instances.size());
				if (instances[newidx]->color.empty()) {
					instances[newidx]->color = inst->color;
					continue;
				}
			}
			// If an old instance does not match any new instances,
			// keep it for the next frame in case it is just missed by the detector
			inst->ttl -= 1;
			if (inst->ttl > 0) {
				extra_instances.push_back(inst);
			}
		}
	}

	// Assign random color to newly-detected instances:
	vector<VisColor> ret;
	ret.reserve(instances.size());
	for (auto inst : instances) {
		if (inst->color.empty()) {
			inst->color = color_random();
		}
		ret.push_back(inst->color);
	}
	m_old_instances.reserve(instances.size() + extra_instances.size());
	m_old_instances = instances;
	m_old_instances.insert(m_old_instances.end(), extra_instances.begin(), extra_instances.end());
	return ret;
}