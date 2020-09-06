#include "Base.h"
#include "PanopticFPN.h"

#include <Detectron2/Structures/PostProcessing.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PanopticFPNImpl::PanopticFPNImpl(CfgNode &cfg) : MetaArchImpl(cfg),
	m_instance_loss_weight(cfg["MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT"].as<float>()),
	m_combine_on(cfg["MODEL.PANOPTIC_FPN.COMBINE.ENABLED"].as<bool>()),
	m_combine_overlap_threshold(cfg["MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH"].as<float>()),
	m_combine_stuff_area_limit(cfg["MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT"].as<float>()),
	m_combine_instances_confidence_threshold(cfg["MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH"].as<float>())
{
	m_sem_seg_head = make_shared<SemSegFPNHeadImpl>(cfg, m_backbone->output_shapes());
	register_module("sem_seg_head", m_sem_seg_head);

	m_roi_heads = build_roi_heads(cfg, m_backbone->output_shapes());
	register_module("roi_heads", m_roi_heads);
}

void PanopticFPNImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	MetaArchImpl::initialize(importer, prefix);
	m_sem_seg_head->initialize(importer, "sem_seg_head");
	m_roi_heads->initialize(importer, "roi_heads");
}

std::tuple<InstancesList, TensorMap> PanopticFPNImpl::forward(
	const std::vector<DatasetMapperOutput> &batched_inputs) {
	auto images = preprocess_image(batched_inputs, m_backbone->size_divisibility());
	auto features = m_backbone(images.tensor());

	TensorMap proposal_losses;
	InstancesList proposals;
	if (batched_inputs[0].proposals) {
		proposals = Instances::to<DatasetMapperOutput>(batched_inputs, device(),
			&DatasetMapperOutput::get_proposals);
	}

	auto gt_sem_seg = get_gt_sem_seg(batched_inputs, m_sem_seg_head->ignore_value());
	Tensor sem_seg_results;
	TensorMap sem_seg_losses;
	tie(sem_seg_results, sem_seg_losses) = m_sem_seg_head(features, gt_sem_seg);

	InstancesList gt_instances = get_gt_instances(batched_inputs);

	if (m_proposal_generator) {
		tie(proposals, proposal_losses) = m_proposal_generator(images, features, gt_instances);
	}
	InstancesList detector_results;
	TensorMap detector_losses;
	tie(detector_results, detector_losses) = m_roi_heads(images, features, proposals, gt_instances);

	if (is_training()) {
		TensorMap losses;
		losses.insert(sem_seg_losses.begin(), sem_seg_losses.end());
		for (auto iter : detector_losses) {
			losses[iter.first] = iter.second * m_instance_loss_weight;
		}
		losses.insert(proposal_losses.begin(), proposal_losses.end());
		return { InstancesList{}, losses };
	}

	int count = batched_inputs.size();
	assert(sem_seg_results.size(0) == count);
	assert(detector_results.size() == count);
	auto &image_sizes = images.image_sizes();
	assert(image_sizes.size() == count);

	InstancesList processed_results;
	for (int i = 0; i < count; i++) {
		auto sem_seg_result = sem_seg_results[i];
		auto &detector_result = detector_results[i];
		auto &input_per_image = batched_inputs[i];
		auto &image_size = image_sizes[i];

		int height = input_per_image.height ? *input_per_image.height : image_size.height;
		int width = input_per_image.width ? *input_per_image.width : image_size.width;

		auto sem_seg_r = PostProcessing::sem_seg_postprocess(sem_seg_result, image_size, height, width);
		auto detector_r = PostProcessing::detector_postprocess(detector_result, height, width);

		auto output = make_shared<Instances>(ImageSize{ height, width }, false);
		output->set("sem_seg", sem_seg_r);
		output->set("instances", detector_r);
		if (m_combine_on) {
			output->set("panoptic_seg", combine_semantic_and_instance_outputs(detector_r, sem_seg_r.argmax(0)));
		}
		processed_results.push_back(output);
	}
	return { processed_results, {} };
}

std::shared_ptr<PanopticSegment> PanopticFPNImpl::combine_semantic_and_instance_outputs(
	const InstancesPtr &instance_results, const torch::Tensor &semantic_results) {
	auto ret = make_shared<PanopticSegment>();
	auto &panoptic_seg = ret->seg;
	auto &segments_info = ret->infos;

	auto scores = instance_results->getTensor("scores");
	auto pred_classes = instance_results->getTensor("pred_classes");

	panoptic_seg = torch::zeros_like(semantic_results, torch::kInt32);

	// sort instance outputs by scores
	auto sorted_inds = torch::argsort(-scores);
	int count = sorted_inds.size(0);

	int current_segment_id = 0;
	segments_info.reserve(count);

	auto instance_masks = instance_results->getTensor("pred_masks").to(
		dtype(torch::kBool).device(panoptic_seg.device()));

	// Add instances one-by-one, check for overlaps with existing ones
	for (int i = 0; i < count; i++) {
		auto inst_id = sorted_inds[i].item<int64_t>();
		auto score = scores[inst_id].item<float>();
		if (score < m_combine_instances_confidence_threshold) {
			break;
		}
		auto mask = instance_masks[inst_id]; // H, W
		auto mask_area = mask.sum().item<float>();

		if (mask_area == 0) {
			continue;
		}

		auto intersect = (mask > 0).bitwise_and(panoptic_seg > 0);
		auto intersect_area = intersect.sum().item<float>();

		if (intersect_area * 1.0 / mask_area > m_combine_overlap_threshold) {
			continue;
		}

		if (intersect_area > 0) {
			mask = mask.bitwise_and(panoptic_seg == 0);
		}

		current_segment_id += 1;
		panoptic_seg.index_put_({ mask }, current_segment_id);

		segments_info.resize(segments_info.size() + 1);
		SegmentInfo &info = segments_info.back();
		info.id = current_segment_id;
		info.isthing = true;
		info.score = score;
		info.category_id = pred_classes[inst_id].item<int64_t>();
		info.instance_id = inst_id;
		info.area = 0.0f;
	}

	// Add semantic results to remaining empty areas
	Tensor semantic_labels = tolist(get<0>(torch::_unique2(semantic_results)).cpu());
	for (int i = 0; i < semantic_labels.size(0); i++) {
		auto semantic_label = semantic_labels[i].item<int64_t>();
		if (semantic_label == 0) {  // 0 is a special "thing" class
			continue;
		}
		auto mask = (semantic_results == semantic_label).bitwise_and(panoptic_seg == 0);
		auto mask_area = mask.sum().item<float>();
		if (mask_area < m_combine_stuff_area_limit) {
			continue;
		}

		current_segment_id += 1;
		panoptic_seg.index_put_({ mask }, current_segment_id);

		segments_info.resize(segments_info.size() + 1);
		SegmentInfo &info = segments_info.back();
		info.id = current_segment_id;
		info.isthing = false;
		info.score = 0.0f;
		info.category_id = semantic_label;
		info.instance_id = 0;
		info.area = mask_area;
		segments_info.push_back(info);
	}

	return ret;
}