#include "Base.h"
#include "GeneralizedRCNN.h"

#include <Detectron2/Utils/EventStorage.h>
#include <Detectron2/Utils/Visualizer.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GeneralizedRCNNImpl::GeneralizedRCNNImpl(CfgNode &cfg) : MetaArchImpl(cfg) {
	m_roi_heads = build_roi_heads(cfg, m_backbone->output_shapes());
	register_module("roi_heads", m_roi_heads);
}

void GeneralizedRCNNImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	MetaArchImpl::initialize(importer, prefix);
	m_roi_heads->initialize(importer, "roi_heads");
}

void GeneralizedRCNNImpl::visualize_training(const std::vector<DatasetMapperOutput> &batched_inputs,
	const InstancesList &proposals) {
	auto &storage = get_event_storage();
	int max_vis_prop = 20;

	assert(batched_inputs.size() == proposals.size());
	for (int i = 0; i < batched_inputs.size(); i++) {
		auto input = batched_inputs[i];
		auto prop = proposals[i];
		auto img = input.image.cpu();
		assert(img.size(0) == 3); // Images should have 3 channels
		if (m_input_format == "BGR") {
			img = torch::flip(img, 0);
		}
		img = img.transpose_(1, 2);

		Visualizer v_gt(img, {});
		auto v_gt_output = v_gt.overlay_instances(input.instances->getTensor("gt_boxes"));
		auto anno_img = v_gt_output.get_image();
		auto box_size = min(prop->getTensor("proposal_boxes").size(0), (int64_t)max_vis_prop);
		Visualizer v_pred(img, {});
		auto v_pred_output = v_pred.overlay_instances(prop->getTensor("proposal_boxes").index({ Slice(0, box_size) })
			.cpu());
		auto prop_img = v_pred_output.get_image();
		auto vis_img = torch::cat({ anno_img, prop_img }, 1);
		vis_img = vis_img.permute({ 2, 0, 1 });
		const char *vis_name = "Left: GT bounding boxes;  Right: Predicted proposals";
		storage.put_image(vis_name, vis_img);
		break; // only visualize one image in a batch
	}
}

std::tuple<InstancesList, TensorMap> GeneralizedRCNNImpl::forward(
	const std::vector<DatasetMapperOutput> &batched_inputs) {
	if (!is_training()) {
		auto inferred = inference(batched_inputs);
		return { inferred, {} };
	}

	auto images = preprocess_image(batched_inputs, m_backbone->size_divisibility());
	auto features = m_backbone(images.tensor());

	InstancesList gt_instances = get_gt_instances(batched_inputs);

	InstancesList proposals; TensorMap proposal_losses;
	if (m_proposal_generator) {
		tie(proposals, proposal_losses) = m_proposal_generator(images, features, gt_instances);
	}
	else {
		assert(batched_inputs[0].proposals);
		proposals = Instances::to<DatasetMapperOutput>(batched_inputs, device(),
			&DatasetMapperOutput::get_proposals);
	}

	auto detector_losses = get<1>(m_roi_heads(images, features, proposals, gt_instances));
	if (m_vis_period > 0) {
		assert(false);
		auto &storage = get_event_storage();
		if (storage.iter() % m_vis_period == 0) {
			visualize_training(batched_inputs, proposals);
		}
	}

	TensorMap losses;
	losses.insert(detector_losses.begin(), detector_losses.end());
	losses.insert(proposal_losses.begin(), proposal_losses.end());
	return { InstancesList{}, losses };
}

InstancesList GeneralizedRCNNImpl::inference(const std::vector<DatasetMapperOutput> &batched_inputs,
	const InstancesList &detected_instances, bool do_postprocess) {
	assert(!is_training());

	auto images = preprocess_image(batched_inputs, m_backbone->size_divisibility());
	auto features = m_backbone(images.tensor());

	InstancesList results;
	if (detected_instances.empty()) {
		InstancesList proposals;
		if (m_proposal_generator) {
			proposals = get<0>(m_proposal_generator(images, features));
		}
		else {
			assert(batched_inputs[0].proposals);
			proposals = Instances::to<DatasetMapperOutput>(batched_inputs, device(),
				&DatasetMapperOutput::get_instances);
		}
		results = get<0>(m_roi_heads(images, features, proposals));
	}
	else {
		InstancesList converted = Instances::to(detected_instances, device());
		results = m_roi_heads->forward_with_given_boxes(features, converted);
	}

	if (do_postprocess) {
		return _postprocess(results, batched_inputs, images.image_sizes());
	}
	return results;
}
