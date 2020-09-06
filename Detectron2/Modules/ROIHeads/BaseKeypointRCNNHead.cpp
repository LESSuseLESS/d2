#include "Base.h"
#include "BaseKeypointRCNNHead.h"

#include <Detectron2/Utils/EventStorage.h>
#include <Detectron2/Structures/Keypoints.h>
#include "KRCNNConvDeconvUpsampleHead.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static int _TOTAL_SKIPPED = 0;

KeypointHead Detectron2::build_keypoint_head(CfgNode &cfg, const ShapeSpec &input_shape) {
	return shared_ptr<BaseKeypointRCNNHeadImpl>(new KRCNNConvDeconvUpsampleHeadImpl(cfg, input_shape));
}

Tensor BaseKeypointRCNNHeadImpl::keypoint_rcnn_loss(torch::Tensor pred_keypoint_logits,
	const InstancesList &instances, torch::optional<float> normalizer) {
	TensorVec heatmaps; heatmaps.reserve(instances.size());
	TensorVec valids; valids.reserve(instances.size());

	auto keypoint_side_len = pred_keypoint_logits.size(2);
	for (auto &instances_per_image : instances) {
		if (instances_per_image->len() == 0) {
			continue;
		}
		auto keypoints = instances_per_image->getTensor("gt_keypoints");
		Tensor heatmaps_per_image, valid_per_image;
		tie(heatmaps_per_image, valid_per_image) = Keypoints(keypoints).to_heatmap(
			instances_per_image->getTensor("proposal_boxes"), keypoint_side_len
		);
		heatmaps.push_back(heatmaps_per_image.view(-1));
		valids.push_back(valid_per_image.view(-1));
	}

	Tensor keypoint_targets;
	Tensor valid;
	if (!heatmaps.empty()) {
		keypoint_targets = cat(heatmaps, 0);
		valid = cat(valids, 0).to(torch::kUInt8);
		valid = torch::nonzero(valid).squeeze(1);
	}

	// torch.mean (in binary_cross_entropy_with_logits) doesn't
	// accept empty tensors, so handle it separately
	if (heatmaps.empty() or valid.numel() == 0) {
		_TOTAL_SKIPPED += 1;
		auto &storage = get_event_storage();
		storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, false);
		return pred_keypoint_logits.sum() * 0;
	}

	auto N = pred_keypoint_logits.size(0);
	auto K = pred_keypoint_logits.size(1);
	auto H = pred_keypoint_logits.size(2);
	auto W = pred_keypoint_logits.size(3);
	pred_keypoint_logits = pred_keypoint_logits.view({ N * K, H * W });

	auto keypoint_loss = nn::functional::cross_entropy(
		pred_keypoint_logits.index(valid), keypoint_targets.index(valid),
		nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum));

	// If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
	if (!normalizer.has_value()) {
		normalizer = valid.numel();
	}
	keypoint_loss /= *normalizer;

	return keypoint_loss;
}

void BaseKeypointRCNNHeadImpl::keypoint_rcnn_inference(const torch::Tensor &pred_keypoint_logits,
	InstancesList &pred_instances) {
	// flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
	auto bboxes = pred_instances.getTensorVec("pred_boxes");
	auto bboxes_flat = cat(bboxes, 0);

	auto keypoint_results = Keypoints::heatmaps_to_keypoints(pred_keypoint_logits, bboxes_flat);
	auto num_instances_per_image = pred_instances.getLenVec();
	TensorVec splitted = keypoint_results.index({ Colon, Colon, torch::tensor({0, 1, 3}) })
		.split_with_sizes(num_instances_per_image, 0);

	int count = splitted.size();
	assert(pred_instances.size() == count);
	for (int i = 0; i < count; i++) {
		auto keypoint_results_per_image = splitted[i];
		auto instances_per_image = pred_instances[i];
		// keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score)
		instances_per_image->set("pred_keypoints", keypoint_results_per_image);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BaseKeypointRCNNHeadImpl::BaseKeypointRCNNHeadImpl(CfgNode &cfg) :
	m_num_keypoints(cfg["MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS"].as<int>()),
	m_loss_weight(cfg["MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT"].as<float>())
{
	bool normalize_by_visible = cfg["MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS"].as<bool>();
	if (!normalize_by_visible) {
		auto batch_size_per_image = cfg["MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE"].as<int>();
		auto positive_sample_fraction = cfg["MODEL.ROI_HEADS.POSITIVE_FRACTION"].as<float>();
		m_loss_normalizer = m_num_keypoints * batch_size_per_image * positive_sample_fraction;
	}
}

std::tuple<TensorMap, InstancesList> BaseKeypointRCNNHeadImpl::forward(torch::Tensor x, InstancesList &instances) {
	x = layers(x);
	if (is_training()) {
		auto num_images = instances.size();
		auto normalizer = m_loss_normalizer;
		if (m_loss_normalizer.has_value()) {
			normalizer = num_images * (*m_loss_normalizer);
		}
		auto loss = keypoint_rcnn_loss(x, instances, normalizer) * m_loss_weight;
		return { { { "loss_keypoint", loss } }, {} };
	}
	else {
		keypoint_rcnn_inference(x, instances);
		return { TensorMap{}, instances };
	}
}
