#include "Base.h"
#include "BaseMaskRCNNHead.h"

#include <Detectron2/Utils/EventStorage.h>
#include <Detectron2/Structures/Masks.h>
#include "MaskRCNNConvUpsampleHead.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MaskHead Detectron2::build_mask_head(CfgNode &cfg, const ShapeSpec &input_shape) {
	return shared_ptr<BaseMaskRCNNHeadImpl>(new MaskRCNNConvUpsampleHeadImpl(cfg, input_shape));
}

torch::Tensor BaseMaskRCNNHeadImpl::mask_rcnn_loss(torch::Tensor pred_mask_logits, const InstancesList &instances,
	int vis_period) {
	auto cls_agnostic_mask = pred_mask_logits.size(1) == 1;
	auto total_num_masks = pred_mask_logits.size(0);
	auto mask_side_len = pred_mask_logits.size(2);
	assert(pred_mask_logits.size(2) == pred_mask_logits.size(3)); // Mask prediction must be square!

	TensorVec gt_classes; gt_classes.reserve(instances.size());
	TensorVec gt_masks; gt_masks.reserve(instances.size());
	for (auto &instances_per_image : instances) {
		if (instances_per_image->len() == 0) {
			continue;
		}
		if (!cls_agnostic_mask) {
			auto gt_classes_per_image = instances_per_image->getTensor("gt_classes").to(torch::kInt64);
			gt_classes.push_back(gt_classes_per_image);
		}

		auto masks = dynamic_pointer_cast<Masks>(instances_per_image->get("gt_masks"));
		auto gt_masks_per_image = masks->crop_and_resize(
			instances_per_image->getTensor("proposal_boxes"), mask_side_len
		).to(pred_mask_logits.device());
		// A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
		gt_masks.push_back(gt_masks_per_image);
	}

	if (gt_masks.empty()) {
		return pred_mask_logits.sum() * 0;
	}

	Tensor t_gt_masks = cat(gt_masks, 0);

	if (cls_agnostic_mask) {
		pred_mask_logits = pred_mask_logits.index({ Colon, 0 });
	}
	else {
		auto indices = torch::arange(total_num_masks);
		auto t_gt_classes = cat(gt_classes, 0);
		pred_mask_logits = pred_mask_logits.index({ indices, t_gt_classes });
	}

	Tensor gt_masks_bool;
	if (t_gt_masks.dtype() == torch::kBool) {
		gt_masks_bool = t_gt_masks;
	}
	else {
		// Here we allow t_gt_masks to be float as well (depend on the implementation of rasterize())
		gt_masks_bool = t_gt_masks > 0.5;
	}
	t_gt_masks = t_gt_masks.to(torch::kFloat32);

    // Log the training accuracy (using gt classes and 0.5 threshold)
	auto mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool;
	auto mask_accuracy = 1 - (mask_incorrect.sum().item<float>() / max((float)mask_incorrect.numel(), 1.0f));
	auto num_positive = gt_masks_bool.sum().item<float>();
	auto false_positive = (mask_incorrect.bitwise_and(gt_masks_bool.neg())).sum().item<float>() / max(
		(float)gt_masks_bool.numel() - num_positive, 1.0f
	);
	auto false_negative = (mask_incorrect.bitwise_and(gt_masks_bool)).sum().item<float>() / max(num_positive, 1.0f);

	auto &storage = get_event_storage();
	storage.put_scalar("mask_rcnn/accuracy", mask_accuracy);
	storage.put_scalar("mask_rcnn/false_positive", false_positive);
	storage.put_scalar("mask_rcnn/false_negative", false_negative);
	if (vis_period > 0 && storage.iter() % vis_period == 0) {
		auto pred_masks = pred_mask_logits.sigmoid();
		auto vis_masks = torch::cat({ pred_masks, t_gt_masks }, 2);
		string name = "Left: mask prediction;   Right: mask GT";
		for (int idx = 0; idx < vis_masks.size(0); idx++) {
			auto vis_mask = vis_masks[idx];
			vis_mask = torch::stack({ vis_mask, vis_mask, vis_mask }, 0);
			storage.put_image(name + FormatString(" ({%d})", idx), vis_mask);
		}
	}

	auto mask_loss = nn::functional::binary_cross_entropy_with_logits(pred_mask_logits, t_gt_masks,
		nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kMean));
	return mask_loss;
}

void BaseMaskRCNNHeadImpl::mask_rcnn_inference(const torch::Tensor &pred_mask_logits, InstancesList &pred_instances) {
	auto cls_agnostic_mask = pred_mask_logits.size(1) == 1;

	Tensor mask_probs_pred;
	if (cls_agnostic_mask) {
		mask_probs_pred = pred_mask_logits.sigmoid();
	}
	else {
		// Select masks corresponding to the predicted classes
		auto num_masks = pred_mask_logits.size(0);
		auto class_pred = cat(pred_instances.getTensorVec("pred_classes"));
		auto indices = torch::arange(num_masks, class_pred.device());
		mask_probs_pred = pred_mask_logits.index({ indices, class_pred }).index({ Colon, None }).sigmoid();
	}
    // mask_probs_pred.shape: (B, 1, Hmask, Wmask)

	auto num_boxes_per_image = pred_instances.getLenVec();
	auto splitted_mask_probs_pred = mask_probs_pred.split_with_sizes(num_boxes_per_image, 0);

	int count = splitted_mask_probs_pred.size();
	assert(pred_instances.size() == count);
	for (int i = 0; i < count; i++) {
		auto prob = splitted_mask_probs_pred[i];
		auto instances = pred_instances[i];
		instances->set("pred_masks", prob);  // (1, Hmask, Wmask)
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BaseMaskRCNNHeadImpl::BaseMaskRCNNHeadImpl(CfgNode &cfg) :
	m_vis_period(cfg["VIS_PERIOD"].as<int>())
{
}

std::tuple<TensorMap, InstancesList> BaseMaskRCNNHeadImpl::forward(torch::Tensor x, InstancesList &instances) {
	x = layers(x);
	if (is_training()) {
		return { { { "loss_mask", mask_rcnn_loss(x, instances, m_vis_period) } }, {} };
	}
	else {
		mask_rcnn_inference(x, instances);
		return { TensorMap{}, instances };
	}
}
