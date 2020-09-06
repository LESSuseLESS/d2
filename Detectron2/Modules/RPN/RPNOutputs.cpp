#include "Base.h"
#include "RPNOutputs.h"

#include <Detectron2/fvcore/fvcore.h>
#include <Detectron2/Utils/EventStorage.h>
#include <Detectron2/Structures/Boxes.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor> RPNOutputs::rpn_losses(
	const torch::Tensor &gt_labels, const torch::Tensor &gt_anchor_deltas,
	const torch::Tensor &pred_objectness_logits, const torch::Tensor &pred_anchor_deltas,
	float smooth_l1_beta) {
	auto pos_masks = (gt_labels == 1);

	assert(false);
	auto localization_loss = fvcore::smooth_l1_loss(
		pred_anchor_deltas.index(pos_masks), gt_anchor_deltas.index(pos_masks), smooth_l1_beta,
		torch::Reduction::Sum
	);

	auto valid_masks = (gt_labels >= 0);
	auto objectness_loss = binary_cross_entropy_with_logits(
		pred_objectness_logits.index(valid_masks),
		gt_labels.index(valid_masks).to(torch::kFloat32), {}, {}, torch::Reduction::Sum);
	return { objectness_loss, localization_loss };
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RPNOutputs::RPNOutputs(const std::shared_ptr<Box2BoxTransform> &box2box_transform, int batch_size_per_image,
	const ImageList &images, const TensorVec &pred_objectness_logits, const TensorVec &pred_anchor_deltas,
	BoxesList anchors, TensorVec gt_labels, BoxesList gt_boxes, float smooth_l1_beta) :
	m_box2box_transform(box2box_transform),
	m_batch_size_per_image(batch_size_per_image),
	m_anchors(std::move(anchors)),
	m_gt_boxes(std::move(gt_boxes)),
	m_gt_labels(std::move(gt_labels)),
	m_num_images(images.length()),
	m_smooth_l1_beta(smooth_l1_beta)
{
	auto B = m_anchors[0].size(1);  // box dimension (4 or 5)

	// Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
	m_pred_objectness_logits = vapply<Tensor>(pred_objectness_logits, [](Tensor score){
		return score.permute({ 0, 2, 3, 1 }).flatten(1);
		});

	// Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B)
	//          -> (N, Hi*Wi*A, B)
	m_pred_anchor_deltas = vapply<Tensor>(pred_anchor_deltas, [=](Tensor x){
		return x.view({ x.size(0), -1, B, x.size(-2), x.size(-1) })
			.permute({ 0, 3, 4, 1, 2 })
			.flatten(1, -2);
		});
}

TensorMap RPNOutputs::losses() {
	auto gt_labels = torch::stack(m_gt_labels);
	auto anchors = Boxes::cat(m_anchors);  // Ax(4 or 5)
	auto gt_anchor_deltas = Boxes::cat(vapply<Tensor>(m_gt_boxes, [=](Tensor k){
		return m_box2box_transform->get_deltas(anchors, k);
	}));
	gt_anchor_deltas = torch::stack(gt_anchor_deltas);

	// Log the number of positive/negative anchors per-image that's used in training
	auto num_pos_anchors = (gt_labels == 1).sum().item<float>();
	auto num_neg_anchors = (gt_labels == 0).sum().item<float>();
	auto &storage = get_event_storage();
	storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / m_num_images);
	storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / m_num_images);

	Tensor objectness_loss, localization_loss;
	tie(objectness_loss, localization_loss) = rpn_losses(
		gt_labels,
		gt_anchor_deltas,
		// concat on the Hi*Wi*A dimension
		cat(m_pred_objectness_logits, 1),
		cat(m_pred_anchor_deltas, 1),
		m_smooth_l1_beta);
	auto normalizer = m_batch_size_per_image * m_num_images;
	return {
		{ "loss_rpn_cls", objectness_loss / normalizer },
		{ "loss_rpn_loc", localization_loss / normalizer },
	};
}

TensorVec RPNOutputs::predict_proposals() {
	TensorVec proposals;
	int count = m_anchors.size();
	assert(m_pred_anchor_deltas.size() == count);
	// For each feature map
	for (int i = 0; i < count; i++) {
		auto &anchors_i = m_anchors[i];
		auto &pred_anchor_deltas_i = m_pred_anchor_deltas[i];

		auto B = anchors_i.size(1);
		auto N = m_num_images;
		pred_anchor_deltas_i = pred_anchor_deltas_i.reshape({ -1, B });
		// Expand anchors to shape (N*Hi*Wi*A, B)
		anchors_i = anchors_i.unsqueeze(0).expand({ N, -1, -1 }).reshape({ -1, B });
		auto proposals_i = m_box2box_transform->apply_deltas(pred_anchor_deltas_i, anchors_i);
		// Append feature map proposals with shape (N, Hi*Wi*A, B)
		proposals.push_back(proposals_i.view({ N, -1, B }));
	}
	return proposals;
}
