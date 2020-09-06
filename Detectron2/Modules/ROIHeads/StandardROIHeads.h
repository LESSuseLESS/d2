#pragma once

#include "ROIHeads.h"
#include <Detectron2/Modules/ROIPooler/ROIPooler.h>
#include "BaseMaskRCNNHead.h"
#include "BaseKeypointRCNNHead.h"
#include "FastRCNNConvFCHead.h"
#include "FastRCNNOutputLayers.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/roi_heads/roi_heads.py

	/**
		It's "standard" in a sense that there is no ROI transform sharing
		or feature sharing between tasks.
		Each head independently processes the input features by each head's
		own pooler and head.

		This class is used by most models, such as FPN and C5.
		To implement more models, you can subclass it and implement a different
		:meth:`forward()` or a head.
	*/
	class StandardROIHeadsImpl : public ROIHeadsImpl {
	public:
		/**
			NOTE: this interface is experimental.

			Args:
				box_in_features (list[str]): list of feature names to use for the box head.
				box_pooler (ROIPooler): pooler to extra region features for box head
				box_head (nn.Module): transform features to make box predictions
				box_predictor (nn.Module): make box predictions from the feature.
					Should have the same interface as :class:`FastRCNNOutputLayers`.
				mask_in_features (list[str]): list of feature names to use for the mask head.
					None if not using mask head.
				mask_pooler (ROIPooler): pooler to extra region features for mask head
				mask_head (nn.Module): transform features to make mask predictions
				keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask*``.
				train_on_pred_boxes (bool): whether to use proposal boxes or
					predicted boxes from the box head to train other heads.
		*/
		StandardROIHeadsImpl(CfgNode &cfg);
		void Create(CfgNode &cfg, const ShapeSpec::Map &input_shapes);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;

		virtual std::tuple<InstancesList, TensorMap> forward(const ImageList &images, const TensorMap &features,
			InstancesList &proposals, const InstancesList &targets = {}) override;

		/**
		Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

		This is useful for downstream tasks where a box is known, but need to obtain
		other attributes (outputs of other heads).
		Test-time augmentation also uses this.

		Args:
			features: same as in `forward()`
			instances (list[Instances]): instances to predict other outputs. Expect the keys
				"pred_boxes" and "pred_classes" to exist.

		Returns:
			instances (list[Instances]):
				the same `Instances` objects, with extra
				fields such as `pred_masks` or `pred_keypoints`.
		*/
		virtual InstancesList forward_with_given_boxes(const TensorMap &features, InstancesList &instances) override;

	protected:
		std::vector<std::string> m_box_in_features;
		ROIPooler m_box_pooler{ nullptr };
		std::vector<BoxHead> m_box_heads;
		std::vector<FastRCNNOutputLayers> m_box_predictors;

		bool m_mask_on;
		std::vector<std::string> m_mask_in_features;
		ROIPooler m_mask_pooler{ nullptr };
		MaskHead m_mask_head{ nullptr };

		bool m_keypoint_on;
		std::vector<std::string> m_keypoint_in_features;
		ROIPooler m_keypoint_pooler{ nullptr };
		KeypointHead m_keypoint_head{ nullptr };
		bool m_train_on_pred_boxes;

		int64_t select_channels(const ShapeSpec::Map &input_shapes, const std::vector<std::string> &in_features);
		TensorVec select_features(const TensorMap &features, const std::vector<std::string> &in_features);
		std::vector<float> get_pooler_scales(const ShapeSpec::Map &input_shapes,
			const std::vector<std::string> &in_features);
		virtual void _init_box_head(CfgNode &cfg, const ShapeSpec::Map &input_shapes);
		virtual void _init_mask_head(CfgNode &cfg, const ShapeSpec::Map &input_shapes);
		virtual void _init_keypoint_head(CfgNode &cfg, const ShapeSpec::Map &input_shapes);

		/**
			Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
				the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

			Args:
				features (dict[str, Tensor]): mapping from feature map names to tensor.
					Same as in :meth:`ROIHeads.forward`.
				proposals (list[Instances]): the per-image object proposals with
					their matching ground truth.
					Each has fields "proposal_boxes", and "objectness_logits",
					"gt_classes", "gt_boxes".

			Returns:
				In training, a dict of losses.
				In inference, a list of `Instances`, the predicted instances.
		*/
		std::tuple<TensorMap, InstancesList> _forward_box(const TensorMap &features, InstancesList &proposals);

		/**
			Forward logic of the mask prediction branch.

			Args:
				features (dict[str, Tensor]): mapping from feature map names to tensor.
					Same as in :meth:`ROIHeads.forward`.
				instances (list[Instances]): the per-image instances to train/predict masks.
					In training, they can be the proposals.
					In inference, they can be the predicted boxes.

			Returns:
				In training, a dict of losses.
				In inference, update `instances` with new fields "pred_masks" and return it.
		*/
		std::tuple<TensorMap, InstancesList> _forward_mask(const TensorMap &features, InstancesList &instances);

		/**
			Forward logic of the keypoint prediction branch.

			Args:
				features (dict[str, Tensor]): mapping from feature map names to tensor.
					Same as in :meth:`ROIHeads.forward`.
				instances (list[Instances]): the per-image instances to train/predict keypoints.
					In training, they can be the proposals.
					In inference, they can be the predicted boxes.

			Returns:
				In training, a dict of losses.
				In inference, update `instances` with new fields "pred_keypoints" and return it.
		*/
		std::tuple<TensorMap, InstancesList> _forward_keypoint(const TensorMap &features, InstancesList &instances);
	};
}