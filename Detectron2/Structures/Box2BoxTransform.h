#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/box_regression.py

	/**
		The box-to-box transform defined in R-CNN. The transformation is parameterized
		by 4 deltas: (dx, dy, dw, dh). The transformation scales the box's width and height
		by exp(dw), exp(dh) and shifts a box's center by the offset (dx * width, dy * height).
	*/
	class Box2BoxTransform {
	public:
		struct Weights {
			float dx;
			float dy;
			float dw;
			float dh;

			float da; // only used by Box2BoxTransformRotated
		};

		static const float _DEFAULT_SCALE_CLAMP;

		static std::shared_ptr<Box2BoxTransform> Create(YAML::Node node,
			float scale_clamp = _DEFAULT_SCALE_CLAMP);
		static std::shared_ptr<Box2BoxTransform> Create(const std::vector<float> &w,
			float scale_clamp = _DEFAULT_SCALE_CLAMP);

		/**
			Args:
				weights (4-element tuple): Scaling factors that are applied to the
					(dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
					such that the deltas have unit variance; now they are treated as
					hyperparameters of the system.
				scale_clamp (float): When predicting deltas, the predicted box scaling
					factors (dw and dh) are clamped such that they are <= scale_clamp.
		*/
		Box2BoxTransform(const Weights &weights, float scale_clamp = _DEFAULT_SCALE_CLAMP);
		virtual ~Box2BoxTransform() {}

		virtual int box_dim() const { return 4; }

		/**
			Get box regression transformation deltas (dx, dy, dw, dh) that can be used
			to transform the `src_boxes` into the `target_boxes`. That is, the relation
			``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
			any delta is too large and is clamped).

			Args:
				src_boxes (Tensor): source boxes, e.g., object proposals
				target_boxes (Tensor): target of the transformation, e.g., ground-truth
					boxes.
		*/
		virtual torch::Tensor get_deltas(torch::Tensor src_boxes, torch::Tensor target_boxes);

		/**
			Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

			Args:
				deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
					deltas[i] represents k potentially different class-specific
					box transformations for the single box boxes[i].
				boxes (Tensor): boxes to transform, of shape (N, 4)
		*/
		virtual torch::Tensor apply_deltas(torch::Tensor deltas, torch::Tensor boxes);

		/**
			Apply transform deltas to boxes. Similar to `box2box_transform.apply_deltas`,
			but allow broadcasting boxes when the second dimension of deltas is a multiple
			of box dimension.

			Args:
				box2box_transform (Box2BoxTransform or Box2BoxTransformRotated): the transform to apply
				deltas (Tensor): tensor of shape (N,B) or (N,KxB)
				boxes (Tensor): tensor of shape (N,B)

			Returns:
				Tensor: same shape as deltas.
		*/
		torch::Tensor apply_deltas_broadcast(torch::Tensor deltas, torch::Tensor boxes);

	protected:
		Weights m_weights;
		float m_scale_clamp;
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
		The box-to-box transform defined in Rotated R-CNN. The transformation is parameterized
		by 5 deltas: (dx, dy, dw, dh, da). The transformation scales the box's width and height
		by exp(dw), exp(dh), shifts a box's center by the offset (dx * width, dy * height),
		and rotate a box's angle by da (radians).
		Note: angles of deltas are in radians while angles of boxes are in degrees.
	*/
	class Box2BoxTransformRotated : public Box2BoxTransform {
	public:
		/**
			Args:
				weights (5-element tuple): Scaling factors that are applied to the
					(dx, dy, dw, dh, da) deltas. These are treated as
					hyperparameters of the system.
				scale_clamp (float): When predicting deltas, the predicted box scaling
					factors (dw and dh) are clamped such that they are <= scale_clamp.
		*/
		Box2BoxTransformRotated(const Weights &weights, float scale_clamp = _DEFAULT_SCALE_CLAMP);

		virtual int box_dim() const override { return 5; }

		/**
			Get box regression transformation deltas (dx, dy, dw, dh, da) that can be used
			to transform the `src_boxes` into the `target_boxes`. That is, the relation
			``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
			any delta is too large and is clamped).

			Args:
				src_boxes (Tensor): Nx5 source boxes, e.g., object proposals
				target_boxes (Tensor): Nx5 target of the transformation, e.g., ground-truth
					boxes.
		*/
		virtual torch::Tensor get_deltas(torch::Tensor src_boxes, torch::Tensor target_boxes) override;

		/**
			Apply transformation `deltas` (dx, dy, dw, dh, da) to `boxes`.

			Args:
				deltas (Tensor): transformation deltas of shape (N, 5).
					deltas[i] represents box transformation for the single box boxes[i].
				boxes (Tensor): boxes to transform, of shape (N, 5)
		*/
		virtual torch::Tensor apply_deltas(torch::Tensor deltas, torch::Tensor boxes) override;
	};
}