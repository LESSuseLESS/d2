#pragma once

#include "BatchNorm.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from layers/batch_norm.py

	/**
		BatchNorm2d where the batch statistics and the affine parameters are fixed.

		It contains non-trainable buffers called
		"weight" and "bias", "running_mean", "running_var",
		initialized to perform identity transformation.

		The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
		which are computed from the original four parameters of BN.
		The affine transform `x * weight + bias` will perform the equivalent
		computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
		When loading a backbone model from Caffe2, "running_mean" and "running_var"
		will be left unchanged as identity transformation.

		Other pre-trained backbone models may contain all 4 parameters.

		The forward is implemented by `F.batch_norm(..., training=False)`.
	*/
	class FrozenBatchNorm2dImpl : public torch::nn::Module, public BatchNormImpl {
	public:
		/**
			Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

			Args:
				module (torch.nn.Module):

			Returns:
				If module is BatchNorm/SyncBatchNorm, returns a new module.
				Otherwise, in-place convert module and return it.

			Similar to convert_sync_batchnorm in
			https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
		*/
		static ModulePtr convert_frozen_batchnorm(const ModulePtr &mod);

	public:
		FrozenBatchNorm2dImpl(int num_features, double eps = 1e-5);

		std::string toString() const;

		// implementing BatchNormImpl
		virtual torch::Tensor &get_weight() override		{ return m_weight; }
		virtual torch::Tensor &get_bias() override			{ return m_bias; }
		virtual torch::Tensor *get_running_mean() override	{ return &m_running_mean; }
		virtual torch::Tensor *get_running_var() override	{ return &m_running_var; }
		virtual torch::Tensor forward(torch::Tensor x) override;

	private:
		int m_num_features;
		double m_eps;

		torch::Tensor m_weight;
		torch::Tensor m_bias;
		torch::Tensor m_running_mean;
		torch::Tensor m_running_var;
	};
	TORCH_MODULE(FrozenBatchNorm2d);
}
