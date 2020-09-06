#pragma once

#include <Detectron2/Detectron2.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from layers/batch_norm.py

	class BatchNormImpl {
	public:
		virtual ~BatchNormImpl() {}

		virtual torch::Tensor &get_weight() = 0;
		virtual torch::Tensor &get_bias() = 0;
		virtual torch::Tensor *get_running_mean() = 0;
		virtual torch::Tensor *get_running_var() = 0;

		virtual void initialize(const ModelImporter &importer, const std::string &prefix, ModelImporter::Fill fill);
		virtual torch::Tensor forward(torch::Tensor x) = 0;
	};

	class BatchNorm : public std::shared_ptr<BatchNormImpl> {
	public:
		enum Type {
			kNone,

			kBN,		// BatchNorm2d, Fixed in https ://github.com/pytorch/pytorch/pull/36382
			kSyncBN,	// NaiveSyncBatchNorm if TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
			kFrozenBN,	// FrozenBatchNorm2d,
			kGN,		// lambda channels : nn.GroupNorm(32, channels),

			// for debugging:
			nnSyncBN,	// nn.SyncBatchNorm,
			naiveSyncBN	// NaiveSyncBatchNorm,
		};

		static Type GetType(const std::string &name);

	public:
		/**
			Args:
				norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
					or a callable that takes a channel number and returns
					the normalization layer as a nn.Module.

			Returns:
				nn.Module or None: the normalization layer
		*/
		BatchNorm(std::nullptr_t) {}
		BatchNorm(Type type, int out_channels);

		template<typename T>
		std::shared_ptr<T> as() {
			return std::dynamic_pointer_cast<T>(*this);
		}
		ModulePtr asModule() { return as<torch::nn::Module>(); }

		torch::Tensor operator()(torch::Tensor x) {
			return get()->forward(x);
		}
	};
}
