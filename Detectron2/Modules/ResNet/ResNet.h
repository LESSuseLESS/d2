#pragma once

#include <Detectron2/Modules/Backbone.h>
#include "CNNBlockBase.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/backbone/resnet.py

	// Implement :paper:`ResNet`.
	class ResNetImpl : public BackboneImpl {
	public:
		/**
			stem (nn.Module): a stem module
			stages (list[list[CNNBlockBase]]): several (typically 4) stages,
				each contains multiple :class:`CNNBlockBase`.
			num_classes (None or int): if None, will not perform classification.
				Otherwise, will create a linear layer.
			out_features (list[str]): name of the layers whose outputs should
				be returned in forward. Can be anything in "stem", "linear", or "res2" ...
				If None, will return the output of the last layer.
		*/
		ResNetImpl(const CNNBlockBase &stem, const std::vector<std::vector<CNNBlockBase>> &stages,
			const std::unordered_set<std::string> &out_features, int num_classes = 0);

		virtual void initialize(const ModelImporter &importer, const std::string &prefix) override;

		virtual TensorMap forward(torch::Tensor x) override;

		/**
			Freeze the first several stages of the ResNet. Commonly used in
			fine-tuning.

			Layers that produce the same feature map spatial size are defined as one
			"stage" by :paper:`FPN`.

			Args:
				freeze_at (int): number of stages to freeze.
					`1` means freezing the stem. `2` means freezing the stem and
					one residual stage, etc.

			Returns:
				nn.Module: this ResNet itself
		*/
		std::shared_ptr<ResNetImpl> freeze(int freeze_at = 0);

	private:
		CNNBlockBase m_stem;
		std::unordered_set<std::string> m_out_features;
		int m_num_classes;

		std::vector<std::string> m_names;
		std::vector<torch::nn::Sequential> m_stages;

		torch::nn::AdaptiveAvgPool2d m_avgpool{ nullptr };
		torch::nn::Linear m_linear{ nullptr };
	};
	TORCH_MODULE(ResNet);

    /**
		Create a ResNet instance from config.

		Returns:
			ResNet: a :class:`ResNet` instance.
	*/
	Backbone build_resnet_backbone(CfgNode &cfg, const ShapeSpec &input_shape);

	/**
		Create a list of blocks just like those in a ResNet stage.

		Args:
			block_class (type): a subclass of ResNetBlockBase
			num_blocks (int):
			first_stride (int): the stride of the first block. The other blocks will have stride=1.
			in_channels (int): input channels of the entire stage.
			out_channels (int): output channels of **every block** in the stage.
			kwargs: other arguments passed to the constructor of every block.

		Returns:
			list[nn.Module]: a list of block module.
	*/
	/**
		* no need to convert: one should just manually call something like "new BottleneckBlockImpl(...)"

		static std::vector<torch::nn::Module> make_stage(block_class, ...);
		blocks = []
		for i in range(num_blocks):
			blocks.append(
				block_class(
					in_channels=in_channels,
					out_channels=out_channels,
					stride=first_stride if i == 0 else 1,		<= NOTE
					**kwargs,
				)
			)
			in_channels = out_channels							<= NOTE
		return blocks
	*/
}