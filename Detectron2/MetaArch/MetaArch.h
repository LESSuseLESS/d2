#pragma once

#include <Detectron2/Utils/CfgNode.h>
#include <Detectron2/Structures/ImageList.h>
#include <Detectron2/Structures/Instances.h>
#include <Detectron2/Modules/FPN/FPN.h>
#include <Detectron2/Modules/RPN/RPN.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	struct DatasetMapperOutput {
		torch::Tensor image;					// in(C, H, W) format
		std::shared_ptr<int> height;
		std::shared_ptr<int> width;
		InstancesPtr instances;					// groundtruth
		InstancesPtr proposals;					// precomputed proposals
		std::shared_ptr<torch::Tensor> sem_seg;	// groundtruth of semantic segments

		InstancesPtr get_instances() const { return instances; }
		InstancesPtr get_proposals() const { return proposals; }
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class MetaArchImpl : public torch::nn::Module {
	public:
		MetaArchImpl(CfgNode &cfg);
		virtual ~MetaArchImpl() {}
		void load_checkpoint(const std::string &checkpointer, bool jit);

		torch::Device device() const;

		virtual void initialize(const ModelImporter &importer, const std::string &prefix);
		virtual std::tuple<InstancesList, TensorMap>
			forward(const std::vector<DatasetMapperOutput> &batched_inputs) = 0;

	protected:
		Backbone m_backbone{ nullptr };
		RPN m_proposal_generator{ nullptr };

		// The period(in terms of steps) for minibatch visualization at train time.
		// Set to 0 to disable.
		int m_vis_period;

		// Whether the model needs RGB, YUV, HSV etc.
		// Should be one of the modes defined here, as we use PIL to read the image:
		// https://pillow.readthedocs.io/en/stable/handbook/concepts.html//concept-modes
		// with BGR being the one exception. One can set image format to BGR, we will
		// internally use RGB for conversion and flip the channels over
		std::string m_input_format;

		torch::Tensor m_pixel_mean;
		torch::Tensor m_pixel_std;

		// Normalize, pad and batch the input images.
		ImageList preprocess_image(const std::vector<DatasetMapperOutput> &batched_inputs, int size_divisibility);

		InstancesList get_gt_instances(const std::vector<DatasetMapperOutput> &batched_inputs);
		torch::Tensor get_gt_sem_seg(const std::vector<DatasetMapperOutput> &batched_inputs, double ignore_value);

		// Rescale the output instances to the target size.
		// note: private function; subject to changes
		InstancesList _postprocess(const InstancesList &instances,
			const std::vector<DatasetMapperOutput> &batched_inputs,
			const std::vector<ImageSize> &image_sizes);
	};
	TORCH_MODULE(MetaArch);

	/**
		Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
		Note that it does not load any weights from ``cfg``.
	*/
	MetaArch build_model(CfgNode &cfg);
}