#pragma once

#include "MetaArch.h"
#include <Detectron2/Modules/FPN/FPN.h>
#include <Detectron2/Modules/RPN/RPN.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from modeling/meta_arch/rcnn.py

	// A meta architecture that only predicts object proposals.
	class ProposalNetworkImpl : public MetaArchImpl {
	public:
		ProposalNetworkImpl(CfgNode &cfg);

        /**
			Args:
				Same as in :class:`GeneralizedRCNN.forward`

			Returns:
				list[dict]:
					Each dict is the output for one input image.
					The dict contains one key "proposals" whose value is a
					:class:`Instances` with keys "proposal_boxes" and "objectness_logits".
		*/
		virtual std::tuple<InstancesList, TensorMap>
			forward(const std::vector<DatasetMapperOutput> &batched_inputs) override;
	};
	TORCH_MODULE(ProposalNetwork);
}