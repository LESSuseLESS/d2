#include "Base.h"
#include "ProposalNetwork.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ProposalNetworkImpl::ProposalNetworkImpl(CfgNode &cfg) : MetaArchImpl(cfg) {
}

std::tuple<InstancesList, TensorMap> ProposalNetworkImpl::forward(
	const std::vector<DatasetMapperOutput> &batched_inputs) {
	auto images = preprocess_image(batched_inputs, m_backbone->size_divisibility());
	auto features = m_backbone(images.tensor());

	InstancesList gt_instances = get_gt_instances(batched_inputs);

	InstancesList proposals; TensorMap proposal_losses;
	tie(proposals, proposal_losses) = m_proposal_generator(images, features, gt_instances);

	// In training, the proposals are not useful at all but we generate them anyway.
	// This makes RPN-only models about 5% slower.
	if (is_training()) {
		return { InstancesList{}, proposal_losses };
	}

	return { _postprocess(proposals, batched_inputs, images.image_sizes()), {} };
}