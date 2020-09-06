#include "Base.h"
#include "SemanticSegmentor.h"

#include <Detectron2/Structures/PostProcessing.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

SemanticSegmentorImpl::SemanticSegmentorImpl(CfgNode &cfg) : MetaArchImpl(cfg) {
	m_sem_seg_head = make_shared<SemSegFPNHeadImpl>(cfg, m_backbone->output_shapes());
	register_module("sem_seg_head", m_sem_seg_head);
}

void SemanticSegmentorImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	MetaArchImpl::initialize(importer, prefix);
	m_sem_seg_head->initialize(importer, "sem_seg_head");
}

std::tuple<InstancesList, TensorMap> SemanticSegmentorImpl::forward(
	const std::vector<DatasetMapperOutput> &batched_inputs) {
	auto images = preprocess_image(batched_inputs, m_backbone->size_divisibility());
	auto features = m_backbone(images.tensor());

	auto gt_sem_seg = get_gt_sem_seg(batched_inputs, m_sem_seg_head->ignore_value());
	Tensor results;
	TensorMap losses;
	tie(results, losses) = m_sem_seg_head(features, gt_sem_seg);

	if (is_training()) {
		return { InstancesList{}, losses };
	}

	int count = batched_inputs.size();
	assert(results.size(0) == count);
	auto &image_sizes = images.image_sizes();
	assert(image_sizes.size() == count);

	InstancesList processed_results;
	for (int i = 0; i < count; i++) {
		auto result = results[i];
		auto &input_per_image = batched_inputs[i];
		auto &image_size = image_sizes[i];

		int height = input_per_image.height ? *input_per_image.height : image_size.height;
		int width = input_per_image.width ? *input_per_image.width : image_size.width;

		auto sem_seg_r = PostProcessing::sem_seg_postprocess(result, image_size, height, width);
		auto m = make_shared<Instances>(ImageSize{ height, width });
		m->set("sem_seg", sem_seg_r);
		processed_results.push_back(m);
	}
	return { processed_results, {} };
}
