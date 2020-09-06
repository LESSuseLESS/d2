#include "Base.h"
#include "DefaultPredictor.h"

#include <Detectron2/Utils/Timer.h>
#include <Detectron2/Data/ResizeShortestEdge.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DefaultPredictor::DefaultPredictor(const CfgNode &cfg) : m_model(nullptr) {
	m_cfg = cfg.clone();  // cfg can be modified by model
	{
		Timer timer("build_model");
		m_model = build_model(m_cfg);
	}
	m_model->eval();

	auto name = CfgNode::parseTuple<string>(cfg["DATASETS.TEST"], { "" })[0];
	m_metadata = MetadataCatalog::get(name);
	{
		Timer timer("load_checkpoint");
		m_model->load_checkpoint(cfg["MODEL.WEIGHTS"].as<string>(""), false);
	}
	m_transform_gen = shared_ptr<TransformGen>(new ResizeShortestEdge(
		{ cfg["INPUT.MIN_SIZE_TEST"].as<int>(), cfg["INPUT.MIN_SIZE_TEST"].as<int>() },
		cfg["INPUT.MAX_SIZE_TEST"].as<int>()
	));

	m_input_format = cfg["INPUT.FORMAT"].as<string>();
	assert(m_input_format == "RGB" || m_input_format == "BGR");
}

InstancesPtr DefaultPredictor::predict(torch::Tensor original_image) {
	torch::NoGradGuard guard; // https://github.com/sphinx-doc/sphinx/issues/4258

	// Apply pre-processing to image.
	if (m_input_format == "RGB") {
		// whether the model expects BGR inputs or RGB
		original_image = torch::flip(original_image, { -1 });
	}
	auto height = original_image.size(0);
	auto width = original_image.size(1);
	auto image = m_transform_gen->get_transform(original_image)->apply_image(original_image);
	image = image.to(torch::kFloat32).permute({ 2, 0, 1 });

	std::vector<DatasetMapperOutput> inputs(1);
	inputs[0].image = image;
	inputs[0].height = make_shared<int>(height);
	inputs[0].width = make_shared<int>(width);
	InstancesPtr predictions;
	{
		Timer timer("forward");
		predictions = get<0>(m_model->forward(inputs))[0];
	}
	return predictions;
}
