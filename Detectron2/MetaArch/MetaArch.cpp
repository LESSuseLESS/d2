#include "Base.h"
#include "MetaArch.h"

#include <Detectron2/Structures/PostProcessing.h>
#include <Detectron2/MetaArch/GeneralizedRCNN.h>
#include <Detectron2/MetaArch/PanopticFPN.h>
#include <Detectron2/MetaArch/ProposalNetwork.h>
#include <Detectron2/MetaArch/SemanticSegmentor.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MetaArch Detectron2::build_model(CfgNode &cfg) {
	MetaArch model{ nullptr };

	string meta_arch = cfg["MODEL.META_ARCHITECTURE"].as<string>();
	if (meta_arch == "GeneralizedRCNN") {
		model = shared_ptr<MetaArchImpl>(new GeneralizedRCNNImpl(cfg));
	}
	else if (meta_arch == "PanopticFPN") {
		model = shared_ptr<MetaArchImpl>(new PanopticFPNImpl(cfg));
	}
	else if (meta_arch == "ProposalNetwork") {
		model = shared_ptr<MetaArchImpl>(new ProposalNetworkImpl(cfg));
	}
	else if (meta_arch == "SemanticSegmentor") {
		model = shared_ptr<MetaArchImpl>(new SemanticSegmentorImpl(cfg));
	} else {
		assert(false);
	}

	auto device = torch::Device(cfg["MODEL.DEVICE"].as<string>());
	if (device == torch::kCUDA && !Detectron2::cudaEnabled()) {
		device = torch::kCPU;
	}
	model->to(device);
	return model;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MetaArchImpl::MetaArchImpl(CfgNode &cfg) :
	m_vis_period(cfg["VIS_PERIOD"].as<int>()),
	m_input_format(cfg["INPUT.FORMAT"].as<string>())
{
	m_backbone = build_backbone(cfg);
	register_module("backbone", m_backbone);

	m_proposal_generator = build_proposal_generator(cfg, m_backbone->output_shapes());
	register_module("proposal_generator", m_proposal_generator);

	auto pixel_mean = cfg["MODEL.PIXEL_MEAN"].as<vector<float>>();
	auto pixel_std = cfg["MODEL.PIXEL_STD"].as<vector<float>>();
	assert(pixel_mean.size() == pixel_std.size());
	m_pixel_mean = register_buffer("pixel_mean", torch::tensor(pixel_mean).view({ -1, 1, 1 }));
	m_pixel_std = register_buffer("pixel_std", torch::tensor(pixel_std).view({ -1, 1, 1 }));
}

void MetaArchImpl::load_checkpoint(const std::string &checkpointer, bool jit) {
	if (!checkpointer.empty()) {
		auto basename = File::Basename(checkpointer);
		if (jit) {
			// TODO: this doesn't work yet:
			string filename = File::ComposeFilename(ModelImporter::DataDir(), basename);
			torch::serialize::InputArchive archive;
			archive.load_from(filename);
			load(archive);
		}
		else {
			ModelImporter importer(basename);
			initialize(importer, "");
			auto count = importer.ReportUnimported();
			assert(count == 0);
		}
	}
	else {
		ModelImporter importer(ModelImporter::kNone);
		initialize(importer, "");
	}
}

void MetaArchImpl::initialize(const ModelImporter &importer, const std::string &prefix) {
	assert(prefix.empty());
	m_backbone->initialize(importer, "backbone");
	m_proposal_generator->initialize(importer, "proposal_generator");
}

torch::Device MetaArchImpl::device() const {
	return m_pixel_mean.device();
}

ImageList MetaArchImpl::preprocess_image(const std::vector<DatasetMapperOutput> &batched_inputs,
	int size_divisibility) {
	TensorVec images;
	images.reserve(batched_inputs.size());
	auto dev = device();
	for (auto &x : batched_inputs) {
		Tensor t = x.image.to(dev);
		t = (t - m_pixel_mean) / m_pixel_std;
		images.push_back(t);
	}
	return ImageList::from_tensors(images, size_divisibility);
}

InstancesList MetaArchImpl::get_gt_instances(const std::vector<DatasetMapperOutput> &batched_inputs) {
	InstancesList gt_instances;
	if (batched_inputs[0].instances) {
		gt_instances = Instances::to<DatasetMapperOutput>(batched_inputs, device(),
			&DatasetMapperOutput::get_instances);
	}
	//elif "targets" in batched_inputs[0]:
	//	log_first_n(
	//		logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
	//	)
	//	gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
	return gt_instances;
}

Tensor MetaArchImpl::get_gt_sem_seg(const std::vector<DatasetMapperOutput> &batched_inputs, double ignore_value) {
	if (batched_inputs[0].sem_seg) {
		TensorVec gt_sem_segs;
		gt_sem_segs.reserve(batched_inputs.size());
		for (auto &item : batched_inputs) {
			gt_sem_segs.push_back(item.sem_seg->to(device()));
		}
		return ImageList::from_tensors(gt_sem_segs, m_backbone->size_divisibility(), ignore_value).tensor();
	}
	return Tensor();
}

InstancesList MetaArchImpl::_postprocess(const InstancesList &instances,
	const std::vector<DatasetMapperOutput> &batched_inputs, const std::vector<ImageSize> &image_sizes) {
	int count = batched_inputs.size();
	assert(instances.size() == count);
	assert(image_sizes.size() == count);

	InstancesList processed_results;
	for (int i = 0; i < count; i++) {
		auto &results_per_image = instances[i];
		auto &input_per_image = batched_inputs[i];
		auto &image_size = image_sizes[i];

		int height = input_per_image.height ? *input_per_image.height : image_size.height;
		int width = input_per_image.width ? *input_per_image.width : image_size.width;
		auto r = PostProcessing::detector_postprocess(results_per_image, height, width);
		auto m = make_shared<Instances>(ImageSize{ height, width });
		m->set("instances", r);
		processed_results.push_back(m);
	}
	return processed_results;
};
