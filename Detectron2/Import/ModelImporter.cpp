#include "Base.h"
#include "ModelImporter.h"

#include <Detectron2/Modules/BatchNorm/BatchNorm.h>

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ModelImporter::FillTensor(Tensor x, ModelImporter::Fill fill) {
	switch (fill) {
	case kNoFill:			break;
	case kZeroFill:			torch::nn::init::constant_(x, 0);			break;
	case kConstantFill:		torch::nn::init::constant_(x, 1);			break;
	case kNormalFill2:		torch::nn::init::normal_(x, 0.0, 0.01);		break;
	case kNormalFill3:		torch::nn::init::normal_(x, 0.0, 0.001);	break;
	case kXavierNormalFill:	torch::nn::init::xavier_normal_(x);			break;
	case kCaffe2XavierFill:	torch::nn::init::kaiming_uniform_(x, 1);	break;
	case kCaffe2MSRAFill:	torch::nn::init::kaiming_normal_(x, 0.0, torch::kFanOut, torch::kReLU); break;
	case kCaffe2MSRAFillIn: torch::nn::init::kaiming_normal_(x, 0.0, torch::kFanIn, torch::kReLU); break;
	default:
		assert(false);
	}
}

ModelImporter::Model ModelImporter::FilenameToModel(const std::string &filename) {
	static unordered_map<string, Model> s_models = {
		{ "model_final_f10217.pkl", kDemo },
		{ "model_final_f6e8b1.pkl", kCOCODetection },
		{ "model_final_997cc7.pkl", kCOCOKeypoints },
		{ "model_final_a3ec72.pkl", kCOCOInstanceSegmentation },
		{ "model_final_cafdb1.pkl", kCOCOPanopticSegmentation }
	};
	auto iter = s_models.find(filename);
	assert(iter != s_models.end());
	return iter->second;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ModelImporter::ModelImporter(const std::string &filename) : ModelImporter(FilenameToModel(filename)) {
}

ModelImporter::ModelImporter(Model model) {
	std::string fullpath;
	switch (model) {
	case kDemo:							fullpath = import_model_final_f10217(); break;
	case kCOCODetection:				fullpath = import_model_final_f6e8b1(); break;
	case kCOCOKeypoints:				fullpath = import_model_final_997cc7(); break;
	case kCOCOInstanceSegmentation:		fullpath = import_model_final_a3ec72(); break;
	case kCOCOPanopticSegmentation:		fullpath = import_model_final_cafdb1(); break;
	}
	if (!fullpath.empty()) {
		m_fullpath = fullpath;
		m_fdata = make_shared<File>(fullpath, true);
	}
}

std::string ModelImporter::DataDir() {
	auto dir = getenv("D2_CHECKPOINTS_DIR");
	assert(dir);
	return dir;
}

void ModelImporter::Add(const char *name, int count) {
	m_sections[name] = { m_size, count };
	m_size += count;
}

void ModelImporter::Import(const std::string &name, torch::nn::Conv2d &conv, Fill fill) const {
	if (m_fdata) {
		Initialize(name + ".weight", conv->weight);
		if (conv->options.bias()) {
			Initialize(name + ".bias", conv->bias);
		}
	}
	else {
		FillTensor(conv->weight, fill);
		if (conv->options.bias()) {
			FillTensor(conv->bias, kZeroFill);
		}
	}
}

void ModelImporter::Import(const std::string &name, torch::nn::ConvTranspose2d &conv, Fill fill) const {
	if (m_fdata) {
		Initialize(name + ".weight", conv->weight);
		if (conv->options.bias()) {
			Initialize(name + ".bias", conv->bias);
		}
	}
	else {
		FillTensor(conv->weight, fill);
		if (conv->options.bias()) {
			FillTensor(conv->bias, kZeroFill);
		}
	}
}

void ModelImporter::Import(const std::string &name, torch::nn::Linear &fc, Fill fill) const {
	if (m_fdata) {
		Initialize(name + ".weight", fc->weight);
		Initialize(name + ".bias", fc->bias);
	}
	else {
		FillTensor(fc->weight, fill);
		FillTensor(fc->bias, kZeroFill);
	}
}

void ModelImporter::Initialize(const std::string &name, torch::Tensor &tensor) const {
	assert(m_imported.find(name) == m_imported.end());
	m_imported.insert(name);

	const auto iter = m_sections.find(name);
	assert(iter != m_sections.end());

	const auto &pos = iter->second;
	auto offset = pos.first;
	auto size = pos.second;
	int count = tensor.numel();
	assert(count == size);
	assert(tensor.dtype() == torch::kFloat32);

	count = size * sizeof(float);
	auto p = (char*)malloc(count);
	m_fdata->Seek(offset * sizeof(float));
	m_fdata->Read(p, count);
	auto created = torch::from_blob(p, tensor.sizes(), torch::Deleter(free), torch::kFloat32);
	tensor = created.to(tensor.device());
}

int ModelImporter::ReportUnimported(const std::string &prefix) const {
	char buf[1024];
	int count = 0;
	for (auto iter : m_sections) {
		const auto &key = iter.first;
		if (m_imported.find(key) == m_imported.end()) {
			if (prefix.empty() || key.find(prefix) == 0) {
				count++;
				snprintf(buf, sizeof(buf), "Not imported: %s\n", key.c_str());
				std::cerr << buf;
			}
		}
	}
	if (count > 0) {
		snprintf(buf, sizeof(buf), "%d not imported from %s\n", count, m_fullpath.c_str());
		std::cerr << buf;
	}
	return count;
}
