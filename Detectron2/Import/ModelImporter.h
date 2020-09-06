#pragma once

#include <Detectron2/Base.h>
#include <Detectron2/Utils/File.h>

namespace Detectron2
{
	class ModelImporter {
	public:
		enum Model {
			kNone,
			kDemo,						// COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml

			// Root: https://github.com/facebookresearch/detectron2/tree/master/configs
			kCOCODetection,				// COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
			kCOCOKeypoints,				// COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml
			kCOCOInstanceSegmentation,	// COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
			kCOCOPanopticSegmentation	// COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml
		};
		static Model FilenameToModel(const std::string &filename);

		enum Fill {
			kNoFill,
			kZeroFill,
			kConstantFill,
			kNormalFill2,
			kNormalFill3,
			kXavierNormalFill,
			kCaffe2XavierFill,
			kCaffe2MSRAFill,
			kCaffe2MSRAFillIn
		};

		static void FillTensor(torch::Tensor x, Fill fill);

		static std::string DataDir();

	public:
		ModelImporter(Model model);
		ModelImporter(const std::string &filename);

		bool HasData() const { return m_fdata.get() != nullptr; }

		void Import(const std::string &name, torch::nn::Conv2d &conv, Fill fill) const;
		void Import(const std::string &name, torch::nn::ConvTranspose2d &conv, Fill fill) const;
		void Import(const std::string &name, torch::nn::Linear &fc, Fill fill) const;

		void Initialize(const std::string &name, torch::Tensor &tensor) const;

		int ReportUnimported(const std::string &prefix = "") const;

	private:
		// implemented in generated files by ImportBaseline.py
		std::string import_model_final_f10217();
		std::string import_model_final_f6e8b1();
		std::string import_model_final_a3ec72();
		std::string import_model_final_997cc7();
		std::string import_model_final_cafdb1();

		std::unordered_map<std::string, std::pair<int, int>> m_sections;
		int m_size;
		void Add(const char *name, int count);

		std::string m_fullpath;
		std::shared_ptr<File> m_fdata;

		mutable std::unordered_set<std::string> m_imported;
	};
}