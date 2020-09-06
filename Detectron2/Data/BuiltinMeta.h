#pragma once

#include "MetadataCatalog.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from data/datasets/builtin_meta.py

	class BuiltinMeta {
	public:
		static Metadata _get_builtin_metadata(const std::string &dataset_name);

	private:
		static void _get_coco_instances_meta(Metadata &metadata);

		// Returns metadata for "separated" version of the panoptic segmentation dataset.
		static void _get_coco_panoptic_separated_meta(Metadata &metadata);

		static void _get_coco_person_meta(Metadata &metadata);
		static void _get_cityscapes_meta(Metadata &metadata);
	};
}
