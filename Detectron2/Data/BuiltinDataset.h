#pragma once

#include "MetadataCatalog.h"

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from data/datasets/builtin.py
	// converted from data/datasets/register_coco.py

	/**
		This file registers pre-defined datasets at hard-coded paths, and their metadata.

		We hard-code metadata for common datasets. This will enable:
		1. Consistency check when loading the datasets
		2. Use models on these standard datasets directly and run demos,
		   without having to download the dataset annotations

		We hard-code some paths to the dataset that's assumed to
		exist in "./datasets/".

		Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
		To add new dataset, refer to the tutorial "docs/DATASETS.md".
	*/

	class BuiltinDataset {
	public:
		static void register_all();

	private:
		static void register_all_coco(const std::string &root);
		static void register_all_lvis(const std::string &root);
		static void register_all_cityscapes(const std::string &root);
		static void register_all_pascal_voc(const std::string &root);

		
		// functions to register a COCO-format dataset to the DatasetCatalog.

		/**
			Register a dataset in COCO's json annotation format for
			instance detection, instance segmentation and keypoint detection.
			(i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
			`instances*.json` and `person_keypoints*.json` in the dataset).

			This is an example of how to register a new dataset.
			You can do something similar to this function, to register new datasets.

			Args:
				name (str): the name that identifies a dataset, e.g. "coco_2014_train".
				metadata (dict): extra metadata associated with this dataset.  You can
					leave it as an empty dict.
				json_file (str): path to the json instance annotation file.
				image_root (str or path-like): directory which contains all the images.
		*/
		static void register_coco_instances(const std::string &name, Metadata metadata,
			const std::string &json_file, const std::string &image_root);

		/**
			Register a COCO panoptic segmentation dataset named `name`.
			The annotations in this registered dataset will contain both instance annotations and
			semantic annotations, each with its own contiguous ids. Hence it's called "separated".

			It follows the setting used by the PanopticFPN paper:

			1. The instance annotations directly come from polygons in the COCO
			   instances annotation task, rather than from the masks in the COCO panoptic annotations.

			   The two format have small differences:
			   Polygons in the instance annotations may have overlaps.
			   The mask annotations are produced by labeling the overlapped polygons
			   with depth ordering.

			2. The semantic annotations are converted from panoptic annotations, where
			   all "things" are assigned a semantic id of 0.
			   All semantic categories will therefore have ids in contiguous
			   range [1, #stuff_categories].

			This function will also register a pure semantic segmentation dataset
			named ``name + '_stuffonly'``.

			Args:
				name (str): the name that identifies a dataset,
					e.g. "coco_2017_train_panoptic"
				metadata (dict): extra metadata associated with this dataset.
				image_root (str): directory which contains all the images
				panoptic_root (str): directory which contains panoptic annotation images
				panoptic_json (str): path to the json panoptic annotation file
				sem_seg_root (str): directory which contains all the ground truth segmentation annotations.
				instances_json (str): path to the json instance annotation file
    	*/
		static void register_coco_panoptic_separated(const std::string &name, Metadata metadata,
			const std::string &image_root, const std::string &panoptic_root, const std::string &panoptic_json,
			const std::string &sem_seg_root, const std::string &instances_json);


		/**
			Create dataset dicts for panoptic segmentation, by
			merging two dicts using "file_name" field to match their entries.

			Args:
				detection_dicts (list[dict]): lists of dicts for object detection or instance segmentation.
				sem_seg_dicts (list[dict]): lists of dicts for semantic segmentation.

			Returns:
				list[dict] (one per input image): Each dict contains all (key, value) pairs from dicts in
					both detection_dicts and sem_seg_dicts that correspond to the same image.
					The function assumes that the same key in different dicts has the same value.
		*/
		//~! static void merge_to_panoptic(detection_dicts, sem_seg_dicts);
	};
}
