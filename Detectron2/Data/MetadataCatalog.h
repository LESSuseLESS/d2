#pragma once

#include <Detectron2/Utils/VisColor.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from data/catalog.py

	/**
		A class that supports simple attribute setter/getter.
		It is intended for storing metadata of a dataset and make it accessible globally.

		Examples:

		.. code-block:: python

			# somewhere when you load the data:
			MetadataCatalog.get("mydataset").thing_classes = ["person", "dog"]

			# somewhere when you print statistics or visualize:
			classes = MetadataCatalog.get("mydataset").thing_classes
	*/
	struct MetadataImpl {
		std::string name;
		std::string image_root;
		std::string json_file;
		std::string evaluator_type;

		std::string panoptic_root;
		std::string panoptic_json;
		std::string sem_seg_root;

		std::vector<ClassColor> thing;
		std::vector<ClassColor> stuff;

		std::vector<std::string> keypoint_names;
		std::vector<std::tuple<std::string, std::string>> keypoint_flip_map;
		std::vector<std::tuple<std::string, std::string, VisColor>> keypoint_connection_rules;

		std::unordered_map<int, int> thing_dataset_id_to_contiguous_id;
		std::unordered_map<int, int> stuff_dataset_id_to_contiguous_id;
	};
	using Metadata = std::shared_ptr<MetadataImpl>;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
		MetadataCatalog provides access to "Metadata" of a given dataset.

		The metadata associated with a certain name is a singleton: once created,
		the metadata will stay alive and will be returned by future calls to `get(name)`.

		It's like global variables, so don't abuse it.
		It's meant for storing knowledge that's constant and shared across the execution
		of the program, e.g.: the class names in COCO.
	*/
	class MetadataCatalog {
	public:
		/**
			Args:
				name (str): name of a dataset (e.g. coco_2014_train).

			Returns:
				Metadata: The :class:`Metadata` instance associated with this name,
				or create an empty one if none is available.
		*/
		static Metadata get(const std::string &name);

		/**
			List all registered metadata.

			Returns:
				list[str]: keys (names of datasets) of all registered metadata
		*/
		static std::vector<std::string> list();
	};
}
