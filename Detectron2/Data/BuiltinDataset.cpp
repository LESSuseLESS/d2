#include "Base.h"
#include "BuiltinDataset.h"
#include "BuiltinMeta.h"

using namespace std;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void BuiltinDataset::register_all_coco(const std::string &root) {
	static map<string, map<string, tuple<string, string>>> _PREDEFINED_SPLITS_COCO = {
		{
			"coco", {
			{"coco_2014_train",				{ "coco/train2014", "coco/annotations/instances_train2014.json"			}},
			{"coco_2014_val",				{ "coco/val2014", "coco/annotations/instances_val2014.json"				}},
			{"coco_2014_minival",			{ "coco/val2014", "coco/annotations/instances_minival2014.json"			}},
			{"coco_2014_minival_100",		{ "coco/val2014", "coco/annotations/instances_minival2014_100.json"		}},
			{"coco_2014_valminusminival",	{ "coco/val2014", "coco/annotations/instances_valminusminival2014.json"	}},
			{"coco_2017_train",				{ "coco/train2017", "coco/annotations/instances_train2017.json"			}},
			{"coco_2017_val",				{ "coco/val2017", "coco/annotations/instances_val2017.json"				}},
			{"coco_2017_test",				{ "coco/test2017", "coco/annotations/image_info_test2017.json"			}},
			{"coco_2017_test-dev",			{ "coco/test2017", "coco/annotations/image_info_test-dev2017.json"		}},
			{"coco_2017_val_100",			{ "coco/val2017", "coco/annotations/instances_val2017_100.json"			}},
		}},
		{
			"coco_person", {
			{"keypoints_coco_2014_train",	{ "coco/train2014", "coco/annotations/person_keypoints_train2014.json"	}},
			{"keypoints_coco_2014_val",		{ "coco/val2014", "coco/annotations/person_keypoints_val2014.json"		}},
			{"keypoints_coco_2014_minival", { "coco/val2014", "coco/annotations/person_keypoints_minival2014.json"	}},
			{"keypoints_coco_2014_valminusminival", {"coco/val2014",
				"coco/annotations/person_keypoints_valminusminival2014.json" }},
			{"keypoints_coco_2014_minival_100", { "coco/val2014",
				"coco/annotations/person_keypoints_minival2014_100.json" }},
			{"keypoints_coco_2017_train",	{ "coco/train2017", "coco/annotations/person_keypoints_train2017.json"	}},
			{"keypoints_coco_2017_val",		{ "coco/val2017", "coco/annotations/person_keypoints_val2017.json"		}},
			{"keypoints_coco_2017_val_100", { "coco/val2017", "coco/annotations/person_keypoints_val2017_100.json"	}},
		}},
	};
	for (auto iter : _PREDEFINED_SPLITS_COCO) {
		auto &dataset_name = iter.first;
		auto &splits_per_dataset = iter.second;
		for (auto it : splits_per_dataset) {
			auto &key = it.first;
			string image_root, json_file;
			tie(image_root, json_file) = it.second;

			// Assume pre-defined datasets live in `./datasets`.
			register_coco_instances(key, BuiltinMeta::_get_builtin_metadata(dataset_name),
				json_file.find("://") == string::npos ? File::ComposeFilename(root, json_file) : json_file,
				File::ComposeFilename(root, image_root));
		}
	}

	static map<string, tuple<string, string, string>> _PREDEFINED_SPLITS_COCO_PANOPTIC = {
		{"coco_2017_train_panoptic", {
		// This is the original panoptic annotation directory
		"coco/panoptic_train2017",
		"coco/annotations/panoptic_train2017.json",
		// This directory contains semantic annotations that are
		// converted from panoptic annotations.
		// It is used by PanopticFPN.
		// You can use the script at detectron2 / datasets / prepare_panoptic_fpn.py
		// to create these directories.
		"coco/panoptic_stuff_train2017",
	}},
	{"coco_2017_val_panoptic", {
		"coco/panoptic_val2017",
		"coco/annotations/panoptic_val2017.json",
		"coco/panoptic_stuff_val2017",
	}},
	{"coco_2017_val_100_panoptic", {
		"coco/panoptic_val2017_100",
		"coco/annotations/panoptic_val2017_100.json",
		"coco/panoptic_stuff_val2017_100",
	}},
	};

	for (auto iter : _PREDEFINED_SPLITS_COCO_PANOPTIC) {
		auto &prefix = iter.first;
		string panoptic_root, panoptic_json, semantic_root;
		tie(panoptic_root, panoptic_json, semantic_root) = iter.second;

		const char *suffix = "_panoptic";
		int pos = prefix.find(suffix);
		assert(pos == prefix.length() - strlen(suffix));
		auto prefix_instances = prefix.substr(0, pos);
		auto instances_meta = MetadataCatalog::get(prefix_instances);
		auto image_root = instances_meta->image_root;
		auto instances_json = instances_meta->json_file;
		register_coco_panoptic_separated(prefix, BuiltinMeta::_get_builtin_metadata("coco_panoptic_separated"),
			image_root,
			File::ComposeFilename(root, panoptic_root),
			File::ComposeFilename(root, panoptic_json),
			File::ComposeFilename(root, semantic_root),
			instances_json);
	}
}

void BuiltinDataset::register_all_lvis(const std::string &root) {
	// ==== Predefined datasets and splits for LVIS ==========
	static map<string, map<string, tuple<string, string>>> _PREDEFINED_SPLITS_LVIS = {
		{"lvis_v0.5", {
			{"lvis_v0.5_train",			{ "coco/train2017", "lvis/lvis_v0.5_train.json"				}},
			{"lvis_v0.5_val",			{ "coco/val2017",	"lvis/lvis_v0.5_val.json"				}},
			{"lvis_v0.5_val_rand_100",	{ "coco/val2017",	"lvis/lvis_v0.5_val_rand_100.json"		}},
			{"lvis_v0.5_test",			{ "coco/test2017", "lvis/lvis_v0.5_image_info_test.json"	}},
		}},
		{"lvis_v0.5_cocofied", {
			{"lvis_v0.5_train_cocofied",{ "coco/train2017", "lvis/lvis_v0.5_train_cocofied.json"	}},
			{"lvis_v0.5_val_cocofied",	{ "coco/val2017", "lvis/lvis_v0.5_val_cocofied.json"		}},
		}},
	};

	for (auto iter : _PREDEFINED_SPLITS_LVIS) {
		auto &dataset_name = iter.first;
		auto &splits_per_dataset = iter.second;
		for (auto it : splits_per_dataset) {
			auto &key = it.first;
			string image_root, json_file;
			tie(image_root, json_file) = it.second;

			// Assume pre-defined datasets live in `./datasets`.
			assert(false);
			/*~!
			register_lvis_instances(key, get_lvis_instances_meta(dataset_name),
				json_file.find("://") == string::npos ? File::ComposeFilename(root, json_file) : json_file,
				File::ComposeFilename(root, image_root));
			*/
		}
	}
}

void BuiltinDataset::register_all_cityscapes(const std::string &root) {
	// ==== Predefined splits for raw cityscapes images ===========
	static map<string, tuple<string, string>> _RAW_CITYSCAPES_SPLITS = {
		{"cityscapes_fine_{task}_train",	{ "cityscapes/leftImg8bit/train", "cityscapes/gtFine/train"	}},
		{"cityscapes_fine_{task}_val",		{ "cityscapes/leftImg8bit/val", "cityscapes/gtFine/val"		}},
		{"cityscapes_fine_{task}_test",		{ "cityscapes/leftImg8bit/test", "cityscapes/gtFine/test"	}},
	};

	for (auto iter : _RAW_CITYSCAPES_SPLITS) {
		auto &key = iter.first;
		string image_dir, gt_dir;
		tie(image_dir, gt_dir) = iter.second;

		auto meta = BuiltinMeta::_get_builtin_metadata("cityscapes");
		image_dir = File::ComposeFilename(root, image_dir);
		gt_dir = File::ComposeFilename(root, gt_dir);

		assert(false);
		/*~!
		inst_key = key.format(task="instance_seg")
		DatasetCatalog.register(
			inst_key,
			lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
				x, y, from_json=True, to_polygons=True
			),
		)
		MetadataCatalog.get(inst_key).set(
			image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
		)

		sem_key = key.format(task="sem_seg")
		DatasetCatalog.register(
			sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
		)
		MetadataCatalog.get(sem_key).set(
			image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_sem_seg", **meta
		)
		*/
	}
}

void BuiltinDataset::register_all_pascal_voc(const std::string &root) {
	// ==== Predefined splits for PASCAL VOC ===========
	static vector<tuple<string, string, string>> SPLITS = {
		{ "voc_2007_trainval",	"VOC2007", "trainval"	},
		{ "voc_2007_train",		"VOC2007", "train"		},
		{ "voc_2007_val",		"VOC2007", "val"		},
		{ "voc_2007_test",		"VOC2007", "test"		},
		{ "voc_2012_trainval",	"VOC2012", "trainval"	},
		{ "voc_2012_train",		"VOC2012", "train"		},
		{ "voc_2012_val",		"VOC2012", "val"		},
	};
	for (auto iter : SPLITS) {
		string name, dirname, split;
		tie(name, dirname, split) = iter;

		auto year = name.find("2007") != string::npos ? 2007 : 2012;
		assert(false);
		/*~!
		register_pascal_voc(name, File::ComposeFilename(root, dirname), split, year);
		MetadataCatalog::get(name)->evaluator_type = "pascal_voc";
		*/
	}
}

void BuiltinDataset::register_all() {
	// Register them all under "./datasets"
	auto env = getenv("DETECTRON2_DATASETS");
	string _root = env ? env : "datasets";
	register_all_coco(_root);
	//register_all_lvis(_root);
	//register_all_cityscapes(_root);
	//register_all_pascal_voc(_root);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void BuiltinDataset::register_coco_instances(const std::string &name, Metadata metadata,
	const std::string &json_file, const std::string &image_root) {
	// 1. register a function which returns dicts
	//~! DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))

	// 2. Optionally, add metadata about this dataset,
	// since they might be useful in evaluation, visualization or logging
	auto clone = MetadataCatalog::get(name);
	*clone = *metadata;
	clone->json_file = json_file;
	clone->image_root = image_root;
	clone->evaluator_type = "coco";
}

void BuiltinDataset::register_coco_panoptic_separated(const std::string &name, Metadata metadata,
	const std::string &image_root, const std::string &panoptic_root, const std::string &panoptic_json,
	const std::string &sem_seg_root, const std::string &instances_json) {
	auto panoptic_name = name + "_separated";
	/*~!
	DatasetCatalog.register(
		panoptic_name,
		lambda: merge_to_panoptic(
			load_coco_json(instances_json, image_root, panoptic_name),
			load_sem_seg(sem_seg_root, image_root),
		),
	)
	*/
	auto panoptic_metadata = MetadataCatalog::get(panoptic_name);
	*panoptic_metadata = *metadata; // clone first
	panoptic_metadata->panoptic_root = panoptic_root;
	panoptic_metadata->image_root = image_root;
	panoptic_metadata->panoptic_json = panoptic_json;
	panoptic_metadata->sem_seg_root = sem_seg_root;
	panoptic_metadata->json_file = instances_json;  // TODO rename
	panoptic_metadata->evaluator_type = "coco_panoptic_seg";

	auto semantic_name = name + "_stuffonly";
	//~! DatasetCatalog.register(semantic_name, lambda: load_sem_seg(sem_seg_root, image_root))
	auto semantic_metadata = MetadataCatalog::get(semantic_name);
	*semantic_metadata = *metadata;
	semantic_metadata->sem_seg_root = sem_seg_root;
	semantic_metadata->image_root = image_root;
	semantic_metadata->evaluator_type = "sem_seg";
}

/*~!
def merge_to_panoptic(detection_dicts, sem_seg_dicts):
    results = []
    sem_seg_file_to_entry = {x["file_name"]: x for x in sem_seg_dicts}
    assert len(sem_seg_file_to_entry) > 0

    for det_dict in detection_dicts:
        dic = copy.copy(det_dict)
        dic.update(sem_seg_file_to_entry[dic["file_name"]])
        results.append(dic)
    return results
*/