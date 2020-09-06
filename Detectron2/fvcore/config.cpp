#include "Base.h"
#include "config.h"

#include <Detectron2/Utils/File.h>

using namespace std;
using namespace Detectron2;
using namespace fvcore;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static const char *BASE_KEY = "_BASE_";

static void merge_a_into_b(YAML::Node a, YAML::Node b) {
	// merge dict a into dict b. values in a will overwrite b.
	for (auto iter : a) {
		auto k = iter.first.as<string>();
		auto v = iter.second;
		if (v.IsMap() && b[k]) {
			assert(b[k].IsMap()); // Cannot inherit key '{k}' from base!
			merge_a_into_b(v, b[k]);
		} else {
			b[k] = v;
		}
	}
}

YAML::Node fvcore::CfgNode::load_yaml_with_base(const std::string &filename, bool allow_unsafe) {
	auto cfg = YAML::LoadFile(filename);
	if (cfg[BASE_KEY]) {
		auto base_cfg_file = cfg[BASE_KEY].as<string>();
		if (!File::IsAbsolutePath(base_cfg_file)) {
			base_cfg_file = File::ComposeFilename(File::Dirname(filename), base_cfg_file);
		}
		auto base_cfg = load_yaml_with_base(base_cfg_file, allow_unsafe);
		cfg.remove(BASE_KEY);

		// pyre-fixme[6]: Expected `Dict[typing.Any, typing.Any]` for 2nd param
		//  but got `None`.
		merge_a_into_b(cfg, base_cfg);
		return base_cfg;
	}
	return cfg;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

fvcore::CfgNode::CfgNode(YAML::Node init_dict) : yacs::CfgNode(init_dict) {
}

void fvcore::CfgNode::merge_from_file(const std::string &cfg_filename, bool allow_unsafe) {
	merge_from_other_cfg(load_yaml_with_base(cfg_filename, allow_unsafe));
}

void fvcore::CfgNode::merge_from_other_cfg(const CfgNode &cfg_other) {
	assert(!cfg_other[BASE_KEY].node()); // The reserved key '{BASE_KEY}' can only be used in files!
	yacs::CfgNode::merge_from_other_cfg(cfg_other);
}

void fvcore::CfgNode::merge_from_list(const OptionList &cfg_list) {
	for (int i = 0; i < cfg_list.size(); i++) {
		assert(std::get<0>(cfg_list[i]) != BASE_KEY); // The reserved key '{BASE_KEY}' can only be used in files!
	}
	yacs::CfgNode::merge_from_list(cfg_list);
}

void fvcore::CfgNode::set(const std::string &name, YAML::Node val) {
	if (name.find("COMPUTED_") == 0) {
		if (m_dict[name]) {
			auto old_val = m_dict[name];
			if (old_val == val) {
				return;
			}
			assert(false); // Computed attributed '{name}' already exists  with a different value!
		}
		m_dict[name] = val;
	}
	else {
		yacs::CfgNode::set(name, val);
	}
}
