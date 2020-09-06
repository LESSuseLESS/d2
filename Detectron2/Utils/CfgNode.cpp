#include "Base.h"
#include "CfgNode.h"

#include "File.h"

using namespace std;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int s_latest_ver = 0;
CfgNode CfgNode::get_cfg() {
	static CfgNode _C;
	if (s_latest_ver == 0) {
		auto defaultConfigDir = getenv("D2_CONFIGS_DEFAULT_DIR");
		assert(defaultConfigDir);
		// This yaml was created by dumping _C into yaml from config/defaults.py.
		_C = load_cfg_from_yaml_file(File::ComposeFilename(defaultConfigDir, "CfgDefaults.yaml"));
		s_latest_ver = _C["VERSION"].as<int>();
	}
	return _C.clone();
}

static CfgNode s_global_cfg;
void CfgNode::set_global_cfg(const CfgNode &cfg) {
	s_global_cfg = cfg;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CfgNode::CfgNode(YAML::Node init_dict) : fvcore::CfgNode(init_dict) {
}

void CfgNode::merge_from_file(const std::string &cfg_filename, bool allow_unsafe) {
	CfgNode loaded_cfg = fvcore::CfgNode::load_yaml_with_base(cfg_filename, allow_unsafe);

	auto ver = m_dict["VERSION"].as<int>();

	// CfgNode.merge_from_file is only allowed on a config object of latest version!
	assert(s_latest_ver == ver);

	auto loaded_ver = loaded_cfg["VERSION"].as<int>();
	/*~!
	if loaded_ver is None:
	    from .compat import guess_version
	    loaded_ver = guess_version(loaded_cfg, cfg_filename)
	*/
	assert(loaded_ver <= ver); // Cannot merge a v{loaded_ver} config into a v{self.VERSION} config.

	if (loaded_ver == ver) {
		merge_from_other_cfg(loaded_cfg);
	}
	else {
		assert(false);
		//~! didn't convert config upgrade/downgrade
	}
}
