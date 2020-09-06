#pragma once

#include <Detectron2/fvcore/config.h>

namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from config/config.py
	
    /**
		The same as `fvcore.common.config.CfgNode`, but different in:

		1. Use unsafe yaml loading by default.
		   Note that this may lead to arbitrary code execution: you must not
		   load a config file from untrusted sources before manually inspecting
		   the content of the file.
		2. Support config versioning.
		   When attempting to merge an old config, it will convert the old config automatically.
	*/
	class CfgNode : public fvcore::CfgNode {
	public:
		/**
			Get a copy of the default config.

			Returns:
				a detectron2 CfgNode instance.
		*/
		static CfgNode get_cfg();

		/**
			Let the global config point to the given cfg.

			Assume that the given "cfg" has the key "KEY", after calling
			`set_global_cfg(cfg)`, the key can be accessed by:

			.. code-block:: python

				from detectron2.config import global_cfg
				print(global_cfg.KEY)

			By using a hacky global config, you can access these configs anywhere,
			without having to pass the config object or the values deep into the code.
			This is a hacky feature introduced for quick prototyping / research exploration.
		*/
		static void set_global_cfg(const CfgNode &cfg);

	public:
		CfgNode(YAML::Node init_dict = {});

		// Note that the default value of allow_unsafe is changed to True
		void merge_from_file(const std::string &cfg_filename, bool allow_unsafe = true);
	};
}
