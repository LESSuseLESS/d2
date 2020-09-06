#pragma once

#include "yacs.h"

namespace Detectron2 { namespace fvcore
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from fvcore/common/config.py
	
    /**
		Our own extended version of :class:`yacs.config.CfgNode`.
		It contains the following extra features:

		1. The :meth:`merge_from_file` method supports the "_BASE_" key,
		   which allows the new CfgNode to inherit all the attributes from the
		   base configuration file.
		2. Keys that start with "COMPUTED_" are treated as insertion-only
		   "computed" attributes. They can be inserted regardless of whether
		   the CfgNode is frozen or not.
		3. With "allow_unsafe=True", it supports pyyaml tags that evaluate
		   expressions in config. See examples in
		   https://pyyaml.org/wiki/PyYAMLDocumentation#yaml-tags-and-python-types
		   Note that this may lead to arbitrary code execution: you must not
		   load a config file from untrusted sources before manually inspecting
		   the content of the file.
	*/
	class CfgNode : public yacs::CfgNode {
	public:
		/**
			Just like `yaml.load(open(filename))`, but inherit attributes from its
				`_BASE_`.

			Args:
				filename (str): the file name of the current config. Will be used to
					find the base config file.
				allow_unsafe (bool): whether to allow loading the config file with
					`yaml.unsafe_load`.

			Returns:
				(dict): the loaded yaml
		*/
		static YAML::Node load_yaml_with_base(const std::string &filename, bool allow_unsafe = false);

	public:
		CfgNode(YAML::Node init_dict = {});

		virtual void set(const std::string &name, YAML::Node val) override;

		/**
			Merge configs from a given yaml file.

			Args:
				cfg_filename: the file name of the yaml config.
				allow_unsafe: whether to allow loading the config file with
					`yaml.unsafe_load`.
		*/
		void merge_from_file(const std::string &cfg_filename, bool allow_unsafe = false);

		// Forward the following calls to base, but with a check on the BASE_KEY.
		/**
			Args:
				cfg_other (CfgNode): configs to merge from.
		*/
		void merge_from_other_cfg(const CfgNode &cfg_other);

		/**
			Args:
				cfg_list (list): list of configs to merge from.
		*/
		void merge_from_list(const OptionList &cfg_list);
	};
}}
