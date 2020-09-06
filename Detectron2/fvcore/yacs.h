#pragma once

#include <Detectron2/Base.h>
#include <yaml-cpp/yaml.h>

namespace Detectron2 { namespace yacs
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from https://github.com/rbgirshick/yacs

	/**
		YACS -- Yet Another Configuration System is designed to be a simple
		configuration management system for academic and industrial research
		projects.

		See README.md for usage and examples.
	*/

    /**
		CfgNode represents an internal node in the configuration tree. It's a simple
		dict-like container that allows for attribute-based access to keys.
	*/
	class CfgNode {
	public:
		using Option = std::tuple<std::string, YAML::Node>;
		using OptionList = std::vector<Option>;

	public:
		// Load a config from a YAML file or a Python source file.
		static YAML::Node load_cfg_from_yaml_file(const std::string &filename);

		// Load a config from a YAML string encoding.
		static YAML::Node load_cfg_from_yaml_str(const std::string &str_obj);

		// handling "(elem, )"
		template<typename T>
		static std::vector<T> parseTuple(YAML::Node node, const std::vector<T> &def = {}) {
			if (node) {
				std::string s = node.as<std::string>();
				assert(s[0] == '(');
				assert(s[s.size() - 1] == ')');
				assert(s[s.size() - 2] == ',');
				s = std::string("x: [") + s.substr(1, s.size() - 3) + "]";
				return YAML::Load(s)["x"].as<std::vector<T>>(def);
			}
			return def;
		}

	public:
		/**
			init_dict (dict): the possibly-nested dictionary to initailize the CfgNode.
			key_list (list[str]): a list of names which index this CfgNode from the root.
				Currently only used for logging purposes.
			new_allowed (bool): whether adding new key is allowed when merging with
				other configs.
		*/
		CfgNode(YAML::Node init_dict = {}, bool new_allowed = false);
		virtual ~CfgNode() {}

		// The only reason NodeSetter exists is to give subclasses a chance to add special handling on sets.
		// But, YAML::Node can be modified directly, without going through the checking. So the promise is,
		// you have to only do two forms of coding, without directly setting a YAML::Node:
		//     auto value = cfg["some_key"].as<some_type>();	// going through get()
		//     cfg["some_key"] = value;							// going through set()
		// Passing around YAML::Node is fine, but just don't write to it directly. Always write through cfg object.
		virtual YAML::Node get(const std::string &name) const;
		virtual void set(const std::string &name, YAML::Node val);
		struct NodeGetter {
			const CfgNode *cfg;
			std::string name;

			YAML::Node node() const {
				return cfg->get(name);
			}
			operator YAML::Node() const {
				return cfg->get(name);
			}
			template<typename T>
			T as() const {
				return cfg->get(name).as<T>();
			}
			template <typename T, typename S>
			T as(const S& fallback) const {
				return cfg->get(name).as<T>(fallback);
			}
		};
		struct NodeSetter {
			CfgNode *cfg;
			std::string name;

			template<typename T>
			void operator=(const T &val) {
				YAML::Node nval;
				nval = val;
				cfg->set(name, nval);
			}
			template<>
			void operator=(const YAML::Node &val);

			YAML::Node node() const {
				return cfg->get(name);
			}
			operator YAML::Node() const {
				return cfg->get(name);
			}
			template<typename T>
			T as() const {
				return cfg->get(name).as<T>();
			}
			template <typename T, typename S>
			T as(const S& fallback) const {
				return cfg->get(name).as<T>(fallback);
			}
		};
		NodeGetter operator[](const std::string &name) const {
			return { this, name };
		}
		NodeSetter operator[](const std::string &name) {
			return { this, name };
		}

		std::string str() const;  // informal and readable
		std::string repr() const; // official
		std::string dump() const; // yaml output

		// Load a yaml config file and merge it this CfgNode.
		void merge_from_file(const std::string &cfg_filename);

		void merge_from_other_cfg(const CfgNode &cfg_other);

		/**
			Merge config (keys, values) in a list (e.g., from command line) into
			this CfgNode. For example, `cfg_list = ['FOO.BAR', 0.5]`.
		*/
		void merge_from_list(const OptionList &cfg_list);

		// Make this CfgNode and all of its children immutable.
		void freeze() { _immutable(true); }
		// Make this CfgNode and all of its children mutable.
		void defrost() { _immutable(false); }
		// Return mutability.
		bool is_frozen() const;

		// Recursively copy this CfgNode.
		YAML::Node clone() const;
		YAML::Node node() const { return m_dict; }

		/**
			Register key (e.g. `FOO.BAR`) a deprecated option. When merging deprecated
			keys a warning is generated and the key is ignored.
		*/
		void register_deprecated_key(const std::string &key);

		/**
			Register a key as having been renamed from `old_name` to `new_name`.
			When merging a renamed key, an exception is thrown alerting to user to
			the fact that the key has been renamed.
		*/
		void register_renamed_key(const std::string &old_name, const std::string &new_name,
			const std::string &message = "");

		// Test if a key is deprecated.
		bool key_is_deprecated(const std::string &full_key) const;
		// Test if a key is renamed.
		bool key_is_renamed(const std::string &full_key) const;

		void raise_key_rename_error(const std::string &full_key);

		bool is_new_allowed() const;

		/**
			Set this config (and recursively its subconfigs) to allow merging
			new keys from other configs.
		*/
		void set_new_allowed(bool is_new_allowed);

	protected:
		YAML::Node m_dict;

		void _recurse(YAML::Node node, std::function<void(YAML::Node node)> func);

		// Set immutability to is_immutable and recursively apply the setting to all nested CfgNodes.
		void _immutable(bool is_immutable);

		void _merge_a_into_b(YAML::Node a, YAML::Node b, const std::string &full_key);
	};
}}
