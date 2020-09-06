#include "Base.h"
#include "yacs.h"

#include <Detectron2/Utils/Utils.h>
#include <Detectron2/Utils/File.h>

using namespace std;
using namespace Detectron2;
using namespace yacs;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static const char *IMMUTABLE = "__immutable__";
static const char *DEPRECATED_KEYS = "__deprecated_keys__";
static const char *RENAMED_KEYS = "__renamed_keys__";
static const char *NEW_ALLOWED = "__new_allowed__";

YAML::Node yacs::CfgNode::load_cfg_from_yaml_file(const std::string &filename) {
	auto content = File(filename).Read();
	return YAML::Load(content.c_str());
}

YAML::Node yacs::CfgNode::load_cfg_from_yaml_str(const std::string &str_obj) {
	return YAML::Load(str_obj);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

yacs::CfgNode::CfgNode(YAML::Node init_dict, bool new_allowed) :
	m_dict(init_dict)
{
	// Recursively convert nested dictionaries in init_dict into CfgNodes
	// Manage if the CfgNode is frozen or not
	m_dict[IMMUTABLE] = false;
	// Deprecated options
	// If an option is removed from the code and you don't want to break existing
	// yaml configs, you can add the full config key as a string to the set below.
	m_dict[DEPRECATED_KEYS] = {};
	// Renamed options
	// If you rename a config option, record the mapping from the old name to the new
	// name in the dictionary below. Optionally, if the type also changed, you can
	// make the value a tuple that specifies first the renamed key and then
	// instructions for how to edit the config file.
	m_dict[RENAMED_KEYS] = {
		// 'EXAMPLE.OLD.KEY': 'EXAMPLE.NEW.KEY',  // Dummy example to follow
		// 'EXAMPLE.OLD.KEY': (                   // A more complex example to follow
		//     'EXAMPLE.NEW.KEY',
		//     "Also convert to a tuple, e.g., 'foo' -> ('foo',) or "
		//     + "'foo:bar' -> ('foo', 'bar')"
		// ),
	};

	// Allow new attributes after initialisation
	m_dict[NEW_ALLOWED] = new_allowed;
}

YAML::Node yacs::CfgNode::get(const std::string &name) const {
	if (name.find('.') == string::npos) {
		return m_dict[name];
	}
	auto key_list = tokenize(name, '.');
	auto d = m_dict;
	int last = key_list.size() - 1;
	for (int i = 0; i < last; i++) {
		auto &subkey = key_list[i];
		d.reset(d[subkey]);
	}
	auto &subkey = key_list[last];
	return d[subkey];
}

void yacs::CfgNode::set(const std::string &name, YAML::Node val) {
	assert(!is_frozen());
	get(name) = val;
}

template<>
void  yacs::CfgNode::NodeSetter::operator=(const YAML::Node &val) {
	assert(false); // because it is ambiguous
}

std::string yacs::CfgNode::str() const {
	return dump();
}

std::string yacs::CfgNode::repr() const {
	return "CfgNode(" + str() + ")";
}

std::string yacs::CfgNode::dump() const {
	return YAML::Dump(m_dict);
}

void yacs::CfgNode::merge_from_file(const std::string &cfg_filename) {
	auto node = load_cfg_from_yaml_file(cfg_filename);
	_merge_a_into_b(node, m_dict, "");
}

void yacs::CfgNode::merge_from_other_cfg(const CfgNode &cfg_other) {
	_merge_a_into_b(cfg_other.clone(), m_dict, "");
}

void yacs::CfgNode::_merge_a_into_b(YAML::Node a, YAML::Node b, const std::string &full_key) {
	for (auto iter : a) {
		auto k = iter.first.as<string>();
		auto v = iter.second;
		if (b[k]) {
			if (b[k].IsMap()) {
				_merge_a_into_b(v, b[k], (full_key.empty() ? k : (full_key + "." + k)));
			} else {
				b[k] = v;
			}
		} else if (b[NEW_ALLOWED].as<bool>()) {
			b[k] = v;
		} else {
			if (key_is_deprecated(full_key)) continue;
			if (key_is_renamed(full_key)) raise_key_rename_error(full_key);
			assert(false); // Non-existent config key: {full_key}
		}
	}
}

void yacs::CfgNode::merge_from_list(const OptionList &cfg_list) {
	for (int i = 0; i < cfg_list.size(); i++) {
		std::string full_key; YAML::Node v;
		tie(full_key, v) = cfg_list[i];
		if (key_is_deprecated(full_key)) {
			continue;
		}
		if (key_is_renamed(full_key)) {
			raise_key_rename_error(full_key);
		}
		set(full_key, v);
	}
}

bool yacs::CfgNode::is_frozen() const {
	return m_dict[IMMUTABLE].as<bool>(false);
}

void yacs::CfgNode::_recurse(YAML::Node node, std::function<void(YAML::Node node)> func) {
	func(node);
	for (auto iter : node) {
		YAML::Node node = iter.second;
		if (node.IsMap()) {
			_recurse(node, func);
		}
	}
}

void yacs::CfgNode::_immutable(bool is_immutable) {
	_recurse(m_dict, [=](YAML::Node node) { node[IMMUTABLE] = is_immutable; });
}

YAML::Node yacs::CfgNode::clone() const {
	return YAML::Load(YAML::Dump(m_dict));
}

void yacs::CfgNode::register_deprecated_key(const std::string &key) {
	assert(!m_dict[DEPRECATED_KEYS][key]);
	m_dict[DEPRECATED_KEYS][key] = true;
}

void yacs::CfgNode::register_renamed_key(const std::string &old_name, const std::string &new_name,
	const std::string &message) {
	assert(!m_dict[RENAMED_KEYS][old_name]);
	if (message.empty()) {
		m_dict[RENAMED_KEYS][old_name] = new_name;
	}
	else {
		m_dict[RENAMED_KEYS][old_name] = std::vector<string>{ new_name, message };
	}
}

bool yacs::CfgNode::key_is_deprecated(const std::string &full_key) const {
	if (m_dict[DEPRECATED_KEYS][full_key]) {
		return true;
	}
	return false;
}

bool yacs::CfgNode::key_is_renamed(const std::string &full_key) const {
	if (m_dict[RENAMED_KEYS][full_key]) {
		return true;
	}
	return false;
}

void yacs::CfgNode::raise_key_rename_error(const std::string &full_key) {
	auto new_key = m_dict[RENAMED_KEYS][full_key];
	string new_key_name, msg;
	if (new_key.IsSequence()) {
		auto new_key2 = new_key.as<std::vector<string>>();
		msg = " Note: " + new_key2[1];
		new_key_name = new_key2[0];
	}
	assert(false); // "Key {} was renamed to {}; please update your config.{}".format(full_key, new_key, msg)
}

bool yacs::CfgNode::is_new_allowed() const {
	return m_dict[NEW_ALLOWED].as<bool>();
}

void yacs::CfgNode::set_new_allowed(bool is_new_allowed) {
	_recurse(m_dict, [=](YAML::Node node) { node[NEW_ALLOWED] = is_new_allowed; });
}
