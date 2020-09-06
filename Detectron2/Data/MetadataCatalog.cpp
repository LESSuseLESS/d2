#include "Base.h"
#include "MetadataCatalog.h"

using namespace std;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static unordered_map<string, Metadata> _NAME_TO_META = {};

Metadata MetadataCatalog::get(const std::string &name) {
	assert(!name.empty());
	auto iter = _NAME_TO_META.find(name);
	if (iter != _NAME_TO_META.end()) {
		return iter->second;
	}
	else {
		auto m = make_shared<MetadataImpl>();
		m->name = name;
		_NAME_TO_META[name] = m;
		return m;
	}
}

std::vector<std::string> MetadataCatalog::list() {
	std::vector<std::string> ret;
	ret.reserve(_NAME_TO_META.size());
	for (auto iter : _NAME_TO_META) {
		ret.push_back(iter.first);
	}
	return ret;
}
