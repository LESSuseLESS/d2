#include "Base.h"
#include "BatchNorm.h"

#include "BatchNorm2d.h"
#include "FrozenBatchNorm2d.h"
#include "GroupNorm.h"
#include "NaiveSyncBatchNorm.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BatchNorm::Type BatchNorm::GetType(const std::string &name) {
	static map<string, Type> lookup_table{
		{ "",				kNone },
		{ "BN",				kBN },
		{ "SyncBN",			kSyncBN },
		{ "FrozenBN",		kFrozenBN },
		{ "GN",				kGN },
		{ "nnSyncBN",		nnSyncBN },
		{ "naiveSyncBN",	naiveSyncBN }
	};
	auto iter = lookup_table.find(name);
	assert(iter != lookup_table.end());
	return iter->second;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BatchNorm::BatchNorm(BatchNorm::Type type, int out_channels) {
	switch (type) {
	case kNone:																				break;
	case kBN:			reset(new BatchNorm2dImpl(out_channels));							break;
	case kFrozenBN:		reset(new FrozenBatchNorm2dImpl(out_channels));						break;
	case naiveSyncBN:	reset(new NaiveSyncBatchNormImpl(out_channels));					break;
	case kGN:			reset(new GroupNormImpl(nn::GroupNormOptions(32, out_channels)));	break;

	case kSyncBN:	//"SyncBN": NaiveSyncBatchNorm if TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
	case nnSyncBN:	// return nn.SyncBatchNorm(out_channels);
	default:
		assert(false);
	}
}

void BatchNormImpl::initialize(const ModelImporter &importer, const std::string &prefix, ModelImporter::Fill fill) {
	if (importer.HasData()) {
		importer.Initialize(prefix + ".weight", get_weight());
		importer.Initialize(prefix + ".bias", get_bias());
		if (get_running_mean()) {
			importer.Initialize(prefix + ".running_mean", *get_running_mean());
			importer.Initialize(prefix + ".running_var", *get_running_var());
		}
	}
	else {
		ModelImporter::FillTensor(get_weight(), fill);
		ModelImporter::FillTensor(get_bias(), ModelImporter::kZeroFill);
	}
}
