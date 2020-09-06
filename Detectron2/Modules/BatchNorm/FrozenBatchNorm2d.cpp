#include "Base.h"
#include "FrozenBatchNorm2d.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// not converted

/*
def _load_from_state_dict(
	self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
	version = local_metadata.get("version", None)

	if version is None or version < 2:
		# No running_mean/var in early versions
		# This will silent the warnings
		if prefix + "running_mean" not in state_dict:
			state_dict[prefix + "running_mean"] = torch.zeros_like(m_running_mean)
		if prefix + "running_var" not in state_dict:
			state_dict[prefix + "running_var"] = torch.ones_like(m_running_var)

	if version is not None and version < 3:
		logger = logging.getLogger(__name__)
		logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
		# In version < 3, running_var are used without +eps.
		state_dict[prefix + "running_var"] -= m_eps

	super()._load_from_state_dict(
		state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
	)

def __repr__(self):
	return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ModulePtr FrozenBatchNorm2dImpl::convert_frozen_batchnorm(const ModulePtr &mod) {
	if (mod->as<nn::BatchNorm2dImpl>() /* || module.as<SyncBatchNorm>() */) {
		auto bn = mod->as<nn::BatchNorm2dImpl>();
		FrozenBatchNorm2d res(bn->options.num_features());
		if (bn->options.affine()) {
			res->m_weight = bn->weight.clone();
			res->m_bias = bn->bias.clone();
		}
		res->m_running_mean = bn->running_mean;
		res->m_running_var = bn->running_var;
		res->m_eps = bn->options.eps();
		return res.ptr();
	}

	ModulePtr res = mod;
	for (auto iter : mod->named_children()) {
		auto new_child = convert_frozen_batchnorm(iter.value());
		res->replace_module(iter.key(), new_child);
	}
	return res;
}

FrozenBatchNorm2dImpl::FrozenBatchNorm2dImpl(int num_features, double eps) :
	m_num_features(num_features),
	m_eps(eps),
	m_weight(register_buffer("weight", torch::ones(num_features))),
	m_bias(register_buffer("bias", torch::zeros(num_features))),
	m_running_mean(register_buffer("running_mean", torch::zeros(num_features))),
	m_running_var(register_buffer("running_var", torch::ones(num_features) - eps)) {
}

std::string FrozenBatchNorm2dImpl::toString() const {
	return FormatString("FrozenBatchNorm2d(num_features=%d", m_num_features) +
		FormatString(", eps=%f)", m_eps);
}

torch::Tensor FrozenBatchNorm2dImpl::forward(torch::Tensor x) {
	if (x.requires_grad()) {
		// When gradients are needed, F.batch_norm will use extra memory
		// because its backward op computes gradients for weight/bias as well.
		auto scale = m_weight * (m_running_var + m_eps).rsqrt();
		auto bias = m_bias - m_running_mean * scale;
		scale = scale.reshape({ 1, -1, 1, 1 });
		bias = bias.reshape({ 1, -1, 1, 1 });
		return x * scale + bias;
	}
	else {
		// When gradients are not needed, F.batch_norm is a single fused op
		// and provide more optimization opportunities.
		return batch_norm(
			x,
			m_weight,
			m_bias,
			m_running_mean,
			m_running_var,
			false,
			0.1f,
			m_eps,
			Detectron2::cudaEnabled()
		);
	}
}
