#include "Base.h"
#include "NaiveSyncBatchNorm.h"

using namespace std;
using namespace torch;
using namespace torch::autograd;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// not converted

// torch::sum() was directly used instead:
/*
class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

NaiveSyncBatchNormImpl::NaiveSyncBatchNormImpl(const torch::nn::BatchNorm2dOptions &options,
	const std::string &stats_mode) : BatchNorm2dImpl(options), m_stats_mode(stats_mode) {
	assert(stats_mode == "" || stats_mode == "N");
}

torch::Tensor NaiveSyncBatchNormImpl::forward(torch::Tensor input) {
	if (get_world_size() == 1 or !is_training()) {
		return BatchNorm2dImpl::forward(input);
	}

	auto B = input.size(0);
	auto C = input.size(1);

	auto mean = torch::mean(input, { 0, 2, 3 });
	auto meansqr = torch::mean(input * input, { 0, 2, 3 });

	double momentum = options.momentum().value_or(0.1);
	if (m_stats_mode.empty()) {
		assert(B > 0); // 'SyncBatchNorm(stats_mode="") does not support zero batch size.'
		auto vec = torch::cat({ mean, meansqr }, 0);
		vec = torch::sum(vec, 0) * (1.0 / get_world_size());
		auto results = torch::split(vec, C);
		mean = results[0];
		meansqr = results[1];
	}
	else {
		Tensor vec;
		auto toptions = dtype(mean.dtype()).device(mean.device());
		if (B == 0) {
			vec = torch::zeros({ 2 * C + 1 }, toptions);
			vec = vec + input.sum(); // make sure there is gradient w.r.t input
		}
		else {
			vec = torch::cat({ mean, meansqr, torch::ones({1}, toptions) });
		}
		vec = torch::sum(vec * B, 0);

		auto total_batch = vec[-1].detach();
		momentum = total_batch.clamp(nullopt, 1).item<double>() * momentum; // no update if total_batch is 0
		total_batch = torch::max(total_batch, torch::ones_like(total_batch)); // avoid div - by - zero
		auto results = torch::split(vec / total_batch, C);
		mean = results[0];
		meansqr = results[1];
	}

	auto var = meansqr - mean * mean;
	auto invstd = torch::rsqrt(var + options.eps());
	auto scale = get_weight() * invstd;
	auto bias = get_bias() - mean * scale;
	scale = scale.reshape({ 1, -1, 1, 1 });
	bias = bias.reshape({ 1, -1, 1, 1 });

	assert(get_running_mean());
	auto running_mean = *get_running_mean();
	auto running_var = *get_running_var();
	running_mean += momentum * (mean.detach() - running_mean);
	running_var += momentum * (var.detach() - running_var);
	return input * scale + bias;
}
