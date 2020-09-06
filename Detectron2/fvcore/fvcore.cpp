#include "Base.h"
#include "fvcore.h"

using namespace std;
using namespace torch;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor fvcore::smooth_l1_loss(const torch::Tensor &input, const torch::Tensor &target,
	float beta, torch::Reduction::Reduction reduction) {
	Tensor loss;
	if (beta < 1e-5) {
		// if beta == 0, then torch.where will result in nan gradients when
		// the chain rule is applied due to pytorch implementation details
		// (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
		// zeros, rather than "no gradient"). To avoid this issue, we define
		// small values of beta to be exactly l1 loss.
		loss = torch::abs(input - target);
	}
	else {
		auto n = torch::abs(input - target);
		auto cond = n < beta;
		loss = torch::where(cond, 0.5 * (n * n) / beta, n - 0.5 * beta);
	}

	if (reduction == Reduction::Mean) {
		loss = loss.mean();
	}
	else if (reduction == Reduction::Sum) {
		loss = loss.sum();
	}
	return loss;
}
