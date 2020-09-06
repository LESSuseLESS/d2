#pragma once

#include <Detectron2/Base.h>

namespace Detectron2
{
	/**
	 * ms-level timing of blocks. In Python, this is equivalent to:
	 *
	 *   import time
	 *   t0 = time.time()
	 *   ...
	 *   print(">>>>>>> {}: {:.2f}ms".format("some name", (time.time() - t0) * 1000))
	 */
	class Timer {
	public:
		Timer(const std::string &name);
		~Timer();

	private:
		std::string m_name;
		int m_t0;
	};
}
