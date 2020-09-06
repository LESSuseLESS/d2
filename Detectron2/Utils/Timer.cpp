#include "Base.h"
#include "Timer.h"

#include <windows.h>
#include <profileapi.h>

using namespace std;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class QueryPerformanceTimer {
public:
	QueryPerformanceTimer() {
		LARGE_INTEGER li;
		auto ret = QueryPerformanceFrequency(&li);
		assert(ret);
		m_freq = double(li.QuadPart) / 1000.0;
		QueryPerformanceCounter(&li);
		m_counter0 = li.QuadPart;
	}

	double get_counter() {
		LARGE_INTEGER li;
		QueryPerformanceCounter(&li);
		return double(li.QuadPart - m_counter0) / m_freq;
	}

private:
	double m_freq = 0.0;
	__int64 m_counter0 = 0;
};
static QueryPerformanceTimer s_hp_timer;

Timer::Timer(const std::string &name) : m_name(name) {
	m_t0 = (int)s_hp_timer.get_counter();
}

Timer::~Timer() {
	char buf[256];
	snprintf(buf, sizeof(buf), ">>>>>>> %s: %dms\n", m_name.c_str(), (int)s_hp_timer.get_counter() - m_t0);
	cout << buf;
}
