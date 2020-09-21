#include "WindowTimer.h"

using namespace std::chrono;

WindowTimer::WindowTimer() noexcept
{
	last = steady_clock::now();
}

float WindowTimer::Mark() noexcept
{
	const auto old = last;
	last = steady_clock::now();
	const duration<float> frameTime = last - old;
	return frameTime.count();
}

float WindowTimer::Peek() const noexcept
{
	return duration<float>( steady_clock::now() - last ).count();
}
