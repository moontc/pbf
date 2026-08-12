#pragma once

#include <chrono>
#include <cstdio>

#include <glad/gl.h>

class FrameProfiler {
public:
    explicit FrameProfiler(bool enabled, int interval = 60)
        : enabled_(enabled), interval_(interval) {}

    void frameStart()
    {
        if (enabled_) t0_ = Clock::now();
    }

    void afterSolve(int steps)
    {
        if (!enabled_) return;
        t1_ = Clock::now();
        steps_ += steps;
    }

    void afterRender()
    {
        if (!enabled_) return;
        glFinish();
        t2_ = Clock::now();
    }

    void afterSwap(int particleCount)
    {
        if (!enabled_) return;

        const Clock::time_point t3 = Clock::now();
        solve_  += millis(t1_ - t0_);
        render_ += millis(t2_ - t1_);
        swap_   += millis(t3 - t2_);

        if (++frames_ < interval_) return;

        const double n = static_cast<double>(frames_);
        const double total = solve_ + render_ + swap_;

        const double solvePerStep = steps_ > 0 ? solve_ / steps_ : 0.0;

        std::printf("%7d particles | solve/frame %6.2f | solver/step %6.2f | render %6.2f "
                    "| swap %6.2f | total %6.2f ms = %5.1f fps\n",
                    particleCount, solve_ / n, solvePerStep,
                    render_ / n, swap_ / n,
                    total / n, 1000.0 * n / total);

        // 主动刷新缓冲区
        std::fflush(stdout);

        solve_ = render_ = swap_= frames_ = steps_ = 0;
    }

private:
    using Clock = std::chrono::steady_clock;

    static double millis(Clock::duration d)
    {
        return std::chrono::duration<double, std::milli>(d).count();
    }

    bool enabled_;
    int  interval_;

    Clock::time_point t0_{}, t1_{}, t2_{};

    double solve_  = 0.0;
    double render_ = 0.0;
    int steps_ = 0;
    double swap_   = 0.0;
    int    frames_ = 0;
};
