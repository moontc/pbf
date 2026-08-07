#pragma once

#include <chrono>
#include <cstdio>

#include <glad/gl.h>

// Splits the frame into solve / render / swap and prints a running average.
//
// This is an instrument on the live render loop rather than a test, which is
// why it lives in src/ and not test/: the loop it measures cannot be lifted out
// of main.cpp, and having src/ include from test/ would invert the dependency.
//
// Reading the output: `swap` is where a vsync wait shows up, so a large swap
// with small solve/render means the frame is finishing early and the refresh
// rate is the limit.  A swap near zero means the frame already overruns the
// refresh interval and vsync is irrelevant.
//
// Disabled instances compile down to nothing worth caring about; enabled ones
// cost a glFinish per frame (see afterRender), so switch it off once a question
// has been answered.
class FrameProfiler {
public:
    explicit FrameProfiler(bool enabled, int interval = 60)
        : enabled_(enabled), interval_(interval) {}

    void frameStart()
    {
        if (enabled_) t0_ = Clock::now();
    }

    // `steps` is how many fixed timesteps the accumulator actually ran.  Worth
    // printing: when the frame overruns the fixed timestep the accumulator
    // starts doing catch-up steps, so the solver is charged for more substeps
    // than one frame of simulation needs -- and the fluid is simultaneously
    // running in slow motion.  An average above 1.0 says so.
    void afterSolve(int steps)
    {
        if (!enabled_) return;
        t1_ = Clock::now();
        steps_ += steps;
    }

    // The glFinish is the point of this call.  GL commands are queued, not
    // executed, so without it the GPU's render cost is not attributed to the
    // render phase at all -- it lands in SwapBuffers together with the vsync
    // wait, and separating those two is usually the whole question.
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

        std::printf("%7d particles | solve %6.2f (%.2f steps) | render %6.2f "
                    "| swap %6.2f | total %6.2f ms = %5.1f fps\n",
                    particleCount,
                    solve_ / n, steps_ / n,
                    render_ / n, swap_ / n,
                    total / n, 1000.0 * n / total);

        // stdout is block buffered when redirected to a file, and this program
        // is normally ended by closing its window -- without the flush the last
        // several reports are simply lost.
        std::fflush(stdout);

        solve_ = render_ = swap_ = steps_ = 0.0;
        frames_ = 0;
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
    double swap_   = 0.0;
    double steps_  = 0.0;
    int    frames_ = 0;
};