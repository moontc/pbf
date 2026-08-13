#pragma once

#include <cstddef>
#include <vector>

#include "pbf_types.h"
#include "../vector.h"

class CudaPbfSolver final {
public:
    explicit CudaPbfSolver(const PbfParams& p = PbfParams());
    ~CudaPbfSolver();

    CudaPbfSolver(const CudaPbfSolver&) = delete;
    CudaPbfSolver& operator=(const CudaPbfSolver&) = delete;

    void initBlock();

    void debug(bool enabled = true) noexcept { m_debug = enabled; }
    void step(float dtFrame);

    int count() const { return m_n; }
    const PbfParams& params() const { return m_p; }

    const std::vector<Vec3>& positions() const;
    const PbfStats& stats() const { return m_stats; }

    void copyPositionsToDevice(Vec3* destination,
                               std::size_t destinationCount) const;

    // Validation hooks.  Not used by the render loop; they exist so the GPU
    // state can be diffed against the CPU solver's.  Each one synchronises.
    void downloadVelocities(std::vector<Vec3>& out) const;
    void downloadDensities(std::vector<float>& out) const;
    void downloadNeighbors(std::vector<std::vector<int>>& out) const;

    // Non-zero means maxNeighbors was too small and neighbours were silently
    // dropped.  The CPU solver's std::vector grows instead, so a non-zero value
    // here is also the one way the two can legitimately disagree.
    int neighborOverflow() const;

private:
    void allocate();
    void release();
    void substep(float dt, bool wantStats);
    void findNeighbors();

    PbfParams m_p;
    int m_n = 0;
    bool m_debug = false;

    // Kernel constants, derived from m_p once in the constructor.
    float m_kPoly6 = 0.0f;
    float m_kSpiky = 0.0f;
    float m_wSelf  = 0.0f;
    float m_wDq    = 1.0f;

    Vec3 m_gridLo;
    int  m_nx = 1, m_ny = 1, m_nz = 1, m_nCells = 1;

    // Device buffers.
    Vec3*  d_x         = nullptr;
    Vec3*  d_v         = nullptr;
    Vec3*  d_xp        = nullptr;
    Vec3*  d_xTmp      = nullptr;
    Vec3*  d_vTmp      = nullptr;
    Vec3*  d_xpTmp     = nullptr;
    Vec3*  d_dp        = nullptr;
    Vec3*  d_omega     = nullptr;
    Vec3*  d_dv        = nullptr;   // XSPH scratch
    float* d_lambda    = nullptr;
    float* d_density   = nullptr;

    int*   d_cellOf    = nullptr;
    int*   d_cellCount = nullptr;   // nCells+1, histogram before the scan
    int*   d_cellStart = nullptr;   // nCells+1, prefix sums
    int*   d_cursor    = nullptr;   // nCells,   scatter write heads
    int*   d_sorted    = nullptr;
    int*   d_id        = nullptr;   // current slot -> original particle id
    int*   d_idTmp     = nullptr;

    int*   d_nbr       = nullptr;
    int*   d_nbrCount  = nullptr;
    int*   d_overflow  = nullptr;

    void*  d_scanTemp  = nullptr;
    unsigned long long m_scanTempBytes = 0;

    void*  d_stats     = nullptr;

    mutable std::vector<Vec3> m_hostX;
    mutable bool m_hostPositionsDirty = false;
    PbfStats m_stats;
};
