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

    void debug(bool enabled = true) noexcept { debug_ = enabled; }
    void step(float dtFrame);

    int count() const { return n_; }
    const PbfParams& params() const { return params_; }

    const std::vector<Vec3>& positions() const;
    const PbfStats& stats() const { return stats_; }

    void copyPositionsToDevice(Vec3* destination,
                               std::size_t destinationCount) const;

    // 验证接口
    void downloadVelocities(std::vector<Vec3>& out) const;
    void downloadDensities(std::vector<float>& out) const;
    void downloadNeighbors(std::vector<std::vector<int>>& out) const;

    // 非零表示 maxNeighbors 太小，部分邻居被静默丢弃
    int neighborOverflow() const;

private:
    void allocate();
    void release();
    void substep(float dt, bool wantStats);
    void findNeighbors();

    PbfParams params_;
    int n_ = 0;
    bool debug_ = false;

    // 内核常量，在构造函数中根据 params_ 计算一次
    float kPoly6_ = 0.0f;
    float kSpiky_ = 0.0f;
    float wSelf_  = 0.0f;
    float wDq_    = 1.0f;

    Vec3 gridLo_;
    int  nx_ = 1, ny_ = 1, nz_ = 1, nCells_ = 1;

    // 设备缓冲区。
    Vec3*  dX_         = nullptr;
    Vec3*  dV_         = nullptr;
    Vec3*  dXp_        = nullptr;
    Vec3*  dXTmp_      = nullptr;
    Vec3*  dVTmp_      = nullptr;
    Vec3*  dXpTmp_     = nullptr;
    Vec3*  dDp_        = nullptr;
    Vec3*  dOmega_     = nullptr;
    Vec3*  dDv_        = nullptr;   // XSPH 临时缓冲区
    float* dLambda_    = nullptr;
    float* dDensity_   = nullptr;

    int*   dCellOf_    = nullptr;
    int*   dCellCount_ = nullptr;
    int*   dCellStart_ = nullptr;
    int*   dCursor_    = nullptr;
    int*   dSorted_    = nullptr;
    int*   dId_        = nullptr;   // 当前槽位 -> 原始粒子编号
    int*   dIdTmp_     = nullptr;

    int*   dNbr_       = nullptr;
    int*   dNbrCount_  = nullptr;
    int*   dOverflow_  = nullptr;

    void*  dScanTemp_  = nullptr;
    unsigned long long scanTempBytes_ = 0;

    void*  dStats_     = nullptr;

    mutable std::vector<Vec3> hostX_;
    mutable bool hostPositionsDirty_ = false;
    PbfStats stats_;
};
