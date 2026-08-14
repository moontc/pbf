#include "cuda_pbf_solver.cuh"

#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                              \
            throw std::runtime_error(                                           \
                std::string("CUDA error at " __FILE__ ":")                      \
                + std::to_string(__LINE__) + " -- "                             \
                + cudaGetErrorString(err_));                                    \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

namespace {

constexpr int kBlock = 256;
inline int gridFor(int n) { return (n + kBlock - 1) / kBlock; }

struct GpuParams {
    float rho0, mass, eps, h, d, omega, cflFactor;
    float kCorr, deltaQ;
    int   nCorr;
    float vorticity, xsph;
    Vec3  gravity, boxLo, boxHi;
    int   maxNeighbors;

    float kPoly6, kSpiky, wSelf, wDq;

    Vec3  gridLo;
    int   nx, ny, nz, nCells;

    int   n;

    int   wantStats;
};

struct GpuStats {
    float rhoSum;
    float rhoMax;
    float vMax;
    float momX, momY, momZ;
    int   clamped;
    int   cflHits;
};

// 仅适用于非负值，对非负值而言，IEEE-754 位模式的排序与数值排序一致
__device__ __forceinline__ void atomicMaxFloat(float* addr, float value)
{
    atomicMax(reinterpret_cast<unsigned int*>(addr), __float_as_uint(value));
}

__device__ __forceinline__ float wPoly6(float r, const GpuParams& p)
{
    if (r < 0.0f || r >= p.h) return 0.0f;
    const float t = p.h * p.h - r * r;
    return p.kPoly6 * t * t * t;
}

__device__ __forceinline__ Vec3 gradWSpiky(const Vec3& rij, const GpuParams& p)
{
    const float r = len(rij);
    if (r >= p.h || r < 1e-9f) return {0, 0, 0};
    const float c = p.kSpiky * (p.h - r) * (p.h - r);
    return rij * (c / r);
}

__device__ __forceinline__ int cellIndex(const Vec3& q, const GpuParams& p)
{
    const float inv = 1.0f / p.h;

    const int cx = min(max(static_cast<int>((q.x - p.gridLo.x) * inv), 0), p.nx - 1);
    const int cy = min(max(static_cast<int>((q.y - p.gridLo.y) * inv), 0), p.ny - 1);
    const int cz = min(max(static_cast<int>((q.z - p.gridLo.z) * inv), 0), p.nz - 1);
    return (cz * p.ny + cy) * p.nx + cx;
}

__global__ void kPredict(Vec3* x, Vec3* v, Vec3* xp, GpuParams p, float dt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;

    Vec3 vi = v[i] + p.gravity * dt;
    v[i]  = vi;
    xp[i] = x[i] + vi * dt;
}

__global__ void kCellOfAndCount(const Vec3* xp, int* cellOf, int* cellCount, GpuParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;

    const int c = cellIndex(xp[i], p);
    cellOf[i] = c;
    atomicAdd(&cellCount[c + 1], 1);
}

__global__ void kScatter(const int* cellOf, int* cursor, int* sorted, GpuParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;

    sorted[atomicAdd(&cursor[cellOf[i]], 1)] = i;
}

__global__ void kInitIds(int* id, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) id[i] = i;
}

// 重排粒子
__global__ void kReorderState(const int* sorted,
                              const Vec3* xIn,
                              const Vec3* vIn,
                              const Vec3* xpIn,
                              const int* idIn,
                              Vec3* xOut,
                              Vec3* vOut,
                              Vec3* xpOut,
                              int* idOut,
                              int n)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n) return;

    const int old = sorted[slot];
    xOut[slot]  = xIn[old];
    vOut[slot]  = vIn[old];
    xpOut[slot] = xpIn[old];
    idOut[slot] = idIn[old];
}

__global__ void kScatterVec3ById(const Vec3* sortedValues,
                                 const int* id,
                                 Vec3* originalOrder,
                                 int n)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n) return;
    originalOrder[id[slot]] = sortedValues[slot];
}

__global__ void kFindNeighbors(const Vec3* xp,
                               const int* cellStart,
                               int* nbr,
                               int* nbrCount,
                               int* overflow,
                               GpuParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;

    const float h2 = p.h * p.h;
    const Vec3  xi = xp[i];

    const float inv = 1.0f / p.h;
    const int cx = min(max(static_cast<int>((xi.x - p.gridLo.x) * inv), 0), p.nx - 1);
    const int cy = min(max(static_cast<int>((xi.y - p.gridLo.y) * inv), 0), p.ny - 1);
    const int cz = min(max(static_cast<int>((xi.z - p.gridLo.z) * inv), 0), p.nz - 1);

    int cnt = 0;

    for (int dz = -1; dz <= 1; ++dz) {
        const int z = cz + dz;  if (z < 0 || z >= p.nz) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            const int y = cy + dy;  if (y < 0 || y >= p.ny) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                const int x = cx + dx;  if (x < 0 || x >= p.nx) continue;

                const int c = (z * p.ny + y) * p.nx + x;

                for (int j = cellStart[c]; j < cellStart[c + 1]; ++j) {
                    if (j == i) continue;
                    const Vec3 rij = xi - xp[j];
                    if (dot(rij, rij) < h2) {
                        if (cnt < p.maxNeighbors) {
                            nbr[cnt * p.n + i] = j;
                            ++cnt;
                        } else {
                            atomicAdd(overflow, 1);
                        }
                    }
                }
            }
        }
    }

    nbrCount[i] = cnt;
}

__global__ void kComputeLambda(const Vec3* xp,
                               const int* nbr,
                               const int* nbrCount,
                               float* lambda,
                               float* density,
                               GpuParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;

    const Vec3 xi = xp[i];

    float rho = p.mass * p.wSelf;

    Vec3  gradI(0, 0, 0);       // 公式 (8) 的 k = i 分支：先累加，再平方
    float sumSq = 0.0f;                // 公式 (8) 的 k = j 分支：先平方，再累加

    const int cnt = nbrCount[i];
    for (int k = 0; k < cnt; ++k) {
        const int j = nbr[k * p.n + i];
        const Vec3 rij = xi - xp[j];
        rho += p.mass * wPoly6(len(rij), p);

        const Vec3 g = gradWSpiky(rij, p) * (p.mass / p.rho0);
        gradI += g;
        sumSq += dot(g, g);
    }
    sumSq += dot(gradI, gradI);

    density[i] = rho;

    const float C = rho / p.rho0 - 1.0f;
    lambda[i] = -C / (sumSq + p.eps);   // 公式 (11)，CFM 正则化
}

__global__ void kComputeDeltaP(const Vec3* xp,
                               const int* nbr,
                               const int* nbrCount,
                               const float* lambda,
                               Vec3* dp,
                               GpuParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;

    const Vec3  xi = xp[i];
    const float li = lambda[i];
    const bool  useScorr = (p.kCorr > 0.0f);

    Vec3 acc(0, 0, 0);

    const int cnt = nbrCount[i];
    for (int k = 0; k < cnt; ++k) {
        const int j = nbr[k * p.n + i];
        const Vec3 rij = xi - xp[j];

        float coef = li + lambda[j];

        if (useScorr) {
            const float ratio = wPoly6(len(rij), p) / p.wDq;
            float pw = 1.0f;
            for (int e = 0; e < p.nCorr; ++e) pw *= ratio;
            coef += -p.kCorr * pw;
        }

        acc += gradWSpiky(rij, p) * coef;
    }

    dp[i] = acc * (p.mass / p.rho0 * p.omega);
}

__global__ void kApplyDeltaPAndBound(Vec3* xp, const Vec3* dp, GpuParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;

    Vec3 q = xp[i] + dp[i];

    const float m = 0.5f * p.d;
    q.x = fminf(fmaxf(q.x, p.boxLo.x + m), p.boxHi.x - m);
    q.y = fminf(fmaxf(q.y, p.boxLo.y + m), p.boxHi.y - m);
    q.z = fminf(fmaxf(q.z, p.boxLo.z + m), p.boxHi.z - m);

    xp[i] = q;
}

__global__ void kVelocityCommit(Vec3* x,
                                Vec3* v,
                                const Vec3* xp,
                                const float* density,
                                GpuStats* st,
                                GpuParams p,
                                float dt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;

    const float lim = p.cflFactor * p.h / dt;
    const float mrg = 0.5f * p.d;

    const Vec3 xi = x[i];
    Vec3 vi = (xp[i] - xi) * (1.0f / dt);
    Vec3 q  = xp[i];

    const float sp = len(vi);
    if (sp > lim) {
        vi = vi * (lim / sp);
        q  = xi + vi * dt;
        if (p.wantStats) atomicAdd(&st->cflHits, 1);
    }

    v[i] = vi;
    x[i] = q;

    if (!p.wantStats) return;

    atomicAdd(&st->rhoSum, density[i]);
    atomicMaxFloat(&st->rhoMax, density[i]);
    atomicMaxFloat(&st->vMax, len(vi));
    atomicAdd(&st->momX, vi.x * p.mass);
    atomicAdd(&st->momY, vi.y * p.mass);
    atomicAdd(&st->momZ, vi.z * p.mass);

    if (q.x < p.boxLo.x + mrg + 1e-6f || q.x > p.boxHi.x - mrg - 1e-6f ||
        q.y < p.boxLo.y + mrg + 1e-6f || q.y > p.boxHi.y - mrg - 1e-6f ||
        q.z < p.boxLo.z + mrg + 1e-6f || q.z > p.boxHi.z - mrg - 1e-6f) {
        atomicAdd(&st->clamped, 1);
    }
}

__global__ void kVorticityOmega(const Vec3* x,
                                const Vec3* v,
                                const float* density,
                                const int* nbr,
                                const int* nbrCount,
                                Vec3* omega,
                                GpuParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;

    const Vec3 xi = x[i];
    const Vec3 vi = v[i];

    Vec3 w(0, 0, 0);

    const int cnt = nbrCount[i];
    for (int k = 0; k < cnt; ++k) {
        const int j = nbr[k * p.n + i];
        const Vec3 g = gradWSpiky(xi - x[j], p) * (-p.mass / density[j]); // 公式 (15)
        w += cross(v[j] - vi, g);
    }

    omega[i] = w;
}

__global__ void kVorticityApply(const Vec3* x,
                                Vec3* v,
                                const float* density,
                                const int* nbr,
                                const int* nbrCount,
                                const Vec3* omega,
                                GpuParams p,
                                float dt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;

    const Vec3  xi = x[i];
    const Vec3  wi = omega[i];
    const float lwi = len(wi);

    Vec3 eta(0, 0, 0);

    const int cnt = nbrCount[i];
    for (int k = 0; k < cnt; ++k) {
        const int j = nbr[k * p.n + i];
        const Vec3 g = gradWSpiky(xi - x[j], p) * (p.mass / density[j]);
        eta += g * (len(omega[j]) - lwi);
    }

    const float e = len(eta);
    if (e < 1e-9f) return;

    const Vec3 N = eta * (1.0f / e);
    v[i] += cross(N, wi) * (p.vorticity * dt);   // 公式 (16)
}

__global__ void kXsphCompute(const Vec3* x,
                             const Vec3* v,
                             const float* density,
                             const int* nbr,
                             const int* nbrCount,
                             Vec3* dv,
                             GpuParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;

    const Vec3  xi   = x[i];
    const Vec3  vi   = v[i];
    const float rhoI = density[i];

    Vec3 acc(0, 0, 0);

    const int cnt = nbrCount[i];
    for (int k = 0; k < cnt; ++k) {
        const int j = nbr[k * p.n + i];
        const float w = wPoly6(len(xi - x[j]), p);

        const float wt = 2.0f * p.mass / (rhoI + density[j]); // 公式 (17)
        acc += (v[j] - vi) * (p.xsph * wt * w);
    }

    dv[i] = acc;
}

// 单独执行应用步骤：如果在上面的聚集循环中直接应用 dv，粒子就可能读到邻居已经
// 更新后的速度。这正是雅可比法与高斯-赛德尔法的区别，会使结果依赖线程执行顺序。
__global__ void kXsphApply(Vec3* v, const Vec3* dv, GpuParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;
    v[i] += dv[i];
}

// 第二次 CFL 限制：涡量和 XSPH 都在第一次限制之后增加速度
__global__ void kCflClamp2(Vec3* v, GpuStats* st, GpuParams p, float dt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;

    const float lim = p.cflFactor * p.h / dt;

    Vec3 vi = v[i];
    const float sp = len(vi);
    if (sp > lim) {
        vi = vi * (lim / sp);
        v[i] = vi;
        if (p.wantStats) atomicAdd(&st->cflHits, 1);
    }
    if (p.wantStats) atomicMaxFloat(&st->vMax, len(vi));
}

} // namespace

CudaPbfSolver::CudaPbfSolver(const PbfParams& p)
{
    params_ = p;

    constexpr float kPi = 3.14159265358979323846f;
    kPoly6_ = 315.0f / (64.0f * kPi * std::pow(params_.h, 9.0f));
    kSpiky_ = -45.0f / (kPi * std::pow(params_.h, 6.0f));
    wSelf_  = kPoly6_ * (params_.h * params_.h) * (params_.h * params_.h) * (params_.h * params_.h);

    const float rq = params_.deltaQ * params_.h;
    const float tq = params_.h * params_.h - rq * rq;
    wDq_ = (rq >= params_.h) ? 1.0f : kPoly6_ * tq * tq * tq;
    if (wDq_ <= 0.0f) wDq_ = 1.0f;

    const float pad = 2.0f * params_.h;
    gridLo_ = Vec3(params_.boxLo.x - pad, params_.boxLo.y - pad, params_.boxLo.z - pad);
    const Vec3 hi(params_.boxHi.x + pad, params_.boxHi.y + pad, params_.boxHi.z + pad);

    const float inv = 1.0f / params_.h;
    nx_ = std::max(1, static_cast<int>((hi.x - gridLo_.x) * inv) + 1);
    ny_ = std::max(1, static_cast<int>((hi.y - gridLo_.y) * inv) + 1);
    nz_ = std::max(1, static_cast<int>((hi.z - gridLo_.z) * inv) + 1);
    nCells_ = nx_ * ny_ * nz_;

    try {
        initBlock();
    } catch (...) {
        release();
        throw;
    }
}

CudaPbfSolver::~CudaPbfSolver()
{
    release();
}

void CudaPbfSolver::release()
{
    cudaFree(dX_);        cudaFree(dV_);         cudaFree(dXp_);
    cudaFree(dXTmp_);     cudaFree(dVTmp_);      cudaFree(dXpTmp_);
    cudaFree(dDp_);       cudaFree(dOmega_);     cudaFree(dDv_);
    cudaFree(dLambda_);   cudaFree(dDensity_);
    cudaFree(dCellOf_);   cudaFree(dCellCount_); cudaFree(dCellStart_);
    cudaFree(dCursor_);   cudaFree(dSorted_);    cudaFree(dId_);
    cudaFree(dIdTmp_);
    cudaFree(dNbr_);      cudaFree(dNbrCount_);  cudaFree(dOverflow_);
    cudaFree(dScanTemp_); cudaFree(dStats_);

    dX_ = dV_ = dXp_ = dDp_ = dOmega_ = dDv_ = nullptr;
    dXTmp_ = dVTmp_ = dXpTmp_ = nullptr;
    dLambda_ = dDensity_ = nullptr;
    dCellOf_ = dCellCount_ = dCellStart_ = dCursor_ = dSorted_ = nullptr;
    dId_ = dIdTmp_ = nullptr;
    dNbr_ = dNbrCount_ = dOverflow_ = nullptr;
    dScanTemp_ = dStats_ = nullptr;
}

void CudaPbfSolver::initBlock()
{
    const Vec3& lo = params_.blockLo;
    const Vec3& hi = params_.blockHi;

    std::mt19937 rng(params_.seed);
    std::uniform_real_distribution<float> jitter(-0.5f, 0.5f);

    const float amp = 1e-4f;

    hostX_.clear();
    for (float px = lo.x; px <= hi.x + 1e-6f; px += params_.d)
        for (float py = lo.y; py <= hi.y + 1e-6f; py += params_.d)
            for (float pz = lo.z; pz <= hi.z + 1e-6f; pz += params_.d) {
                hostX_.emplace_back(px + jitter(rng) * amp,
                                       py + jitter(rng) * amp,
                                       pz + jitter(rng) * amp);
            }
    n_ = static_cast<int>(hostX_.size());
    allocate();

    CUDA_CHECK(cudaMemcpy(dX_, hostX_.data(),
                          sizeof(Vec3) * n_, cudaMemcpyHostToDevice));
    hostPositionsDirty_ = false;
}

void CudaPbfSolver::allocate()
{
    release();

    const auto n  = static_cast<size_t>(n_);
    const auto nc = static_cast<size_t>(nCells_);

    CUDA_CHECK(cudaMalloc(&dX_,       sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&dV_,       sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&dXp_,      sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&dXTmp_,    sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&dVTmp_,    sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&dXpTmp_,   sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&dDp_,      sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&dOmega_,   sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&dDv_,      sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&dLambda_,  sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(&dDensity_, sizeof(float) * n));

    CUDA_CHECK(cudaMalloc(&dCellOf_,    sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&dCellCount_, sizeof(int) * (nc + 1)));
    CUDA_CHECK(cudaMalloc(&dCellStart_, sizeof(int) * (nc + 1)));
    CUDA_CHECK(cudaMalloc(&dCursor_,    sizeof(int) * nc));
    CUDA_CHECK(cudaMalloc(&dSorted_,    sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&dId_,        sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&dIdTmp_,     sizeof(int) * n));

    CUDA_CHECK(cudaMalloc(&dNbr_,      sizeof(int) * n * params_.maxNeighbors));
    CUDA_CHECK(cudaMalloc(&dNbrCount_, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&dOverflow_, sizeof(int)));

    CUDA_CHECK(cudaMalloc(&dStats_, sizeof(GpuStats)));

    CUDA_CHECK(cudaMemset(dV_, 0, sizeof(Vec3) * n));
    CUDA_CHECK(cudaMemset(dOverflow_, 0, sizeof(int)));

    kInitIds<<<gridFor(n_), kBlock>>>(dId_, n_);
    CUDA_CHECK_LAUNCH();

    size_t bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, bytes,
                                  dCellCount_, dCellStart_,
                                  static_cast<int>(nc + 1));
    scanTempBytes_ = bytes;
    CUDA_CHECK(cudaMalloc(&dScanTemp_, bytes));
}

static GpuParams makeGpuParams(const PbfParams& p,
                        float kPoly6, float kSpiky, float wSelf, float wDq,
                        const Vec3& gridLo, int nx, int ny, int nz, int nCells,
                        int n, bool wantStats)
{
    GpuParams g{};
    g.rho0 = p.rho0;  g.mass = p.mass;  g.eps = p.eps;
    g.h = p.h;        g.d = p.d;        g.omega = p.omega;
    g.cflFactor = p.cflFactor;
    g.kCorr = p.kCorr;  g.deltaQ = p.deltaQ;  g.nCorr = p.nCorr;
    g.vorticity = p.vorticity;  g.xsph = p.xsph;
    g.gravity = p.gravity;  g.boxLo = p.boxLo;  g.boxHi = p.boxHi;
    g.maxNeighbors = p.maxNeighbors;
    g.kPoly6 = kPoly6;  g.kSpiky = kSpiky;  g.wSelf = wSelf;  g.wDq = wDq;
    g.gridLo = gridLo;
    g.nx = nx;  g.ny = ny;  g.nz = nz;  g.nCells = nCells;
    g.n = n;
    g.wantStats = wantStats ? 1 : 0;
    return g;
}

void CudaPbfSolver::findNeighbors()
{
    const GpuParams gp = makeGpuParams(params_, kPoly6_, kSpiky_, wSelf_, wDq_,
                                       gridLo_, nx_, ny_, nz_, nCells_, n_,
                                       false);
    const int blocks = gridFor(n_);

    CUDA_CHECK(cudaMemset(dCellCount_, 0, sizeof(int) * (nCells_ + 1)));

    kCellOfAndCount<<<blocks, kBlock>>>(dXp_, dCellOf_, dCellCount_, gp);
    CUDA_CHECK_LAUNCH();

    auto bytes = static_cast<size_t>(scanTempBytes_);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(dScanTemp_, bytes,
                                             dCellCount_, dCellStart_,
                                             nCells_ + 1));

    CUDA_CHECK(cudaMemcpy(dCursor_, dCellStart_, sizeof(int) * nCells_,
                          cudaMemcpyDeviceToDevice));

    kScatter<<<blocks, kBlock>>>(dCellOf_, dCursor_, dSorted_, gp);
    CUDA_CHECK_LAUNCH();

    kReorderState<<<blocks, kBlock>>>(dSorted_,
                                      dX_, dV_, dXp_, dId_,
                                      dXTmp_, dVTmp_, dXpTmp_, dIdTmp_,
                                      n_);
    CUDA_CHECK_LAUNCH();

    std::swap(dX_, dXTmp_);
    std::swap(dV_, dVTmp_);
    std::swap(dXp_, dXpTmp_);
    std::swap(dId_, dIdTmp_);

    kFindNeighbors<<<blocks, kBlock>>>(dXp_, dCellStart_,
                                       dNbr_, dNbrCount_, dOverflow_, gp);
    CUDA_CHECK_LAUNCH();
}

void CudaPbfSolver::substep(float dt, bool wantStats)
{
    const GpuParams gp = makeGpuParams(params_, kPoly6_, kSpiky_, wSelf_, wDq_,
                                       gridLo_, nx_, ny_, nz_, nCells_, n_,
                                       wantStats);
    const int blocks = gridFor(n_);

    kPredict<<<blocks, kBlock>>>(dX_, dV_, dXp_, gp, dt);
    CUDA_CHECK_LAUNCH();

    findNeighbors();

    for (int it = 0; it < params_.solverIters; ++it) {
        kComputeLambda<<<blocks, kBlock>>>(dXp_, dNbr_, dNbrCount_,
                                           dLambda_, dDensity_, gp);
        CUDA_CHECK_LAUNCH();

        kComputeDeltaP<<<blocks, kBlock>>>(dXp_, dNbr_, dNbrCount_,
                                           dLambda_, dDp_, gp);
        CUDA_CHECK_LAUNCH();

        kApplyDeltaPAndBound<<<blocks, kBlock>>>(dXp_, dDp_, gp);
        CUDA_CHECK_LAUNCH();
    }

    if (wantStats) {
        CUDA_CHECK(cudaMemset(dStats_, 0, sizeof(GpuStats)));
    }

    kVelocityCommit<<<blocks, kBlock>>>(dX_, dV_, dXp_, dDensity_,
                                        static_cast<GpuStats*>(dStats_), gp, dt);
    CUDA_CHECK_LAUNCH();

    if (params_.vorticity > 0.0f) {
        kVorticityOmega<<<blocks, kBlock>>>(dX_, dV_, dDensity_,
                                            dNbr_, dNbrCount_, dOmega_, gp);
        CUDA_CHECK_LAUNCH();

        kVorticityApply<<<blocks, kBlock>>>(dX_, dV_, dDensity_,
                                            dNbr_, dNbrCount_, dOmega_, gp, dt);
        CUDA_CHECK_LAUNCH();
    }

    if (params_.xsph > 0.0f) {
        kXsphCompute<<<blocks, kBlock>>>(dX_, dV_, dDensity_,
                                         dNbr_, dNbrCount_, dDv_, gp);
        CUDA_CHECK_LAUNCH();

        kXsphApply<<<blocks, kBlock>>>(dV_, dDv_, gp);
        CUDA_CHECK_LAUNCH();
    }

    kCflClamp2<<<blocks, kBlock>>>(dV_, static_cast<GpuStats*>(dStats_), gp, dt);
    CUDA_CHECK_LAUNCH();
}

void CudaPbfSolver::step(float dtFrame)
{
    if (n_ == 0) return;

    const float dt = dtFrame / static_cast<float>(params_.substeps);

    for (int s = 0; s < params_.substeps; ++s) {
        substep(dt, debug_ && s == params_.substeps - 1);
    }

    if (debug_) {
        GpuStats gs{};
        CUDA_CHECK(cudaMemcpy(&gs, dStats_, sizeof(GpuStats), cudaMemcpyDeviceToHost));

        stats_ = PbfStats();
        stats_.rhoAvg   = n_ ? gs.rhoSum / static_cast<float>(n_) : 0.0f;
        stats_.rhoMax   = gs.rhoMax;
        stats_.vMax     = gs.vMax;
        stats_.momentum = Vec3(gs.momX, gs.momY, gs.momZ);
        stats_.clamped  = gs.clamped;
        stats_.cflHits  = gs.cflHits;
    } else {
        stats_ = PbfStats();
    }

    hostPositionsDirty_ = true;
}

const std::vector<Vec3>& CudaPbfSolver::positions() const
{
    if (!hostPositionsDirty_) return hostX_;

    kScatterVec3ById<<<gridFor(n_), kBlock>>>(dX_, dId_, dXTmp_, n_);
    CUDA_CHECK_LAUNCH();

    CUDA_CHECK(cudaMemcpy(hostX_.data(), dXTmp_,
                          sizeof(Vec3) * n_, cudaMemcpyDeviceToHost));
    hostPositionsDirty_ = false;
    return hostX_;
}

void CudaPbfSolver::copyPositionsToDevice(Vec3* destination,
                                          std::size_t destinationCount) const
{
    if (destination == nullptr) {
        throw std::invalid_argument("CUDA position destination is null");
    }
    if (destinationCount < static_cast<std::size_t>(n_)) {
        throw std::invalid_argument("CUDA position destination is too small");
    }

    CUDA_CHECK(cudaMemcpyAsync(destination, dX_, sizeof(Vec3) * n_,
                               cudaMemcpyDeviceToDevice));
}

void CudaPbfSolver::downloadVelocities(std::vector<Vec3>& out) const
{
    out.resize(n_);
    kScatterVec3ById<<<gridFor(n_), kBlock>>>(dV_, dId_, dVTmp_, n_);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(out.data(), dVTmp_, sizeof(Vec3) * n_,
                          cudaMemcpyDeviceToHost));
}

void CudaPbfSolver::downloadDensities(std::vector<float>& out) const
{
    std::vector<float> sortedDensity(n_);
    std::vector<int> ids(n_);

    CUDA_CHECK(cudaMemcpy(sortedDensity.data(), dDensity_, sizeof(float) * n_,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids.data(), dId_, sizeof(int) * n_,
                          cudaMemcpyDeviceToHost));

    out.resize(n_);
    for (int slot = 0; slot < n_; ++slot) {
        out[ids[slot]] = sortedDensity[slot];
    }
}

void CudaPbfSolver::downloadNeighbors(std::vector<std::vector<int>>& out) const
{
    std::vector<int> flat(static_cast<size_t>(n_) * params_.maxNeighbors);
    std::vector<int> counts(n_);
    std::vector<int> ids(n_);

    CUDA_CHECK(cudaMemcpy(flat.data(), dNbr_,
                          sizeof(int) * flat.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(counts.data(), dNbrCount_,
                          sizeof(int) * n_, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids.data(), dId_,
                          sizeof(int) * n_, cudaMemcpyDeviceToHost));

    out.assign(n_, {});
    for (int slot = 0; slot < n_; ++slot) {
        const int originalId = ids[slot];
        out[originalId].resize(counts[slot]);
        for (int k = 0; k < counts[slot]; ++k) {
            const int neighborSlot = flat[static_cast<size_t>(k) * n_ + slot];
            out[originalId][k] = ids[neighborSlot];
        }
    }
}

int CudaPbfSolver::neighborOverflow() const
{
    int v = 0;
    CUDA_CHECK(cudaMemcpy(&v, dOverflow_, sizeof(int), cudaMemcpyDeviceToHost));
    return v;
}
