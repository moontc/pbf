#include "cuda_pbf_solver.cuh"

#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>

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

// Only correct for non-negative values -- for those, IEEE-754 bit patterns sort
// in the same order as the numbers, so an integer max *is* a float max.  Both
// call sites here (a density and a speed) are non-negative by construction.
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
    if (r >= p.h || r < 1e-9f) return Vec3(0, 0, 0);
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

// ------------ kernels ---------------

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

__global__ void kFindNeighbors(const Vec3* xp,
                               const int* cellStart,
                               const int* sorted,
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

                for (int k = cellStart[c]; k < cellStart[c + 1]; ++k) {
                    const int j = sorted[k];
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

    Vec3  gradI(0, 0, 0);   // eq (8), k = i branch: accumulate, then square
    float sumSq = 0.0f;     // eq (8), k = j branch: square, then accumulate

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
    lambda[i] = -C / (sumSq + p.eps);   // eq (11), CFM
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
        const Vec3 g = gradWSpiky(xi - x[j], p) * (-p.mass / density[j]); // eq (15)
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
    v[i] += cross(N, wi) * (p.vorticity * dt);   // eq (16)
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

        const float wt = 2.0f * p.mass / (rhoI + density[j]); // eq (17)
        acc += (v[j] - vi) * (p.xsph * wt * w);
    }

    dv[i] = acc;
}

// Separate pass: applying dv in the gather loop above would let a particle read
// its neighbours' already-updated velocities, which is exactly the Jacobi /
// Gauss-Seidel distinction that makes the result thread-order dependent.
__global__ void kXsphApply(Vec3* v, const Vec3* dv, GpuParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;
    v[i] += dv[i];
}

// Second CFL clamp: vorticity and XSPH both add velocity *after* the first one,
// so without this stats.vMax overshoots the limit by 8-12% and the next
// substep's prediction starts from an unbounded velocity.  No position rewrite
// here -- x is already committed.
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

// ============================================================================

CudaPbfSolver::CudaPbfSolver(const PbfParams& p)
{
    m_p = p;

    constexpr float kPi = 3.14159265358979323846f;
    m_kPoly6 = 315.0f / (64.0f * kPi * std::pow(m_p.h, 9.0f));
    m_kSpiky = -45.0f / (kPi * std::pow(m_p.h, 6.0f));
    m_wSelf  = m_kPoly6 * (m_p.h * m_p.h) * (m_p.h * m_p.h) * (m_p.h * m_p.h);

    const float rq = m_p.deltaQ * m_p.h;
    const float tq = m_p.h * m_p.h - rq * rq;
    m_wDq = (rq >= m_p.h) ? 1.0f : m_kPoly6 * tq * tq * tq;
    if (m_wDq <= 0.0f) m_wDq = 1.0f;

    const float pad = 2.0f * m_p.h;
    m_gridLo = Vec3(m_p.boxLo.x - pad, m_p.boxLo.y - pad, m_p.boxLo.z - pad);
    const Vec3 hi(m_p.boxHi.x + pad, m_p.boxHi.y + pad, m_p.boxHi.z + pad);

    const float inv = 1.0f / m_p.h;
    m_nx = std::max(1, static_cast<int>((hi.x - m_gridLo.x) * inv) + 1);
    m_ny = std::max(1, static_cast<int>((hi.y - m_gridLo.y) * inv) + 1);
    m_nz = std::max(1, static_cast<int>((hi.z - m_gridLo.z) * inv) + 1);
    m_nCells = m_nx * m_ny * m_nz;
}

CudaPbfSolver::~CudaPbfSolver()
{
    release();
}

void CudaPbfSolver::release()
{
    cudaFree(d_x);        cudaFree(d_v);         cudaFree(d_xp);
    cudaFree(d_dp);       cudaFree(d_omega);     cudaFree(d_dv);
    cudaFree(d_lambda);   cudaFree(d_density);
    cudaFree(d_cellOf);   cudaFree(d_cellCount); cudaFree(d_cellStart);
    cudaFree(d_cursor);   cudaFree(d_sorted);
    cudaFree(d_nbr);      cudaFree(d_nbrCount);  cudaFree(d_overflow);
    cudaFree(d_scanTemp); cudaFree(d_stats);

    d_x = d_v = d_xp = d_dp = d_omega = d_dv = nullptr;
    d_lambda = d_density = nullptr;
    d_cellOf = d_cellCount = d_cellStart = d_cursor = d_sorted = nullptr;
    d_nbr = d_nbrCount = d_overflow = nullptr;
    d_scanTemp = d_stats = nullptr;
}

void CudaPbfSolver::initBlock(const Vec3& lo, const Vec3& hi, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> jitter(-0.5f, 0.5f);

    const float amp = 1e-4f;

    m_hostX.clear();
    for (float px = lo.x; px <= hi.x + 1e-6f; px += m_p.d)
    for (float py = lo.y; py <= hi.y + 1e-6f; py += m_p.d)
    for (float pz = lo.z; pz <= hi.z + 1e-6f; pz += m_p.d) {
        m_hostX.push_back(Vec3(px + jitter(rng) * amp,
                               py + jitter(rng) * amp,
                               pz + jitter(rng) * amp));
    }

    m_n = static_cast<int>(m_hostX.size());
    allocate();

    CUDA_CHECK(cudaMemcpy(d_x, m_hostX.data(),
                          sizeof(Vec3) * m_n, cudaMemcpyHostToDevice));
}

void CudaPbfSolver::allocate()
{
    release();

    const size_t n  = static_cast<size_t>(m_n);
    const size_t nc = static_cast<size_t>(m_nCells);

    CUDA_CHECK(cudaMalloc(&d_x,       sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&d_v,       sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&d_xp,      sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&d_dp,      sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&d_omega,   sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&d_dv,      sizeof(Vec3) * n));
    CUDA_CHECK(cudaMalloc(&d_lambda,  sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(&d_density, sizeof(float) * n));

    CUDA_CHECK(cudaMalloc(&d_cellOf,    sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d_cellCount, sizeof(int) * (nc + 1)));
    CUDA_CHECK(cudaMalloc(&d_cellStart, sizeof(int) * (nc + 1)));
    CUDA_CHECK(cudaMalloc(&d_cursor,    sizeof(int) * nc));
    CUDA_CHECK(cudaMalloc(&d_sorted,    sizeof(int) * n));

    CUDA_CHECK(cudaMalloc(&d_nbr,      sizeof(int) * n * m_p.maxNeighbors));
    CUDA_CHECK(cudaMalloc(&d_nbrCount, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d_overflow, sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_stats, sizeof(GpuStats)));

    CUDA_CHECK(cudaMemset(d_v, 0, sizeof(Vec3) * n));
    CUDA_CHECK(cudaMemset(d_overflow, 0, sizeof(int)));

    size_t bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, bytes,
                                  d_cellCount, d_cellStart,
                                  static_cast<int>(nc + 1));
    m_scanTempBytes = bytes;
    CUDA_CHECK(cudaMalloc(&d_scanTemp, bytes));
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
    const GpuParams gp = makeGpuParams(m_p, m_kPoly6, m_kSpiky, m_wSelf, m_wDq,
                                       m_gridLo, m_nx, m_ny, m_nz, m_nCells, m_n,
                                       false);
    const int blocks = gridFor(m_n);

    CUDA_CHECK(cudaMemset(d_cellCount, 0, sizeof(int) * (m_nCells + 1)));

    kCellOfAndCount<<<blocks, kBlock>>>(d_xp, d_cellOf, d_cellCount, gp);
    CUDA_CHECK_LAUNCH();

    size_t bytes = static_cast<size_t>(m_scanTempBytes);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_scanTemp, bytes,
                                             d_cellCount, d_cellStart,
                                             m_nCells + 1));

    CUDA_CHECK(cudaMemcpy(d_cursor, d_cellStart, sizeof(int) * m_nCells,
                          cudaMemcpyDeviceToDevice));

    kScatter<<<blocks, kBlock>>>(d_cellOf, d_cursor, d_sorted, gp);
    CUDA_CHECK_LAUNCH();

    kFindNeighbors<<<blocks, kBlock>>>(d_xp, d_cellStart, d_sorted,
                                       d_nbr, d_nbrCount, d_overflow, gp);
    CUDA_CHECK_LAUNCH();
}

void CudaPbfSolver::substep(float dt, bool wantStats)
{
    const GpuParams gp = makeGpuParams(m_p, m_kPoly6, m_kSpiky, m_wSelf, m_wDq,
                                       m_gridLo, m_nx, m_ny, m_nz, m_nCells, m_n,
                                       wantStats);
    const int blocks = gridFor(m_n);

    kPredict<<<blocks, kBlock>>>(d_x, d_v, d_xp, gp, dt);
    CUDA_CHECK_LAUNCH();

    findNeighbors();

    for (int it = 0; it < m_p.solverIters; ++it) {
        kComputeLambda<<<blocks, kBlock>>>(d_xp, d_nbr, d_nbrCount,
                                           d_lambda, d_density, gp);
        CUDA_CHECK_LAUNCH();

        kComputeDeltaP<<<blocks, kBlock>>>(d_xp, d_nbr, d_nbrCount,
                                           d_lambda, d_dp, gp);
        CUDA_CHECK_LAUNCH();

        kApplyDeltaPAndBound<<<blocks, kBlock>>>(d_xp, d_dp, gp);
        CUDA_CHECK_LAUNCH();
    }

    CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(GpuStats)));

    kVelocityCommit<<<blocks, kBlock>>>(d_x, d_v, d_xp, d_density,
                                        static_cast<GpuStats*>(d_stats), gp, dt);
    CUDA_CHECK_LAUNCH();

    if (m_p.vorticity > 0.0f) {
        kVorticityOmega<<<blocks, kBlock>>>(d_x, d_v, d_density,
                                            d_nbr, d_nbrCount, d_omega, gp);
        CUDA_CHECK_LAUNCH();

        kVorticityApply<<<blocks, kBlock>>>(d_x, d_v, d_density,
                                            d_nbr, d_nbrCount, d_omega, gp, dt);
        CUDA_CHECK_LAUNCH();
    }

    if (m_p.xsph > 0.0f) {
        kXsphCompute<<<blocks, kBlock>>>(d_x, d_v, d_density,
                                         d_nbr, d_nbrCount, d_dv, gp);
        CUDA_CHECK_LAUNCH();

        kXsphApply<<<blocks, kBlock>>>(d_v, d_dv, gp);
        CUDA_CHECK_LAUNCH();
    }

    kCflClamp2<<<blocks, kBlock>>>(d_v, static_cast<GpuStats*>(d_stats), gp, dt);
    CUDA_CHECK_LAUNCH();
}

void CudaPbfSolver::step(float dtFrame)
{
    if (m_n == 0) return;

    const float dt = dtFrame / static_cast<float>(m_p.substeps);

    for (int s = 0; s < m_p.substeps; ++s) {
        substep(dt, s == m_p.substeps - 1);
    }

    GpuStats gs{};
    CUDA_CHECK(cudaMemcpy(&gs, d_stats, sizeof(GpuStats), cudaMemcpyDeviceToHost));

    m_stats = PbfStats();
    m_stats.rhoAvg   = m_n ? gs.rhoSum / static_cast<float>(m_n) : 0.0f;
    m_stats.rhoMax   = gs.rhoMax;
    m_stats.vMax     = gs.vMax;
    m_stats.momentum = Vec3(gs.momX, gs.momY, gs.momZ);
    m_stats.clamped  = gs.clamped;
    m_stats.cflHits  = gs.cflHits;

    CUDA_CHECK(cudaMemcpy(m_hostX.data(), d_x,
                          sizeof(Vec3) * m_n, cudaMemcpyDeviceToHost));
}

void CudaPbfSolver::downloadVelocities(std::vector<Vec3>& out) const
{
    out.resize(m_n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_v, sizeof(Vec3) * m_n,
                          cudaMemcpyDeviceToHost));
}

void CudaPbfSolver::downloadDensities(std::vector<float>& out) const
{
    out.resize(m_n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_density, sizeof(float) * m_n,
                          cudaMemcpyDeviceToHost));
}

void CudaPbfSolver::downloadNeighbors(std::vector<std::vector<int>>& out) const
{
    std::vector<int> flat(static_cast<size_t>(m_n) * m_p.maxNeighbors);
    std::vector<int> counts(m_n);

    CUDA_CHECK(cudaMemcpy(flat.data(), d_nbr,
                          sizeof(int) * flat.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(counts.data(), d_nbrCount,
                          sizeof(int) * m_n, cudaMemcpyDeviceToHost));

    out.assign(m_n, {});
    for (int i = 0; i < m_n; ++i) {
        out[i].resize(counts[i]);
        for (int k = 0; k < counts[i]; ++k) {
            out[i][k] = flat[static_cast<size_t>(k) * m_n + i];   // un-transpose
        }
    }
}

int CudaPbfSolver::neighborOverflow() const
{
    int v = 0;
    CUDA_CHECK(cudaMemcpy(&v, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
    return v;
}