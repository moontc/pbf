#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "vector.h"

static_assert(sizeof(Vec3) == 3 * sizeof(float),
              "Vec3 must be tightly packed for direct GPU upload");

struct PbfParams {
    float rho0 = 1000.0f;         // 静止密度
    float d    = 0.0250f;           // 粒子间距
    float h    = 0.050f;           // 光滑核半径 (= 2d)
    float mass = 0.01547374;     // 标定：静止规则排布时 rho_i 必须 = rho0
    float eps  = 2.92960;         // 标定：CFM 松弛，量纲 1/m^2

    // --- 求解器 ---
    int   substeps    = 4;        // 论文 Table 1: Dam Break = 4
    int   solverIters = 4;        // 论文 Table 1: Dam Break = 3

    // omega ~ 4/solverIters
    float omega = 1.0f;

    float cflFactor = 0.4f;

    // --- 场景 ---
    Vec3  gravity = Vec3(0.0f, -9.81f, 0.0f);
    Vec3  boxLo   = Vec3(0.0f, 0.0f, 0.0f);
    Vec3  boxHi   = Vec3(1.0f, 1.0f, 0.5f);

    int   maxNeighbors = 128;     // 邻居表预留容量

    float kCorr  = 0.0f;   // 0 = 关闭
    float deltaQ = 0.3f;
    int   nCorr  = 4;

    float vorticity = 1;

    float xsph = 0.05f;
};

struct PbfStats {
    float rhoAvg   = 0.0f;
    float rhoMax   = 0.0f;
    float vMax     = 0.0f;
    Vec3  momentum = Vec3(0, 0, 0);
    int   clamped  = 0;
    int   cflHits  = 0;
};

// ============================================================================
class PbfSolver {
public:
    explicit PbfSolver(const PbfParams& p = PbfParams()) { setParams(p); }

    void setParams(const PbfParams& p) {
        m_p = p;

        // 315/(64 pi h^9)
        m_kPoly6 = 315.0f / (64.0f * kPi * std::pow(m_p.h, 9.0f));
        // -45/(pi h^6)
        m_kSpiky = -45.0f / (kPi * std::pow(m_p.h, 6.0f));

        m_wSelf = wPoly6(0.0f);

        m_wDq = wPoly6(m_p.deltaQ * m_p.h);
        if (m_wDq <= 0.0f) m_wDq = 1.0f;
    }

    const PbfParams& params() const { return m_p; }

    void initBlock(const Vec3& lo, const Vec3& hi, unsigned seed = 12345) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> jitter(-0.5f, 0.5f);

        const float amp = 1e-4f;

        m_x.clear();
        for (float px = lo.x; px <= hi.x + 1e-6f; px += m_p.d)
        for (float py = lo.y; py <= hi.y + 1e-6f; py += m_p.d)
        for (float pz = lo.z; pz <= hi.z + 1e-6f; pz += m_p.d) {
            m_x.push_back(Vec3(px + jitter(rng) * amp,
                               py + jitter(rng) * amp,
                               pz + jitter(rng) * amp));
        }
        allocate();
    }

    void step(float dtFrame) {
        const float dt = dtFrame / static_cast<float>(m_p.substeps);
        for (int s = 0; s < m_p.substeps; ++s) substep(dt);
    }


    int  count() const { return static_cast<int>(m_x.size()); }
    const std::vector<Vec3>&  positions()  const { return m_x; }
    const std::vector<Vec3>&  velocities() const { return m_v; }
    const std::vector<float>& lambdas()    const { return m_lambda; }
    const std::vector<float>& densities()  const { return m_density; }
    const PbfStats& stats() const { return m_stats; }

    const std::vector<std::vector<int>>& neighbors() const { return m_nbrs; }

    float wPoly6(float r) const {
        if (r < 0.0f || r >= m_p.h) return 0.0f;
        const float t = m_p.h * m_p.h - r * r;
        return m_kPoly6 * t * t * t;
    }

    Vec3 gradWSpiky(const Vec3& rij) const {
        const float r = len(rij);
        if (r >= m_p.h || r < 1e-9f) return Vec3(0, 0, 0);
        const float c = m_kSpiky * (m_p.h - r) * (m_p.h - r);
        return rij * (c / r);
    }

private:
    void allocate() {
        const size_t n = m_x.size();
        m_v.assign(n, Vec3(0, 0, 0));
        m_xp.assign(n, Vec3(0, 0, 0));
        m_dp.assign(n, Vec3(0, 0, 0));
        m_lambda.assign(n, 0.0f);
        m_density.assign(n, 0.0f);
        m_nbrs.assign(n, {});
        for (auto& v : m_nbrs) v.reserve(m_p.maxNeighbors);
    }

    void buildGrid() {
        const int n = count();

        const float pad = 2.0f * m_p.h;
        m_gridLo = Vec3(m_p.boxLo.x - pad, m_p.boxLo.y - pad, m_p.boxLo.z - pad);
        const Vec3 hi(m_p.boxHi.x + pad, m_p.boxHi.y + pad, m_p.boxHi.z + pad);

        const float inv = 1.0f / m_p.h;
        m_nx = std::max(1, static_cast<int>((hi.x - m_gridLo.x) * inv) + 1);
        m_ny = std::max(1, static_cast<int>((hi.y - m_gridLo.y) * inv) + 1);
        m_nz = std::max(1, static_cast<int>((hi.z - m_gridLo.z) * inv) + 1);
        const int nCells = m_nx * m_ny * m_nz;

        m_cellOf.resize(n);
        m_sorted.resize(n);
        m_cellStart.assign(nCells + 1, 0);

        for (int i = 0; i < n; ++i) {
            const int cx = std::clamp(static_cast<int>((m_xp[i].x - m_gridLo.x) * inv), 0, m_nx - 1);
            const int cy = std::clamp(static_cast<int>((m_xp[i].y - m_gridLo.y) * inv), 0, m_ny - 1);
            const int cz = std::clamp(static_cast<int>((m_xp[i].z - m_gridLo.z) * inv), 0, m_nz - 1);
            m_cellOf[i] = (cz * m_ny + cy) * m_nx + cx;
        }

        for (int i = 0; i < n; ++i) ++m_cellStart[m_cellOf[i] + 1];

        for (int c = 0; c < nCells; ++c) m_cellStart[c + 1] += m_cellStart[c];

        m_cursor.assign(m_cellStart.begin(), m_cellStart.end() - 1);
        for (int i = 0; i < n; ++i) m_sorted[m_cursor[m_cellOf[i]]++] = i;
    }

    void findNeighbors() {
        buildGrid();

        const int n = count();
        const float h2 = m_p.h * m_p.h;
        const float inv = 1.0f / m_p.h;

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            m_nbrs[i].clear();

            const int cx = std::clamp(static_cast<int>((m_xp[i].x - m_gridLo.x) * inv), 0, m_nx - 1);
            const int cy = std::clamp(static_cast<int>((m_xp[i].y - m_gridLo.y) * inv), 0, m_ny - 1);
            const int cz = std::clamp(static_cast<int>((m_xp[i].z - m_gridLo.z) * inv), 0, m_nz - 1);

            for (int dz = -1; dz <= 1; ++dz) {
                const int z = cz + dz;  if (z < 0 || z >= m_nz) continue;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int y = cy + dy;  if (y < 0 || y >= m_ny) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int x = cx + dx;  if (x < 0 || x >= m_nx) continue;

                        const int c = (z * m_ny + y) * m_nx + x;

                        for (int k = m_cellStart[c]; k < m_cellStart[c + 1]; ++k) {
                            const int j = m_sorted[k];
                            if (j == i) continue;
                            const Vec3 d = m_xp[i] - m_xp[j];
                            if (dot(d, d) < h2) m_nbrs[i].push_back(j);
                        }
                    }
                }
            }
        }
    }

    void computeLambda() {
        const int n = count();

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {

            float rho = m_p.mass * m_wSelf;

            Vec3  gradI(0, 0, 0);   // 公式 (8) 上半支 k=i：向量累加
            float sumSq = 0.0f;     // 公式 (8) 下半支 k=j：标量累加

            for (int j : m_nbrs[i]) {
                const Vec3 rij = m_xp[i] - m_xp[j];
                rho += m_p.mass * wPoly6(len(rij));

                // grad C = (m/rho0) * grad W
                const Vec3 g = gradWSpiky(rij) * (m_p.mass / m_p.rho0);
                gradI += g;
                sumSq += dot(g, g);
            }
            sumSq += dot(gradI, gradI);

            m_density[i] = rho;

            const float C = rho / m_p.rho0 - 1.0f;

            // 公式 (11)：CFM 正则化。
            m_lambda[i] = -C / (sumSq + m_p.eps);
        }
    }

    void computeDeltaP() {
        const int n = count();
        const bool useScorr = (m_p.kCorr > 0.0f);

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            Vec3 dp(0, 0, 0);
            for (int j : m_nbrs[i]) {
                const Vec3 rij = m_xp[i] - m_xp[j];

                float coef = m_lambda[i] + m_lambda[j];

                if (useScorr) {
                    float ratio = wPoly6(len(rij)) / m_wDq;
                    float pw = 1.0f;
                    for (int e = 0; e < m_p.nCorr; ++e) pw *= ratio;
                    coef += -m_p.kCorr * pw;
                }

                dp += gradWSpiky(rij) * coef;
            }

            m_dp[i] = dp * (m_p.mass / m_p.rho0 * m_p.omega);
        }
    }

    void enforceBoundary() {
        const int   n = count();
        const float m = 0.5f * m_p.d;
        const float lo[3] = { m_p.boxLo.x + m, m_p.boxLo.y + m, m_p.boxLo.z + m };
        const float hi[3] = { m_p.boxHi.x - m, m_p.boxHi.y - m, m_p.boxHi.z - m };

        for (int i = 0; i < n; ++i) {
            float* p = &m_xp[i].x;
            for (int a = 0; a < 3; ++a) {
                if (p[a] < lo[a]) p[a] = lo[a];
                if (p[a] > hi[a]) p[a] = hi[a];
            }
        }
    }

    void applyVorticity(float dt) {
        if (m_p.vorticity <= 0.0f) return;      // ★ 关闭时直接返回，别白跑
        const int n = count();

        // 公式 15
        m_omega.assign(n, Vec3(0, 0, 0));
        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
            for (int j : m_nbrs[i]) {

                const Vec3 g = gradWSpiky(m_x[i] - m_x[j]) * (-m_p.mass / m_density[j]);
                m_omega[i] += cross(m_v[j] - m_v[i], g);
            }

        // eta = grad|omega| -> N -> 加力（公式 16）
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            Vec3 eta(0, 0, 0);
            for (int j : m_nbrs[i]) {
                const Vec3 g = gradWSpiky(m_x[i] - m_x[j]) * (m_p.mass / m_density[j]);
                eta += g * (len(m_omega[j]) - len(m_omega[i]));
            }
            const float e = len(eta);
            if (e < 1e-9f) continue;
            const Vec3 N = eta * (1.0f / e);
            m_v[i] += cross(N, m_omega[i]) * (m_p.vorticity * dt);
        }
    }

    void applyXSPH(float /*dt*/) {
        if (m_p.xsph <= 0.0f) return;
        const int n = count();

        std::vector<Vec3> dv(n, Vec3(0, 0, 0));
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            const float rhoI = m_density[i];
            for (int j : m_nbrs[i]) {
                const float w = wPoly6(len(m_x[i] - m_x[j]));

                const float wt = 2.0f * m_p.mass / (rhoI + m_density[j]);
                dv[i] += (m_v[j] - m_v[i]) * (m_p.xsph * wt * w);
            }
        }
        for (int i = 0; i < n; ++i) m_v[i] += dv[i];
    }

    void substep(float dt) {
        const int n = count();

        // apply forces
        for (int i = 0; i < n; ++i) {
            m_v[i] += m_p.gravity * dt;
            m_xp[i] = m_x[i] + m_v[i] * dt;
        }

        findNeighbors();

        for (int it = 0; it < m_p.solverIters; ++it) {
            computeLambda();
            computeDeltaP();
            for (int i = 0; i < n; ++i) m_xp[i] += m_dp[i];
            enforceBoundary();
        }

        // 反推速度，提交位置
        m_stats = PbfStats();
        const float lim = m_p.cflFactor * m_p.h / dt;
        const float mrg = 0.5f * m_p.d;

        for (int i = 0; i < n; ++i) {
            m_v[i] = (m_xp[i] - m_x[i]) * (1.0f / dt);

            // CFL 安全网
            const float sp = len(m_v[i]);
            if (sp > lim) {
                m_v[i]  = m_v[i] * (lim / sp);
                m_xp[i] = m_x[i] + m_v[i] * dt;
                ++m_stats.cflHits;
            }

            m_x[i] = m_xp[i];

            // 收集诊断量
            m_stats.rhoAvg   += m_density[i];
            m_stats.rhoMax    = std::max(m_stats.rhoMax, m_density[i]);
            m_stats.vMax      = std::max(m_stats.vMax, len(m_v[i]));
            m_stats.momentum += m_v[i] * m_p.mass;

            const float* p = &m_x[i].x;
            if (p[0] < m_p.boxLo.x + mrg + 1e-6f || p[0] > m_p.boxHi.x - mrg - 1e-6f ||
                p[1] < m_p.boxLo.y + mrg + 1e-6f || p[1] > m_p.boxHi.y - mrg - 1e-6f ||
                p[2] < m_p.boxLo.z + mrg + 1e-6f || p[2] > m_p.boxHi.z - mrg - 1e-6f)
                ++m_stats.clamped;
        }
        if (n) m_stats.rhoAvg /= static_cast<float>(n);

        applyVorticity(dt);
        applyXSPH(dt);

        for (int i = 0; i < n; ++i) {
            float sp = len(m_v[i]);
            if (sp > lim) { m_v[i] = m_v[i] * (lim / sp); ++m_stats.cflHits; }
            m_stats.vMax = std::max(m_stats.vMax, len(m_v[i]));
        }
    }


    static constexpr float kPi = 3.14159265358979323846f;

    PbfParams m_p;
    float m_kPoly6 = 0.0f;   // Poly6 归一化系数
    float m_kSpiky = 0.0f;   // Spiky 梯度系数
    float m_wSelf  = 0.0f;   // W(0,h)，密度的自身项

    std::vector<Vec3>  m_x;        // 位置（帧开始时）
    std::vector<Vec3>  m_v;        // 速度
    std::vector<Vec3>  m_xp;       // 预测位置 x*
    std::vector<Vec3>  m_dp;       // 位置修正
    std::vector<float> m_lambda;   // 拉格朗日乘子
    std::vector<float> m_density;  // 密度

    std::vector<std::vector<int>> m_nbrs;
    Vec3 m_gridLo = Vec3(0, 0, 0);
    int  m_nx = 1, m_ny = 1, m_nz = 1;
    std::vector<int> m_cellOf;
    std::vector<int> m_sorted;
    std::vector<int> m_cellStart;
    std::vector<int> m_cursor;

    float m_wDq = 1.0f;

    std::vector<Vec3> m_omega;

    PbfStats m_stats;
};