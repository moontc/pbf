#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "../vector.h"
#include "pbf_types.h"

static_assert(sizeof(Vec3) == 3 * sizeof(float),
              "Vec3 must be tightly packed for direct GPU upload");

class PbfSolver {
public:

    explicit PbfSolver(const PbfParams& p = PbfParams()) {
        setParams(p);
        initBlock();
    }

    void setParams(const PbfParams& p) {
        params_ = p;

        // 315/(64 π h^9)
        kPoly6_ = 315.0f / (64.0f * kPi_ * std::pow(params_.h, 9.0f));
        // -45/(π h^6)
        kSpiky_ = -45.0f / (kPi_ * std::pow(params_.h, 6.0f));

        wSelf_ = wPoly6(0.0f);

        wDq_ = wPoly6(params_.deltaQ * params_.h);
        if (wDq_ <= 0.0f) wDq_ = 1.0f;
    }

    const PbfParams& params() const { return params_; }


    void initBlock() {
        const Vec3& lo = params_.blockLo;
        const Vec3& hi = params_.blockHi;

        std::mt19937 rng(params_.seed);
        std::uniform_real_distribution<float> jitter(-0.5f, 0.5f);

        const float amp = 1e-4f;

        positions_.clear();
        for (float px = lo.x; px <= hi.x + 1e-6f; px += params_.d)
        for (float py = lo.y; py <= hi.y + 1e-6f; py += params_.d)
        for (float pz = lo.z; pz <= hi.z + 1e-6f; pz += params_.d) {
            positions_.push_back(Vec3(px + jitter(rng) * amp,
                               py + jitter(rng) * amp,
                               pz + jitter(rng) * amp));
        }
        allocate();
    }

    void step(float dtFrame) {
        const float dt = dtFrame / static_cast<float>(params_.substeps);
        for (int s = 0; s < params_.substeps; ++s) substep(dt);
    }


    int  count() const { return static_cast<int>(positions_.size()); }
    const std::vector<Vec3>&  positions()  const { return positions_; }
    const std::vector<Vec3>&  velocities() const { return velocities_; }
    const std::vector<float>& lambdas()    const { return lambdas_; }
    const std::vector<float>& densities()  const { return densities_; }
    const PbfStats& stats() const { return stats_; }

    const std::vector<std::vector<int>>& neighbors() const { return neighbors_; }

    float wPoly6(float r) const {
        if (r < 0.0f || r >= params_.h) return 0.0f;
        const float t = params_.h * params_.h - r * r;
        return kPoly6_ * t * t * t;
    }

    Vec3 gradWSpiky(const Vec3& rij) const {
        const float r = len(rij);
        if (r >= params_.h || r < 1e-9f) return Vec3(0, 0, 0);
        const float c = kSpiky_ * (params_.h - r) * (params_.h - r);
        return rij * (c / r);
    }

private:
    void allocate() {
        const size_t n = positions_.size();
        velocities_.assign(n, Vec3(0, 0, 0));
        predictedPositions_.assign(n, Vec3(0, 0, 0));
        positionDeltas_.assign(n, Vec3(0, 0, 0));
        lambdas_.assign(n, 0.0f);
        densities_.assign(n, 0.0f);
        neighbors_.assign(n, {});
        for (auto& v : neighbors_) v.reserve(params_.maxNeighbors);
    }

    void buildGrid() {
        const int n = count();

        const float pad = 2.0f * params_.h;
        gridLo_ = Vec3(params_.boxLo.x - pad, params_.boxLo.y - pad, params_.boxLo.z - pad);
        const Vec3 hi(params_.boxHi.x + pad, params_.boxHi.y + pad, params_.boxHi.z + pad);

        const float inv = 1.0f / params_.h;
        nx_ = std::max(1, static_cast<int>((hi.x - gridLo_.x) * inv) + 1);
        ny_ = std::max(1, static_cast<int>((hi.y - gridLo_.y) * inv) + 1);
        nz_ = std::max(1, static_cast<int>((hi.z - gridLo_.z) * inv) + 1);
        const int nCells = nx_ * ny_ * nz_;

        cellOf_.resize(n);
        sorted_.resize(n);
        cellStart_.assign(nCells + 1, 0);

        for (int i = 0; i < n; ++i) {
            const int cx = std::clamp(static_cast<int>((predictedPositions_[i].x - gridLo_.x) * inv), 0, nx_ - 1);
            const int cy = std::clamp(static_cast<int>((predictedPositions_[i].y - gridLo_.y) * inv), 0, ny_ - 1);
            const int cz = std::clamp(static_cast<int>((predictedPositions_[i].z - gridLo_.z) * inv), 0, nz_ - 1);
            cellOf_[i] = (cz * ny_ + cy) * nx_ + cx;
        }

        for (int i = 0; i < n; ++i) ++cellStart_[cellOf_[i] + 1];

        for (int c = 0; c < nCells; ++c) cellStart_[c + 1] += cellStart_[c];

        cursor_.assign(cellStart_.begin(), cellStart_.end() - 1);
        for (int i = 0; i < n; ++i) sorted_[cursor_[cellOf_[i]]++] = i;
    }

    void findNeighbors() {
        buildGrid();

        const int n = count();
        const float h2 = params_.h * params_.h;
        const float inv = 1.0f / params_.h;

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            neighbors_[i].clear();

            const int cx = std::clamp(static_cast<int>((predictedPositions_[i].x - gridLo_.x) * inv), 0, nx_ - 1);
            const int cy = std::clamp(static_cast<int>((predictedPositions_[i].y - gridLo_.y) * inv), 0, ny_ - 1);
            const int cz = std::clamp(static_cast<int>((predictedPositions_[i].z - gridLo_.z) * inv), 0, nz_ - 1);

            for (int dz = -1; dz <= 1; ++dz) {
                const int z = cz + dz;  if (z < 0 || z >= nz_) continue;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int y = cy + dy;  if (y < 0 || y >= ny_) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int x = cx + dx;  if (x < 0 || x >= nx_) continue;

                        const int c = (z * ny_ + y) * nx_ + x;

                        for (int k = cellStart_[c]; k < cellStart_[c + 1]; ++k) {
                            const int j = sorted_[k];
                            if (j == i) continue;
                            const Vec3 d = predictedPositions_[i] - predictedPositions_[j];
                            if (dot(d, d) < h2) neighbors_[i].push_back(j);
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

            float rho = params_.mass * wSelf_;

            Vec3  gradI(0, 0, 0);   // 公式 (8) 上半支 k=i：向量累加
            float sumSq = 0.0f;     // 公式 (8) 下半支 k=j：标量累加

            for (int j : neighbors_[i]) {
                const Vec3 rij = predictedPositions_[i] - predictedPositions_[j];
                rho += params_.mass * wPoly6(len(rij));

                // 约束梯度：grad C = (m/rho0) * grad W
                const Vec3 g = gradWSpiky(rij) * (params_.mass / params_.rho0);
                gradI += g;
                sumSq += dot(g, g);
            }
            sumSq += dot(gradI, gradI);

            densities_[i] = rho;

            const float C = rho / params_.rho0 - 1.0f;

            // 公式 (11)：CFM 正则化。
            lambdas_[i] = -C / (sumSq + params_.eps);
        }
    }

    void computeDeltaP() {
        const int n = count();
        const bool useScorr = (params_.kCorr > 0.0f);

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            Vec3 dp(0, 0, 0);
            for (int j : neighbors_[i]) {
                const Vec3 rij = predictedPositions_[i] - predictedPositions_[j];

                float coef = lambdas_[i] + lambdas_[j];

                if (useScorr) {
                    float ratio = wPoly6(len(rij)) / wDq_;
                    float pw = 1.0f;
                    for (int e = 0; e < params_.nCorr; ++e) pw *= ratio;
                    coef += -params_.kCorr * pw;
                }

                dp += gradWSpiky(rij) * coef;
            }

            positionDeltas_[i] = dp * (params_.mass / params_.rho0 * params_.omega);
        }
    }

    void enforceBoundary() {
        const int   n = count();
        const float m = 0.5f * params_.d;
        const float lo[3] = { params_.boxLo.x + m, params_.boxLo.y + m, params_.boxLo.z + m };
        const float hi[3] = { params_.boxHi.x - m, params_.boxHi.y - m, params_.boxHi.z - m };

        for (int i = 0; i < n; ++i) {
            float* p = &predictedPositions_[i].x;
            for (int a = 0; a < 3; ++a) {
                if (p[a] < lo[a]) p[a] = lo[a];
                if (p[a] > hi[a]) p[a] = hi[a];
            }
        }
    }

    void applyVorticity(float dt) {
        if (params_.vorticity <= 0.0f) return;
        const int n = count();

        // 公式 15
        omega_.assign(n, Vec3(0, 0, 0));
        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
            for (int j : neighbors_[i]) {

                const Vec3 g = gradWSpiky(positions_[i] - positions_[j]) * (-params_.mass / densities_[j]);
                omega_[i] += cross(velocities_[j] - velocities_[i], g);
            }

        // eta = grad|omega| -> N -> 加力（公式 16）
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            Vec3 eta(0, 0, 0);
            for (int j : neighbors_[i]) {
                const Vec3 g = gradWSpiky(positions_[i] - positions_[j]) * (params_.mass / densities_[j]);
                eta += g * (len(omega_[j]) - len(omega_[i]));
            }
            const float e = len(eta);
            if (e < 1e-9f) continue;
            const Vec3 N = eta * (1.0f / e);
            velocities_[i] += cross(N, omega_[i]) * (params_.vorticity * dt);
        }
    }

    void applyXSPH(float /*dt*/) {
        if (params_.xsph <= 0.0f) return;
        const int n = count();

        std::vector<Vec3> dv(n, Vec3(0, 0, 0));
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            const float rhoI = densities_[i];
            for (int j : neighbors_[i]) {
                const float w = wPoly6(len(positions_[i] - positions_[j]));

                const float wt = 2.0f * params_.mass / (rhoI + densities_[j]);
                dv[i] += (velocities_[j] - velocities_[i]) * (params_.xsph * wt * w);
            }
        }
        for (int i = 0; i < n; ++i) velocities_[i] += dv[i];
    }

    void substep(float dt) {
        const int n = count();

        // 施加外力
        for (int i = 0; i < n; ++i) {
            velocities_[i] += params_.gravity * dt;
            predictedPositions_[i] = positions_[i] + velocities_[i] * dt;
        }

        findNeighbors();

        for (int it = 0; it < params_.solverIters; ++it) {
            computeLambda();
            computeDeltaP();
            for (int i = 0; i < n; ++i) predictedPositions_[i] += positionDeltas_[i];
            enforceBoundary();
        }

        // 反推速度，提交位置
        stats_ = PbfStats();
        const float lim = params_.cflFactor * params_.h / dt;
        const float mrg = 0.5f * params_.d;

        for (int i = 0; i < n; ++i) {
            velocities_[i] = (predictedPositions_[i] - positions_[i]) * (1.0f / dt);

            // CFL 安全网
            const float sp = len(velocities_[i]);
            if (sp > lim) {
                velocities_[i]  = velocities_[i] * (lim / sp);
                predictedPositions_[i] = positions_[i] + velocities_[i] * dt;
                ++stats_.cflHits;
            }

            positions_[i] = predictedPositions_[i];

            // 收集诊断量
            stats_.rhoAvg   += densities_[i];
            stats_.rhoMax    = std::max(stats_.rhoMax, densities_[i]);
            stats_.vMax      = std::max(stats_.vMax, len(velocities_[i]));
            stats_.momentum += velocities_[i] * params_.mass;

            const float* p = &positions_[i].x;
            if (p[0] < params_.boxLo.x + mrg + 1e-6f || p[0] > params_.boxHi.x - mrg - 1e-6f ||
                p[1] < params_.boxLo.y + mrg + 1e-6f || p[1] > params_.boxHi.y - mrg - 1e-6f ||
                p[2] < params_.boxLo.z + mrg + 1e-6f || p[2] > params_.boxHi.z - mrg - 1e-6f)
                ++stats_.clamped;
        }
        if (n) stats_.rhoAvg /= static_cast<float>(n);

        applyVorticity(dt);
        applyXSPH(dt);

        for (int i = 0; i < n; ++i) {
            float sp = len(velocities_[i]);
            if (sp > lim) { velocities_[i] = velocities_[i] * (lim / sp); ++stats_.cflHits; }
            stats_.vMax = std::max(stats_.vMax, len(velocities_[i]));
        }
    }


    static constexpr float kPi_ = 3.14159265358979323846f;

    PbfParams params_;
    float kPoly6_ = 0.0f;          // Poly6 归一化系数
    float kSpiky_ = 0.0f;          // Spiky 梯度系数
    float wSelf_  = 0.0f;          // W(0,h)，密度的自身项

    std::vector<Vec3>  positions_;                  // 位置（帧开始时）
    std::vector<Vec3>  velocities_;                 // 速度
    std::vector<Vec3>  predictedPositions_;         // 预测位置 x*
    std::vector<Vec3>  positionDeltas_;             // 位置修正
    std::vector<float> lambdas_;                    // 拉格朗日乘子
    std::vector<float> densities_;                  // 密度

    std::vector<std::vector<int>> neighbors_;
    Vec3 gridLo_ = Vec3(0, 0, 0);
    int  nx_ = 1, ny_ = 1, nz_ = 1;
    std::vector<int> cellOf_;
    std::vector<int> sorted_;
    std::vector<int> cellStart_;
    std::vector<int> cursor_;

    float wDq_ = 1.0f;

    std::vector<Vec3> omega_;

    PbfStats stats_;
};