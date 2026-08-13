#pragma once

#include "../vector.h"

struct PbfParams {
    float rho0 = 1000.0f;           // 静止密度
    float d    = 0.010f;            // 粒子间距
    float h    = 0.020f;            // 光滑核半径 (= 2d)
    float mass = 0.00099032;        // 标定：静止规则排布时 rho_i 必须 = rho0
    float eps  = 18.31001;          // 标定：CFM 松弛，量纲 1/m^2

    // --- 求解器 ---
    int   substeps    = 4;        // 论文 Table 1: Dam Break = 4
    int   solverIters = 4;        // 论文 Table 1: Dam Break = 3

    // omega ~ 4/solverIters
    float omega = 1.0f;

    float cflFactor = 0.4f;

    // --- 场景 ---
    Vec3  gravity = Vec3(0.0f, -9.81f, 0.0f);
    Vec3  boxLo   = Vec3(-1.0f, 0.0f, -0.5f);
    Vec3  boxHi   = Vec3(1.0f, 2.0f, 0.5f);

    // 初始水块
    Vec3  blockLo = Vec3(0.05f, 0.05f, 0.05f);
    Vec3  blockHi = Vec3(0.45f, 0.90f, 0.45f);
    unsigned seed = 10000;

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