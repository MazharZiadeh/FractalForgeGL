#pragma once
#include <vector>
#include <random>

// ─── genome + stats ───────────────────────────────────────────────────────────
struct Individual {
    int   maxIter = 256;      // genome
    float obj[4] = {};       // 0:FPSerr 1:GPUms 2:-boundary 3:-density
    int   rank = 0;
    float crowd = 0.0f;
};

// ─── minimalist NSGA‑II wrapper ──────────────────────────────────────────────
class NSGAII {
public:
    NSGAII(int population, int minIter, int maxIter);

    Individual& current();
    bool nextIndividual();                       // true when gen done
    void setFitness(float fpsErr, float gpuMs,
        float boundary, float density);

    void evolve();                               // next generation

    // for main.cpp
    void recalcRanks();                          // recompute ranks only
    const std::vector<Individual>& population() const { return pop; }
    const Individual& best() const;

private:
    int popSize, minIter, maxIter;
    int evalIndex = 0;

    std::vector<Individual> pop;
    std::vector<Individual> offspring;
    std::mt19937 rng{ std::random_device{}() };

    // core helpers
    static bool dominates(const Individual& a, const Individual& b);
    void assignRanks();
    void calcCrowding(std::vector<Individual*>& front);
    Individual tournament();

    // fallback clamp (works even if std::clamp missing)
    template<typename T>
    static T clampVal(T v, T lo, T hi) { return (v < lo) ? lo : (v > hi) ? hi : v; }
};
