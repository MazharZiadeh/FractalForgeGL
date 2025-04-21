#include "NSGAII.h"
#include <algorithm>
#include <cmath>

// ─── dominance test ──────────────────────────────────────────────────────────
bool NSGAII::dominates(const Individual& a, const Individual& b)
{
    bool better = false;
    for (int k = 0; k < 4; ++k) {
        float av = a.obj[k], bv = b.obj[k];
        if (k >= 2) { av = -av;  bv = -bv; }      // maximise -> minimise negative

        if (av > bv) return false;                // worse in at least one
        if (av < bv) better = true;               // strictly better somewhere
    }
    return better;
}

// ─── ctor ────────────────────────────────────────────────────────────────────
NSGAII::NSGAII(int population, int mn, int mx)
    : popSize(population), minIter(mn), maxIter(mx),
    pop(population), offspring(population)
{
    std::uniform_int_distribution<int> dist(minIter, maxIter);
    for (auto& ind : pop) ind.maxIter = dist(rng);
}

// ─── getters / setters ───────────────────────────────────────────────────────
Individual& NSGAII::current() { return pop[evalIndex]; }
const Individual& NSGAII::best() const {
    return *std::min_element(pop.begin(), pop.end(),
        [](const Individual& a, const Individual& b) {
            return a.obj[0] < b.obj[0]; });
}
bool NSGAII::nextIndividual()
{
    ++evalIndex;
    if (evalIndex == popSize) { evalIndex = 0; return true; }
    return false;
}
void NSGAII::setFitness(float fpsErr, float gpuMs,
    float boundary, float density)
{
    pop[evalIndex].obj[0] = fpsErr;
    pop[evalIndex].obj[1] = gpuMs;
    pop[evalIndex].obj[2] = -boundary;
    pop[evalIndex].obj[3] = -density;
}

// ─── ranking / crowding ──────────────────────────────────────────────────────
void NSGAII::calcCrowding(std::vector<Individual*>& F)
{
    const int M = 4, s = (int)F.size();
    for (auto* p : F) p->crowd = 0.0f;

    for (int m = 0; m < M; ++m) {
        std::sort(F.begin(), F.end(),
            [m](Individual* a, Individual* b) { return a->obj[m] < b->obj[m]; });

        F.front()->crowd = F.back()->crowd = INFINITY;
        float minv = F.front()->obj[m], maxv = F.back()->obj[m];
        if (maxv - minv == 0) continue;

        for (int i = 1; i < s - 1; ++i)
            F[i]->crowd += (F[i + 1]->obj[m] - F[i - 1]->obj[m]) / (maxv - minv);
    }
}

void NSGAII::assignRanks()
{
    std::vector<std::vector<int>> fronts;
    std::vector<int> dominated(popSize, 0);
    std::vector<std::vector<int>> domList(popSize);

    for (int p = 0; p < popSize; ++p)
        for (int q = 0; q < popSize; ++q) if (p != q) {
            if (dominates(pop[p], pop[q])) domList[p].push_back(q);
            else if (dominates(pop[q], pop[p])) ++dominated[p];
        }

    for (int i = 0; i < popSize; ++i) if (!dominated[i]) {
        pop[i].rank = 0;
        if (fronts.empty()) fronts.emplace_back();
        fronts[0].push_back(i);
    }

    int i = 0;
    while (i < (int)fronts.size()) {
        std::vector<int> next;
        for (int p : fronts[i])
            for (int q : domList[p])
                if (--dominated[q] == 0) {
                    pop[q].rank = i + 1;
                    next.push_back(q);
                }
        if (!next.empty()) fronts.push_back(std::move(next));
        ++i;
    }
    for (auto& f : fronts) {
        std::vector<Individual*> ptrs;
        for (int idx : f) ptrs.push_back(&pop[idx]);
        calcCrowding(ptrs);
    }
}
void NSGAII::recalcRanks() { assignRanks(); }

// ─── selection & variation ───────────────────────────────────────────────────
Individual NSGAII::tournament()
{
    std::uniform_int_distribution<int> pick(0, popSize - 1);
    Individual& A = pop[pick(rng)], & B = pop[pick(rng)];
    if (A.rank < B.rank) return A;
    if (B.rank < A.rank) return B;
    return (A.crowd > B.crowd) ? A : B;
}

void NSGAII::evolve()
{
    assignRanks();
    std::uniform_real_distribution<float> pb(0, 1);
    std::uniform_int_distribution<int>    delta(-128, 128);

    for (int i = 0; i < popSize; ++i) {
        Individual child = tournament();
        if (pb(rng) < 0.8f)
            child.maxIter = clampVal(child.maxIter + delta(rng), minIter, maxIter);
        offspring[i] = std::move(child);
    }
    pop.swap(offspring);
}
