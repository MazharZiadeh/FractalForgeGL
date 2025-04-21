// ─────────────────────────────────────────────────────────────────────────────
//  Mandelbrot + NSGA‑II  — all‑tweakables‑up‑front version
//  Build (Linux/macOS):
//      g++ -std=c++17 -O2 main.cpp MandelbrotRenderer.cpp NSGAII.cpp glad.c \
//          -lglfw -ldl -lGL -pthread -o MandelbrotNSGA
//  Build (MSVC):
//      cl /std:c++17 /O2 main.cpp MandelbrotRenderer.cpp NSGAII.cpp glad.c\
//          glfw3.lib opengl32.lib user32.lib gdi32.lib shell32.lib
// ─────────────────────────────────────────────────────────────────────────────
#include "MandelbrotRenderer.h"
#include "NSGAII.h"
#include "Logger.h"
#include <iostream>
#include <cmath>

// ─────────── 1. ALL PARAMETERS IN ONE PLACE ─────────────────────────────────
namespace CFG {
    // Window / view
    constexpr int   winW = 1280;
    constexpr int   winH = 720;
    constexpr float panSpeed = 0.004f;     // relative to zoom
    constexpr float zoomFactor = 1.07f;

    // Evolution
    constexpr int   popSize = 48;
    constexpr int   minIterLOD = 128;
    constexpr int   maxIterLOD = 20000;
    constexpr float mutateProb = 0.9f;       // (inside NSGAII::evolve)
    constexpr int   mutateDelta = 256;

    // Performance target
    constexpr float targetFPS = 60.0f;

    // Off‑screen fitness buffer (increase to raise GPU cost & metric fidelity)
    constexpr int   evalW = 1024;
    constexpr int   evalH = 1024;

    // CSV output
    constexpr const char* csvFile = "run_log.csv";
}
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    // ─── GLFW / GLAD init ───────────────────────────────────────────────────
    if (!glfwInit()) { std::cerr << "GLFW failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* win = glfwCreateWindow(CFG::winW, CFG::winH,
        "Mandelbrot‑NSGA (CFG edition)", nullptr, nullptr);
    if (!win) { std::cerr << "Window failed\n"; return -1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD failed\n"; return -1;
    }

    // ─── components ─────────────────────────────────────────────────────────
    MandelbrotRenderer renderer(CFG::winW, CFG::winH);          // off‑screen size set inside
    NSGAII             evo(CFG::popSize, CFG::minIterLOD, CFG::maxIterLOD);
    CSVLogger          log(CFG::csvFile);

    // Camera state
    float cx = -0.5f, cy = 0.0f, zoom = 1.0f;
    float view[3] = { cx, cy, zoom };
    glfwSetWindowUserPointer(win, view);
    glfwSetKeyCallback(win, [](GLFWwindow* w, int key, int, int act, int) {
        if (act != GLFW_PRESS && act != GLFW_REPEAT) return;
        auto* v = static_cast<float*>(glfwGetWindowUserPointer(w));
        float& cx = v[0], & cy = v[1], & z = v[2];
        float pan = CFG::panSpeed * z;
        switch (key) {
        case GLFW_KEY_UP: cy += pan; break;
        case GLFW_KEY_DOWN: cy -= pan; break;
        case GLFW_KEY_LEFT: cx -= pan; break;
        case GLFW_KEY_RIGHT: cx += pan; break;
        case GLFW_KEY_Z: z /= CFG::zoomFactor; break;
        case GLFW_KEY_X: z *= CFG::zoomFactor; break;
        }
        });

    // ─── evolutionary loop ─────────────────────────────────────────────────
    int gen = 0, idx = 0;
    while (!glfwWindowShouldClose(win)) {
        // fetch updated view
        cx = view[0]; cy = view[1]; zoom = view[2];

        // set genome + view
        renderer.setView(cx, cy, zoom);
        renderer.setMaxIter(evo.current().maxIter);
        renderer.renderOffscreen();

        // ----- fitness metrics -----
        float fpsErr = std::abs(renderer.fps() - CFG::targetFPS);
        float gpuMs = renderer.lastGpuTimeMs();

        const unsigned char* px = renderer.pixelPtr();
        int edges = 0; double sum = 0, sum2 = 0;
        for (int y = 1; y < MandelbrotRenderer::OFF_H; ++y)
            for (int x = 1; x < MandelbrotRenderer::OFF_W; ++x) {
                int i = y * MandelbrotRenderer::OFF_W + x;
                if (px[i] != px[i - 1] || px[i] != px[i - MandelbrotRenderer::OFF_W]) ++edges;
            }
        for (int i = 0; i < MandelbrotRenderer::OFF_W * MandelbrotRenderer::OFF_H; ++i) {
            float v = px[i] / 255.f; sum += v; sum2 += v * v;
        }
        int N = MandelbrotRenderer::OFF_W * MandelbrotRenderer::OFF_H;
        float var = (sum2 / N) - float(sum / N) * float(sum / N);

        evo.setFitness(fpsErr, gpuMs, float(edges), var);
        log.row("EVAL", gen, idx, evo.current().maxIter,
            fpsErr, gpuMs, edges, var, -1);

        // draw best individual onscreen
        renderer.setMaxIter(evo.best().maxIter);
        renderer.renderOnscreen();

        glfwSwapBuffers(win);
        glfwPollEvents();

        // move to next individual / generation
        if (evo.nextIndividual()) {
            evo.recalcRanks();
            for (size_t i = 0; i < evo.population().size(); ++i)
                if (evo.population()[i].rank == 0)
                    log.row("FRONT", gen, static_cast<int>(i), evo.population()[i].maxIter,
                        evo.population()[i].obj[0], evo.population()[i].obj[1],
                        -evo.population()[i].obj[2], -evo.population()[i].obj[3],
                        evo.population()[i].rank);
            evo.evolve();
            ++gen; idx = 0;
        }
        else ++idx;
    }
    glfwTerminate();
    std::cout << "Run complete. CSV written to " << CFG::csvFile << "\n";
    return 0;
}
