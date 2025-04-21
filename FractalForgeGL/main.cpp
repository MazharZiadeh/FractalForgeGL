#include "MandelbrotRenderer.h"
#include "NSGAII.h"
#include "Logger.h"
#include <iostream>

constexpr int   SCR_W = 1280, SCR_H = 720;
constexpr float TARGET_FPS = 60.0f;

int main() {
    // ─── init GLFW + GLAD ────────────────────────────────────────────────────
    if (!glfwInit()) { std::cerr << "GLFW failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* win = glfwCreateWindow(SCR_W, SCR_H, "Mandelbrot‑NSGA CSV", nullptr, nullptr);
    if (!win) { std::cerr << "Window failed\n"; return -1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD failed\n"; return -1;
    }

    // ─── components ──────────────────────────────────────────────────────────
    MandelbrotRenderer renderer(SCR_W, SCR_H);
    NSGAII             evo(32, 64, 4096);
    CSVLogger          log("run_log.csv");

    // ‑‑ basic camera state
    float cx = -0.5f, cy = 0.0f, zoom = 1.0f;
    float viewPack[3] = { cx,cy,zoom };
    glfwSetWindowUserPointer(win, viewPack);
    glfwSetKeyCallback(win, [](GLFWwindow* w, int k, int, int a, int) {
        if (a != GLFW_PRESS && a != GLFW_REPEAT) return;
        float* p = (float*)glfwGetWindowUserPointer(w);
        float& cx = *p, & cy = *(p + 1), & zoom = *(p + 2);
        float pan = 0.005f * zoom;
        switch (k) {
        case GLFW_KEY_UP:   cy += pan; break;
        case GLFW_KEY_DOWN: cy -= pan; break;
        case GLFW_KEY_LEFT: cx -= pan; break;
        case GLFW_KEY_RIGHT:cx += pan; break;
        case GLFW_KEY_Z:    zoom /= 1.1f; break;
        case GLFW_KEY_X:    zoom *= 1.1f; break;
        }
        });

    // ─── evolutionary loop ───────────────────────────────────────────────────
    int generation = 0, idxInGen = 0;
    while (!glfwWindowShouldClose(win)) {
        // update view from keys
        cx = viewPack[0]; cy = viewPack[1]; zoom = viewPack[2];

        // set genome + view
        renderer.setView(cx, cy, zoom);
        renderer.setMaxIter(evo.current().maxIter);

        // render off‑screen for fitness
        renderer.renderOffscreen();

        // collect metrics
        float fpsErr = std::abs(renderer.fps() - TARGET_FPS);
        float gpuMs = renderer.lastGpuTimeMs();

        // boundary complexity & variance on small FBO
        const unsigned char* pix = renderer.pixelPtr();
        int edges = 0; double sum = 0, sum2 = 0;
        for (int y = 1;y < MandelbrotRenderer::OFF_H;++y)
            for (int x = 1;x < MandelbrotRenderer::OFF_W;++x) {
                int i = y * MandelbrotRenderer::OFF_W + x;
                if (pix[i] != pix[i - 1] || pix[i] != pix[i - MandelbrotRenderer::OFF_W]) ++edges;
            }
        for (int i = 0;i < MandelbrotRenderer::OFF_W * MandelbrotRenderer::OFF_H;++i) {
            float v = pix[i] / 255.0f; sum += v; sum2 += v * v;
        }
        int N = MandelbrotRenderer::OFF_W * MandelbrotRenderer::OFF_H;
        float var = (sum2 / N) - float(sum / N) * float(sum / N);

        // push fitness & log eval
        evo.setFitness(fpsErr, gpuMs, (float)edges, var);
        log.row("EVAL", generation, idxInGen, evo.current().maxIter,
            fpsErr, gpuMs, edges, var, -1);

        // onscreen draw best so far
        renderer.setMaxIter(evo.best().maxIter);
        renderer.renderOnscreen();

        // GUI & swap
        glfwSwapBuffers(win);
        glfwPollEvents();

        // advance individual / generation
        if (evo.nextIndividual()) {
            // generation finished → log Pareto front
            evo.recalcRanks();
            const auto& pop = evo.population();
            for (size_t i = 0;i < pop.size();++i)
                if (pop[i].rank == 0)
                    log.row("FRONT", generation, (int)i, pop[i].maxIter,
                        pop[i].obj[0], pop[i].obj[1],
                        -pop[i].obj[2], -pop[i].obj[3],
                        pop[i].rank);

            evo.evolve();
            ++generation;
            idxInGen = 0;
        }
        else ++idxInGen;
    }
    glfwTerminate();
    std::cout << "Run complete.  CSV written to run_log.csv\n";
    return 0;
}
