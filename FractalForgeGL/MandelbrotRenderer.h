#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>

class MandelbrotRenderer {
public:
    MandelbrotRenderer(int winW, int winH);
    ~MandelbrotRenderer();

    void setView(float cx, float cy, float zoom);
    void setMaxIter(int it);

    void renderOnscreen();   // draw best individual to screen
    void renderOffscreen();  // draw to 256×256 FBO for fitness

    float  lastGpuTimeMs() const { return gpuTimeMs; }
    float  fps()           const { return 1000.0f / gpuTimeMs; }
    const unsigned char* pixelPtr() const { return pixels.data(); }

    static constexpr int OFF_W = 256;
    static constexpr int OFF_H = 256;

private:
    GLuint prog, vao, vbo;
    GLuint fbo, tex, rbo, timerQuery;
    GLint  uCenter, uZoom, uRes, uMaxIter;

    float gpuTimeMs = 0.0f;
    std::vector<unsigned char> pixels;

    void initShader();
    void initQuad();
    void initFBO();
};
