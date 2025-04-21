#include "MandelbrotRenderer.h"
#include <iostream>

// ─── GLSL sources ────────────────────────────────────────────────────────────
static const char* VS = R"(#version 410 core
layout(location=0) in vec2 p; out vec2 uv;
void main(){ uv=p*0.5+0.5; gl_Position=vec4(p,0,1);} )";

static const char* FS = R"(#version 410 core
in vec2 uv; out vec4 frag;
uniform vec2  uCenter;
uniform float uZoom;
uniform vec2  uRes;
uniform int   uMaxIter;

void main(){
    vec2 c;
    c.x = (uv.x-0.5)*uZoom*(uRes.x/uRes.y)+uCenter.x;
    c.y = (uv.y-0.5)*uZoom+uCenter.y;

    vec2 z = vec2(0.0);
    int  i = 0;
    for(; i<uMaxIter && dot(z,z)<4.0; ++i)
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;

    float t  = float(i)/uMaxIter;
    frag = vec4(t, t*t, sqrt(t), 1);
} )";

// ─── helper helpers ──────────────────────────────────────────────────────────
static GLuint compile(GLenum tp, const char* src) {
    GLuint s = glCreateShader(tp);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512]; glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "Shader:" << log << "\n"; std::exit(1);
    }
    return s;
}
static GLuint link(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512]; glGetProgramInfoLog(p, 512, nullptr, log);
        std::cerr << "Link:" << log << "\n"; std::exit(1);
    }
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

// ─── ctor/dtor ───────────────────────────────────────────────────────────────
MandelbrotRenderer::MandelbrotRenderer(int winW, int winH) {
    initShader(); initQuad(); initFBO();
    glUseProgram(prog);
    glUniform2f(uRes, (float)winW, (float)winH);
}
MandelbrotRenderer::~MandelbrotRenderer() {
    glDeleteProgram(prog); glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo); glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex); glDeleteRenderbuffers(1, &rbo);
}

// ─── init helpers ────────────────────────────────────────────────────────────
void MandelbrotRenderer::initShader() {
    prog = link(compile(GL_VERTEX_SHADER, VS), compile(GL_FRAGMENT_SHADER, FS));
    uCenter = glGetUniformLocation(prog, "uCenter");
    uZoom = glGetUniformLocation(prog, "uZoom");
    uRes = glGetUniformLocation(prog, "uRes");
    uMaxIter = glGetUniformLocation(prog, "uMaxIter");
}
void MandelbrotRenderer::initQuad() {
    float tri[6] = { -1,-1, 3,-1, -1,3 };
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
}
void MandelbrotRenderer::initFBO() {
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, OFF_W, OFF_H, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glGenRenderbuffers(1, &rbo); glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, OFF_W, OFF_H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "FBO incomplete!\n";
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenQueries(1, &timerQuery);
    pixels.resize(OFF_W * OFF_H);
}

// ─── per‑frame interface ─────────────────────────────────────────────────────
void MandelbrotRenderer::setView(float cx, float cy, float zoom) {
    glUseProgram(prog);
    glUniform2f(uCenter, cx, cy);
    glUniform1f(uZoom, zoom);
}
void MandelbrotRenderer::setMaxIter(int it) {
    glUseProgram(prog);
    glUniform1i(uMaxIter, it);
}
void MandelbrotRenderer::renderOnscreen() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    int w, h; glfwGetFramebufferSize(glfwGetCurrentContext(), &w, &h);
    glViewport(0, 0, w, h);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindVertexArray(vao);
    glUseProgram(prog);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}
void MandelbrotRenderer::renderOffscreen() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, OFF_W, OFF_H);

    glBeginQuery(GL_TIME_ELAPSED, timerQuery);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndQuery(GL_TIME_ELAPSED);

    GLuint timeNS = 0; glGetQueryObjectuiv(timerQuery, GL_QUERY_RESULT, &timeNS);
    gpuTimeMs = timeNS * 1e-6f;

    glReadPixels(0, 0, OFF_W, OFF_H, GL_RED, GL_UNSIGNED_BYTE, pixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
