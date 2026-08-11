#include "fluid_renderer.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <glm/gtc/type_ptr.hpp>

#include "glsl_program.h"

namespace {

static_assert(std::is_standard_layout_v<Vec3>,
              "Vec3 must have a standard memory layout");
static_assert(sizeof(Vec3) == 3 * sizeof(float),
              "Vec3 must contain exactly three packed floats");

constexpr char kImposterVertexShader[] =
    // language=GLSL
        R"glsl(#version 460 core

layout(location = 0) in vec3 aPos;

uniform mat4  uView;
uniform mat4  uProjection;
uniform float uRadius;
uniform float uViewportH;   // 帧缓冲高度

out vec3 vViewPos;

void main() {
    vec4 view = uView * vec4(aPos, 1.0);
    vViewPos = view.xyz;

    gl_Position = uProjection * view;

    gl_PointSize = uViewportH * uProjection[1][1] * uRadius / -view.z;
}
)glsl";

// 深度平滑、厚度平滑和表面着色共用
constexpr char kFullscreenVertexShader[] =
    // language=GLSL
        R"glsl(#version 460 core
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)glsl";

} // namespace

FluidRenderer::FluidRenderer(std::size_t particleCount)
{
    initDepthPass();
    initThicknessPass();
    initBlurPass();
    initSurfacePass();

    glCreateVertexArrays(1, &depthVao_);
    glCreateVertexArrays(1, &emptyVao_);
    glCreateBuffers(1, &depthVbo_);

    if (depthVao_ == 0 || emptyVao_ == 0 || depthVbo_ == 0) {
        shutdown();
        throw std::runtime_error("Failed to create the particle VAO or VBO");
    }

    particleCapacity_ = particleCount > 0 ? particleCount : 1;

    glNamedBufferData(
        depthVbo_,
        static_cast<GLsizeiptr>(particleCapacity_ * sizeof(Vec3)),
        nullptr,
        GL_DYNAMIC_DRAW
    );

    // 绑定点 0 每个顶点读一个紧凑排列的 Vec3。
    glVertexArrayVertexBuffer(
        depthVao_,
        0,
        depthVbo_,
        0,
        static_cast<GLsizei>(sizeof(Vec3))
    );
    glEnableVertexArrayAttrib(depthVao_, 0);
    glVertexArrayAttribFormat(depthVao_, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(depthVao_, 0, 0);
}

FluidRenderer::~FluidRenderer()
{
    shutdown();
}

void FluidRenderer::ensureTarget(int width, int height)
{
    if (width == targetWidth_ && height == targetHeight_) {
        return;
    }

    // glTextureStorage2D 创建的是不可变纹理，所以尺寸变化只能删掉重建，
    // 不能重新分配。FBO 本身可以复用，只换附件。
    if (depthField_ != 0)  { glDeleteTextures(1, &depthField_); depthField_ = 0; }
    if (blurField_ != 0)   { glDeleteTextures(1, &blurField_); blurField_ = 0; }
    if (thicknessField_ != 0) { glDeleteTextures(1, &thicknessField_); thicknessField_ = 0; }
    if (thicknessFbo_ == 0)   { glCreateFramebuffers(1, &thicknessFbo_); }
    if (depthBuffer_ != 0) { glDeleteRenderbuffers(1, &depthBuffer_); depthBuffer_ = 0; }
    if (fbo_ == 0)         { glCreateFramebuffers(1, &fbo_); }
    if (blurFbo_ == 0)     { glCreateFramebuffers(1, &blurFbo_); }

    targetWidth_  = width;
    targetHeight_ = height;

    glCreateTextures(GL_TEXTURE_2D, 1, &depthField_);
    glTextureStorage2D(depthField_, 1, GL_R32F, width, height);
    // NEAREST：后面的 pass 只在精确的 texel 中心采样，插值不但没有意义，
    // 还会跨轮廓把前后两团流体的深度混在一起。
    // CLAMP_TO_EDGE：滤波会采到图像外面，包裹模式会把屏幕另一边折进表面。
    glTextureParameteri(depthField_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(depthField_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(depthField_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(depthField_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // 平滑的中转纹理，格式和采样方式必须与 depthField_ 完全一致——横向那一遍的
    // 输出就是竖向那一遍的输入，两者精度不同会让第二遍在第一遍的量化台阶上工作。
    glCreateTextures(GL_TEXTURE_2D, 1, &blurField_);
    glTextureStorage2D(blurField_, 1, GL_R32F, width, height);
    glTextureParameteri(blurField_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(blurField_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(blurField_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(blurField_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glCreateRenderbuffers(1, &depthBuffer_);
    glNamedRenderbufferStorage(depthBuffer_, GL_DEPTH_COMPONENT24, width, height);

    glNamedFramebufferTexture(fbo_, GL_COLOR_ATTACHMENT0, depthField_, 0);
    glNamedFramebufferRenderbuffer(fbo_, GL_DEPTH_ATTACHMENT,
                                   GL_RENDERBUFFER, depthBuffer_);

    // 厚度场。同样不需要深度附件——这个 pass 的深度测试本来就是关掉的，那正是
    // 它能把前表面后面的球也累加进来的原因。
    glCreateTextures(GL_TEXTURE_2D, 1, &thicknessField_);
    glTextureStorage2D(thicknessField_, 1, GL_R16F, width, height);
    glTextureParameteri(thicknessField_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(thicknessField_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(thicknessField_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(thicknessField_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glNamedFramebufferTexture(thicknessFbo_, GL_COLOR_ATTACHMENT0, thicknessField_, 0);

    if (glCheckNamedFramebufferStatus(thicknessFbo_, GL_FRAMEBUFFER)
            != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Fluid thickness framebuffer is incomplete");
    }

    // blurFbo_ 不需要深度附件：平滑是全屏 pass，深度测试是关掉的。
    glNamedFramebufferTexture(blurFbo_, GL_COLOR_ATTACHMENT0, blurField_, 0);

    if (glCheckNamedFramebufferStatus(blurFbo_, GL_FRAMEBUFFER)
            != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Fluid blur framebuffer is incomplete");
    }

    const GLenum status = glCheckNamedFramebufferStatus(fbo_, GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Fluid framebuffer is incomplete: 0x"
                                 + std::to_string(status));
    }
}

void FluidRenderer::render(
    const std::vector<Vec3>& positions,
    const glm::mat4& view,
    const glm::mat4& projection,
    float radius,
    int viewportWidth,
    int viewportHeight
)
{
    if (positions.empty() || viewportWidth <= 0 || viewportHeight <= 0) {
        return;
    }

    ensureTarget(viewportWidth, viewportHeight);
    uploadPositions(positions);

    const GLsizei count = static_cast<GLsizei>(positions.size());

    drawDepthField(view, projection, radius, viewportHeight, count);
    drawThicknessField(view, projection, radius, viewportHeight, count);
    blurDepthField(projection, radius, viewportWidth, viewportHeight);
    // 顺序不能换：厚度平滑的半径和空白判断都读 depthField_，要的是上一行平滑
    // 之后的版本；而且它借用 blurField_ 当中转，那张纹理得等深度平滑用完。
    blurThicknessField(projection, radius, viewportWidth, viewportHeight);
    shadeSurface(projection, radius);

    // 把 GL 状态恢复成 main.cpp 在初始化时设定的那一套。上面的 pass 改过
    // 深度测试、深度写和混合，不恢复的话下一帧的地板 pass 会用错的状态画。
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

// 把粒子位置搬到 VBO。
void FluidRenderer::uploadPositions(const std::vector<Vec3>& positions)
{
    // 缓冲区只增不减。glNamedBufferData 会丢弃旧内容，但下面反正整体覆写。
    if (positions.size() > particleCapacity_) {
        particleCapacity_ = positions.size();
        glNamedBufferData(
            depthVbo_,
            static_cast<GLsizeiptr>(particleCapacity_ * sizeof(Vec3)),
            nullptr,
            GL_DYNAMIC_DRAW
        );
    }

    // Vec3 是三个紧凑排列的 float（文件开头两个 static_assert 保证了这点），
    // 所以 std::vector<Vec3> 的内存能整块拷进 VBO，不需要逐个转换。
    glNamedBufferSubData(
        depthVbo_,
        0,
        static_cast<GLsizeiptr>(positions.size() * sizeof(Vec3)),
        positions.data()
    );
}

// 深度场
void FluidRenderer::initDepthPass()
{
    constexpr char kFragmentShader[] =
        // language=GLSL
            R"glsl(#version 460 core
in vec3 vViewPos;

uniform mat4  uProjection;
uniform float uRadius;

layout(location = 0) out float fragDistance;

void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    c.y = -c.y;                          // gl_PointCoord 的 y 朝下
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;

    vec3 n = vec3(c, sqrt(1.0 - r2));           // 视空间球面法线
    vec3 spherePos = vViewPos + n * uRadius;    // 这个片元真正的 3D 位置

    // 真实球面深度
    vec4 clip = uProjection * vec4(spherePos, 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;

    fragDistance = -spherePos.z;
}
)glsl";

    depthProgram_ = buildGlslProgram(
        kImposterVertexShader,
        kFragmentShader,
        "fluid depth field"
    );
    if (depthProgram_ == 0) {
        throw std::runtime_error("Failed to create the fluid depth program");
    }

    viewLocation_       = glGetUniformLocation(depthProgram_, "uView");
    projectionLocation_ = glGetUniformLocation(depthProgram_, "uProjection");
    radiusLocation_     = glGetUniformLocation(depthProgram_, "uRadius");
    viewportHLocation_  = glGetUniformLocation(depthProgram_, "uViewportH");
    if (viewLocation_ == -1 || projectionLocation_ == -1 ||
        radiusLocation_ == -1 || viewportHLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find a required depth shader uniform");
    }
}

// 每个像素记录最靠前的那个球面的视空间距离，没有流体的地方是 0。
void FluidRenderer::drawDepthField(
    const glm::mat4& view,
    const glm::mat4& projection,
    float radius,
    int viewportHeight,
    GLsizei count
)
{
    glProgramUniformMatrix4fv(depthProgram_, viewLocation_, 1, GL_FALSE,
                              glm::value_ptr(view));
    glProgramUniformMatrix4fv(depthProgram_, projectionLocation_, 1, GL_FALSE,
                              glm::value_ptr(projection));
    glProgramUniform1f(depthProgram_, radiusLocation_, radius);
    glProgramUniform1f(depthProgram_, viewportHLocation_,
                       static_cast<float>(viewportHeight));

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);

    // 深度测试 + 深度写都要开：只保留最靠前的球面，这才叫"深度场"。
    // 混合必须关：混合会把前后两个球的距离按 alpha 平均掉，得到的距离不属于
    // 任何一个真实表面，后面重建出来的法线也就没有意义。
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    // 清成 0。0 是"这里没有流体"的约定，后面每个 pass 都靠它判断空白区域。
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 顶点着色器写的 gl_PointSize 只有在 GL_PROGRAM_POINT_SIZE 打开时才生效，
    // 这一项是 main.cpp 全局打开的，全程没人关，所以这里不重复设置。
    glUseProgram(depthProgram_);
    glBindVertexArray(depthVao_);
    glDrawArrays(GL_POINTS, 0, count);
    glBindVertexArray(0);
    glUseProgram(0);
}

// 厚度场
void FluidRenderer::initThicknessPass()
{
    constexpr char kFragmentShader[] =
        // language=GLSL
            R"glsl(#version 460 core

uniform float uRadius;

layout(location = 0) out float fragThickness;

void main() {
    vec2  c  = gl_PointCoord * 2.0 - 1.0;
    float c2 = dot(c, c);
    if (c2 > 1.0) discard;

    // 视线穿过一颗球的弦长  L = 2·r·sqrt(1 - |c|²)
    fragThickness = 2.0 * uRadius * sqrt(1.0 - c2);
}
)glsl";

    thicknessProgram_ = buildGlslProgram(
        kImposterVertexShader,
        kFragmentShader,
        "fluid thickness field"
    );
    if (thicknessProgram_ == 0) {
        shutdown();
        throw std::runtime_error("Failed to create the fluid thickness program");
    }

    thickViewLocation_       = glGetUniformLocation(thicknessProgram_, "uView");
    thickProjectionLocation_ = glGetUniformLocation(thicknessProgram_, "uProjection");
    thickRadiusLocation_     = glGetUniformLocation(thicknessProgram_, "uRadius");
    thickViewportHLocation_  = glGetUniformLocation(thicknessProgram_, "uViewportH");
    if (thickViewLocation_ == -1 || thickProjectionLocation_ == -1 ||
        thickRadiusLocation_ == -1 || thickViewportHLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find a required thickness shader uniform");
    }
}

// 视线穿过水体的总长度，米，喂给 Beer-Lambert 当 d。
void FluidRenderer::drawThicknessField(
    const glm::mat4& view,
    const glm::mat4& projection,
    float radius,
    int viewportHeight,
    GLsizei count
)
{
    // 顶点着色器和深度场那一遍是同一份源码，但 uniform 位置属于 program 对象，
    // 每个 program 各有一份，所以要单独再传一次。
    glProgramUniformMatrix4fv(thicknessProgram_, thickViewLocation_, 1, GL_FALSE,
                              glm::value_ptr(view));
    glProgramUniformMatrix4fv(thicknessProgram_, thickProjectionLocation_, 1, GL_FALSE,
                              glm::value_ptr(projection));
    glProgramUniform1f(thicknessProgram_, thickRadiusLocation_, radius);
    glProgramUniform1f(thicknessProgram_, thickViewportHLocation_,
                       static_cast<float>(viewportHeight));

    glBindFramebuffer(GL_FRAMEBUFFER, thicknessFbo_);

    // 深度测试关：故意让前表面后面的球也画上去，累加起来才是总路径长度。
    // 加性混合：GL_ONE, GL_ONE 就是 dst = dst + src，于是每颗球的弦长相加。
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    // 清成 0 是加性混合的前提，否则累加会从上一帧的残留继续往上加。
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(thicknessProgram_);
    glBindVertexArray(depthVao_);
    glDrawArrays(GL_POINTS, 0, count);
    glBindVertexArray(0);
    glUseProgram(0);
}

// 深度平滑，同时也是厚度平滑用的那个 program
void FluidRenderer::initBlurPass()
{
    // 可分离高斯，横竖各一遍。深度场和厚度场用同一份代码，只换输入纹理。
    //
    // 权重里**没有**任何依赖 z 的项，这是刻意的。二维高斯之所以能拆成两个一维
    // 高斯，前提是 G(x,y) = G(x)·G(y)，即权重只由偏移决定。一旦权重里掺进
    // 逐像素的深度（双边的范围项、窄范围滤波的丢弃/夹取），可分离性就不成立，
    // 横一遍的误差会被竖一遍沿另一个方向拉长，屏幕上表现为轴向的竖条纹。
    //
    // 唯一保留的深度相关判断是 `zi <= 0`（空白），它是二值的、且轮廓两侧一致，
    // 不引入方向性偏置。
    constexpr char kFragmentShader[] =
        // language=GLSL
            R"glsl(#version 460 core

uniform sampler2D uSource;        // 要平滑的场：深度场或厚度场
uniform sampler2D uDepth;         // 深度场；平滑深度时它和 uSource 是同一张纹理
uniform vec2      uDirection;     // 一个 texel 的步长：(1/w,0) 或 (0,1/h)
uniform float     uRadiusScale;   // 粒子在 z=1m 处的像素半径
uniform float     uBlurScale;     // 滤波半径 = 几倍粒子半径

layout(location = 0) out float fragValue;

// 循环上界必须是编译期常量，实际半径靠 break 截断。
const int kMaxRadius = 32;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(uSource, 0));
    vec2 uv    = gl_FragCoord.xy * texel;

    // 滤波半径和"这里有没有流体"这两件事都只看深度：厚度是一个沿视线的长度，
    // 没有"距摄像机多远"的含义，定不出以米为单位的窗口，也算不出像素半径。
    float z0 = texture(uDepth, uv).r;
    if (z0 <= 0.0) {
        fragValue = 0.0;
        return;
    }

    float v0 = texture(uSource, uv).r;   // 中心的待平滑值

    // R = uBlurScale · uRadiusScale / z，最大不超过 kMaxRadius。
    int R = int(min(uBlurScale * uRadiusScale / z0, float(kMaxRadius)));
    if (R < 1) {
        fragValue = v0;
        return;
    }

    // 高斯权重  w(i) = exp(-i² / (2σ²))，取 σ = R/2。
    // 最外圈 i=R 处 w = exp(-R²/(2·(R/2)²)) = exp(-2) ≈ 0.135，截断不可见。
    float sigma  = float(R) * 0.5;
    float inv2s2 = 1.0 / (2.0 * sigma * sigma);

    float sum     = v0;      // i=0 的中心样本，w(0) = exp(0) = 1
    float weights = 1.0;

    for (int i = 1; i <= kMaxRadius; ++i) {
        if (i > R) break;

        float w = exp(-float(i * i) * inv2s2);

        for (int side = -1; side <= 1; side += 2) {   // 对称的两侧
            vec2  suv = uv + uDirection * float(i * side);
            float zi  = texture(uDepth, suv).r;

            // 空白区域一律不参与平均。少了这一句，轮廓附近会把外面的 0 混进来，
            // 深度被拉远、厚度被拉薄，边缘出现一圈过度透明的晕。
            if (zi <= 0.0) continue;

            sum     += w * texture(uSource, suv).r;
            weights += w;
        }
    }

    fragValue = sum / weights;
}
)glsl";

    blurProgram_ = buildGlslProgram(
        kFullscreenVertexShader,
        kFragmentShader,
        "fluid depth smoothing"
    );
    if (blurProgram_ == 0) {
        shutdown();
        throw std::runtime_error("Failed to create the depth smoothing program");
    }

    blurDirectionLocation_   = glGetUniformLocation(blurProgram_, "uDirection");
    blurRadiusScaleLocation_ = glGetUniformLocation(blurProgram_, "uRadiusScale");
    blurScaleLocation_       = glGetUniformLocation(blurProgram_, "uBlurScale");
    if (blurDirectionLocation_ == -1 || blurRadiusScaleLocation_ == -1 ||
        blurScaleLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find a required blur shader uniform");
    }

    // sampler uniform 存的是"纹理单元编号"，不是纹理本身。这里定死 uSource 去
    // 0 号单元、uDepth 去 1 号单元取数据，之后每帧只需要 glBindTextureUnit 换纹理。
    glProgramUniform1i(blurProgram_, glGetUniformLocation(blurProgram_, "uSource"), 0);
    glProgramUniform1i(blurProgram_, glGetUniformLocation(blurProgram_, "uDepth"), 1);
}

// 深度平滑，横竖各一遍。
void FluidRenderer::blurDepthField(
    const glm::mat4& projection,
    float radius,
    int viewportWidth,
    int viewportHeight
)
{
    // 全屏 pass，不需要深度测试；深度写也关掉，免得往 fbo_ 的深度 renderbuffer
    // 里写进无意义的 0.5（全屏三角形的 z 是写死的）。
    // 混合关：这一遍的输出要原样覆盖目标纹理，不是和里面的旧值混。
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_BLEND);

    // uRadiusScale = viewportH · proj[1][1] · r / 2
    // 也就是这颗粒子在 z = 1 m 处占的像素半径。shader 里再除以逐像素的 z，
    // 于是远处的流体用小窗口、近处用大窗口，滤波强度在视觉上保持一致。
    const float radiusScale =
        static_cast<float>(viewportHeight) * projection[1][1] * radius * 0.5f;

    glProgramUniform1f(blurProgram_, blurRadiusScaleLocation_, radiusScale);
    glProgramUniform1f(blurProgram_, blurScaleLocation_, blurScale);

    glUseProgram(blurProgram_);
    glBindVertexArray(emptyVao_);

    // 平滑深度时 uSource 和 uDepth 是同一张纹理：待平滑的值本身就是深度。
    // 第二遍读的是第一遍的输出，所以 uDepth 也要跟着换成 blurField_，否则半径
    // 和空白判断会用未平滑的深度，两遍的支撑域对不上。

    // 横向：depthField_ -> blurField_
    glBindFramebuffer(GL_FRAMEBUFFER, blurFbo_);
    glBindTextureUnit(0, depthField_);
    glBindTextureUnit(1, depthField_);
    glProgramUniform2f(blurProgram_, blurDirectionLocation_,
                       1.0f / static_cast<float>(viewportWidth), 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // 竖向：blurField_ -> depthField_。后面的厚度平滑和表面着色都约定最终深度
    // 位于 depthField_，写回这里之后 blurField_ 就空出来给厚度当中转。
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glBindTextureUnit(0, blurField_);
    glBindTextureUnit(1, blurField_);
    glProgramUniform2f(blurProgram_, blurDirectionLocation_,
                       0.0f, 1.0f / static_cast<float>(viewportHeight));
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexArray(0);
    glUseProgram(0);
}

// 厚度平滑，横竖各一遍。和深度平滑同一个 program，只是换了输入纹理。
void FluidRenderer::blurThicknessField(
    const glm::mat4& projection,
    float radius,
    int viewportWidth,
    int viewportHeight
)
{
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_BLEND);

    const float radiusScale =
        static_cast<float>(viewportHeight) * projection[1][1] * radius * 0.5f;

    glProgramUniform1f(blurProgram_, blurRadiusScaleLocation_, radiusScale);
    glProgramUniform1f(blurProgram_, blurScaleLocation_, blurScale);

    glUseProgram(blurProgram_);
    glBindVertexArray(emptyVao_);

    // uDepth 全程绑已经平滑好的 depthField_：半径和空白判断都靠它，而这两件事
    // 都要用最终的深度，这就是本函数必须排在 blurDepthField 之后的原因。
    glBindTextureUnit(1, depthField_);

    // 横向：读 thicknessField_ -> 写 blurField_
    // 借用深度平滑的中转纹理。它是 R32F，装 R16F 的厚度富富有余；深度平滑跑完
    // 结果已经回到 depthField_，这张纹理就空出来了。
    glBindFramebuffer(GL_FRAMEBUFFER, blurFbo_);
    glBindTextureUnit(0, thicknessField_);
    glProgramUniform2f(blurProgram_, blurDirectionLocation_,
                       1.0f / static_cast<float>(viewportWidth), 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // 竖向：读 blurField_ -> 写回 thicknessField_
    glBindFramebuffer(GL_FRAMEBUFFER, thicknessFbo_);
    glBindTextureUnit(0, blurField_);
    glProgramUniform2f(blurProgram_, blurDirectionLocation_,
                       0.0f, 1.0f / static_cast<float>(viewportHeight));
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexArray(0);
    glUseProgram(0);
}

// 表面着色
void FluidRenderer::initSurfacePass()
{
    // 从深度场重建表面法线，然后打光
    constexpr char kFragmentShader[] =
        // language=GLSL
            R"glsl(#version 460 core

uniform sampler2D uDepthField;   // R32F，视空间距离，0 表示这里没有流体
uniform sampler2D uThickness;    // R16F，视线穿过的水的总长度，米
uniform vec2      uProjXY;       // (proj[0][0], proj[1][1])，见 viewFromUv
uniform vec3      uAbsorb;       // Beer-Lambert 吸收系数 σ，1/m，分通道
uniform float     uNormalDepthThreshold; // 中央差分允许跨过的最大深度差，米

out vec4 FragColor;

float depthAt(vec2 uv) {
    return texture(uDepthField, uv).r;
}

// ndc.x = proj[0][0] * view.x / (-view.z)
// view.x = ndc.x * z / proj[0][0]
vec3 viewFromUv(vec2 uv, float z) {
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(ndc.x * z / uProjXY.x,
                ndc.y * z / uProjXY.y,
                -z);
}

vec3 tangent(vec2 uv, vec2 step, vec3 here, float z0) {
    float zp = depthAt(uv + step);
    float zm = depthAt(uv - step);

    if (zp <= 0.0 && zm <= 0.0) return vec3(0.0);
    if (zm <= 0.0) return viewFromUv(uv + step, zp) - here;
    if (zp <= 0.0) return here - viewFromUv(uv - step, zm);

    float deltaPlus  = abs(zp - z0);
    float deltaMinus = abs(zm - z0);
    float span       = abs(zp - zm);

    // 左右（或上下）仍属于同一层连续水面时，优先用中央差分：它同时利用两侧，
    // 一阶误差相消，对残留的一像素深度噪声比单侧差分稳定得多。
    // 只有两侧真的跨过深度断层时，才退回下面的单侧差分。
    if (span <= uNormalDepthThreshold) {
        return 0.5 * (viewFromUv(uv + step, zp)
                    - viewFromUv(uv - step, zm));
    }

    return deltaPlus < deltaMinus
        ? viewFromUv(uv + step, zp) - here
        : here - viewFromUv(uv - step, zm);
}

void main() {
    vec2 texel = 1.0 / vec2(textureSize(uDepthField, 0));
    vec2 uv    = gl_FragCoord.xy * texel;

    float z = depthAt(uv);
    if (z <= 0.0) discard;

    vec3 here = viewFromUv(uv, z);

    vec3 tx = tangent(uv, vec2(texel.x, 0.0), here, z);
    vec3 ty = tangent(uv, vec2(0.0, texel.y), here, z);

    vec3 n = cross(tx, ty);
    n = dot(n, n) > 1e-18 ? normalize(n) : vec3(0.0, 0.0, 1.0);
    if (n.z < 0.0) n = -n;

    // Beer-Lambert：光穿过吸收介质，强度按路径长度指数衰减
    //     I(d) = I₀ · exp(-σ · d)
    // σ 分通道，水对红光的吸收远强于蓝光——**这才是水呈蓝绿色的原因**，不是因为
    // 它反射蓝光。所以这里不再乘任何人为的蓝色调，蓝绿是算出来的：薄处
    // exp(0) = 1 出射接近白色，厚处红光先被吃掉，剩下青蓝。
    float thickness = texture(uThickness, uv).r;
    vec3  transmit  = exp(-uAbsorb * thickness);

    vec3  L    = normalize(vec3(0.5, 0.8, 0.6));
    float diff = max(dot(n, L), 0.0);

    float alpha = 1.0 - exp(-8 * thickness);
    FragColor  = vec4(transmit * (0.25 + 0.75 * diff), alpha);
}
)glsl";

    surfaceProgram_ = buildGlslProgram(
        kFullscreenVertexShader,
        kFragmentShader,
        "fluid surface shader"
    );
    if (surfaceProgram_ == 0) {
        shutdown();
        throw std::runtime_error("Failed to create the fluid surface program");
    }

    surfaceProjXYLocation_ = glGetUniformLocation(surfaceProgram_, "uProjXY");
    surfaceAbsorbLocation_ = glGetUniformLocation(surfaceProgram_, "uAbsorb");
    surfaceNormalThresholdLocation_ =
        glGetUniformLocation(surfaceProgram_, "uNormalDepthThreshold");
    if (surfaceProjXYLocation_ == -1 || surfaceAbsorbLocation_ == -1 ||
        surfaceNormalThresholdLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find a required surface shader uniform");
    }

    // 两个 sampler 分别去 0 号和 1 号纹理单元取数据
    glProgramUniform1i(surfaceProgram_, glGetUniformLocation(surfaceProgram_, "uDepthField"), 0);
    glProgramUniform1i(surfaceProgram_, glGetUniformLocation(surfaceProgram_, "uThickness"), 1);
}

// 读深度场和厚度场，重建法线 + Beer-Lambert，画到屏幕。
void FluidRenderer::shadeSurface(const glm::mat4& projection, float radius)
{
    // 绑回 0 号帧缓冲，也就是窗口本身。不解绑的话，下一帧 main.cpp 的清屏和地板
    // 会画进 FBO 里，屏幕就再也不刷新了（症状是画面卡在第一帧）。
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // 深度测试要关。全屏三角形的 gl_Position.z 写死是 0，换算成深度值是 0.5，
    // 这个数和地板谁近谁远纯属偶然——开着深度测试会随机剔掉一部分像素。
    // 挡不挡地板是靠片元里的 discard 决定的，不靠深度。
    //
    // 表面 shader 的 alpha 随厚度变化：轮廓越薄越接近 0。这里必须开启普通
    // straight-alpha 混合，否则本应渐隐的薄边会以不透明暗色写入，形成一圈黑边。
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // viewFromUv 反投影只需要这两个矩阵元素。glm 是列主序，projection[列][行]，
    // 所以 [0][0] 和 [1][1] 就是对角线上的那两项。
    glProgramUniform2f(surfaceProgram_, surfaceProjXYLocation_,
                       projection[0][0], projection[1][1]);

    // 两侧深度差不超过两个范围标准差时，把它们视为同一层连续表面并使用中央差分。
    // 当前默认是 2·4r = 8r；真正的前后层断层通常远大于这个值。
    glProgramUniform1f(surfaceProgram_, surfaceNormalThresholdLocation_,
                       2.0f * sigmaRange * radius);

    // σ = absorbScale · (0.45, 0.074, 0.02)，括号里是真实水的吸收系数 (1/m)。
    // 红光被吸收得比蓝光强 22 倍，这就是水呈蓝绿色的来源。
    glProgramUniform3f(surfaceProgram_, surfaceAbsorbLocation_,
                       absorbScale * 0.45f,
                       absorbScale * 0.074f,
                       absorbScale * 0.02f);

    glUseProgram(surfaceProgram_);
    glBindTextureUnit(0, depthField_);       // 0 号纹理单元 = uDepthField
    glBindTextureUnit(1, thicknessField_);   // 1 号纹理单元 = uThickness
    glBindVertexArray(emptyVao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);        // 三个顶点，全屏三角形

    glBindVertexArray(0);
    glUseProgram(0);
}

void FluidRenderer::shutdown()
{
    if (depthVbo_ != 0) {
        glDeleteBuffers(1, &depthVbo_);
        depthVbo_ = 0;
    }

    if (depthVao_ != 0) {
        glDeleteVertexArrays(1, &depthVao_);
        depthVao_ = 0;
    }

    if (emptyVao_ != 0) {
        glDeleteVertexArrays(1, &emptyVao_);
        emptyVao_ = 0;
    }

    if (depthField_ != 0) {
        glDeleteTextures(1, &depthField_);
        depthField_ = 0;
    }

    if (blurField_ != 0) {
        glDeleteTextures(1, &blurField_);
        blurField_ = 0;
    }

    if (thicknessField_ != 0) {
        glDeleteTextures(1, &thicknessField_);
        thicknessField_ = 0;
    }

    if (thicknessFbo_ != 0) {
        glDeleteFramebuffers(1, &thicknessFbo_);
        thicknessFbo_ = 0;
    }

    if (thicknessProgram_ != 0) {
        glDeleteProgram(thicknessProgram_);
        thicknessProgram_ = 0;
    }

    thickViewLocation_       = -1;
    thickProjectionLocation_ = -1;
    thickRadiusLocation_     = -1;
    thickViewportHLocation_  = -1;
    surfaceAbsorbLocation_   = -1;

    if (blurFbo_ != 0) {
        glDeleteFramebuffers(1, &blurFbo_);
        blurFbo_ = 0;
    }

    if (blurProgram_ != 0) {
        glDeleteProgram(blurProgram_);
        blurProgram_ = 0;
    }

    blurDirectionLocation_   = -1;
    blurRadiusScaleLocation_ = -1;
    blurScaleLocation_       = -1;

    if (depthBuffer_ != 0) {
        glDeleteRenderbuffers(1, &depthBuffer_);
        depthBuffer_ = 0;
    }

    if (fbo_ != 0) {
        glDeleteFramebuffers(1, &fbo_);
        fbo_ = 0;
    }

    targetWidth_ = 0;
    targetHeight_ = 0;

    if (depthProgram_ != 0) {
        glDeleteProgram(depthProgram_);
        depthProgram_ = 0;
    }

    if (surfaceProgram_ != 0) {
        glDeleteProgram(surfaceProgram_);
        surfaceProgram_ = 0;
    }

    surfaceProjXYLocation_ = -1;
    surfaceNormalThresholdLocation_ = -1;

    viewLocation_ = -1;
    projectionLocation_ = -1;
    viewportHLocation_ = -1;
    radiusLocation_ = -1;
    particleCapacity_ = 0;
}
