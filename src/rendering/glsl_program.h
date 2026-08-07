#pragma once

#include <cstdio>
#include <string>

#include <glad/gl.h>

inline GLuint compileGlslShader(
    GLenum type,
    const char* source,
    const char* name
)
{
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success == GL_TRUE) {
        return shader;
    }

    GLint logLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);

    std::string log(
        static_cast<std::size_t>(logLength > 0 ? logLength : 1),
        '\0'
    );
    glGetShaderInfoLog(shader, logLength, nullptr, log.data());

    const char* typeName = type == GL_VERTEX_SHADER
        ? "vertex shader"
        : "fragment shader";
    std::fprintf(
        stderr,
        "Failed to compile %s '%s':\n%s\n",
        typeName,
        name,
        log.c_str()
    );

    glDeleteShader(shader);
    return 0;
}

inline GLuint linkGlslProgram(
    const char* vertexSource,
    const char* fragmentSource,
    const char* name
)
{
    const GLuint vertexShader = compileGlslShader(
        GL_VERTEX_SHADER,
        vertexSource,
        name
    );
    if (vertexShader == 0) {
        return 0;
    }

    const GLuint fragmentShader = compileGlslShader(
        GL_FRAGMENT_SHADER,
        fragmentSource,
        name
    );
    if (fragmentShader == 0) {
        glDeleteShader(vertexShader);
        return 0;
    }

    const GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint success = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == GL_TRUE) {
        return program;
    }

    GLint logLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);

    std::string log(
        static_cast<std::size_t>(logLength > 0 ? logLength : 1),
        '\0'
    );
    glGetProgramInfoLog(program, logLength, nullptr, log.data());
    std::fprintf(stderr, "Failed to link program '%s':\n%s\n", name, log.c_str());

    glDeleteProgram(program);
    return 0;
}
