#version 330 core

in vec3 v_color;
in float v_fade;

out vec4 f_color;

void main() {
    f_color = vec4(v_color, clamp(v_fade, 0.0, 1.0) * 0.85);
}
