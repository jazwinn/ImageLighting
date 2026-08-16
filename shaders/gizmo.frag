#version 330 core

// Sphere impostor for a light gizmo, with a selection ring and an occlusion
// test against the scene depth buffer.  Occluded gizmos are drawn faintly
// rather than hidden: a light tucked behind the subject still has to be
// selectable.

in vec2 v_local;
in vec3 v_color;
in float v_selected;
in float v_depth;

out vec4 f_color;

uniform sampler2D u_depth;
uniform float u_depth_scale;
uniform vec2 u_image_scale;
uniform vec2 u_image_offset;
uniform vec2 u_viewport;

void main() {
    float r2 = dot(v_local, v_local);
    if (r2 > 1.0) discard;

    float r = sqrt(r2);

    // Recover this fragment's position in image space to look up scene depth.
    vec2 ndc = (gl_FragCoord.xy / u_viewport) * 2.0 - 1.0;
    vec2 img = (ndc - u_image_offset) / max(u_image_scale, vec2(1e-5));
    vec2 uv = vec2((img.x + 1.0) * 0.5, (1.0 - img.y) * 0.5);

    float occlusion = 1.0;
    if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
        float scene_z = texture(u_depth, uv).r * u_depth_scale;
        if (v_depth > scene_z + 0.01) {
            occlusion = 0.32;
        }
    }

    // Shaded ball, lit from the upper left so it reads as a 3D handle.
    float nz = sqrt(max(1.0 - r2, 0.0));
    vec3 n = vec3(v_local.x, -v_local.y, nz);
    float lambert = clamp(dot(n, normalize(vec3(-0.5, 0.6, 0.7))), 0.0, 1.0);
    vec3 body = v_color * (0.45 + 0.55 * lambert);
    body += vec3(pow(lambert, 24.0)) * 0.6;   // small specular pip

    float alpha = smoothstep(1.0, 0.94, r);

    // Selection ring just inside the silhouette.
    if (v_selected > 0.5) {
        float ring = smoothstep(0.74, 0.80, r) * (1.0 - smoothstep(0.92, 0.99, r));
        body = mix(body, vec3(1.0), ring * 0.9);
        alpha = max(alpha, ring);
    }

    f_color = vec4(body, alpha * occlusion);
}
