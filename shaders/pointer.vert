#version 330 core

// Directional pointer for spot and directional lights: a two-vertex line per
// instance running from the light along its aim direction.  Rendered with the
// same pinhole projection as the gizmo billboards so the two stay locked
// together while the light is dragged.

#define MAX_LIGHTS 8

uniform vec4 u_pointer_start[MAX_LIGHTS];  // xyz = light position, w = length
uniform vec4 u_pointer_dir[MAX_LIGHTS];    // xyz = aim direction,  w = visible
uniform vec4 u_gizmo_color[MAX_LIGHTS];

uniform vec4 u_intrinsics;
uniform vec2 u_resolution;
uniform vec2 u_image_scale;
uniform vec2 u_image_offset;

out vec3 v_color;
out float v_fade;

void main() {
    vec4 start = u_pointer_start[gl_InstanceID];
    vec4 dir = u_pointer_dir[gl_InstanceID];

    if (dir.w < 0.5) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_color = vec3(0.0);
        v_fade = 0.0;
        return;
    }

    float t = float(gl_VertexID);          // 0 at the light, 1 at the tip
    vec3 world = start.xyz + normalize(dir.xyz) * start.w * t;
    v_color = u_gizmo_color[gl_InstanceID].rgb;
    v_fade = 1.0 - 0.65 * t;               // fades out toward the tip

    if (world.z <= 1e-3) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }

    vec2 px = vec2(
        world.x * u_intrinsics.x / world.z + u_intrinsics.z,
        -world.y * u_intrinsics.y / world.z + u_intrinsics.w
    );
    vec2 uv = px / u_resolution;
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) * u_image_scale + u_image_offset;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
