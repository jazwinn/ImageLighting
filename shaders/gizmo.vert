#version 330 core

// Instanced, attribute-less billboards for the light gizmos.  One instance
// per light; four vertices per instance form a screen-aligned quad of fixed
// pixel size centred on the light's projected position, so a gizmo stays
// grabbable regardless of how far away in Z the light has been pushed.

#define MAX_LIGHTS 8

uniform vec4 u_gizmo_position[MAX_LIGHTS];  // xyz = camera-space position, w = radius px
uniform vec4 u_gizmo_color[MAX_LIGHTS];     // rgb = colour, w = 1.0 when selected

uniform vec4 u_intrinsics;     // fx, fy, cx, cy
uniform vec2 u_resolution;     // G-buffer resolution
uniform vec2 u_image_scale;    // image-space -> NDC fit
uniform vec2 u_image_offset;
uniform vec2 u_viewport;       // widget size in device pixels

out vec2 v_local;
out vec3 v_color;
out float v_selected;
out float v_depth;

void main() {
    vec3 world = u_gizmo_position[gl_InstanceID].xyz;
    float radius_px = u_gizmo_position[gl_InstanceID].w;

    v_color = u_gizmo_color[gl_InstanceID].rgb;
    v_selected = u_gizmo_color[gl_InstanceID].w;
    v_depth = world.z;

    // Behind the camera: collapse the quad off-screen rather than letting the
    // projection wrap it around to a bogus position.
    if (world.z <= 1e-3) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_local = vec2(0.0);
        return;
    }

    vec2 px = vec2(
        world.x * u_intrinsics.x / world.z + u_intrinsics.z,
        -world.y * u_intrinsics.y / world.z + u_intrinsics.w
    );
    vec2 uv = px / u_resolution;

    vec2 center_ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) * u_image_scale + u_image_offset;

    vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1)) * 2.0 - 1.0;
    v_local = corner;

    vec2 offset = corner * (radius_px * 2.0) / u_viewport;
    gl_Position = vec4(center_ndc + offset, 0.0, 1.0);
}
