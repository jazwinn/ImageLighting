#version 330 core

// Draws the image plane as a screen-aligned quad.
//
// No vertex buffer is bound: the four corners come from gl_VertexID, which
// keeps the viewport free of per-frame buffer churn.  u_image_scale and
// u_image_offset carry the aspect-fit, zoom and pan mapping from image
// space into NDC, so the same transform can be reused when projecting
// light gizmos.

uniform vec2 u_image_scale;
uniform vec2 u_image_offset;

out vec2 v_uv;

void main() {
    // 0 -> (0,0), 1 -> (1,0), 2 -> (0,1), 3 -> (1,1) as a triangle strip.
    vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    v_uv = corner;

    // v_uv.y == 0 is the top row of the image, hence the Y flip.
    vec2 ndc = vec2(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0);
    gl_Position = vec4(ndc * u_image_scale + u_image_offset, 0.0, 1.0);
}
