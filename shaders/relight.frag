#version 330 core

// ---------------------------------------------------------------------------
// 2.5D deferred relighting.
//
// Every pixel of the image is treated as a surfel: its camera-space position
// is unprojected from the depth buffer with the pinhole intrinsics, its
// orientation comes from the normal buffer, and its reflectance comes from
// the de-lit albedo buffer.  Lighting is then evaluated per pixel against up
// to MAX_LIGHTS virtual sources, with visibility resolved by raymarching the
// depth buffer in screen space.
//
// Colour management: the albedo/original textures are sRGB-encoded, so they
// are decoded to linear light before any arithmetic and re-encoded at the
// very end.  All lighting maths happens in linear space.
// ---------------------------------------------------------------------------

#define MAX_LIGHTS 8

#define LIGHT_POINT 0
#define LIGHT_SPOT 1
#define LIGHT_DIRECTIONAL 2

#define VIEW_BEAUTY 0
#define VIEW_ALBEDO 1
#define VIEW_NORMAL 2
#define VIEW_DEPTH 3
#define VIEW_SHADING 4
#define VIEW_SPECULAR 5
#define VIEW_ORIGINAL 6

#define SPEC_BLINN_PHONG 0
#define SPEC_GGX 1

in vec2 v_uv;
out vec4 f_color;

// --- G-buffer --------------------------------------------------------------
uniform sampler2D u_albedo;    // sRGB-encoded de-lit base colour
uniform sampler2D u_original;  // sRGB-encoded source image
uniform sampler2D u_normal;    // camera-space unit normals, [-1, 1]
uniform sampler2D u_depth;     // R32F metric depth, > 0
uniform sampler2D u_shading;   // scalar shading removed by the de-lighter

// --- Camera ----------------------------------------------------------------
uniform vec4 u_intrinsics;   // fx, fy, cx, cy in buffer pixels
uniform vec2 u_resolution;   // buffer width, height
uniform float u_depth_scale;

// --- Lights ----------------------------------------------------------------
uniform int  u_light_count;
uniform vec4 u_light_position[MAX_LIGHTS];   // xyz = position,  w = type
uniform vec4 u_light_direction[MAX_LIGHTS];  // xyz = aim dir,   w = cos(outer)
uniform vec4 u_light_color[MAX_LIGHTS];      // rgb = colour*intensity, w = cos(inner)
uniform vec4 u_light_atten[MAX_LIGHTS];      // kc, kl, kq,      w = casts shadow

// --- Material --------------------------------------------------------------
uniform float u_ambient;
uniform float u_diffuse;
uniform float u_specular;
uniform float u_shininess;
uniform float u_roughness;
uniform float u_metallic;
uniform float u_base_light;
uniform float u_exposure;
uniform float u_normal_strength;
uniform vec3  u_ambient_color;
uniform int   u_spec_model;

// --- Screen-space shadows --------------------------------------------------
uniform int   u_shadow_enabled;
uniform int   u_shadow_steps;
uniform int   u_shadow_rays;
uniform float u_shadow_max_distance;
uniform float u_shadow_bias;
uniform float u_shadow_softness;
uniform float u_shadow_strength;

// --- Output control --------------------------------------------------------
uniform int u_view_mode;
uniform int u_tonemap;
uniform int u_use_original;   // 1 = relight the raw image instead of albedo
uniform float u_frame_seed;   // animates the raymarch dither

const float PI = 3.14159265359;

// ---------------------------------------------------------------------------
// Colour space
// ---------------------------------------------------------------------------
vec3 srgb_to_linear(vec3 c) {
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}

vec3 linear_to_srgb(vec3 c) {
    c = max(c, vec3(0.0));
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

// Narkowicz's ACES filmic curve: cheap, and it rolls highlights off instead
// of clipping them, which matters when several lights overlap.
vec3 aces_tonemap(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------
float sample_depth(vec2 uv) {
    return texture(u_depth, uv).r * u_depth_scale;
}

// Unproject a pixel into camera space: X right, Y up, Z away from camera.
vec3 unproject(vec2 uv, float z) {
    vec2 px = uv * u_resolution;
    float x = (px.x - u_intrinsics.z) * z / u_intrinsics.x;
    float y = -(px.y - u_intrinsics.w) * z / u_intrinsics.y;
    return vec3(x, y, z);
}

// Inverse of unproject: camera space back to normalised image coordinates.
vec2 project(vec3 p) {
    float z = max(p.z, 1e-5);
    vec2 px = vec2(
        p.x * u_intrinsics.x / z + u_intrinsics.z,
        -p.y * u_intrinsics.y / z + u_intrinsics.w
    );
    return px / u_resolution;
}

vec3 fetch_normal(vec2 uv) {
    vec3 n = texture(u_normal, uv).xyz * 2.0 - 1.0;
    // Flatten toward, or exaggerate away from, the viewer.  Scaling only the
    // tangential components keeps the normal a plausible unit vector.
    n.xy *= max(u_normal_strength, 0.0);
    return normalize(n);
}

// ---------------------------------------------------------------------------
// Screen-space raymarched shadows
// ---------------------------------------------------------------------------
// Marches from the shaded point toward the light, reprojecting each sample
// into the depth buffer.  A sample is occluded when the recorded surface sits
// in front of the ray by more than the bias.  Occluders thicker than
// `thickness` are ignored: with a single depth layer we cannot tell a real
// blocker from a distant background surface, and rejecting them prevents the
// long shadow smears that naive SSRS produces.
float trace_shadow(vec3 origin, vec3 light_dir, float light_distance, vec3 normal) {
    if (u_shadow_enabled == 0) {
        return 1.0;
    }

    int steps = clamp(u_shadow_steps, 4, 64);
    float march = min(u_shadow_max_distance, light_distance);
    if (march <= 1e-4) {
        return 1.0;
    }
    float step_size = march / float(steps);

    // Interleaved gradient noise breaks the march's banding into a fine
    // stipple.  u_frame_seed is deliberately constant: with no temporal
    // filter to resolve it, animating the offset turns a static stipple
    // into shimmer, which is far more distracting in a still image.
    vec2 frag = gl_FragCoord.xy + vec2(u_frame_seed);
    float dither = fract(52.9829189 * fract(dot(frag, vec2(0.06711056, 0.00583715))));

    float thickness = max(u_shadow_bias * 4.0, u_shadow_softness * march * 0.5 + 0.02);

    // Average several rays whose start offsets are spread evenly across one
    // step.  A single ray decides "occluded or not" from wherever its
    // jittered samples happen to land, so an occluder thinner than a step
    // is a coin flip and the penumbra breaks into stipple -- which more
    // steps barely fix, because the variance is in the offset, not the
    // resolution.  Averaging a handful of phases turns that binary decision
    // into a graded fraction, which is both quieter and a better
    // approximation of a real penumbra.
    int rays = clamp(u_shadow_rays, 1, 4);
    float total = 0.0;

    for (int r = 0; r < 4; ++r) {
        if (r >= rays) break;

        float phase = fract(dither + float(r) / float(rays));
        // Start off the surface along the normal as well as the light
        // direction, so grazing angles do not shadow themselves.
        vec3 pos = origin
                 - normal * (u_shadow_bias * 0.5)
                 + light_dir * step_size * (0.5 + phase);

        float occlusion = 0.0;
        for (int i = 0; i < 64; ++i) {
            if (i >= steps) break;

            vec2 uv = project(pos);
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                break;
            }

            float scene_z = sample_depth(uv);
            float delta = pos.z - scene_z;  // > 0: the ray is behind a surface

            if (delta > u_shadow_bias && delta < thickness) {
                float travelled = step_size * float(i + 1);
                // Penumbra: distant occluders cast a softer, weaker shadow.
                float penumbra = 1.0 - u_shadow_softness * clamp(travelled / march, 0.0, 1.0);
                float edge = smoothstep(u_shadow_bias, u_shadow_bias * 3.0 + 0.004, delta);
                occlusion = max(occlusion, edge * penumbra);
                if (occlusion > 0.995) break;
            }

            pos += light_dir * step_size;
        }
        total += occlusion;
    }

    return 1.0 - (total / float(rays)) * clamp(u_shadow_strength, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// BRDF
// ---------------------------------------------------------------------------
float distribution_ggx(float n_dot_h, float roughness) {
    float a = max(roughness * roughness, 1e-3);
    float a2 = a * a;
    float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-6);
}

float geometry_smith(float n_dot_v, float n_dot_l, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    float gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// ---------------------------------------------------------------------------
// Debug visualisation
// ---------------------------------------------------------------------------
// A five-stop ramp in the spirit of Turbo: navy -> cyan -> green -> amber ->
// red, with near surfaces at the hot end.
vec3 heat_ramp(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.10, 0.04, 0.32);
    const vec3 c1 = vec3(0.08, 0.55, 0.85);
    const vec3 c2 = vec3(0.15, 0.82, 0.38);
    const vec3 c3 = vec3(0.97, 0.80, 0.14);
    const vec3 c4 = vec3(0.86, 0.15, 0.09);
    float s = t * 4.0;
    if (s < 1.0) return mix(c0, c1, s);
    if (s < 2.0) return mix(c1, c2, s - 1.0);
    if (s < 3.0) return mix(c2, c3, s - 2.0);
    return mix(c3, c4, s - 3.0);
}

// ---------------------------------------------------------------------------
void main() {
    vec2 uv = v_uv;

    float z = sample_depth(uv);
    vec3 position = unproject(uv, z);
    vec3 normal = fetch_normal(uv);

    vec3 base_srgb = (u_use_original == 1)
        ? texture(u_original, uv).rgb
        : texture(u_albedo, uv).rgb;
    vec3 base = srgb_to_linear(base_srgb);

    // The camera sits at the origin of camera space.
    vec3 view_dir = normalize(-position);
    float n_dot_v = max(dot(normal, view_dir), 1e-4);

    // Dielectrics reflect ~4% at normal incidence; metals tint the specular
    // with their own base colour and lose the diffuse lobe entirely.
    vec3 f0 = mix(vec3(0.04), base, clamp(u_metallic, 0.0, 1.0));
    vec3 diffuse_base = base * (1.0 - clamp(u_metallic, 0.0, 1.0));

    vec3 diffuse_accum = vec3(0.0);
    vec3 specular_accum = vec3(0.0);

    for (int i = 0; i < MAX_LIGHTS; ++i) {
        if (i >= u_light_count) break;

        int type = int(u_light_position[i].w + 0.5);
        vec3 light_pos = u_light_position[i].xyz;
        vec3 aim = normalize(u_light_direction[i].xyz);
        vec3 radiance = u_light_color[i].rgb;
        vec3 atten_k = u_light_atten[i].xyz;
        bool shadowing = u_light_atten[i].w > 0.5;

        vec3 to_light;
        float distance;
        float attenuation;

        if (type == LIGHT_DIRECTIONAL) {
            to_light = -aim;
            distance = u_shadow_max_distance;   // no falloff, but still shadowed
            attenuation = 1.0;
        } else {
            vec3 delta = light_pos - position;
            distance = length(delta);
            if (distance < 1e-5) continue;
            to_light = delta / distance;
            attenuation = 1.0 / max(
                atten_k.x + atten_k.y * distance + atten_k.z * distance * distance,
                1e-4
            );
        }

        if (type == LIGHT_SPOT) {
            float cos_outer = u_light_direction[i].w;
            float cos_inner = u_light_color[i].w;
            float cos_angle = dot(-to_light, aim);
            float cone = clamp(
                (cos_angle - cos_outer) / max(cos_inner - cos_outer, 1e-4), 0.0, 1.0
            );
            // Square the falloff so the cone edge reads as soft rather than
            // as a hard linear ramp.
            attenuation *= cone * cone;
        }

        if (attenuation <= 1e-5) continue;

        float n_dot_l = dot(normal, to_light);
        if (n_dot_l <= 0.0) continue;

        float visibility = 1.0;
        if (shadowing) {
            visibility = trace_shadow(position, to_light, distance, normal);
            if (visibility <= 1e-3) continue;
        }

        vec3 energy = radiance * attenuation * visibility;
        diffuse_accum += n_dot_l * energy;

        // Specular
        vec3 half_vec = normalize(to_light + view_dir);
        float n_dot_h = max(dot(normal, half_vec), 0.0);

        if (u_spec_model == SPEC_GGX) {
            float d = distribution_ggx(n_dot_h, clamp(u_roughness, 0.03, 1.0));
            float g = geometry_smith(n_dot_v, n_dot_l, clamp(u_roughness, 0.03, 1.0));
            vec3 f = fresnel_schlick(max(dot(half_vec, view_dir), 0.0), f0);
            vec3 brdf = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
            specular_accum += brdf * n_dot_l * energy;
        } else {
            float power = pow(n_dot_h, max(u_shininess, 1.0));
            specular_accum += power * energy * f0;
        }
    }

    vec3 ambient = u_ambient * u_ambient_color;
    vec3 irradiance = ambient + u_diffuse * diffuse_accum;

    vec3 color = diffuse_base * irradiance + u_specular * specular_accum;

    // A touch of the un-relit image keeps heavily de-lit results readable.
    color += base * u_base_light;

    color *= max(u_exposure, 0.0);

    // ---- debug views ------------------------------------------------------
    if (u_view_mode == VIEW_ALBEDO) {
        f_color = vec4(texture(u_albedo, uv).rgb, 1.0);
        return;
    } else if (u_view_mode == VIEW_ORIGINAL) {
        f_color = vec4(texture(u_original, uv).rgb, 1.0);
        return;
    } else if (u_view_mode == VIEW_NORMAL) {
        vec3 n = normal;
        n.z = -n.z;   // conventional normal-map look: viewer-facing is blue
        f_color = vec4(n * 0.5 + 0.5, 1.0);
        return;
    } else if (u_view_mode == VIEW_DEPTH) {
        float raw = texture(u_depth, uv).r;
        // The buffer is normalised to roughly [near, far] on the CPU side;
        // remap against the same window for a stable ramp.
        float t = 1.0 - clamp((raw - 0.5) / 5.5, 0.0, 1.0);
        f_color = vec4(heat_ramp(t), 1.0);
        return;
    } else if (u_view_mode == VIEW_SHADING) {
        vec3 shade = irradiance;
        f_color = vec4(linear_to_srgb(u_tonemap == 1 ? aces_tonemap(shade) : shade), 1.0);
        return;
    } else if (u_view_mode == VIEW_SPECULAR) {
        vec3 spec = u_specular * specular_accum;
        f_color = vec4(linear_to_srgb(u_tonemap == 1 ? aces_tonemap(spec) : spec), 1.0);
        return;
    }

    if (u_tonemap == 1) {
        color = aces_tonemap(color);
    }
    f_color = vec4(linear_to_srgb(color), 1.0);
}
