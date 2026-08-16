# Relighting Studio

Real-time 2.5D single-image de-lighting and virtual relighting.

Load one ordinary photograph. The app estimates its depth, unprojects it to a
3D point cloud, derives surface normals, and strips the existing illumination
into a flat albedo texture. You then place virtual lights in 3D and see the
scene relit at 100+ FPS, with screen-space raymarched cast shadows.

The AI runs **once** per image on a background thread. Everything after that —
moving lights, changing colours, materials, shadows — is pure GLSL.

![Relighting Studio: five-pass G-buffer inspector on the left, relit viewport
with light gizmos in the centre, lighting controls on the
right](assets/screenshot.png)

---

## Quick start

```bash
pip install -r requirements.txt
```

Install `torch` from the right index first, or pip will silently give you the
CPU-only wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then run it against the bundled synthetic test scene:

```bash
python main.py --sample
```

The sample is ray-traced on first use into `assets/sample_scene.png`: three
spheres on a checkerboard floor under one warm key light. It is deliberately
chosen as the smoke test because the ground truth is known — the shading
gradient and hard cast shadows are exactly what the de-lighter has to remove,
and the checkerboard is exactly what it must *not* remove.

Or use your own image:

```bash
python main.py path/to/photo.jpg
```

You can also drag an image onto the window, or use **File → Open**.

On first run the depth model (~100 MB) downloads from Hugging Face. Every run
after that is offline and takes about a second.

---

## Viewport controls

| Input | Action |
| --- | --- |
| Left-drag a gizmo | Move that light in the image plane (X/Y at fixed Z) |
| Left-click the image | Move the selected light to the cursor |
| Ctrl / Alt + drag | Push the light toward or away from the camera |
| Right-drag a gizmo | Same, on the light under the cursor |
| Double-click | Snap the light just off the surface under the cursor |
| Wheel | Zoom |
| Ctrl + wheel | Change the selected light's depth |
| Middle-drag, Shift+drag | Pan |
| `F` | Reset the view |
| `F1`–`F7` | Switch view mode (beauty, albedo, normals, depth, …) |
| `Ctrl+1/2/3` | Add a point / spot / directional light |

---

## What the pipeline does

```
Input image I(u,v)
   │
   ├─► YOLO26 depth (yolo26m-depth.pt) ─► Z(u,v)   metric depth, in metres
   │                                        │
   │                                        ├─► unproject with K ──► P(X,Y,Z)
   │                                        │
   │                                        └─► ∂P/∂u × ∂P/∂v ─────► N(u,v)
   │
   └─► inverse shading solver ─────────────────────────────────────► A(u,v)
```

Depth comes from **Ultralytics YOLO26** (`yolo26m-depth.pt`), which predicts
*metric* depth — each pixel is a distance in metres from an unbounded
log-depth head covering roughly 0.02–150 m. That matters here because the
rest of the pipeline is physically parameterised: inverse-square falloff,
shadow ray lengths and light placement are all distances, and with metric
depth they are distances in real units rather than in an arbitrary window.
Inference is ~30 ms on a mid-range GPU, and the map comes back already
aligned to the input resolution.

The G-buffer records which regime produced it (`depth_units` in the export
manifest, and the Depth tab caption), so the units are never ambiguous.

**Unprojection.** With pinhole intrinsics `K` synthesised from an assumed
vertical FOV (`--fov`, default 55°):

```
X = (u - cx)·Z / fx      Y = -(v - cy)·Z / fy      Z = Z(u,v)
```

Camera space is X right, Y up, Z away from the camera, so viewer-facing
normals have a negative Z. The whole app uses this one convention.

**Deferred shading.** Per pixel, per light, in `shaders/relight.frag`:

```
L = normalize(P_light − P)          d  = |P_light − P|
Att = 1 / (kc + kl·d + kq·d²)
S_diff = max(N·L, 0) · I_color · Att
H = normalize(L + V)                V  = normalize(−P)
S_spec = GGX(N,H,V,roughness) · I_color · Att      [or (N·H)^shininess]
Color  = A · (ka·I_amb + Σ Vᵢ·S_diff,ᵢ) + Σ Vᵢ·S_spec,ᵢ
```

Lighting is evaluated in linear light and ACES-tonemapped; the albedo and
original textures are sRGB-decoded on sample and re-encoded at the end.

**Screen-space shadows.** `Vᵢ` comes from marching the depth buffer from `P`
toward each light, reprojecting every sample through `K` and comparing against
the recorded depth. Occluders further behind the ray than a thickness bound
are rejected — with a single depth layer there is no way to tell a real
blocker from a distant background, and an unbounded test smears shadows across
the frame. Each pixel averages **Rays** marches at evenly spread start
phases, which is what keeps the penumbra smooth rather than stippled.

**Normals.** Derived from the depth point cloud, with two details that
decide whether the result is usable on a real photograph:

- The depth is smoothed first with an edge-preserving filter whose support
  is a *fraction of the image size*, not a fixed pixel count — see the
  troubleshooting note on streaked walls for why.
- The tangents come from **one-sided differences chosen per pixel**, not a
  centred stencil. A centred stencil straddles an occlusion boundary — one
  neighbour on the object, the other on the background — and the difference
  describes a connecting surface that does not exist. Taking both one-sided
  differences and keeping whichever has the smaller depth change keeps the
  stencil on one surface layer.

That second point is worth qualifying, because it is easy to over-claim: it
only helps where the boundary is genuinely a step. This depth comes from a
network that predicts at lower internal resolution and upsamples, so
boundaries arrive spread over several pixels, and inside a ramp both
one-sided differences are moderate — there is nothing to reject. Applied to
raw depth it is markedly *worse* than a centred stencil (halo 1.42 against
0.64). Applied after the smoothing above it is modestly better, 0.64 → 0.59,
for about 135 ms. The smoothing is what keeps the boundary sharp enough for
it to bite.

---

## De-lighting

Relighting an image that still contains its original illumination gives you
double shadows and double highlights. The **De-lighting** panel toggles between
relighting the extracted albedo (default) and the raw image, so you can see the
difference directly.

### Neural decomposition (recommended)

The best albedo comes from **[compphoto/Intrinsic][ci]** — the SFU
Computational Photography Lab's ordinal-shading pipeline. It is optional
and off by default only because of its licence (below); when installed the
app detects it, prefers it automatically, and shows a **Neural
decomposition** checkbox in the De-lighting panel.

```bash
pip install "intrinsic @ git+https://github.com/compphoto/Intrinsic@main"
```

It is dramatically better than the analytic solver: shading gradients
flatten almost completely and most of a hard cast shadow disappears. On the
bundled sample the spheres come out as uniform flat colour with the
checkerboard intact — something no filter-based method achieves. Warm cost
is ~1.1 s, on par with the analytic solver. First run downloads ~1 GB of
weights.

> **Licence — read before shipping.** compphoto/Intrinsic is released for
> **academic use only**, and the method is **patent-protected**, with
> commercial licensing through the SFU Technology Licensing Office. This is
> stricter than AGPL: there is no compliance path for commercial use short
> of negotiating a licence. It is deliberately kept out of
> `requirements.txt` so it cannot be installed by accident, and the app is
> fully functional without it.

Two implementation details, both established by measurement:

- **Working resolution is capped at 1024 px.** Runtime is flat up to there
  and then falls off a cliff — the same image takes 1.4 s at 1024 px and
  **81 s at 1600 px** on an 8 GB card, as the allocator spills to host
  memory.
- **Full detail is preserved anyway.** Rather than upsampling the albedo,
  which would soften its texture, the *shading* is upsampled and albedo is
  re-derived at full resolution from the identity the pipeline guarantees,
  `linear_image = albedo × diffuse_shading + residual`. Shading is smooth
  so it survives resampling. Verified: the identity holds to 0.000000 mean
  absolute error, and re-deriving this way matches the network's own
  `hr_alb` to 0.008.

Inputs under ~256 px are upscaled to 384 px first, because `chrislib`'s
resolution helper raises `UnboundLocalError` below that. A uniformly black
frame makes the network emit all-NaN; that is detected and falls back to
the analytic solver rather than reaching the GPU.

[ci]: https://github.com/compphoto/Intrinsic

### Analytic solver (fallback, always available)

Used when the neural backend is absent or switched off. Three stages:

1. **Geometric shading.** Fit image luminance to a second-order spherical
   harmonic basis evaluated on the estimated normals. Nine coefficients
   describe any distant illumination of a Lambertian surface.
2. **Cast shadows.** The order-1 SH coefficients *are* the dominant light
   direction. Knowing it, the solver raymarches the depth buffer toward that
   light and finds the pixels it could not reach — the same trace the viewport
   shader runs, pointed at the photograph's own illumination. That visibility
   mask is appended to the basis and the fit is redone, so least squares
   decides how much of the image is cast shadow. This is what lets hard
   shadows come out at all: to a filter, a cast shadow and a dark patch of
   paint are identical, and only geometry separates them.
3. **Residual falloff.** What is left — vignetting, inverse-square falloff,
   bounce light — is captured as a smooth low-pass of the log residual.

### Honest limitations

These apply to the **analytic solver**; the neural backend is materially
better on all of them except the last two.

- The light-direction recovery assumes **one dominant source**. On the sample
  scene it lands within ~13° of ground truth; multiple strong sources will
  give a compromise direction.
- The shadow trace sees only the **single depth layer** the camera captured.
  A shadow whose occluder is outside the frame is invisible to it.
- Hard shadow edges are **attenuated, not erased**. Expect a residual ghost.
- YOLO26's metric scale is trustworthy on **real photographs**, its training
  domain — on a street scene it puts the road at 2.3 m, the bus at 4.8 m and
  the background at 12–16 m, all plausible. On synthetic or stylised images
  it is out of distribution and the scale compresses: on the bundled
  ray-traced sample it reads the back wall at 3.7 m against a true 7.5 m.
  The *ordering* stays correct, which is what the geometry needs; use the
  **Depth scale** slider to correct the absolute figure.
- YOLO26 upsamples from its internal resolution, so silhouettes can carry a
  slight staircase. A median pre-filter in the normal estimator suppresses
  most of it, but on high-contrast edges a faint crenellated rim can survive.
- **Occlusion boundaries are the hard limit of 2.5D.** A depth map records
  only the front surface, so the handful of pixels spanning an object's
  silhouette belong to neither the object nor the background, and there is
  no correct normal for them — whatever you put there renders as a thin band
  that catches light differently from both sides. The normal estimator is
  built to keep that band as narrow as possible (see
  `NORMAL_MEDIAN_PASSES` / `NORMAL_SMOOTH_PASSES`), which on the interior
  test image cuts the halo by 43%, but it cannot be removed without real 3D.
  Thin structures — chair legs, cables — are the visible case, because the
  depth model cannot resolve them from the background at all.
- Roughness and metallic are **scene-wide** artistic controls. A single image
  gives no per-pixel material map.

For production work, install the neural backend above.

---

## Model backends

Depth resolves in this order, and **Help → Pipeline backends** always shows
which one is live:

| Order | Backend | Units | Notes |
| --- | --- | --- | --- |
| 1 | `yolo26m-depth.pt` (Ultralytics YOLO26) | metres | Default. ~45 MB, ~30 ms |
| 2 | Depth Anything V2 Small (`transformers`) | relative | Used if ultralytics is missing |
| 3 | Analytic monocular prior | relative | No models, no network |

Pick a different checkpoint with `--depth-model`. A `.pt` name routes to
Ultralytics (`yolo26n-depth.pt` … `yolo26x-depth.pt`); anything else is
treated as a Hugging Face model id:

```bash
python main.py --depth-model yolo26x-depth.pt photo.jpg
python main.py --depth-model depth-anything/Depth-Anything-V2-Base-hf photo.jpg
```

> **Licence.** Ultralytics is **AGPL-3.0**. That is fine for personal,
> research and other open-source use, but distributing this app or offering
> it over a network obliges you to release your source under AGPL, or to buy
> an Ultralytics Enterprise licence. If neither works for you, uninstall
> `ultralytics` and the app falls back to Depth Anything V2 (Apache-2.0)
> automatically, with no code change.

## Optional deep models

### Surface normals from RGB — StableNormal (default, recommended)

Deriving normals by differentiating depth has a hard ceiling: a depth map
cannot resolve thin structures — chair legs, cables, window bars — so no
depth-derived method can recover them, and occlusion boundaries always cost
something. **StableNormal** predicts normals straight from the image and
sidesteps all of it. No differentiation means no `fx/Z` noise amplification
and no smoothing needed at all; boundaries come from appearance, so they are
steps rather than ramps; and thin structures appear because they are plainly
visible in the photograph even when invisible in the depth.

It is used automatically when `diffusers` is installed:

```bash
pip install diffusers
```

Measured on the 1504 px interior test image against the tuned analytic
estimator:

| | analytic | StableNormal | |
| --- | --- | --- | --- |
| halo (edge artifacts) | 0.600 | **0.250** | −58% |
| noise (flat surfaces) | 0.00718 | **0.00585** | −19% |
| time | 0.9 s | 3.6 s | one-off per import |
| peak VRAM | — | 4.8 GB | |

Disable with `--no-neural-normals`, or the **Geometry** checkbox in the
inspector; both fall back to the analytic estimator, which remains fully
supported.

**Licence: Apache-2.0 on both code and weights** — no commercial
restriction, unlike the depth and albedo backends.

Two compatibility measures live in `pipeline/stablenormal_backend.py`, both
deliberate and both commented there: the weights are pre-fetched into
`models/stablenormal/` because `hubconf` builds the Hugging Face repo id
with `os.path.join` (a backslash on Windows, which fails repo-id
validation), and one relocated `diffusers` module is aliased because
StableNormal targets diffusers 0.28 while pinning its full requirements set
would drag transformers and torch back far enough to break the depth and
albedo backends.

The axis convention was determined by fitting all eight sign combinations
against known-good analytic normals rather than read off a paper:
`(−x, +y, −z)`, at +0.969 mean agreement, and +0.894 versus +0.055 for the
runner-up on the pixels where the X component actually carries signal.

#### Alternatives

Licences checked directly, not taken from READMEs:

| Model | Code | Weights | Commercial use |
| --- | --- | --- | --- |
| **StableNormal** *(used)* | Apache-2.0 | Apache-2.0 | **Yes** |
| **Lotus** | Apache-2.0 | Apache-2.0 (`jingheya/lotus-normal-g-v1-1`) | **Yes** |
| DSINE | Proprietary | Manual download | No — non-commercial, no modification, no redistribution |
| Omnidata v2 | Dataset EULA | — | No — research/educational only |

DSINE is the most commonly cited of the four and the most restrictive; it
forbids even modification without written permission. Lotus is the closest
permissive alternative if you would rather not carry `diffusers` at this
version.

An ONNX export of any normal model also works — point `--normal-onnx` at it.

### Optional ONNX models

Both hooks accept an ONNX file and fall back silently when unset:

```bash
# DSINE / Omnidata-style surface normals instead of depth-derived ones
set IMAGELIGHTING_NORMAL_ONNX=C:\models\dsine.onnx

# IntrinsicAnything / Total-Relighting-style albedo instead of the solver
set IMAGELIGHTING_ALBEDO_ONNX=C:\models\intrinsic.onnx

python main.py --sample
```

Equivalent flags: `--normal-onnx`, `--albedo-onnx`, `--depth-model`.

**Help → Pipeline backends** shows exactly which one is live for each stage,
so you always know whether you are looking at a deep model or the fallback.

The app degrades all the way down: no ONNX models → analytic normals and the
inverse-shading solver; no torch or no network → an analytic depth prior too.
It stays usable offline, and says so in the status bar.

---

## Exports

- **Export relit image** (`Ctrl+E`) — PNG/JPEG/WebP, rendered offscreen at full
  G-buffer resolution rather than grabbed from the widget.
- **Export G-buffer archive** (`Ctrl+Shift+E`) — a `.zip` with `original`,
  `albedo`, `normal`, `depth` (16-bit), `depth_visualised`, `shading` and
  `beauty` as PNGs, lossless `depth.npy` / `normal.npy`, and a `manifest.json`
  carrying the intrinsics, the decode formula for every normalised buffer, and
  the full scene state.
- **Save / load lighting preset** — the scene as JSON, reusable across images.

---

## Layout

```
main.py                        entry point, surface format, CLI
core/
  qt_compat.py                 PySide6 / PyQt6 shim
  gbuffer.py                   GBuffer + CameraIntrinsics (unprojection)
  scene.py                     Light, Material, ShadowSettings, Scene
  imageio.py                   load/save, Unicode-safe on Windows
pipeline/
  base.py                      device select, colour space, torch warm-up
  depth_engine.py              YOLO26 metric depth, + relative/analytic fallbacks
  normal_engine.py             point-cloud normals + ONNX hook
  intrinsic_backend.py         compphoto/Intrinsic wrapper (optional)
  delighting_engine.py         backend selection + inverse-shading solver
  worker.py                    QThread worker and its GUI-side controller
render/
  relight_renderer.py          ModernGL resources, uniforms, offscreen render
shaders/
  fullscreen.vert relight.frag GLSL deferred relighting + SSRS
  gizmo.* pointer.*            light gizmo billboards and aim pointers
ui/
  main_window.py               layout, menus, wiring
  gl_viewport.py               QOpenGLWidget, mouse interaction, picking
  inspector.py                 right sidebar
  gbuffer_tabs.py              five-pass inspector
  widgets.py theme.py          controls and dark theme
export/exporter.py             PNG and G-buffer archive
tools/make_sample.py           ray-traced sample scene
```

---

## Troubleshooting

**"Could not create an OpenGL 3.3 core context"** — update your GPU driver.
Over RDP or in a VM, force software rendering:

```bash
set LIBGL_ALWAYS_SOFTWARE=1
```

**Shadows look stippled.** Raise **Rays**, not Steps. The stipple is variance
in the march's *start offset*, not its resolution: a single ray decides
"occluded or not" from wherever its jittered samples land, so an occluder
thinner than one step is a coin flip. Averaging phases fixes it and more
steps barely does — measured on the sample scene, going from 1 to 2 rays cut
the residual high-frequency energy by 30% for about 1 ms, while doubling
steps gained 2%. The dither itself is deliberately static rather than
per-frame: without a temporal filter, animating it turns a fixed pattern
into shimmer.

**Shading breaks into streaks or a fine crosshatch on walls and ceilings.**
This should be fixed, but if you meet it again, raise the normal smoothing
(`NORMAL_SMOOTH_FRACTION` in `pipeline/normal_engine.py`). The cause is worth
understanding: turning depth into a normal divides the depth gradient by the
pixel footprint, so sensitivity to depth error scales with `fx / Z`, and `fx`
grows with image width. On a 1504 px interior shot `fx ≈ 1383`, where a depth
wobble of one millimetre per pixel already tilts the normal by ten degrees.
On a surface facing the camera the true gradient is near zero, so on a flat
wall that wobble is the entire signal. This is why the artifact appears on
big photographs of rooms and not on small test images — and why the smoothing
support is a fraction of the image size rather than a fixed pixel count.

**Shadow acne, or shadows detached from their objects.** Adjust **Bias** —
raise it until the acne clears, lower it until contact shadows reattach.

**Everything is blown out.** Lower **Exposure**, or **Intensity** on the
brightest light. Defaults are solved from the scene scale, so this mostly
happens after manual edits.

**Out of VRAM on a large photo.** Lower `--buffer-max-side` (textures) and
`--max-side` (inference). Defaults are 1600 and 1024.

**Slow, or no GPU.** `--cpu` forces CPU inference. The viewport stays on the
GPU either way; only the one-time import gets slower.

**Slow startup on a machine with no internet.** Once the model is cached,
`transformers` still contacts Hugging Face on load to check for updates and
waits for the timeout. Skip it:

```bash
set HF_HUB_OFFLINE=1
```

---

## Notes for anyone extending this

`torch`'s intra-op thread pool **must** be created on the main thread before
any worker thread runs inference — `pipeline.base.warm_up_torch`, called from
`PipelineController.__init__`, does this. Skipping it corrupts the process heap
on the *second* inference on Windows, and the crash surfaces later as a bare
`0xC0000374` in an unrelated allocation with no Python traceback.

GL resources are owned explicitly: `RelightRenderer.upload_gbuffer` releases
the previous textures before allocating, and the viewport re-detects Qt's
default framebuffer only when Qt swaps it. Reloading images repeatedly is the
normal workflow, so the texture count is expected to stay flat.
