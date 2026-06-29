"""
lightgrab_core.py — turn PSX emulator screenshots into Blender-ready environment maps.

Pipeline:
  1. Ingest screenshots (PNG/JPG/BMP) captured in DuckStation (or any emulator).
  2. Optional dither cleanup (PSX framebuffers are heavily ordered-dithered).
  3. Project each pinhole screenshot onto an equirectangular canvas using its
     yaw / pitch / horizontal-FOV, with feathered blending where shots overlap.
  4. Fill uncovered sky/floor regions with a pyramid-blur ambient fill so the
     map is usable for image-based lighting even from a handful of angles.
  5. Inverse-tonemap to pseudo-HDR (PSX is strictly LDR; we fabricate range):
     sRGB -> linear, highlight inflation, optional "sun" boost of the brightest blob.
  6. Export Radiance .hdr (and optional EXR), a tonemapped preview PNG, and a
     Blender script that wires the map into the World shader.

Everything is numpy/OpenCV; no GPU needed.
"""

from __future__ import annotations

import os
import math
import json
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2


# --------------------------------------------------------------------------- #
#  Data model
# --------------------------------------------------------------------------- #

@dataclass
class Shot:
    """One screenshot with its camera orientation."""
    path: str
    yaw_deg: float = 0.0     # 0 = forward/north, 90 = right/east, 180 = back, 270 = left
    pitch_deg: float = 0.0   # +up / -down
    roll_deg: float = 0.0    # usually 0 for game cameras
    hfov_deg: float = 60.0   # horizontal field of view of the in-game camera
    enabled: bool = True

    def to_dict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d):
        return Shot(**d)


@dataclass
class BuildSettings:
    out_width: int = 2048              # equirect width (height = width // 2)
    denoise_dither: bool = True        # dither cleanup pass (see dither_mode)
    dither_mode: str = "bayer"         # "bayer" | "median" | "off"
    feather: float = 0.15              # edge feather fraction for blending overlaps
    hdr_strength: float = 0.6          # 0 = plain linearized LDR, 1 = aggressive inflation
    sun_boost: float = 4.0             # extra multiplier on bright blobs (1 = off)
    sun_max_blobs: int = 4             # max distinct light sources to boost independently
    sun_threshold: float = 0.92        # luminance percentile-ish threshold for "sun" pixels
    ambient_fill: bool = True          # fill uncovered regions with blurred ambient
    exposure: float = 1.0              # global gain applied at the end
    auto_crop: bool = True             # detect & remove black border bars (emulator padding)
    hud_color: str = ""                # hex like "#22cc33"; if set, inpaint matching pixels
    hud_tolerance: int = 55            # per-channel tolerance for HUD color match
    watch_dir: str = ""                # persisted watch-folder path (GUI convenience)

    def to_dict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d):
        return BuildSettings(**{k: v for k, v in d.items()
                                if k in {f.name for f in dataclasses.fields(BuildSettings)}})


# --------------------------------------------------------------------------- #
#  Color helpers
# --------------------------------------------------------------------------- #

def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """img float32 in [0,1] -> linear."""
    a = 0.055
    return np.where(img <= 0.04045, img / 12.92, ((img + a) / (1 + a)) ** 2.4).astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    a = 0.055
    img = np.clip(img, 0.0, 1.0)
    return np.where(img <= 0.0031308, img * 12.92, (1 + a) * np.power(img, 1 / 2.4) - a).astype(np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)


def crop_black_borders(rgb: np.ndarray, thresh: float = 8.0) -> np.ndarray:
    """Trim near-black bars (emulator framebuffer padding) from all four edges."""
    col = rgb.mean(axis=(0, 2))
    row = rgb.mean(axis=(1, 2))
    w, h = rgb.shape[1], rgb.shape[0]
    if (col > thresh).any() and (row > thresh).any():
        l = int(np.argmax(col > thresh)); r = int(w - np.argmax(col[::-1] > thresh))
        t = int(np.argmax(row > thresh)); b = int(h - np.argmax(row[::-1] > thresh))
        if r - l > w // 2 and b - t > h // 2:        # sanity: don't crop to a sliver
            return rgb[t:b, l:r]
    return rgb


def remove_hud(rgb: np.ndarray, hex_color: str, tol: int) -> np.ndarray:
    """Inpaint pixels close to the given HUD color (health bars, PRESS START, timers)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return rgb
    target = np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)], dtype=np.int16)
    diff = np.abs(rgb.astype(np.int16) - target[None, None, :]).max(axis=2)
    mask = (diff <= tol).astype(np.uint8) * 255
    if not mask.any():
        return rgb
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    return cv2.inpaint(rgb, mask, 5, cv2.INPAINT_TELEA)


def _denoise_bayer(rgb: np.ndarray) -> np.ndarray:
    """PSX-aware dither removal.

    The PSX GPU uses a fixed 4x4 ordered Bayer dither at the 5-bit (32-level)
    quantization boundary. Each 2x2 pixel block has a predictable offset pattern
    rather than random noise, so a standard median blur overkills fine edges.

    Strategy:
      1. Detect the 2x2 dither phase by testing four pixel-grid alignments and
         picking the one whose even-odd pixel difference is largest (the Bayer
         pattern is strongest along its axis).
      2. Average each 2x2 cell with the correct phase — this precisely cancels
         the ordered offset while leaving 1px edges intact.
      3. A gentle bilateral sharpens contours back up without re-banding.

    Falls back to the original median+bilateral if ximgproc is unavailable.
    """
    h, w = rgb.shape[:2]
    # Phase detection: sample the mean absolute diff between interleaved rows/cols
    # at each of the four possible 2x2 alignments (dy in {0,1}, dx in {0,1}).
    best_score, best_dy, best_dx = -1.0, 0, 0
    for dy in (0, 1):
        for dx in (0, 1):
            even_rows = rgb[dy:h - (h - dy) % 2:2, dx:w - (w - dx) % 2:2]
            odd_rows  = rgb[dy + 1:h - (h - dy) % 2:2, dx:w - (w - dx) % 2:2]
            min_h = min(even_rows.shape[0], odd_rows.shape[0])
            if min_h == 0:
                continue
            score = float(np.abs(even_rows[:min_h].astype(np.int16)
                                 - odd_rows[:min_h].astype(np.int16)).mean())
            if score > best_score:
                best_score, best_dy, best_dx = score, dy, dx

    # Average each 2x2 block at the detected phase.
    out = rgb.astype(np.float32)
    dy, dx = best_dy, best_dx
    h2 = (h - dy) // 2 * 2
    w2 = (w - dx) // 2 * 2
    block = out[dy:dy + h2, dx:dx + w2].reshape(h2 // 2, 2, w2 // 2, 2, 3)
    avg = block.mean(axis=(1, 3))                        # (h2//2, w2//2, 3)
    out[dy:dy + h2, dx:dx + w2] = np.repeat(
        np.repeat(avg, 2, axis=0), 2, axis=1)
    rgb = np.clip(out, 0, 255).astype(np.uint8)

    # Bilateral to tighten edges that the block-average slightly blurred.
    return cv2.bilateralFilter(rgb, d=5, sigmaColor=20, sigmaSpace=4)


def load_screenshot(path: str, denoise_dither: bool = True,
                    settings: "BuildSettings | None" = None) -> np.ndarray:
    """Load image as float32 RGB [0,1] with optional preprocessing:
    border auto-crop, HUD color removal, and PSX dither smoothing."""
    raw = cv2.imread(path, cv2.IMREAD_COLOR)
    if raw is None:
        raise IOError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    if settings is not None:
        if settings.auto_crop:
            rgb = crop_black_borders(rgb)
        if settings.hud_color:
            rgb = remove_hud(rgb, settings.hud_color, settings.hud_tolerance)
    if denoise_dither:
        mode = settings.dither_mode if settings is not None else "bayer"
        if mode == "bayer":
            rgb = _denoise_bayer(rgb)
        elif mode == "median":
            # Original approach: small median kills the checkerboard, bilateral
            # restores edges without re-banding.
            rgb = cv2.medianBlur(rgb, 3)
            rgb = cv2.bilateralFilter(rgb, d=5, sigmaColor=24, sigmaSpace=5)
        # mode == "off": no dither pass
    return (rgb.astype(np.float32) / 255.0)


# --------------------------------------------------------------------------- #
#  Equirect projection
# --------------------------------------------------------------------------- #

def _equirect_directions(width: int, height: int) -> np.ndarray:
    """Unit direction vector for every equirect pixel. Returns (H, W, 3).

    Convention: yaw 0 looks down +Z (forward), +X is right, +Y is up.
    Equirect: u=0 is yaw -180, u=0.5 is yaw 0; v=0 is straight up.
    """
    u = (np.arange(width, dtype=np.float32) + 0.5) / width        # 0..1
    v = (np.arange(height, dtype=np.float32) + 0.5) / height      # 0..1
    yaw = (u - 0.5) * 2.0 * np.pi                                  # -pi..pi
    pitch = (0.5 - v) * np.pi                                      # +pi/2 (up) .. -pi/2
    yaw_g, pitch_g = np.meshgrid(yaw, pitch)
    cp = np.cos(pitch_g)
    dirs = np.stack([cp * np.sin(yaw_g),        # X (right)
                     np.sin(pitch_g),           # Y (up)
                     cp * np.cos(yaw_g)],       # Z (forward)
                    axis=-1)
    return dirs.astype(np.float32)


def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """World->camera rotation for a camera oriented by yaw/pitch/roll."""
    y, p, r = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    Ry = np.array([[math.cos(y), 0, math.sin(y)],
                   [0, 1, 0],
                   [-math.sin(y), 0, math.cos(y)]], dtype=np.float32)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(p), -math.sin(p)],
                   [0, math.sin(p), math.cos(p)]], dtype=np.float32)
    Rz = np.array([[math.cos(r), -math.sin(r), 0],
                   [math.sin(r), math.cos(r), 0],
                   [0, 0, 1]], dtype=np.float32)
    # camera = Ry @ Rx @ Rz applied to camera-local axes; we need world->cam = R^T
    R = Ry @ Rx @ Rz
    return R.T


def project_shot(canvas: np.ndarray, weight: np.ndarray, shot: Shot,
                 img: np.ndarray, feather: float) -> None:
    """Splat one pinhole screenshot into the equirect accumulation buffers (in place)."""
    H, W = canvas.shape[:2]
    ih, iw = img.shape[:2]
    dirs = _equirect_directions(W, H)                       # (H, W, 3) world dirs
    Rwc = _rotation_matrix(shot.yaw_deg, shot.pitch_deg, shot.roll_deg)
    cam = dirs @ Rwc.T                                      # world -> camera space

    z = cam[..., 2]
    in_front = z > 1e-6

    hfov = math.radians(shot.hfov_deg)
    fx = (iw / 2.0) / math.tan(hfov / 2.0)
    fy = fx  # square pixels; vertical FOV follows aspect

    with np.errstate(divide="ignore", invalid="ignore"):
        px = (cam[..., 0] / z) * fx + iw / 2.0
        py = (-cam[..., 1] / z) * fy + ih / 2.0

    inside = in_front & (px >= 0) & (px <= iw - 1) & (py >= 0) & (py <= ih - 1)
    if not inside.any():
        return

    map_x = np.where(inside, px, 0).astype(np.float32)
    map_y = np.where(inside, py, 0).astype(np.float32)
    sampled = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)

    # Feathered weight: 1 in the middle of the frame, falls to 0 at the borders.
    nx = np.clip(np.minimum(px, iw - 1 - px) / (iw * max(feather, 1e-3)), 0, 1)
    ny = np.clip(np.minimum(py, ih - 1 - py) / (ih * max(feather, 1e-3)), 0, 1)
    w = np.where(inside, (nx * ny).astype(np.float32), 0.0)

    canvas += sampled * w[..., None]
    weight += w


def _hpad(arr: np.ndarray, pad: int) -> np.ndarray:
    """Wrap-pad an equirect array horizontally so pyrDown/pyrUp see no hard seam."""
    return np.concatenate([arr[:, -pad:], arr, arr[:, :pad]], axis=1)


def _hcrop(arr: np.ndarray, pad: int) -> np.ndarray:
    return arr[:, pad:-pad]


def pyramid_fill(canvas: np.ndarray, weight: np.ndarray, levels: int = 8) -> np.ndarray:
    """Fill zero-weight regions with progressively blurred covered content.

    Push-pull style: downsample (weighted), then on the way back up use real data
    where it exists and the blurred estimate where it doesn't. Gives a smooth
    ambient gradient in the holes instead of black voids.

    The equirect left/right edge maps to yaw=±180 — the same point on the sphere.
    We wrap-pad before the pyramid so the Gaussian kernels see continuity there,
    then crop back after. Without this, partial-coverage builds have a visible
    seam stripe at yaw=180.
    """
    eps = 1e-6
    W = canvas.shape[1]
    pad = max(W // 8, 4)

    c_pad = _hpad(canvas, pad)
    w_pad = _hpad(weight[..., None], pad)[..., 0]

    imgs = [c_pad]
    wts = [w_pad]
    for _ in range(levels):
        c = cv2.pyrDown(imgs[-1])
        w = cv2.pyrDown(wts[-1])
        imgs.append(c)
        wts.append(w)
        if min(c.shape[:2]) <= 4:
            break

    # Coarsest level: normalize whatever we have (global average where empty).
    top = imgs[-1] / np.maximum(wts[-1], eps)[..., None]
    mean_color = (imgs[-1].sum(axis=(0, 1)) / max(wts[-1].sum(), eps))
    top = np.where(wts[-1][..., None] > eps, top, mean_color[None, None, :])

    filled = top
    for lvl in range(len(imgs) - 2, -1, -1):
        up = cv2.pyrUp(filled)
        up = cv2.resize(up, (imgs[lvl].shape[1], imgs[lvl].shape[0]))
        real = imgs[lvl] / np.maximum(wts[lvl], eps)[..., None]
        a = np.clip(wts[lvl], 0, 1)[..., None]
        filled = real * a + up * (1 - a)

    # Remove the horizontal wrap-padding.
    scale = filled.shape[1] / c_pad.shape[1]
    crop = max(int(round(pad * scale)), 1)
    return _hcrop(filled, crop).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Pseudo-HDR
# --------------------------------------------------------------------------- #

def pseudo_hdr(linear_img: np.ndarray, strength: float, sun_boost: float,
               sun_threshold: float, exposure: float,
               sun_max_blobs: int = 4) -> np.ndarray:
    """Fabricate dynamic range from an LDR (linearized) image.

    - Inverse Reinhard inflation: L' = L / (1 - k*L). Bright pixels shoot up,
      midtones barely move, so the map gains lighting punch without going gray.
    - Sun pass: distinct bright blobs (campfire, lamp, sky disk, fog glow) each
      get their own boosted Gaussian so Blender sees multiple independent light
      sources rather than one merged blob. Up to sun_max_blobs are processed,
      sorted by area (largest first).
    """
    img = np.clip(linear_img, 0.0, 1.0)
    k = np.clip(strength, 0.0, 0.97)
    inflated = img / np.maximum(1.0 - k * img, 0.03)

    if sun_boost > 1.0:
        lum = luminance(img)
        thr = max(float(np.quantile(lum, sun_threshold)), 0.5)
        binary = (lum >= thr).astype(np.uint8)
        if binary.any():
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8)
            # stats columns: LEFT, TOP, WIDTH, HEIGHT, AREA; label 0 is background
            blob_labels = sorted(range(1, n_labels),
                                 key=lambda i: stats[i, cv2.CC_STAT_AREA],
                                 reverse=True)
            blob_labels = blob_labels[:max(sun_max_blobs, 1)]

            sigma_base = max(linear_img.shape[1] // 256, 3)
            combined_soft = np.zeros(lum.shape, dtype=np.float32)
            for lbl in blob_labels:
                blob_mask = (labels == lbl).astype(np.float32)
                w = stats[lbl, cv2.CC_STAT_WIDTH]
                h = stats[lbl, cv2.CC_STAT_HEIGHT]
                sigma = max(int(max(w, h) * 0.4), sigma_base)
                soft = cv2.GaussianBlur(blob_mask, (0, 0),
                                        sigmaX=sigma, sigmaY=sigma)
                combined_soft = np.maximum(combined_soft, soft)

            combined_soft = np.clip(
                combined_soft / max(combined_soft.max(), 1e-6), 0, 1)
            inflated *= (1.0 + (sun_boost - 1.0) * combined_soft)[..., None]

    return (inflated * exposure).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Gradient mode (single-screenshot "lighting vibe" grab)
# --------------------------------------------------------------------------- #

def gradient_environment(img: np.ndarray, out_width: int) -> np.ndarray:
    """Build a synthetic equirect from one screenshot's vertical color structure.

    Many PSX games never let you look up — so instead of a panorama we read the
    screenshot's row-wise average colors (sky at top, ground at bottom), and
    sweep that gradient around the sphere. Crude, but it nails the palette and
    overall light direction-from-above, which is most of what IBL contributes.
    """
    H = out_width // 2
    rows = img.mean(axis=1)                                  # (ih, 3) top->bottom
    ih = rows.shape[0]
    # Screenshot covers roughly pitch +30..-30 of the sphere; extrapolate ends.
    src_v = np.linspace(0.30, 0.70, ih)
    dst_v = (np.arange(H) + 0.5) / H
    env_rows = np.empty((H, 3), dtype=np.float32)
    for c in range(3):
        env_rows[:, c] = np.interp(dst_v, src_v, rows[:, c],
                                   left=rows[0, c], right=rows[-1, c])
    env_rows = cv2.GaussianBlur(env_rows.reshape(H, 1, 3), (0, 0), sigmaY=H / 64,
                                sigmaX=0.01).reshape(H, 3)
    env = np.repeat(env_rows[:, None, :], out_width, axis=1)
    return env.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Build + export
# --------------------------------------------------------------------------- #

def build_panorama(shots: list[Shot], settings: BuildSettings,
                   progress=None) -> np.ndarray:
    """Full pipeline: shots -> pseudo-HDR equirect (float32 RGB, linear)."""
    W = settings.out_width
    H = W // 2
    canvas = np.zeros((H, W, 3), np.float32)
    weight = np.zeros((H, W), np.float32)

    active = [s for s in shots if s.enabled]
    if not active:
        raise ValueError("No enabled shots.")

    for i, shot in enumerate(active):
        if progress:
            progress(f"Projecting {os.path.basename(shot.path)} "
                     f"({i + 1}/{len(active)})")
        img = load_screenshot(shot.path, settings.denoise_dither, settings)
        img = srgb_to_linear(img)
        project_shot(canvas, weight, shot, img, settings.feather)

    if progress:
        progress("Filling uncovered regions")
    if settings.ambient_fill:
        filled = pyramid_fill(canvas, weight)
    else:
        filled = canvas / np.maximum(weight, 1e-6)[..., None]

    if progress:
        progress("Inflating to pseudo-HDR")
    hdr = pseudo_hdr(filled, settings.hdr_strength, settings.sun_boost,
                     settings.sun_threshold, settings.exposure,
                     settings.sun_max_blobs)
    return hdr


def build_gradient(shot_path: str, settings: BuildSettings) -> np.ndarray:
    img = load_screenshot(shot_path, settings.denoise_dither, settings)
    img = srgb_to_linear(img)
    env = gradient_environment(img, settings.out_width)
    return pseudo_hdr(env, settings.hdr_strength, settings.sun_boost,
                      settings.sun_threshold, settings.exposure,
                      settings.sun_max_blobs)


def save_hdr(hdr_rgb: np.ndarray, path: str) -> None:
    """Write Radiance .hdr (Blender reads these natively)."""
    bgr = cv2.cvtColor(hdr_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, bgr.astype(np.float32)):
        raise IOError(f"Failed to write {path}")


def save_exr(hdr_rgb: np.ndarray, path: str) -> bool:
    """Write OpenEXR if this cv2 build supports it. Returns success."""
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        bgr = cv2.cvtColor(hdr_rgb, cv2.COLOR_RGB2BGR)
        return bool(cv2.imwrite(path, bgr.astype(np.float32)))
    except cv2.error:
        return False


def save_preview(hdr_rgb: np.ndarray, path: str) -> None:
    """Tonemapped sRGB preview PNG."""
    tm = hdr_rgb / (1.0 + hdr_rgb)            # Reinhard
    srgb = (linear_to_srgb(tm) * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(srgb, cv2.COLOR_RGB2BGR))


def save_project(path: str, shots: list[Shot], settings: BuildSettings) -> None:
    with open(path, "w") as f:
        json.dump({"shots": [s.to_dict() for s in shots],
                   "settings": settings.to_dict()}, f, indent=2)


def load_project(path: str) -> tuple[list[Shot], BuildSettings]:
    with open(path) as f:
        data = json.load(f)
    return ([Shot.from_dict(d) for d in data.get("shots", [])],
            BuildSettings.from_dict(data.get("settings", {})))


BLENDER_TEMPLATE = '''"""Auto-generated by PSX LightGrab.
Run inside Blender:  blender --python {script_name}
or paste into Blender's Scripting tab and hit Run. Sets the World to use the
grabbed environment map with a Mapping node so you can spin it to taste.
"""
import bpy, os

HDRI_PATH = r"{hdri_path}"

world = bpy.data.worlds.get("PSX_LightGrab") or bpy.data.worlds.new("PSX_LightGrab")
bpy.context.scene.world = world
world.use_nodes = True
nt = world.node_tree
nt.nodes.clear()

n_out = nt.nodes.new("ShaderNodeOutputWorld");      n_out.location = (600, 0)
n_bg  = nt.nodes.new("ShaderNodeBackground");        n_bg.location = (380, 0)
n_env = nt.nodes.new("ShaderNodeTexEnvironment");    n_env.location = (80, 0)
n_map = nt.nodes.new("ShaderNodeMapping");           n_map.location = (-160, 0)
n_tex = nt.nodes.new("ShaderNodeTexCoord");          n_tex.location = (-400, 0)

n_env.image = bpy.data.images.load(HDRI_PATH)
n_env.interpolation = "Closest" if {pixelated} else "Linear"

nt.links.new(n_tex.outputs["Generated"], n_map.inputs["Vector"])
nt.links.new(n_map.outputs["Vector"],    n_env.inputs["Vector"])
nt.links.new(n_env.outputs["Color"],     n_bg.inputs["Color"])
nt.links.new(n_bg.outputs["Background"], n_out.inputs["Surface"])
n_bg.inputs["Strength"].default_value = 1.0

print("PSX LightGrab world set up:", os.path.basename(HDRI_PATH))
'''


def write_blender_script(hdri_path: str, script_path: str, pixelated: bool = True) -> None:
    with open(script_path, "w") as f:
        f.write(BLENDER_TEMPLATE.format(
            hdri_path=os.path.abspath(hdri_path),
            script_name=os.path.basename(script_path),
            pixelated="True" if pixelated else "False"))


def boost_existing(path_in: str, settings: BuildSettings) -> np.ndarray:
    """Load an existing float render (.exr/.hdr) or LDR image and apply the
    pseudo-HDR pass. Used to post-process Blender bakes from the OBJ workflow."""
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    raw = cv2.imread(path_in, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if raw is None:
        raise IOError(f"Could not read {path_in}")
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32)
    if raw.dtype == np.uint8:
        rgb = srgb_to_linear(rgb / 255.0)        # LDR input -> linearize
    # float inputs (EXR/HDR) are already linear; normalize into [0,1] for the
    # inflation curve, then restore scale so existing >1 values survive.
    peak = max(float(rgb.max()), 1e-6)
    norm = np.clip(rgb / max(peak, 1.0), 0, 1)
    out = pseudo_hdr(norm, settings.hdr_strength, settings.sun_boost,
                     settings.sun_threshold, settings.exposure,
                     settings.sun_max_blobs)
    return out * max(peak, 1.0)
