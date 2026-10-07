"""Per-object 'shader-like' effects. Pure numpy/OpenCV (no torch needed).

Every effect gets: img (uint8 RGB HxWx3), m (float32 HxW mask 0..1),
s (strength 0..1), c (r,g,b 0..255), code (custom python) and returns
(out_image, alpha) -> the result is blended into the frame using `alpha`.
"""
import os
import re

import cv2
import numpy as np

# ---------------------------------------------------------------- mask utils
def pack_mask(m):
    if m is None or not np.any(m):
        return None
    return np.packbits(np.asarray(m, bool).ravel())


def unpack_mask(p, shape):
    if p is None:
        return np.zeros(shape, bool)
    return np.unpackbits(p)[: shape[0] * shape[1]].reshape(shape).astype(bool)


def soften(m, px):
    f = np.asarray(m, np.float32)
    if px and px > 0:
        f = cv2.GaussianBlur(f, (0, 0), float(px))
    return f


def parse_color(c, default=(0, 255, 136)):
    if isinstance(c, (tuple, list)) and len(c) >= 3:
        return tuple(int(x) for x in c[:3])
    if isinstance(c, str):
        c = c.strip()
        if c.startswith("#"):
            h = c[1:]
            if len(h) == 3:
                h = "".join(ch * 2 for ch in h)
            if len(h) >= 6:
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        nums = re.findall(r"[\d.]+", c)
        if len(nums) >= 3:
            return tuple(int(float(x)) for x in nums[:3])
    return default


def _k(px):
    px = max(1, int(px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))


def _mix(a, b, s):
    return a.astype(np.float32) * (1 - s) + b.astype(np.float32) * s


def _gray3(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.repeat(g[..., None], 3, axis=2)

# ------------------------------------------------------------------- effects
def fx_blur(img, m, s, c, code):
    return cv2.GaussianBlur(img, (0, 0), 1 + s * 35), m


def fx_pixelate(img, m, s, c, code):
    h, w = img.shape[:2]
    b = max(2, int(3 + s * 60))
    small = cv2.resize(img, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST), m


def fx_bw(img, m, s, c, code):
    return _mix(img, _gray3(img), s), m


def fx_invert(img, m, s, c, code):
    return _mix(img, 255 - img, s), m


def fx_sepia(img, m, s, c, code):
    k = np.array([[.393, .769, .189], [.349, .686, .168], [.272, .534, .131]], np.float32)
    sep = np.clip(img.astype(np.float32) @ k.T, 0, 255)
    return _mix(img, sep, s), m


def fx_tint(img, m, s, c, code):
    lum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    tinted = np.clip(lum[..., None] * np.array(c, np.float32) * 1.3, 0, 255)
    return _mix(img, tinted, s), m


def fx_brightness(img, m, s, c, code):  # 0.5 = unchanged
    return np.clip(img.astype(np.float32) * (s * 2), 0, 255), m


def fx_thermal(img, m, s, c, code):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    t = cv2.cvtColor(cv2.applyColorMap(g, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    return _mix(img, t, s), m


def fx_outline(img, m, s, c, code):
    mb = (m > 0.5).astype(np.uint8)
    ring = cv2.dilate(mb, _k(2 + s * 12)) - mb
    out = np.empty_like(img)
    out[:] = np.array(c, np.uint8)
    return out, cv2.GaussianBlur(ring.astype(np.float32), (0, 0), 0.8)


def fx_glow(img, m, s, c, code):
    mb = (m > 0.5).astype(np.uint8)
    d = cv2.dilate(mb, _k(3 + s * 25)).astype(np.float32)
    glow = cv2.GaussianBlur(d, (0, 0), 4 + s * 25)
    add = glow[..., None] * np.array(c, np.float32) * 1.1 * (1 - m)[..., None]
    return np.clip(img.astype(np.float32) + add, 0, 255), np.ones_like(m)


def fx_neon(img, m, s, c, code):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(g, 60, 140)
    e = cv2.dilate(e, _k(1)).astype(np.float32) / 255.0
    glow = np.minimum(cv2.GaussianBlur(e, (0, 0), 2 + s * 6) * 3 + e, 1.0)
    out = img.astype(np.float32) * 0.25 + glow[..., None] * np.array(c, np.float32)
    return np.clip(out, 0, 255), m


def fx_cartoon(img, m, s, c, code):
    levels = int(3 + (1 - s) * 10)
    step = max(8, 256 // levels)
    sm = cv2.medianBlur(img, 5)
    q = (sm // step) * step + step // 2
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 7)
    edges = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7)
    return np.clip(q.astype(np.float32) * (edges / 255.0)[..., None], 0, 255), m


def fx_glitch(img, m, s, c, code):
    dx = int(2 + s * 30)
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], -dx, axis=1)
    out[..., 2] = np.roll(img[..., 2], dx, axis=1)
    h = img.shape[0]
    rng = np.random.default_rng()
    for _ in range(int(2 + s * 8)):
        y = int(rng.integers(0, max(1, h - 8)))
        hh = int(rng.integers(3, 3 + int(s * 25) + 2))
        out[y:y + hh] = np.roll(out[y:y + hh], int(rng.integers(-dx * 3, dx * 3 + 1)), axis=1)
    return out, m


def fx_fill(img, m, s, c, code):  # strength = opacity
    out = np.empty_like(img)
    out[:] = np.array(c, np.uint8)
    return out, m * s


def fx_inpaint(img, m, s, c, code):
    mb = ((m > 0.3).astype(np.uint8)) * 255
    mb = cv2.dilate(mb, _k(4 + s * 14))
    out = cv2.inpaint(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), mb, 5, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), cv2.GaussianBlur(mb.astype(np.float32) / 255, (0, 0), 1.5)


def fx_custom(img, m, s, c, code):
    if os.environ.get("ALLOW_CUSTOM_CODE") != "1":
        raise ValueError("Custom code band hai. Apne private Colab/Kaggle/Space mein ALLOW_CUSTOM_CODE=1 set karo.")
    ns = {"np": np, "cv2": cv2}
    exec(code or "", ns)  # personal use only!
    if "effect" not in ns:
        raise ValueError("Custom code mein `def effect(img, mask, strength, color): ...` hona chahiye.")
    out = np.asarray(ns["effect"](img.copy(), m.copy(), s, c))
    return out, m


EFFECT_FUNCS = {
    "Blur": fx_blur, "Pixelate": fx_pixelate, "Black & white": fx_bw, "Invert": fx_invert,
    "Sepia": fx_sepia, "Tint (color)": fx_tint, "Brightness (0.5 = normal)": fx_brightness,
    "Thermal": fx_thermal, "Outline": fx_outline, "Glow": fx_glow, "Neon edges": fx_neon,
    "Cartoon": fx_cartoon, "RGB glitch": fx_glitch, "Solid color fill": fx_fill,
    "Remove (inpaint)": fx_inpaint, "Custom code": fx_custom,
}
EFFECT_NAMES = ["None"] + list(EFFECT_FUNCS)

CUSTOM_EXAMPLE = '''def effect(img, mask, strength, color):
    # img: uint8 RGB (H,W,3) | mask: float32 (H,W) 0..1 | strength: 0..1 | color: (r,g,b)
    out = img.astype(np.float32)
    out[..., 0] = np.clip(out[..., 0] * (1 + strength * 2), 0, 255)  # push red
    return out.astype(np.uint8)
'''


def apply_layers(frame, layers, softness=2.0, code=""):
    """layers: [(mask HxW bool/float, effect_name, strength, (r,g,b)), ...] applied in order."""
    cur = frame.astype(np.float32)
    for mask, name, s, color in layers:
        if name == "None" or name not in EFFECT_FUNCS:
            continue
        mf = soften(mask, softness)
        out, alpha = EFFECT_FUNCS[name](np.clip(cur, 0, 255).astype(np.uint8), mf, float(s), color, code)
        a = np.clip(np.asarray(alpha, np.float32), 0, 1)[..., None]
        cur = cur * (1 - a) + np.asarray(out, np.float32) * a
    return np.clip(cur, 0, 255).astype(np.uint8)
