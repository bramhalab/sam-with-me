"""My SAM Studio: SAM 3 segmentation + per-object effects for images and videos."""
import os
import re
import shutil
import subprocess
import tempfile

import cv2
import gradio as gr
import numpy as np

import effects as fx
import sam as sam_backend

# ZeroGPU decorator if available (HF PRO); no-op everywhere else (Colab/Kaggle/CPU).
try:
    import spaces
    GPU = spaces.GPU
except Exception:
    def GPU(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda f: f

if os.environ.get("SPACES_ZERO_GPU"):  # ZeroGPU needs models loaded at startup
    sam_backend.preload()

MAX_OBJ = 8
PALETTE = [(255, 64, 64), (64, 160, 255), (255, 200, 0), (80, 220, 100),
           (200, 90, 255), (255, 130, 30), (0, 220, 220), (255, 90, 170)]
SLOT_COLORS = ["#00ff88", "#ff4040", "#40a0ff", "#ffc800", "#50dc64", "#c85aff", "#ff821e", "#00dcdc", "#ff5aaa"]
OUT_MODES = ["Effects applied", "Green screen (selected objects only)", "Black & white mask (selected objects only)"]


def new_state():
    return {"frames": [], "fps": 24.0, "shape": None, "objs": [], "pend": [[]], "t": 0,
            "src": None, "full_image": None}


def _need(S):
    if not S or not S.get("frames"):
        raise gr.Error("Pehle image ya video load karo (Step 1).")
    return S


# ------------------------------------------------------------------- loading
def _fit(img, max_side):
    h, w = img.shape[:2]
    sc = max_side / max(h, w)
    if sc < 1:
        img = cv2.resize(img, (int(round(w * sc)), int(round(h * sc))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


# --------------------------------------------------------------- GPU wrappers
@GPU(duration=90)
def gpu_text(frames, prompts, thresh, split):
    if len(frames) == 1:
        return sam_backend.detect_image(frames[0], prompts, thresh, split)
    return sam_backend.detect_video(frames, prompts, split)


@GPU(duration=120)
def gpu_text_video(frames, prompts, split):
    return sam_backend.detect_video(frames, prompts, split)


@GPU(duration=90)
def gpu_click(frames, groups, t):
    if len(frames) == 1:
        return sam_backend.click_image(frames[0], groups)
    return sam_backend.click_video(frames, groups, t)


# ------------------------------------------------------------------- preview
def preview(S, t=None):
    t = S["t"] if t is None else t
    img = S["frames"][t]
    over = img.astype(np.float32)
    masks = [fx.unpack_mask(o["m"][t], S["shape"]) for o in S["objs"]]
    for k, m in enumerate(masks):
        if m.any():
            over[m] = over[m] * 0.45 + np.array(PALETTE[k % 8], np.float32) * 0.55
    out = np.ascontiguousarray(over.astype(np.uint8))
    fs = max(0.5, min(out.shape[:2]) / 700)
    for k, m in enumerate(masks):
        if not m.any():
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 255), 2)
        ys, xs = np.nonzero(m)
        cv2.putText(out, str(k + 1), (int(xs.mean()) - 8, int(ys.mean()) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, (255, 255, 255), 4)
        cv2.putText(out, str(k + 1), (int(xs.mean()) - 8, int(ys.mean()) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, (0, 0, 0), 2)
    for gi, pts in enumerate(S["pend"]):
        col = PALETTE[(len(S["objs"]) + gi) % 8]
        for (x, y) in pts:
            cv2.circle(out, (int(x), int(y)), 9, (255, 255, 255), -1)
            cv2.circle(out, (int(x), int(y)), 6, col, -1)
    return out


def slots_update(S):
    n = len(S["objs"]) if S else 0
    loaded = bool(S and S.get("frames"))
    ups = []
    for i in range(MAX_OBJ + 1):
        if i == 0:
            vis, label = loaded, "Background (everything else): effect"
        else:
            vis = i <= n
            label = f"[{i}] {S['objs'][i - 1]['label']}: effect" if vis else "effect"
        ups += [gr.update(visible=vis), gr.update(label=label)]
    return ups


def status(S):
    if not S or not S.get("frames"):
        return "Kuch load nahi hua."
    T = len(S["frames"])
    h, w = S["shape"]
    names = ", ".join(f"[{i + 1}] {o['label']}" for i, o in enumerate(S["objs"])) or "koi object nahi"
    pend = sum(len(g) for g in S["pend"])
    return f"**{'Video' if T > 1 else 'Image'}** {w}x{h}, {T} frame(s) | Objects: {names} | Pending clicks: {pend}"


def full_update(S):
    return [S, preview(S), status(S)] + slots_update(S)


# ------------------------------------------------------------------ handlers
def on_load_image(img, max_side):
    if img is None:
        raise gr.Error("Image upload karo.")
    S = new_state()
    f = _fit(np.asarray(img)[..., :3], int(max_side))
    S["frames"], S["shape"] = [f], f.shape[:2]
    S["full_image"] = np.ascontiguousarray(np.asarray(img)[..., :3])
    return full_update(S) + [gr.Slider(0, 1, value=0, step=1, visible=False)]


def on_load_video(path, max_side, max_frames):
    if not path:
        raise gr.Error("Video upload karo.")
    cap = cv2.VideoCapture(path)
    S = new_state()
    S["fps"] = cap.get(cv2.CAP_PROP_FPS) or 24.0
    S["src"] = path
    frames = []
    while len(frames) < int(max_frames):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(_fit(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), int(max_side)))
    cap.release()
    if not frames:
        raise gr.Error("Video read nahi hua.")
    S["frames"], S["shape"] = frames, frames[0].shape[:2]
    return full_update(S) + [gr.Slider(0, len(frames) - 1, value=0, step=1, visible=len(frames) > 1)]


def on_frame(S, t):
    S = _need(S)
    S["t"] = int(t)
    return S, preview(S)


def _add_items(S, items):
    room = MAX_OBJ - len(S["objs"])
    area = lambda it: sum(int(np.unpackbits(p).sum()) for p in it[1] if p is not None)
    items = [it for it in items if any(p is not None for p in it[1])]
    items.sort(key=area, reverse=True)
    for lb, ms in items[:room]:
        S["objs"].append({"label": lb, "m": ms})
    return len(items), len(items[:room])


def on_detect(S, prompts, thresh, split, progress=gr.Progress()):
    S = _need(S)
    plist = [p.strip() for p in re.split(r"[,\n]", prompts or "") if p.strip()]
    if not plist:
        raise gr.Error("Text prompt likho, jaise: football, person")
    progress(0.05, desc="SAM chal raha hai (pehli baar model download hoga)...")
    try:
        if len(S["frames"]) > 1:
            items = gpu_text_video(S["frames"], plist, bool(split))
        else:
            items = gpu_text(S["frames"], plist, float(thresh), bool(split))
    except RuntimeError as e:
        if "SAM3_REQUIRED" in str(e):
            raise gr.Error("Text prompt ke liye SAM 3 chahiye (access abhi pending). Abhi 'B) Click se' use karo.")
        raise
    found, kept = _add_items(S, items)
    msg = status(S) + (f" | ⚠️ {found} mile, sirf {kept} rakhe (max {MAX_OBJ})" if kept < found else "")
    if found == 0:
        msg += " | ⚠️ Kuch detect nahi hua, prompt/threshold badlo."
    return full_update(S)[:2] + [msg] + slots_update(S)


def on_click(S, evt: gr.SelectData):
    S = _need(S)
    x, y = evt.index
    S["pend"][-1].append((int(x), int(y)))
    return S, preview(S), status(S)


def on_new_click_obj(S):
    S = _need(S)
    if S["pend"][-1]:
        S["pend"].append([])
    return S, preview(S), status(S)


def on_undo(S):
    S = _need(S)
    for g in reversed(S["pend"]):
        if g:
            g.pop()
            break
    while len(S["pend"]) > 1 and not S["pend"][-1] and not S["pend"][-2]:
        S["pend"].pop()
    return S, preview(S), status(S)


def on_clear_clicks(S):
    S = _need(S)
    S["pend"] = [[]]
    return S, preview(S), status(S)


def on_run_clicks(S, progress=gr.Progress()):
    S = _need(S)
    groups = [g for g in S["pend"] if g]
    if not groups:
        raise gr.Error("Pehle image par object ke upar click karo.")
    progress(0.05, desc="Tracking...")
    items = gpu_click(S["frames"], groups, int(S["t"]))
    base = sum(1 for o in S["objs"] if o["label"].startswith("click"))
    items = [(f"click {base + i + 1}", ms) for i, (_, ms) in enumerate(items)]
    _add_items(S, items)
    S["pend"] = [[]]
    return full_update(S)[:3] + slots_update(S)


def on_remove_last(S):
    S = _need(S)
    if S["objs"]:
        S["objs"].pop()
    return full_update(S)[:3] + slots_update(S)


def on_clear_objs(S):
    S = _need(S)
    S["objs"], S["pend"] = [], [[]]
    return full_update(S)[:3] + slots_update(S)


# -------------------------------------------------------------------- render
def _frames_iter(S, full):
    if full and S["src"]:
        cap = cv2.VideoCapture(S["src"])
        for _ in range(len(S["frames"])):
            ok, f = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        cap.release()
    elif full and S["full_image"] is not None:
        yield S["full_image"]
    else:
        yield from S["frames"]


def _render_frame(S, t, frame, cfgs, softness, mode, code):
    H, W = frame.shape[:2]
    ms = []
    for o in S["objs"]:
        m = fx.unpack_mask(o["m"][t], S["shape"]).astype(np.float32)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        ms.append(m)
    union = np.maximum.reduce(ms) if ms else np.zeros((H, W), np.float32)
    if mode == OUT_MODES[2]:
        g = (fx.soften(union, softness) * 255).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)
    layers = [(1 - union, *cfgs[0])] + [(m, *cfgs[k + 1]) for k, m in enumerate(ms)]
    res = fx.apply_layers(frame, layers, softness, code)
    if mode == OUT_MODES[1]:
        a = fx.soften(union, softness)[..., None]
        green = np.zeros_like(res)
        green[:] = (0, 255, 0)
        res = (green * (1 - a) + res * a).astype(np.uint8)
    return res


def _write_video(frames, fps, src):
    tmp = tempfile.mktemp(suffix=".mp4")
    vw = None
    for fr in frames:
        if vw is None:
            h, w = fr.shape[:2]
            vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    if vw is None:
        raise gr.Error("Koi frame render nahi hua.")
    vw.release()
    final = tmp.replace(".mp4", "_h264.mp4")
    try:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = shutil.which("ffmpeg")
        cmd = [exe, "-y", "-loglevel", "error", "-i", tmp]
        if src:
            cmd += ["-i", src, "-map", "0:v", "-map", "1:a?", "-shortest"]
        cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", final]
        subprocess.run(cmd, check=True)
        return final
    except Exception as e:
        print("ffmpeg re-encode failed:", e)
        return tmp


def _cfgs(d):
    out = []
    for i in range(MAX_OBJ + 1):
        out.append((d[slot_eff[i]], d[slot_str[i]], fx.parse_color(d[slot_col[i]])))
    return out


def _apply(d, only_frame, progress):
    S = _need(d[state])
    if not S["objs"]:
        raise gr.Error("Pehle kam se kam ek object select karo (Step 2).")
    cfgs, softness, mode = _cfgs(d), float(d[softness_s]), d[outmode]
    code, full = d[custom_code], bool(d[fullres])
    T = len(S["frames"])
    if only_frame or T == 1:
        t = S["t"]
        frame = next(iter(_frames_iter(S, full))) if T == 1 else None
        if frame is None:
            frame = S["frames"][t]
            if full and S["src"]:
                cap = cv2.VideoCapture(S["src"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, t)
                ok, f = cap.read()
                cap.release()
                if ok:
                    frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        return _render_frame(S, t, frame, cfgs, softness, mode, code), None

    def gen():
        for t, frame in enumerate(_frames_iter(S, full)):
            progress((t + 1) / T, desc=f"Frame {t + 1}/{T}")
            yield _render_frame(S, t, frame, cfgs, softness, mode, code)
    return None, _write_video(gen(), S["fps"], S["src"])


def on_preview(d, progress=gr.Progress()):
    img, _ = _apply(d, True, progress)
    return img, None


def on_render(d, progress=gr.Progress()):
    return _apply(d, False, progress)


# ------------------------------------------------------------------------ UI
with gr.Blocks(title="My SAM Studio") as demo:
    state = gr.State(None)
    gr.Markdown("# ✂️ My SAM Studio\nObjects select karo (text ya click), phir **har object par alag effect** lagao. Image + Video dono.")
    backend_md = gr.Markdown(sam_backend.backend_text())
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1) Load")
            with gr.Tabs():
                with gr.Tab("Image"):
                    in_img = gr.Image(type="numpy", label="Image", height=220)
                    b_img = gr.Button("Load image", variant="primary")
                with gr.Tab("Video"):
                    in_vid = gr.Video(label="Video", height=220)
                    b_vid = gr.Button("Load video", variant="primary")
            with gr.Row():
                max_side = gr.Slider(256, 1920, value=720, step=32, label="Work size (long side px)")
                max_frames = gr.Slider(8, 600, value=120, step=8, label="Max frames (video)")
            gr.Markdown("### 2) Objects chuno")
            with gr.Accordion("A) Text se (AI): SAM 3 chahiye", open=False):
                prompts = gr.Textbox(label="Prompts (comma se alag)", placeholder="football, person in blue shirt")
                with gr.Row():
                    thresh = gr.Slider(0.05, 0.95, value=0.45, step=0.05, label="Confidence (image)")
                    split = gr.Checkbox(value=False, label="Har instance alag object")
                b_detect = gr.Button("🔍 Find objects", variant="primary")
            with gr.Accordion("B) Click se (SAM 2.1 / SAM 3 dono)", open=True):
                gr.Markdown("Preview par object ke upar click karo (video mein pehle frame slider se frame chuno). Dusra object chahiye to **New object** dabao.")
                with gr.Row():
                    b_new = gr.Button("➕ New object")
                    b_undo = gr.Button("↩️ Undo click")
                    b_clr = gr.Button("🧹 Clear clicks")
                b_runclk = gr.Button("🎯 Segment / track clicked objects", variant="primary")
            with gr.Row():
                b_rm = gr.Button("🗑️ Last object hatao")
                b_clrall = gr.Button("🗑️ Sab objects hatao")
        with gr.Column(scale=2):
            prev = gr.Image(label="Preview (click yahin karo)", type="numpy", interactive=False, height=460)
            frame_s = gr.Slider(0, 1, value=0, step=1, label="Frame", visible=False)
            stat = gr.Markdown("Kuch load nahi hua.")
            gr.Markdown("### 3) Effects (har object ka alag)")
            slot_rows, slot_eff, slot_str, slot_col = [], [], [], []
            for i in range(MAX_OBJ + 1):
                with gr.Row(visible=False) as r:
                    e = gr.Dropdown(fx.EFFECT_NAMES, value="None", label="effect", scale=3)
                    s = gr.Slider(0, 1, value=0.6, step=0.05, label="Strength", scale=2)
                    c = gr.ColorPicker(value=SLOT_COLORS[i], label="Color", scale=1)
                slot_rows.append(r); slot_eff.append(e); slot_str.append(s); slot_col.append(c)
            with gr.Row():
                softness_s = gr.Slider(0, 12, value=1.5, step=0.5, label="Edge softness")
                outmode = gr.Dropdown(OUT_MODES, value=OUT_MODES[0], label="Output mode")
                fullres = gr.Checkbox(value=True, label="Original resolution export")
            with gr.Accordion("Custom code effect (advanced, sirf private use)", open=False):
                custom_code = gr.Code(value=fx.CUSTOM_EXAMPLE, language="python", label="def effect(img, mask, strength, color)")
                gr.Markdown("Effect dropdown mein **Custom code** chuno. `ALLOW_CUSTOM_CODE=1` env set hona chahiye (Colab/Kaggle notebook mein already set hai).")
            with gr.Row():
                b_prev = gr.Button("👁️ Preview this frame")
                b_render = gr.Button("🎬 Render final", variant="primary")
            out_img = gr.Image(label="Result (image / preview)", type="numpy", interactive=False)
            out_vid = gr.Video(label="Result video")

    slot_out = []
    for r_, e_ in zip(slot_rows, slot_eff):
        slot_out += [r_, e_]
    base_out = [state, prev, stat]
    demo.load(lambda: sam_backend.backend_text(), None, backend_md)

    b_img.click(on_load_image, [in_img, max_side], base_out + slot_out + [frame_s])
    b_vid.click(on_load_video, [in_vid, max_side, max_frames], base_out + slot_out + [frame_s])
    frame_s.release(on_frame, [state, frame_s], [state, prev])
    b_detect.click(on_detect, [state, prompts, thresh, split], [state, prev, stat] + slot_out)
    prev.select(on_click, [state], [state, prev, stat])
    b_new.click(on_new_click_obj, [state], [state, prev, stat])
    b_undo.click(on_undo, [state], [state, prev, stat])
    b_clr.click(on_clear_clicks, [state], [state, prev, stat])
    b_runclk.click(on_run_clicks, [state], [state, prev, stat] + slot_out)
    b_rm.click(on_remove_last, [state], [state, prev, stat] + slot_out)
    b_clrall.click(on_clear_objs, [state], [state, prev, stat] + slot_out)

    apply_inputs = {state, softness_s, outmode, fullres, custom_code, *slot_eff, *slot_str, *slot_col}
    b_prev.click(on_preview, apply_inputs, [out_img, out_vid])
    b_render.click(on_render, apply_inputs, [out_img, out_vid])

if __name__ == "__main__":
    demo.queue().launch(share=os.environ.get("SHARE") == "1")
