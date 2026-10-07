"""SAM 3 backend (torch / transformers imported lazily).

Every detect/track function returns "items": [(label, [packed_mask_or_None per frame]), ...]
"""
import os

import numpy as np

from effects import pack_mask

MODEL_ID = "facebook/sam3"           # gated model: request access once on HF
TOKEN = os.environ.get("HF_TOKEN")
SAM2_ID = os.environ.get("SAM2_ID", "facebook/sam2.1-hiera-large")  # or ...-small / ...-tiny (faster)
KEEP_ALL = False                     # True on ZeroGPU (models preloaded at startup)
_cache = {}
_sam3_ok = False
_last_check = 0.0


def sam3_available():
    """True once your HF token has been granted access to facebook/sam3."""
    global _sam3_ok, _last_check
    if _sam3_ok:
        return True
    import time
    if time.time() - _last_check < 30:
        return False
    _last_check = time.time()
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(MODEL_ID, "config.json", token=TOKEN)
        _sam3_ok = True
    except Exception:
        _sam3_ok = False
    return _sam3_ok


def version():
    """SAM_VERSION env: auto (default) | 3 | 2.1.  auto = SAM 3 if access granted, else SAM 2.1."""
    v = os.environ.get("SAM_VERSION", "auto").lower()
    if v in ("3", "sam3"):
        return "3"
    if v in ("2", "2.1", "sam2"):
        return "2.1"
    return "3" if sam3_available() else "2.1"


def backend_text():
    v = version()
    if v == "3":
        return "🟢 **SAM 3** active: text prompts + click, image + video."
    return ("🟡 **SAM 2.1** active (SAM 3 access abhi nahi mila): sirf **click** se object chuno, image + video. "
            "Text prompt (Find objects) SAM 3 approve hone ke baad apne aap chalu ho jayega.")


def need_sam3():
    if version() != "3":
        raise RuntimeError("SAM3_REQUIRED")


def _torch():
    import torch
    return torch


def device():
    t = _torch()
    return "cuda" if t.cuda.is_available() else "cpu"


def vid_dtype():
    t = _torch()
    if t.cuda.is_available() and t.cuda.is_bf16_supported():
        return t.bfloat16
    return t.float32  # T4 / P100 / CPU


def load(kind):
    """kind: img | trk | vid | trkvid  (img/vid = text prompts, SAM 3 only)"""
    t = _torch()
    ver = version()
    key = f"{ver}:{kind}"
    if key in _cache:
        return _cache[key]
    if not KEEP_ALL:  # keep VRAM low on small GPUs: only one model at a time
        _cache.clear()
        if t.cuda.is_available():
            t.cuda.empty_cache()
    dev, dt = device(), vid_dtype()
    if ver == "2.1":
        from transformers import Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor
        if kind == "trk":
            obj = (Sam2Model.from_pretrained(SAM2_ID).to(dev), Sam2Processor.from_pretrained(SAM2_ID))
        elif kind == "trkvid":
            obj = (Sam2VideoModel.from_pretrained(SAM2_ID).to(dev, dtype=dt), Sam2VideoProcessor.from_pretrained(SAM2_ID))
        else:
            raise RuntimeError("SAM3_REQUIRED")
    else:
        from transformers import (
            Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor,
            Sam3VideoModel, Sam3VideoProcessor, Sam3TrackerVideoModel, Sam3TrackerVideoProcessor)
        kw = dict(token=TOKEN)
        if kind == "img":
            obj = (Sam3Model.from_pretrained(MODEL_ID, **kw).to(dev), Sam3Processor.from_pretrained(MODEL_ID, **kw))
        elif kind == "trk":
            obj = (Sam3TrackerModel.from_pretrained(MODEL_ID, **kw).to(dev), Sam3TrackerProcessor.from_pretrained(MODEL_ID, **kw))
        elif kind == "vid":
            obj = (Sam3VideoModel.from_pretrained(MODEL_ID, **kw).to(dev, dtype=dt), Sam3VideoProcessor.from_pretrained(MODEL_ID, **kw))
        else:
            obj = (Sam3TrackerVideoModel.from_pretrained(MODEL_ID, **kw).to(dev, dtype=dt), Sam3TrackerVideoProcessor.from_pretrained(MODEL_ID, **kw))
    _cache[key] = obj
    return obj


def preload():
    global KEEP_ALL
    KEEP_ALL = True
    for k in (("img", "trk", "vid", "trkvid") if version() == "3" else ("trk", "trkvid")):
        load(k)


# --------------------------------------------------------------- text, image
def detect_image(frame, prompts, thresh, split):
    need_sam3()
    from PIL import Image
    t = _torch()
    model, proc = load("img")
    dev = device()
    pil = Image.fromarray(frame)
    items = []
    for p in prompts:
        inp = proc(images=pil, text=p, return_tensors="pt").to(dev)
        with t.no_grad():
            out = model(**inp)
        res = proc.post_process_instance_segmentation(
            out, threshold=thresh, mask_threshold=0.5,
            target_sizes=inp.get("original_sizes").tolist())[0]
        ms = res["masks"].cpu().numpy().astype(bool)
        if len(ms) == 0:
            continue
        if split:
            items += [(f"{p} #{i + 1}", [pack_mask(m)]) for i, m in enumerate(ms)]
        else:
            items.append((p, [pack_mask(np.any(ms, axis=0))]))
    return items


# --------------------------------------------------------------- text, video
def detect_video(frames, prompts, split, progress=None):
    need_sam3()
    t = _torch()
    model, proc = load("vid")
    dev, dt = device(), vid_dtype()
    T = len(frames)
    sess = proc.init_video_session(video=list(frames), inference_device=dev, dtype=dt)
    sess = proc.add_text_prompt(inference_session=sess, text=list(prompts))
    per_obj, label_of = {}, {}
    with t.no_grad():
        for mo in model.propagate_in_video_iterator(inference_session=sess, max_frame_num_to_track=T):
            post = proc.postprocess_outputs(sess, mo)
            f = int(mo.frame_idx)
            for pr, oids in post["prompt_to_obj_ids"].items():
                for o in oids:
                    label_of[int(o)] = pr
            ms = post["masks"].cpu().numpy().astype(bool)
            for o, m in zip(post["object_ids"].tolist(), ms):
                per_obj.setdefault(int(o), [None] * T)[f] = m
            if progress:
                progress((f + 1) / T)
    items = []
    if split:
        cnt = {}
        for o, ms in sorted(per_obj.items()):
            lb = label_of.get(o, "object")
            cnt[lb] = cnt.get(lb, 0) + 1
            items.append((f"{lb} #{cnt[lb]}", [pack_mask(m) for m in ms]))
    else:
        groups = {}
        for o, ms in per_obj.items():
            groups.setdefault(label_of.get(o, "object"), []).append(ms)
        for lb, lst in groups.items():
            merged = []
            for f in range(T):
                cur = [x[f] for x in lst if x[f] is not None]
                merged.append(pack_mask(np.any(cur, axis=0)) if cur else None)
            items.append((lb, merged))
    return items


# ------------------------------------------------------------- clicks, image
def click_image(frame, groups):
    from PIL import Image
    t = _torch()
    model, proc = load("trk")
    dev = device()
    pil = Image.fromarray(frame)
    items = []
    for gi, pts in enumerate(groups, 1):
        pts = [[float(x), float(y)] for x, y in pts]
        inp = proc(images=pil, input_points=[[pts]], input_labels=[[[1] * len(pts)]],
                   return_tensors="pt").to(dev)
        with t.no_grad():
            out = model(**inp, multimask_output=False)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"], binarize=True)[0]
        m = masks.reshape(-1, *masks.shape[-2:])[0].numpy().astype(bool)
        items.append((f"click {gi}", [pack_mask(m)]))
    return items


# ------------------------------------------------------------- clicks, video
def click_video(frames, groups, frame_idx, progress=None):
    t = _torch()
    model, proc = load("trkvid")
    dev, dt = device(), vid_dtype()
    T = len(frames)
    H, W = frames[0].shape[:2]
    sess = proc.init_video_session(video=list(frames), inference_device=dev, dtype=dt)
    for k, pts in enumerate(groups, 1):
        pts = [[float(x), float(y)] for x, y in pts]
        proc.add_inputs_to_inference_session(
            inference_session=sess, frame_idx=frame_idx, obj_ids=k,
            input_points=[[pts]], input_labels=[[[1] * len(pts)]])
    store = {k: [None] * T for k in range(1, len(groups) + 1)}

    def run(reverse):
        with t.no_grad():
            for out in model.propagate_in_video_iterator(
                    inference_session=sess, start_frame_idx=frame_idx, reverse=reverse):
                res = proc.post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=False)[0]
                res = res.reshape(res.shape[0], *res.shape[-2:])
                for i, o in enumerate(out.object_ids):
                    store[int(o)][int(out.frame_idx)] = (res[i] > 0).cpu().numpy()
                if progress:
                    progress(min(1.0, (int(out.frame_idx) + 1) / T))

    run(False)
    if frame_idx > 0:
        try:
            run(True)
        except Exception as e:  # reverse tracking is a bonus
            print("reverse tracking failed:", e)
    return [(f"click {k}", [pack_mask(m) for m in ms]) for k, ms in store.items()]
