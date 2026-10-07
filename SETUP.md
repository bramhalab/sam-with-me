# SETUP: My SAM Studio

## SAM 2.1 vs SAM 3 (automatic)
- App khud check karti hai ki aapke `HF_TOKEN` ko `facebook/sam3` ka access mila ya nahi.
  - **Access nahi mila** -> **SAM 2.1** (public, Apache 2.0): sirf **click** se object chuno, image + video dono. Text prompt band.
  - **Access mil gaya** -> **SAM 3** apne aap: text prompt + click dono. Kuch badalna nahi, bas app restart/refresh.
- Force karna ho: env `SAM_VERSION=2.1` ya `SAM_VERSION=3`. SAM 2.1 ka model: env `SAM2_ID` (default `facebook/sam2.1-hiera-large`, tez ke liye `facebook/sam2.1-hiera-small`).
- Page ke upar ek line dikhati hai ki abhi kaunsa SAM active hai.

Common first steps (once)
1. https://huggingface.co/facebook/sam3 -> log in -> accept terms (gated model, needed for every option below).
2. https://huggingface.co/settings/tokens -> New token -> *Write* (or Read for Colab/Kaggle only).

## A) Google Colab (free GPU, easiest)
1. Push this folder to GitHub (public repo) or keep the zip.
2. Open `notebooks/colab.ipynb` in Colab (File -> Upload notebook, or open from GitHub).
3. Runtime -> T4 GPU. Add Colab Secret `HF_TOKEN`. Edit `REPO`. Run all.
4. Open the printed `https://xxxx.gradio.live` link. Session ends after ~12 h idle/limit, link dies with it.

## B) Kaggle (free GPU, 30 h/week)
1. Kaggle -> Create -> New Notebook -> File -> Import Notebook -> `notebooks/kaggle.ipynb`.
2. Session options: GPU T4 x2 (or P100), Internet ON. Add-ons -> Secrets -> `HF_TOKEN`.
3. Edit `REPO`, Run all, open the `gradio.live` link.
   (Phone verification is required on Kaggle for GPU + internet.)

## C) Hugging Face Space + GitHub Actions (always-on URL)
1. HF -> New Space -> SDK **Gradio**, Private. Hardware:
   - CPU basic (free): works for single images only (slow). Video is too slow.
   - **ZeroGPU** (needs HF PRO): fast, app auto-detects it and preloads models.
   - Paid GPU (T4 etc.): works, billed per hour.
2. Space -> Settings -> Variables and secrets -> secret `HF_TOKEN`.
3. GitHub repo with these files. Settings -> Secrets and variables -> Actions:
   secret `HF_TOKEN`, variables `HF_USERNAME`, `HF_SPACE`.
4. `git push` to `main` -> Actions deploys to the Space -> build -> live.
   Do NOT set `ALLOW_CUSTOM_CODE` on a Space others can open (it runs Python code).

## D) Local PC (NVIDIA GPU)
    pip install -r requirements.txt
    set HF_TOKEN=...          (Windows)   |   export HF_TOKEN=...   (Linux/Mac)
    python app.py

## How to use the app
1. Load image or video.
2. Pick objects: type `football, person` -> Find objects, and/or open "Click" and click objects
   (video: choose the frame first; tracking runs both directions from that frame).
   Objects are numbered [1], [2] ... in the preview.
3. For each object (and the background) choose its own effect + strength + color.
4. Preview this frame -> Render final. Output mode can also give a green-screen
   cut-out or a black/white mask video of the selected objects.

## Limits / notes
- Max 8 objects per project. "Har instance alag object" splits e.g. every person.
- Remove (inpaint) is simple (OpenCV): fine for small objects, not magic.
- Effects are numpy/OpenCV "shader-like" filters, not GLSL. Custom code lets you write your own.
- Large videos: lower "Work size" and "Max frames" (SAM 3 video is memory hungry).
- Not tested on a real GPU here; if a model call errors, send the traceback.
