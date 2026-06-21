# Reproduction Log: Generating & Visualizing Example Grasps

End-to-end steps used to install Dexonomy, synthesize example grasps for the
**Allegro** hand, and visualize them. Verified on this machine:

- Ubuntu, NVIDIA **RTX 5090** (Blackwell, sm_120), driver **575.64.03 / CUDA 12.9**
- conda (miniforge) at `/home/chenyu/miniforge3`
- A single GPU → all GPU ops use `init_gpu=[0]`

> Result of this run: **1000 init → 832 refined (83%) → 602 validated grasps (72%)** across 19/20 objects, using the `fingertip_small` template.

---

## 0. Prerequisites already present
- GitHub SSH access (needed for `git submodule`, which uses `git@github.com` URLs).
- `git-lfs` installed (not actually needed here — assets come from HuggingFace as a tarball).
- Network access to `huggingface.co`.

---

## 1. Submodules

Only `mujoco_menagerie` is required (it provides the Allegro/Shadow/Leap mesh
assets referenced by the hand XMLs). `BODex` is **optional** — it is only used
for collision-free trajectory synthesis, and `skip_traj: True` is the default in
`dexonomy/config/base.yaml`, so we skip it.

```bash
cd /home/chenyu/workspace/grasp_ws/Dexonomy
git submodule update --init --progress third_party/mujoco_menagerie
# verify allegro meshes exist:
ls third_party/mujoco_menagerie/wonik_allegro/assets   # base_link.stl, link_*.stl, ...
```

---

## 2. Conda environment + install

```bash
source /home/chenyu/miniforge3/etc/profile.d/conda.sh
conda create -n dexonomy python=3.10 -y
conda activate dexonomy
pip install -e .
```

### 2a. Fix torch for the RTX 5090 (IMPORTANT)
`pip install -e .` pulls a torch built for **CUDA 13.0**, which is too new for the
installed driver (CUDA 12.9). Symptom:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12090)
torch.cuda.is_available()  ->  False
```

Reinstall a CUDA-12.8 build of torch (supports Blackwell sm_120 **and** works with
the 12.9 driver):

```bash
pip install --force-reinstall "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
# -> installs torch 2.11.0+cu128
```

Verify the GPU is usable:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); x=torch.randn(1000,1000,device='cuda'); print('matmul ok', float((x@x).sum()) is not None)"
# torch 2.11.0+cu128 True NVIDIA GeForce RTX 5090
# matmul ok True
```

---

## 3. Object assets (DGN_5k)

Download the pre-processed assets from HuggingFace (1.55 GB) and extract into
`assets/object/` (the tarball already contains the `DGN_5k/` top-level folder).

```bash
mkdir -p assets/object && cd assets/object
curl -L -C - -o DGN_5k_processed.tar.gz \
  "https://huggingface.co/datasets/JiayiChenPKU/Dexonomy/resolve/main/DGN_5k_processed.tar.gz?download=true"
gzip -t DGN_5k_processed.tar.gz          # integrity check
tar xzf DGN_5k_processed.tar.gz
cd ../..
```

Expected layout (`assets/object/DGN_5k/{processed_data,scene_cfg,valid_split}`):
- `processed_data/` → 5751 object folders
- `scene_cfg/`      → 5697 folders (the `init` op globs `scene_cfg/**/floating/*.npy`)
- `valid_split/`    → `all.json`, `train.json`, `test.json`

---

## 4. Prepare grasp templates (`op=tmpl`)

Converts the human annotations in `assets/hand/allegro/raw_anno/*.yaml` into
valid `.npy` templates.

```bash
conda activate dexonomy
dexrun op=tmpl hand=allegro
# -> "Find 11 annotations" / "Finish template conversion"
# output: assets/hand/allegro/init_tmpl/*.npy  (11 templates incl. fingertip_small)
```

`dexrun` = `python -m dexonomy.main`; `dexsyn` = `python -m dexonomy.script`.

---

## 5. Synthesize grasps (`dexsyn` = init → grasp → eval)

`dexsyn` runs three ops as parallel polling processes that terminate naturally:
- `init` (GPU): sample/optimize object pose against a template
- `grasp` (CPU/MuJoCo): refine palm pose + qpos
- `eval` (CPU/MuJoCo): physically validate

Single GPU, one template, small `n_cfg=20` for a quick example run:

```bash
dexsyn hand=allegro exp_name=first_try 'tmpl_name=[fingertip_small]' 'init_gpu=[0]' +init.object.n_cfg=20
```

Notes:
- Do **not** export `MUJOCO_GL=egl` for the core pipeline — it is CPU-only for
  grasp/eval and the egl backend conflicts with pyrender (see §6).
- `skip_traj: True` (default) → BODex is not needed.

Check results:

```bash
dexrun op=stat hand=allegro exp_name=first_try
# fingertip_small | Init | SynGrasp  | EvalGrasp  |
# Grasp:          | 1000 | 832,83%   | 602,72%    |
# Object:         | 20   | 19,95%    | 19,100%    |
```

Output directories under `output/first_try_allegro/`:
- `init_data/`  — initializations
- `grasp_data/` — refined grasps
- `succ_grasp/` — **602 validated grasp `.npy` files** (nested `…/floating/scaleXXX/*_grasp.npy`)
- `log/`        — per-op logs

---

## 6. Visualize

### 6a. Built-in formats
```bash
# All grasps in one USD (open with usdview):
dexrun op=vusd hand=allegro exp_name=first_try op.n_max=20 op.data=succ_grasp op.succ=True
# -> output/first_try_allegro/vis_usd/usd/fingertip_small.usd

# Per-grasp OBJ meshes (hand + object). Run WITHOUT MUJOCO_GL=egl:
dexrun op=v3d hand=allegro exp_name=first_try op.data=succ_grasp op.n_max=5 \
  debug_name=core_jar_3e2dd918360309554b3c42e318f3affc
# -> output/first_try_allegro/vis_3d/.../<id>_grasp_hand.obj  +  <id>_grasp_obj.obj
```

### 6b. Headless PNG renders
`vusd` needs a GUI (`usdview`); the `.obj` pairs can be rendered headlessly.
Install pyrender and use the helper script.

> ⚠️ Installing `pyrender` downgrades PyOpenGL 3.1.10 → 3.1.0, which **breaks
> MuJoCo's `MUJOCO_GL=egl`** (`EGLDeviceEXT` AttributeError). That only affects
> MuJoCo debug-render GIFs — synthesis and `op=v3d` are unaffected because they
> don't render on GPU. So: export OBJs first (no egl), then render with pyrender.

```bash
pip install pyrender   # PyOpenGL 3.1.0; pyrender renders via PYOPENGL_PLATFORM=egl

# Render a directory of *_hand.obj / *_obj.obj pairs to 4-angle PNG grids:
python output/render_grasps.py \
  "output/first_try_allegro/vis_3d/succ_grasp/fingertip_small/core_jar_.../floating/scale008" \
  "output/renders"
```

The script `output/render_grasps.py` (committed in this run) loads each
`*_hand.obj` + matching `*_obj.obj`, colors the hand blue / object grey, and
renders 4 azimuth views into one PNG per grasp under `output/renders/`.

---

## 7. Quick reference — re-run later

```bash
source /home/chenyu/miniforge3/etc/profile.d/conda.sh && conda activate dexonomy
cd /home/chenyu/workspace/grasp_ws/Dexonomy

dexrun op=tmpl hand=allegro
dexsyn hand=allegro exp_name=first_try 'tmpl_name=[fingertip_small]' 'init_gpu=[0]' +init.object.n_cfg=20
dexrun op=stat hand=allegro exp_name=first_try
dexrun op=vusd hand=allegro exp_name=first_try op.n_max=20 op.data=succ_grasp op.succ=True
```

Other grasp types available (from `assets/hand/allegro/raw_anno/`):
`1_Large_Diameter`, `4_Adducted_Thumb`, `6_Prismatic_4_Finger`, `10_Power_Disk`,
`12_Precision_Disk`, `16_Lateral`, `18_Extension_Type`, `fingertip_{small,mid,large}`.
Pass several with multiple GPUs, e.g.
`'tmpl_name=[fingertip_small,fingertip_mid]' 'init_gpu=[0,1]'`.
