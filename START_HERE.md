# START HERE — new laptop / USB copy

This repo is a **deepfake detector** (REAL vs FAKE), not a face-swap generator.

Work from a **local disk copy**, not the USB stick. USB + PyTorch is slow and venvs break across machines.

```text
USB  →  copy folder to Desktop/Documents  →  follow this file
```

**First run needs internet** (PyTorch wheels, npm packages, Hugging Face `google/vit-base-patch16-224`, MTCNN face weights). After that, offline use is usually fine.

---

## 0. What must be on the USB

`.gitignore` excludes the trained weights. If you only copy the git tree, **detection will fail**.

| Include | Skip (recreate on the new PC) |
|---------|--------------------------------|
| Whole `deepfake_detector` project | `deepfake_env/` (old Windows venv — do not reuse) |
| `data/models/best_model.pth` (**required**, ~330 MB) | `frontend/node_modules/` |
| `START_HERE.md`, `api/`, `frontend/`, `data/*.py` | `api/_jobs/`, `frontend/dist/` |
| Optional: `data/videos/` for your own tests | Huge `data/DFDC_Frames/` unless you need to retrain |

Confirm this file exists before unplugging:

```text
deepfake_detector/data/models/best_model.pth
```

---

## 1. Software on the new laptop

Install if missing:

- **Python 3.10–3.12** ([python.org](https://www.python.org/downloads/)) — tick **Add python.exe to PATH**
- **Node.js 18+** ([nodejs.org](https://nodejs.org/)) — needed for the web UI

Open **PowerShell**. Check:

```powershell
python --version
node --version
npm --version
```

---

## 2. Copy off the USB

```powershell
# Example: USB is E: — change the drive letter if needed
$usb = "E:\deepfake_detector"
$dest = "$env:USERPROFILE\Desktop\deepfake_detector"
Copy-Item -Path $usb -Destination $dest -Recurse -Force
cd $dest
```

If the folder is already on Desktop, only:

```powershell
cd $env:USERPROFILE\Desktop\deepfake_detector
```

---

## 3. Python environment (CPU vs GPU)

Create a **new** venv on this machine. Do not activate a venv copied from another PC.

```powershell
cd $env:USERPROFILE\Desktop\deepfake_detector
python -m venv deepfake_env
.\deepfake_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If `Activate.ps1` is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### A) No GPU / unknown GPU (safe default)

Install **CPU PyTorch first**, then the rest:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r api\requirements.txt
```

Detection still works on CPU. A short clip can take **several minutes**. The UI will show `device: cpu`.

### B) NVIDIA GPU (optional, faster)

Only if `nvidia-smi` works in PowerShell. Pick a CUDA build that matches the driver from [pytorch.org](https://pytorch.org/get-started/locally/), for example CUDA 12.1:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r api\requirements.txt
```

If GPU install fails, go back to **section A (CPU)**. The API already falls back to CPU when CUDA is missing.

---

## 4. Frontend packages

```powershell
cd $env:USERPROFILE\Desktop\deepfake_detector\frontend
npm install
```

---

## 5. Run the app (two terminals)

**Terminal 1 — API** (first start downloads ViT/MTCNN; can take several minutes):

```powershell
cd $env:USERPROFILE\Desktop\deepfake_detector
.\deepfake_env\Scripts\Activate.ps1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Wait until you see `Uvicorn running on http://127.0.0.1:8000`.

Check the model:

```powershell
# another PowerShell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

You want `"loaded": true`. `"device"` will be `cpu` or `cuda`.

**Terminal 2 — UI:**

```powershell
cd $env:USERPROFILE\Desktop\deepfake_detector\frontend
npm run dev
```

Open **http://localhost:5173**

Upload a video (MP4/WebM/MOV) or image → **Analyze authenticity**. Verdict is REAL or FAKE with confidence and the frames the model used.

---

## 6. Training scripts (optional)

Only if you are retraining, not for the web demo. Those scripts expect the process cwd to be the **parent** of `deepfake_detector`:

```powershell
cd $env:USERPROFILE\Desktop
.\deepfake_detector\deepfake_env\Scripts\Activate.ps1
python deepfake_detector\data\prepare\prepare_data.py
python deepfake_detector\data\train_vit2.py
```

Use **train_vit2.py** (VideoViT v2). It writes `deepfake_detector/data/models/best_model.pth`.

---

## 7. If something breaks

| Symptom | Fix |
|---------|-----|
| `No VideoViT checkpoint found` | Copy `data/models/best_model.pth` from the USB |
| `Cannot reach API on port 8000` | Start Terminal 1 first; keep it open |
| Health `"loaded": false` | First-run download failed — need internet, then restart API |
| `Activate.ps1` cannot run | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Out of memory on CPU | Use a short clip; close other apps |
| GPU OOM | Use the CPU torch install in section 3A |
| Very slow | Expected on CPU; 8 frames + MTCNN + ViT is heavy |
| npm / pip SSL or timeout | Connect to internet; retry |

The web stack does **not** change `train_vit.py` / `train_vit2.py`. It wraps VideoViT v2 only.
