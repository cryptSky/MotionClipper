# Build Intel (`x86_64`) one-file macOS executable on Apple Silicon

## Goal
Build a **PyInstaller one-file executable** for **Intel Macs** while working on an **Apple Silicon Mac**.

## Important
Running Terminal under **Rosetta** is **not enough**.
Your **Python environment itself must be `x86_64`**.

Check it with:

```bash
python -c "import platform; print(platform.machine())"
```

It must print:

```bash
x86_64
```

---

## Steps

### 1. Start a Rosetta shell
```bash
arch -x86_64 /bin/bash
arch
```
Expected output:
```bash
i386
```

### 2. Load Conda in the current shell
```bash
source /Users/user948776/miniforge3/etc/profile.d/conda.sh
```

### 3. Create an Intel Conda environment
```bash
export CONDA_SUBDIR=osx-64
conda create -n mc_x86 python=3.10 -y
conda activate mc_x86
```

### 4. Verify Python architecture
```bash
python -c "import platform; print(platform.machine())"
```
Expected output:
```bash
x86_64
```

### 5. Install dependencies
```bash
pip install -r requirements.txt
```

### 6. Build one-file executable
```bash
pyinstaller \
  --onefile \
  --windowed \
  --noconfirm \
  --paths /Users/user948776/miniforge3/envs/mc_x86/lib/python3.10/site-packages/PyQt5/Qt/bin \
  --add-binary "/Users/user948776/miniforge3/envs/mc_x86/lib/python3.10/site-packages/cv2/.dylibs:." \
  MotionClipper.py
```

---

## Verify the output
Check the produced binary:

```bash
file dist/MotionClipper
```

Expected output should include:

```bash
x86_64
```

---

## Notes
- If `python -c "import platform; print(platform.machine())"` prints `arm64`, the build will be **Apple Silicon only**.
- `arch` showing `i386` only means the shell is under Rosetta.
- The final architecture follows the **Python interpreter**, not just the shell.

