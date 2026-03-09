# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['MotionClipper.py'],
    pathex=['/Users/user948776/miniforge3/envs/mc_x86/lib/python3.10/site-packages/PyQt5/Qt/bin'],
    binaries=[('/Users/user948776/miniforge3/envs/mc_x86/lib/python3.10/site-packages/cv2/.dylibs', '.')],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MotionClipper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
app = BUNDLE(
    exe,
    name='MotionClipper.app',
    icon=None,
    bundle_identifier=None,
)
