# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['MotionClipper.py'],
             pathex=['c:\\Users\\Kryvol\\Anaconda3\\envs\\tensorflow\\Lib\\site-packages\\PyQt5\\Qt\\bin\\', 'D:\\Projects\\Deep Learning\\Prokofolio\\MotionClipper'],
             binaries=[('c:\\Users\\Kryvol\\Anaconda3\\envs\\tensorflow\\Lib\\site-packages\\cv2\\opencv_ffmpeg342_64.dll', '.')],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='MotionClipper',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False )
