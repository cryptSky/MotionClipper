# -*- mode: python -*-

block_cipher = None


a = Analysis(['MotionClipper_v1.py'],
             pathex=['c:\\Users\\Kryvol\\Anaconda3\\envs\\tensorflow\\Lib\\site-packages\\PyQt6\\Qt\\bin\\', 'D:\\Projects\\Deep Learning\\Prokofolio\\MotionClipper'],
             binaries=[('c:\\Users\\Kryvol\\Anaconda3\\envs\\tensorflow\\Library\\bin\\opencv_ffmpeg330_64.dll', '.')],
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
          [],
          exclude_binaries=True,
          name='MotionClipper_v1',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='MotionClipper_v1')
