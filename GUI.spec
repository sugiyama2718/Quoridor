# -*- mode: python ; coding: utf-8 -*-


from kivy_deps import sdl2, glew  # 追記1 依存ファイルのインポート
block_cipher = None


a = Analysis(
    ['GUI.py'],
    pathex=[],
    binaries=[('State.cp310-win_amd64.pyd', '.')],
    datas=[],
    hiddenimports=['win32file', 'win32timezone'],  # 追記2 FileChooserを使用している場合に必要
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],  # 追記3 依存関係の設定
    name='GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(exe, Tree('.'),  # 追記4 以下行すべて
               a.binaries,
               a.zipfiles,
               a.datas,
               *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
               strip=False,
               upx=True,
               name='GUI')
