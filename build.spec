# -*- mode: python ; coding: utf-8 -*-


version = "0.0.1"

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[('logo.ico', 'icons'), ('logo.png', 'icons')],
    hiddenimports=[
        'matplotlib.backends.backend_qtagg',
        'scipy.special._ufuncs',
        'scipy.integrate',
        'scipy.optimize',
        'scipy.signal',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=f"Baldur_SFC_Analyzer_{version}",
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
    icon='logo.ico',
)
