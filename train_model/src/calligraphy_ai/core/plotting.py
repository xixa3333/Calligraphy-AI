from pathlib import Path

import matplotlib
from matplotlib import font_manager


def configure_chinese_font():
    """Configure a bundled Windows CJK font before creating figures."""
    candidates = (
        Path("C:/Windows/Fonts/msjh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/mingliu.ttc"),
    )
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            matplotlib.rcParams["font.family"] = family
            matplotlib.rcParams["font.sans-serif"] = [family]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return family
    raise RuntimeError("找不到可用的中文字型。")
