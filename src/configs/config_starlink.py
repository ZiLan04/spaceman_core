"""
Configuration for satellite URDF model.
"""
import sys
from pathlib import Path
import genesis as gs

current_file_path = Path(__file__).resolve().parent
sys.path.insert(0, str(current_file_path))

to_posix = lambda p: p.as_posix()

_ROOT_PATH = current_file_path.parent.parent
_ASSET_PATH = _ROOT_PATH / 'src' / 'assets'

_STARLINK_PATHS = {
    # 新模型：base 由 body + cylinder 两个 link 用 fixed joint 拼成，
    # 再挂上原 panel。详见 assets/urdf/starlink_new/README.md
    'urdf': _ASSET_PATH / 'urdf' / 'starlink_new' / 'urdf' / 'starlink.urdf',
}

STARLINK_PARAMS = {
    "name": "starlink",
    # base_star_body_link 是合并后的根 link；base_star_cylinder_link 与它共坐标系，
    # 所以下面的 grasp_offset_local 数值无需改动。
    "base": "base_star_body_link",
    "grasp_link": "base_star_body_link",
    "grasp_offset_local": [-0.33519137, 0.06895784, 0.22731882],
    "grasp_marker_offset_local": [0.0, 0.0, 0.0],
    "path": _STARLINK_PATHS['urdf'],
}

# 需要在 fixed-link 合并时保留的 link。
#
# Genesis 默认 merge_fixed_links=True，会把 base 的 body / cylinder 和 panel 合并成
# 一个刚体再整体做凸分解，圆筒的凹坑会被大凸包填平（新做的精细圆筒就白费了）。
# 把圆筒加进 links_to_keep，它会作为独立 link 单独做凸分解，凹坑得以保留
#（与 assets/urdf/cylinder/view.py 的用法一致）。
STARLINK_LINKS_TO_KEEP = ("base_star_cylinder_link",)

# 精细圆筒的凸分解参数（取自 assets/urdf/cylinder/view.py 的 CoacdOptions）。
#
# 只加 links_to_keep 还不够：默认 CoACD 阈值会把圆筒那一段合成 1 个实心凸包，孔又被
# 填平。配上这组参数后圆筒会被拆成几十个凸块、保留筒壁（实测约 60 块，碰撞总体积
# ≈ 真实筒壁体积）。代价是碰撞几何变重、接触计算更慢。
STARLINK_FINE_CYLINDER_COACD_KWARGS = {
    "threshold": 0.05,
    "max_convex_hull": -1,
    "resolution": 10000,
    "preprocess_mode": "auto",
}

STARLINK_PID = {
    "name": "starlink",
    "enable_pid": False,  # 控制是否启用PID控制器
    "P": [0, 0, 0, 0, 0, 0],
    "I": [0, 0, 0, 0, 0, 0],
    "D": [0, 0, 0, 0, 0, 0],
    "setpoint": [0, 0, 0, 0, 0, 0],
    "dt": 0.01,
    "limits": [
        None,        # x
        None,        # y
        None,        # z
        None,        # roll
        None,        # pitch
        None         # yaw
    ]
}

STARLINK_CAMERA = {
    "name": "starlink",
    "wrist_camera": False,
    "camera": {
        "res": (640, 480),
        "pos": (0.0, 0.0, 0.0),  # 将在机器人初始化时设置
        "lookat": (0.0, 0.0, 0.0),  # 将在机器人初始化时设置
        "fov": 40,
        "GUI": False,
    },
    "enable_recording": False  # 控制是否启用录制的flag
}

def make_starlink_morph(
    *,
    collision: bool = True,
    visualization: bool = True,
    keep_cylinder_link: bool = True,
    fine_cylinder_collision: bool = True,
    **overrides,
):
    """构造 starlink 目标的 URDF morph —— 全项目唯一的构造入口。

    任何需要覆写参数的地方（RL 训练、各 test 里切换 target collision 等）都应该调用
    本函数，这样 `links_to_keep` 之类的公共设置只要维护这一处，而不是每次建场景时
    各自重写一遍 morph。

    Args:
        collision: 是否启用物理碰撞（各 test / 训练通过它切换 target collision）。
        visualization: 是否生成可视几何。
        keep_cylinder_link: 是否把圆筒 link 排除在 fixed-link 合并之外，
            以便它独立做凸分解、保留凹坑（见 STARLINK_LINKS_TO_KEEP 的说明）。
        fine_cylinder_collision: 是否使用 view.py 那组 CoACD 参数，让圆筒拆成几十个
            凸块、保住筒壁。只在 keep_cylinder_link=True 时有意义。它会加重碰撞几何，
            训练/仿真变慢时可以置 False 换回默认（会退回 1 个实心凸包）。
        **overrides: 其余 gs.morphs.URDF 参数，直接覆盖默认值。
    """
    kwargs = {
        "file": to_posix(_STARLINK_PATHS['urdf']),
        "pos": (-2.0, 0.0, 0.5),
        "euler": (0.0, 0.0, 180.0),
        "scale": 7e-1,
        "fixed": False,
        "collision": bool(collision),
        "visualization": bool(visualization),
        "links_to_keep": list(STARLINK_LINKS_TO_KEEP) if keep_cylinder_link else [],
    }
    if keep_cylinder_link and fine_cylinder_collision:
        kwargs["convexify"] = True
        kwargs["decompose_object_error_threshold"] = 0.0
        kwargs["coacd_options"] = gs.options.CoacdOptions(
            **STARLINK_FINE_CYLINDER_COACD_KWARGS
        )
    kwargs.update(overrides)
    return gs.morphs.URDF(**kwargs)


def _make_starlink():
    return {
        "morph": make_starlink_morph(),
        "material": gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    }

__all__ = [
    'STARLINK_PARAMS',
    'STARLINK_PID',
    'STARLINK_CAMERA',
    'STARLINK_LINKS_TO_KEEP',
    'STARLINK_FINE_CYLINDER_COACD_KWARGS',
    'make_starlink_morph',
    '_make_starlink',
]
