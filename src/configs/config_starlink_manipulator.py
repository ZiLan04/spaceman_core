
"""
Configuration for starlink_space_manipulator_merged URDF model.
"""
from pathlib import Path
import genesis as gs
import torch
to_posix = lambda p: p.as_posix()

_current_file_path = Path(__file__).resolve().parent
_ROOT_PATH = _current_file_path.parent.parent
_ASSET_PATH = _ROOT_PATH / 'src' / 'assets'

_STARLINK_MANIPULATOR_URDF = (
    _ASSET_PATH / 'urdf' / 'starlink_combine_starlink_space_manipulator_merged.SLDASM9' 
    / 'starlink_combine_starlink_space_manipulator_merged.SLDASM9.urdf'
)

FRANKA_S_Q_CONFIG = {
    "initial_dofs": [0.14, -0.55, -1.36, 0.81, 1.31, -0.52, 0.0, 0.0],
    "control": {
        "kp": [4500, 4500, 3500, 3500, 2000, 2000, 100, 100],
        "kv": [450, 450, 350, 350, 200, 200, 10, 10],
        "force_range_min": [-87, -87, -87, -87, -12, -12, -100, -100],
        "force_range_max": [87, 87, 87, 87, 12, 12, 100, 100],
    }
}

IK_PARAMS = {
    "smooth_factor": 0.3,
    "max_joint_change": 0.05,
}

FRANKA_S_Q_PARAMS = {
    "name": "franka_merge",
    "end_effector": "starlink_space_manipulator_merged.SLDASM9_starlink__space_manipulator_2F-Body_Link",
    "base": "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_base_link",
    "pid_apply_link": "starlink_base_star_link",
    "config": FRANKA_S_Q_CONFIG,
    "joints": (
        # "root_joint",
        # "qf_space_manipulator_Shoulder_link_1_yaw",
        # "qf_space_manipulator_Upper_arm_Link_1_roll",
        # "qf_space_manipulator_Mid_arm_Link_1_roll",
        # "qf_space_manipulator_Upper_wrist_Link_1_yaw",
        # "qf_space_manipulator_Upper_wrist_Link_2_roll",
        # "qf_space_manipulator_2F-Body_Link_pitch", 
        # # hand
        # "qf_space_manipulator_Finger1_2_Link_roll",
        # "qf_space_manipulator_Finger3_2_Link_roll", # all gripper joints mimic this
        # "qf_space_manipulator_Finger3_1_Link_roll",
        # "qf_space_manipulator_Finger1_1_Link_roll", # the only positive mutiplier
        # "qf_space_manipulator_Finger4_2_Link_roll",
        # "qf_space_manipulator_Finger4_1_Link_roll",
        #"root_joint",
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Shoulder_link_1_yaw",
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Upper_arm_Link_1_roll",
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Mid_arm_Link_1_roll",
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Upper_wrist_Link_1_yaw",
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Upper_wrist_Link_2_roll",
        "starlink_space_manipulator_merged.SLDASM9_starlink__space_manipulator_2F-Body_Link_yaw", 
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger1_2_Link_roll",
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger2_1_Link_roll",
       
    ),
    "motor": 6,
    "finger": 8,
    "gripper_revolute": [1, 1],
    "gripper_force_scale": 1.0,
    
    "finger_open": [0, 0],
    "finger_close": [0.4, 0.4],
    "ik_params": IK_PARAMS,
    "path": _STARLINK_MANIPULATOR_URDF,
}

# ---------------------------------------------------------------------------
# 自由漂浮基座的 6 自由度 PID：x/y/z 平动 + roll/pitch/yaw 姿态。
#
# 被控对象是重力补偿后的刚体，没有摩擦也没有气阻，所以每个通道都是纯双积分器：
#     平动  m * x'' = F        转动  I * theta'' = tau
# 加上 PD 之后就是质量-弹簧-阻尼系统：
#     omega_n = sqrt(Kp / m)                 # Kp 决定响应快慢（周期 = 2*pi/omega_n）
#     zeta    = Kd / (2 * sqrt(Kp * m))      # Kd 决定阻尼：zeta<1 会来回荡、冲过头
# 太空里唯一的阻尼来源就是 D 项，所以 D 必须按目标 zeta 反算，不能凭手感给；
# 而且同一组增益换到不同质量的星体上表现完全不同（omega_n 与 sqrt(m) 成反比），
# 所以下面每个数字都是连着实测的 m / I 一起算出来的。
#
# stage-1 场景实测（torque impulse 标定，src/test/stage1_pid_long_range_test.py
# 的 measure_base_inertia 也会在启动时复测）：
#     基座质量    m      = 516.7 kg
#     I_roll             = 215   kg*m^2
#     I_pitch            = 1177  kg*m^2
#     I_yaw              = 1386  kg*m^2
#     120 N 平移力作用在 root 连杆原点（不是质心），会带出约 14 N*m 扰动力矩
#
# dt 必须等于 PID 两次调用之间的真实时间：它只用在
#     integral += error * dt
#     D = Kd * (error - prev_error) / dt
# 两处，作用是把"两次调用之间的误差变化"换算成变化率。stage-1 的控制步是
# base_control_decimation=10 个物理步，每步 gsim.step() 前进 3 ms，所以
# 控制器实际每 3 ms 被调用一次。若 dt 仍写 0.01，等效阻尼会变成
# Kd * (0.003/0.01) = 0.3 * Kd（只有名义值的 1/3），I 项则被放大 3.3 倍。
# 这里对齐到 3 ms，于是 Kd 数值本身就等于物理阻尼系数 C [N*s/m] 或 [N*m*s/rad]。
#
# 增益要同时满足两个条件，只按 zeta 反算是不够的：
#   (1) 阻尼比:   Kd = 2*zeta*omega_n*I
#   (2) D 项不能被削顶: Kd * omega_peak <= tau_max
#       （一旦 D 项饱和，环路就丢掉阻尼，会变成持续摆动/极限环）
# 一次转角 theta 的滑行段峰值角速度 omega_peak = sqrt(0.6 * tau_max/I * theta)，
# 于是约束 (2) 给出 Kd <= sqrt(tau_max*I/(0.6*theta))，再用 (1) 反推 omega_n。
# 本例 tau_max=400 N*m、I_pitch=1177、theta=0.325 rad：
#     Kd <= sqrt(400*1177/0.195) = 1553  ->  omega_n <= 0.73 rad/s
# 所以取 omega_n = 0.73 rad/s（周期 8.6 s）、zeta ~ 0.86，三个姿态轴用同一组值
# （按最大的 pitch 惯量算）。注意不要按各轴惯量分别算"等带宽"：
# roll 的 I 只有 215 kg*m^2，等带宽会把它算成 Kp=115，比 pitch 弱 5 倍，而平移
# 力带出的扰动力矩是 roll/pitch 一起扛的，roll 一软整个基座就会翻滚（实测 5 m
# 站位下 roll 从 -25 度晃到 +49 度直接失控）。统一取 pitch 的值，roll 因此变成
# omega_n=1.7 rad/s、zeta~2.0 的过阻尼轴，抗扰更好也更安全。
#     转动（三轴统一）: Kp = 627 N*m/rad, Kd = 1478 N*m*s/rad
#     平动: Kp = 800 N/m（omega_n = 1.24 rad/s，周期 5.1 s）
#           Kd = 600 N*s/m（zeta ~ 0.47；再大一些 D 项就会在滑行段削顶，
#           实测 1155 会与姿态通道耦合出 ±4 度的持续摆动）
# 实测对比：之前那组"只按 zeta=0.9 算"的激进增益（姿态 1695/2540）会让
# D 项在 omega>0.157 rad/s 时贴顶，基座就一直在 ±3~4 度来回晃、永远停不下来；
# 换成下面这组后，5 m 站位起飞能在 ~10.7 s 稳定停住（终态位置误差 1.5 mm、
# 姿态误差 0.30 度），判据里的"停止"条件也能满足。
#
# limits 是执行器能力上限。它有两个作用：
# 一是决定"最快能多快"（饱和后 P 增益失效，退化成 bang-bang，最短时间
#     t = 2*sqrt(2d/a_max) 或 2*sqrt(2*theta/alpha_max)）；
# 二是必须给 D 项留足够的线性区，否则阻尼项被削顶，环路会变成极限环来回抖：
#     tau > Kd * omega   才能保证阻尼不被削顶（omega 是滑行段的峰值角速度）。
# 按上面这组增益：滑行段 omega_peak ~ 0.14 rad/s，Kd = 2540
#     -> 需要 2540 * 0.14 ~ 360 N*m，所以取 400 N*m（原来 40 N*m 会把 D 项
#        全程削顶，实测就是到点后一直漂，永远停不下来）。
# 力上限 250 N 的依据：a_max = 0.48 m/s^2，5 m 站位走 3.9 m 约需 8 s，十几秒的
# episode 够用；同时动作限幅是 0.3 m，Kp * 0.3 = 240 N 略小于限幅，所以策略
# 下发的偏移在整段动作范围内都近似线性映射成推力，学起来更平滑。
# 若以后要把力矩调小，就得把姿态 omega_n 和 Kd 一起降下来（例如 tau=120 N*m
# 时 omega_n 大约只能取 0.5~0.7 rad/s），不能只砍限幅。
# ---------------------------------------------------------------------------
FRANKA_S_Q_PID = {
    "name": "franka_merge",
    "enable_pid": True,
    "P": [800, 800, 800, 627, 627, 627],
    "I": [0, 0, 0, 0, 0, 0],
    "D": [600, 600, 600, 1478, 1478, 1478],
    "setpoint": [0, 0, 0, 0, 0, 0],
    "dt": 3e-3,
    "limits": [
        250,        # x  [N]
        250,        # y  [N]
        250,        # z  [N]
        400,        # roll  [N*m]
        400,        # pitch [N*m]
        400         # yaw   [N*m]
    ]
}

FRANKA_S_Q_CAMERA = {
    "name": "franka",
    # Regular debug/recording camera. RL observations are controlled by
    # rl_obs.enabled below and do not require this to be True.
    "wrist_camera": False,
    "camera": {
        # Keep a small training resolution here because the current batched RL
        # camera implementation reuses this value directly.
        "res": (84, 84),
        "pos": (-1, -1, -1),
        "lookat": (0, 0, 0),
        "fov": 60,
        "GUI": False,
    },
    "enable_recording": False,  # 训练时先关闭录制，避免额外渲染负担
    "episodes_per_video": 1,
    "end_effector_link": "starlink_space_manipulator_merged.SLDASM9_starlink__space_manipulator_2F-Body_Link",
    "pos_offset": torch.tensor([0.05, -0.08, -0.07], dtype=torch.float32),
    "lookat_offset": torch.tensor([0.05, -1.0, -0.1], dtype=torch.float32),
    "up_offset": torch.tensor([0.15, -1.0, -1.0], dtype=torch.float32),
    "base_camera": {
        # Regular debug/recording base camera. RL base observations are
        # controlled by base_camera.rl_obs.enabled below.
        "enabled": False,
        "camera": {
            # Match the RL visual pipeline resolution first; we can increase it later
            # after the mounting pose is stable.
            "res": (84, 84),
            "pos": (-1, -1, -1),
            "lookat": (0, 0, 0),
            "fov": 82,
            "GUI": False,
        },
        "enable_recording": False,
        "episodes_per_video": 1,
        "mount_link": "starlink_base_star_link",
        # Camera mounted on the left-rear side of the satellite base, looking
        # toward the arm workspace in the right-front / forward-up direction.
        "pos_offset": torch.tensor([0.25, 0.5, 0.4], dtype=torch.float32),
        "lookat_offset": torch.tensor([-0.5, 0.55, 0.75], dtype=torch.float32),
        "up_offset": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        "rl_obs": {
            "enabled": True,
            "channels_first": True,
            "normalize": False,
            "enable_recording": True,
            "record_env_idx": 0,
            "record_fps": 30,
            "episodes_per_video": 10,
            "use_batch_renderer": True,
            "update_interval": 1,
        },
    },
    "rl_obs": {
        "enabled": True,
        # SB3 expects image tensors in (C, H, W) for MultiInputPolicy.
        "channels_first": True,
        # Keep uint8 in the env and normalize in the extractor by default.
        "normalize": False,
        "enable_recording": True,
        "record_env_idx": 0,
        "record_fps": 30,
        "episodes_per_video": 50,
        "use_batch_renderer": True,
        # Render every control step first; if training is slow, try 2 or 4.
        "update_interval": 1,
    },
}

def _make_starlink_manipulator():
    return {
        "morph": gs.morphs.URDF(
            file=to_posix(_STARLINK_MANIPULATOR_URDF),
            pos=(0.5, 0.0, 0.0),
            euler=(0.0, 0.0, 0.0),
            merge_fixed_links=False,
            fixed=False,
        ),
        "material": gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
        #"vis_mode": "collision",
    }

__all__ = [
    'FRANKA_S_Q_PARAMS',
    'FRANKA_S_Q_PID',
    'FRANKA_S_Q_CAMERA',
    '_make_starlink_manipulator',
]
