import os
import math
import datetime
import torch
from typing import Dict, Any

# 
def parse_asset_config(config_dict, config_key):
    if config_key not in config_dict:
        print(f"Key '{config_key}' not found in assets dictionary")
        return None, None
    
    entity_config = config_dict[config_key]

    if hasattr(entity_config, '__iter__') and isinstance(entity_config, list):
        if len(entity_config) == 1:
            return entity_config[0], None
        elif len(entity_config) == 2:
            return entity_config[0], entity_config[1]
        else:
            print(f"Value for key '{config_key}' has {len(entity_config)} elements, expected 1 or 2")
            return None, None
    else:
        return entity_config, None

#
def convert_dict_to_tensors(nested_dict: Dict[str, Any], 
                            dtype: torch.dtype = torch.float32,
                            device: str = None) -> Dict[str, Any]:
    """
    将嵌套字典中的所有列表转换为PyTorch张量
    
    Args:
        nested_dict: 包含列表的嵌套字典
        device: 目标设备（如'cpu', 'cuda:0'），如果为None则使用默认设备
        dtype: 目标数据类型，默认为torch.float32
    
    Returns:
        转换后的字典，其中所有列表都变为PyTorch张量
    """
    if nested_dict is None:
        return None
    
    result = {}
    
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            # 递归处理嵌套字典
            result[key] = convert_dict_to_tensors(value, dtype, device)
        elif isinstance(value, (list, tuple)):
            # 转换列表或元组为PyTorch张量
            tensor = torch.tensor(value, dtype=dtype)
            if device is not None:
                tensor = tensor.to(device)
            result[key] = tensor
        elif isinstance(value, torch.Tensor):
            # 如果已经是张量，确保设备和数据类型正确
            tensor = value
            if dtype is not None and tensor.dtype != dtype:
                tensor = tensor.to(dtype)
            if device is not None and tensor.device != torch.device(device):
                tensor = tensor.to(device)
            result[key] = tensor
        else:
            # 保持其他类型不变（数字、字符串等）
            result[key] = value
    
    return result

#
def generate_filename(prefix="video", extension="mp4", folder_path=""):
    """
    生成带时间戳的文件名
    Args:
        prefix (str): 文件名前缀，默认为"video"
        extension (str): 文件扩展名，默认为"mp4"
        folder_path (str): 文件夹路径，如果提供会检查并创建目录
    Returns:
        str: 完整的文件路径
    """
    # 获取当前时间
    now = datetime.datetime.now()
    
    # 格式化时间字符串
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    # 组合文件名
    filename = f"{prefix}_{timestamp}.{extension}"
    
    # 如果提供了文件夹路径，确保目录存在并返回完整路径
    if folder_path:
        # 创建目录（如果不存在）
        os.makedirs(folder_path, exist_ok=True)
        return os.path.join(folder_path, filename)
    
    return filename

#
def angle_difference(target, current):
    """
    使用PyTorch计算角度差值, 处理-π到π的跨越问题
    Args:
        target: 目标角度张量
        current: 当前角度张量
    Returns:
        最短路径的角度差值张量
    """
    diff = target - current
    # Limit the difference in [-π, π]
    if isinstance(diff, torch.Tensor):
        # 使用PyTorch操作确保梯度可计算
        diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
    else:
        # 处理标量
        diff = ((diff + math.pi) % (2 * math.pi)) - math.pi
    return diff

def angular_velocity(current_qpos, prev_qpos, dt=None):
    """
    基于当前位置和上一次位置计算角速度
    Args:
        current_qpos: 当前关节位置张量
        prev_qpos: 上一次关节位置张量  
        dt: 持续时间
    Returns:
        目标角速度张量
    """
    if prev_qpos is None:
        return torch.zeros_like(current_qpos)
    
    if dt is None:
        dt = 0.1
    
    # 计算位置差值，处理角度跨越边界的情况
    angle_diff = angle_difference(current_qpos, prev_qpos)
    
    # 应用折扣系数计算目标角速度
    target_velocity = angle_diff / dt
    
    return target_velocity


#### Camera operation

def with_camera(cam_settings):
    """
    Check whether CAM_SETTINGS contains camera configuration
    Args:
        cam_settings (dict): camera configurations
    Returns:
        bool: return True if containing any camera configuration, otherwise return False
    """
    if not isinstance(cam_settings, dict):
        return False
    
    # Check all keys. Look for keys starting with "camera" (e.g., "camera", "camera0", "camera1")
    camera_keys = [key for key in cam_settings.keys() 
                  if isinstance(key, str) and key.startswith("camera")]
    
    # Check if recording is enabled
    recording_enabled = cam_settings.get("enable_recording", True)
    
    return len(camera_keys) > 0 and recording_enabled

def calculate_camera_pose(wrist_position, wrist_quaternion, 
                          pos_offset=[0.07, 0.0, -0.12], 
                          lookat_offset=[0.0, 0.0, 1.0],
                          up_offset=[1.0, 0.0, 0.0]):
    """
    使用pos、lookat和up参数设置手腕相机位姿
    
    参数:
        wrist_position: torch.Tensor, 手腕位置 [x, y, z]
        wrist_quaternion: torch.Tensor, 手腕四元数 [qw, qx, qy, qz]
        pos_offset: list or torch.Tensor, 相机位置偏移 [x, y, z]
        lookat_offset: list or torch.Tensor, 观察点偏移 [x, y, z]
        up_offset: list or torch.Tensor, 上方向偏移 [x, y, z]
    """
    # 确保所有张量在相同设备上
    device = wrist_position.device
    dtype = wrist_position.dtype
    
    # 转换为张量（如果还不是的话）
    pos_offset = torch.tensor(pos_offset, dtype=dtype, device=device)
    lookat_offset = torch.tensor(lookat_offset, dtype=dtype, device=device)
    up_offset = torch.tensor(up_offset, dtype=dtype, device=device)
    
    # 计算相机的朝向方向（假设相机朝向手腕的z轴负方向）
    transform_matrix = as_transform_matrix(wrist_position, wrist_quaternion)
    rotation_matrix = transform_matrix[..., :3, :3]  # 3x3旋转矩阵
    
    # 相机位置：手腕位置 + 旋转后的偏移量
    pos = wrist_position + torch.matmul(rotation_matrix, pos_offset)
    
    # 观察点：手腕位置 + 旋转后的观察偏移量
    lookat = wrist_position + torch.matmul(rotation_matrix, lookat_offset)
    
    # 上方向：使用手腕的y轴方向
    up = torch.matmul(rotation_matrix, up_offset)

    return pos, lookat, up


### Attitude-related state variables

def reorder_quaternion(quat: torch.Tensor, to_format: str = "xyzw") -> torch.Tensor:
    """
    Convert quaternion between scalar-first [w, x, y, z] and scalar-last [x, y, z, w] formats.
    Args:
        quat: Input quaternion tensor
        to_format: Target format - "xyzw" (x, y, z, w) or "wxyz" (w, x, y, z)
    Returns:
        Quaternion in the specified format
    Raises:
        ValueError: If quaternion doesn't have 4 elements or invalid format specified
    """
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion must have 4 elements, got {quat.shape[-1]}")
    
    if to_format not in ["xyzw", "qxqyqzqw", "wxyz", "qwqxqyqz"]:
        raise ValueError(f"Target format must be 'xyzw' or 'wxyz', got '{to_format}'")
    
    if to_format in ["xyzw", "qxqyqzqw"]:
        # Rearrange from [w, x, y, z] to [x, y, z, w]
        return torch.roll(quat, shifts=-1, dims=-1)
    elif to_format in ["wxyz", "qwqxqyqz"]:
        # Rearrange from [x, y, z, w] to [w, x, y, z]
        return torch.roll(quat, shifts=1, dims=-1)
    else:
        raise ValueError(f"Target format must be 'xyzw' or 'wxyz', got '{to_format}'")

def as_rotation_matrix(quaternion, order="wxyz"):
    """
    根据姿态四元数，计算旋转矩阵
    参数:
        quaternion: torch.Tensor, 末端姿态四元数 
        order: string, 定义四元数的元素顺序
    返回:
        rotation_matrix: 旋转矩阵
    """
    quaternion = torch.as_tensor(quaternion)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Quaternion must have 4 elements, got shape {tuple(quaternion.shape)}")

    # 提取四元数分量
    if order in ["wxyz", "qwqxqyqz"]:
        qw, qx, qy, qz = (
            quaternion[..., 0],
            quaternion[..., 1],
            quaternion[..., 2],
            quaternion[..., 3],
        )
    elif order in ["xyzw", "qxqyqzqw"]:
        qx, qy, qz, qw = (
            quaternion[..., 0],
            quaternion[..., 1],
            quaternion[..., 2],
            quaternion[..., 3],
        )
    else:
        raise ValueError(f"不支持的顺序格式: {order}。支持的格式: 'wxyz', 'xyzw', 'qwqxqyqz', 'qxqyqzqw'")
    
    # 计算旋转矩阵（从四元数到旋转矩阵的转换）
    # 第一列 (x轴方向)
    r00 = 1 - 2*(qy*qy + qz*qz)
    r10 = 2*(qx*qy + qw*qz)
    r20 = 2*(qx*qz - qw*qy)
    
    # 第二列 (y轴方向)
    r01 = 2*(qx*qy - qw*qz)
    r11 = 1 - 2*(qx*qx + qz*qz)
    r21 = 2*(qy*qz + qw*qx)
    
    # 第三列 (z轴方向)
    r02 = 2*(qx*qz + qw*qy)
    r12 = 2*(qy*qz - qw*qx)
    r22 = 1 - 2*(qx*qx + qy*qy)
    
    # 构建旋转矩阵，兼容单体 [3, 3] 和 batch [N, 3, 3]
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    rotation_matrix = torch.stack([row0, row1, row2], dim=-2)

    return rotation_matrix

def as_transform_matrix(position, quaternion, order="wxyz"):
    """
    将位置和四元数转换为4x4平移旋转变换矩阵
    
    参数:
        position: torch.Tensor, 位置 [x, y, z]
        quaternion: torch.Tensor, 四元数 
        order: string, 四元数的元素顺序 (默认"wxyz")
        
    返回:
        torch.Tensor: 4x4变换矩阵
    """
    position = torch.as_tensor(position)
    quaternion = torch.as_tensor(quaternion, device=position.device, dtype=position.dtype)

    if position.shape[-1] != 3:
        raise ValueError(f"Position must have 3 elements, got shape {tuple(position.shape)}")

    device = position.device
    dtype = position.dtype
    
    # 使用as_rotation_matrix函数获取3x3旋转矩阵
    rotation_matrix = as_rotation_matrix(quaternion, order=order)
    
    if rotation_matrix.dim() == 2:
        transform_matrix = torch.eye(4, device=device, dtype=dtype)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = position
        return transform_matrix

    batch_size = rotation_matrix.shape[0]
    transform_matrix = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    transform_matrix[:, :3, :3] = rotation_matrix
    transform_matrix[:, :3, 3] = position
    
    return transform_matrix

def quaternion_conjugate(q, order="xyzw"):
    """[qx, qy, qz, qw] -> [-qx, -qy, -qz, qw]"""
    q = torch.as_tensor(q)
    if q.shape[-1] != 4:
        raise ValueError(f"Quaternion must have 4 elements, got shape {tuple(q.shape)}")
    if order in ["xyzw", "qxqyqzqw"]:
        result = q.clone()
        result[..., 0:3] = -result[..., 0:3]
        return result
    elif order in ["wxyz", "qwqxqyqz"]:
        result = q.clone()
        result[..., 1:4] = -result[..., 1:4]
        return result
    raise ValueError(f"Unsupported quaternion order: {order}")

def quaternion_multiply(q1, q2):
    """[qx, qy, qz, qw]"""
    q1 = torch.as_tensor(q1)
    q2 = torch.as_tensor(q2, device=q1.device, dtype=q1.dtype)
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError(
            f"Quaternion multiply expects trailing dim 4, got {tuple(q1.shape)} and {tuple(q2.shape)}"
        )

    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    
    return torch.stack([x, y, z, w], dim=-1)


def map_to_range(x, x_close, x_open, new_close, new_open):
    """
    将x从原区间[x_close, x_open]线性映射到新区间[new_close, new_open]
    
    参数:
    x: 输入值（在原区间内）
    x_close: 原区间的起始端点
    x_open: 原区间的结束端点  
    new_close: 新区间的起始端点
    new_open: 新区间的结束端点
    """
    if any(isinstance(v, torch.Tensor) for v in (x, x_close, x_open, new_close, new_open)):
        ref = next(v for v in (x, x_close, x_open, new_close, new_open) if isinstance(v, torch.Tensor))
        device = ref.device
        dtype = ref.dtype

        def to_tensor(v):
            if isinstance(v, torch.Tensor):
                return v.to(device=device, dtype=dtype)
            return torch.tensor(v, device=device, dtype=dtype)

        x_t = to_tensor(x)
        x_close_t = to_tensor(x_close)
        x_open_t = to_tensor(x_open)
        new_close_t = to_tensor(new_close)
        new_open_t = to_tensor(new_open)

        x_min = torch.minimum(x_close_t, x_open_t)
        x_max = torch.maximum(x_close_t, x_open_t)
        out_of_range = (x_t < x_min) | (x_t > x_max)
        try:
            if out_of_range.any().item():
                pass
        except Exception:
            pass

        original_range = x_open_t - x_close_t
        new_range = new_open_t - new_close_t
        eps = torch.tensor(1e-12, device=device, dtype=dtype)
        is_zero = torch.abs(original_range) < eps
        mid = (new_close_t + new_open_t) / 2.0

        t = (x_open_t - x_t) / original_range
        mapped = new_open_t - t * new_range
        return torch.where(is_zero, mid, mapped)

    # 输入验证
    lower = min(x_close, x_open)
    upper = max(x_close, x_open)
    if not (lower <= x <= upper):
        # 可以选择返回边界值或抛出异常
        print(f"警告: x={x} 不在原区间 [{lower}, {upper}] 内")

    # 计算原区间和新区间的长度
    original_range = x_open - x_close
    new_range = new_open - new_close
    
    # 处理区间长度为0的特殊情况
    if abs(original_range) < 1e-12:  # 浮点数容差
        # 原区间是一个点，映射到新区间的中点或者根据需求处理
        return (new_close + new_open) / 2.0
    
    # 核心映射公式：线性插值
    # 这个公式适用于所有单调情况（递增或递减）
    t = (x_open - x) / original_range  # t在[0, 1]范围内
    return new_open - t * new_range

def quat_to_euler(quat):
    """
    将四元数转换为欧拉角 (使用scipy标准库)
    
    Args:
        quat: 四元数 (4,) - [qx, qy, qz, qw]
        
    Returns:
        tuple: (roll, pitch, yaw)
    """
    from scipy.spatial.transform import Rotation

    quat = torch.as_tensor(quat)
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion must have 4 elements, got shape {tuple(quat.shape)}")

    quat_np = quat.detach().cpu().numpy()
    rotation = Rotation.from_quat(quat_np)
    euler = rotation.as_euler('xyz', degrees=False)

    if quat.dim() == 1:
        return euler[0], euler[1], euler[2]
    return torch.as_tensor(euler, dtype=quat.dtype, device=quat.device)
