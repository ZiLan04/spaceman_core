import sys
import torch
import logging
import math

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from envs.genesis_env import GenesisSim
from configs.asset_configs import *
from controllers.backend import Backend
from utils.setup_logger import setup_logger

class PIDController(Backend):  # 继承Backend
    def __init__(self, P, I, D, setpoint, dt=0.01, output_limits=None, name="pid"):
        super().__init__()  # 调用Backend的__init__
        self.logger = setup_logger(f"PIDController.{name}")
        self._name = name
        """
        PyTorch GPU版本的6自由度PID控制器
        """
        self.device = GenesisSim().device
        self.datatype = GenesisSim().datatype
        self.dt = dt

        self.Kp = torch.tensor(P, device=self.device, dtype=self.datatype)
        self.Ki = torch.tensor(I, device=self.device, dtype=self.datatype)
        self.Kd = torch.tensor(D, device=self.device, dtype=self.datatype)
        self.setpoint = torch.tensor(setpoint, device=self.device, dtype=self.datatype)
        self.output_limits = self._parse_output_limits(output_limits)

        self.init_P = self.Kp.clone()
        self.init_I = self.Ki.clone()
        self.init_D = self.Kd.clone()
        self.init_setpoint = self.setpoint.clone()

        # 新增：存储当前状态和控制输出
        self._current_position = None
        self._current_orientation = None
        self._control_output = None
        self.enabled = True
        self.prev_error = None
        self.integral = None
        self._ensure_state_buffers(self.setpoint)

    def _as_pid_tensor(self, value, expected_last_dim: int, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device, dtype=self.datatype)
        if tensor.shape[-1] != expected_last_dim:
            raise ValueError(
                f"{name} should have trailing dim {expected_last_dim}, got {tuple(tensor.shape)}"
            )
        return tensor

    def _match_batch_shape(self, tensor: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
        if reference.dim() == 1:
            if tensor.dim() != 1:
                raise ValueError(
                    f"{name} batch shape {tuple(tensor.shape)} does not match reference {tuple(reference.shape)}"
                )
            return tensor

        if tensor.dim() == 1:
            return tensor.unsqueeze(0).repeat(reference.shape[0], 1)
        if tensor.dim() == 2 and tensor.shape[0] == reference.shape[0]:
            return tensor
        raise ValueError(
            f"{name} batch shape {tuple(tensor.shape)} does not match reference {tuple(reference.shape)}"
        )

    def _ensure_state_buffers(self, reference_state: torch.Tensor):
        if reference_state.dim() == 1:
            target_shape = (6,)
        else:
            target_shape = tuple(reference_state.shape)

        if self.prev_error is None or tuple(self.prev_error.shape) != target_shape:
            self.prev_error = torch.zeros(
                target_shape, device=self.device, dtype=self.datatype
            )
            self.integral = torch.zeros(
                target_shape, device=self.device, dtype=self.datatype
            )

    def _parse_output_limits(self, output_limits):
        if output_limits is None:
            return None
        elif isinstance(output_limits, (int, float)):
            return [(float(-output_limits), float(output_limits))] * 6
        elif isinstance(output_limits, (list, tuple)):
            if len(output_limits) == 6:
                limits = []
                for limit in output_limits:
                    if limit is None:
                        limits.append((-float('inf'), float('inf')))
                    elif isinstance(limit, (int, float)):
                        limits.append((float(-limit), float(limit)))
                    elif isinstance(limit, (list, tuple)) and len(limit) == 2:
                        limits.append((float(limit[0]), float(limit[1])))
                return limits

    def _wrap_error(self, error, setpoint, current):
        wrapped_error = error.clone()
        
        for i in range(3, 6):
            angle_diff = setpoint[..., i] - current[..., i]
            wrapped_error[..., i] = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        
        return wrapped_error
    
    def _apply_output_limits(self, output):
        if self.output_limits is None:
            return output
        
        limited_output = output.clone()
        for i in range(6):
            min_val, max_val = self.output_limits[i]
            limited_output[..., i] = torch.clamp(output[..., i], min_val, max_val)
        
        return limited_output

    def control(self, pos, orien):
        """
        计算控制输出
        
        Args:
            pos: 当前位置 [x, y, z]
            orien: 当前姿态 [roll, pitch, yaw]
            
        Returns:
            pos_control: 位置控制输出 [x, y, z]
            orien_control: 姿态控制输出 [roll, pitch, yaw]
        """
        pos = self._as_pid_tensor(pos, 3, "pos")
        orien = self._as_pid_tensor(orien, 3, "orien")

        if pos.dim() != orien.dim():
            raise ValueError(
                f"pos and orien should have the same rank, got {tuple(pos.shape)} and {tuple(orien.shape)}"
            )

        current_state = torch.cat([pos, orien], dim=-1)
        self.setpoint = self._match_batch_shape(self.setpoint, current_state, "setpoint")
        self._ensure_state_buffers(current_state)
        
        raw_error = self.setpoint - current_state
        error = self._wrap_error(raw_error, self.setpoint, current_state)

        self.integral = self.integral + error * self.dt

        P = self.Kp * error
        I = self.Ki * self.integral
        error_diff = error - self.prev_error

        for i in range(3, 6):
            needs_wrap = torch.abs(error_diff[..., i]) > math.pi
            if torch.any(needs_wrap):
                positive = error_diff[..., i] > 0
                error_diff[..., i] = torch.where(
                    needs_wrap & positive,
                    error_diff[..., i] - 2 * math.pi,
                    error_diff[..., i],
                )
                error_diff[..., i] = torch.where(
                    needs_wrap & (~positive),
                    error_diff[..., i] + 2 * math.pi,
                    error_diff[..., i],
                )
        
        D = self.Kd * error_diff / self.dt
        
        self.prev_error = error.clone()
        output = P + I + D

        output = self._apply_output_limits(output)

        pos_control = output[..., :3]
        orien_control = output[..., 3:]
        
        return pos_control, orien_control
    
    def reset(self):
        self.Kp = self.init_P.clone()
        self.Ki = self.init_I.clone()
        self.Kd = self.init_D.clone()
        self.setpoint = self.init_setpoint.clone()
        self._ensure_state_buffers(self.setpoint)
        self.prev_error.zero_()
        self.integral.zero_()
        self._current_position = None
        self._current_orientation = None
        self._control_output = None

    def reset_envs(self, env_ids):
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_tensor.numel() == 0:
            return

        self._ensure_state_buffers(self.setpoint)

        if self.prev_error.dim() == 1:
            self.prev_error.zero_()
            self.integral.zero_()
            return

        self.prev_error[env_ids_tensor] = 0.0
        self.integral[env_ids_tensor] = 0.0
        if self._current_position is not None and self._current_position.dim() > 1:
            self._current_position[env_ids_tensor] = 0  # 或None
        if self._current_orientation is not None and self._current_orientation.dim() > 1:
            self._current_orientation[env_ids_tensor] = 0  # 或None
        if self._control_output is not None:
            self._control_output[env_ids_tensor] = 0

    def _wrap_to_pi(self, angles):
        wrapped = (angles + math.pi) % (2 * math.pi) - math.pi
        return wrapped
    
    def update_setpoint(self, setpoint):
        self.setpoint = torch.as_tensor(setpoint, device=self.device, dtype=self.datatype)
        if self.setpoint.dim() == 1:
            for i in range(3, 6):
                if self.setpoint[i].abs() > math.pi:
                    self.setpoint[i] = self._wrap_to_pi(self.setpoint[i])
        else:
            self.setpoint[..., 3:6] = self._wrap_to_pi(self.setpoint[..., 3:6])
        self._ensure_state_buffers(self.setpoint)
    
    def update_gains(self, P=None, I=None, D=None):
        if P is not None:
            self.Kp = torch.as_tensor(P, device=self.device, dtype=self.datatype)
        if I is not None:
            self.Ki = torch.as_tensor(I, device=self.device, dtype=self.datatype)
        if D is not None:
            self.Kd = torch.as_tensor(D, device=self.device, dtype=self.datatype)

    # ------------------------------------------------------
    # 新增：Backend接口要求的实现
    # ------------------------------------------------------
    
    def step(self):
        """Backend接口：执行PID控制步"""
        if not self.enabled:
            self._control_output = None
            return
        if self._current_position is None or self._current_orientation is None:
            self.logger.warning("No current state available for PID control")
            return
            
        # 调用你现有的control方法
        pos_control, orien_control = self.control(
            self._current_position, 
            self._current_orientation
        )
        
        # 应用到机器人（可选，可以在外部调用apply_force）
        self._control_output = {
            'position': pos_control,
            'orientation': orien_control
        }
        
        self.logger.debug(
            "PID控制输出: 位置shape=%s, 姿态shape=%s",
            tuple(pos_control.shape),
            tuple(orien_control.shape),
        )

    def quaternion_to_euler(self, quat):
        """将四元数 [qw, qx, qy, qz] 转换为欧拉角 [roll, pitch, yaw]"""
        quat = self._as_pid_tensor(quat, 4, "quat")
        qw = quat[..., 0]
        qx = quat[..., 1]
        qy = quat[..., 2]
        qz = quat[..., 3]
        # 滚转角 (x轴旋转)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # 俯仰角 (y轴旋转)
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)
        
        # 偏航角 (z轴旋转)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return torch.stack([roll, pitch, yaw], dim=-1)
    
    def update_state(self, position, orien):
        """Backend接口：更新当前状态"""
        self._current_position = self._as_pid_tensor(position, 3, "position")
        orien = torch.as_tensor(orien, device=self.device, dtype=self.datatype)
        if orien.shape[-1] == 4:
            orien = self.quaternion_to_euler(orien)
        elif orien.shape[-1] != 3:
            raise ValueError(
                f"orien should have trailing dim 3 or 4, got {tuple(orien.shape)}"
            )
        self._current_orientation = orien
    
    def input_reference(self):
        """Backend接口：返回控制输出"""
        return self._control_output
    
    def start(self):
        """Backend接口：启动"""
        self.logger.info(f"PIDController '{self._name}' started")
        self.reset()  # 启动时重置
    
    def stop(self):
        """Backend接口：停止"""
        self.logger.info(f"PIDController '{self._name}' stopped")
        self.reset()  # 停止时重置

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)
        if not self.enabled:
            self._control_output = None
    
    @property
    def name(self):
        return self._name
