
import math
import torch
from typing import Optional, Union

try:
    from .utils import as_rotation_matrix, quat_to_euler
except Exception:
    from utils import as_rotation_matrix, quat_to_euler


class StateVectors:
    """
    Stores and manages the state vectors of a given robot, including position, orientation and velocity.
    Automatically synchronizes all orientation variables. All quaternion orders follow [qw, qx, qy, qz].
    Supports device management for GPU computation.
    """

    def __init__(self, euler_order: str = "xyz", datatype = torch.float32, device: str = "cpu"):
        """
        Initialize the State object.
        Args:
            euler_order: Euler angle rotation order (default external rotation : "xyz" for roll-pitch-yaw)
        """
        self.euler_order = euler_order
        self.device = torch.device(device)
        self.datatype = datatype

        # Initialize all state variables
        self._initialize_states()

    def _initialize_states(self):
        """Initialize all state variables with zeros"""
        # Global frame states
        self._position = torch.tensor([0.0, 0.0, 0.0], dtype=self.datatype, device=self.device)
        self._quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=self.datatype, device=self.device)  # [qw, qx, qy, qz]
        self._orient = torch.tensor([0.0, 0.0, 0.0], dtype=self.datatype, device=self.device) # -pi ~ pi
        self._R_mat = torch.eye(3, dtype=self.datatype, device=self.device) # identity matrix
        self.R = None

        self._linear_velocity = torch.tensor([0.0, 0.0, 0.0], dtype=self.datatype, device=self.device)
        self._angular_velocity = torch.tensor([0.0, 0.0, 0.0], dtype=self.datatype, device=self.device)

    def _to_state_tensor(self, value, trailing_dim: int, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=self.datatype, device=self.device)
        if tensor.shape[-1] != trailing_dim:
            raise ValueError(
                f"{name} must have trailing dim {trailing_dim}, got shape {tuple(tensor.shape)}"
            )
        return tensor

    @property
    def position(self):
        """position: [x, y, z]"""
        return self._position.clone()
    
    @property
    def quat(self):
        """quaternion: [qw, qx, qy, qz]"""
        return self._quat.clone()
    
    @property
    def orient(self):
        """rotation order: self.euler_order"""
        return self._orient.clone()
    
    @property
    def R_mat(self):
        """rotation matrix, transfer from other frame to this frame."""
        return self._R_mat.clone()
    
    @property
    def linear_velocity(self):
        return self._linear_velocity.clone()
    
    @property
    def angular_velocity(self):
        return self._angular_velocity.clone()

    def update(self, 
               position: Optional[torch.Tensor] = None,
               orient: Optional[torch.Tensor] = None,
               quat: Optional[torch.Tensor] = None,
               linear_velocity: Optional[torch.Tensor] = None,
               angular_velocity: Optional[torch.Tensor] = None):
        """
        Update state variables with automatic synchronization of orientation representations.
        Args:
            position: Position vector [x, y, z] 
            orient: Euler angles [roll, pitch, yaw] 
            quat: Quaternion [qw, qx, qy, qz] 
            linear_velocity: Linear velocity vector [vx, vy, vz] 
            angular_velocity: Angular velocity vector [ωx, ωy, ωz] 
        Raises:
            ValueError: If both orient and quat are provided simultaneously
        """
        # Update position if provided
        if position is not None:
            self._position = self._to_state_tensor(position, 3, "position")
        
        # Update orientation representations with priority handling
        if orient is not None and quat is not None:
            raise ValueError("Cannot provide both orient and quat simultaneously. "
                           "Please provide only one orientation representation.")
        
        if orient is not None:
            # Update from Euler angles
            self._update_rotation_matrices_from_euler(orient)
        elif quat is not None:
            # Update from quaternion
            self._update_rotation_matrices_from_quat(quat)
        
        # Update velocities if provided
        if linear_velocity is not None:
            self._linear_velocity = self._to_state_tensor(
                linear_velocity, 3, "linear_velocity"
            )
        
        if angular_velocity is not None:
            self._angular_velocity = self._to_state_tensor(
                angular_velocity, 3, "angular_velocity"
            )

    def add(self, 
            position: Optional[torch.Tensor] = None,
            orient: Optional[torch.Tensor] = None,
            quat: Optional[torch.Tensor] = None,
            linear_velocity: Optional[torch.Tensor] = None,
            angular_velocity: Optional[torch.Tensor] = None):
        """
        Add incremental values to state variables with automatic synchronization of orientation representations.
                
        Args:
            position: Position increment [dx, dy, dz]
            orient: Euler angle increment [d_roll, d_pitch, d_yaw]
            quat: Quaternion increment [dqw, dqx, dqy, dqz]
            linear_velocity: Linear velocity increment [dvx, dvy, dvz]
            angular_velocity: Angular velocity increment [dωx, dωy, dωz]
            
        Raises:
            ValueError: If both orient and quat are provided simultaneously
        """
        # Add position increment if provided
        if position is not None:
            position_increment = self._to_state_tensor(position, 3, "position")
            self._position = self._position + position_increment
        
        # Add orientation increments with priority handling
        if orient is not None and quat is not None:
            raise ValueError("Cannot provide both orient and quat simultaneously. "
                        "Please provide only one orientation representation.")
        
        if orient is not None:
            # Add to Euler angles and update other representations
            orient_increment = self._to_state_tensor(orient, 3, "orient")
            new_orient = self._orient + orient_increment
            self._update_rotation_matrices_from_euler(new_orient)
            # new_R = Rotation.from_euler(self.euler_order, orient.cpu().numpy()) * self.R
            # new_quat = new_R.as_quat(scalar_first=True)
            # new_quat = torch.tensor(new_quat, dtype=self.datatype, device=self.device)
            # self._update_rotation_matrices_from_quat(new_quat)
        elif quat is not None:
            # Add to quaternion and update other representations
            quat_increment = self._to_state_tensor(quat, 4, "quat")
            new_quat = self._quat + quat_increment
            # Normalize quaternion after addition to maintain unit quaternion property
            new_quat = new_quat / torch.clamp(
                torch.norm(new_quat, dim=-1, keepdim=True),
                min=torch.tensor(1e-12, dtype=self.datatype, device=self.device),
            )
            self._update_rotation_matrices_from_quat(new_quat)
        
        # Add velocity increments if provided
        if linear_velocity is not None:
            linear_velocity_increment = self._to_state_tensor(
                linear_velocity, 3, "linear_velocity"
            )
            self._linear_velocity = self._linear_velocity + linear_velocity_increment
        
        if angular_velocity is not None:
            angular_velocity_increment = self._to_state_tensor(
                angular_velocity, 3, "angular_velocity"
            )
            self._angular_velocity = self._angular_velocity + angular_velocity_increment

    def _update_rotation_matrices_from_quat(self, quat: torch.Tensor):
        """Update rotation matrices from global quaternion [qw, qx, qy, qz]"""
        quat = self._to_state_tensor(quat, 4, "quat")

        # Check if quaternion is not zero
        if torch.allclose(quat, torch.zeros_like(quat)):
            quat_normalized = torch.zeros_like(quat)
            quat_normalized[..., 0] = 1.0
        else:
            quat_normalized = quat / torch.clamp(
                torch.norm(quat, dim=-1, keepdim=True),
                min=torch.tensor(1e-12, dtype=self.datatype, device=self.device),
            )

        self._quat = quat_normalized.to(dtype=self.datatype, device=self.device)

        # Update orientation angles
        self.R = None
        euler = quat_to_euler(self._quat[..., [1, 2, 3, 0]])
        if isinstance(euler, tuple):
            euler = torch.tensor(euler, dtype=self.datatype, device=self.device)
        self._orient = euler.to(dtype=self.datatype, device=self.device)
        self._R_mat = as_rotation_matrix(self._quat, order="wxyz").to(
            dtype=self.datatype, device=self.device
        )

    def _update_rotation_matrices_from_euler(self, euler: torch.Tensor):
        """Update rotation matrices from global Euler angles"""
        euler = self._to_state_tensor(euler, 3, "orient")
        # -pi ~ pi
        euler = torch.remainder(euler + math.pi, 2*math.pi) - math.pi
        self._orient = euler.to(dtype=self.datatype, device=self.device)

        roll = self._orient[..., 0]
        pitch = self._orient[..., 1]
        yaw = self._orient[..., 2]

        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        self._quat = torch.stack([qw, qx, qy, qz], dim=-1).to(
            dtype=self.datatype, device=self.device
        )
        self._R_mat = as_rotation_matrix(self._quat, order="wxyz").to(
            dtype=self.datatype, device=self.device
        )
        self.R = None

    def to(self, device: Union[str, torch.device]):
        """Move all state tensors to the specified device"""
        device = torch.device(device)
        if device == self.device:
            return self
        
        self.device = device
        
        # Move all tensors to the new device
        for attr_name in dir(self):
            if attr_name.startswith('_') and not attr_name.startswith('__'):
                attr = getattr(self, attr_name)
                if isinstance(attr, torch.Tensor):
                    setattr(self, attr_name, attr.to(device))
        
        return self

    def __repr__(self):
        return (f"State(euler_order='{self.euler_order}')\n"
                f"Position: {self.position}\n"
                f"Orientation: {self.orient}\n"
                f"Quaternion (wxyz): {self.quat}\n"
                f"Linear_Velocity: {self.linear_velocity}\n"
                f"Angular_Velocity: {self.angular_velocity}\n")
    

if __name__ == '__main__':
    state = StateVectors(euler_order="XYZ")

    pos = torch.tensor([1.0, 0.0, 0.0])
    orient = torch.tensor([1.57, 0.0, 0.0])
    state.update(position=pos, orient=orient)
    print(state)

    pos = torch.tensor([1.0, 0.0, 0.0])
    orient = torch.tensor([0.0, 1.57, 0.0])
    state.add(position=pos, orient=orient)
    print(state)
