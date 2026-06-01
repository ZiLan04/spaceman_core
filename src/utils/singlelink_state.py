import torch
from typing import Optional, Union

# Extension APIs
try:
    from .state_vectors import StateVectors
    from .utils import reorder_quaternion
except:
    from state_vectors import StateVectors
    from utils import reorder_quaternion


class SingleLinkState:
    """
    Stores and manages the state of a given single-link robot in both global and body frames.
    Assuming the body frame is established at the center of global position and attitude.
    Automatically synchronizes all state variables when any coordinate frame is updated.
    Supports device management for GPU computation. Input/Output quaternion follow [qw, qx, qy, qz].
    """

    def __init__(self, euler_order: str = "XYZ", datatype = torch.float32, device: str = "cpu"):
        """
        Initialize the State object.
        Args:
            euler_order: Euler angle rotation order (default: "XYZ" for roll-pitch-yaw)
            datatype: Data type for tensors (default: torch.float32)
            device: Device for tensor storage (default: "cpu")
        """
        self.euler_order = euler_order
        self.device = torch.device(device)
        self.datatype = datatype
        
        # Initialize StateVectors for global and body frames
        self._global_state = StateVectors(euler_order, datatype, device) # [qx,qy,qz,qw]
        self._body_state = StateVectors(euler_order, datatype, device) # zero position and identity orientation 

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

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure the tensor is on the correct device"""
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor

    def _to_tensor(self, array) -> torch.Tensor:
        """Convert numpy array to tensor on the correct device"""
        return torch.tensor(array, dtype=self.datatype, device=self.device)
    
    def get_state_dict(self, frame: str = "global") -> dict:
        """Get complete state dictionary for specified frame, where quaternion [qw,qx,qy,qz]."""
        if frame.lower() == "global":
            return {
                "position": self.position_global,
                "quat": reorder_quaternion(self.quat_global,"wxyz"),
                "orient": self.orient_global,
                "linear_velocity": self.linear_velocity_global,
                "angular_velocity": self.angular_velocity_global,
                "R_matrix": self.R_body_to_global
            }
        else:  # body frame
            return {
                "position": self.position_body,
                "quat": reorder_quaternion(self.quat_body,"wxyz"),
                "orient": self.orient_body,
                "linear_velocity": self.linear_velocity_body,
                "angular_velocity": self.angular_velocity_body,
                "R_matrix": self.R_global_to_body
            }

    def __repr__(self):
        return (f"State(euler_order='{self.euler_order}', device='{self.device}')\n"
                f"Global Position: {self.position_global.detach().cpu().numpy()}\n"
                f"Global Orientation: {self.orient_global.detach().cpu().numpy()}\n"
                f"Body Position: {self.position_body.detach().cpu().numpy()}\n"
                f"Body Orientation: {self.orient_body.detach().cpu().numpy()}")

    @property
    def global_state(self):
        """Whose quaternion order [qx, qy, qz, qw]"""
        return self._global_state

    @property
    def body_state(self):
        """Whose quaternion order [qx, qy, qz, qw]"""
        return self._body_state

    @property
    def position_global(self):
        return self._global_state.position
    
    @property
    def position_body(self):
        return self._body_state.position
    
    @property
    def quat_global(self):
        """Returned quaternion order [qw, qx, qy, qz]"""
        return reorder_quaternion(self._global_state.quat,"wxyz")
    
    @property
    def quat_body(self):
        """Returned quaternion order [qw, qx, qy,qz]"""
        return reorder_quaternion(self._body_state.quat,"wxyz")
    
    @property
    def orient_global(self):
        return self._global_state.orient
    
    @property
    def orient_body(self):
        return self._body_state.orient
    
    @property
    def linear_velocity_global(self):
        return self._global_state.linear_velocity
    
    @property
    def linear_velocity_body(self):
        return self._body_state.linear_velocity
    
    @property
    def angular_velocity_global(self):
        return self._global_state.angular_velocity
    
    @property
    def angular_velocity_body(self):
        return self._body_state.angular_velocity

    @property
    def R_body_to_global(self):
        """Warning: as_quat() [qx, qy, qz, qw]."""
        return self._global_state._R_mat

    @property
    def R_global_to_body(self):
        """Warning: as_quat() [qx, qy, qz, qw]."""
        return self._global_state._R_mat.transpose(-1, -2)

    def _update_rotation_matrices(self):
        """Update rotation matrices based on global quaternion"""
        quat_global = self._global_state.quat
        if quat_global is not None:
            # Update rotation from body to global
            # self._global_state.update(quat=quat_global)
            self._R_body_to_global = self._global_state._R_mat
            
            # Update rotation from global to body
            self._R_global_to_body = self._R_body_to_global.transpose(-1, -2)

    def _synchronize_global_to_body(self):
        """Transform velocities from global frame to body frame.
        no need to transform position and orientation (fixed in body frame)
        """
        # self._update_rotation_matrices()
        
        # Transform velocities
        lin_vel_global = self._global_state.linear_velocity
        ang_vel_global = self._global_state.angular_velocity
        
        # Use rotation matrix from global frame to body frame
        linear_velocity_body = torch.matmul(
            self.R_global_to_body, lin_vel_global.unsqueeze(-1)
        ).squeeze(-1)
        angular_velocity_body = torch.matmul(
            self.R_global_to_body, ang_vel_global.unsqueeze(-1)
        ).squeeze(-1)
        self._body_state.update(
            linear_velocity=linear_velocity_body,
            angular_velocity=angular_velocity_body
        )

    def _synchronize_body_to_global(self):
        """Transform velocities from global frame to body frame.
        no need to transform position and orientation (fixed in body frame)
        """
        # self._update_rotation_matrices()

        # Transform velocities
        lin_vel_body = self._body_state.linear_velocity
        ang_vel_body = self._body_state.angular_velocity
        
        # Use rotation matrix from body frame to global frame
        linear_velocity_global = torch.matmul(
            self.R_body_to_global, lin_vel_body.unsqueeze(-1)
        ).squeeze(-1)
        angular_velocity_global = torch.matmul(
            self.R_body_to_global, ang_vel_body.unsqueeze(-1)
        ).squeeze(-1)
        self._global_state.update(
            linear_velocity=linear_velocity_global,
            angular_velocity=angular_velocity_global
        )

    def update_from_global_frame(self, 
                               position: Optional[torch.Tensor] = None,
                               quat: Optional[torch.Tensor] = None, 
                               orient: Optional[torch.Tensor] = None,
                               linear_velocity: Optional[torch.Tensor] = None,
                               angular_velocity: Optional[torch.Tensor] = None):
        """
        Update state from global frame values and synchronize body frame.
        quaternion: [qw,qx,qy,qz]
        """
        if quat is not None:
            quat = reorder_quaternion(quat,"xyzw")

        # Update global state using StateVectors
        self._global_state.update(
            position=position,
            quat=quat,
            orient=orient,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity
        )
        
        # Synchronize body frame states
        self._synchronize_global_to_body()
    
    def update_from_body_frame(self, 
                               position: Optional[torch.Tensor] = None,
                               quat: Optional[torch.Tensor] = None,
                               orient: Optional[torch.Tensor] = None,
                               linear_velocity: Optional[torch.Tensor] = None,
                               angular_velocity: Optional[torch.Tensor] = None):
        """
        Update state from global frame values and synchronize body frame.
        quaternion: [qw,qx,qy,qz]
        """
        if quat is not None:
            quat = reorder_quaternion(quat,"xyzw")

        # Update global state using StateVectors
        self._body_state.update(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity
        )
        
        # Synchronize body frame states
        self._synchronize_body_to_global()
    

# test main
if __name__ == "__main__":
    # 创建状态对象，指定device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = SingleLinkState(euler_order="xyz", device=device)
    
    print("Initial state:")
    print(state)
    print()
    
    # 从全局坐标系更新，自动处理device
    print("=== Updating from global frame ===")
    state.update_from_global_frame(
        position=torch.tensor([1.0, 0.0, 0.0], device=device),
        quat=torch.tensor([0.707, 0.0, 0.0, 0.707], device=device),  # quaternion [qw,qx,qy,qz]
        linear_velocity=torch.tensor([0.1, 0.2, 0.3], device=device),
        angular_velocity=torch.tensor([0.01, 0.02, 0.03], device=device)
    )
    # state.update_from_global_frame(
    #     position=torch.tensor([-0.5074,  0.0000,  0.4000], device=device),
    #     quat=torch.tensor([ 0.7074,  0.0000, -0.7068,  0.0000], device=device),  # quaternion [qw,qx,qy,qz]
    # )

    print("After global update:")
    print(f"Global position: {state.position_global.cpu().numpy()}")
    print(f"Body position: {state.position_body.cpu().numpy()}")
    print(f"Global orientation: {state.orient_global.cpu().numpy()}")
    print(f"Body orientation: {state.orient_body.cpu().numpy()}")
    print(f"Global linear velocity: {state.linear_velocity_global.cpu().numpy()}")
    print(f"Body linear velocity: {state.linear_velocity_body.cpu().numpy()}")
    print()
    
    # 从body坐标系更新
    print("=== Updating from body frame ===")
    state.update_from_body_frame(
        position=torch.tensor([2.0, 0.0, 0.0], device=device),
        orient=torch.tensor([0.0, 0.0, -1.570796], device=device),  # small rotation
        linear_velocity=torch.tensor([1.0, 0.0, 0.0], device=device)
    )
    
    print("After body update:")
    print(f"Global position: {state.position_global.cpu().numpy()}")
    print(f"Body position: {state.position_body.cpu().numpy()}")
    print(f"Global orientation: {state.orient_global.cpu().numpy()}")
    print(f"Body orientation: {state.orient_body.cpu().numpy()}")
    print(f"Global linear velocity: {state.linear_velocity_global.cpu().numpy()}")
    print(f"Body linear velocity: {state.linear_velocity_body.cpu().numpy()}")
    
    # 检查所有tensor都在正确device上
    print(f"\nState device: {state.device}")
    print(f"Global position device: {state.position_global.device}")
    print(f"Body position device: {state.position_body.device}")
    
    # 与其他tensor运算（确保device一致）
    other_tensor = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    result = state.position_global + other_tensor
    print(f"Result device: {result.device}")
    print(f"Result: {result.cpu().numpy()}")
