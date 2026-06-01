import torch
from typing import Optional

# Extension APIs
try:
    from .state_vectors import StateVectors
    from .utils import quat_to_euler, reorder_quaternion
except:
    from state_vectors import StateVectors
    from utils import quat_to_euler, reorder_quaternion

class TwoLinkState:
    """
    Stores and manages the state of a given two-link robot, from parent Link i to child Link i+1. 
    States includes global states of Link i and Link i+1, body states of Link i+1 in terms of Link i,
    Transformation matrices between parent Link i and i+1,
    Automatically synchronizes all state variables when any coordinate frame is updated.
    Supports device management for GPU computation. 
    Input/Output quaternion follow scalar-first format [qw, qx, qy, qz].
    """
    def __init__(self, euler_order: str = "xyz", datatype = torch.float32, device: str = "cpu"):
        """
        Initialize the State object.
        Args:
            euler_order: Euler angle rotation order (default: "xyz" for roll-pitch-yaw)
            datatype: Data type for tensors (default: torch.float32)
            device: Device for tensor storage (default: "cpu")
        """
        self.euler_order = euler_order
        self.device = torch.device(device)
        self.datatype = datatype
        
        # Initialize global StateVectors of parent Link i and child Link i+1
        # parent link could be world frame, and child link could be the robot frame
        self._link_parent_global_state = StateVectors(euler_order, datatype, device) # [qx,qy,qz,qw] 
        self._link_child_global_state = StateVectors(euler_order, datatype, device) # [qx,qy,qz,qw]
        
        # Initialize StateVectors of child link i+1 in parent Link i frame
        self._link_child_in_parent = StateVectors(euler_order, datatype, device)

        # Initialize rotation matrices
        self._update_parent_rotation_matrices()
        self._update_child_rotation_matrices()
        self._update_R_child_to_parent()

    @property
    def link_parent_global_state(self):
        return self._link_parent_global_state
    
    @property
    def link_child_global_state(self):
        return self._link_child_global_state
    
    @property
    def link_child_in_parent(self):
        return self._link_child_in_parent
    
    @property
    def R_child_to_parent(self):
        return self._R_child_to_parent
    
    @property
    def R_parent_to_child(self):
        return self._R_parent_to_child

    def _update_parent_rotation_matrices(self):
        """Update rotation matrices between parent link and global frame"""
        # Update rotation matrices from parent link to global frame
        self._R_parent_to_global = self._link_parent_global_state.R_mat
        
        # Update rotation matrices from global frame to parent link
        self._R_global_to_parent = self._R_parent_to_global.transpose(-1, -2)

    def _update_child_rotation_matrices(self):
        """Update rotation matrices between child link and global frame"""
        # Update rotation matrices from child link to global frame
        self._R_child_to_global = self._link_child_global_state.R_mat

        # Update rotation matrices from parent link frame to child link frame
        self._R_global_to_child = self._R_child_to_global.transpose(-1, -2)

    def _update_R_child_to_parent(self):
        """Update rotation matrices between parent link and child link"""
        # Update rotation matrices from child link frame to parent link frame
        self._R_child_to_parent = self._link_child_in_parent.R_mat

        # Update rotation matrices from parent link frame to child link frame
        self._R_parent_to_child = self._R_child_to_parent.transpose(-1, -2)

    def update_parent_from_global_frame(self, 
                               position: Optional[torch.Tensor] = None,
                               quat: Optional[torch.Tensor] = None, 
                               orient: Optional[torch.Tensor] = None,
                               linear_velocity: Optional[torch.Tensor] = None,
                               angular_velocity: Optional[torch.Tensor] = None):
        """
        Update parent link state from global frame values.
        quaternion: [qw,qx,qy,qz]
        """
        
        # Update parent link global state using StateVectors
        self._link_parent_global_state.update(
            position=position,
            quat=quat,
            orient=orient,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity
        )
        #
        self._update_parent_rotation_matrices()

    def update_child_from_global_frame(self, 
                               position: Optional[torch.Tensor] = None,
                               quat: Optional[torch.Tensor] = None, 
                               orient: Optional[torch.Tensor] = None,
                               linear_velocity: Optional[torch.Tensor] = None,
                               angular_velocity: Optional[torch.Tensor] = None):
        """
        Update child link state from global frame values.
        quaternion: [qw,qx,qy,qz]
        """

        # Update global state using StateVectors
        self._link_child_global_state.update(
            position=position,
            quat=quat,
            orient=orient,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity
        )
        #
        self._update_child_rotation_matrices()

    def update_child_in_parent(self):
        """Update state of child link in parent link frame"""
        # 获取父连杆和子连杆的全局状态
        parent_global = self._link_parent_global_state
        child_global = self._link_child_global_state
        
        # 计算相对位置
        position_diff = child_global.position - parent_global.position
        relative_position = torch.matmul(
            self._R_global_to_parent, position_diff.unsqueeze(-1)
        ).squeeze(-1)
        
        # 计算相对旋转矩阵：R_child_in_parent = R_parent^T * R_child
        self._R_child_to_parent = torch.matmul(self._R_global_to_parent, self._R_child_to_global)
        
        # Update rotation matrices from parent link frame to child link frame
        self._R_parent_to_child = self._R_child_to_parent.transpose(-1, -2)

        parent_quat_xyzw = reorder_quaternion(parent_global.quat, "xyzw")
        child_quat_xyzw = reorder_quaternion(child_global.quat, "xyzw")
        parent_conj_xyzw = parent_quat_xyzw.clone()
        parent_conj_xyzw[..., 0:3] = -parent_conj_xyzw[..., 0:3]

        px, py, pz, pw = (
            parent_conj_xyzw[..., 0],
            parent_conj_xyzw[..., 1],
            parent_conj_xyzw[..., 2],
            parent_conj_xyzw[..., 3],
        )
        cx, cy, cz, cw = (
            child_quat_xyzw[..., 0],
            child_quat_xyzw[..., 1],
            child_quat_xyzw[..., 2],
            child_quat_xyzw[..., 3],
        )
        rel_x = pw * cx + px * cw + py * cz - pz * cy
        rel_y = pw * cy - px * cz + py * cw + pz * cx
        rel_z = pw * cz + px * cy - py * cx + pz * cw
        rel_w = pw * cw - px * cx - py * cy - pz * cz
        relative_quat_xyzw = torch.stack([rel_x, rel_y, rel_z, rel_w], dim=-1)
        relative_quat = reorder_quaternion(relative_quat_xyzw, "wxyz")
        # 更新子连杆在父连杆坐标系下的状态
        self._link_child_in_parent.update(
            position=relative_position,
            quat=relative_quat,
        )


        
if __name__ == '__main__':
    # 创建二连杆状态对象
    two_link = TwoLinkState(euler_order="xyz", device="cpu")
    
    print("=== 初始状态 ===")
    print("父连杆位置:", two_link.link_parent_global_state.position)
    print("子连杆位置:", two_link.link_child_global_state.position)
    print("相对位置:", two_link.link_child_in_parent.position)
    print("R_child_to_parent:\n", two_link.R_child_to_parent)
    print()
    
    # 测试1: 父连杆在原点，子连杆在x轴方向1单位处
    print("=== 测试1: 简单平移 ===")
    two_link.update_parent_from_global_frame(
        position=torch.tensor([0.0, 0.0, 0.0]),
        quat=torch.tensor([1.0, 0.0, 0.0, 0.0])  # 单位四元数 [qw, qx, qy, qz]
    )
    
    two_link.update_child_from_global_frame(
        position=torch.tensor([1.0, 0.0, 0.0]),
        quat=torch.tensor([1.0, 0.0, 0.0, 0.0])  # 单位四元数
    )
    
    two_link.update_child_in_parent()
    
    print("父连杆位置:", two_link.link_parent_global_state.position)
    print("子连杆位置:", two_link.link_child_global_state.position)
    print("相对位置:", two_link.link_child_in_parent.position)
    print("相对四元数:", two_link.link_child_in_parent.quat)
    print("R_child_to_parent:\n", two_link.R_child_to_parent)
    print()
    
    # 测试2: 父连杆绕z轴旋转90度，子连杆在x轴方向1单位处
    print("=== 测试2: 父连杆旋转90度 ===")
    two_link.update_parent_from_global_frame(
        position=torch.tensor([0.0, 0.0, 0.0]),
        orient=torch.tensor([0.0, 0.0, -np.pi/2])  # 绕z轴旋转90度
    )
    
    two_link.update_child_from_global_frame(
        position=torch.tensor([1.0, 0.0, 0.0]),
        quat=torch.tensor([1.0, 0.0, 0.0, 0.0])  # 单位四元数
    )
    
    two_link.update_child_in_parent()
    
    print("父连杆欧拉角:", two_link.link_parent_global_state.orient)
    print("子连杆位置:", two_link.link_child_global_state.position)
    print("相对位置:", two_link.link_child_in_parent.position)  # 应该在父连杆的y轴负方向
    print("相对欧拉角:", two_link.link_child_in_parent.orient)
    print("R_child_to_parent:\n", two_link.R_child_to_parent)
    print()
    
    # 测试3: 验证旋转矩阵的正确性
    print("=== 测试3: 旋转矩阵验证 ===")
    # 从父连杆到全局的旋转矩阵
    R_p2g = two_link.link_parent_global_state.R_mat
    # 从子连杆到父连杆的旋转矩阵  
    R_c2p = two_link.R_child_to_parent
    # 从子连杆到全局的旋转矩阵（应该等于 R_p2g * R_c2p）
    R_c2g_calculated = torch.matmul(R_p2g, R_c2p)
    R_c2g_actual = two_link.link_child_global_state.R_mat
    
    print("计算得到的 R_child_to_global:\n", R_c2g_calculated)
    print("实际的 R_child_to_global:\n", R_c2g_actual)
    print("两者是否相等:", torch.allclose(R_c2g_calculated, R_c2g_actual, atol=1e-6))
