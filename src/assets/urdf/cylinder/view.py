import genesis as gs
import trimesh
import os

gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=False)

robot = scene.add_entity(
    morph=gs.morphs.URDF(
        file="D:/cylinder/whole.urdf",
        convexify=True,
        decompose_robot_error_threshold=0.0,
        recompute_inertia=True,
        # 关键：默认 merge_fixed_links=True 会把 fixed 关节连接的两个 link 合并成一个刚体，
        # 圆柱就和大 body 一起做凸分解，凹坑被大凸包填平。
        # 用 links_to_keep 保留圆柱 link，让它独立凸分解，凹坑得以保留。
        links_to_keep=["base_star_cylinder_link"],
        # 凸包永远是平直面。默认 threshold=0.1 只会把圆筒拆成约 10 个粗楔块（看起来像"三角锥"）。
        # 调小 threshold / 调大 resolution 会分出更多、更贴近弧面的小块，圆筒更圆滑。
        # 权衡：块数越多越圆，但碰撞几何越重（模拟变慢）。
        #   0.1(默认) -> 圆筒~10块   0.05 -> ~几十块   0.03 -> 圆筒150块
        coacd_options=gs.options.CoacdOptions(
            threshold=0.05,
            max_convex_hull=-1,
            resolution=10000,
            preprocess_mode="auto",
        ),
    ),
)
scene.build()

import numpy as np

def quat_to_R(q):
    """wxyz 四元数 -> 3x3 旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])

# 建立空的场景，用来存放所有凸包
combined_mesh = trimesh.util.concatenate([])

for link in robot.links:
    for geom in link.geoms:
        tm = geom.get_trimesh()
        # 关键：get_trimesh() 只返回凸块在【自身局部系】里的网格（重居中、主轴对齐后），
        # 它在世界里的真实位姿在 geom.get_pos(relative=False) / get_quat(relative=False) 里。
        # 必须把这个世界变换应用到顶点上，否则所有凸块都堆在原点附近，
        # 圆筒就会"嵌"进 body 里。
        pos = np.asarray(geom.get_pos(relative=False))
        quat = np.asarray(geom.get_quat(relative=False))
        T = np.eye(4)
        T[:3, :3] = quat_to_R(quat)
        T[:3, 3] = pos
        tm.apply_transform(T)
        combined_mesh += tm

# 导出合并后的整体模型（用绝对路径，避免运行目录不同导致写错位置）
out_dir = r"D:\cylinder\convex_out"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "combined_convex.obj")
combined_mesh.export(out_path)
print("✅ 合并完成!文件:", out_path)