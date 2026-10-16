# starlink_new

把「新 base link」和「原 panel」拼起来的完整 starlink 卫星模型。

## 来源

| 部分 | 来源 | 说明 |
| --- | --- | --- |
| base 主体 `base_star_body_link` | `assets/urdf/cylinder/whole.urdf` | `body.stl` + `object.stl` |
| base 圆筒 `base_star_cylinder_link` | `assets/urdf/cylinder/whole.urdf` | `cylinder_1.stl` ~ `cylinder_8.stl` |
| panel `panel_Link` | `assets/urdf/starlink/urdf/starlink.urdf` | `panel_Link.STL`，惯量 / 材质 / 关节 origin 均保持原值 |

原 `starlink` 的 `base_star_link` 上有一段圆筒被拆下来重新精细建模，`whole.urdf`
把剩余主体和圆筒拼成两个 link（fixed joint 连接）。本模型在此基础上把 panel 接回主体：

```
base_star_body_link  ──fixed(base_body_to_cylinder, origin=0)──  base_star_cylinder_link
        │
        └──fixed(panel, origin="3.1501 0.12613 0.143")──  panel_Link
```

所有 base 相关 mesh 与原 `base_star_link.STL` 处于同一坐标系，因此各 mesh 的
`<origin>` 都是 0；panel 关节 origin 沿用原 `starlink.urdf` 的
`xyz="3.1501 0.12613 0.143"`，parent 由 `base_star_link` 改为 `base_star_body_link`。

## 材质 / 颜色（与原 starlink 一致）

- base：`starlink_base_star_link_material`，`rgba="0.937255 0.894118 0.690196 1"`，`../textures/1.png`
- panel：`starlink_panel_Link_material`，`rgba="0 0.172549 0.694118 1"`，`../textures/5.png`

## 惯量处理

`whole.urdf` 里两个 base link 用的是占位惯量（mass=1.0，转动惯量=0.01）。本模型改成
能反映原 starlink 的量级：

- `base_star_body_link` 沿用原 `base_star_link` 的质心与惯量张量，mass = 365.27 − 5.50 = 359.77；
- `base_star_cylinder_link` 按圆筒包围盒（半径≈0.1，高≈0.4 的薄壁圆筒）估算，
  mass ≈ 5.50（约占 base 体积 1.5%），质心取圆筒包围盒中心。

两者相加 ≈ 原 `base_star_link`（mass 365.27）。若想按 mesh 精确重算，可在 Genesis 里加
`recompute_inertia=True`，或直接替换这两个 `<inertial>`。

## 目录结构

```
starlink_new/
├── urdf/starlink.urdf          # 合并后的完整模型
├── meshes/                     # URDF 引用的全部 STL
│   ├── body.stl  object.stl
│   ├── cylinder_1.stl ~ cylinder_8.stl
│   └── panel_Link.STL
├── textures/1.png  5.png       # base / panel 贴图
├── config/joint_names_starlink.yaml
└── launch/display.launch  gazebo.launch
```

## 在 Genesis 中加载

```python
import genesis as gs

gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=False)
scene.add_entity(
    morph=gs.morphs.URDF(
        file="src/assets/urdf/starlink_new/urdf/starlink.urdf",
        # 需要单独对圆筒做凸分解时，和 cylinder/view.py 一样保留该 link：
        links_to_keep=["base_star_cylinder_link"],
    ),
)
scene.build()
```

## 已接入项目

`configs/config_starlink.py` 的 `_STARLINK_PATHS['urdf']` 已指向本模型，
`STARLINK_PARAMS` 的 `base` / `grasp_link` 已改为 `base_star_body_link`
（`base_star_cylinder_link` 与它共坐标系，所以 `grasp_offset_local` 数值不用改）。

starlink 目标的 morph 统一由 `config_starlink.make_starlink_morph()` 构造，全项目只有
这一个入口。RL 训练和各 test 里需要切换 target collision 的地方，都改成调用它并只覆写
`collision=`，因此下列设置只需要在这里维护一处：

- `links_to_keep=["base_star_cylinder_link"]`：默认 `merge_fixed_links=True` 会把 base
  的两个 link 和 panel 合并成一个刚体再整体凸分解，圆筒的孔会被大凸包填平；保留该 link
  后它单独分解。
- 一组 CoACD 参数（`convexify=True`、`decompose_object_error_threshold=0.0`、
  `coacd_options(threshold=0.05, ...)`，同 `assets/urdf/cylinder/view.py`）。

实测（scale=0.7）：只加 `links_to_keep` 时圆筒仍是 **1 个实心凸包**（体积 0.0043 ≈
实心圆柱，孔被填平）；再加上那组 CoACD 参数后变成 **60 个凸块**，碰撞总体积 0.0019
≈ 真实筒壁，孔保留。代价是碰撞几何变重，若训练/仿真变慢，用
`make_starlink_morph(fine_cylinder_collision=False)` 换回默认。
