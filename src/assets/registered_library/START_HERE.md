# Historical library notes

These notes describe an earlier gripper revision. For current operation use `.agents/skills/spaceman-asset-pair/SKILL.md` and `docs/asset-pairs/README.md` at repository root. Current arms have four rotary finger joints. JSON reports beside this document are historical snapshots; inspect current registry status and revisions before applying them.

# VISPA + Starlink 原始夹爪（2026-11-05）

27卫星×2机械臂，共54组合。VISPA保留6个旋转关节，固定转接原始Starlink夹爪主体及两个手指，不添加Starlink原腕部旋转关节。Starlink整臂不变，Franka备选版本保存在之前独立目录。

原夹爪3个STL字节不变，手指关节原点、轴、0–0.04m行程及惯量保留。仅修改名称和末端固定安装变换，新增几何转接件。源素材为用户提供的Starlink资源；不声称新增第三方授权或航天适用认证。

控制：两个手指分别为vispa_finger_joint1/2，均设q=0.04*u，u=0张开，u=1向内闭合。无mimic，必须同步发给两关节。0.04m是单指行程，不是夹爪净开口，也不能把Franka的80mm口径套用到本模型。详见arms/airbus_vispa/gripper_control.json。

TCP=vispa_hand_tcp，为手指末端区域的近似几何参考，尚待接触标定。+Z接近方向，X开合方向。转接件未验证孔位、紧固、承载。

验证：3种开合状态下夹爪对VISPA零位姿表面检查；27款卫星检查新增夹爪及转接件在开/闭端点的表面相交。未验证全行程、自碰撞、实体包含、抓取成功、动力学。预览是几何展示。旧validation.json等为历史裸臂报告；当前新增验证见GRIPPER_*.json及registry.json。

VISPA主臂仍缺完整碰撞/惯量，不能直接认定训练就绪；卫星6缺件、catch01有12个未网格化面，卫星17支座仍需加强设计。

查看：单臂包解压后导入整个文件夹的arm.urdf；individual/中ZIP为单独卫星组合，主共享包不重复收录。所有旧版本保留。
