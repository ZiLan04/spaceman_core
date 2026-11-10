# 卫星与机械臂组合：通用调用与交接

本分支提供 27 个卫星登记项、Starlink/VISPA 两款机械臂及 54 个组合。两款机械臂均换装 2× 几何缩放的 Kinova JACO 旋转二指手：6 个臂关节 + 4 个手指转动关节。`satellite_6` 素材不完整，默认随机选择排除，因此随机池为 26 款卫星。登记数量不代表物理抓取任务已通过。

## 给任意 agent

让 agent 阅读 [SKILL.md](../../../.agents/skills/spaceman-asset-pair/SKILL.md)：

说明文件是普通 Markdown；调用入口为 Python 函数和输出 JSON 的命令行，不依赖任何模型厂商。支持 skill 自动发现的 agent 可按自身约定安装/链接这个目录；不支持的 agent 直接读该文件即可。无需 agent 也能运行。

从仓库根运行：

```bash
python scripts/assets/materialize_meshes.py
python src/asset_pair_api.py list
python src/asset_pair_api.py plan --seed 42 --output artifacts/plan.json
python safe_run.py --seconds 240 --rss-gib 8 --log artifacts/smoke.log -- python src/test/asset_pair_agent_smoke.py --carrier satellite_17 --arm airbus_vispa --target gps --render
```

安装位可调整，不必手改 54 个组合：修改一款卫星的 mount 配置，运行 `scripts/assets/maintain.py --satellite <id> --apply`，会重新生成该卫星的两个机械臂组合并同步安装帧、边界和校准状态。也可通过 `--xyz`、`--rpy` 指定新值；不加 `--apply` 为预览。改动前工具自动备份其负责的文件，失败时回滚。

卫星上的安装位、机械臂相对安装位的底座偏移、场景初始位姿、目标抓取点是四类独立参数。移动安装位后，原支撑垫的贴合形状可能需要重新制作；工具不会假定原碰撞检查仍有效。替换卫星 mesh 后也必须重新检查支撑、抓取点、可达性与碰撞。完整参数地图及流程见 skill 的 `references/operations.md`。

## 验证范围

2026-11-10 原始替换版完成 54 组合零臂姿态、夹爪开/闭端点表面相交检查。两款机械臂分别完成 CPU Genesis 加载、按名称初始化关节、两步仿真与渲染；峰值 RSS 约 1.64/1.54 GiB。日志摘要见本目录 JSON。后续编辑会撤销检查状态，这些历史记录不能替代新版本评测。

这套入口目前只做固定基座、关闭碰撞的分离场景检查，`task_ready=false`。抓取点仍是候选体表位置；并未验证接触抓取成功率或自由漂浮动力学。旧 8 关节策略/数据不能直接用于新 10 关节模型。

## 后续

与之前 RL + StarVLA 自训练工作的结合只记录在 [FUTURE_TRAINING.md](FUTURE_TRAINING.md)，本分支不启动训练，也不宣称已经实现任意场景自主学习。下一阶段需先确认任务定义、动力学、TCP、动作/观测契约、随机化范围和评测集，再集成学习流程。
