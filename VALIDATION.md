# 验证记录

日期：2026-09-21。测试对象为本整理目录内的代码与资产。

- Python 语法检查通过：116 个文件（不含 third_party；原 DynamixelSDK 示例含 Python 2 语法，按原样保留）。
- `bash -n run.sh` 与 `bash run.sh --help` 通过。
- 本任务源码、场景及机械臂资产与来源逐文件比较一致；缩略图和缓存排除。
- Git 配置检查通过：场景与机械臂资产没有被 gitignore 排除，并使用 LFS。
- 私钥、local.env 和运行日志被 gitignore 排除；打包内容不含 PEM / KEY 文件。
- 新增目录中的单文件均未超过 100 MiB；资产仍统一使用 LFS。
- 在 Isaac Sim 4.5.0 + 本机 Isaac Lab 2.3.2 源码环境中运行 headless smoke，180 秒超时限制，进程返回码 0。
- smoke 内确认导入的 leisaac.tasks 来自本整理目录。

自动解析的结果：

```json
{"task": "task3", "status": "passed", "reset": true, "steps": 3, "action_shape": [1, 16], "policy_evaluated": false, "cameras_enabled": false}
```

原始日志保存在本地 `logs/smoke_*.log`，不会随 Git 提交。日志存在 Isaac 扩展、OmniHub 和机械臂碰撞网格警告，本次检查正常退出，未出现 traceback。该检查只覆盖环境创建、reset 和 3 步动力学执行，未覆盖相机画面、遥操作交互、模型推理、任务完成或比赛得分。

可在配置环境后使用 `bash run.sh smoke` 重跑。使用 `sha256sum -c SHA256SUMS` 验证整理文件；校验清单不含自身、本机 local.env、运行日志和缓存。

## 技术文档核对

2026-09-21 新增 [TECHNICAL.md](TECHNICAL.md)，核对任务配置、判定、几何辅助函数及共享控制、观测、采集、推理流程。所有文档源码链接存在，Bash 示例通过 `bash -n`。

对源码中的 `task_done` 和 `is_xtrainer_at_rest_pose` 做了 CPU Torch 隔离调用，以模拟检测点验证区域聚合与 rest pose 行为，具体输入条件及结果见技术文档。该检查没有启动物理引擎，不覆盖视觉中心求解或模型执行。

本次文档更新前后的全部 Python 源码校验和一致，未修改仿真或策略代码。未为纯文档变更重复运行 Isaac Sim；上方 smoke 结果仍指本次整理时的测试。

## 2026-09-21 上传集合检查

本次整理仅针对文件发布边界与说明文档。通过独立临时 Git 索引核对实际上传文件和忽略文件，并检查 Git LFS 对资产的暂存结果；逐文件校验原 Python 源码和资产内容没有变化。已排除 `local.env`、日志、系统杂文件及两份含密码的硬件 INI。更新后的 `SHA256SUMS` 只覆盖上传文件（不含清单自身）。没有在本次 Windows 整理过程中重跑 Isaac Sim，前文 smoke 和隔离调用结果为已有记录。
