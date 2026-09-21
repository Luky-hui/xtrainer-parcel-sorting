# 来源与整理记录

整理日期：2026-09-21。

上游地址：https://gitee.com/dobot-opensource/Guangdong_Collegiate_Computing_Competition
本机上游 HEAD：`41758269ab71cc2c37e8c8b66a9b5eeb1b98bf4c`。

来源为本机 `official_sources_20260429/Guangdong_Collegiate_Computing_Competition/x-trainer` 的工作目录快照，**并非干净的上游 commit 导出**。已有改动包含机械臂配置、回放、推理和策略客户端及文档演示资源，均随本次整理保留。同时保留已有的 convert_task3_0428_success.py 转换脚本。

本目录包含 task3 的任务代码和场景，以及公共 source、scripts、机器人资产、文档资源和原许可。其他任务的专用目录和场景没有复制。公共工具中可能仍出现其他任务名称，这些兼容分支保留原样。

新增 README、环境配置示例、run.sh 和 smoke_test.py；调整 gitignore 使资产可提交，并为 assets 全目录配置 Git LFS。为修复原 pyproject 的 README 引用，补充 source/leisaac/README.md。README.upstream.md 为来源 README 的原文副本。

未复制：原 Git 历史、无目标检出的 .gitmodules、IDE 配置、缓存、日志、数据集、checkpoints 占位文件、VR PEM 证书与私钥、重复的第三方 SDK 7z 压缩包。解压后的 SDK 源码及其许可保留。

docker/ 保留原项目的仿真环境示例，其中主机路径和镜像名需要调整；它不是已有训练模型镜像。数据转换脚本里的演示路径和参数也需要按新数据修改。

运行依赖记录：Isaac Sim 4.5.0；本机 Isaac Lab VERSION 2.3.2，commit a859a5f。此次整理不包含这两个大型外部依赖，不附带 ACT 模型和数据集。

验证结果见 VALIDATION.md；逐文件校验和见 SHA256SUMS。

2026-09-21 补充 TECHNICAL.md：依据当前源码记录架构、接口、任务判定、数据流程、赛题评分差异和已知限制。文档编写进行了判定函数隔离验证，没有更改原 Python 实现。

## 2026-09-21 GitHub 上传整理

保留源码和资产原路径。分组整理 `.gitignore`；将两份含明文密码的硬件 INI 限定为本地文件，新增清空密码和相机序列号的 `dobot_settings.example.ini`。补充 Docker 构建上下文的本地配置排除项，文本属性固定为 LF，新增 `GITHUB_UPLOAD.md` 并更新主 README。原本机文件没有删除。`SHA256SUMS` 改为仅覆盖实际上传集合；原 Python 源码和仿真资产未改动。
