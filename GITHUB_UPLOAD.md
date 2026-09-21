# GitHub 上传说明：xtrainer-task3

整理日期：2026-09-21。按独立仓库保留原目录结构，避免移动文件后破坏 Python 导入、USD 引用和说明链接。

## 上传哪些文件

| 路径 | 处理 | 原因 |
| --- | --- | --- |
| `source/` | 上传源码、依赖描述、第三方源码和许可；两份本地硬件 INI 除外 | 仿真与工具的实现 |
| `scripts/`、`run.sh` | 上传 | 遥操作、评测、采集与数据转换入口 |
| `assets/robots/` | 上传，使用 Git LFS | 机械臂模型及其引用资源 |
| `assets/scenes/task3/` | 上传，使用 Git LFS | 本任务场景、对象及配套纹理 |
| `assets/docs/` | 保留，使用 Git LFS | 原项目说明配图和演示动图；属于文档资源 |
| `docker/` | 上传 | 原项目容器示例，使用前检查主机路径和镜像名 |
| `README.md`、`README.upstream.md`、`TECHNICAL.md`、`PROVENANCE.md`、`VALIDATION.md`、`CHANGELOG.md` | 上传 | 使用说明、来源和已有验证记录 |
| `LICENSE`、第三方目录中的许可 | 上传 | 保留原版权与授权声明 |
| `env.example`、`dobot_settings.example.ini` | 上传 | 不含原密码和相机序列号的配置入口 |
| `.gitignore`、`.gitattributes`、`.dockerignore`、`.flake8`、`.pre-commit-config.yaml` | 上传 | 项目规则与原开发工具配置 |
| `SHA256SUMS`、`GITHUB_UPLOAD.md` | 上传 | 文件校验与上传说明 |

## 哪些留在本地

- `local.env`：本机 Isaac Sim / Isaac Lab 路径。
- `source/leisaac/leisaac/xtrainer_utils/utils/dobot_config/dobot_settings.ini`：含明文 `passcode`、相机序列号和硬件参数。
- `source/leisaac/leisaac/xtrainer_utils/utils/dobot_config/dobot_settings copy.ini`：同样含上述本地信息，且与主配置的标定参数不同；保留本地文件，不当作重复文件删除。
- `logs/`、`.DS_Store`：运行日志和系统杂文件。
- 缓存、虚拟环境、数据集、训练权重、录制结果、私钥和证书继续按 `.gitignore` 排除；本次目录中不存在的类别只是保留原有忽略规则。

`.gitignore` 控制 Git 提交，不会在浏览器手动拖拽文件时自动过滤。请从项目根目录使用 Git + Git LFS 提交。

## 新机器配置

仿真按主 README 从 `env.example` 创建 `local.env`。使用读取硬件 INI 的实体设备工具前，在项目根目录的 Git Bash / Linux 终端运行：

```bash
cp source/leisaac/leisaac/xtrainer_utils/utils/dobot_config/dobot_settings.example.ini source/leisaac/leisaac/xtrainer_utils/utils/dobot_config/dobot_settings.ini
```

然后在本地填写 `COMPUTER.passcode`、`CAMERA.top`、`CAMERA.left`、`CAMERA.right`，并按设备核对串口、关节偏移与夹爪标定。示例中的其他数值原样取自已有主配置，不代表新设备的校准结果。

## 首次上传

两个目录分别初始化、分别上传；不混合不同任务的场景。下面的命令在所选项目根目录执行：

```bash
git init -b main
git lfs install --local
git add .
git lfs ls-files
git status --short
git commit -m "Import task3 source and simulation assets"
```

创建 GitHub 仓库后，将其真实 URL 填入主 README 中的 `git remote add origin` 命令，再执行 `git push -u origin main`。本次整理没有创建远程仓库、提交或推送。

克隆后运行 `git lfs pull` 获取资产，再运行 `sha256sum -c SHA256SUMS`。校验清单对应真实资产内容，不是 LFS 指针；清单排除自身及全部忽略文件。`.gitattributes` 将文本工作副本固定为 LF，保证跨平台校验一致。

Git LFS 配置依据：[GitHub 官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage)。

## 已有配置的限制

原 `.pre-commit-config.yaml` 的两个 `insert-license` 钩子引用 `.github/LICENSE_HEADER.txt` 和 `.github/LICENSE_HEADER_MIMIC.txt`，这两个文件不在本目录中。本次保留原配置供查阅；原配置不能直接作为已验证的提交检查使用，也没有运行会批量改写源码的格式化工具。

本次只验证上传集合、忽略规则、LFS 暂存结果及文件校验，没有重跑 Isaac Sim 仿真。`VALIDATION.md` 原有 smoke 结果属于此前的验证记录。
