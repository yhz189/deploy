# AGENTS.md — 给 AI 编码助手的约定文件

> 本文件是 Codex / AI 助手接手本项目时自动加载的规则文件。
> **详细背景、架构、知识点见 `PROJECT_CONTEXT.md`（必读）**，本文件只列核心规则。

## 项目一句话

RK3568 嵌入式智能冰箱：摄像头 + YOLOv8 RKNN 推理，自动识别食材放入/取出，
维护 SQLite 库存，手机经局域网查看（Flask 网页 + 安卓 App）。研电赛参赛项目。

## 开始任何任务前

1. **先读 `PROJECT_CONTEXT.md`** —— 完整背景、算法、模块、坑、约束都在里面。
2. 不确定的地方先问，不要猜着改。

## 绝对约束（违反会导致系统损坏，不可碰）

- `utils.CLASSES` 的 19 类内容与顺序，必须与 `models/fridge_yolo_v2.rknn` 训练标签一致。
- `IMG_SIZE = 640`、`utils.postprocess()` 的张量解析逻辑，与模型输出 `(1,23,8400,1)` 绑定。
- 事件系统是**增量语义**（PUT_IN 累加 / TAKE_OUT 递减），不可改成绝对值覆盖。
- `main.py` 的 seed 双路逻辑：`inv.has_stock()` 为真时只恢复位置记忆、不写 DB —— 不可退回无条件播种。
- `event_detector.update()` 返回**二元组** `(events, state)`。
- 板端用 `RKNNLite`（不是 `RKNN`）；`init_runtime()` 不带 `core_mask` 参数（RK3568 不支持）。
- `main.py` 必须在 `/root/2_deploy` 目录运行（`inventory.db` 是相对路径）。
- REST API 路径与 JSON 结构（`/api/stock`、`/api/events`、`/camera`、`/api/stock/adjust`）
  已被安卓 App 依赖，改动需同步 `mock_server.py`。

## 验证要求

- 改任何核心模块（utils / event_detector / change_locator / inventory）后，
  **必须运行 `python -m pytest tests/ -q`**，当前基线 **25 passed**。
- 不要声称"完成"或"修复"，除非验证命令实际通过并贴出输出。

## 运行命令

```bash
# 板子（/root/2_deploy 目录，conda 环境 rknn）
python3 main.py                 # 实时
python3 main.py --show          # 实时 + HDMI 显示窗口
python3 main.py --replay <目录>  # 回放 jpg 序列
python3 web_server.py           # Web 服务（独立进程）

# PC（安卓 App 开发用 Mock 后端）
python mock_server.py

# 测试
python -m pytest tests/ -q
```

## 协作偏好

- **用中文沟通。**
- 改完文件后，直接给出 scp 部署命令供复制粘贴，格式：
  ```
  scp "C:\Users\hp-pc\Desktop\deploy\<文件>" root@192.168.3.100:/root/2_deploy/
  ```
- 网络：开发板 `192.168.3.100:5000`，开发 PC `192.168.3.5:5000`。
- 当前分支 `feature/stability-optimization`，未合并 master。

## 当前待办（详见 PROJECT_CONTEXT.md 第 7、9 章）

- 板子端到端实测（v2 模型 + 全部改动用真实蔬果跑通）
- 依实测调参（diff_thresh / min_area / area_thresh / IOU_MATCH_THRESH）
- 清理 `main.py` 里的 `[DEBUG]` 打印
- 录演示视频
- 安卓 App（Codex 负责，背景见 `docs/android_app_brief.md`）
