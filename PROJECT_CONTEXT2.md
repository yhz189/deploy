# PROJECT_CONTEXT2.md — AI 交接文档

> 面向接手本项目的 AI 助手。读完本文件应能立即接手，无需额外上下文。
> 最后更新：2026-05-22

---

## 0. 一句话项目定位

研电赛（小米赛题五）嵌入式智能冰箱：RK3568 开发板 + USB 摄像头，
通过运动门控 + 图像差分 + YOLOv8 RKNN 推理，自动识别食材放入/取出并维护
SQLite 库存，手机经局域网 HTTP 查看（网页 + 安卓 App）。

---

## 1. 硬件 / 软件环境

### 硬件
- 开发板：iTOP-RK3568（RK3568 SoC，含 NPU）
- 输入：USB 摄像头（接冰箱内部），`cv2.VideoCapture(0)`，640×480
- 输出：HDMI 显示器（用于 `--show` 实时窗口调试）
- 网络：局域网 WiFi/有线，单冰箱场景，无云端

### 软件（板子端）
- 系统：开发板自带 Linux
- Python：3.9（板子 rknn 环境；仓库 `__pycache__` 另见 3.8 残留，以 3.9 为准）
- 推理：`rknnlite.api.RKNNLite`（rknn-toolkit2 **1.4.0**，板子端 runtime）
- 依赖：OpenCV (`cv2`)、NumPy、Flask、SQLite3（标准库）
- 模型转换在 PC 端的 `rknn` conda 环境完成（rknn-toolkit2 1.4.0）

### 网络地址（重要）
| 角色 | 地址 |
|------|------|
| 开发板 | `192.168.3.100`，端口 `5000` |
| 开发 PC（Mock 后端） | `192.168.3.5`，端口 `5000` |
| 板子部署目录 | `/root/2_deploy/` |
| 预览图路径（板子） | `/tmp/fridge_latest.jpg` |

### 代码仓库
- 本地：`C:\Users\hp-pc\Desktop\deploy`（Windows 开发 PC）
- 当前分支：`feature/stability-optimization`（**尚未合并到 master**）
- 计划：板子实测通过后合并到 master，并推 GitHub（由协作者「小龙虾」处理）

### scp 部署命令格式（用户偏好，每次改完文件直接给）
```
scp "C:\Users\hp-pc\Desktop\deploy\<文件>" root@192.168.3.100:/root/2_deploy/
```

---

## 2. 当前模型信息

- 文件：`models/fridge_yolo_v2.rknn` ← **当前唯一生效模型**
- 来源：闲鱼采购（¥10），卖家提供训练好的 YOLOv8n（`yolov8_model/weights/best.pt`）
- 卖家声称指标：mAP@0.5 ≈ 0.94，19 类水果蔬菜，训练 imgsz=640、epochs=100
- 转换链路：`best.pt` →（ultralytics export，`opset=11`，`simplify=True`）→ ONNX →（rknn-toolkit2 1.4.0）→ `.rknn`
- 输入尺寸：640×640，fp16
- 模型输出张量：`(1, 23, 8400, 1)` = 4 坐标(x1y1x2y2，640 尺度) + 19 类得分
- 旧模型 `models/fridge_yolo_fp16.rknn`、`fridge_yolo_int8.rknn` 为 17 类老模型，**已弃用**

### CLASSES 列表（19 类，顺序绝对不可改）
```python
CLASSES = [
    'apple', 'Onion', 'banana', 'garlic', 'pear',
    'orange', 'Capsicum', 'Beet', 'Tomato', 'Cucumber',
    'carrot', 'Eggplant', 'Cabbage', 'Potato', 'Zucchini',
    'pineapple', 'Garlic', 'Cauliflower', 'Calabash',
]
```
索引 0–18 必须与 `fridge_yolo_v2.rknn` 训练标签严格对应。

### 粗分类映射（COARSE_MAP）
- `水果`：apple, banana, pear, orange, pineapple（5 类）
- `蔬菜`：其余 14 类（Onion, garlic, Capsicum, Beet, Tomato, Cucumber, carrot, Eggplant, Cabbage, Potato, Zucchini, Garlic, Cauliflower, Calabash）
- 注：`garlic` 与 `Garlic` 是同物，数据集大小写重复，保留原样。

### 模型识别效果现状（如实记录）
- 旧 17 类模型：仅 1000 余张图训练，静止物体都常识别不出，**已废弃**。
- v2 模型：静止物体识别明显改善；但实测香蕉裁剪图置信度仅约 0.22（故 `CONF_THRESH` 调到 0.20）。
- 已知弱点：紧挨着的同类物体会被 NMS 合并，导致计数偏少。
- **尚未在板子上用真实蔬果做完整端到端实测**。

---

## 3. 已解决的问题

| # | 问题 | 解决方式 |
|---|------|---------|
| 1 | 旧模型识别极差 | 换 v2 模型（19 类，CLASSES/COARSE_MAP/IMG_SIZE=640 全部更新） |
| 2 | RKNN 转换报错 `unsupported MaxPool attribute 'dilations'` | rknn-toolkit2 1.4.0 不支持 opset 12，ONNX 导出改 `opset=11` |
| 3 | convert.py 中 ONNX 路径填错 | 用户已自行修正 |
| 4 | 取出物体不减库存 | ①`CONF_THRESH` 0.30→0.20；②`change_locator` 两边都识别失败时返回 `NOISE` 区域而非丢弃，`event_detector` 用 `placed_items` 位置记忆 IoU 兜底派生 `TAKE_OUT` |
| 5 | 一串香蕉只记 1 个 | `identify_crop` 增加 `count` 字段（同类检测框数），`_put_in` 按 count 记账 |
| 6 | 看不到实时推理画面 | `main.py` 加 `--show`，`cv2.imshow` 窗口叠加状态/帧号/事件/变化框，按 q 退出 |
| 7 | mock_server 加事件库存不变 | 加 `_apply_to_stock()`，`/dev/add_event` 同步更新库存；`/dev/reset` 同时重置库存与事件 |
| 8 | 用户手动改的数量被重启时的自动识别覆盖 | ①`inventory.has_stock()`；②`main.py` seed 拆分：DB 空才入账，DB 非空只恢复位置记忆；③加 `POST /api/stock/adjust` + `inventory.adjust_quantity()`，写 `MANUAL_ADJUST` 事件 |

---

## 4. 未解决 / 待办

| 优先级 | 事项 | 说明 |
|--------|------|------|
| 高 | 板子端到端实测 | 新模型 + 全部改动未在真板子用真实蔬果跑通完整流程 |
| 高 | 参数调优 | `diff_thresh`/`min_area`(change_locator)、`diff_thresh`/`area_thresh`(motion)、`IOU_MATCH_THRESH` 依实测数据调 |
| 中 | 清理 DEBUG 打印 | `main.py` 的 `classify_fn`/`locate_fn` 内仍有 `[DEBUG]` 打印，演示前清掉 |
| 中 | NMS 合并紧贴物体 | 多个同类物体挨太近被合并，计数偏少；暂以「物体摆开」规避 |
| 中 | 演示视频 | 工位 + 白盒子假冰箱方案录制 |
| 低 | 合并到 master | 实测通过后合并并推 GitHub |
| 低 | `postprocess` 注释过时 | utils.py `postprocess` docstring/注释仍写 17/21/3549（旧模型遗留），逻辑正确，注释可顺手更正 |
| 进行中 | 安卓 App | 已交给 Codex（Kotlin+Compose+Material3），背景见 `docs/android_app_brief.md` |

---

## 5. 绝对不能破坏的约束

1. **`utils.CLASSES` 列表的内容与顺序**——与 `fridge_yolo_v2.rknn` 标签绑定，改动 = 全部识别错位。
2. **`IMG_SIZE = 640`**——模型输入尺寸，与 rknn 模型绑定。
3. **`utils.postprocess()` 的张量解析逻辑**——与输出 `(1,23,8400,1)` 布局绑定（注释可改，解析代码不可动）。
4. **事件系统的增量语义**——`PUT_IN` 累加、`TAKE_OUT` 递减，绝不可改成「绝对值覆盖」，否则用户手动修正会被冲掉。
5. **seed 的双路逻辑**——`main.py` 中 `inv.has_stock()` 为真时**只恢复位置记忆、不写 DB**；这是保护用户手动修正数据的关键，不可退回到「无条件 PUT_IN 播种」。
6. **板子双进程架构**——`main.py` 与 `web_server.py` 各自独立进程，经 `inventory.db` 与 `/tmp/fridge_latest.jpg` 文件解耦，不要改成进程内直接调用。
7. **REST API 契约**——`/api/stock`、`/api/events`、`/camera`、`/api/stock/adjust` 的路径与 JSON 结构已被安卓 App 依赖，改动需同步 `mock_server.py` 与 App。
8. **改任何核心模块（utils/event_detector/change_locator/inventory）后必须重跑 `pytest`**，当前基线 **25 passed**。

---

## 6. 关键参数（实测调过，勿随意改）

| 参数 | 值 | 文件 |
|------|-----|------|
| `IMG_SIZE` | 640 | utils.py |
| `CONF_THRESH` | 0.20 | utils.py（为 v2 实测调低，调高会漏取出事件） |
| `NMS_THRESH` | 0.45 | utils.py |
| `find_change_regions` `diff_thresh` / `min_area` | 30 / 400 | change_locator.py |
| `is_moving` `diff_thresh` / `area_thresh` | 25 / 3000 | motion.py |
| `enter_frames` / `exit_frames` / `settle_frames` | 3 / 10 / 5 | event_detector.py |
| `PARTIAL_AREA_RATIO` | 0.7 | event_detector.py |
| `IOU_MATCH_THRESH` | 0.3 | event_detector.py |

---

## 7. 运行命令与部署步骤

### 板子端运行（需开两个终端 / 两个进程）
```bash
cd /root/2_deploy
# 终端 1：识别 + 记账主程序
python3 main.py                       # 实时
python3 main.py --show                # 实时 + HDMI 显示窗口（按 q 退出）
python3 main.py --replay results/clip # 回放某目录的 jpg 序列

# 终端 2：Web 服务（网页 + App API）
python3 web_server.py
```
手机浏览器访问 `http://192.168.3.100:5000`。

### PC 端（安卓 App 开发用 Mock 后端）
```bash
python mock_server.py        # PC 上跑，监听 0.0.0.0:5000
# App 设置里填 PC IP 192.168.3.5:5000
# 手动造数据：浏览器访问 http://localhost:5000/dev/add_event/PUT_IN/apple?n=3
# 重置：http://localhost:5000/dev/reset
```

### 测试
```bash
cd C:\Users\hp-pc\Desktop\deploy
python -m pytest tests/ -q          # 基线：25 passed
```

### 部署（改完文件传板子）
```
scp "C:\Users\hp-pc\Desktop\deploy\<文件>" root@192.168.3.100:/root/2_deploy/
```

---

## 8. REST API 契约

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/stock` | 库存数组：`[{name,qty,first_in,last_update}]` |
| GET | `/api/events` | 最近 20 条：`[{time,type,note}]`，倒序 |
| GET | `/camera` | `image/jpeg`，建议加 `?t=<时间戳>` 防缓存 |
| POST | `/api/stock/adjust` | body `{"name":"banana","qty":4}`；200 `{ok:true,msg}` / 404 `{ok:false,error}` |

事件 `type` 取值：`PUT_IN` / `TAKE_OUT` / `PARTIAL_TAKE_OUT` / `MANUAL_ADJUST`。

---

## 9. 架构与数据流（速览）

```
摄像头 → main.py(open_source) → EventDetector.update()
  → motion.is_moving 门控 → 三态状态机 STABLE/BUSY/SETTLING
  → SETTLING 稳定 → _analyze → change_locator(find+classify)
  → utils.identify_crop(RKNN 推理) → ChangedRegion[]
  → _analyze_regions → 事件 → inventory.process_event → SQLite
                                main.py → /tmp/fridge_latest.jpg
─────────── 文件解耦 ───────────
web_server.py 读 SQLite + 预览图 → HTML / JSON API → 手机 / App
```

`ChangedRegion.kind`：`APPEAR`(放入) / `DISAPPEAR`(取出) / `REPLACE`(更换) /
`SAME`(同位同粗类，按面积比判部分取出/追加) / `NOISE`(两边识别失败，走位置记忆兜底)。

---

## 10. 下一步建议（接手后优先做）

1. **板子实测**：部署 v2 模型与全部改动，用真实蔬果跑完整「放入→取出→部分取出」流程，验证事件准确率。
2. **依实测调参**：重点 `change_locator` 的 `diff_thresh/min_area` 与 `event_detector` 的 `IOU_MATCH_THRESH`。
3. **清理 `main.py` 的 `[DEBUG]` 打印**（演示前）。
4. **验证 `/api/stock/adjust` 与 seed 双路逻辑**：手动改数 → 重启板子 → 确认数据未被覆盖。
5. **录演示视频**：工位 + 白盒子假冰箱。
6. 全部通过后 `pytest` 复核 → 合并 `feature/stability-optimization` 到 master。

---

## 11. 用户背景与协作偏好

- 用户：研一学生，嵌入式 Linux 方向，RK3568 开发板。
- 每次修改文件后，直接给出 scp 命令供复制粘贴（格式见第 1 节）。
- 安卓 App 由用户用 Codex/ChatGPT 做 vibe coding，本项目窗口聚焦端侧部署与核心算法。
- 沟通用中文。
