# 冰箱食材识别与管理系统 — 项目核心记忆文件

> 本文件是所有 AI 接手本项目时必须首先阅读的上下文文件。
> 最后更新：2026-05-22
> 当前阶段：核心算法已重构跑通，v2 新模型已部署，待板子端到端实测 + 调参 + 录演示视频
>
> ⚠️ 重要：本文件已根据**当前最终版代码**全面更新。早期版本描述的
> 「IDLE/WATCHING/WAITING 状态机 + 数量对比」算法**已废弃**，现在用的是
> 「三态状态机 + 图像差分变化定位」，详见第 3、4 章。

---

## 0. 快速定位

| 我要做什么 | 看哪个章节 |
|---|---|
| 了解项目是什么 | 第 1 章 |
| 了解核心算法（差分变化定位） | 第 3 章 |
| 修改识别类别 | 第 4 章 utils.py |
| 修改事件判断逻辑 | 第 4 章 event_detector.py |
| 修改库存逻辑 | 第 4 章 inventory.py |
| 修改 Web 界面 / API | 第 4 章 web_server.py |
| 安卓 App 开发 | 第 4 章 mock_server.py + `docs/android_app_brief.md` |
| 转换新模型 | 第 5 章 模型转换流程 |
| 排查 Bug | 第 6、7 章 |
| 了解下一步做什么 | 第 9 章 |
| 绝对不能碰的地方 | 第 10 章 |

---

## 1. 项目概述

### 基本信息
- **项目名称**：冰箱食材识别与管理系统
- **赛事**：全国研究生电子设计竞赛（研电赛），小米赛题五
- **开发者背景**：研究生在读，方向嵌入式 Linux 应用/驱动
- **剩余时间**：约 1 个月（截止答辩）

### 项目目标
在 RK3568 嵌入式开发板上部署完整的冰箱食材识别与库存管理系统，
实现「摄像头采集 → 事件识别 → 库存更新 → 手机展示」的完整闭环。

### 评分重点
1. **系统完整性**：链路必须跑通
2. **真实事件识别**：区分放入/取出/部分取出/无效干扰（核心难点）
3. **物体识别与分类**：稳定优先，不要求极高精度
4. **库存管理与更新**：数量变化正确同步
5. **加分项**：低算力实时性、复杂光照稳定性、多目标进出、嵌入式部署细节

### 核心功能
1. 摄像头实时采集
2. 运动门控 + 图像差分变化定位
3. YOLOv8n 食材识别（RK3568 NPU，RKNN 推理）
4. 三态状态机事件判断（PUT_IN / TAKE_OUT / PARTIAL_TAKE_OUT / EXCHANGE）
5. SQLite 库存与事件持久化
6. Flask Web 界面 + REST API（手机网页 / 安卓 App）
7. 用户手动修正库存数量

---

## 2. 硬件与环境

### 硬件配置
- 开发板：迅为 iTOP-RK3568
- CPU：Quad-core Cortex-A55 @ 2.0GHz
- NPU：0.8 TOPS，支持 RKNN 推理
- 系统：Linux
- 连接：SSH 远程 + HDMI 显示器（`--show` 实时窗口用）
- 摄像头：OV5695，设备节点 `/dev/video0`，`cv2.VideoCapture(0)`
- 采集分辨率：640×480

### 网络地址（重要）
| 角色 | 地址 |
|------|------|
| 开发板 | `192.168.3.100`，Web 端口 `5000` |
| 开发 PC（Mock 后端） | `192.168.3.5`，端口 `5000` |
| 板子部署目录 | `/root/2_deploy/`（⚠️ 已从旧版 `/deploy` 迁移） |
| 板子预览图路径 | `/tmp/fridge_latest.jpg` |

### 软件环境（开发板）
- conda 环境名：`rknn`
- Python：3.9
- 关键包：
  - `rknn-toolkit-lite2`（板端推理，**注意是 lite 版本**）
  - `opencv-python`（含 Qt，有字体警告但功能正常）
  - `flask`
  - `numpy`
  - `sqlite3`（Python 内置）

### 软件环境（模型转换，PC / Ubuntu）
- conda 环境名：`rknn`
- `rknn-toolkit2 v1.4.0`（⚠️ 版本固定，不要升级）
- `onnx`、`numpy`、`opencv-python-headless`

### 软件环境（Windows，模型训练 / 本地仓库）
- 本地仓库：`C:\Users\hp-pc\Desktop\deploy`
- 训练用 `ultralytics`（YOLOv8）—— 注：v2 模型为采购，训练环境一般用不到

### 代码仓库与分支
- 当前分支：`feature/stability-optimization`（**尚未合并到 master**）
- 计划：板子实测通过 → `pytest` 复核 → 合并 master → 推 GitHub（协作者「小龙虾」处理）

### 部署命令格式（用户偏好：改完文件直接给 scp 命令）
```
scp "C:\Users\hp-pc\Desktop\deploy\<文件>" root@192.168.3.100:/root/2_deploy/
```

---

## 3. 核心算法：门控 + 差分变化定位（⭐ 必读）

> ⚠️ 这是与早期版本最大的不同。早期是「每隔几帧全画面检测 → 统计前后数量差」，
> 现在是「运动门控 → 状态机 → 稳定后对差分区域定向识别」。

### 设计动机
低算力嵌入式 NPU 扛不住每帧全画面推理。所以：
1. 用**极轻量的运动检测**（灰度差分）做门控，绝大多数帧只做这个。
2. 只在画面**从忙碌恢复到稳定的那一刻**，做一次「参考图 vs 新图」的差分，
   只对**变化的局部区域**做 YOLO 识别。

### 三态状态机
```
STABLE（稳定）  ──连续 enter_frames 帧有运动──▶  BUSY（忙碌）
BUSY（忙碌）    ──连续 exit_frames 帧静止──────▶  SETTLING（沉降）
SETTLING（沉降）──连续 settle_frames 帧静止────▶  STABLE（触发分析）
SETTLING（沉降）──期间又检测到运动────────────▶  BUSY（回退）
```
- STABLE 期：持续把当前帧刷新为 `ref_frame`（参考图）。
- SETTLING 是为了过滤「人手中途停顿」造成的假稳定。
- SETTLING 稳定后：用 `ref_frame`（动作前）和 `settle_frame`（动作后）做分析。

### 变化定位与事件派生
```
ref_frame, new_frame
   │  change_locator.find_change_regions()  差分+形态学+轮廓 → 变化 bbox 列表
   ▼
change_locator.classify_regions()  逐区域裁剪 ref/new 两图做 YOLO 识别
   │  判定每个区域的 kind：
   │    APPEAR    : ref 认不出、new 认出      → 放入
   │    DISAPPEAR : ref 认出、new 认不出      → 取出
   │    REPLACE   : 两边粗类不同              → 更换
   │    SAME      : 两边粗类相同（按面积比判部分取出/追加/抖动）
   │    NOISE     : 两边都认不出 → 走位置记忆兜底
   ▼
event_detector._analyze_regions()  → 事件列表
   ├─ _cross_check：同粗类、尺寸相近的一出一进配对为「整理」，不计事件
   └─ NOISE 兜底：用 placed_items 位置记忆做 IoU 匹配派生 TAKE_OUT
```

### 位置记忆 placed_items
`EventDetector` 维护一个 `placed_items` 列表，每项记录某个物品的
`{id, class_id, fine, coarse, bbox}`。作用：
- 物体被拿走后那块变成空背景，模型识别不出 → 用位置记忆兜底判定 TAKE_OUT。
- 与库存数据库**职责分离**：位置记忆只管「东西在画面哪里」，
  数据库才是「有几个」的真值。

---

## 4. 代码结构与核心模块

### 目录结构
```
deploy/                          # 板子部署到 /root/2_deploy/
├── main.py                      # ⭐ 主程序：实时/回放主循环
├── motion.py                    # 运动门控（新增）
├── event_detector.py            # ⭐ 三态状态机 + 位置记忆 + 事件派生（已重构）
├── change_locator.py            # ⭐ 图像差分变化定位（新增）
├── utils.py                     # ⭐ YOLO 预处理/后处理 + 19 类 CLASSES
├── inventory.py                 # ⭐ SQLite 库存与事件管理
├── web_server.py                # Flask Web 服务（板子上独立进程）
├── mock_server.py               # 假后端：PC 上模拟板子接口（App 开发用，新增）
├── infer_image.py               # 单图推理调试工具
├── models/
│   ├── fridge_yolo_v2.rknn      # ★ 当前生效模型（19 类）
│   ├── fridge_yolo_fp16.rknn    # 旧 17 类模型（已弃用）
│   └── fridge_yolo_int8.rknn    # 旧 17 类模型（已弃用）
├── tests/                       # pytest 测试（替代旧的 test_event.py）
│   ├── test_utils.py
│   ├── test_motion.py
│   ├── test_change_locator.py
│   └── test_event_detector.py
├── docs/
│   ├── android_app_brief.md     # 安卓 App 交接文档
│   ├── 项目复盘_白话版.md
│   └── 代码结构分析.md
├── yolov8_model/                # 采购的训练产物存档（运行时不参与）
├── inventory.db                 # SQLite（相对路径，运行时生成，.gitignore 忽略）
└── PROJECT_CONTEXT.md / PROJECT_CONTEXT2.md   # 本文件 + AI 交接文档
```

### utils.py — 最关键文件

```python
# 19 类清单（顺序严格对应 class_id 0–18，绝对不可改）
CLASSES = [
    'apple', 'Onion', 'banana', 'garlic', 'pear',
    'orange', 'Capsicum', 'Beet', 'Tomato', 'Cucumber',
    'carrot', 'Eggplant', 'Cabbage', 'Potato', 'Zucchini',
    'pineapple', 'Garlic', 'Cauliflower', 'Calabash',
]

# 粗分类
COARSE_MAP → 水果: apple/banana/pear/orange/pineapple
             蔬菜: 其余 14 类
# 注：garlic 与 Garlic 是同物，数据集大小写重复，保留原样

# 关键常量
IMG_SIZE   = 640    # ⚠️ 模型输入尺寸，与 rknn 模型绑定，改了要重新转换
CONF_THRESH = 0.20  # 为 v2 模型实测调低（香蕉裁剪图置信度仅约 0.22），调高会漏取出事件
NMS_THRESH  = 0.45

# 核心函数
letterbox / preprocess  # 保持宽高比缩放+padding，BGR→RGB
postprocess(outputs, ratio, pad, orig_shape)
  # 模型输出 (1, 23, 8400, 1) = 4 坐标(x1y1x2y2,640尺度) + 19 类得分
  # ⚠️ 函数内注释仍写着旧的 17/21/3549，注释过时但解析逻辑正确
identify_crop(model, crop)
  # 对裁剪图识别，返回 {class_id, fine, coarse, conf, area, count}
  # count = 同类检测框个数，支持「一串香蕉」按个数计
```

### event_detector.py — 核心算法（已完全重构）

```python
class EventDetector:
    EventDetector(motion_fn, locate_fn,
                  enter_frames=3, exit_frames=10, settle_frames=5)
    # motion_fn = motion.is_moving
    # locate_fn = main.py 注入，内部 = find_change_regions + classify_regions

    seed(frame, detections)   # 开机播种参考帧 + placed_items 位置记忆
    update(frame) -> (events, state)
        # ⚠️ 返回二元组 (events, state)
        #    events 为 [(event_type, details), ...]，可为空列表
        #    （早期版本返回三元组，已废弃）

# 关键常量
PARTIAL_AREA_RATIO   = 0.7   # SAME 区域面积比阈值，判部分取出/追加
IOU_MATCH_THRESH     = 0.3   # 位置记忆匹配 IoU 阈值
SIZE_MATCH_TOLERANCE = 0.3   # _cross_check 尺寸相近判定容差

# last_regions：最近一次分析的变化区域，供 main.py --show 可视化读取
```

### change_locator.py — 变化定位（新增）

```python
@dataclass ChangedRegion: bbox, kind, ref_ident, new_ident

find_change_regions(ref, new, diff_thresh=30, min_area=400)
  # 灰度差分 → 阈值化 → 形态学开闭运算 → 轮廓 → 变化 bbox 列表

classify_regions(ref, new, bboxes, classify_fn)
  # 逐区域裁剪 ref/new 识别，判定 kind
  # 两边都识别失败 → 返回 NOISE 区域（早期是直接丢弃，会漏取出事件）
```

### motion.py — 运动门控（新增）

```python
is_moving(prev, cur, diff_thresh=25, area_thresh=3000)
  # 两帧灰度差分，变化像素数 > area_thresh 即判定有运动
```

### inventory.py — 库存管理

```python
class InventoryManager:
    # DB_PATH='inventory.db'（相对路径，必须在 /root/2_deploy 目录运行）
    # check_same_thread=False（允许 Flask 线程访问）
    # 两张表：inventory（库存）、events（事件历史）

    process_event(event_type, details)
      # PUT_IN / TAKE_OUT / PARTIAL_TAKE_OUT / EXCHANGE，增量式更新
      #   PUT_IN  → 有则数量叠加，无则新增记录
      #   TAKE_OUT/PARTIAL → 减少数量，为 0 时 status='out'
    has_stock()                  # 新增：DB 是否已有在库记录（判断是否首次启动）
    adjust_quantity(name, qty)   # 新增：用户手动修正数量，写 MANUAL_ADJUST 事件
    get_current_stock / get_recent_events / print_stock / close
```

### main.py — 主程序

```python
RKNN_MODEL = 'models/fridge_yolo_v2.rknn'   # 换模型改这里
CAMERA_ID  = 0
PREVIEW_PATH = '/tmp/fridge_latest.jpg'      # web_server 的 /camera 读这里
PREVIEW_INTERVAL  = 30                       # 每 30 帧存一次预览图
REGION_SHOW_FRAMES = 90                      # 分析后变化框持续显示帧数

# 命令行
python main.py                       # 实时
python main.py --show                # 实时 + HDMI 窗口（按 q 退出）
python main.py --replay results/clip # 回放某目录的 jpg 序列

# 初始化顺序
1. RKNNLite → load_rknn → init_runtime()   # ⚠️ 无参数，不能加 core_mask
2. InventoryManager()
3. EventDetector(is_moving, locate_fn)
4. open_source() 生成器（实时摄像头 / 回放目录）

# seed 双路逻辑（⭐ 保护用户手动修正，不可退回）
seed_detections(首帧) → detector.seed()  # 始终恢复位置记忆
if inv.has_stock():    # 重启：DB 已有数据 → 只恢复位置记忆，不写 DB
else:                  # 首次启动：DB 空 → 逐物品 PUT_IN 入账

# 主循环
每帧 detector.update(frame) → 收事件 → inv.process_event() → 每 30 帧存预览图
SHOW 时叠加 draw_overlay 并 imshow
```

### web_server.py — Web 服务（板子上独立进程，可安全改样式）

```python
GET  /                  # HTML 页面（库存表+事件表+摄像头，移动端响应式，3 秒整页刷新）
GET  /camera            # 返回 /tmp/fridge_latest.jpg
GET  /api/stock         # JSON: [{name, qty, first_in, last_update}]
GET  /api/events        # JSON: [{time, type, note}]，最近 20 条
POST /api/stock/adjust  # 新增：用户修正数量，body {"name","qty"}
                        #   200 {ok:true,msg} / 404 {ok:false,error}
python3 web_server.py   # 端口 5000，独立启动
```

### mock_server.py — 假后端（PC 上跑，安卓 App 开发用，新增）
模拟板子全部接口（`/api/stock`、`/api/events`、`/camera` 占位图、`/api/stock/adjust`），
外加 `/dev/add_event/<etype>/<food>?n=N`（造数据）和 `/dev/reset`（重置基线）。
让 App 开发期间不必一直开着板子。详见 `docs/android_app_brief.md`。

---

## 5. 运行方式

### 开发板完整启动流程
```bash
ssh root@192.168.3.100
conda activate rknn
cd /root/2_deploy        # ⚠️ 必须在此目录，inventory.db 是相对路径

# 双进程（tmux 分屏或后台）
tmux new -s fridge       # Ctrl+B % 分屏
#   左：python3 main.py   （或 python3 main.py --show 看 HDMI 窗口）
#   右：python3 web_server.py

# 后台方式
nohup python3 web_server.py > web.log 2>&1 &
python3 main.py

# 清库重来
rm -f inventory.db && python3 main.py
```
手机浏览器访问 `http://192.168.3.100:5000`。

### PC 端（安卓 App 开发用 Mock 后端）
```bash
python mock_server.py    # PC 上跑，监听 0.0.0.0:5000
# App 设置里填 PC IP 192.168.3.5:5000
```

### 测试（pytest，替代旧 test_event.py）
```bash
cd C:\Users\hp-pc\Desktop\deploy
python -m pytest tests/ -q      # 当前基线：25 passed
```

### 模型转换流程（v2 模型）
```
best.pt ──(ultralytics export, opset=11, simplify=True)──▶ best.onnx
best.onnx ──(rknn-toolkit2 v1.4.0, convert.py, fp16, target rk3568)──▶ fridge_yolo_v2.rknn
scp fridge_yolo_v2.rknn root@192.168.3.100:/root/2_deploy/models/
```
ONNX 导出（ultralytics）：
```python
from ultralytics import YOLO
m = YOLO('yolov8_model/weights/best.pt')
m.export(format='onnx', imgsz=640, opset=11, simplify=True, dynamic=False)
```
若误用 opset=12 导出，需手动删 MaxPool 的 dilations 属性：
```python
import onnx
model = onnx.load('best.onnx')
for node in model.graph.node:
    if node.op_type == 'MaxPool':
        keep = [a for a in node.attribute if a.name != 'dilations']
        del node.attribute[:]; node.attribute.extend(keep)
onnx.save(model, 'best_fixed.onnx')
```

---

## 6. 已解决的问题

| 问题 | 原因 | 解决方案 |
|---|---|---|
| 旧 17 类模型识别极差 | 仅 1000 余张图训练，静物都认不出 | 换 v2 模型（19 类，闲鱼采购 YOLOv8n，卖家称 mAP@0.5≈0.94） |
| MaxPool dilations 报错 | rknn v1.4.0 不支持 opset 12 | ONNX 导出用 opset=11，或手动删 dilations 属性 |
| convert.py 中 ONNX 路径填错 | 笔误 | 用户已自行修正 |
| 取出物体不减库存 | 拿走后变空背景，ref/new 两边都识别失败被丢弃 | ①CONF_THRESH 0.30→0.20；②change_locator 返回 NOISE 区域；③event_detector 用 placed_items 位置记忆 IoU 兜底派生 TAKE_OUT |
| 一串香蕉只记 1 个 | 未统计同类目标数 | identify_crop 增加 count 字段，_put_in 按 count 记账 |
| 看不到实时推理画面 | 无可视化 | main.py 加 --show，cv2.imshow 叠加状态/帧号/事件/变化框 |
| mock_server 加事件库存不变 | 只改事件没改库存 | 加 _apply_to_stock()，事件联动库存；/dev/reset 同时重置两者 |
| 用户改的数量被重启时自动识别覆盖 | 重启时 seed 无条件 PUT_IN 入账 | inv.has_stock() + seed 双路逻辑（DB 非空只恢复位置记忆）+ /api/stock/adjust 接口 |
| 重启后取出库存不更新（旧版遗留 bug） | 早期 seed 不恢复记忆，TAKE_OUT 找不到记录 | seed 双路逻辑：重启恢复位置记忆 + DB 保留，TAKE_OUT 可正常命中 |
| MaxPool dilations / core_mask / INT8 量化 / reorder_channel（旧版排查） | rknn v1.4.0 兼容性 | 见第 10 章模型转换约束 |

---

## 7. 未解决 / 待办

| 事项 | 现状 | 优先级 |
|---|---|---|
| 板子端到端实测 | v2 模型 + 全部改动未在真板子用真实蔬果跑完整流程 | 🔴 高 |
| 参数调优 | diff_thresh/min_area(change_locator)、diff_thresh/area_thresh(motion)、IOU_MATCH_THRESH 依实测调 | 🔴 高 |
| 清理 DEBUG 打印 | main.py 的 classify_fn/locate_fn 内仍有 [DEBUG] 打印，演示前清掉 | 🟡 中 |
| NMS 合并紧贴物体 | 同类物体挨太近被合并，计数偏少；暂以「物体摆开」规避 | 🟡 中 |
| postprocess 注释过时 | utils.py 注释仍写 17/21/3549，逻辑正确，可顺手更正 | 🟢 低 |
| 位置记忆与 DB 数量可能不一致 | 用户手动改数后两者会偏差；已知可接受（记忆只用于配对取出） | 🟢 低 |
| 演示视频 | 计划工位 + 白盒子假冰箱录制 | 🟡 中 |
| 合并到 master | 实测通过后合并并推 GitHub | 🟢 低 |
| 安卓 App | 已交给 Codex（Kotlin+Compose+Material3），背景见 docs/android_app_brief.md | 进行中 |

---

## 8. Web 界面与 API

### REST API 契约
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/stock` | `[{name,qty,first_in,last_update}]` |
| GET | `/api/events` | `[{time,type,note}]`，最近 20 条倒序 |
| GET | `/camera` | `image/jpeg`，建议加 `?t=<时间戳>` 防缓存 |
| POST | `/api/stock/adjust` | body `{"name","qty"}`，用户手动修正 |

事件 `type`：`PUT_IN` / `TAKE_OUT` / `PARTIAL_TAKE_OUT` / `EXCHANGE` / `MANUAL_ADJUST`。

### 界面现状
`web_server.py` 内嵌 HTML 已做移动端响应式（viewport + `@media`），
中文字体栈，库存表/事件表/摄像头三张卡片，库存 3 秒整页刷新、画面 1 秒刷新。
样式可安全美化。安卓 App 由 Codex 另做（4 页 Tab：库存/事件/摄像头/设置）。

---

## 9. 后续开发计划

### 优先级
**第一步：板子端到端实测**
- [ ] 部署 v2 模型与全部改动到 `/root/2_deploy/`
- [ ] 真实蔬果跑完整「放入 → 取出 → 部分取出」流程，看事件准确率
- [ ] 验证 seed 双路逻辑：手动改数 → 重启板子 → 确认数据未被覆盖

**第二步：依实测调参**
- [ ] 重点 change_locator 的 `diff_thresh/min_area`、event_detector 的 `IOU_MATCH_THRESH`
- [ ] 固定演示场景，选 2~3 类最稳定的食材

**第三步：收尾**
- [ ] 清理 main.py 的 `[DEBUG]` 打印
- [ ] 录演示视频（工位 + 白盒子假冰箱）
- [ ] pytest 复核 → 合并 master

### 答辩重点准备
- **为什么用「门控 + 差分变化定位」而不是每帧全画面检测？**
  → 低算力嵌入式 NPU 扛不住每帧推理；运动门控极轻量，只在稳定时刻对变化区域定向识别，省算力又准。
- **三态状态机的作用？**
  → SETTLING「沉降」阶段专门过滤人手中途停顿造成的假稳定，降低误触发。
- **物体识别不出还能记账？**
  → 位置记忆 placed_items 兜底：东西被拿走后那块变空背景识别失败，靠记忆判定取出。AI 识别 + 传统视觉规则结合。
- **为什么选 RK3568？** → 0.8 TOPS NPU 原生支持 RKNN，功耗低、成本合理。
- **模型为什么用 fp16 不用 INT8？** → rknn-toolkit2 v1.4.0 对 YOLOv8 的 INT8 量化有 bug（类别得分全 0），fp16 精度损失可接受。
- **模型是自己训练的吗？** → 模型为采购；**端侧部署、RKNN 转换、事件检测算法、状态机、库存系统均为自研**——答辩重点讲系统工程与算法，这才是核心工作量。
- **系统局限？** → 多层冰箱需多摄像头；严重遮挡识别率下降；紧贴物体 NMS 会合并；建议门内顶部安装减少遮挡。

---

## 10. 绝对约束（不能破坏）

### 代码层面
- **`utils.CLASSES` 的内容与顺序**必须与 `fridge_yolo_v2.rknn` 训练标签完全一致 —— 改动 = 所有识别错位、历史库存混乱。
- **`IMG_SIZE = 640`** 与 rknn 模型绑定 —— 改了必须重新转换模型。
- **`utils.postprocess()` 的张量解析逻辑**与输出 `(1,23,8400,1)` 布局绑定 —— 注释可改，解析代码不可动。
- **`event_detector.update()` 返回二元组 `(events, state)`** —— 早期三元组写法已废弃，main.py 按二元组解包。
- **事件系统的增量语义**（PUT_IN 累加 / TAKE_OUT 递减）—— 不可改成绝对值覆盖，否则用户手动修正会被冲掉。
- **seed 双路逻辑** —— `inv.has_stock()` 为真时只恢复位置记忆、不写 DB —— 不可退回到无条件 PUT_IN 播种。
- **板端必须用 `RKNNLite`，不能用 `RKNN`** —— 否则导入失败。
- **`init_runtime()` 不能带 `core_mask` 参数** —— RK3568 不支持（那是 RK3588 专用）。
- **必须在 `/root/2_deploy` 目录运行 `python3 main.py`** —— `inventory.db` 是相对路径。
- 改任何核心模块（utils/event_detector/change_locator/inventory）后**必须重跑 `pytest`**，基线 25 passed。

### 模型转换层面
- `rknn-toolkit2` 版本固定 **v1.4.0**，不要升级。
- ONNX 必须 **opset=11**，或修复 MaxPool dilations 属性。
- 转换配置 **`do_quantization=False`**（用 fp16），否则 INT8 量化致类别得分全 0。
- `target_platform` 必须是 `'rk3568'`。
- 不能加 `reorder_channel` 参数（v1.4.0 不支持）。

### 数据层面
- 换模型必须**同时更新 `CLASSES` 并清空 `inventory.db`**，否则 class_name 与 class_id 错位。

---

## 11. 耦合关系图

```
utils.CLASSES ──强耦合，必须同步──▶ fridge_yolo_v2.rknn 训练标签
      │                                      │
      └── class_id 索引 ──▶ inventory.db 的 class_name ──▶ web_server / App 展示

utils.py (preprocess/postprocess/identify_crop)
   ▲ 被 import
   ├── main.py
   ├── infer_image.py
   └── change_locator.py（经 classify_fn 注入）

motion.is_moving ┐
                 ├──注入──▶ EventDetector(motion_fn, locate_fn)
change_locator ──┘                  ▲ 被 main.py 调用，返回 (events, state)

inventory.py
   ▲ 被 main.py（写）与 web_server.py（读）同时访问同一个 inventory.db
   │ 经 SQLite 文件解耦，潜在锁冲突（check_same_thread=False 缓解）

main.py ──写──▶ /tmp/fridge_latest.jpg ──读──▶ web_server.py /camera
```

板子上 `main.py` 与 `web_server.py` 为**两个独立进程**，
经 `inventory.db` 与 `/tmp/fridge_latest.jpg` 文件解耦，不直接通信。

---

## 12. 给接手 AI 的特别说明

### 这个项目当前最重要的事
**算法已重构跑通，v2 模型已就位，最后一公里是板子端到端实测 + 调参 + 录演示视频。**
不要再去碰算法架构，重点是用真实数据验证和调参。

### 最容易踩的坑（按严重度排序）
1. **CLASSES 写错或顺序不对** → 食材识别全错。
2. **以为 `update()` 返回三元组** → 早期文档/代码已废弃，现在是二元组 `(events, state)`。
3. **不在 `/root/2_deploy` 目录运行** → 数据库路径错误。
4. **用 `RKNN` 而非 `RKNNLite`** → 板端导入失败。
5. **给 `init_runtime()` 加 `core_mask`** → RK3568 报错。
6. **退回无条件 seed 播种** → 用户手动修正数据被覆盖。

### 开发建议
- 改任何核心模块后立即 `python -m pytest tests/ -q` 验证（基线 25 passed）。
- 换模型前先 `print(m.names)` 确认 CLASSES，再 `rm inventory.db` 清库。
- 调试优先用 `python main.py --replay <目录>` 回放固定帧序列，比开摄像头可复现。
- 用 `--show` 在 HDMI 看实时状态/变化框叠加，方便定位问题。
- 改完文件给用户的 scp 命令格式见第 2 章。

### 演示场景建议（答辩用）
选识别最稳定的 2~3 类食材，固定三个场景：
- 场景 1 放入：空冰箱放入 1 根香蕉 → PUT_IN → 库存更新
- 场景 2 取出：取走香蕉 → TAKE_OUT → 库存清零
- 场景 3 部分取出：放入 3 个苹果 → 取走 1 个 → PARTIAL_TAKE_OUT → 库存 3→2
- 加分：App 上手动修正某数量，演示「AI 识别 + 人工修正」容错设计

这几个场景跑通，答辩基础分稳了。
```
