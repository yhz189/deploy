# 基于 RK3568 的端侧智能冰箱食材识别与库存管理系统

本项目是一个面向全国研究生电子设计竞赛的嵌入式端侧视觉识别系统，基于 RK3568 Linux 开发板、摄像头、YOLOv8n-RKNN、SQLite、Flask 和 Android App，实现冰箱指定取放区域内食材放入、取出、部分取出、位置整理和包装食品建档等库存管理能力。

项目重点不是单帧识别演示，而是将连续摄像头画面转换为稳定的库存事件：通过 OpenCV 运动门控和稳定前后帧差分定位变化区域，只在取放动作结束后对局部变化区域进行 RKNN/NPU 推理，从而降低端侧无效计算量。

## 项目特点

- **RK3568 端侧部署**：在 RK3568 Linux 开发板上运行摄像头采集、事件检测、RKNNLite 推理和库存服务。
- **事件触发式识别**：不是每帧全图推理，而是先用轻量运动检测判断是否发生取放动作，再对稳定前后帧的变化区域识别。
- **YOLOv8n-RKNN 本地推理**：使用 RKNNLite 1.6.0 加载 `models/fridge_yolo_opset11_rknn16.rknn`，完成 20 类食材检测。
- **库存增量更新**：根据视觉事件生成 `PUT_IN`、`TAKE_OUT`、`PARTIAL_TAKE_OUT`、`EXCHANGE` 等库存事件，并写入 SQLite。
- **混合计量策略**：鸡蛋采用目标计数；香蕉、胡萝卜等不规则食材支持基于颜色分割和面积估计的余量分级。
- **包装食品 OCR 协同**：端侧保存疑似包装候选图，由 App 端 OCR 后确认名称、数量和保质期，再写回库存。
- **Web 与 App 联动**：Flask 提供库存、事件、摄像头画面和包装候选接口，Android App 通过局域网访问。
- **复杂条件辅助**：提供低光增强、高光检测和反光风险分析工具，用于复杂光照条件下的离线验证和演示。

## 系统架构

```text
摄像头 /dev/video0
    |
    v
main.py 主循环
    |
    +-- motion.py              运动门控
    +-- event_detector.py      STABLE / BUSY / SETTLING 状态机
    +-- change_locator.py      稳定前后帧差分与变化区域定位
    +-- utils.py               YOLOv8-RKNN 预处理、后处理、NMS
    +-- banana_area.py         香蕉面积估计与余量分级
    +-- carrot_area.py         胡萝卜面积估计与余量分级
    +-- package_ocr.py         包装候选图保存与确认
    |
    v
inventory.py / SQLite inventory.db
    |
    v
web_server.py / Flask REST API
    |
    +-- Web 页面
    +-- Android App
```

## 核心流程

```text
1. 摄像头持续采集 640x480 画面
2. 灰度帧差判断是否有运动
3. 状态机从 STABLE -> BUSY -> SETTLING -> STABLE
4. 动作结束后取稳定前后帧进行差分
5. 通过阈值、形态学和轮廓提取变化区域 bbox
6. 对变化区域的 ref/new 裁剪图分别进行 YOLOv8-RKNN 识别
7. 根据识别结果判断 APPEAR / DISAPPEAR / SAME / REPLACE / NOISE
8. 生成库存事件并写入 SQLite
9. Web/App 通过 REST API 查询库存和事件
```

## 视觉算法

项目实际使用的视觉算法包括：

- **灰度帧差运动检测**：`cv2.absdiff` + 二值阈值 + 变化像素统计。
- **变化区域定位**：稳定前后帧差分、阈值分割、形态学开闭运算、轮廓提取、bbox 外扩与合并。
- **YOLOv8n 目标检测**：RKNNLite 调用 RK3568 NPU 推理，输出解析、置信度过滤和 NMS。
- **HSV 颜色分割**：用于香蕉、胡萝卜面积估计。
- **连通域过滤**：剔除颜色分割中的小噪声区域。
- **低光增强**：Gamma 校正、双边滤波、LAB/CLAHE。
- **高光检测与抑制**：HSV 高亮低饱和区域检测、局部峰值、形态学处理和亮度压制。

当前没有使用 YOLO-seg、GrabCut、Watershed、光流或多目标跟踪算法。

## 模型说明

当前主模型：

```text
models/fridge_yolo_opset11_rknn16.rknn
```

关键参数：

- 输入尺寸：`640 x 640`
- 输入格式：RGB、uint8、NHWC，形状为 `1 x 640 x 640 x 3`
- 输出解析：`8400 x 24`
- 输出含义：前 4 个为 `xywh`，后 20 个为类别分数
- 后处理：不做 sigmoid，不乘 objectness，直接类别分数过滤 + NMS
- 端侧运行：`rknn-toolkit-lite2 1.6.0` + `/usr/lib/librknnrt.so 1.6.0`

支持类别：

```text
apple, Onion, banana, garlic, pear,
orange, Capsicum, Beet, Tomato, Cucumber,
carrot, Eggplant, Cabbage, Potato, Zucchini,
pineapple, Garlic, Cauliflower, Calabash, egg
```

## 目录结构

```text
deploy/
├── main.py                 # 板端主程序：摄像头采集、状态机、推理和库存事件处理
├── motion.py               # 运动门控
├── change_locator.py       # 变化区域定位
├── event_detector.py       # 事件状态机和事件派生
├── utils.py                # YOLOv8 预处理、后处理、类别映射
├── inventory.py            # SQLite 库存和事件管理
├── web_server.py           # Flask Web 页面和 REST API
├── package_ocr.py          # 包装食品 OCR 候选图管理
├── banana_area.py          # 香蕉颜色分割和面积等级估计
├── carrot_area.py          # 胡萝卜颜色分割和面积等级估计
├── image_enhancement.py    # 低光增强和反光检测工具
├── mock_server.py          # PC 端 App 联调用 Mock 后端
├── infer_image.py          # 单图推理调试脚本
├── tests/                  # pytest 测试用例
├── docs/                   # 技术文档、测试记录和复盘材料
├── models/                 # RKNN 模型文件
└── tools/                  # 离线评估和文档辅助工具
```

## 运行环境

### 开发板

- 开发板：iTOP-RK3568
- 系统：Linux
- Python：3.9
- Conda 环境：`rknn16`
- NPU Runtime：RKNNLite 1.6.0
- 摄像头：`/dev/video0`
- 部署目录：`/root/2_deploy`

### Python 依赖

板端主要依赖：

```text
numpy
opencv-python
flask
rknn-toolkit-lite2==1.6.0
```

SQLite 使用 Python 内置 `sqlite3` 模块。

## 板端运行

进入板端部署目录：

```bash
cd /root/2_deploy
conda activate rknn16
```

启动主程序：

```bash
python3 main.py
```

启动带显示窗口的主程序：

```bash
python3 main.py --show
```

如果需要实时显示 YOLO 推理框：

```bash
python3 main.py --show --show-infer
```

启动 Web/API 服务：

```bash
python3 web_server.py
```

访问 Web 页面：

```text
http://192.168.3.100:5000
```

## PC 端调试

PC 端可启动 Mock 后端用于 Android App 联调：

```bash
python mock_server.py
```

单图推理调试：

```bash
python infer_image.py --image images/test_eggs.jpg
```

## REST API

常用接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | Web 库存页面 |
| GET | `/camera` | 最新摄像头画面 |
| GET | `/api/stock` | 当前库存 JSON |
| GET | `/api/events` | 最近事件 JSON |
| POST | `/api/stock/adjust` | 手动修正计数型食材数量 |
| POST | `/api/stock/adjust-level` | 手动修正面积等级型食材余量 |
| GET | `/api/package/candidate` | 获取包装入库候选 |
| POST | `/api/package/confirm` | App OCR 后确认包装入库 |
| GET | `/api/package/takeout_candidate` | 获取包装取出候选 |
| POST | `/api/package/takeout` | 确认包装取出 |

示例：

```bash
curl http://192.168.3.100:5000/api/stock
```

```bash
curl -X POST http://192.168.3.100:5000/api/stock/adjust \
  -H "Content-Type: application/json" \
  -d '{"name":"egg","qty":12}'
```

## 数据库设计

数据库文件：

```text
inventory.db
```

主要表：

- `inventory`：当前库存状态，包括名称、数量、类型、来源、保质期、位置、面积等级等。
- `events`：历史事件记录，包括事件时间、事件类型、食材名称、数量变化和备注。

网页端和 App 端不直接访问 SQLite，而是通过 Flask REST API 与 `web_server.py` 通信；`web_server.py` 再通过 `InventoryManager` 使用 `sqlite3` 读写 `inventory.db`。

## 测试

运行全部单元测试：

```bash
python -m pytest tests/ -q
```

测试覆盖方向包括：

- 运动检测
- 变化区域定位
- 事件状态机
- YOLO 后处理
- SQLite 库存更新
- 包装候选和包装确认接口
- 香蕉/胡萝卜面积分级
- 复杂光照增强工具

## 部署到开发板

示例 scp 命令：

```powershell
scp "C:\Users\hp-pc\Desktop\deploy\main.py" root@192.168.3.100:/root/2_deploy/
scp "C:\Users\hp-pc\Desktop\deploy\web_server.py" root@192.168.3.100:/root/2_deploy/
scp "C:\Users\hp-pc\Desktop\deploy\inventory.py" root@192.168.3.100:/root/2_deploy/
scp "C:\Users\hp-pc\Desktop\deploy\utils.py" root@192.168.3.100:/root/2_deploy/
```

模型部署：

```powershell
scp "C:\Users\hp-pc\Desktop\deploy\models\fridge_yolo_opset11_rknn16.rknn" root@192.168.3.100:/root/2_deploy/models/
```

## 注意事项

- `main.py` 必须在 `/root/2_deploy` 下运行，因为 `inventory.db` 使用相对路径。
- RK3568 板端必须使用 RKNNLite 1.6.0，旧版 RKNN Runtime 可能导致 YOLOv8 bbox 坐标异常。
- `--show-infer` 会实时跑全帧 YOLO，帧率会明显下降；常规演示建议只使用 `--show`。
- 包装食品 OCR 由 App 端完成，板端只保存候选图和提供确认接口。
- `inventory.db`、debug 裁剪图、旧模型和临时输出不建议提交到代码仓库。

## 项目定位

本项目是一个端侧 AI + 嵌入式 Linux 应用系统原型，重点体现：

- 真实硬件平台上的摄像头采集和 NPU 推理部署
- 低算力场景下的事件触发式视觉识别设计
- 从视觉识别结果到库存业务事件的状态机建模
- SQLite、Flask 和 Android App 的端到端数据闭环
- 面向演示和测试的工程调试、日志记录和文档沉淀

它不是医疗级或工业级产品，也不是完整的 BSP/驱动项目；后续可继续向 C/C++ V4L2 采集、RKNN C API、RGA/MPP 硬件加速和进程守护方向演进。
