# 冰箱食材管理 Android App — Codex 交接文档

> 给 ChatGPT/Codex 的两段提示词：
> - **第一段**：项目背景（建立上下文，让它理解你在做什么）
> - **第二段**：初始任务（让它开始写代码）
>
> 建议把"第一段"作为第一条消息，等它确认理解后再发"第二段"。

---

## 第一段：项目背景

```text
我正在做一个嵌入式智能冰箱项目（小米研电赛赛题五），后端已经在 RK3568 开发板上跑通了，现在要做一个 Android App 作为前端管理界面。请帮我用 Kotlin + Jetpack Compose 实现。

## 项目背景

一个内嵌摄像头的智能冰箱，能自动识别食材的放入和取出，维护实时库存。后端逻辑已全部在板子上完成，App 只需要做展示层。

## 系统架构

[Android App (本次要做)]  <---HTTP/REST 局域网--->  [Flask 后端 (已实现)]
                                                      |
                                            [YOLOv8 RKNN 推理 + 事件状态机]
                                                      |
                                                [USB 摄像头 → 冰箱内部]

## 硬件 / 软件

- 后端硬件：RK3568 开发板（iTOP-RK3568），带 NPU
- 后端语言：Python 3 + Flask
- 模型：YOLOv8n RKNN 推理（19 类水果蔬菜，mAP@0.5=0.94）
- 数据库：SQLite（库存与事件历史）
- 通信：HTTP，局域网，板子默认 IP `192.168.3.100:5000`（App 内可配置）

## 后端 REST API 契约（已实现，可直接调用）

### 1. GET /api/stock — 当前库存

响应（JSON 数组）：
```
[
  {
    "name": "banana",
    "qty": 2,
    "first_in": "2026-05-20 23:21:09",
    "last_update": "2026-05-20 23:30:11"
  },
  {
    "name": "apple",
    "qty": 1,
    "first_in": "2026-05-21 09:15:30",
    "last_update": "2026-05-21 09:15:30"
  }
]
```

字段：name（食材英文名）、qty（数量）、first_in（首次入库时间）、last_update（最近更新时间，字符串格式 "YYYY-MM-DD HH:MM:SS"）

### 2. GET /api/events — 最近 20 条事件

响应（JSON 数组，按时间倒序）：
```
[
  {
    "time": "2026-05-20 23:30:11",
    "type": "PUT_IN",
    "note": "放入1个banana"
  },
  {
    "time": "2026-05-20 23:25:03",
    "type": "TAKE_OUT",
    "note": "取出1个apple"
  }
]
```

type 取值：
- `PUT_IN` —— 放入
- `TAKE_OUT` —— 取出
- `PARTIAL_TAKE_OUT` —— 部分取出（如取走一把葡萄中的几颗）

### 3. GET /camera — 摄像头当前帧

响应：image/jpeg 二进制图片，由板子每秒生成一张到 /tmp/fridge_latest.jpg。

注意：请求时建议加时间戳参数防止 HTTP 缓存，例如 `/camera?t=1716345678901`。

## 食材中英文映射（用于显示中文名）

水果类（5 种）：
- apple = 苹果
- banana = 香蕉
- pear = 梨
- orange = 橙子
- pineapple = 菠萝

蔬菜类（14 种）：
- Onion = 洋葱
- garlic = 大蒜
- Garlic = 蒜（与 garlic 是同一物，模型数据集大小写重复）
- Capsicum = 甜椒
- Beet = 甜菜
- Tomato = 番茄
- Cucumber = 黄瓜
- carrot = 胡萝卜
- Eggplant = 茄子
- Cabbage = 卷心菜
- Potato = 土豆
- Zucchini = 西葫芦
- Cauliflower = 花椰菜
- Calabash = 葫芦瓜

App 显示时优先用中文名，找不到映射则原样显示英文。

## 范围约束

- 这是一个学校研电赛项目，1 个月内要做演示视频和答辩
- 局域网内单冰箱使用，不需要云端、账户、推送
- 重点是演示效果好，代码整洁、UI 美观即可，不需要生产级架构
- 我用 Codex/ChatGPT 辅助开发，之后会逐步迭代

你先确认理解了项目背景，告诉我 1-2 个你认为需要澄清的关键点，然后我再发后续的任务。
```

---

## 第二段：初始任务提示词

```text
好，下面是 App 的功能与技术规格。请按"交付步骤"逐步实现，每一步交付完先停下来等我反馈再继续。

## 功能需求（4 个核心页面，底部 Tab 切换）

### 1. 库存页（Stock）
- 卡片列表展示所有食材
- 每张卡片：中文名（大字）+ 数量徽章 + 首次入库时间 + 最近更新时间
- 下拉刷新触发 GET /api/stock
- 库存为空时显示"冰箱是空的"的空状态插画/文字

### 2. 事件页（Events）
- 时间轴样式列表，最新事件在顶部
- 每条事件：时间 + 类型徽章（PUT_IN 绿色"放入"，TAKE_OUT 红色"取出"，PARTIAL_TAKE_OUT 橙色"部分取出"）+ note 详情
- 下拉刷新触发 GET /api/events

### 3. 摄像头页（Camera）
- 全屏显示摄像头实时画面
- 每 1 秒自动刷新（通过给图片 URL 加时间戳实现）
- 顶部小字显示最后刷新时间
- 加载失败显示重试按钮

### 4. 设置页（Settings）
- 输入框配置后端 IP 和端口
- "测试连接"按钮 → 调用 /api/stock 看是否返回 200
- 保存到 SharedPreferences / DataStore
- 修改后所有页面自动用新地址刷新
- **默认值**：IP = `192.168.3.5`，端口 = `5000`
  （192.168.3.5 是开发期间 PC Mock 后端的地址；联调时改成板子的 `192.168.3.100`）

## 技术栈

- Kotlin（语言）
- Jetpack Compose + Material 3（UI）
- 单 Activity + Navigation Compose（路由）
- Retrofit + OkHttp + Kotlinx Serialization（网络）
- Coil（图片加载，支持时间戳缓存策略）
- ViewModel + StateFlow（状态管理）
- DataStore Preferences（持久化设置）
- 最低 minSdk 26（Android 8.0），targetSdk 34

## 错误处理

- 全局：网络异常 → Snackbar 提示"连接板子失败"
- 列表：加载失败 → 显示错误页 + 重试按钮
- 摄像头：加载失败 → 占位图 + 重试按钮
- 不要崩溃，所有异常都要 catch 并友好提示

## 交付步骤（每一步停下来等我反馈）

1. **项目骨架**：Gradle 配置（含所有依赖）、AndroidManifest、目录结构、Application 类
2. **数据层**：定义 API 接口（Retrofit）、数据模型（@Serializable data class）、Repository、ViewModel
3. **导航与主题**：Material 3 Theme、底部 Tab 导航、4 个空白 Composable 占位
4. **库存页 + 事件页**：完整实现两个列表页（含加载/错误/空状态）
5. **摄像头页**：定时刷新策略 + Coil 集成
6. **设置页**：IP 配置 + DataStore + 测试连接
7. **联调指南**：怎么在本地测试（包括用 Mock 数据先跑通 UI）

请开始第 1 步。给我完整的项目目录结构 + build.gradle.kts（含 versionCatalog/libs.versions.toml）+ AndroidManifest + 主 Application 类。

我的开发环境是 Android Studio 最新稳定版，Windows 平台。
```

---

## 重要：开发期间用 Mock 后端，不依赖板子

为了不让板子一直跑着（省 CPU、省电、省心），我准备了一个 Mock 后端 `mock_server.py`，
在 Windows 上直接跑就能模拟板子的全部接口（库存、事件、摄像头都有占位数据）。

**开发流程**：

```
[Android 手机]  ──同 WiFi──>  [Windows PC: mock_server.py]
                                 实现完所有 UI、调通逻辑
                                          ↓
                                    切换到真板子
                                          ↓
[Android 手机]  ──同 WiFi──>  [RK3568: main.py + web_server.py]
                                    最后联调 + 录视频
```

**操作步骤**：

1. 在 PC 上：`python mock_server.py`（在项目根目录）
2. cmd 里跑 `ipconfig`，找 WLAN 网卡的 IPv4，比如 `192.168.1.42`
3. 手机连同一个 WiFi
4. App 设置里 IP 填 PC 的 IP，端口 5000
5. 第一次手机访问可能被 Windows 防火墙拦，弹窗选「允许」即可

**测试技巧**：

- 浏览器访问 `http://localhost:5000/dev/add_event/PUT_IN/apple` 手动追加一条事件，
  测 App 下拉刷新很方便
- `http://localhost:5000/dev/reset` 重置事件列表
- 直接改 `mock_server.py` 顶部的 `MOCK_STOCK` / `MOCK_EVENTS` 改返回数据

这样你的板子可以完全关掉，开发节奏由 PC 主导，到联调阶段再开板子。

---

## 使用建议

1. **第一段先发出去**，等 ChatGPT 反馈"我理解了，但有两个问题想确认……"之类，回答它的问题，确认它理解了再发第二段
2. **每一步交付完，先在 Android Studio 跑通编译**，再让它进下一步。出错就把错误信息粘回去让它修
3. **Mock 数据先行**：第 4 步实现列表页时，先不连板子，让 ChatGPT 给一份硬编码的假数据跑通 UI，然后再切换到真实 API。这样 UI 调试和后端联调解耦
4. **联调阶段板子要在线**：测试 App 时板子上要同时跑 `python3 main.py` 和 `python3 web_server.py`，否则 App 拿不到数据
5. **演示视频的 App 部分**：录的时候手机也要连同一个 WiFi，建议手机用屏幕录制（Android 自带或秒影），可以单独录一段 App 的 demo 拼到主演示视频里
