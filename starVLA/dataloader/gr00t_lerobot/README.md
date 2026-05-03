# gr00t_lerobot — 数据集 / 本体 / 混合 设计说明

> 本文持续更新；目前为「现状梳理 + 重构提案」阶段，**未落地代码**。

---

## 1. 现状（3 个并列的全局注册表）

```
ROBOT_TYPE_CONFIG_MAP          robot_type ──► DataConfig 实例 (modality_config + transform)
ROBOT_TYPE_TO_EMBODIMENT_TAG   robot_type ──► EmbodimentTag (枚举)
DATASET_NAMED_MIXTURES         mixture    ──► [(dataset, weight, robot_type)]
```

定义位置：

| 注册表 | 基线（package 内） | 自动发现源（per-bench） |
|---|---|---|
| `ROBOT_TYPE_CONFIG_MAP`        | `data_config.py`        | `examples/<bench>/train_files/data_registry/data_config.py` |
| `ROBOT_TYPE_TO_EMBODIMENT_TAG` | `embodiment_tags.py`    | 同上（同一 `data_config.py` 里也可以暴露） |
| `DATASET_NAMED_MIXTURES`       | `mixtures.py`           | 同上 |

合并入口：[registry.py](registry.py) `discover_and_merge()`，import 时一次性把 `examples/*/train_files/data_registry/data_config.py` 里的同名顶层 dict `update` 到全局。

`EmbodimentTag` 还有第二层用途——`EMBODIMENT_TAG_MAPPING: tag → projector_index`（Action Expert 模块里用来选择 head 的 index）。**这是 tag 真正不可省的语义**：tag 不是装饰，它是「我用第几个 action-projector」。

---

## 2. 痛点（你提出的 + 我看到的）

1. **三处分散**：加一个新 robot_type 要在 `ROBOT_TYPE_CONFIG_MAP` + `ROBOT_TYPE_TO_EMBODIMENT_TAG` 两个 dict 里同时写，容易忘 / 不一致。
2. **`ROBOT_TYPE_TO_EMBODIMENT_TAG` 在 `embodiment_tags.py` 里，但和 DataConfig 强耦合**——它本质上是 DataConfig 的一个属性，却放在了别处。
3. `OxeDroidDataConfig` 这种类已经是 robot 的"自描述"了，缺的只是一行 `embodiment_tag = EmbodimentTag.OXE_DROID`。
4. `EmbodimentTag` 枚举本身要不要保留？保留——因为 `EMBODIMENT_TAG_MAPPING` 还要用它做 projector 路由 key（要求强类型、有限集合）。

---

## 3. 重构提案（待你确认）

### 提案 A：把 tag 折进 DataConfig，干掉 `ROBOT_TYPE_TO_EMBODIMENT_TAG` dict

```python
# data_config.py
class OxeDroidDataConfig:
    embodiment_tag = EmbodimentTag.OXE_DROID   # ← 新增 classvar
    video_keys = [...]
    state_keys = [...]
    ...
```

`embodiment_tags.py` **保留**：

- `class EmbodimentTag(Enum)` ✅ 保留（被 `EMBODIMENT_TAG_MAPPING` 和 DataConfig 引用）
- `EMBODIMENT_TAG_MAPPING: tag → projector_index` ✅ 保留（Action Expert 用）
- `ROBOT_TYPE_TO_EMBODIMENT_TAG` ❌ **删除**

`registry.py` 改造：

```python
ROBOT_TYPE_CONFIG_MAP: dict = ...                      # 仍然保留（auto-merged）
def get_embodiment_tag(robot_type: str) -> EmbodimentTag:
    return ROBOT_TYPE_CONFIG_MAP[robot_type].embodiment_tag
# 兼容老代码：保留一个动态 dict-like 视图
ROBOT_TYPE_TO_EMBODIMENT_TAG = {rt: cfg.embodiment_tag for rt, cfg in ROBOT_TYPE_CONFIG_MAP.items()}
```

**收益**：
- 加新本体只改一个类，tag 跟 schema 在同一处定义；
- `embodiment_tags.py` 退化为「枚举 + projector 路由表」纯定义文件；
- 调用方零改动（`ROBOT_TYPE_TO_EMBODIMENT_TAG` 仍可用，只是从 ConfigMap 派生）。

**代价**：
- 所有现存 DataConfig 类需要补一行 `embodiment_tag = ...`（一次性、机械操作）；
- `examples/<bench>/data_registry/data_config.py` 里也不再需要写 `ROBOT_TYPE_TO_EMBODIMENT_TAG`，迁移期可同时支持「类上有就用类，没有就 fallback 到 dict」。

### 提案 B（更激进，不推荐）：连 `EmbodimentTag` 枚举一起删
直接用 robot_type 字符串当 projector 路由 key。**风险**：失去强类型 + projector index 表会变成 robot_type → int 的弱耦合大字典，每加一个 robot_type 都要去改 Action Expert 端配置；当前结构已经把"逻辑本体（tag）"和"具体硬件（robot_type）"做了多对一收敛，这个抽象层是有用的（例：`libero_franka` 和 `demo_sim_franka_delta_joints` 都是 `FRANKA`，共享同一 head）。**所以建议保留枚举层**。

---

## 4. ALOHA（双臂）整合计划

### 4.1 转换器
`convert_robochallenge_to_lerobot.py` 现有 `_process_episode_single_arm`（`state[7]/action[8]`）；新增 `_process_episode_aloha` 分支：

- ALOHA `task_info` 里有 `task_tag=["ALOHA"]`，`CAMERAS["ALOHA"]` 已写好（cam_high/cam_left_wrist/cam_right_wrist）；
- 双臂 schema **待你确认**（见 §5 问题 2）。预计：
  - state: `joint_left(7) + grip_left(1) + joint_right(7) + grip_right(1) = 16`
  - action: 同结构 16 维（或 14：去掉 grip？）
- `state_dim/action_dim` 改为按 robot 分支取（不再硬编码 7/8）。

### 4.2 注册
新增 `_AlohaDataConfig`（双臂 video_keys + state/action keys 改名带 `_left/_right`）+ `_ALOHA_TASKS` set + `aloha_robochallenge → EmbodimentTag.???`（建议新建 `EmbodimentTag.ALOHA` 或先借 `NEW_EMBODIMENT`，见 §5 问题 3）。

### 4.3 bg 下载脚本
`bg_download_convert_all.sh` 加 `KEEP_RAW_ON_SKIP=1` 默认值；ALOHA 分支不再 `rm -rf raw/<task>`。后续等 ALOHA 转换器写好后改 `SINGLE_ARM` 集合 / 移除 ALOHA 跳过分支即可重跑。

### 4.4 训练 smoke
单任务（建议 `lint_roller_remove_dirt` 因为已下载 tar）→ 只跑 50–100 step，验证 dataloader 出 batch、loss 不爆。

---

## 5. 等你确认的问题（请在这里回答）

1. **提案选 A 还是 B？** （我倾向 A：保留枚举，干掉 `ROBOT_TYPE_TO_EMBODIMENT_TAG` dict，把 tag 作为 DataConfig 的 classvar。）
2. **ALOHA 双臂 state/action 维度？** 我没看到已解压的 `task_info.json`，需要先解一个 ALOHA tar 来确认 `joint_positions` / `gripper_width` / `ee_positions` 的形状。是否允许我**只解压一个** ALOHA tar 用于探查（不会再删 raw）？
3. **ALOHA 用哪个 EmbodimentTag？** 选项：
   - (a) 新增 `EmbodimentTag.ALOHA` + 在 `EMBODIMENT_TAG_MAPPING` 里给一个 projector index（请你给我一个未占用的 index，目前已用 17/18/19/24/25/26/31）；
   - (b) 先复用 `NEW_EMBODIMENT`（=31）跑通；
   - (c) 其它。
4. **bg 脚本的 raw 清理策略**：默认对**所有**任务都不清理，还是只对 ALOHA 不清理？（磁盘占用考量：单臂 raw 平均 ~10–30G/task。）
5. **smoke 训练用哪个配置**？沿用之前 `examples/RoboChallenge_table30v2/train_files/run_robochallenge_table30v2.sh` 的配置 + 替换 mixture 为 `robochallenge_table30v2_aloha_*`，对吗？
6. **迁移节奏**：是先做 ALOHA 整合（仍写两个 dict），还是先做架构重构（提案 A），再用新规范写 ALOHA？我建议**先重构再写 ALOHA**——只多 1 步、避免新建一个又要改的烂账。

---

## 6. 进度

### 用户已确认（v0 → v1 决策）
- 提案 **A**：保留 `EmbodimentTag` 枚举 + `EMBODIMENT_TAG_MAPPING`，干掉 `ROBOT_TYPE_TO_EMBODIMENT_TAG` 这个独立 dict（推迟到 ALOHA 下载完成后再做）
- bg 清理策略：**仅** ALOHA 保留 raw（其他单臂任务原本就清掉，且都已完成不再下载）
- ALOHA `EmbodimentTag.ALOHA`，`EMBODIMENT_TAG_MAPPING` 中 projector index = **7**
- ALOHA 双臂 schema：state(14) = L.joint(6)+L.grip(1)+R.joint(6)+R.grip(1)；action(16) = L.ee(7)+L.grip(1)+R.ee(7)+R.grip(1)
- smoke 训练：用 Qwen3.5-0.8B + QwenOFT，目标 OOD（后续任务）
- 节奏：**先**下载完 ALOHA + 通过 smoke 训练，**再**做架构重构

### 已完成
- [x] 现状分析、痛点列出、提案草稿（v0）
- [x] §5 问题用户已回答（见上）
- [x] 探查 ALOHA `task_info.json` schema（lint_roller_remove_dirt: 6+1+6+1 / 7+1+7+1）
- [x] 加 `_process_episode_aloha` + 双分支 modality.json + state/action_dim 切换 → converter 支持 ALOHA
- [x] `bg_download_convert_all.sh` 改：仅 ALOHA 保留 raw（基于 task_info.json 的 `"ALOHA"` tag 判定），删掉 NotImplementedError 跳过分支
- [x] 注册 `EmbodimentTag.ALOHA = 'aloha'`（projector index 7）
- [x] 注册 `_RoboChallengeAlohaConfig` + `aloha_robochallenge` → CONFIG_MAP / EMBODIMENT_TAG
- [x] 加 `_ALOHA_TASKS` set + `robochallenge_table30v2_aloha_all` mixture（auto-填充由 update_data_config.py 负责）
- [x] update_data_config.py 加 ALOHA 分支
- [x] **smoke 训练通过**（lint_roller_remove_dirt × 100 step × bs2 × 2 H800 = 1m25s, loss=0.243, ckpt 保存成功）
- [x] 后台 ALOHA 下载+转换流水线启动中（`tmp/logs/bg_aloha_*.log`）

### 后台进行中
- [ ] 9/10 ALOHA 任务下载+转换（lint_roller 已完成；pack_the_items 进行中）
- [ ] 下载完成后：`update_data_config.py` 自动写回 `_ALOHA_TASKS`
- [ ] 多任务 ALOHA 大 mixture 训练（待你确认配置）

### 待做（下载完成后）
- [ ] 架构重构：`embodiment_tags.py` + `data_config.py`（每个 DataConfig 加 `embodiment_tag` classvar；删 `ROBOT_TYPE_TO_EMBODIMENT_TAG`）
- [ ] `registry.py` 改为 `ROBOT_TYPE_TO_EMBODIMENT_TAG` 从 ConfigMap 派生
- [ ] 各 bench 的 `data_registry/data_config.py` 同步迁移（删除独立 dict）
- 重构data_config 的时候记得 全部examples 下面的都需要变化

然后你要验证rebochallenge 不同本体的训练是否完备，啊验证load 不同本体的 数据， aloha 也是很关键的一个点


