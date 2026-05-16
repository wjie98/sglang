# DVR v3 与原始参考分支对比

本文记录当前 v3 分支与原始 `origin/dvr_qwen3.5` 的设计差异，用作 v4
继续优化的起点。分析只用于后续合并决策，不代表最终上游提交说明。

## 参照点

- 原始 DVR 分支：`origin/dvr_qwen3.5`
- 原始 DVR 旧起点：`7ace64d1d88de1dfd3cfb6622f3fcbba4fe7eef8`
- 当前 v3 起点：最新 `upstream/sglang-miles`
- v3 清理后提交：`87689acd4 chore(dvr): reduce upstream merge noise`

原始 DVR 相对旧起点修改约 38 个文件，`2726 insertions / 193 deletions`。
v3 相对当前上游修改约 21 个文件，文件范围更小，但核心文件内部逻辑更重。

## 流程差异

原始 DVR 更贴近 EAGLE 的外层 verify 流程：

1. 用 target model 的普通 decode 做 self draft。
2. target verify 外层仍然只传 `num_draft_tokens` 个 token。
3. 对 GDN，backend 内部维护 `CHUNK_SIZE + num_draft_tokens` 的 q/k/v/g/beta
   rolling buffer。
4. GDN target verify 内部用 chunkwise scan 计算 `64 + draft` 窗口，但只把
   draft suffix 的 logits/hidden states 交给 EAGLE verify 后处理。
5. verify 后根据 accepted length 更新 live state、conv state、边界 state，并滚动
   q/k/v/g/beta buffer。

v3 当前流程更直接：

1. 用 target model 的普通 decode 做 self draft。
2. DVR worker 构造物理
   `verified_tokens + draft_tokens + padding_tokens = CHUNK_SIZE + draft` 窗口。
3. 整个模型 target verify 都跑这个固定物理窗口。
4. verify 后把 logits/hidden states 过滤回 draft suffix。
5. GDN backend 从完整窗口 forward 中导出 q/k/v/g/beta、conv window 和边界 state。

v3 的优势是语义直观，容易作为 KL=0 correctness baseline。缺点是所有层都重复
计算 verified tail，不只是 GDN 层。

## 性能影响

以 `CHUNK_SIZE=64`、`num_draft_tokens=16` 为例：

- 原始 DVR：full attention、MLP、RMSNorm 等非 GDN 层通常只处理 16 rows；
  GDN 内部处理 80 rows。
- v3：整个 target verify forward 都处理 80 rows。

因此，v3 在 verify forward 的非 GDN 部分大约有 `80 / 16 = 5x` 的 row 数开销。
端到端不一定慢 5 倍，因为还包含 draft decode、采样、调度和 KV 操作，但这个
物理窗口会明显增加 verify 阶段成本。

## 显存影响

按本地 `Qwen3.5-0.8B`、TP=1、18 个 GDN 层粗略估算，仅计算 DVR/GDN 额外
状态缓存：

- 原始 DVR：q/k/v/g/beta `64+16` buffer + draft conv window，约 `27 MB / req`。
- 当前 v3：额外把 `intermediate_ssm` 和 conv window 扩展到 `64+16`，约
  `1.47 GB / req`。
- 理想优化版：q/k/v/g/beta `80` + draft conv window + compact boundary state，
  约 `44 MB / req`。

这个估算说明 v3 最大的问题不是原理，而是缓存形状过大。尤其
`intermediate_ssm: [layers, reqs, 80, state_shape]` 代价很高，v4 应优先压缩。

## v3 是否可以作为新的 DVR

可以把 v3 作为新的 correctness baseline，但不建议把当前形态作为最终上游版本。

v3 的优点：

- 基于最新 `sglang-miles`，比原始分支更容易继续 rebase。
- 对 Qwen3/Qwen3.5 的 KL=0 测试经验更完整。
- `custom_mask=None` 的 causal verify 路径比复用 tree mask 更符合 DVR chain verify
  语义。
- 文件范围比原始分支小，减少了很多旧版本时代的无关改动。

v3 的缺点：

- 整个模型跑 `64+draft` 物理 verify，性能不如原始分支。
- GDN state cache 显存占用过大。
- `server_args.py` 里还有一些策略性覆盖，合并上游时 review 成本较高。
- `gdn_backend.py` 仍然包含较多 DVR/spec 语义，需要继续收敛为数据接口。

## v4 优化方向

v4 应当保留 v3 的正确性经验，但把性能路径向原始 DVR 靠拢：

1. target verify 外层恢复 EAGLE 风格，只传 `num_draft_tokens` rows。
2. DVR chain verify 继续设置 `custom_mask=None`，保证 causal attention。
3. GDN backend 内部维护 `CHUNK_SIZE + draft` 的 q/k/v/g/beta rolling window。
4. GDN target verify 内部用 chunkwise scan 计算完整窗口，但只返回 draft suffix。
5. 删除 worker 里的 fixed physical window 构造、padding loc 和 80-row graph token
   特判。
6. 把 `intermediate_ssm` 从 80-token cache 压缩为 compact boundary state。
7. conv window 只保留 draft token 级别，必要时从 exported q/k/v/g/beta 和边界 conv
   state 重建。
8. FLA deterministic 修复单独保留为前置 commit，避免和 DVR control flow 混在一起。
9. 手动 KL 测试脚本、开发报告、实验记录不进入最终上游 diff，保留在 archive 或
   manual 路径。

目标形态是“原始 DVR 的现代化重写”：流程贴近参考分支，性能接近参考分支，
但代码基于最新上游，并保持 v3 已验证过的 KL=0 测试方法。
