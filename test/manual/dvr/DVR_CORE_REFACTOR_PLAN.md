# DVR Core Refactor Plan

目标结构：

- 少量 DVR core 代码负责 target verify、线性状态 restore/commit、accepted suffix repair、输出前缀和延后后处理。
- self draft 和 EAGLE/MTP 都围绕 core 搭建，只在 draft adapter 和 next-draft-input 构造阶段分化。
- self draft 不分配独立 draft KV cache，只复用 scheduler 准备好的 spec KV window；EAGLE/MTP 继续由 upstream draft worker 管理自己的 draft KV/cache。
- spec v1/v2 只作为最外层胶水：v1 同步消费 core 结果，v2 保存 `DVRDeferredActions` 并在 scheduler 后处理阶段统一消费。

执行顺序：

1. 提交当前已验证通过版本，作为大重构安全点。
2. 引入小而明确的 core 数据对象：`DVRDraftResult`、`DVRVerifyResult`、`DVRDeferredActions`、`DVRDeferredOutput`。
3. 合并 verify 后半段：output prefix、compact logprob、accepted suffix repair、linear-state commit、pending checkpoint 都由 core 统一收口。
4. self draft 和 EAGLE/MTP 都产出统一 `DVRDraftResult`，尽量推迟分支条件。
5. 检查 EAGLE/MTP 边界修复是否仍必要；必要时挂在 core 后处理内部，不再散落在 EAGLE worker。
6. 收缩 spec v2 延后处理：外层只保存一个 `DVRDeferredActions` 对象，后续只通过 `None` 判断是否需要 DVR 后处理。
7. 清理通用文件触点：result processor、output streamer、scheduler 只调用 DVR 后处理入口，不直接理解 self/EAGLE 细节。

验证要求：

- 每批关键改动后至少跑 `git diff --check`、`py_compile`、DVR unit tests。
- 触碰 EAGLE/MTP 路径后跑 35B MTP smoke，覆盖 sync/overlap 和 `return_logprob=True/False`。
- 最终跑 0.8B self-DVR v1/v2 KL boundary、35B MTP/EAGLE smoke、80B self-DVR ShareGPT/LongBench 长输出吞吐。

功能边界：

- GDN/KDA 线性状态抽象必须保留，不能为了当前 GDN 实现写死。
- `return_logprob=True` 是确定性 DVR 的常用路径，不能通过关闭或降级它换吞吐。
- 如果 final-logprob full-prefix replay 能被 suffix oracle + deferred output 替代，应删除；如果还必须保留，只能封装在 core 的 `DVRDeferredActions` 内部。
