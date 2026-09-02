# SYNTHETIC_DEMO 场景

这里的 24 个场景只用于可重复工程演示，不是真实学生记录，也不能作为学习效果、留存或用户规模证据。每个场景和 episode 都显式标记 `"synthetic": true`。

## 隔离原则

- 重放器不调用网络或付费 LLM。
- 默认输出应放在 `/tmp/yher_synthetic_demo_replays/`。
- 重放器拒绝写入 `data/local_store/`、`data/study_logs/` 或仓库 `data/` 下的其他位置。
- 合成用户 ID 统一使用 `SYNTHETIC_DEMO_` 前缀。
- 当前产品开放 27 个节点；计划矩阵保留第 28 个“化学反应速率”，并把它重放为预期关闭，而不是绕过五题族开放门。

## 验证与重放

```bash
python3 -m demo.synthetic_scenarios.validate
python3 -m demo.synthetic_scenarios.replay \
  --output /tmp/yher_synthetic_demo_replays/manual_run
```

第一次命令校验 24 场景、32 episodes、预算与结局矩阵、节点覆盖和当前开放状态。第二次命令使用真实 R5 catalog 和正式 `SessionService` 做完全离线的确定性重放。
