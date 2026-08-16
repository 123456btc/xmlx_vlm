# Governance Map & Module Trigger Index — 治理策略索引与模块映射表

本文档为 `xmlx_vlm` 项目的唯一真实源（SSOT），将代码目录、功能模块与具体的宪法治理策略进行强制绑定。

**在修改任何目录中的代码时，工程师与 AI Agent 必须通读并遵守该模块绑定的治理规范。**

---

## 🗺️ 核心模块与治理策略映射表

| 业务领域 / 目标代码路径 | 核心模块 / 文件 | 强制遵循的治理规范与核心文件 | 核心不变量与关注焦点 |
| :--- | :--- | :--- | :--- |
| **Agent 核心与护栏** | `xmlx_vlm/agent_core/`<br>`xmlx_vlm/agent.py` | [`ENGINEERING_CONSTITUTION.md`](./ENGINEERING_CONSTITUTION.md)<br>[`GOVERNANCE_PRIMACY.md`](./GOVERNANCE_PRIMACY.md) | 工具变异与幂等隔离、死循环硬停、反过度交易与频次节流、严禁伪造测试。 |
| **多 Agent 策略与共识** | `xmlx_vlm/ai_trader/agent/`<br>`xmlx_vlm/ai_trader/decision/` | [`ENGINEERING_CONSTITUTION.md`](./ENGINEERING_CONSTITUTION.md) §1, §2<br>[`AI_GOVERNANCE_OPERATING_MODEL.md`](./AI_GOVERNANCE_OPERATING_MODEL.md) | 分析师与风控官权责分离、盈亏比(RR)与ATR动态止损硬核验、历史反思注入。 |
| **订单执行与 OMS 风控** | `xmlx_vlm/ai_trader/oms/` | [`ENGINEERING_CONSTITUTION.md`](./ENGINEERING_CONSTITUTION.md) §1, §2, §3 | 订单唯一 SSOT 网关、日内亏损硬熔断、Correlation ID 追踪、仓位与滑点保护。 |
| **时钟与确定性环境** | `xmlx_vlm/ai_trader/oms/utils/clock.py`<br>`xmlx_vlm/ai_trader/research/` | [`ENGINEERING_CONSTITUTION.md`](./ENGINEERING_CONSTITUTION.md) §2, §3 | 严禁 Wall-clock 时钟，统一注入 `ClockProvider`，100% 确定性回测与重播。 |
| **行情流与技术指标** | `xmlx_vlm/ai_trader/market_service/` | [`ENGINEERING_CONSTITUTION.md`](./ENGINEERING_CONSTITUTION.md) §3 | 毫秒级内存状态机、无界内存队列防溢出、离线降级与断线重连抖动退避。 |
| **密钥与资产安全 (KMS)** | `xmlx_vlm/ai_trader/web/`<br>`xmlx_vlm/auth.py` | [`ENGINEERING_CONSTITUTION.md`](./ENGINEERING_CONSTITUTION.md) §5 | 默认 Paper 模式、实盘双重确认、私钥零明文日志、最小权限原则。 |
| **技能扩展与安全扫描** | `xmlx_vlm/skills/` | [`ENGINEERING_CONSTITUTION.md`](./ENGINEERING_CONSTITUTION.md) §4 | 技能威胁扫描（防信息外发、防破坏性指令、防 Prompt 注入越狱）。 |
| **测试与质量门禁** | `tests/`<br>`.governance/scripts/` | [`ENGINEERING_CONSTITUTION.md`](./ENGINEERING_CONSTITUTION.md) §4<br>[`PROJECT_BUGS_POSTMORTEM.md`](./PROJECT_BUGS_POSTMORTEM.md) | TDD 红绿闭环、CI 门禁扫描、真实终端输出验证。 |

---

## ⚡ 快速查询治理规范命令

在修改任何代码路径前，可运行以下命令快速查询需要阅读并遵守的治理条款：

```bash
python3 .governance/scripts/resolve_governance.py --scope xmlx_vlm/ai_trader/oms
```
