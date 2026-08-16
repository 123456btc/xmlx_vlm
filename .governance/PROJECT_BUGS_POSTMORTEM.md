# Project Bugs & Postmortem — 历史重大缺陷复盘与规则免疫库

> **强制阅读要求**：在修改金融资产核算、并发异步或状态机之前，必须阅读本文档中对应的事故复盘。

---

## 历史重大缺陷分类与免疫规则

### §1. 浮点数资金精度事故 (Floating Point Accounting)
* **故障模式**：在订单撮合结算中直接使用 Python 原生 `float` 进行累计加减，导致浮点精度丢失产生尾差（如平仓后持仓残留 `1e-18` BTC）。策略判断持仓非空，导致无法发起新一轮反向开仓。
* **根本原因**：IEEE 754 浮点二进制表示法在十进制小数转换时存在天生精度误差。
* **规则免疫 (Rule Immunization)**：
  1. 财务与持仓路径 100% 强制使用 `Decimal`，并使用字符串初始化（如 `Decimal("0.01")` 而非 `Decimal(0.01)`）。
  2. 平仓归零判断必须引入 Epsilon 容差：`if abs(position_qty) < Decimal("1e-9"): position_qty = Decimal("0")`。

---

### §2. 异步协程跨 await 持锁死锁 (Concurrency & Deadlocks)
* **故障模式**：在 `async` 异步处理流程中，某协程获取了 `threading.Lock` 同步锁后调用了 `await websocket.send(...)`。在网络 I/O 阻塞期间，其他协程在事件循环中尝试获取同一锁，导致整个事件循环挂起。
* **根本原因**：同步锁（`threading.Lock`）会阻塞当前操作系统线程，而 `asyncio` 的多协程运行在单一事件循环线程上，一旦阻塞线程则所有协程全部停滞。
* **规则免疫 (Rule Immunization)**：
  1. 严禁在 `async def` 异步函数中跨 `await` 持有任何同步锁。
  2. 异步保护必须使用 `asyncio.Lock`，且保持锁的临界区最小化。

---

### §3. 时钟穿越与回测未来函数 (Wall-Clock Contamination)
* **故障模式**：在历史 K 线回测中，策略内部调用了 `datetime.now()` 进行冷却时间判断，导致回测系统读到了当前真实系统时间，回测结果完全虚假。
* **根本原因**：将系统当前时间与数据时间轴混用。
* **规则免疫 (Rule Immunization)**：
  1. 严禁策略逻辑直接调用 `datetime.now()` / `time.time()`。
  2. 必须统一由 `ClockProvider` 注入虚拟时钟或 Tick 事件时间戳。

---

### §4. 假测试通过与占位符静默穿透 (Stubs & Phantom Passes)
* **故障模式**：在风控模块的检查函数中，开发阶段为了快速通过测试写了 `return True # TODO: implement margin check`，并在测试用例中断言为 True。该伪代码合入主干后，导致实盘无保证金检查直接穿仓。
* **根本原因**：占位代码未完成即宣称交付，测试未对拒绝场景进行断言。
* **规则免疫 (Rule Immunization)**：
  1. 严禁在生产风控和资金路径中留下任何占位符。
  2. CI 门禁对 `TODO` 占位和单向返回进行硬性静态扫描拦截。
