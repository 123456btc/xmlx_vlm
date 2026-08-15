"""微内核插件系统与生命周期管理 (Microkernel & Plugin Architecture).

借鉴 Cordis 与 DeepSeek Harness 的“一切皆插件”和“时空可组合性 (Spatiotemporal Composability)”思想：
- 策略、告警处理器、外部集成均封装为标准插件。
- 具备严格的生命周期闭环：加载时按需注册服务与订阅，卸载时确定性回收所有副作用（事件监听、后台协程任务、内存状态），杜绝残留与状态污染。
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

from xmlx_vlm.ai_trader.oms.events.bus import EventBus, SyncEventBus
from xmlx_vlm.ai_trader.sdk.client import TraderSDK

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """插件元数据."""

    name: str
    version: str = "0.1.0"
    author: str = "AI Trader Community"
    description: str = ""
    tags: List[str] = field(default_factory=list)


class PluginContext:
    """插件运行时上下文，提供受控的资源访问与自动副作用追踪."""

    def __init__(
        self,
        plugin_name: str,
        event_bus: EventBus,
        sdk: TraderSDK,
    ):
        self.plugin_name = plugin_name
        self.event_bus = event_bus
        self.sdk = sdk
        self.logger = logging.getLogger(f"plugin.{plugin_name}")

        # 追踪该插件注册的副作用，以便卸载时自动对称回收
        self._registered_handlers: List[Tuple[Any, Callable]] = []
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._state_store: Dict[str, Any] = {}

    def subscribe(self, event_type: Any, handler: Callable[[Any], None]) -> None:
        """订阅事件总线事件并记录副作用."""
        self.event_bus.subscribe(event_type, handler)
        self._registered_handlers.append((event_type, handler))
        self.logger.debug("Subscribed to event %s", event_type)

    def spawn_task(self, name: str, coro: Coroutine) -> asyncio.Task:
        """创建受控后台任务，卸载插件时会自动取消并回收."""
        if name in self._active_tasks and not self._active_tasks[name].done():
            self.logger.warning("Task %s already running in plugin %s, cancelling old one", name, self.plugin_name)
            self._active_tasks[name].cancel()

        task = asyncio.create_task(coro)
        self._active_tasks[name] = task
        self.logger.debug("Spawned background task %s", name)
        return task

    def set_state(self, key: str, value: Any) -> None:
        """存储插件状态."""
        self._state_store[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """获取插件状态."""
        return self._state_store.get(key, default)

    def cleanup_all(self) -> None:
        """卸载时确定性撤销所有副作用."""
        # 1. 取消所有后台协程
        for name, task in list(self._active_tasks.items()):
            if not task.done():
                task.cancel()
                self.logger.debug("Cancelled task %s", name)
        self._active_tasks.clear()

        # 2. 注销事件监听
        for event_type, handler in self._registered_handlers:
            try:
                self.event_bus.unsubscribe(event_type, handler)
                self.logger.debug("Unsubscribed handler from %s", event_type)
            except Exception as e:
                self.logger.warning("Error unsubscribing handler: %s", e)
        self._registered_handlers.clear()

        # 3. 清理内存状态
        self._state_store.clear()
        self.logger.info("Plugin %s resources fully cleaned up", self.plugin_name)


class BaseTraderPlugin(ABC):
    """交易插件抽象基类."""

    metadata: PluginMetadata

    def __init__(self):
        self.ctx: Optional[PluginContext] = None

    @abstractmethod
    def on_load(self, ctx: PluginContext) -> None:
        """插件加载生命周期钩子."""
        pass

    def on_unload(self) -> None:
        """插件卸载前自定义清理逻辑（默认已自动由 Context 撤销副作用）."""
        pass


class PluginManager:
    """微内核插件管理器."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        sdk: Optional[TraderSDK] = None,
    ):
        self.event_bus = event_bus or SyncEventBus()
        self.sdk = sdk or TraderSDK()
        self._plugins: Dict[str, BaseTraderPlugin] = {}
        self._contexts: Dict[str, PluginContext] = {}

    def load_plugin(self, plugin: BaseTraderPlugin) -> bool:
        """动态挂载并启动一个插件."""
        name = plugin.metadata.name
        if name in self._plugins:
            logger.warning("Plugin %s is already loaded, unloading first", name)
            self.unload_plugin(name)

        try:
            ctx = PluginContext(plugin_name=name, event_bus=self.event_bus, sdk=self.sdk)
            plugin.ctx = ctx
            plugin.on_load(ctx)

            self._plugins[name] = plugin
            self._contexts[name] = ctx
            logger.info("Successfully loaded plugin: %s (v%s)", name, plugin.metadata.version)
            return True
        except Exception as exc:
            logger.exception("Failed to load plugin %s: %s", name, exc)
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """动态卸载插件并确定性回收所有副作用."""
        if plugin_name not in self._plugins:
            logger.warning("Plugin %s not found for unload", plugin_name)
            return False

        plugin = self._plugins.pop(plugin_name)
        ctx = self._contexts.pop(plugin_name, None)

        try:
            plugin.on_unload()
        except Exception as exc:
            logger.warning("Error in on_unload hook of %s: %s", plugin_name, exc)

        if ctx:
            ctx.cleanup_all()

        logger.info("Successfully unloaded plugin: %s", plugin_name)
        return True

    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出当前加载的所有插件信息."""
        return [
            {
                "name": p.metadata.name,
                "version": p.metadata.version,
                "description": p.metadata.description,
                "tags": p.metadata.tags,
                "tasks_count": len(self._contexts[name]._active_tasks) if name in self._contexts else 0,
            }
            for name, p in self._plugins.items()
        ]

    def unload_all(self) -> None:
        """卸载所有插件并释放全部资源."""
        for name in list(self._plugins.keys()):
            self.unload_plugin(name)


# ── 示例内置插件 ──

class TrendFollowPlugin(BaseTraderPlugin):
    """示例插件：动量突破与趋势跟踪策略."""

    metadata = PluginMetadata(
        name="trend_follow_strategy",
        version="1.0.0",
        description="监听价格突破与均线金叉，生成趋势跟踪交易提案",
        tags=["strategy", "trend", "momentum"],
    )

    def on_load(self, ctx: PluginContext) -> None:
        ctx.logger.info("TrendFollowPlugin loaded and monitoring breakouts.")

        # 示例：通过 ctx 注册事件或启动监控任务
        async def background_scanner():
            try:
                while True:
                    await asyncio.sleep(60)
                    # 模拟轻量扫描
                    ctx.logger.debug("TrendFollowPlugin scan loop tick")
            except asyncio.CancelledError:
                ctx.logger.debug("TrendFollowPlugin scanner task successfully cancelled")
                raise

        ctx.spawn_task("scanner", background_scanner())

    def on_unload(self) -> None:
        if self.ctx:
            self.ctx.logger.info("TrendFollowPlugin gracefully unloading...")
