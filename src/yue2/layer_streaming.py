from __future__ import annotations
import functools
import itertools
import logging
from typing import Any
from collections import OrderedDict
import torch
from torch import nn

logger = logging.getLogger(__name__)



def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    """Resolve a dotted attribute path like ``'model.language_model.layers'``."""
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj


class SimpleLayerStore_Fast:
    """简化版层存储，支持按需加载和立即释放"""
    
    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)
        
        # 保留CPU端的原始参数引用
        self._cpu_params: list[dict[str, torch.Tensor]] = []
        for layer in layers:
            cpu_copy = {}
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                cpu_copy[name] = tensor.data.cpu()  # 保留在CPU上
            self._cpu_params.append(cpu_copy)
    
    def load_layer_to_gpu(self, idx: int, layer: nn.Module) -> None:
        """将指定层加载到GPU"""
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if name in self._cpu_params[idx]:
                param.data = self._cpu_params[idx][name].to(self.target_device)
    
    def unload_layer_from_gpu(self, idx: int, layer: nn.Module) -> None:
        """将指定层从GPU卸载回CPU"""
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if name in self._cpu_params[idx]:
                param.data = self._cpu_params[idx][name]  # 恢复为CPU副本


class SimpleLayerFastWrapper(nn.Module):
    """单层流式处理包装器"""
    
    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
    ) -> None:
        super().__init__()
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device

        self._store = SimpleLayerStore_Fast(self._layers, self._target_device)
        
        # 将非层参数移到GPU
        self._move_non_layer_params_to_gpu()
        
        # 注册钩子
        self._register_simple_hooks()
    
    def _move_non_layer_params_to_gpu(self) -> None:
        """移动非层参数到GPU"""
        layer_tensor_ids = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)
    
    def _register_simple_hooks(self) -> None:
        """注册简单的加载/释放钩子"""
        idx_map = {id(layer): idx for idx, layer in enumerate(self._layers)}
        
        def _pre_hook(module: nn.Module, input, *, idx: int):
            # 加载当前层到GPU
            self._store.load_layer_to_gpu(idx, module)
            # 记录流，防止内存被提前回收
            for param in itertools.chain(module.parameters(), module.buffers()):
                param.data.record_stream(torch.cuda.current_stream(self._target_device))
        
        def _post_hook(module: nn.Module, input, output, *, idx: int):
            # 处理完后立即将层移回CPU
            self._store.unload_layer_from_gpu(idx, module)
        
        for layer in self._layers:
            idx = idx_map[id(layer)]
            pre_hook = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            post_hook = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """代理属性访问到原始模型"""
        try:
            # 首先尝试从包装器自身获取属性
            return super().__getattr__(name)
        except AttributeError:
            # 如果失败，则从原始模型获取
            return getattr(self._model, name)

class _SimpleLayerStore:
    """Layer store that preserves original CPU tensors to avoid reallocation."""

    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)
        # 按层存储每个参数的原始 CPU 张量引用
        self._cpu_refs: list[dict[str, torch.Tensor]] = []

    def _ensure_cpu_refs(self, idx: int, layer: nn.Module) -> None:
        """首次加载时保存当前参数的 CPU 引用（如果参数还在 CPU 上）。"""
        if idx < len(self._cpu_refs):
            return  # 已经保存过
        refs = {}
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if param.data.device.type == 'cpu':
                refs[name] = param.data  # 直接保存当前 CPU 张量的引用
            else:
                # 正常情况下不会发生，但安全起见先移到 CPU 再保存引用
                refs[name] = param.data.cpu()
        # 确保 _cpu_refs 长度足够
        while len(self._cpu_refs) <= idx:
            self._cpu_refs.append({})
        self._cpu_refs[idx] = refs

    def load_layer_to_gpu(self, idx: int, layer: nn.Module) -> None:
        # 第一次加载时记录原始 CPU 引用
        self._ensure_cpu_refs(idx, layer)
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            param.data = self._cpu_refs[idx][name].to(self.target_device)

    def unload_layer_from_gpu(self, idx: int, layer: nn.Module) -> None:
        """恢复为最初的 CPU 张量（不分配新内存）。"""
        refs = self._cpu_refs[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if param.data.is_cuda:
                param.data = refs[name]   # 直接指回原始 CPU 张量



class SimpleLayerStreamingWrapper(nn.Module):
    """单层流式处理包装器"""
    
    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
    ) -> None:
        super().__init__()
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device

        self._store = _SimpleLayerStore(self._layers, self._target_device)
        
        # 将非层参数移到GPU
        self._move_non_layer_params_to_gpu()
        
        # 注册钩子
        self._register_simple_hooks()
    
    def _move_non_layer_params_to_gpu(self) -> None:
        """移动非层参数到GPU"""
        layer_tensor_ids = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)
    
    def _register_simple_hooks(self) -> None:
        """注册简单的加载/释放钩子"""
        idx_map = {id(layer): idx for idx, layer in enumerate(self._layers)}
        
        def _pre_hook(module: nn.Module, input, *, idx: int):
            # 加载当前层到GPU
            self._store.load_layer_to_gpu(idx, module)
            # 记录流，防止内存被提前回收
            compute_stream = torch.cuda.current_stream(self._target_device)
            for param in itertools.chain(module.parameters(), module.buffers()):
                if param.data.is_cuda and param.data.data_ptr() != 0:
                    current_tensor_stream = torch.cuda.current_stream(param.data.device)
                    if current_tensor_stream != compute_stream:
                        param.data.record_stream(compute_stream)
        
        def _post_hook(module: nn.Module, input, output, *, idx: int):
            # 处理完后立即将层移回CPU
            self._store.unload_layer_from_gpu(idx, module)
        
        for layer in self._layers:
            idx = idx_map[id(layer)]
            pre_hook = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            post_hook = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """代理属性访问到原始模型"""
        try:
            # 首先尝试从包装器自身获取属性
            return super().__getattr__(name)
        except AttributeError:
            # 如果失败，则从原始模型获取
            return getattr(self._model, name)





# ---------------------------------------------------------------------------
# 对YYQ的代码进行修改，避免强制锁页
# ---------------------------------------------------------------------------


def _layer_bytes(layer: nn.Module) -> int:
    """Total bytes of a layer's params + buffers (uses real ``element_size``)."""
    return sum(
        t.numel() * t.element_size()
        for t in itertools.chain(layer.parameters(), layer.buffers())
    )


def _detect_capacity(
    layers: nn.ModuleList,
    target_device: torch.device,
    reserved_mb: int,
    model: nn.Module = None,  # 新增：传入整个模型以计算非层参数
) -> tuple[int, int, int, int, int]:
    """Detect how many layers can stay GPU-resident, accounting for non-layer params."""
    per_layer = [_layer_bytes(l) for l in layers]
    max_layer_bytes = max(per_layer) if per_layer else 0
    avg_layer_bytes = sum(per_layer) // max(len(per_layer), 1)

    # ---- 新增：计算非层模块的显存占用 ----
    non_layer_bytes = 0
    if model is not None:
        layer_tensor_ids: set[int] = set()
        for layer in layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))
        
        for p in model.parameters():
            if id(p) not in layer_tensor_ids:
                non_layer_bytes += p.numel() * p.element_size()
        for b in model.buffers():
            if id(b) not in layer_tensor_ids:
                non_layer_bytes += b.numel() * b.element_size()

    try:
        free, _total = torch.cuda.mem_get_info(target_device)
    except Exception:
        free = 8 * 1024**3

    # 可用显存 = 空闲显存 - 预留缓冲 - 非层模块占用
    available = max(0, free - reserved_mb * 1024 * 1024 - non_layer_bytes)

    if max_layer_bytes <= 0:
        return 1, 0, 0, free, available

    # 额外预留 2 层的 Buffer 空间给 CUDA 计算上下文和激活值
    safe_available = max(0, available - 2 * max_layer_bytes)
    
    max_resident = max(1, safe_available // max_layer_bytes)
    max_resident = int(min(len(layers), max_resident))
    
    return max_resident, max_layer_bytes, avg_layer_bytes, free, available



# ---------------------------------------------------------------------------
# LRU layer store (彻底放弃锁页，使用普通 CPU 内存 + 异步拷贝)
# ---------------------------------------------------------------------------

class _LRULayerStore:
    """普通 CPU 缓存 + LRU GPU 驻留追踪器 (不再使用 pin_memory 避免共享显存占用)"""

    def __init__(
        self,
        layers: nn.ModuleList,
        target_device: torch.device,
    ) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)
        self._layers = layers

        # 保存普通 CPU 张量的引用，不执行任何锁页操作
        self._cpu_refs: list[dict[str, torch.Tensor]] = [{} for _ in layers]
        self._gpu_lru: "OrderedDict[int, None]" = OrderedDict()
        # 预取缓冲区，用于在预取流中暂存数据，避免直接修改 param.data 导致数据竞争
        self._prefetch_buffers: dict[int, dict[str, torch.Tensor]] = {}

    # -- introspection ------------------------------------------------------

    def is_on_gpu(self, idx: int) -> bool:
        return idx in self._gpu_lru

    def gpu_count(self) -> int:
        return len(self._gpu_lru)

    def gpu_indices(self) -> list[int]:
        return list(self._gpu_lru.keys())

    # -- residency ops ------------------------------------------------------

    def _ensure_cpu_refs(self, idx: int) -> None:
        """首次加载时记录原始 CPU 张量的引用（不克隆、不锁页）"""
        if self._cpu_refs[idx]:
            return
            
        layer = self._layers[idx]
        refs = {}
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if param.data.device.type == 'cpu':
                refs[name] = param.data  # 直接保存当前 CPU 张量的引用
            else:
                # 安全降级：如果不知为何已在 GPU 上，移回 CPU 保存引用
                refs[name] = param.data.cpu()
        self._cpu_refs[idx] = refs

    def touch(self, idx: int) -> None:
        """Mark ``idx`` as MRU (move to LRU tail)."""
        if idx in self._gpu_lru:
            self._gpu_lru.move_to_end(idx, last=True)

    def move_to_gpu(self, idx: int, *, non_blocking: bool = False, to_buffer: bool = False) -> None:
        """Bring ``idx`` to GPU and mark MRU.  No-op if already resident."""
        if idx in self._gpu_lru and not to_buffer:
            self._gpu_lru.move_to_end(idx, last=True)
            return
        
        # 确保已保存 CPU 引用
        self._ensure_cpu_refs(idx)
        
        layer = self._layers[idx]
        refs = self._cpu_refs[idx]
        
        if to_buffer:
            # 异步预取模式：将数据拷贝到缓冲区，不直接修改 param.data
            buffers = {}
            for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                buffers[name] = refs[name].to(self.target_device, non_blocking=non_blocking)
            self._prefetch_buffers[idx] = buffers
        else:
            # 同步加载模式：直接替换 param.data
            for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                param.data = refs[name].to(self.target_device, non_blocking=non_blocking)
            self._gpu_lru[idx] = None

    def swap_from_buffer(self, idx: int) -> None:
        """将预取缓冲区的数据交换到模型参数中，并标记为驻留 GPU"""
        if idx not in self._prefetch_buffers:
            return
            
        layer = self._layers[idx]
        buffers = self._prefetch_buffers.pop(idx)
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if name in buffers:
                param.data = buffers[name]
        self._gpu_lru[idx] = None

    def evict_to_cpu(self, idx: int) -> None:
        if idx not in self._gpu_lru:
            return
        layer = self._layers[idx]
        refs = self._cpu_refs[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            param.data = refs[name]
        del self._gpu_lru[idx]

    def evict_lru_until(
        self, max_resident: int, protected: set[int] | None = None
    ) -> list[int]:
        protected = protected or set()
        evicted: list[int] = []
        candidates = [i for i in self._gpu_lru.keys() if i not in protected]
        excess = len(self._gpu_lru) - max_resident
        for idx in candidates:
            if excess <= 0:
                break
            self.evict_to_cpu(idx)
            evicted.append(idx)
            excess -= 1
        return evicted

    def evict_all(self) -> None:
        for idx in list(self._gpu_lru.keys()):
            self.evict_to_cpu(idx)

    def cleanup(self) -> None:
        for d in self._cpu_refs:
            d.clear()
        self._cpu_refs.clear()
        self._gpu_lru.clear()
        self._prefetch_buffers.clear()


# ---------------------------------------------------------------------------
# Async prefetcher
# ---------------------------------------------------------------------------

class _AsyncPrefetcher:
    def __init__(self, store: _LRULayerStore) -> None:
        self._store = store
        self._stream = torch.cuda.Stream(device=store.target_device)
        self._events: dict[int, torch.cuda.Event] = {}

    def prefetch(self, idx: int) -> None:
        if self._store.is_on_gpu(idx):
            self._store.touch(idx)
            return
        if idx in self._events:
            return
        with torch.cuda.stream(self._stream):
            # 使用 to_buffer=True 将数据预取到缓冲区，避免直接修改 param.data
            self._store.move_to_gpu(idx, non_blocking=True, to_buffer=True)
            event = torch.cuda.Event()
            event.record(self._stream)
            self._events[idx] = event


    def wait(self, idx: int) -> None:
        event = self._events.pop(idx, None)
        if event is not None:
            torch.cuda.current_stream(self._store.target_device).wait_event(event)

    def cleanup(self) -> None:
        self._events.clear()
        self._stream = None



class LayerStreamingWrapper(nn.Module):
    """Adaptive layer-streaming wrapper with VRAM-aware LRU caching."""

    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        prefetch_count: int = 2,
        max_resident: int | None = None,
    ) -> None:
        if prefetch_count < 1:
            raise ValueError("prefetch_count must be >= 1")
        super().__init__()
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._prefetcher: _AsyncPrefetcher | None = None
        self._store: _LRULayerStore | None = None

        reserved_mb = 1024 # 预留1G 的 VRAM
        auto_resident, max_layer_bytes, avg_layer_bytes, free_bytes, avail_bytes = (
            _detect_capacity(self._layers, self._target_device, reserved_mb, model=self._model)
        )

        if max_resident is None:
            self._max_resident = auto_resident
            mode = "AUTO"
        else:
            self._max_resident = max(1, min(len(self._layers), int(max_resident)))
            mode = "MANUAL"

        self._prefetch_count = min(
            prefetch_count,
            max(1, self._max_resident - 1),
            max(1, len(self._layers) - 1),
        )

        n = len(self._layers)
        free_gb = free_bytes / 1024**3
        avail_gb = avail_bytes / 1024**3
        layer_mb = max_layer_bytes / 1024**2
        avg_mb = avg_layer_bytes / 1024**2
        cached_pct = 100 * self._max_resident / max(n, 1)
        banner = (
            f"[layer_streaming] mode={mode} layers={n} "
            f"layer_size~{avg_mb:.0f}MB(max {layer_mb:.0f}MB) "
            f"free_vram={free_gb:.1f}GB reserve={reserved_mb}MB "
            f"avail={avail_gb:.1f}GB "
            f"max_resident={self._max_resident}/{n} ({cached_pct:.0f}%) "
            f"prefetch_count={self._prefetch_count}"
        )
        print(banner)
        logger.info(banner)

        self._setup()

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        self._store = _LRULayerStore(self._layers, self._target_device)

        layer_tensor_ids: set[int] = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        # 【关键修改】：不进行任何锁页操作，只将非层参数移到 GPU
        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)

        initial = min(
            self._prefetch_count + 1, self._max_resident, len(self._layers)
        )
        for idx in range(initial):
            self._store.move_to_gpu(idx)

        self._prefetcher = _AsyncPrefetcher(self._store)
        self._register_hooks()

    def _register_hooks(self) -> None:
        idx_map: dict[int, int] = {
            id(layer): idx for idx, layer in enumerate(self._layers)
        }
        n = len(self._layers)

        def _pre_hook(module: nn.Module, _args: Any, *, idx: int) -> None:
            assert self._prefetcher is not None and self._store is not None

            # 1. 等待预取流完成数据拷贝到缓冲区
            self._prefetcher.wait(idx)
            
            # 2. 如果预取缓冲区有数据，将其交换到模型参数中
            self._store.swap_from_buffer(idx)
            
            # 3. 确保层在 GPU 上（如果没有预取，则同步加载）
            if not self._store.is_on_gpu(idx):
                self._store.move_to_gpu(idx)
            else:
                self._store.touch(idx)

            compute_stream = torch.cuda.current_stream(self._target_device)
            for param in itertools.chain(module.parameters(), module.buffers()):
                param.data.record_stream(compute_stream)

            protected = {(idx + off) % n for off in range(0, self._prefetch_count + 1)}
            self._store.evict_lru_until(self._max_resident, protected=protected)

            for off in range(1, self._prefetch_count + 1):
                next_idx = (idx + off) % n
                if self._store.is_on_gpu(next_idx) or self._store.gpu_count() < self._max_resident:
                    self._prefetcher.prefetch(next_idx)

        def _post_hook(
            module: nn.Module, _args: Any, _output: Any, *, idx: int
        ) -> None:
            assert self._store is not None
            self._store.touch(idx)

        for layer in self._layers:
            idx = idx_map[id(layer)]
            h1 = layer.register_forward_pre_hook(
                functools.partial(_pre_hook, idx=idx)
            )
            h2 = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
            self._hooks.extend([h1, h2])

    def teardown(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        # 强制同步，确保所有 CUDA 操作完成
        torch.cuda.synchronize(device=self._target_device)
        
        # 【关键修复】：彻底清理预取器，防止 Event 残留
        if self._prefetcher is not None:
            self._prefetcher.cleanup()
            # 额外确保流对象被置空，防止持有引用
            self._prefetcher._stream = None
            self._prefetcher = None

        if self._store is not None:
            self._store.evict_all()

        for p in self._model.parameters():
            p.data = p.data.to("cpu")
        for b in self._model.buffers():
            b.data = b.data.to("cpu")

        if self._store is not None:
            self._store.cleanup()
            self._store = None
            
        # 【关键修复】：强制释放 CUDA 缓存分配器中的空闲块
        torch.cuda.empty_cache()


    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)

