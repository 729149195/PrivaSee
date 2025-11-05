# GPU 显存优化指南

## 问题描述
DeepSeek-OCR 模型在处理多个文档时可能会遇到 CUDA OOM (Out of Memory) 错误。

## 已实施的优化措施

### 1. ✅ 半精度浮点数 (bfloat16)
模型已配置为使用 `bfloat16` 而非 `float32`，减少约 50% 的显存占用。

### 2. ✅ 自动清理 GPU 缓存
每次推理完成后自动调用 `torch.cuda.empty_cache()`，释放未使用的显存。

### 3. ✅ PyTorch 显存分配优化
启动脚本已设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，避免显存碎片化。

### 4. ✅ Flash Attention 2
优先使用 Flash Attention 2（如果可用），减少注意力机制的显存占用。

## 如果仍然遇到 OOM 错误

### 方法 1：重启服务（推荐）
```bash
# 停止当前服务 (Ctrl+C)
# 重新启动服务
bash start_deepseek_ocr_service.sh
```

### 方法 2：减少并发处理
一次只处理一个文档，等待完成后再处理下一个。

### 方法 3：使用更小的分辨率模式
在前端选择更低的分辨率模式（如 "standard" 而非 "gundam"）。

### 方法 4：监控显存使用
```bash
# 实时监控 GPU 显存
watch -n 1 nvidia-smi
```

### 方法 5：手动清理显存（临时方案）
```python
# 在 Python 中执行
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
```

## 显存使用说明

- **模型加载**: ~4-6 GB
- **单次推理**: ~2-4 GB (取决于图片大小和分辨率模式)
- **推荐 GPU 显存**: ≥ 12 GB

## 优化建议

1. **关闭其他占用 GPU 的程序**
2. **避免在服务运行时训练其他模型**
3. **定期重启服务以清理累积的显存**
4. **使用较低的分辨率模式处理大型文档**

## 技术细节

当前优化配置：
- 模型精度: `bfloat16`
- 注意力机制: Flash Attention 2（如可用）
- 显存分配: 可扩展段（expandable_segments）
- 自动清理: 每次推理后执行

## 故障排除

如果持续遇到 OOM 错误：

1. 检查 GPU 是否被其他进程占用：
   ```bash
   nvidia-smi
   ```

2. 重启服务以清理显存：
   ```bash
   bash start_deepseek_ocr_service.sh
   ```

3. 考虑升级到更大显存的 GPU（推荐 16GB 或 24GB）

## 性能监控

建议使用以下命令监控服务性能：
```bash
# GPU 监控
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 1

# 服务日志
tail -f deepseek_ocr_service.log
```

