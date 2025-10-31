#!/bin/bash
#
# 修复 DeepSeek-OCR 与 transformers 4.56.2 的兼容性问题
# 替换 seen_tokens 为 get_seq_length()
#

MODEL_FILE="$HOME/.cache/huggingface/modules/transformers_modules/deepseek_hyphen_ocr/modeling_deepseekocr.py"

if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ 错误：找不到模型文件"
    echo "路径: $MODEL_FILE"
    exit 1
fi

# 检查是否需要修复
if grep -q "past_length = past_key_values.seen_tokens" "$MODEL_FILE"; then
    echo "🔧 正在修复 DeepSeek-OCR 兼容性问题..."
    
    # 备份原文件
    cp "$MODEL_FILE" "${MODEL_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
    
    # 执行替换
    sed -i 's/past_length = past_key_values\.seen_tokens/past_length = past_key_values.get_seq_length()  # Fixed: seen_tokens -> get_seq_length()/g' "$MODEL_FILE"
    
    # 验证修复
    if grep -q "past_length = past_key_values.get_seq_length()" "$MODEL_FILE"; then
        echo "✅ 修复成功！"
        echo "已将 seen_tokens 替换为 get_seq_length()"
        echo "备份文件已保存"
    else
        echo "❌ 修复失败，请手动检查"
        exit 1
    fi
else
    echo "✅ 已经修复，无需再次操作"
fi

