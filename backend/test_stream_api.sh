#!/bin/bash
# 测试流式API

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

API_URL="http://localhost:5001/api/process/stream"
TEST_IMAGE="/home/zhangxiangxuan/桌面/Projects/PrivaSee/data/1217.jpg"

echo "========================================"
echo "测试 DeepSeek-OCR 流式API"
echo "========================================"
echo ""

if [ ! -f "${TEST_IMAGE}" ]; then
    echo -e "${YELLOW}测试图片不存在: ${TEST_IMAGE}${NC}"
    echo "请指定一个有效的图片路径"
    exit 1
fi

echo -e "${BLUE}测试图片: $(basename ${TEST_IMAGE})${NC}"
echo -e "${BLUE}API地址: ${API_URL}${NC}"
echo ""
echo "开始流式处理..."
echo "----------------------------------------"

curl -N -X POST "${API_URL}" \
  -F "file=@${TEST_IMAGE}" \
  -F "function=free_ocr" \
  -F "resolution=small" \
  2>/dev/null | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
        # 提取JSON数据
        json_data="${line#data: }"
        
        # 解析并显示
        type=$(echo "$json_data" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('type', ''))" 2>/dev/null)
        
        case "$type" in
            start)
                message=$(echo "$json_data" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('message', ''))" 2>/dev/null)
                echo -e "${GREEN}[开始] ${message}${NC}"
                ;;
            progress)
                stage=$(echo "$json_data" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('stage', ''))" 2>/dev/null)
                progress=$(echo "$json_data" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('progress', 0))" 2>/dev/null)
                echo -e "${YELLOW}[${progress}%] ${stage}${NC}"
                ;;
            content)
                text=$(echo "$json_data" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('text', ''), end='')" 2>/dev/null)
                echo -n "$text"
                ;;
            done)
                echo ""
                echo ""
                echo -e "${GREEN}[完成]${NC}"
                metadata=$(echo "$json_data" | python3 -c "import sys, json; data=json.load(sys.stdin); print(json.dumps(data.get('metadata', {}), indent=2))" 2>/dev/null)
                echo "元数据:"
                echo "$metadata"
                ;;
            error)
                error=$(echo "$json_data" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('error', ''))" 2>/dev/null)
                echo -e "${RED}[错误] ${error}${NC}"
                ;;
        esac
    fi
done

echo ""
echo "----------------------------------------"
echo -e "${GREEN}流式测试完成${NC}"

