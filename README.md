# PrivaSee

这是一个隐私感知的 AI 助手，具备信息元提取和隐私推理能力。

## 新功能：用户登录系统 ✨

PrivaSee 现已支持完整的用户登录和历史数据保存功能！

### 主要特性

- ✅ 用户注册和登录
- ✅ 自动头像生成
- ✅ 历史对话自动保存
- ✅ 刷新页面后自动恢复
- ✅ 无痕模式（未登录时）
- ✅ 多用户数据隔离

### 快速开始

1. **启动前端**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

2. **使用系统**
   - 点击左侧栏底部 Context window 右上角的"登录"按钮
   - 注册新账号或登录已有账号
   - 开始对话，系统会自动保存历史记录

3. **测试无痕模式**
   - 未登录状态下使用系统
   - 刷新页面，对话将不会保留

### 详细文档

- [用户认证系统说明](frontend/USER_AUTH_README.md) - 完整功能文档
- [演示账号](frontend/DEMO_ACCOUNTS.md) - 测试账号创建指南

## 目录结构

- `frontend/`: React 前端界面
  - `src/users/`: 用户管理模块（新增）
    - `userStore.js`: 用户状态管理
    - `UserAuth.jsx`: 用户认证组件
    - `historyStorage.js`: 历史数据存储工具
- `backend/`: Python 后端服务
- `data/`: 数据存储
- `users/`: 用户数据（可用于后端扩展）

## 启动方式

### 前端（Vite + React）
```bash
cd frontend
npm install
npm run dev
```

### 后端（FastAPI，conda 环境 `privasee`）
```bash
conda activate privasee
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 技术栈

- **前端**: React 18 + Zustand 5 + Ant Design 5
- **状态管理**: Zustand with persist middleware
- **数据存储**: LocalStorage (演示版本)
- **样式**: CSS Modules

## 开发说明

当前版本是前端演示实现，数据存储在浏览器 LocalStorage。生产环境建议：

1. 使用后端 API 进行用户认证
2. 将历史数据存储到数据库
3. 实施安全的密码加密
4. 添加 JWT/Session 管理

## License

(To be added)
