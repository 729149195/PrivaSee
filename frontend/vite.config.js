import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// 允许通过 Tailscale Funnel 的 ts.net 域名访问开发服务器
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ['mammoth', 'xlsx', 'jszip'],
    esbuildOptions: {
      target: 'es2020'
    }
  },
  build: {
    target: 'es2020',
    commonjsOptions: {
      include: [/mammoth/, /xlsx/, /node_modules/]
    }
  },
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    allowedHosts: [
      'zhangxiangxuan-precision-3660.tail7b7f0c.ts.net',
      '*.ts.net'
    ],
    proxy: {
      // PrivaSee 统一后端 API 代理 - 端口 5000
      // OCR 服务: /ocr-api -> /api/ocr
      '/ocr-api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/ocr-api/, '/api/ocr')
      },
      // Whisper 服务: /whisper-api -> /api/whisper
      '/whisper-api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/whisper-api/, '/api/whisper')
      },
      // Whisper 直接访问（兼容旧路径）
      '/whisper': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/whisper/, '/api/whisper')
      },
      '/api': {
        target: 'http://127.0.0.1:11434',
        changeOrigin: true,
        secure: false,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            try { proxyReq.removeHeader('origin') } catch (_) {}
            try { proxyReq.removeHeader('referer') } catch (_) {}
            try { proxyReq.setHeader('host', '127.0.0.1:11434') } catch (_) {}
          })
        }
      },
      '/v1': {
        target: 'http://127.0.0.1:11434',
        changeOrigin: true,
        secure: false,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            try { proxyReq.removeHeader('origin') } catch (_) {}
            try { proxyReq.removeHeader('referer') } catch (_) {}
            try { proxyReq.setHeader('host', '127.0.0.1:11434') } catch (_) {}
          })
        }
      }
    }
  }
})
