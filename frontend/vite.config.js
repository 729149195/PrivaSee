import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// 允许通过 Tailscale Funnel 的 ts.net 域名访问开发服务器
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    allowedHosts: [
      'zhangxiangxuan-precision-3660.tail7b7f0c.ts.net',
      '*.ts.net'
    ],
    proxy: {
      // DeepSeek OCR API 代理 - 端口 5001
      '/ocr-api': {
        target: 'http://127.0.0.1:5001',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/ocr-api/, '/api')
      },
      // Whisper API 代理 - 端口 5000，必须放在 /api 之前，避免路径冲突
      '/whisper-api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/whisper-api/, '/api')
      },
      '/whisper': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/whisper/, '')
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
