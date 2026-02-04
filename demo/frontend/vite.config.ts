import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/plan': 'http://localhost:8000',
      '/execute': 'http://localhost:8000',
      '/run': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    }
  }
})
