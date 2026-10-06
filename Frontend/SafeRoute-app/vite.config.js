import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Configuración del compilador y servidor de desarrollo del frontend.
// Habilita la transformación de JSX y la actualización de componentes durante el desarrollo.
export default defineConfig({
  plugins: [react()],
})
