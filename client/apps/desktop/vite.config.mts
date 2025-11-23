import path from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { tamaguiPlugin } from '@tamagui/vite-plugin';

const resolvePackage = (name: string) => path.resolve(__dirname, `../../packages/${name}/src`);

export default defineConfig({
  plugins: [
    react(),
    tamaguiPlugin({
      config: '../../tamagui.config.ts',
      components: ['tamagui'],
      disableExtraction: process.env.NODE_ENV !== 'production' ? true : false,
      useReactNativeWebLite: false,
      resolveExtensions: ['.web.tsx', '.web.ts', '.tsx', '.ts'],
    }),
  ],
  resolve: {
    alias: {
      'react-native$': 'react-native-web',
      '@timbre/api': resolvePackage('api'),
      '@timbre/config': resolvePackage('config'),
      '@timbre/ui': resolvePackage('ui'),
      '@timbre/state': resolvePackage('state'),
    },
  },
  define: {
    'process.env': {},
  },
  optimizeDeps: {
    include: ['react-native-web'],
  },
});
