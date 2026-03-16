import { defineConfig } from 'vite';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';

const __dirname = import.meta.dirname;

export default defineConfig({
  root: 'src',
  build: { outDir: '../dist', emptyOutDir: true },
  base: '/kifu/',
  server: {
    fs: { allow: ['..'] },
  },
  plugins: [{
    name: 'serve-dev',
    configureServer(server) {
      const prefix = '/kifu/dev/';
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith(prefix)) return next();
        const relPath = req.url.slice(prefix.length).split('?')[0];
        const filePath = resolve(__dirname, 'dev', relPath);
        if (existsSync(filePath) && filePath.endsWith('.html')) {
          let html = readFileSync(filePath, 'utf8');
          // Inject Vite HMR client so the page connects to the dev server
          html = html.replace('<head>', '<head><script type="module" src="/kifu/@vite/client"></script>');
          res.setHeader('Content-Type', 'text/html');
          res.end(html);
        } else {
          next();
        }
      });
    },
  }],
});
