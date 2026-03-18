import { defineConfig } from 'vite';
import { readFileSync, existsSync, statSync } from 'node:fs';
import { resolve, extname } from 'node:path';

const __dirname = import.meta.dirname;

const MIME_TYPES = {
  '.json': 'application/json',
  '.jpg':  'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png':  'image/png',
  '.webp': 'image/webp',
};

function serveFile(filePath, req, res) {
  if (!existsSync(filePath) || statSync(filePath).isDirectory()) return false;
  const mime = MIME_TYPES[extname(filePath).toLowerCase()] ?? 'application/octet-stream';
  res.setHeader('Content-Type', mime);
  if (req.method === 'HEAD') {
    res.end();
  } else {
    res.end(readFileSync(filePath));
  }
  return true;
}

export default defineConfig({
  root: 'src',
  build: { outDir: '../dist', emptyOutDir: true },
  base: '/kifu/',
  server: {
    fs: { allow: ['..'] },
  },
  plugins: [{
    name: 'serve-dev-and-fixtures',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url ?? '';
        const relPath = decodeURIComponent(url.split('?')[0]);

        // Serve dev/ HTML tools at /kifu/dev/
        if (relPath.startsWith('/kifu/dev/')) {
          const filePath = resolve(__dirname, 'dev', relPath.slice('/kifu/dev/'.length));
          if (existsSync(filePath) && filePath.endsWith('.html')) {
            let html = readFileSync(filePath, 'utf8');
            // Inject Vite HMR client so the page connects to the dev server
            html = html.replace('<head>', '<head><script type="module" src="/kifu/@vite/client"></script>');
            res.setHeader('Content-Type', 'text/html');
            res.end(html);
            return;
          }
        }

        // Serve fixtures/ at /kifu/fixtures/ (Vite root is src/, so fixtures/ isn't
        // reachable via the normal root-relative URL mapping)
        if (relPath.startsWith('/kifu/fixtures/')) {
          const filePath = resolve(__dirname, 'fixtures', relPath.slice('/kifu/fixtures/'.length));
          if (serveFile(filePath, req, res)) return;
        }

        next();
      });
    },
  }],
});
