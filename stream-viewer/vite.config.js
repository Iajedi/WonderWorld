import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
/** Output directory: <stream-viewer>/debug-export (see .gitignore). */
const DEBUG_EXPORT_DIR = path.join(__dirname, 'debug-export');

function debugSavePlugin() {
  return {
    name: 'debug-save-export',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url !== '/__debug/save-image' || req.method !== 'POST') {
          next();
          return;
        }
        let body = '';
        req.on('data', (chunk) => {
          body += chunk;
        });
        req.on('end', () => {
          try {
            const { filename, dataUrl } = JSON.parse(body);
            if (!filename || typeof dataUrl !== 'string') {
              res.statusCode = 400;
              res.end('missing filename or dataUrl');
              return;
            }
            const m = /^data:image\/png;base64,(.+)$/i.exec(dataUrl);
            if (!m) {
              res.statusCode = 400;
              res.end('expected data:image/png;base64,...');
              return;
            }
            const safe = path.basename(filename).replace(/[^a-zA-Z0-9._-]/g, '_');
            const dest = path.join(DEBUG_EXPORT_DIR, safe);
            fs.mkdirSync(DEBUG_EXPORT_DIR, { recursive: true });
            fs.writeFileSync(dest, Buffer.from(m[1], 'base64'));
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ ok: true, file: safe }));
          } catch (e) {
            res.statusCode = 500;
            res.end(e instanceof Error ? e.message : 'save failed');
          }
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), debugSavePlugin()],
});
