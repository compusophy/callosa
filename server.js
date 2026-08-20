// Local static file server, for development only.
//
// The deployed build is pure static hosting with no backend of any kind: peers
// find each other through BroadcastChannel or by exchanging a blob by hand, so
// there is nothing here to relay signaling. This exists purely so `npm start`
// serves `public/` with the right MIME types.

import http from 'node:http';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT) || 3000;
const HOST = process.env.HOST || '0.0.0.0';
const PUBLIC_DIR = path.resolve(__dirname, 'public');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

/** Resolve a request path inside PUBLIC_DIR, or null if it escapes. */
function resolveStatic(urlPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(urlPath.split('?')[0].split('#')[0]);
  } catch {
    return null;
  }
  if (decoded === '/' || decoded === '') decoded = '/index.html';

  const resolved = path.resolve(PUBLIC_DIR, `.${path.posix.normalize(decoded)}`);
  if (resolved !== PUBLIC_DIR && !resolved.startsWith(PUBLIC_DIR + path.sep)) {
    return null;
  }
  return resolved;
}

const server = http.createServer(async (req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.writeHead(405, { Allow: 'GET, HEAD' });
    return res.end('method not allowed');
  }

  const filePath = resolveStatic(req.url);
  if (!filePath) {
    res.writeHead(403);
    return res.end('forbidden');
  }

  let stats;
  try {
    stats = await fsp.stat(filePath);
  } catch {
    res.writeHead(404);
    return res.end('not found');
  }
  if (!stats.isFile()) {
    res.writeHead(404);
    return res.end('not found');
  }

  res.writeHead(200, {
    'Content-Type': MIME_TYPES[path.extname(filePath).toLowerCase()] || 'application/octet-stream',
    'Content-Length': stats.size,
    // The wasm bundle is rebuilt constantly during development; never let a
    // stale copy survive a reload.
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
  });
  if (req.method === 'HEAD') return res.end();

  const stream = fs.createReadStream(filePath);
  stream.on('error', () => res.destroy());
  stream.pipe(res);
});

server.listen(PORT, HOST, () => {
  console.log(`[server] serving public/ at http://localhost:${PORT}`);
  console.log('[server] open two tabs: leave one as node 0, set the other to node 1');
});
