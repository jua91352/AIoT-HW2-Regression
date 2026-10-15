// 本機開發伺服器：node scripts/dev-server.js → http://localhost:5173
// 提供靜態檔案，並把 /api/stock 交給 api/stock.js（模擬 Vercel Serverless Function）
const http = require("http");
const fs = require("fs");
const path = require("path");
const stock = require("../api/stock.js");

const ROOT = path.resolve(__dirname, "..");
const PORT = Number(process.env.PORT) || 5173;
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".csv": "text/csv; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
};

http
  .createServer((req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    if (url.pathname === "/api/stock") {
      req.query = Object.fromEntries(url.searchParams);
      res.status = (code) => ((res.statusCode = code), res);
      res.json = (obj) => {
        res.setHeader("Content-Type", "application/json; charset=utf-8");
        res.end(JSON.stringify(obj));
      };
      res.send = (body) => res.end(body);
      return stock(req, res);
    }
    let file = path.join(ROOT, decodeURIComponent(url.pathname));
    if (!file.startsWith(ROOT)) {
      res.statusCode = 403;
      return res.end();
    }
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
    fs.readFile(file, (err, data) => {
      if (err) {
        res.statusCode = 404;
        return res.end("Not found");
      }
      res.setHeader("Content-Type", TYPES[path.extname(file)] || "application/octet-stream");
      res.setHeader("Cache-Control", "no-store");
      res.end(data);
    });
  })
  .listen(PORT, () => console.log(`HW2 dev server: http://localhost:${PORT}`));
