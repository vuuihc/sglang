import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "/Users/vuuihc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/.pnpm/marked@17.0.5/node_modules/marked/lib/marked.esm.js";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "../..");
const sourcePath = path.join(repoRoot, "study-notes/推理优化全景-六个动词.md");
const markdownOut = path.join(repoRoot, "study-notes/推理优化全景-知乎回答粘贴版.md");
const htmlOut = path.join(repoRoot, "study-notes/推理优化全景-知乎回答粘贴版.html");

function normalizeTextSegment(text) {
  if (!/[\u3400-\u9fff]/.test(text)) return text;
  return text
    .replace(/"([^"\n]+)"/g, "“$1”")
    .replace(/,/g, "，")
    .replace(/;/g, "；")
    .replace(/:/g, "：")
    .replace(/\?/g, "？")
    .replace(/!/g, "！");
}

function normalizeProseLine(line) {
  const protectedParts = line.split(/(`[^`]*`|\[[^\]]+\]\([^)]+\)|https?:\/\/\S+)/g);
  return protectedParts
    .map((part) => {
      if (/^`[^`]*`$/.test(part)) return part;
      if (/^\[[^\]]+\]\([^)]+\)$/.test(part)) {
        return part.replace(/^\[([^\]]+)\]/, (_, label) => `[${normalizeTextSegment(label)}]`);
      }
      if (/^https?:\/\//.test(part)) return part;
      return normalizeTextSegment(part);
    })
    .join("");
}

function prepareMarkdown(source) {
  const lines = source.split(/\r?\n/);
  const out = [];
  let inFence = false;
  let skippedTitle = false;

  for (const line of lines) {
    if (!skippedTitle && /^#\s+/.test(line)) {
      skippedTitle = true;
      continue;
    }
    if (/^!\[一条公式,四类杠杆,六个动词\]/.test(line)) continue;
    if (/^```/.test(line)) {
      inFence = !inFence;
      out.push(line);
      continue;
    }
    out.push(inFence ? line : normalizeProseLine(line));
  }

  while (out.length && out[0].trim() === "") out.shift();
  return `${out.join("\n").trim()}\n`;
}

function inlineLocalImages(html) {
  return html.replace(/<img src="([^"]+)" alt="([^"]*)">/g, (full, src, alt) => {
    if (/^(https?:|data:)/.test(src)) return full;
    const absPath = path.resolve(path.dirname(markdownOut), src);
    if (!fs.existsSync(absPath)) throw new Error(`Missing image: ${absPath}`);
    const ext = path.extname(absPath).toLowerCase();
    const mime = ext === ".jpg" || ext === ".jpeg" ? "image/jpeg" : "image/png";
    const base64 = fs.readFileSync(absPath).toString("base64");
    return `<figure><img src="data:${mime};base64,${base64}" alt="${alt}"><figcaption>${alt}</figcaption></figure>`;
  });
}

const source = fs.readFileSync(sourcePath, "utf8");
const preparedMarkdown = prepareMarkdown(source);
fs.writeFileSync(markdownOut, preparedMarkdown);

marked.setOptions({ gfm: true, breaks: false });
const articleHtml = inlineLocalImages(marked.parse(preparedMarkdown));
const documentHtml = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>大模型推理优化｜知乎回答粘贴版</title>
  <style>
    :root { color-scheme: light; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f6f7f8; color: #121212; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",Arial,sans-serif; }
    .toolbar { position: sticky; top: 0; z-index: 10; display: flex; align-items: center; justify-content: center; gap: 14px; padding: 12px 20px; background: rgba(255,255,255,.96); border-bottom: 1px solid #e5e7eb; backdrop-filter: blur(8px); }
    .toolbar button { border: 0; border-radius: 6px; padding: 9px 18px; color: #fff; background: #1772f6; font-size: 15px; cursor: pointer; }
    .toolbar span { color: #6b7280; font-size: 13px; }
    #article { width: min(760px, calc(100% - 32px)); margin: 24px auto 80px; padding: 48px 56px; background: #fff; box-shadow: 0 2px 18px rgba(0,0,0,.06); font-size: 16px; line-height: 1.82; }
    #article > blockquote:first-child { margin-top: 0; font-size: 18px; font-weight: 600; color: #1f2937; }
    h2 { margin: 2.2em 0 .8em; font-size: 25px; line-height: 1.35; }
    h3 { margin: 1.8em 0 .65em; font-size: 20px; line-height: 1.4; }
    h4 { margin: 1.5em 0 .55em; font-size: 17px; line-height: 1.45; }
    p { margin: .9em 0; }
    strong { font-weight: 650; }
    a { color: #175199; text-decoration: none; }
    ul,ol { padding-left: 1.65em; margin: .8em 0; }
    li { margin: .35em 0; }
    blockquote { margin: 1.2em 0; padding: .25em 1em; color: #4b5563; border-left: 4px solid #d1d5db; background: #f9fafb; }
    pre { overflow-x: auto; margin: 1.2em 0; padding: 16px 18px; border-radius: 6px; background: #f6f8fa; font: 14px/1.6 SFMono-Regular,Consolas,"Liberation Mono",Menlo,monospace; white-space: pre-wrap; }
    code { padding: .1em .3em; border-radius: 3px; background: #f3f4f6; font-family: SFMono-Regular,Consolas,"Liberation Mono",Menlo,monospace; font-size: .92em; }
    pre code { padding: 0; background: transparent; }
    table { width: 100%; margin: 1.25em 0; border-collapse: collapse; font-size: 14px; line-height: 1.55; }
    th,td { padding: 9px 10px; border: 1px solid #d8dee4; vertical-align: top; }
    th { background: #f6f8fa; font-weight: 650; }
    hr { margin: 2.3em 0; border: 0; border-top: 1px solid #e5e7eb; }
    figure { margin: 1.6em 0; }
    figure img { display: block; width: 100%; height: auto; border-radius: 4px; }
    figcaption { margin-top: 8px; color: #6b7280; font-size: 13px; text-align: center; }
    @media (max-width: 720px) { #article { width: 100%; margin: 0; padding: 28px 20px 64px; box-shadow: none; } .toolbar span { display: none; } table { font-size: 12px; } th,td { padding: 6px; } }
  </style>
</head>
<body>
  <div class="toolbar">
    <button id="copyButton" type="button">复制正文（富文本）</button>
    <span id="copyStatus">复制后直接粘贴到知乎回答编辑框</span>
  </div>
  <article id="article">${articleHtml}</article>
  <script>
    const button = document.getElementById('copyButton');
    const status = document.getElementById('copyStatus');
    button.addEventListener('click', async () => {
      const article = document.getElementById('article');
      const html = article.innerHTML;
      const plain = article.innerText;
      try {
        if (navigator.clipboard && window.ClipboardItem) {
          await navigator.clipboard.write([new ClipboardItem({
            'text/html': new Blob([html], {type: 'text/html'}),
            'text/plain': new Blob([plain], {type: 'text/plain'})
          })]);
        } else {
          const range = document.createRange();
          range.selectNodeContents(article);
          const selection = window.getSelection();
          selection.removeAllRanges();
          selection.addRange(range);
          document.execCommand('copy');
          selection.removeAllRanges();
        }
        status.textContent = '已复制富文本，可切到知乎粘贴';
        button.textContent = '复制成功';
      } catch (error) {
        const range = document.createRange();
        range.selectNodeContents(article);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        status.textContent = '浏览器未授权剪贴板：正文已全选，请按 Cmd/Ctrl+C';
      }
    });
  </script>
</body>
</html>`;

fs.writeFileSync(htmlOut, documentHtml);
console.log(JSON.stringify({ sourcePath, markdownOut, htmlOut, markdownChars: preparedMarkdown.length, htmlBytes: Buffer.byteLength(documentHtml) }, null, 2));
