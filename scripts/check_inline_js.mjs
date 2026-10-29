// Syntax-check the inline <script> blocks of HTML pages without running them.
// Usage: node scripts/check_inline_js.mjs static/dashboard.html [more.html ...]
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const files = process.argv.slice(2);
if (!files.length) {
  console.error('usage: node scripts/check_inline_js.mjs <page.html> [...]');
  process.exit(2);
}

let failures = 0;
for (const file of files) {
  const html = readFileSync(file, 'utf8');
  const blocks = [...html.matchAll(/<script(?![^>]*\bsrc=)([^>]*)>([\s\S]*?)<\/script>/g)];
  if (!blocks.length) {
    console.error(`${file}: no inline scripts found`);
    failures += 1;
    continue;
  }
  blocks.forEach((match, index) => {
    const line = html.slice(0, match.index).split('\n').length;
    if (/\btype\s*=\s*["']?module/.test(match[1])) {
      console.error(`FAIL ${file} script ${index + 1} (line ${line}): module scripts are not supported by this checker`);
      failures += 1;
      return;
    }
    try {
      // Classic scripts are compiled exactly as the browser parses them, never run.
      new vm.Script(match[2], { filename: `${file}#script${index + 1}`, lineOffset: line - 1 });
      console.log(`ok   ${file} script ${index + 1} (line ${line})`);
    } catch (error) {
      console.error(`FAIL ${file} script ${index + 1} (line ${line}): ${error.message}`);
      failures += 1;
    }
  });
}
process.exit(failures ? 1 : 0);
