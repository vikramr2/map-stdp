// Render every $...$ / $$...$$ span with KaTeX (strict) and flag macros banned by CLAUDE.md.
// Usage: node math_check.js FILE...   Exit 1 with problems on stderr if any are found.
const fs = require('fs');
const path = require('path');

// katex is installed globally in the project env, next to the node binary running this script.
const katexPath = process.env.KATEX_PATH ||
  path.join(path.dirname(process.execPath), '..', 'lib', 'node_modules', 'katex');
const katex = require(katexPath);

const BANNED = ['\\,', '\\;', '\\tag', '\\{', '\\}', '\\|'];
const problems = [];
let spans = 0;

for (const file of process.argv.slice(2)) {
  // Code is not math: drop fenced blocks and inline code before scanning.
  const text = fs.readFileSync(file, 'utf8')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/`[^`\n]*`/g, '');
  const found = [];
  const rest = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, x) => { found.push([x, true]); return ''; });
  rest.replace(/\$([^$\n]+?)\$/g, (_, x) => { found.push([x, false]); return ''; });
  for (const [tex, display] of found) {
    spans++;
    const short = tex.trim().slice(0, 70);
    try {
      katex.renderToString(tex, { displayMode: display, throwOnError: true, strict: 'error' });
    } catch (e) {
      problems.push(`${file}: KaTeX error in "${short}": ${e.message.slice(0, 120)}`);
    }
    for (const b of BANNED) {
      if (tex.includes(b)) problems.push(`${file}: banned macro ${b} in "${short}" (see CLAUDE.md math rules)`);
    }
  }
}

if (problems.length) {
  console.error(problems.join('\n'));
  process.exit(1);
}
console.log(`math_check: ${spans} spans OK`);
