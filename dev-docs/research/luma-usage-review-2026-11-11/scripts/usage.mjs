import fs from 'node:fs';
import path from 'node:path';
const root = process.cwd();
const exportsByEntry = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const areas = {
  'analysis-src': 'modules/analysis/src', 'lenses-src': 'modules/lenses/src', 'kepler-src': 'modules/kepler/src',
  'tests': ['modules/analysis/test', 'modules/lenses/test', 'modules/kepler/test', 'test'],
  'website': 'website/src', 'examples': 'examples', 'docs': 'docs'
};
function* walk(dir) {
  if (!fs.existsSync(dir)) return;
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    if (entry.name === 'node_modules' || entry.name === 'dist' || entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(full);
    else if (/\.(ts|tsx|mts|js|mjs|md|mdx)$/.test(entry.name)) yield full;
  }
}
const usage = {}; // name -> area -> {value, type}
const specifierCounts = {};
const importPattern = /import\s+(type\s+)?([\s\S]*?)\s+from\s+['"](@luma\.gl\/[^'"]+)['"]/g;
for (const [area, dirs] of Object.entries(areas)) {
  for (const dir of [].concat(dirs)) for (const file of walk(path.join(root, dir))) {
    const text = fs.readFileSync(file, 'utf8');
    for (const match of text.matchAll(importPattern)) {
      const typeOnly = Boolean(match[1]);
      const spec = match[3];
      specifierCounts[spec] ??= {};
      specifierCounts[spec][area] = (specifierCounts[spec][area] || 0) + 1;
      const clause = match[2];
      const braces = clause.match(/\{([\s\S]*)\}/);
      const names = [];
      if (braces) for (const part of braces[1].split(',')) {
        const piece = part.trim(); if (!piece) continue;
        const isType = typeOnly || piece.startsWith('type ');
        names.push([piece.replace(/^type\s+/, '').split(/\s+as\s+/)[0].trim(), isType]);
      }
      if (/\*\s+as\s+/.test(clause)) names.push(['*', typeOnly]);
      for (const [name, isType] of names) {
        const key = `${spec}#${name}`;
        usage[key] ??= {};
        usage[key][area] ??= {value: 0, type: 0};
        usage[key][area][isType ? 'type' : 'value']++;
      }
    }
  }
}
fs.writeFileSync(process.argv[3], JSON.stringify({usage, specifierCounts}));
console.log('SPECIFIERS (import statements per area)');
for (const [spec, counts] of Object.entries(specifierCounts).sort()) console.log(' ', spec.padEnd(44), JSON.stringify(counts));
// Unused value exports per entry (matching by name across any specifier, since symbols are re-exported from several entries)
const usedNames = new Map();
for (const [key, areasUsed] of Object.entries(usage)) {
  const name = key.split('#')[1];
  const total = usedNames.get(name) || {src: 0, other: 0};
  for (const [area, c] of Object.entries(areasUsed)) { if (area.endsWith('-src')) total.src += c.value + c.type; else total.other += c.value + c.type; }
  usedNames.set(name, total);
}
const report = {};
for (const [entry, map] of Object.entries(exportsByEntry)) {
  if (!map) continue;
  const classes = Object.entries(map).filter(([, v]) => v.kind === 'class' || v.kind === 'function' || v.kind === 'const');
  const unused = classes.filter(([n]) => !usedNames.has(n)).map(([n, v]) => `${n}`);
  const testOnly = classes.filter(([n]) => usedNames.has(n) && usedNames.get(n).src === 0).map(([n]) => n);
  report[entry] = {values: classes.length, usedInSrc: classes.filter(([n]) => usedNames.has(n) && usedNames.get(n).src > 0).length, unused, testOnly};
}
fs.writeFileSync(process.argv[4], JSON.stringify(report, null, 1));
console.log('\nENTRY value exports / used in src / used only outside src / unused');
for (const [entry, r] of Object.entries(report)) console.log(' ', entry.padEnd(44), r.values, r.usedInSrc, r.testOnly.length, r.unused.length);
