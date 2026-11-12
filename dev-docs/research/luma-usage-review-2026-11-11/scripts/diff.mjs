import fs from 'node:fs';
const [a, b] = process.argv.slice(2).map(f => JSON.parse(fs.readFileSync(f, 'utf8')));
for (const entry of new Set([...Object.keys(a), ...Object.keys(b)])) {
  if (!a[entry]) { console.log('NEW ENTRY', entry, Object.keys(b[entry] || {}).length); continue; }
  if (!b[entry]) { console.log('REMOVED ENTRY', entry); continue; }
  const added = Object.keys(b[entry]).filter(n => !(n in a[entry]));
  const removed = Object.keys(a[entry]).filter(n => !(n in b[entry]));
  if (added.length || removed.length) console.log(`${entry}\n  + ${added.map(n => `${n}(${b[entry][n].kind})`).join(', ')}\n  - ${removed.join(', ')}`);
}
