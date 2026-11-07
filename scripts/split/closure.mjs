// Compute the luma graph files that contributor code reaches, and classify them vs alpha.10.
import {execSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const repo = process.argv[2];
const base = process.argv[3] || 'v10.0.0-alpha.10'; // published luma
const masterBase = process.argv[4] || '704183953'; // master at last merge
process.chdir(repo);
const graphSrc = 'modules/graph/src';
const lsTree = rev =>
  new Map(
    execSync(`git ls-tree -r ${rev} ${graphSrc}`, {maxBuffer: 1 << 28})
      .toString()
      .trim()
      .split('\n')
      .map(line => {
        const [meta, file] = line.split('\t');
        return [file, meta.split(' ')[2]];
      })
  );
const alphaFiles = lsTree(base);
const masterFiles = lsTree(masterBase);

const walk = dir =>
  fs.readdirSync(dir, {withFileTypes: true}).flatMap(entry => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return walk(full);
    return /\.(ts|tsx)$/.test(entry.name) ? [full] : [];
  });
const allGraph = walk(graphSrc);
const hashes = new Map();
const chunk = 400;
for (let i = 0; i < allGraph.length; i += chunk) {
  const part = allGraph.slice(i, i + chunk);
  const out = execSync(`git hash-object ${part.map(f => `'${f}'`).join(' ')}`).toString().trim().split('\n');
  part.forEach((f, j) => hashes.set(f, out[j]));
}
const CONTRIB_DIRS = ['analysis-core', 'computation-record', 'gpu-network', 'gpu-spatial-analysis', 'gpu-terrain', 'testing'];
const isContributor = f => {
  const rel = f.slice(graphSrc.length + 1);
  if (CONTRIB_DIRS.some(d => rel.startsWith(d + '/'))) return true;
  return !masterFiles.has(f); // branch-added file
};
const importRe = /(?:from\s+|import\s*\(\s*|import\s+)['"]([^'"]+)['"]/g;
const resolveRel = (from, spec) => {
  const base0 = path.normalize(path.join(path.dirname(from), spec)).replace(/\.js$/, '');
  for (const c of [base0 + '.ts', base0 + '.tsx', base0 + '/index.ts', base0]) if (fs.existsSync(c) && fs.statSync(c).isFile()) return c;
  return null;
};
const subpathUses = new Map();
const graphPkg = JSON.parse(fs.readFileSync('modules/graph/package.json', 'utf8'));
const graphExports = {};
for (const [key, value] of Object.entries(graphPkg.exports)) {
  const types = typeof value === 'string' ? value : value.types || value.import;
  if (!types || key.includes('*')) continue;
  const src = types.replace(/^\.\/dist\//, graphSrc + '/').replace(/\.d\.ts$|\.js$/, '.ts');
  if (fs.existsSync(src)) graphExports[key] = src;
}
const extraSeeds = process.argv.slice(5); // other dirs (graph-lenses, kepler) using @luma.gl/graph/*
const scanFile = f => {
  const text = fs.readFileSync(f, 'utf8');
  const rel = [];
  for (const m of text.matchAll(importRe)) {
    const spec = m[1];
    if (spec.startsWith('.')) {
      const r = resolveRel(f, spec);
      if (r && r.startsWith(graphSrc)) rel.push(r);
    } else if (spec.startsWith('@luma.gl/')) {
      subpathUses.set(spec, (subpathUses.get(spec) || 0) + 1);
      if (process.env.FOLLOW_SUBPATHS && (spec === '@luma.gl/graph' || spec.startsWith('@luma.gl/graph/'))) {
        const target = graphExports[spec === '@luma.gl/graph' ? '.' : './' + spec.slice('@luma.gl/graph/'.length)];
        if (target) rel.push(target);
      }
    }
  }
  return rel;
};
const seeds = allGraph.filter(isContributor);
const needed = new Map(); // non-contributor file -> status
const queue = [];
const seen = new Set();
const visit = r => {
  if (seen.has(r)) return;
  seen.add(r);
  if (isContributor(r)) { queue.push(r); return; }
  const status = !alphaFiles.has(r) ? 'new-since-alpha10' : alphaFiles.get(r) === hashes.get(r) ? 'same' : 'modified';
  needed.set(r, status);
  if (status !== 'same' || process.env.FOLLOW_ALL) queue.push(r);
};
seeds.forEach(visit);
for (const dir of (process.env.EXTRA_DIRS || '').split(',').filter(Boolean)) walk(`${graphSrc}/${dir}`).forEach(visit);
for (const dir of extraSeeds) for (const f of walk(dir)) scanFile(f).forEach(visit);
while (queue.length) scanFile(queue.pop()).forEach(visit);
const byStatus = {};
for (const [f, s] of needed) (byStatus[s] ||= []).push(f);
console.log('contributor seed files:', seeds.length);
for (const [s, list] of Object.entries(byStatus)) {
  console.log(`\n${s}: ${list.length}`);
  const dirs = {};
  for (const f of list) { const d = f.split('/')[3]; dirs[d] = (dirs[d] || 0) + 1; }
  console.log(' by dir', JSON.stringify(dirs));
}
console.log('\n@luma.gl imports:', JSON.stringify(Object.fromEntries([...subpathUses].sort((a, b) => b[1] - a[1]))));
fs.writeFileSync(process.env.OUT || '/dev/null', JSON.stringify({seeds, needed: Object.fromEntries(needed)}, null, 1));
// Whole-graph diff vs alpha10 for reference
let same = 0, mod = 0, added = 0;
for (const f of allGraph) { if (!alphaFiles.has(f)) added++; else if (alphaFiles.get(f) === hashes.get(f)) same++; else mod++; }
console.log(`\nwhole graph src vs ${base}: same ${same}, modified ${mod}, added ${added}`);
