// Copy the contributor closure into a standalone repo and rewrite package specifiers.
import fs from 'node:fs';
import path from 'node:path';

const [sourceRepo, targetRepo, closureFile] = process.argv.slice(2);
const closure = JSON.parse(fs.readFileSync(closureFile, 'utf8'));
const graphFiles = new Set([...closure.seeds, ...Object.keys(closure.needed)]);

export const SPECIFIER_REWRITES = [
  [/@luma\.gl\/graph-lenses(?=['"/])/g, '@luma-contributors/lenses'],
  [/@luma\.gl\/kepler(?=['"/])/g, '@luma-contributors/kepler'],
  [/@luma\.gl\/graph(?=['"/])/g, '@luma-contributors/graph']
];
const TEXT_EXTENSIONS = /\.(ts|tsx|js|mjs|cjs|json|md|wgsl|glsl|txt|csv)$/;
const rewrite = text => SPECIFIER_REWRITES.reduce((acc, [pattern, value]) => acc.replace(pattern, value), text);

const copyFile = (from, to) => {
  fs.mkdirSync(path.dirname(to), {recursive: true});
  if (TEXT_EXTENSIONS.test(from)) fs.writeFileSync(to, rewrite(fs.readFileSync(from, 'utf8')));
  else fs.copyFileSync(from, to);
};
const walk = dir =>
  fs.existsSync(dir)
    ? fs.readdirSync(dir, {withFileTypes: true}).flatMap(entry => {
        if (['node_modules', 'dist'].includes(entry.name) || entry.name.endsWith('.tsbuildinfo')) return [];
        const full = path.join(dir, entry.name);
        return entry.isDirectory() ? walk(full) : [full];
      })
    : [];

// 1. graph fork: closure source files plus non-TS assets that live next to them
const graphDirs = new Set();
let copied = 0;
for (const file of graphFiles) {
  copyFile(path.join(sourceRepo, file), path.join(targetRepo, file));
  graphDirs.add(path.dirname(file));
  copied++;
}
for (const dir of graphDirs) {
  for (const entry of fs.readdirSync(path.join(sourceRepo, dir), {withFileTypes: true})) {
    if (entry.isFile() && !/\.(ts|tsx)$/.test(entry.name)) {
      copyFile(path.join(sourceRepo, dir, entry.name), path.join(targetRepo, dir, entry.name));
      copied++;
    }
  }
}
console.log('graph files copied', copied);

// 2. lenses and kepler: whole module (src + test + README), minus build output
for (const [from, to] of [
  ['modules/graph-lenses', 'modules/lenses'],
  ['modules/kepler', 'modules/kepler']
]) {
  let count = 0;
  for (const file of walk(path.join(sourceRepo, from))) {
    const relative = path.relative(path.join(sourceRepo, from), file);
    if (relative === 'package.json' || relative.startsWith('tsconfig')) continue;
    copyFile(file, path.join(targetRepo, to, relative));
    count++;
  }
  console.log(from, '->', to, count);
}
