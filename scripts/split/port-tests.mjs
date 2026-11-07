// Copy tests into the standalone repo, then prune test files whose imports cannot resolve
// inside the fork (e.g. tests of luma graph subpaths that were not forked).
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';

const [sourceRepo, targetRepo] = process.argv.slice(2);
const REWRITES = [
  [/@luma\.gl\/graph-lenses(?=['"/])/g, '@luma-contributors/lenses'],
  [/@luma\.gl\/kepler(?=['"/])/g, '@luma-contributors/kepler'],
  [/@luma\.gl\/graph(?=['"/])/g, '@luma-contributors/graph']
];
const TEXT = /\.(ts|tsx|js|mjs|cjs|json|md|wgsl|glsl|txt|csv|py|geojson)$/;
const SKIP = new Set(['node_modules', 'dist', '__pycache__']);
const walk = dir =>
  fs.existsSync(dir)
    ? fs.readdirSync(dir, {withFileTypes: true}).flatMap(entry => {
        if (SKIP.has(entry.name)) return [];
        const full = path.join(dir, entry.name);
        return entry.isDirectory() ? walk(full) : [full];
      })
    : [];

const MAPPINGS = [
  ['modules/graph/test', 'modules/graph/test'],
  ['modules/graph-lenses/test', 'modules/lenses/test'],
  ['modules/kepler/test', 'modules/kepler/test']
];
for (const [from, to] of MAPPINGS) {
  for (const file of walk(path.join(sourceRepo, from))) {
    const target = path.join(targetRepo, to, path.relative(path.join(sourceRepo, from), file));
    fs.mkdirSync(path.dirname(target), {recursive: true});
    if (TEXT.test(file)) {
      let text = fs.readFileSync(file, 'utf8');
      for (const [pattern, value] of REWRITES) text = text.replace(pattern, value);
      fs.writeFileSync(target, text);
    } else {
      fs.copyFileSync(file, target);
    }
  }
}
for (const file of ['modules/graph-lenses/tsconfig.test.json']) {
  if (fs.existsSync(path.join(sourceRepo, file))) {
    fs.copyFileSync(path.join(sourceRepo, file), path.join(targetRepo, 'modules/lenses/tsconfig.test.json'));
  }
}

process.chdir(targetRepo);
const graphExports = JSON.parse(fs.readFileSync('modules/graph/package.json', 'utf8')).exports;
const npmExports = {};
for (const name of ['core', 'engine', 'shaders']) {
  npmExports[name] = JSON.parse(fs.readFileSync(`node_modules/@luma.gl/${name}/package.json`, 'utf8')).exports;
}
const resolveFile = base => {
  const stripped = base.replace(/\.js$/, '');
  for (const candidate of [base, stripped + '.ts', stripped + '.tsx', stripped + '/index.ts']) {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return candidate;
  }
  return null;
};
const ts = createRequire(path.join(targetRepo, 'package.json'))('typescript');
const getImports = text => ts.preProcessFile(text, true, true).importedFiles.map(entry => entry.fileName.replace(/\?.*$/, ''));
const checkSpecifier = (file, specifier) => {
  if (specifier.startsWith('.')) return Boolean(resolveFile(path.join(path.dirname(file), specifier)));
  if (specifier.startsWith('@luma-contributors/graph/test/')) {
    return Boolean(resolveFile(specifier.replace('@luma-contributors/graph/test/', 'modules/graph/test/')));
  }
  if (specifier === '@luma-contributors/graph' || specifier.startsWith('@luma-contributors/graph/')) {
    const key = specifier === '@luma-contributors/graph' ? '.' : './' + specifier.slice('@luma-contributors/graph/'.length);
    return key in graphExports;
  }
  if (specifier.startsWith('@luma-contributors/lenses')) {
    const rest = specifier.slice('@luma-contributors/lenses'.length);
    return !rest || Boolean(resolveFile('modules/lenses/src' + rest)) || fs.existsSync('modules/lenses/src' + rest);
  }
  const lumaMatch = specifier.match(/^@luma\.gl\/(core|engine|shaders)(\/.*)?$/);
  if (lumaMatch) return (lumaMatch[2] ? '.' + lumaMatch[2] : '.') in npmExports[lumaMatch[1]];
  if (specifier.startsWith('@luma.gl/')) return false; // anari, three, deck layers etc. are not dependencies
  return true;
};

const removed = [];
for (let pass = 0; pass < 20; pass++) {
  let removedThisPass = 0;
  for (const root of ['modules/graph/test', 'modules/lenses/test', 'modules/kepler/test']) {
    for (const file of walk(root).filter(f => /\.(ts|tsx)$/.test(f))) {
      const text = fs.readFileSync(file, 'utf8');
      const bad = getImports(text).find(s => !checkSpecifier(file, s));
      if (bad) {
        fs.rmSync(file);
        removed.push(`${file}  <-  ${bad}`);
        removedThisPass++;
      }
    }
  }
  if (!removedThisPass) break;
}
// drop directories that no longer contain any TypeScript (orphaned fixtures stay if a test remains nearby)
const pruneEmpty = dir => {
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    if (entry.isDirectory()) pruneEmpty(path.join(dir, entry.name));
  }
  if (!fs.readdirSync(dir).length) fs.rmdirSync(dir);
};
for (const root of ['modules/graph/test', 'modules/lenses/test', 'modules/kepler/test']) pruneEmpty(root);
fs.writeFileSync(process.env.REMOVED_LOG, removed.join('\n') + '\n');
const remaining = root => walk(root).filter(f => /\.spec\.tsx?$/.test(f)).length;
console.log('removed test files', removed.length);
for (const root of ['modules/graph/test', 'modules/lenses/test', 'modules/kepler/test']) console.log(root, 'specs kept', remaining(root));
