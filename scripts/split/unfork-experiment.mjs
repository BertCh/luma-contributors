// Experiment: delete forked luma graph files and point contributor code at npm @luma.gl/graph.
// Usage: node unfork.mjs <scratch repo>   (scratch repo is a copy of luma-contributors-lib)
import fs from 'node:fs';
import path from 'node:path';

const root = process.argv[2];
process.chdir(root);
const closure = JSON.parse(fs.readFileSync('scripts/split/closure.json', 'utf8'));
const forked = new Set(Object.keys(closure.needed));
// the compat shim was added by hand and is not luma code
for (const file of forked) fs.rmSync(file, {force: true});

const npmGraph = JSON.parse(fs.readFileSync('node_modules/@luma.gl/graph/package.json', 'utf8'));
const npmSubpaths = new Set(Object.keys(npmGraph.exports).filter(key => !key.includes('*')).map(key => key.slice(2)));
const CONTRIBUTOR_SUBPATHS = ['gpu-spatial-analysis', 'gpu-network', 'gpu-terrain', 'computation-record', 'testing'];
const isContributorSubpath = sub => CONTRIBUTOR_SUBPATHS.some(c => sub === c || sub.startsWith(c + '/'));

const walk = dir =>
  fs.readdirSync(dir, {withFileTypes: true}).flatMap(entry => {
    if (['node_modules', 'dist'].includes(entry.name)) return [];
    const full = path.join(dir, entry.name);
    return entry.isDirectory() ? walk(full) : /\.tsx?$/.test(entry.name) ? [full] : [];
  });
const exists = base => {
  const stripped = base.replace(/\.js$/, '');
  return [base, stripped + '.ts', stripped + '.tsx', stripped + '/index.ts'].some(
    c => fs.existsSync(c) && fs.statSync(c).isFile()
  );
};
const stats = {packageRewrites: 0, relativeToNpm: 0, relativeToInternal: 0, unmapped: []};
const graphSrc = path.resolve('modules/graph/src');
const toNpmSpecifier = absoluteTarget => {
  const relative = path.relative(graphSrc, absoluteTarget).replace(/\\/g, '/').replace(/(\/index)?(\.ts)?$/, '');
  const [top, ...rest] = relative.split('/');
  if (top === 'execution') {
    stats.relativeToInternal++;
    return `@luma.gl/graph/_internal/${rest.join('/')}`;
  }
  if (npmSubpaths.has(relative)) return `@luma.gl/graph/${relative}`;
  if (npmSubpaths.has(top)) return `@luma.gl/graph/${top}`; // symbol must be re-exported by the subpath
  if (top === 'utils' || top === 'index') return '@luma.gl/graph';
  return null;
};

for (const file of [...walk('modules/graph/src'), ...walk('modules/lenses/src'), ...walk('modules/kepler/src')]) {
  let text = fs.readFileSync(file, 'utf8');
  const original = text;
  text = text.replace(/(['"])@luma-contributors\/graph(\/[^'"]*)?\1/g, (match, quote, sub = '') => {
    const subpath = sub.slice(1);
    if (subpath && isContributorSubpath(subpath)) return match;
    stats.packageRewrites++;
    return `${quote}@luma.gl/graph${sub}${quote}`;
  });
  text = text.replace(/(from\s+|import\s*\(\s*)(['"])(\.{1,2}\/[^'"]+)\2/g, (match, prefix, quote, specifier) => {
    const target = path.resolve(path.dirname(file), specifier);
    if (exists(target)) return match;
    const npmSpecifier = toNpmSpecifier(target);
    if (!npmSpecifier) {
      stats.unmapped.push(`${file}: ${specifier}`);
      return match;
    }
    stats.relativeToNpm++;
    return `${prefix}${quote}${npmSpecifier}${quote}`;
  });
  if (text !== original) fs.writeFileSync(file, text);
}
// Point tsconfig at npm for luma subpaths; keep source aliases only for contributor subpaths.
const tsconfig = JSON.parse(fs.readFileSync('tsconfig.json', 'utf8'));
const paths = {};
for (const [key, value] of Object.entries(tsconfig.compilerOptions.paths)) {
  if (!key.startsWith('@luma-contributors/graph')) paths[key] = value;
}
for (const sub of CONTRIBUTOR_SUBPATHS) {
  paths[`@luma-contributors/graph/${sub}`] = [`./modules/graph/src/${sub}/index.ts`];
  paths[`@luma-contributors/graph/${sub}/*`] = [`./modules/graph/src/${sub}/*`];
}
tsconfig.compilerOptions.paths = paths;
fs.writeFileSync('tsconfig.json', JSON.stringify(tsconfig, null, 2));
console.log(JSON.stringify({deleted: forked.size, ...stats, unmapped: stats.unmapped.length}, null, 1));
fs.writeFileSync('unmapped.txt', stats.unmapped.join('\n'));
