// Generate package manifests and tsconfigs for the standalone repo.
import fs from 'node:fs';

const [sourceRepo, targetRepo] = process.argv.slice(2);
process.chdir(targetRepo);
const VERSION = '0.1.0-alpha.0';
const LUMA = '10.0.0-alpha.10';
const write = (file, json) => fs.writeFileSync(file, JSON.stringify(json, null, 2) + '\n');
const read = file => JSON.parse(fs.readFileSync(`${sourceRepo}/${file}`, 'utf8'));
const repository = directory => ({type: 'git', url: 'https://github.com/visgl/luma-contributors', directory});

// graph fork: keep only exports whose source made it into the fork
const graph = read('modules/graph/package.json');
const graphExports = {};
const dropped = [];
for (const [key, value] of Object.entries(graph.exports)) {
  const types = typeof value === 'string' ? value : value.types;
  const source = types.replace(/^\.\/dist\//, 'modules/graph/src/').replace(/\.d\.ts$/, '.ts');
  if (key.includes('*') || !fs.existsSync(source)) {
    dropped.push(key);
    continue;
  }
  // ./testing imports vitest, which is ESM-only, so it cannot offer a CommonJS entry.
  graphExports[key] = key === './testing' ? {types: value.types, import: value.import} : value;
}
console.log('graph exports kept:', Object.keys(graphExports).join(' '));
console.log('graph exports dropped:', dropped.join(' '));
write('modules/graph/package.json', {
  name: '@luma-contributors/graph',
  version: VERSION,
  publishConfig: {access: 'public'},
  description:
    'GPU analysis operations (spatial analysis, network, terrain, dataframe, raster) for luma.gl. Carries its own fork of the @luma.gl/graph execution layer.',
  license: 'MIT',
  type: 'module',
  repository: repository('modules/graph'),
  types: 'dist/index.d.ts',
  main: 'dist/index.cjs',
  module: 'dist/index.js',
  exports: graphExports,
  files: ['src', 'dist', 'README.md'],
  sideEffects: false,
  dependencies: {
    '@math.gl/core': graph.dependencies['@math.gl/core'],
    '@math.gl/geospatial': graph.dependencies['@math.gl/geospatial'],
    'probe.gl': graph.dependencies['probe.gl']
  },
  peerDependencies: {
    '@luma.gl/core': `~${LUMA}`,
    '@luma.gl/shaders': `~${LUMA}`,
    'apache-arrow': '>=17.0.0',
    vitest: '>=4',
    wgsl_reflect: graph.peerDependencies.wgsl_reflect
  },
  peerDependenciesMeta: {
    'apache-arrow': {optional: true},
    vitest: {optional: true},
    wgsl_reflect: {optional: true}
  }
});

const lenses = read('modules/graph-lenses/package.json');
lenses.name = '@luma-contributors/lenses';
lenses.version = VERSION;
lenses.repository = repository('modules/lenses');
lenses.dependencies = {
  '@luma-contributors/graph': VERSION,
  '@luma.gl/core': LUMA,
  '@luma.gl/shaders': LUMA,
  '@math.gl/core': lenses.dependencies['@math.gl/core'],
  '@math.gl/geospatial': lenses.dependencies['@math.gl/geospatial']
};
write('modules/lenses/package.json', lenses);

const kepler = read('modules/kepler/package.json');
kepler.name = '@luma-contributors/kepler';
kepler.version = VERSION;
kepler.repository = repository('modules/kepler');
const keplerDependencies = {...kepler.dependencies};
delete keplerDependencies['@luma.gl/graph'];
delete keplerDependencies['@luma.gl/graph-lenses'];
// kepler only used engine for DynamicBuffer, which now comes from @luma.gl/core/dynamic
delete keplerDependencies['@luma.gl/engine'];
keplerDependencies['@luma-contributors/graph'] = VERSION;
keplerDependencies['@luma-contributors/lenses'] = VERSION;
kepler.dependencies = Object.fromEntries(Object.entries(keplerDependencies).sort());
write('modules/kepler/package.json', kepler);

// tsconfigs: per-module composite builds, references follow package dependencies
const moduleTsconfig = references => ({
  extends: '../../tsconfig.json',
  include: ['src/**/*', 'src/**/*.json'],
  exclude: ['node_modules'],
  compilerOptions: {composite: true, rootDir: 'src', outDir: 'dist'},
  references: references.map(name => ({path: `../${name}`}))
});
write('modules/graph/tsconfig.json', moduleTsconfig([]));
write('modules/lenses/tsconfig.json', moduleTsconfig(['graph']));
write('modules/kepler/tsconfig.json', moduleTsconfig(['graph', 'lenses']));
write('tsconfig.build.json', {
  files: [],
  references: ['graph', 'lenses', 'kepler'].map(name => ({path: `./modules/${name}`}))
});
