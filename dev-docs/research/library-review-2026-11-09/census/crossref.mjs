// Cross-reference exported contributors against docs, tests, catalog and consumers.
// Usage: node crossref.mjs <repoRoot> <outJson>
import {readdirSync, readFileSync, statSync, writeFileSync, existsSync} from 'node:fs';
import {join, relative} from 'node:path';

const [repositoryRoot, outputPath] = process.argv.slice(2);

function walk(directory, test, files = []) {
  if (!existsSync(directory)) return files;
  for (const name of readdirSync(directory)) {
    if (name === 'node_modules' || name === 'dist' || name.startsWith('.')) continue;
    const path = join(directory, name);
    const stats = statSync(path);
    if (stats.isDirectory()) walk(path, test, files);
    else if (test(name)) files.push(path);
  }
  return files;
}

function identifierFileCounts(directory, test) {
  const counts = new Map();
  for (const file of walk(directory, test)) {
    const text = readFileSync(file, 'utf8');
    const seen = new Set(text.match(/\b(?:GPU[A-Za-z0-9]+|add[A-Z][A-Za-z0-9]+Recipe)\b/g) || []);
    for (const identifier of seen) counts.set(identifier, (counts.get(identifier) || 0) + 1);
  }
  return counts;
}

const isCode = name => /\.(ts|tsx|mjs|js)$/.test(name);
const isMarkdown = name => name.endsWith('.md');

const sourceRoot = join(repositoryRoot, 'modules/analysis/src');
const symbols = [];
for (const file of walk(sourceRoot, name => name.endsWith('.ts'))) {
  const text = readFileSync(file, 'utf8');
  const path = relative(sourceRoot, file);
  for (const match of text.matchAll(/^export class ([A-Za-z0-9_]+)/gm)) {
    symbols.push({name: match[1], kind: 'class', path});
  }
  for (const match of text.matchAll(/^export function (add[A-Za-z0-9_]+Recipe)/gm)) {
    symbols.push({name: match[1], kind: 'recipe', path});
  }
}

// Markdown headings that name a symbol.
const docHeadings = new Map();
for (const file of walk(join(repositoryRoot, 'docs'), isMarkdown)) {
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    if (!line.startsWith('#')) continue;
    for (const identifier of line.match(/\b(?:GPU[A-Za-z0-9]+|add[A-Z][A-Za-z0-9]+Recipe)\b/g) || []) {
      docHeadings.set(identifier, (docHeadings.get(identifier) || 0) + 1);
    }
  }
}

const corpora = {
  docs: identifierFileCounts(join(repositoryRoot, 'docs'), isMarkdown),
  matrix: identifierFileCounts(join(repositoryRoot, 'dev-docs'), name => name === 'coverage-matrix.md'),
  tests: identifierFileCounts(join(repositoryRoot, 'modules/analysis/test'), isCode),
  ontology: identifierFileCounts(join(sourceRoot, 'gpu-spatial-analysis/ontology'), isCode),
  websiteCatalog: identifierFileCounts(join(repositoryRoot, 'website/src/catalog'), isCode),
  websiteRuns: identifierFileCounts(join(repositoryRoot, 'website/src/gpu'), isCode),
  lenses: identifierFileCounts(join(repositoryRoot, 'modules/lenses/src'), isCode),
  kepler: identifierFileCounts(join(repositoryRoot, 'modules/kepler/src'), isCode),
  examples: identifierFileCounts(join(repositoryRoot, 'examples'), isCode)
};

// Barrel reachability: is the symbol's file re-exported (directly or by name) from its package index?
const rows = symbols.map(symbol => {
  const row = {...symbol, package: symbol.path.split('/')[0], family: symbol.path.split('/').slice(0, 2).join('/')};
  for (const [corpus, counts] of Object.entries(corpora)) row[corpus] = counts.get(symbol.name) || 0;
  row.docHeading = docHeadings.get(symbol.name) || 0;
  return row;
});

writeFileSync(outputPath, JSON.stringify(rows, null, 1));

const packages = [...new Set(rows.map(row => row.package))];
const header = ['package', 'symbols', 'docHeading', 'docMention', 'matrix', 'tests', 'ontology', 'websiteCatalog', 'websiteRuns', 'lenses', 'kepler', 'examples'];
console.log(header.join(' | '));
for (const packageName of [...packages, 'ALL']) {
  const selected = packageName === 'ALL' ? rows : rows.filter(row => row.package === packageName);
  const count = key => selected.filter(row => row[key] > 0).length;
  console.log(
    [packageName, selected.length, count('docHeading'), count('docs'), count('matrix'), count('tests'), count('ontology'), count('websiteCatalog'), count('websiteRuns'), count('lenses'), count('kepler'), count('examples')].join(' | ')
  );
}
const list = (label, predicate) => {
  const selected = rows.filter(predicate);
  console.log(`\n${label} (${selected.length}):\n` + selected.map(row => `${row.name} [${row.family}]`).join(', '));
};
list('No test mention', row => row.tests === 0);
list('No docs mention at all', row => row.docs === 0);
list('Not on website (catalog or runs), not in lenses, kepler or examples: no consumer', row => !row.websiteCatalog && !row.websiteRuns && !row.lenses && !row.kepler && !row.examples);
