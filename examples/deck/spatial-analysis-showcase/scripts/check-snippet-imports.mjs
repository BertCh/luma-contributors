// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Checks the import statements shown to the reader in code snippets. Snippets are string
 * templates (`snippet:` in `*.scene.ts`) that no compiler sees, so they drift from the packages.
 * Every `import ... from '...'` found inside a string literal of the showcase and explorer
 * sources, or inside a markdown file of either example, is resolved with the TypeScript compiler:
 * `@luma-contributors/analysis/<entry>` against the barrel in `modules/analysis/src/<directory>`,
 * every other package against its published declaration files. A missing module or a name the
 * module does not export is reported with file, line, name and module.
 *
 * Not checked: the shape of props and outputs (see the neighbouring `*.compute.ts` for those),
 * code that is not an import statement, and names used in a snippet without being imported.
 *
 * Usage: `node scripts/check-snippet-imports.mjs`. Exit code 1 on problems. Needs no browser.
 */
import {existsSync, readdirSync, readFileSync, statSync} from 'node:fs';
import {dirname, join, relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';

const SCRIPT_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const SHOWCASE_DIRECTORY = join(SCRIPT_DIRECTORY, '..');
const EXPLORER_DIRECTORY = join(SHOWCASE_DIRECTORY, '../spatial-analysis-explorer');
const REPOSITORY_DIRECTORY = join(SHOWCASE_DIRECTORY, '../../..');
const ANALYSIS_SOURCE_DIRECTORY = join(REPOSITORY_DIRECTORY, 'modules/analysis/src');
const SKIPPED_DIRECTORIES = new Set(['node_modules', 'dist', 'scripts', '.git']);
const IMPORT_PATTERN = /\bimport\s+(type\s+)?([^;'"`]*?)\s*from\s*['"]([^'"\n]+)['"]/g;

/** @returns {string[]} */
function listFiles(directory) {
  if (!existsSync(directory)) {
    return [];
  }
  const files = [];
  for (const entry of readdirSync(directory)) {
    if (SKIPPED_DIRECTORIES.has(entry)) {
      continue;
    }
    const path = join(directory, entry);
    if (statSync(path).isDirectory()) {
      files.push(...listFiles(path));
    } else if (/\.(ts|tsx|md)$/.test(entry) && !/\.d\.ts$/.test(entry)) {
      files.push(path);
    }
  }
  return files;
}

/** Parses `{A, B as C}`, default and namespace clauses into the imported names to verify. */
function parseImportedNames(clause) {
  const names = [];
  const namedMatch = clause.match(/\{([\s\S]*)\}/);
  if (namedMatch) {
    for (const part of namedMatch[1].split(',')) {
      const name = part
        .replace(/\/\/.*$/gm, '')
        .trim()
        .replace(/^type\s+/, '')
        .split(/\s+as\s+/)[0]
        .trim();
      if (/^[A-Za-z_$][\w$]*$/.test(name)) {
        names.push(name);
      }
    }
  }
  const defaultName = clause.replace(/\{[\s\S]*\}/, '').replace(/,/g, '').trim();
  if (/^[A-Za-z_$][\w$]*$/.test(defaultName)) {
    names.push('default');
  }
  return names;
}

/** Finds import statements inside the text of one string-like region. */
function collectImports(text, firstLine, file, imports) {
  IMPORT_PATTERN.lastIndex = 0;
  for (let match = IMPORT_PATTERN.exec(text); match; match = IMPORT_PATTERN.exec(text)) {
    const line = firstLine + text.slice(0, match.index).split('\n').length - 1;
    imports.push({file, line, clause: match[2], moduleName: match[3]});
  }
}

function collectFileImports(file, imports) {
  const text = readFileSync(file, 'utf8');
  if (file.endsWith('.md')) {
    collectImports(text, 1, file, imports);
    return;
  }
  const sourceFile = typescript.createSourceFile(
    file,
    text,
    typescript.ScriptTarget.ESNext,
    true,
    file.endsWith('.tsx') ? typescript.ScriptKind.TSX : typescript.ScriptKind.TS
  );
  const lineOf = node => sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1;
  const visit = node => {
    if (
      typescript.isStringLiteral(node) ||
      typescript.isNoSubstitutionTemplateLiteral(node)
    ) {
      // The literal's own text keeps its source line breaks only for templates.
      collectImports(node.text, lineOf(node), file, imports);
    } else if (typescript.isTemplateExpression(node)) {
      let line = lineOf(node);
      collectImports(node.head.text, line, file, imports);
      line += node.head.text.split('\n').length - 1;
      for (const span of node.templateSpans) {
        collectImports(span.literal.text, line, file, imports);
        line += span.literal.text.split('\n').length - 1;
      }
      node.templateSpans.forEach(span => visit(span.expression));
      return;
    }
    typescript.forEachChild(node, visit);
  };
  visit(sourceFile);
}

const compilerOptions = {
  target: typescript.ScriptTarget.ESNext,
  module: typescript.ModuleKind.ESNext,
  moduleResolution: typescript.ModuleResolutionKind.Bundler,
  skipLibCheck: true,
  noEmit: true,
  types: [],
  baseUrl: REPOSITORY_DIRECTORY,
  paths: {'@luma-contributors/analysis/*': ['modules/analysis/src/*']}
};
const containingFile = join(SHOWCASE_DIRECTORY, 'src/snippet-imports-check.ts');

/** Maps `@luma-contributors/analysis/<entry>` to the barrel under `src`; the root maps to `src/index.ts`. */
function resolveModule(moduleName) {
  if (moduleName === '@luma-contributors/analysis') {
    return join(ANALYSIS_SOURCE_DIRECTORY, 'index.ts');
  }
  if (moduleName.startsWith('@luma-contributors/analysis/')) {
    const barrel = join(
      ANALYSIS_SOURCE_DIRECTORY,
      moduleName.slice('@luma-contributors/analysis/'.length),
      'index.ts'
    );
    return existsSync(barrel) ? barrel : null;
  }
  const resolved = typescript.resolveModuleName(
    moduleName,
    containingFile,
    compilerOptions,
    typescript.sys
  ).resolvedModule;
  return resolved ? resolved.resolvedFileName : null;
}

const files = [...listFiles(SHOWCASE_DIRECTORY), ...listFiles(EXPLORER_DIRECTORY)];
const imports = [];
for (const file of files) {
  collectFileImports(file, imports);
}

const moduleFiles = new Map();
for (const {moduleName} of imports) {
  if (!moduleFiles.has(moduleName)) {
    moduleFiles.set(moduleName, resolveModule(moduleName));
  }
}
const program = typescript.createProgram(
  [...moduleFiles.values()].filter(Boolean),
  compilerOptions
);
const checker = program.getTypeChecker();
const exportCache = new Map();

/** @returns {Set<string> | null} null when the file is not a module with exports. */
function getExportNames(moduleFile) {
  if (!exportCache.has(moduleFile)) {
    const sourceFile = program.getSourceFile(moduleFile);
    const symbol = sourceFile && checker.getSymbolAtLocation(sourceFile);
    exportCache.set(
      moduleFile,
      symbol ? new Set(checker.getExportsOfModule(symbol).map(exported => exported.name)) : null
    );
  }
  return exportCache.get(moduleFile);
}

const problems = [];
let checkedNames = 0;
const snippetFiles = new Set();
for (const {file, line, clause, moduleName} of imports) {
  if (moduleName.startsWith('.')) {
    continue;
  }
  snippetFiles.add(file);
  const location = `${relative(REPOSITORY_DIRECTORY, file)}:${line}`;
  const moduleFile = moduleFiles.get(moduleName);
  if (!moduleFile) {
    problems.push(`${location}  module '${moduleName}' does not resolve`);
    continue;
  }
  const exportNames = getExportNames(moduleFile);
  for (const name of parseImportedNames(clause)) {
    checkedNames++;
    if (exportNames && name !== 'default' && !exportNames.has(name)) {
      problems.push(`${location}  '${name}' is not exported by '${moduleName}'`);
    }
  }
}

console.log(
  `Checked ${imports.length} import statements (${checkedNames} names) in ${snippetFiles.size} files.`
);
if (problems.length > 0) {
  console.error(`${problems.length} problems:`);
  for (const problem of problems) {
    console.error(`  ${problem}`);
  }
  process.exit(1);
}
console.log('Every snippet import resolves.');
