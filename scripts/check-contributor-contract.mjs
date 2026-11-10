// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Report departures from dev-docs/contributor-contract.md across modules/analysis/src.
//
//   node scripts/check-contributor-contract.mjs                  new findings only; exit 1 if any
//   node scripts/check-contributor-contract.mjs --all            every finding
//   node scripts/check-contributor-contract.mjs --rule <id>      one rule only
//   node scripts/check-contributor-contract.mjs --update-baseline
//
// The baseline (scripts/contributor-contract-baseline.json) records existing findings by rule and
// file (a count, or a list of symbols), never by line number, so unrelated edits do not churn it.
// Heuristic checks are text based on purpose: no TypeScript compiler is used.
import fs from 'node:fs';
import path from 'node:path';

const SOURCE_ROOT = 'modules/analysis/src';
const BASELINE_PATH = 'scripts/contributor-contract-baseline.json';
const SKIPPED_DIRECTORIES = new Set(['testing', 'computation-record']);
const ROOT_FILE_DIRECTORIES = new Set([
  'gpu-network',
  'gpu-terrain',
  'raster-analysis',
  'dataframe-analysis',
  'gpu-spatial-analysis'
]);
const ONTOLOGY_DIRECTORY = `${SOURCE_ROOT}/gpu-spatial-analysis/ontology`;

// Rules reported as one finding per file with an occurrence count. The rest are one per symbol.
const COUNT_RULES = new Set([
  'bare-assert',
  'single-invocation-data-loop',
  'luma-reexport',
  'multiple-classes',
  'root-file'
]);
const RULE_IDENTIFIERS = [
  'bare-assert',
  'missing-producer-declaration',
  'single-invocation-data-loop',
  'internal-export',
  'luma-reexport',
  'multiple-classes',
  'root-file',
  'unregistered-contributor'
];

const walk = directory =>
  fs.readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    if (entry.name === 'node_modules' || entry.name === 'dist') return [];
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) return walk(fullPath);
    return entry.name.endsWith('.ts') && !entry.name.endsWith('.d.ts') ? [fullPath] : [];
  });

const getLineNumber = (text, offset) => text.slice(0, offset).split('\n').length;

const isCommentLine = line => /^\s*(\/\/|\*|\/\*)/.test(line);

// Offset of the closing parenthesis that matches the opening one at `openOffset`. Skips string
// and template literals so WGSL source inside backticks cannot unbalance the scan.
const findMatchingParenthesis = (text, openOffset) => {
  let depth = 0;
  for (let offset = openOffset; offset < text.length; offset++) {
    const character = text[offset];
    if (character === '"' || character === "'" || character === '`') {
      for (offset++; offset < text.length && text[offset] !== character; offset++) {
        if (text[offset] === '\\') offset++;
      }
    } else if (character === '(') {
      depth++;
    } else if (character === ')' && --depth === 0) {
      return offset;
    }
  }
  return text.length;
};

// Text of every initializer of `name` in `text` (`const|let|var name = ...` and `name += ...`), each
// read to the end of its statement. Returns '' when `name` is not declared in the file, as for a
// function parameter. Skips string and template literals like findMatchingParenthesis.
const resolveIdentifierText = (text, name) => {
  const pieces = [];
  const pattern = new RegExp(
    `(?:\\b(?:const|let|var)\\s+${name}\\b[^=;]*=(?!=)|(?<![\\w$.])${name}\\s*\\+=)`,
    'g'
  );
  for (const match of text.matchAll(pattern)) {
    const start = match.index + match[0].length;
    let depth = 0;
    let offset = start;
    for (; offset < text.length; offset++) {
      const character = text[offset];
      if (character === '"' || character === "'" || character === '`') {
        for (offset++; offset < text.length && text[offset] !== character; offset++) {
          if (text[offset] === '\\') offset++;
        }
      } else if ('([{'.includes(character)) {
        depth++;
      } else if (')]}'.includes(character)) {
        if (--depth < 0) break;
      } else if (character === ';' && depth === 0) {
        break;
      }
    }
    pieces.push(text.slice(start, offset));
  }
  return pieces.join('\n');
};

// The text a kernel call's `body` and `declarations` props resolve to when they are identifiers
// (`body: name`, or the shorthand `body,`) declared in the same file.
const getReferencedKernelText = (text, callText) => {
  const pieces = [];
  const pattern = /\b(?:body|declarations)\s*:\s*([A-Za-z_$][\w$]*)\s*(?=[,}\n])|[\n,{]\s*(body|declarations)\s*(?=[,}\n])/g;
  for (const match of callText.matchAll(pattern)) {
    pieces.push(resolveIdentifierText(text, match[1] ?? match[2]));
  }
  return pieces.join('\n');
};

const findings = [];
const addFinding = (rule, file, details) => findings.push({rule, file, ...details});

const sourceFiles = walk(SOURCE_ROOT);
const fileTexts = new Map(sourceFiles.map(file => [file, fs.readFileSync(file, 'utf8')]));
const getTopDirectory = file => path.relative(SOURCE_ROOT, file).split(path.sep)[0];
const isCheckedFile = file => !SKIPPED_DIRECTORIES.has(getTopDirectory(file));
const checkedFiles = sourceFiles.filter(isCheckedFile);

// Names exported by every file under the ontology directory.
const ontologyText = walk(ONTOLOGY_DIRECTORY)
  .map(file => fileTexts.get(file))
  .join('\n');

// Export classes: producers are classes that name GPUCommandNodeProducer, or extend one.
const CLASS_PATTERN = /^export (?:abstract )?class (\w+)([^{]*)\{/gm;
const producerClassNames = new Set();
const parentByClassName = new Map();
for (const text of fileTexts.values()) {
  for (const match of text.matchAll(CLASS_PATTERN)) {
    if (/GPUCommandNodeProducer/.test(match[2])) producerClassNames.add(match[1]);
    const parentMatch = match[2].match(/\bextends\s+(\w+)/);
    if (parentMatch) parentByClassName.set(match[1], parentMatch[1]);
  }
}
const isProducerClass = className => {
  for (let name = className, depth = 0; name && depth < 10; name = parentByClassName.get(name), depth++) {
    if (producerClassNames.has(name)) return true;
  }
  return false;
};

for (const file of checkedFiles) {
  const text = fileTexts.get(file);
  const lines = text.split('\n');
  const relativeToSource = path.relative(SOURCE_ROOT, file).split(path.sep);

  // 1. bare-assert: assert imported from @luma.gl/core and called.
  if (/import\s*\{[^}]*\bassert\b[^}]*\}\s*from\s*'@luma\.gl\/core'/.test(text)) {
    const assertLines = lines
      .map((line, index) => (!isCommentLine(line) && /\bassert\(/.test(line) ? index + 1 : 0))
      .filter(Boolean);
    if (assertLines.length) {
      addFinding('bare-assert', file, {count: assertLines.length, lines: assertLines});
    }
  }

  // Classes with their body text, for rules 2, 6 and 8.
  const classMatches = [...text.matchAll(CLASS_PATTERN)];
  const classBodies = classMatches.map((match, index) => ({
    name: match[1],
    header: match[2],
    offset: match.index,
    body: text.slice(match.index, classMatches[index + 1]?.index ?? text.length)
  }));
  const contributorClasses = classBodies.filter(
    ({name, body}) => /^GPU/.test(name) && /\bgetCommandNodes\s*\(/.test(body)
  );

  // 2. missing-producer-declaration.
  for (const {name, offset} of contributorClasses) {
    if (!isProducerClass(name)) {
      addFinding('missing-producer-declaration', file, {symbol: name, lines: [getLineNumber(text, offset)]});
    }
  }

  // 3. single-invocation-data-loop (heuristic): a for loop inside a one-invocation kernel.
  const invocationLines = [];
  for (const match of text.matchAll(/createWGSLKernelNode\s*\(/g)) {
    const openOffset = match.index + match[0].length - 1;
    const callText = text.slice(openOffset, findMatchingParenthesis(text, openOffset) + 1);
    const invocationMatch = callText.match(/invocationCount:\s*1\b(?!\s*[0-9.*+])/);
    if (!invocationMatch) continue;
    if (!/\bfor\s*\(/.test(`${callText}\n${getReferencedKernelText(text, callText)}`)) continue;
    const line = getLineNumber(text, openOffset + invocationMatch.index);
    if (/contract-allow:\s*single-invocation/.test(lines[line - 2] ?? '')) continue;
    invocationLines.push(line);
  }
  if (invocationLines.length) {
    addFinding('single-invocation-data-loop', file, {count: invocationLines.length, lines: invocationLines});
  }

  // 5. luma-reexport.
  const reexportLines = [
    ...text.matchAll(/^export\s+(?:type\s+)?(?:\*(?:\s+as\s+\w+)?|\{[^}]*\})\s*from\s*['"]@luma\.gl\//gm)
  ].map(match => getLineNumber(text, match.index));
  if (reexportLines.length) {
    addFinding('luma-reexport', file, {count: reexportLines.length, lines: reexportLines});
  }

  // 6. multiple-classes.
  if (classMatches.length > 1) {
    addFinding('multiple-classes', file, {
      count: classMatches.length,
      lines: classMatches.map(match => getLineNumber(text, match.index))
    });
  }

  // 7. root-file: only index.ts directly in a package root directory.
  if (
    relativeToSource.length === 2 &&
    ROOT_FILE_DIRECTORIES.has(relativeToSource[0]) &&
    relativeToSource[1] !== 'index.ts'
  ) {
    addFinding('root-file', file, {count: 1, lines: [1]});
  }

  // 8. unregistered-contributor: exported contributor classes and add*Recipe functions.
  const registrableNames = [
    ...contributorClasses.map(({name, offset}) => ({name, offset})),
    ...[...text.matchAll(/^export function (add\w+Recipe)\b/gm)].map(match => ({
      name: match[1],
      offset: match.index
    }))
  ];
  for (const {name, offset} of registrableNames) {
    if (!new RegExp(`\\b${name}\\b`).test(ontologyText)) {
      addFinding('unregistered-contributor', file, {symbol: name, lines: [getLineNumber(text, offset)]});
    }
  }
}

// 4. internal-export: resolve export chains from each entry barrel by reading files.
const resolveModule = (fromFile, specifier) => {
  if (!specifier.startsWith('.')) return null;
  const base = path.join(path.dirname(fromFile), specifier);
  return [`${base}.ts`, path.join(base, 'index.ts'), base].find(candidate => fileTexts.has(candidate)) ?? null;
};
const DECLARATION_PATTERN =
  /^export\s+(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?(?:const|let|function\*?|class|interface|type|enum)\s+(\w+)/gm;
const internalSymbols = new Set(); // `${file}#${name}`
for (const [file, text] of fileTexts) {
  const documentedPattern = new RegExp(
    `/\\*\\*((?:(?!\\*/)[\\s\\S])*?)\\*/\\s*${DECLARATION_PATTERN.source.slice(1)}`,
    'gm'
  );
  for (const match of text.matchAll(documentedPattern)) {
    if (/@internal\b/.test(match[1])) internalSymbols.add(`${file}#${match[2]}`);
  }
}
const exportCache = new Map();
// Map of exported name to the `${file}#${name}` that defines it.
const getExports = (file, visiting = new Set()) => {
  if (exportCache.has(file)) return exportCache.get(file);
  if (visiting.has(file)) return new Map();
  visiting.add(file);
  const text = fileTexts.get(file);
  const exported = new Map();
  for (const match of text.matchAll(DECLARATION_PATTERN)) exported.set(match[1], `${file}#${match[1]}`);
  for (const match of text.matchAll(/^export\s+(type\s+)?\*\s*from\s*['"]([^'"]+)['"]/gm)) {
    const target = resolveModule(file, match[2]);
    if (target) for (const [name, origin] of getExports(target, visiting)) exported.set(name, origin);
  }
  for (const match of text.matchAll(/^export\s+(?:type\s+)?\{([^}]*)\}(?:\s*from\s*['"]([^'"]+)['"])?/gm)) {
    const target = match[2] ? resolveModule(file, match[2]) : file;
    const targetExports = target && target !== file ? getExports(target, visiting) : exported;
    for (const specifier of match[1].split(',').map(part => part.trim()).filter(Boolean)) {
      const [originalName, exportedName = originalName] = specifier
        .replace(/^type\s+/, '')
        .split(/\s+as\s+/);
      exported.set(exportedName, targetExports.get(originalName) ?? `${target ?? match[2]}#${originalName}`);
    }
  }
  visiting.delete(file);
  exportCache.set(file, exported);
  return exported;
};
const barrelFiles = [`${SOURCE_ROOT}/index.ts`, ...fs
  .readdirSync(SOURCE_ROOT, {withFileTypes: true})
  .filter(entry => entry.isDirectory())
  .map(entry => `${SOURCE_ROOT}/${entry.name}/index.ts`)].filter(file => fileTexts.has(file));
for (const barrel of barrelFiles) {
  for (const [exportedName, origin] of getExports(barrel)) {
    if (internalSymbols.has(origin)) {
      addFinding('internal-export', barrel, {symbol: exportedName, lines: [1], origin});
    }
  }
}

// Baseline handling: { rule: { file: count | [symbols] } }.
const buildBaseline = foundList => {
  const baseline = {};
  for (const finding of [...foundList].sort(compareFindings)) {
    baseline[finding.rule] ??= {};
    if (COUNT_RULES.has(finding.rule)) baseline[finding.rule][finding.file] = finding.count;
    else (baseline[finding.rule][finding.file] ??= []).push(finding.symbol);
  }
  return baseline;
};
function compareFindings(left, right) {
  return (
    left.rule.localeCompare(right.rule) ||
    left.file.localeCompare(right.file) ||
    (left.symbol ?? '').localeCompare(right.symbol ?? '')
  );
}
const loadBaseline = () => (fs.existsSync(BASELINE_PATH) ? JSON.parse(fs.readFileSync(BASELINE_PATH, 'utf8')) : {});
const isNewFinding = (finding, baseline) => {
  const recorded = baseline[finding.rule]?.[finding.file];
  if (recorded === undefined) return true;
  return COUNT_RULES.has(finding.rule) ? finding.count > recorded : !recorded.includes(finding.symbol);
};

// Command line.
const arguments_ = process.argv.slice(2);
const ruleIndex = arguments_.indexOf('--rule');
const selectedRule = ruleIndex >= 0 ? arguments_[ruleIndex + 1] : null;
if (selectedRule && !RULE_IDENTIFIERS.includes(selectedRule)) {
  process.stderr.write(`Unknown rule '${selectedRule}'. Rules: ${RULE_IDENTIFIERS.join(', ')}\n`);
  process.exit(2);
}

if (arguments_.includes('--update-baseline')) {
  const baseline = buildBaseline(findings);
  fs.writeFileSync(BASELINE_PATH, `${JSON.stringify(baseline, null, 2)}\n`);
  process.stdout.write(`Wrote ${BASELINE_PATH} with ${findings.length} findings.\n`);
  process.exit(0);
}

const baseline = loadBaseline();
const shownFindings = findings
  .filter(finding => !selectedRule || finding.rule === selectedRule)
  .sort(compareFindings);
const newFindings = shownFindings.filter(finding => isNewFinding(finding, baseline));
const printedFindings = arguments_.includes('--all') ? shownFindings : newFindings;

process.stdout.write('rule                           total  new\n');
for (const rule of selectedRule ? [selectedRule] : RULE_IDENTIFIERS) {
  const ruleFindings = shownFindings.filter(finding => finding.rule === rule);
  const total = COUNT_RULES.has(rule)
    ? ruleFindings.reduce((sum, finding) => sum + finding.count, 0)
    : ruleFindings.length;
  const newCount = newFindings.filter(finding => finding.rule === rule).length;
  process.stdout.write(`${rule.padEnd(30)} ${String(total).padStart(5)} ${String(newCount).padStart(4)}\n`);
}
for (const finding of printedFindings) {
  const detail = finding.symbol ? ` ${finding.symbol}` : finding.count > 1 ? ` x${finding.count}` : '';
  const marker = isNewFinding(finding, baseline) ? 'NEW ' : '';
  process.stdout.write(`${marker}${finding.rule} ${finding.file}:${finding.lines[0]}${detail}\n`);
}
if (newFindings.length) {
  process.stdout.write(`${newFindings.length} new contributor-contract findings\n`);
  process.exitCode = 1;
} else {
  process.stdout.write('contributor-contract: no new findings\n');
}
