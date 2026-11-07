// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Verifies src/catalog/entries/*.ts against the generated inventory and the catalog contract.
//
// Usage (from the website directory, Node 22.6+):
//   node --experimental-strip-types scripts/extract-inventory.ts   # refresh the inventory first
//   node --experimental-strip-types scripts/check-catalog.ts       # check every entry file
//   node --experimental-strip-types scripts/check-catalog.ts local-moran.ts   # only matching files
//
// Errors (exit code 1): duplicate id, unknown symbol, wrong importPath or kind, a param name that
// is not a prop of the symbol, unknown facet ids, summary over 120 characters, explanation over
// three sentences, more than eight params, example over 25 lines, unknown related ids, missing
// lens scenario, unknown reference slug, param or output descriptions over 80 characters, and
// preview captions over 80 characters.
// Also errors: types containing an ellipsis, types that are just `object` or `unknown`, and more than
// eight outputs. Warnings: omitted required props (`output` is exempt: the Returns section covers it) and documented
// defaults that are not listed.
// Warnings: params that are not in the props but only because the symbol has no resolved props,
// and the coverage report (inventory symbols without an entry, grouped by directory).
// Output checks (when the inventory lists output fields): ERROR for an output that is not a field,
// or a required field that is omitted; WARNING for omitted optional fields when fewer than three
// outputs are listed. Errors are also written to the machine-readable report (REPORT_FILE) grouped
// by entry file.
// Pass --quiet to hide the per-directory coverage listing.

import {existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import ts from 'typescript';

const websiteDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const entriesDirectory = join(websiteDirectory, 'src/catalog/entries');
const REPORT_FILE =
  '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-vis-gl-build-luma-contributors/8b3242ff-688f-4f91-850e-f132a3086553/scratchpad/site2/checker/report.json';
const MINIMUM_OUTPUTS_FOR_QUIET_OPTIONAL = 3;

type InventoryFile = {
  entryPoints: string[];
  docSlugs: string[];
  lensScenarioIds: string[];
  symbols: {
    symbol: string;
    kind: string;
    importPath: string;
    directory: string;
    props: {name: string; optional?: boolean; jsdoc?: string}[];
    outputs?: {name: string; optional: boolean}[];
  }[];
};

type LooseEntry = Record<string, any>;
type Facet = {id: string};
type Taxonomy = {DATA_TYPES: Facet[]; USE_CASES: Facet[]; DOMAINS: Facet[]};

async function importTypeScriptFile<T>(file: string): Promise<T> {
  const output = ts.transpileModule(readFileSync(file, 'utf8'), {
    compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}
  });
  return import(`data:text/javascript;base64,${Buffer.from(output.outputText).toString('base64')}`);
}

function countSentences(text: string): number {
  // Ignore periods inside inline code, numbers and abbreviations such as "e.g.".
  const stripped = text.replace(/`[^`]*`/g, 'code').replace(/\b(e\.g|i\.e|vs|etc)\./gi, '$1').replace(/\d\.\d/g, '0');
  return (stripped.match(/[.!?](\s|$)/g) ?? []).length || 1;
}

async function main(): Promise<void> {
  const quiet = process.argv.includes('--quiet');
  const filters = process.argv.slice(2).filter(argument => !argument.startsWith('--'));
  const inventory: InventoryFile = JSON.parse(readFileSync(join(websiteDirectory, 'src/catalog/generated/inventory.json'), 'utf8'));
  const taxonomy = await importTypeScriptFile<Taxonomy>(join(websiteDirectory, 'src/catalog/taxonomy.ts'));
  const inventoryBySymbol = new Map(inventory.symbols.map(symbol => [symbol.symbol, symbol]));
  const dataTypeIds = new Set(taxonomy.DATA_TYPES.map(facet => facet.id));
  const useCaseIds = new Set(taxonomy.USE_CASES.map(facet => facet.id));
  const domainIds = new Set(taxonomy.DOMAINS.map(facet => facet.id));
  const lensScenarioIds = new Set(inventory.lensScenarioIds);
  const docSlugs = new Set(inventory.docSlugs);

  const files = existsSync(entriesDirectory)
    ? readdirSync(entriesDirectory).filter(name => name.endsWith('.ts') && (filters.length === 0 || filters.some(filter => name.includes(filter))))
    : [];

  const errors: string[] = [];
  const errorsByFile: Record<string, string[]> = {};
  const warnings: string[] = [];
  const entries: {file: string; entry: LooseEntry}[] = [];
  for (const file of files) {
    try {
      const loaded = await importTypeScriptFile<{entries?: LooseEntry[]}>(join(entriesDirectory, file));
      if (!Array.isArray(loaded.entries)) errors.push(`${file}: does not export \`entries\` as an array`);
      else for (const entry of loaded.entries) entries.push({file, entry});
    } catch (error) {
      errors.push(`${file}: failed to load (${error instanceof Error ? error.message : String(error)})`);
    }
  }

  const allIds = new Set<string>();
  const seenIds = new Map<string, string>();
  for (const {file, entry} of entries) {
    if (typeof entry.id === 'string') {
      if (seenIds.has(entry.id)) errors.push(`${file}: duplicate id "${entry.id}" (also in ${seenIds.get(entry.id)})`);
      seenIds.set(entry.id, file);
      allIds.add(entry.id);
    }
  }
  const coveredSymbols = new Set<string>();

  for (const {file, entry} of entries) {
    const label = `${file}: ${entry.id ?? '(no id)'}`;
    const fail = (message: string) => {
      errors.push(`${label}: ${message}`);
      (errorsByFile[file] ??= []).push(`${entry.id ?? '(no id)'}: ${message}`);
    };
    if (typeof entry.id !== 'string' || !/^[a-z0-9]+(-[a-z0-9]+)*$/.test(entry.id)) fail('id must be kebab-case');

    const inventorySymbol = inventoryBySymbol.get(entry.symbol);
    if (!inventorySymbol) {
      fail(`symbol "${entry.symbol}" is not in the inventory`);
    } else {
      coveredSymbols.add(entry.symbol);
      if (entry.importPath !== inventorySymbol.importPath) fail(`importPath "${entry.importPath}" should be "${inventorySymbol.importPath}"`);
      if (entry.kind !== inventorySymbol.kind) fail(`kind "${entry.kind}" should be "${inventorySymbol.kind}"`);
      const propNames = new Set(inventorySymbol.props.map(prop => prop.name));
      const outputFields = inventorySymbol.outputs ?? [];
      if (outputFields.length > 0 && Array.isArray(entry.outputs)) {
        const fieldNames = new Set(outputFields.map(field => field.name));
        // Entries may group fields ("a / b") or document nested members ("status.converged",
        // "levels[i].vertices"); each part counts as naming its root field.
        const listed = new Set<string>();
        const unknownNames: string[] = [];
        for (const output of entry.outputs as {name: string}[]) {
          for (const part of output.name.split(/\s*[\/,]\s*/)) {
            const root = part.split(/[.\[]/)[0];
            if (fieldNames.has(root)) listed.add(root);
            else if (propNames.has(root)) continue; // a result nested in an input, e.g. raster.output.triangles
            else unknownNames.push(part);
          }
        }
        for (const name of unknownNames) fail(`output "${name}" is not a field of ${entry.symbol} outputs (fields: ${[...fieldNames].join(', ')})`);
        for (const field of outputFields) {
          if (listed.has(field.name)) continue;
          if (!field.optional) fail(`required output "${field.name}" is not listed`);
          else if (listed.size < MINIMUM_OUTPUTS_FOR_QUIET_OPTIONAL) warnings.push(`${label}: optional output "${field.name}" is not listed`);
        }
      }
      const listedParams = new Set((entry.params ?? []).map((param: {name: string}) => param.name));
      const omittedRequired = inventorySymbol.props.filter(prop => prop.optional === false && prop.name !== 'output' && !listedParams.has(prop.name)).map(prop => prop.name);
      if (omittedRequired.length > 0) warnings.push(`${label}: required props not listed: ${omittedRequired.join(', ')}`);
      for (const param of entry.params ?? []) {
        const documented = inventorySymbol.props.find(prop => prop.name === param.name);
        if (documented?.jsdoc && /\bdefaults? (to|is)\b/i.test(documented.jsdoc) && !param.default && !param.required) {
          warnings.push(`${label}: param "${param.name}" has a documented default but lists none`);
        }
        if (propNames.size === 0) warnings.push(`${label}: param "${param.name}" cannot be checked (symbol has no resolved props)`);
        else if (!propNames.has(param.name)) fail(`param "${param.name}" is not a prop of ${entry.symbol} (props: ${[...propNames].join(', ')})`);
      }
    }

    if (!entry.title) fail('missing title');
    if (typeof entry.summary !== 'string' || entry.summary.length === 0) fail('missing summary');
    else if (entry.summary.length > 120) fail(`summary is ${entry.summary.length} characters (max 120)`);
    if (typeof entry.explanation !== 'string' || entry.explanation.length === 0) fail('missing explanation');
    else if (countSentences(entry.explanation) > 3) fail(`explanation has ${countSentences(entry.explanation)} sentences (max 3)`);

    for (const [key, ids, valid] of [
      ['dataTypes.input', entry.dataTypes?.input, dataTypeIds],
      ['dataTypes.output', entry.dataTypes?.output, dataTypeIds],
      ['useCases', entry.useCases, useCaseIds],
      ['domains', entry.domains, domainIds]
    ] as const) {
      if (!Array.isArray(ids)) fail(`${key} must be an array`);
      else for (const id of ids) if (!valid.has(id)) fail(`${key} has unknown id "${id}"`);
    }
    if (!(entry.useCases?.length >= 1 && entry.useCases.length <= 3)) fail('useCases must have one to three ids');
    if (!(entry.domains?.length >= 1 && entry.domains.length <= 3)) fail('domains must have one to three ids');
    if (!(entry.dataTypes?.input?.length >= 1)) fail('dataTypes.input must have at least one id');
    if (!(entry.dataTypes?.output?.length >= 1)) fail('dataTypes.output must have at least one id');

    if ((entry.params?.length ?? 0) > 8) fail(`${entry.params.length} params (max 8)`);
    for (const param of entry.params ?? []) {
      if ((param.description?.length ?? 0) > 80) fail(`param "${param.name}" description is over 80 characters`);
      if (!param.description) fail(`param "${param.name}" has no description`);
    }
    if (!Array.isArray(entry.outputs) || entry.outputs.length === 0) fail('outputs must list at least one output');
    else if (entry.outputs.length > 8) fail(`${entry.outputs.length} outputs (max 8)`);
    for (const item of [...(entry.params ?? []), ...(entry.outputs ?? [])] as {name: string; type: string}[]) {
      if (/…|\.\.\./.test(item.type)) fail(`"${item.name}" has a truncated type: ${item.type}`);
      if (/^(object|unknown)$/.test(item.type)) fail(`"${item.name}" has a vague type "${item.type}"; name the real type or shape`);
    }
    for (const output of entry.outputs ?? []) {
      if ((output.description?.length ?? 0) > 80) fail(`output "${output.name}" description is over 80 characters`);
    }
    if (typeof entry.example !== 'string' || entry.example.trim().length === 0) fail('missing example');
    else {
      const lineCount = entry.example.replace(/\n+$/, '').split('\n').length;
      if (lineCount > 25) fail(`example is ${lineCount} lines (max 25)`);
      if (!entry.example.includes(entry.symbol)) fail(`example does not use ${entry.symbol}`);
    }
    if (!entry.preview?.kind) fail('missing preview.kind');
    if ((entry.preview?.caption?.length ?? 0) > 80) fail('preview caption is over 80 characters');
    for (const relatedId of entry.related ?? []) {
      if (!allIds.has(relatedId) && filters.length === 0) fail(`related id "${relatedId}" does not exist`);
      else if (!allIds.has(relatedId)) warnings.push(`${label}: related id "${relatedId}" not in the filtered files`);
    }
    if (entry.lensScenario) {
      if (!lensScenarioIds.has(entry.lensScenario) || !existsSync(join(websiteDirectory, '..', 'docs/lens-records', `${entry.lensScenario}.json`))) {
        fail(`lensScenario "${entry.lensScenario}" does not exist in docs/lens-records`);
      }
    }
    if (entry.reference && !docSlugs.has(entry.reference)) fail(`reference "${entry.reference}" is not a docs page (see docSlugs in inventory.json)`);
    if (!['stable', 'partial', 'experimental'].includes(entry.maturity)) fail('maturity must be stable, partial or experimental');
  }

  const uncovered = inventory.symbols.filter(symbol => !coveredSymbols.has(symbol.symbol));
  console.log(`entries: ${entries.length} in ${files.length} files; inventory: ${inventory.symbols.length} symbols`);
  console.log(`coverage: ${inventory.symbols.length - uncovered.length}/${inventory.symbols.length} (${uncovered.length} uncovered)`);
  if (!quiet && uncovered.length > 0 && filters.length === 0) {
    const byDirectory = new Map<string, string[]>();
    for (const symbol of uncovered) byDirectory.set(symbol.directory, [...(byDirectory.get(symbol.directory) ?? []), symbol.symbol]);
    for (const [directory, names] of [...byDirectory].sort(([left], [right]) => left.localeCompare(right))) {
      console.log(`  uncovered ${directory}: ${names.join(', ')}`);
    }
  }
  for (const warning of warnings) console.log(`warning: ${warning}`);
  for (const error of errors) console.log(`error: ${error}`);
  console.log(`${errors.length} errors, ${warnings.length} warnings`);
  mkdirSync(dirname(REPORT_FILE), {recursive: true});
  const counts = Object.fromEntries(Object.entries(errorsByFile).map(([file, list]) => [file, list.length]));
  writeFileSync(REPORT_FILE, JSON.stringify({totalErrors: errors.length, totalWarnings: warnings.length, counts, errorsByFile, warnings}, null, 2));
  process.exitCode = errors.length > 0 ? 1 : 0;
}

await main();
