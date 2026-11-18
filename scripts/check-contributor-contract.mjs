// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
// Report departures from dev-docs/contributor-contract.md across modules/analysis/src.
//
//   node scripts/check-contributor-contract.mjs                  new findings only; exit 1 if any
//   node scripts/check-contributor-contract.mjs --all            every finding
//   node scripts/check-contributor-contract.mjs --rule <id>      one rule only
//   node scripts/check-contributor-contract.mjs --all --paths <file-or-directory> [...]
//   node scripts/check-contributor-contract.mjs --update-baseline
//   node scripts/check-contributor-contract.mjs --update-baseline --reason '<reviewed departure>'
//   node scripts/check-contributor-contract.mjs --check-stale   read-only expired-site audit
//
// The baseline (scripts/contributor-contract-baseline.json) records existing findings by rule and
// file and stable site signature, with a reason; line-number shifts do not renew an allowance.
// TypeScript is parsed without typechecking. WGSL checks inspect statically resolvable snippets;
// they are heuristics, not a replacement for compiling and testing the shaders.
import fs from 'node:fs';
import path from 'node:path';
import typeScript from 'typescript';
import {findInternalPublicMembers} from './contributor-contract-public-surface.mjs';

const SOURCE_ROOT = 'modules/analysis/src';
const BASELINE_PATH = 'scripts/contributor-contract-baseline.json';
const SKIPPED_DIRECTORIES = new Set(['testing', 'computation-record']);
// These public entries organize infrastructure rather than contributor families.
const ROOT_FILE_EXCEPTIONS = new Map([
  ['.', 'Package-wide forwarding barrel; source-root placement is checked by upstream-first.'],
  ['./computation-record', 'Recording infrastructure, not an operator family.'],
  ['./gpu-spatial-analysis/ontology', 'Shared catalog infrastructure, not an operator family.']
]);
const analysisManifest = JSON.parse(fs.readFileSync('modules/analysis/package.json', 'utf8'));
const SOURCE_EXTENSION_PATTERN = /\.(?:[cm]?[jt]s|[jt]sx)$/;
const DECLARATION_SOURCE_PATTERN = /\.d\.[cm]?ts$/;
const SOURCE_SUFFIX_PATTERN = /\.(?:d\.[cm]?ts|[cm]?[jt]s|[jt]sx)$/;

function collectExportTargets(entry, target) {
  if (typeof target === 'string') return [target];
  if (target === null) return []; // An explicitly disabled export has no public source root.
  if (target && typeof target === 'object')
    return Object.values(target).flatMap(value => collectExportTargets(entry, value));
  throw new Error(`Cannot resolve root-file export ${entry} target ${JSON.stringify(target)}`);
}

// Root coverage is independent of the .ts-only traversal used by the other contract rules.
function listExportSourceFiles(directory) {
  return fs.readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    if (entry.name === 'node_modules' || entry.name === 'dist') return [];
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) return listExportSourceFiles(file);
    return entry.isFile() && SOURCE_EXTENSION_PATTERN.test(entry.name) ? [file] : [];
  });
}

function resolvePublicRootDirectories() {
  const directories = new Set();
  let sourceCandidates;
  for (const [entry, exported] of Object.entries(analysisManifest.exports)) {
    if (ROOT_FILE_EXCEPTIONS.has(entry)) continue;
    for (const target of collectExportTargets(entry, exported)) {
      if (
        !target.startsWith('./dist/') ||
        !SOURCE_SUFFIX_PATTERN.test(target) ||
        target
          .split('/')
          .slice(1)
          .some(segment => segment === '.' || segment === '..')
      ) {
        throw new Error(
          `Cannot resolve root-file export ${entry} target ${target}: unsupported source target`
        );
      }
      const sourceStem = target.slice('./dist/'.length).replace(SOURCE_SUFFIX_PATTERN, '');
      if (!sourceStem.includes('*')) {
        const directory = path.posix.dirname(sourceStem);
        directories.add(directory === '.' ? '' : directory);
        continue;
      }
      // Node export patterns replace '*' with a subpath, which can itself contain '/'.
      const [prefix, ...suffixes] = sourceStem
        .split('*')
        .map(part => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
      const pattern = new RegExp(`^${prefix}(?<subpath>.*)${suffixes.join('\\k<subpath>')}$`);
      sourceCandidates ??= listExportSourceFiles(SOURCE_ROOT);
      const matchingFiles = sourceCandidates.filter(file =>
        pattern.test(
          path
            .relative(SOURCE_ROOT, file)
            .split(path.sep)
            .join('/')
            .replace(SOURCE_SUFFIX_PATTERN, '')
        )
      );
      if (!matchingFiles.length) {
        throw new Error(
          `Cannot resolve root-file export ${entry} target ${target}: no matching source files`
        );
      }
      for (const file of matchingFiles) {
        const directory = path.relative(SOURCE_ROOT, path.dirname(file));
        directories.add(directory.split(path.sep).join('/'));
      }
    }
  }
  return directories;
}

let ROOT_FILE_DIRECTORIES;
try {
  ROOT_FILE_DIRECTORIES = resolvePublicRootDirectories();
} catch (error) {
  process.stderr.write(`${error.message}\n`);
  process.exit(2);
}
const ONTOLOGY_DIRECTORY = `${SOURCE_ROOT}/gpu-spatial-analysis/ontology`;

// Rules reported as one finding per file with an occurrence count. The rest are one per symbol.
const COUNT_RULES = new Set([
  'bare-assert',
  'single-invocation-data-loop',
  'shader-analysis-limit',
  'luma-reexport',
  'multiple-classes',
  'root-file'
]);
const RULE_IDENTIFIERS = [
  'bare-assert',
  'missing-producer-declaration',
  'single-invocation-data-loop',
  'duplicate-wgsl-declaration',
  'shader-analysis-limit',
  'internal-export',
  'luma-reexport',
  'multiple-classes',
  'root-file',
  'unregistered-contributor',
  'storage-binding-preflight'
];

// Generic baseline updates cannot excuse strict-rule regressions. Final reconciliation can retain
// explicitly reviewed pre-pass internal exports; their site, owner and revision must all match.
const STRICT_RULES = new Set([
  'bare-assert',
  'missing-producer-declaration',
  'internal-export',
  'luma-reexport',
  'unregistered-contributor',
  'duplicate-wgsl-declaration'
]);

const walk = directory =>
  fs.readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    if (entry.name === 'node_modules' || entry.name === 'dist') return [];
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) return walk(fullPath);
    return entry.name.endsWith('.ts') && !entry.name.endsWith('.d.ts') ? [fullPath] : [];
  });

const getLineNumber = (text, offset) => text.slice(0, offset).split('\n').length;

const isCommentLine = line => /^\s*(\/\/|\*|\/\*)/.test(line);

const findings = [];
const normalizeSite = text =>
  text
    .replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '')
    .match(/\w+|[^\s]/g)
    ?.join(' ') ?? '';
const hashSite = text => createHash('sha256').update(text).digest('hex');
const addFinding = (rule, file, details) => {
  const descriptions =
    details.siteDescriptions ??
    (rule === 'root-file'
      ? ['non-index family root file']
      : details.symbol
        ? [details.symbol]
        : details.lines.map(line => normalizeSite(fileTexts.get(file).split('\n')[line - 1])));
  const occurrences = new Map();
  const sites = descriptions.map(description => {
    const signature = hashSite(description);
    const occurrence = (occurrences.get(signature) ?? 0) + 1;
    occurrences.set(signature, occurrence);
    return {id: `${signature}:${occurrence}`, description};
  });
  findings.push({rule, file, ...details, sites});
};

const sourceFiles = walk(SOURCE_ROOT);
const fileTexts = new Map(sourceFiles.map(file => [file, fs.readFileSync(file, 'utf8')]));
const getTopDirectory = file => path.relative(SOURCE_ROOT, file).split(path.sep)[0];
const isCheckedFile = file => !SKIPPED_DIRECTORIES.has(getTopDirectory(file));
const checkedFiles = sourceFiles.filter(isCheckedFile);

// Inspect immediate implementations in each concrete public directory, even under an
// infrastructure subtree skipped by the other rules. Only exact entries are exceptions.
for (const directory of ROOT_FILE_DIRECTORIES) {
  const sourceDirectory = path.join(SOURCE_ROOT, directory);
  if (!fs.existsSync(sourceDirectory)) continue;
  for (const entry of fs.readdirSync(sourceDirectory, {withFileTypes: true})) {
    if (
      entry.isFile() &&
      SOURCE_EXTENSION_PATTERN.test(entry.name) &&
      !DECLARATION_SOURCE_PATTERN.test(entry.name) &&
      entry.name.replace(SOURCE_SUFFIX_PATTERN, '') !== 'index'
    ) {
      addFinding('root-file', path.join(sourceDirectory, entry.name), {count: 1, lines: [1]});
    }
  }
}

const sourceTrees = new Map(
  sourceFiles.map(file => [
    file,
    typeScript.createSourceFile(file, fileTexts.get(file), typeScript.ScriptTarget.Latest, true)
  ])
);

// Read the actual catalog construction inputs. Comments, descriptions, annotations and kind
// records are not registrations. These collections feed ALL_OPERATORS in the public catalog.
const registeredIdentities = new Set();
const SPATIAL_ENTRY_POINT = '@luma-contributors/analysis/gpu-spatial-analysis';
const catalogCollections = new Set([
  'GPU_SPATIAL_ANALYSIS_CAPABILITIES',
  'ENTRY_POINT_CAPABILITIES',
  'CAPABILITY_DEFINITIONS',
  'KEPLER_ANALYSIS_CAPABILITIES',
  'COVERAGE_EXPANSION_CAPABILITIES',
  'COVERAGE_FAMILIES',
  'SPATIOTEMPORAL_REVIEW_OPERATOR_OVERRIDES',
  'NESTED_VALIDATION_CAPABILITY',
  'REUSED_ANALYSIS_OPERATORS',
  'SPATIOTEMPORAL_REVIEW_OPERATORS',
  'FITTING_KERNEL_OPERATORS',
  'ALL_OPERATORS'
]);
const catalogTrees = [
  ...walk(ONTOLOGY_DIRECTORY),
  `${SOURCE_ROOT}/gpu-spatial-analysis/spatial-validation/spatial-validation-descriptors.ts`
]
  .filter(file => sourceTrees.has(file))
  .map(file => sourceTrees.get(file));
const visitCatalog = (node, visitor) => {
  visitor(node);
  typeScript.forEachChild(node, child => visitCatalog(child, visitor));
};
const unwrapCatalog = node => {
  while (
    node &&
    (typeScript.isParenthesizedExpression(node) ||
      typeScript.isAsExpression(node) ||
      typeScript.isSatisfiesExpression(node) ||
      typeScript.isNonNullExpression(node))
  )
    node = node.expression;
  return node;
};
const catalogPropertyName = name => {
  const key = typeScript.isComputedPropertyName(name) ? unwrapCatalog(name.expression) : name;
  return typeScript.isStringLiteralLike(key) ||
    typeScript.isNumericLiteral(key) ||
    (!typeScript.isComputedPropertyName(name) && typeScript.isIdentifier(key))
    ? key.text
    : undefined;
};
const catalogProperty = (object, name) =>
  object.properties.findLast(
    property =>
      typeScript.isPropertyAssignment(property) && catalogPropertyName(property.name) === name
  )?.initializer;
const catalogDeclarations = catalogTrees.map(tree => {
  const declarations = new Map();
  visitCatalog(tree, node => {
    if (
      typeScript.isVariableDeclaration(node) &&
      typeScript.isIdentifier(node.name) &&
      node.initializer
    )
      declarations.set(node.name.text, node.initializer);
  });
  const readText = (expression, visiting = new Set()) => {
    const node = unwrapCatalog(expression);
    if (!node) return undefined;
    if (typeScript.isStringLiteralLike(node)) return node.text;
    if (typeScript.isIdentifier(node) && !visiting.has(node.text))
      return readText(declarations.get(node.text), new Set([...visiting, node.text]));
    if (typeScript.isTemplateExpression(node)) {
      let text = node.head.text;
      for (const span of node.templateSpans) {
        const value = readText(span.expression, visiting);
        if (value === undefined) return undefined;
        text += value + span.literal.text;
      }
      return text;
    }
    return undefined;
  };
  return {declarations, readText};
});
// Details and overrides supply package identities, but cannot register an operator by themselves.
const entryPointPackages = new Map();
const spatialPackages = new Map();
for (const {declarations, readText} of catalogDeclarations) {
  for (const name of [
    'ENTRY_POINT_OPERATOR_DETAILS',
    'OPERATOR_OVERRIDES',
    'KEPLER_ANALYSIS_OPERATOR_OVERRIDES',
    'COVERAGE_EXPANSION_OPERATOR_OVERRIDES'
  ]) {
    const object = unwrapCatalog(declarations.get(name));
    if (!object || !typeScript.isObjectLiteralExpression(object)) continue;
    for (const property of object.properties) {
      if (!typeScript.isPropertyAssignment(property)) continue;
      const detail = unwrapCatalog(property.initializer);
      if (!typeScript.isObjectLiteralExpression(detail)) continue;
      const entryPoint = readText(catalogProperty(detail, 'package'));
      if (entryPoint)
        (name === 'ENTRY_POINT_OPERATOR_DETAILS' ? entryPointPackages : spatialPackages).set(
          property.name.getText().replace(/['"]/g, ''),
          entryPoint
        );
    }
  }
}
const clonedRegistrations = [];
for (const {declarations, readText} of catalogDeclarations) {
  for (const [collection, initializer] of declarations) {
    if (!catalogCollections.has(collection)) continue;
    const register = (name, entryPoint) => {
      if (name && entryPoint) registeredIdentities.add(`${entryPoint}:${name}`);
    };
    if (collection === 'SPATIOTEMPORAL_REVIEW_OPERATOR_OVERRIDES') {
      const object = unwrapCatalog(initializer);
      if (typeScript.isObjectLiteralExpression(object))
        for (const property of object.properties) {
          if (typeScript.isPropertyAssignment(property))
            register(property.name.getText().replace(/['"]/g, ''), SPATIAL_ENTRY_POINT);
        }
    }
    visitCatalog(initializer, node => {
      if (!typeScript.isObjectLiteralExpression(node)) return;
      if (collection === 'ALL_OPERATORS') {
        // A literal nested inside a record contributes fields, not a separate catalog entry.
        for (let parent = node.parent; parent && parent !== initializer; parent = parent.parent) {
          if (typeScript.isObjectLiteralExpression(parent)) return;
        }
        let exportName;
        let entryPoint;
        let sourceExportName;
        // Object spreads and assignments overwrite earlier identity fields in source order.
        const applyProperties = properties => {
          for (const property of properties) {
            if (typeScript.isSpreadAssignment(property)) {
              const spread = unwrapCatalog(property.expression);
              if (typeScript.isObjectLiteralExpression(spread)) {
                // A literal spread overwrites only the fields it actually contains.
                applyProperties(spread.properties);
                continue;
              }
              exportName = entryPoint = sourceExportName = undefined;
              const call = spread;
              if (
                !typeScript.isCallExpression(call) ||
                !typeScript.isPropertyAccessExpression(call.expression) ||
                call.expression.name.text !== 'find' ||
                call.expression.expression.getText() !== 'GPU_SPATIAL_ANALYSIS_OPERATORS'
              )
                continue;
              const predicate = unwrapCatalog(call.arguments[0]);
              const condition =
                predicate && typeScript.isArrowFunction(predicate)
                  ? unwrapCatalog(predicate.body)
                  : undefined;
              if (
                !condition ||
                !typeScript.isBinaryExpression(condition) ||
                condition.operatorToken.kind !== typeScript.SyntaxKind.EqualsEqualsEqualsToken ||
                !typeScript.isPropertyAccessExpression(condition.left) ||
                condition.left.name.text !== 'exportName'
              )
                continue;
              sourceExportName = readText(condition.right);
              exportName = sourceExportName;
              entryPoint = SPATIAL_ENTRY_POINT;
            } else if (property.name) {
              const key = typeScript.isComputedPropertyName(property.name)
                ? (readText(property.name.expression) ?? catalogPropertyName(property.name))
                : catalogPropertyName(property.name);
              const value = typeScript.isPropertyAssignment(property)
                ? readText(property.initializer)
                : typeScript.isShorthandPropertyAssignment(property)
                  ? readText(property.name)
                  : undefined;
              if (key === undefined) exportName = entryPoint = undefined;
              if (key === 'exportName') exportName = value;
              if (key === 'package') entryPoint = value;
            }
          }
        };
        applyProperties(node.properties);
        if (exportName && entryPoint) {
          if (sourceExportName)
            clonedRegistrations.push({sourceExportName, exportName, entryPoint});
          else register(exportName, entryPoint);
        }
        return;
      }
      const explicitPackage = readText(catalogProperty(node, 'package'));
      const defaultPackage =
        collection === 'FITTING_KERNEL_OPERATORS'
          ? '@luma-contributors/analysis/analysis-core'
          : SPATIAL_ENTRY_POINT;
      const name = readText(catalogProperty(node, 'exportName'));
      if (name) register(name, explicitPackage ?? defaultPackage);
      const operators = unwrapCatalog(catalogProperty(node, 'operators'));
      if (operators && typeScript.isArrayLiteralExpression(operators))
        for (const operator of operators.elements) {
          const exportName = readText(operator);
          register(
            exportName,
            explicitPackage ??
              (collection === 'ENTRY_POINT_CAPABILITIES'
                ? entryPointPackages.get(exportName)
                : (spatialPackages.get(exportName) ?? defaultPackage))
          );
        }
    });
  }
}
for (const {sourceExportName, exportName, entryPoint} of clonedRegistrations) {
  // A copy requires its source record, but registers only its final, possibly renamed identity.
  if (registeredIdentities.has(`${SPATIAL_ENTRY_POINT}:${sourceExportName}`))
    registeredIdentities.add(`${entryPoint}:${exportName}`);
}

// 4. internal-export: resolve export chains from each entry barrel by reading files.
const resolveModule = (fromFile, specifier) => {
  if (!specifier.startsWith('.')) return null;
  const base = path.join(path.dirname(fromFile), specifier);
  return (
    [`${base}.ts`, path.join(base, 'index.ts'), base].find(candidate => fileTexts.has(candidate)) ??
    null
  );
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
  const exported = new Map();
  for (const declaration of sourceTrees.get(file).statements) {
    if (
      declaration.modifiers?.some(modifier => modifier.kind === typeScript.SyntaxKind.ExportKeyword)
    ) {
      const declarations = typeScript.isVariableStatement(declaration)
        ? declaration.declarationList.declarations
        : [declaration];
      for (const entry of declarations) {
        if (entry.name && typeScript.isIdentifier(entry.name)) {
          exported.set(entry.name.text, `${file}#${entry.name.text}`);
        }
      }
    }
    if (!typeScript.isExportDeclaration(declaration)) continue;
    const specifier = declaration.moduleSpecifier?.text;
    const target = specifier ? resolveModule(file, specifier) : file;
    const targetExports = target && target !== file ? getExports(target, visiting) : exported;
    if (!declaration.exportClause) {
      if (target) for (const [name, origin] of targetExports) exported.set(name, origin);
    } else if (typeScript.isNamedExports(declaration.exportClause)) {
      for (const entry of declaration.exportClause.elements) {
        const originalName = (entry.propertyName ?? entry.name).text;
        exported.set(
          entry.name.text,
          targetExports.get(originalName) ?? `${target ?? specifier}#${originalName}`
        );
      }
    }
  }
  visiting.delete(file);
  exportCache.set(file, exported);
  return exported;
};
const barrelFiles = [
  `${SOURCE_ROOT}/index.ts`,
  ...fs
    .readdirSync(SOURCE_ROOT, {withFileTypes: true})
    .filter(entry => entry.isDirectory())
    .map(entry => `${SOURCE_ROOT}/${entry.name}/index.ts`)
]
  .filter(file => fileTexts.has(file))
  .sort();
// Public identities come from package entry points and resolved barrel exports, including aliases.
const publicIdentities = new Map();
const packageName = analysisManifest.name ?? '@luma-contributors/analysis';
for (const [entry, exported] of Object.entries(analysisManifest.exports)) {
  for (const target of collectExportTargets(entry, exported)) {
    if (!target.startsWith('./dist/') || !SOURCE_SUFFIX_PATTERN.test(target)) continue;
    const stem = target.slice('./dist/'.length).replace(SOURCE_SUFFIX_PATTERN, '');
    const pattern = new RegExp(
      `^${stem
        .split('*')
        .map(part => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
        .join('(.*)')}$`
    );
    for (const file of sourceFiles) {
      const match = pattern.exec(
        path.relative(SOURCE_ROOT, file).replace(SOURCE_SUFFIX_PATTERN, '')
      );
      if (!match) continue;
      const entryPoint =
        packageName + (entry === '.' ? '' : entry.slice(1).replaceAll('*', match[1] ?? ''));
      for (const [exportName, origin] of getExports(file)) {
        if (!publicIdentities.has(origin)) publicIdentities.set(origin, new Set());
        publicIdentities.get(origin).add(`${entryPoint}:${exportName}`);
      }
    }
  }
}
for (const barrel of barrelFiles) {
  for (const [exportedName, origin] of getExports(barrel)) {
    if (internalSymbols.has(origin)) {
      addFinding('internal-export', barrel, {symbol: exportedName, lines: [1], origin});
    }
  }
}

// Parse each source once. Bindings are resolved by lexical scope and import origin, rather than
// by class name across the whole repository. No type checker or dependency program is created.
const visitNodes = (node, callback) => {
  callback(node);
  typeScript.forEachChild(node, child => visitNodes(child, callback));
};
const scopeBindings = new WeakMap();
const getScopeBindings = scope => {
  if (scopeBindings.has(scope)) return scopeBindings.get(scope);
  const bindings = new Map();
  const addDeclaration = declaration => {
    if (declaration.name && typeScript.isIdentifier(declaration.name)) {
      bindings.set(declaration.name.text, declaration);
    } else if (
      declaration.name &&
      (typeScript.isObjectBindingPattern(declaration.name) ||
        typeScript.isArrayBindingPattern(declaration.name))
    ) {
      for (const element of declaration.name.elements) {
        if (typeScript.isBindingElement(element)) addDeclaration(element);
      }
    }
  };
  for (const parameter of scope.parameters ?? []) addDeclaration(parameter);
  for (const statement of scope.statements ?? []) {
    if (typeScript.isVariableStatement(statement)) {
      for (const declaration of statement.declarationList.declarations) addDeclaration(declaration);
    } else if (typeScript.isImportDeclaration(statement)) {
      const clause = statement.importClause;
      if (clause?.name) bindings.set(clause.name.text, clause);
      if (clause?.namedBindings && typeScript.isNamedImports(clause.namedBindings)) {
        for (const specifier of clause.namedBindings.elements) addDeclaration(specifier);
      }
    } else {
      addDeclaration(statement);
    }
  }
  scopeBindings.set(scope, bindings);
  return bindings;
};
const findBinding = identifier => {
  for (let scope = identifier.parent; scope; scope = scope.parent) {
    const binding = getScopeBindings(scope).get(identifier.text);
    if (binding) return binding;
  }
  return null;
};
// Shader analysis is synchronous. Resolve a local factory's parameters only for the current
// call site, then restore the previous context; callers must never share shader fragments.
let kernelParameterBindings = new Map();
const withKernelParameterBindings = (bindings, callback) => {
  const previous = kernelParameterBindings;
  kernelParameterBindings = bindings;
  try {
    return callback();
  } finally {
    kernelParameterBindings = previous;
  }
};
const isExpressionWrapper = expression =>
  typeScript.isParenthesizedExpression(expression) ||
  typeScript.isAsExpression(expression) ||
  typeScript.isSatisfiesExpression(expression) ||
  typeScript.isTypeAssertionExpression(expression) ||
  typeScript.isNonNullExpression(expression);
const resolveDeclaration = (expression, visiting = new Set()) => {
  if (!expression || visiting.has(expression)) return null;
  visiting.add(expression);
  if (typeScript.isParameter(expression) && kernelParameterBindings.has(expression)) {
    return resolveDeclaration(kernelParameterBindings.get(expression), visiting);
  }
  if (typeScript.isIdentifier(expression)) {
    const binding = findBinding(expression);
    return binding ? resolveDeclaration(binding, visiting) : null;
  }
  if (typeScript.isImportSpecifier(expression)) {
    const declaration = expression.parent.parent.parent;
    const target = resolveModule(
      expression.getSourceFile().fileName,
      declaration.moduleSpecifier.text
    );
    if (!target) return expression;
    const name = (expression.propertyName ?? expression.name).text;
    const origin = getExports(target).get(name);
    if (!origin) return expression;
    const separator = origin.lastIndexOf('#');
    const source = sourceTrees.get(origin.slice(0, separator));
    const binding = source && getScopeBindings(source).get(origin.slice(separator + 1));
    return binding ? resolveDeclaration(binding, visiting) : expression;
  }
  if (typeScript.isVariableDeclaration(expression) && expression.initializer) {
    return resolveDeclaration(expression.initializer, visiting);
  }
  if (isExpressionWrapper(expression)) return resolveDeclaration(expression.expression, visiting);
  return expression;
};
// Internal members also leak through exported props aliases and visible class signatures.
// Resolve each declaration origin once, even when several package barrels re-export it.
const checkedPublicOrigins = new Set();
const reportedPublicSites = new Set();
const resolvePublicType = reference => {
  if (typeScript.isQualifiedName(reference) || typeScript.isPropertyAccessExpression(reference)) {
    const owner = typeScript.isQualifiedName(reference) ? reference.left : reference.expression;
    const member = typeScript.isQualifiedName(reference) ? reference.right : reference.name;
    if (typeScript.isIdentifier(owner)) {
      const source = owner.getSourceFile();
      const imported = source.statements.find(
        statement =>
          typeScript.isImportDeclaration(statement) &&
          statement.importClause?.namedBindings &&
          typeScript.isNamespaceImport(statement.importClause.namedBindings) &&
          statement.importClause.namedBindings.name.text === owner.text
      );
      const target = imported && resolveModule(source.fileName, imported.moduleSpecifier.text);
      const origin = target && getExports(target).get(member.text);
      if (origin) {
        const separator = origin.lastIndexOf('#');
        const targetSource = sourceTrees.get(origin.slice(0, separator));
        return targetSource && getScopeBindings(targetSource).get(origin.slice(separator + 1));
      }
    }
  }
  if (typeScript.isIdentifier(reference)) {
    for (let scope = reference.parent; scope; scope = scope.parent) {
      const parameter = scope.typeParameters?.find(
        parameter => parameter.name.text === reference.text
      );
      if (parameter) return parameter;
    }
  }
  return resolveDeclaration(reference);
};
for (const barrel of barrelFiles) {
  for (const [publicName, origin] of getExports(barrel)) {
    if (checkedPublicOrigins.has(origin)) continue;
    const separator = origin.lastIndexOf('#');
    const source = sourceTrees.get(origin.slice(0, separator));
    const declaration = source && getScopeBindings(source).get(origin.slice(separator + 1));
    if (
      !declaration ||
      !(
        typeScript.isClassDeclaration(declaration) ||
        ((/Props$/.test(publicName) || /Props$/.test(declaration.name?.text ?? '')) &&
          (typeScript.isTypeAliasDeclaration(declaration) ||
            typeScript.isInterfaceDeclaration(declaration)))
      )
    )
      continue;
    checkedPublicOrigins.add(origin);
    for (const {node, name} of findInternalPublicMembers(
      declaration,
      resolvePublicType,
      publicName
    )) {
      const file = node.getSourceFile().fileName;
      const site = `${file}:${node.getStart()}`;
      if (reportedPublicSites.has(site)) continue;
      reportedPublicSites.add(site);
      addFinding('internal-export', file, {
        symbol: name,
        lines: [getLineNumber(fileTexts.get(file), node.getStart())],
        origin
      });
    }
  }
}

const isNamedReference = (expression, name) => {
  const declaration = resolveDeclaration(expression);
  if (declaration && typeScript.isImportSpecifier(declaration)) {
    return (declaration.propertyName ?? declaration.name).text === name;
  }
  return (
    (typeScript.isIdentifier(expression) && expression.text === name) ||
    (declaration &&
      typeScript.isFunctionDeclaration(declaration) &&
      declaration.name?.text === name)
  );
};
const getClassMembers = (declaration, visiting = new Set()) => {
  if (!declaration || !typeScript.isClassDeclaration(declaration) || visiting.has(declaration))
    return [];
  visiting.add(declaration);
  const parent = declaration.heritageClauses?.find(
    clause => clause.token === typeScript.SyntaxKind.ExtendsKeyword
  )?.types[0]?.expression;
  return [...declaration.members, ...getClassMembers(resolveDeclaration(parent), visiting)];
};
const isProducerClass = (declaration, visiting = new Set()) => {
  if (!declaration || !typeScript.isClassDeclaration(declaration) || visiting.has(declaration))
    return false;
  visiting.add(declaration);
  return (declaration.heritageClauses ?? []).some(clause =>
    clause.types.some(
      type =>
        isNamedReference(type.expression, 'GPUCommandNodeProducer') ||
        (clause.token === typeScript.SyntaxKind.ExtendsKeyword &&
          isProducerClass(resolveDeclaration(type.expression), visiting))
    )
  );
};

// Keep unknown fragments inside tokens: whitespace would turn `helper${suffix}` into a
// different, falsely certain identifier. Conditional strings remain separate shader variants.
const UNKNOWN_WGSL_TEXT = '\u0000';
// Bound the materialized text/choice combinations of one shader property. Larger builders
// need an explicit finding rather than unbounded memory use or silently skipped analysis.
const MAXIMUM_SHADER_VARIANT_COUNT = 256;
const SHADER_ANALYSIS_LIMIT = Symbol('shader-analysis-limit');
const createTextVariant = text => ({text, choices: new Map()});
// Collapse equal text only when the two paths cover both values of one predicate. This
// removes redundant unknown interpolations without dropping correlations between branches.
const normalizeTextVariants = variants => {
  const result = [];
  for (const variant of variants) {
    let candidate = variant;
    for (let index = 0; index < result.length; index++) {
      const previous = result[index];
      if (previous.text !== candidate.text) continue;
      const keys = new Set([...previous.choices.keys(), ...candidate.choices.keys()]);
      const different = [...keys].filter(
        key => previous.choices.get(key) !== candidate.choices.get(key)
      );
      if (!different.length) {
        candidate = null;
        break;
      }
      if (different.length === 1) {
        const key = different[0];
        const choices = new Map(previous.choices);
        choices.delete(key);
        candidate = {...candidate, choices};
        result.splice(index, 1);
        index = -1;
      }
    }
    if (candidate) result.push(candidate);
  }
  return result;
};
const combineTextVariants = (leftVariants, rightVariants, append = true) => {
  const result = [];
  for (const left of leftVariants) {
    for (const right of rightVariants) {
      if (
        [...right.choices].some(
          ([condition, value]) =>
            left.choices.has(condition) && left.choices.get(condition) !== value
        )
      )
        continue;
      if (result.length >= MAXIMUM_SHADER_VARIANT_COUNT) throw SHADER_ANALYSIS_LIMIT;
      result.push({
        text: append ? left.text + right.text : right.text,
        choices: new Map([...left.choices, ...right.choices])
      });
    }
  }
  return normalizeTextVariants(result);
};
const UNKNOWN_STATIC_VALUE = Symbol('unknown-static-value');
const getStaticValue = (expression, visiting = new Set()) => {
  if (!expression || visiting.has(expression)) return UNKNOWN_STATIC_VALUE;
  const nextVisiting = new Set(visiting).add(expression);
  const evaluate = node => getStaticValue(node, nextVisiting);
  if (typeScript.isParameter(expression) && kernelParameterBindings.has(expression)) {
    const argument = kernelParameterBindings.get(expression);
    return argument ? evaluate(argument) : undefined;
  }
  if (expression.kind === typeScript.SyntaxKind.TrueKeyword) return true;
  if (expression.kind === typeScript.SyntaxKind.FalseKeyword) return false;
  if (expression.kind === typeScript.SyntaxKind.NullKeyword) return null;
  if (typeScript.isStringLiteralLike(expression)) return expression.text;
  if (typeScript.isNumericLiteral(expression)) return Number(expression.text);
  if (isExpressionWrapper(expression)) return evaluate(expression.expression);
  if (
    typeScript.isPrefixUnaryExpression(expression) &&
    expression.operator === typeScript.SyntaxKind.ExclamationToken
  ) {
    const value = evaluate(expression.operand);
    return value === UNKNOWN_STATIC_VALUE ? value : !value;
  }
  if (typeScript.isIdentifier(expression)) {
    const binding = findBinding(expression);
    if (!binding && expression.text === 'undefined') return undefined;
    return evaluate(binding);
  }
  if (
    typeScript.isVariableDeclaration(expression) &&
    expression.parent.flags & typeScript.NodeFlags.Const
  )
    return evaluate(expression.initializer);
  return UNKNOWN_STATIC_VALUE;
};
const getStaticBoolean = expression => {
  const value = getStaticValue(expression);
  return value === UNKNOWN_STATIC_VALUE ? undefined : Boolean(value);
};
const getStaticNullish = expression => {
  const value = getStaticValue(expression);
  return value === UNKNOWN_STATIC_VALUE ? undefined : value === null || value === undefined;
};
const stableConditionBindings = new WeakMap();
const getStableConditionText = (expression, visiting = new Set()) => {
  if (!expression || visiting.has(expression)) return null;
  const nextVisiting = new Set(visiting).add(expression);
  const expand = node => getStableConditionText(node, nextVisiting);
  if (typeScript.isParenthesizedExpression(expression)) return expand(expression.expression);
  if (typeScript.isStringLiteralLike(expression) || typeScript.isNumericLiteral(expression)) {
    return `${expression.kind}:${expression.text}`;
  }
  if (typeScript.isPropertyAccessExpression(expression)) {
    const owner = expand(expression.expression);
    return owner === null ? null : `${owner}.${expression.name.text}`;
  }
  if (
    typeScript.isBinaryExpression(expression) &&
    expression.operatorToken.kind < typeScript.SyntaxKind.FirstAssignment
  ) {
    const left = expand(expression.left);
    const right = expand(expression.right);
    return left === null || right === null
      ? null
      : `(${left} ${expression.operatorToken.kind} ${right})`;
  }
  if (!typeScript.isIdentifier(expression)) return null;
  const binding = findBinding(expression);
  if (!binding) return null;
  if (kernelParameterBindings.has(binding)) {
    const argument = kernelParameterBindings.get(binding);
    return argument ? expand(argument) : null;
  }
  if (!stableConditionBindings.has(binding)) {
    let scope = binding.parent;
    while (scope.parent && !typeScript.isFunctionLike(scope) && !typeScript.isSourceFile(scope)) {
      scope = scope.parent;
    }
    let stable = true;
    visitNodes(scope, node => {
      const target =
        typeScript.isBinaryExpression(node) &&
        node.operatorToken.kind >= typeScript.SyntaxKind.FirstAssignment &&
        node.operatorToken.kind <= typeScript.SyntaxKind.LastAssignment
          ? node.left
          : (typeScript.isPrefixUnaryExpression(node) ||
                typeScript.isPostfixUnaryExpression(node)) &&
              [typeScript.SyntaxKind.PlusPlusToken, typeScript.SyntaxKind.MinusMinusToken].includes(
                node.operator
              )
            ? node.operand
            : null;
      let root = target;
      while (
        root &&
        (typeScript.isPropertyAccessExpression(root) || typeScript.isElementAccessExpression(root))
      )
        root = root.expression;
      if (root && typeScript.isIdentifier(root) && findBinding(root) === binding) stable = false;
    });
    stableConditionBindings.set(binding, stable);
  }
  // Repeated reads of an unchanged parameter/local share a branch choice. This is essential
  // for nested templates to describe two shaders rather than an exponential cross product.
  return stableConditionBindings.get(binding)
    ? `${binding.getSourceFile().fileName}:${binding.pos}`
    : null;
};
const getConditionKey = expression => getStableConditionText(expression) ?? expression;
const nullishConditionKeys = new WeakMap();
const constrainTextVariants = (variants, condition, value, kind = 'truthy') => {
  if (isExpressionWrapper(condition)) {
    return constrainTextVariants(variants, condition.expression, value, kind);
  }
  if (
    kind === 'truthy' &&
    typeScript.isPrefixUnaryExpression(condition) &&
    condition.operator === typeScript.SyntaxKind.ExclamationToken
  ) {
    return constrainTextVariants(variants, condition.operand, !value);
  }
  const knownValue = kind === 'nullish' ? getStaticNullish(condition) : getStaticBoolean(condition);
  if (knownValue !== undefined) return knownValue === value ? variants : [];
  // A conjunction/disjunction's truthiness constrains its operands, not a separate unrelated key.
  if (kind === 'truthy' && typeScript.isBinaryExpression(condition)) {
    const operator = condition.operatorToken.kind;
    if (
      operator === typeScript.SyntaxKind.AmpersandAmpersandToken ||
      operator === typeScript.SyntaxKind.BarBarToken
    ) {
      const required = operator === typeScript.SyntaxKind.AmpersandAmpersandToken;
      const leftMatches = constrainTextVariants(variants, condition.left, required);
      const both = constrainTextVariants(leftMatches, condition.right, value);
      return value === required
        ? both
        : normalizeTextVariants([
            ...constrainTextVariants(variants, condition.left, !required),
            ...both
          ]);
    }
  }
  let key = getConditionKey(condition);
  if (kind === 'nullish') {
    if (typeof key === 'string') key = `nullish:${key}`;
    else {
      if (!nullishConditionKeys.has(key)) nullishConditionKeys.set(key, {});
      key = nullishConditionKeys.get(key);
    }
  }
  let result = variants
    .filter(variant => !variant.choices.has(key) || variant.choices.get(key) === value)
    .map(variant => ({text: variant.text, choices: new Map([...variant.choices, [key, value]])}));
  // Null/undefined are falsy, and truthy values cannot be nullish.
  if (kind === 'nullish' && value) result = constrainTextVariants(result, condition, false);
  if (kind === 'truthy' && value)
    result = constrainTextVariants(result, condition, false, 'nullish');
  return result;
};

// Share path discovery between writes and kernel calls. Walking the AST does not imply that
// an if/ternary branch or the right operand of a logical expression actually executes.
const getExecutionGuards = (node, stop = null) => {
  const guards = [];
  let unsupported = false;
  let nestedFunction = false;
  for (
    let child = node, parent = child.parent;
    parent && parent !== stop;
    child = parent, parent = parent.parent
  ) {
    if (typeScript.isFunctionLike(parent)) {
      nestedFunction = true;
      break;
    }
    if (typeScript.isIfStatement(parent)) {
      if (child === parent.thenStatement || child === parent.elseStatement) {
        guards.unshift([parent.expression, child === parent.thenStatement]);
      }
    } else if (typeScript.isConditionalExpression(parent)) {
      if (child !== parent.condition) guards.unshift([parent.condition, child === parent.whenTrue]);
    } else if (typeScript.isBinaryExpression(parent) && child === parent.right) {
      if (parent.operatorToken.kind === typeScript.SyntaxKind.AmpersandAmpersandToken) {
        guards.unshift([parent.left, true]);
      } else if (parent.operatorToken.kind === typeScript.SyntaxKind.BarBarToken) {
        guards.unshift([parent.left, false]);
      } else if (parent.operatorToken.kind === typeScript.SyntaxKind.QuestionQuestionToken) {
        guards.unshift([parent.left, true, 'nullish']);
      }
    } else if (
      typeScript.isIterationStatement(parent, false) ||
      typeScript.isSwitchStatement(parent)
    ) {
      unsupported = true;
    }
  }
  return {guards, unsupported, nestedFunction};
};
const applyExecutionGuards = (variants, guards) =>
  guards.reduce(
    (result, [condition, value, kind]) => constrainTextVariants(result, condition, value, kind),
    variants
  );

// Apply = and += in source order up to the identifier's use. Branch guards keep then/else
// alternatives separate, and aliases observe the string at their initializer, not a later write.
const getStaticTextVariants = (expression, visiting = new Set()) => {
  if (!expression || visiting.has(expression)) return [createTextVariant(UNKNOWN_WGSL_TEXT)];
  const nextVisiting = new Set(visiting).add(expression);
  const expand = node => getStaticTextVariants(node, nextVisiting);
  if (typeScript.isParameter(expression) && kernelParameterBindings.has(expression)) {
    return expand(kernelParameterBindings.get(expression));
  }
  if (typeScript.isStringLiteralLike(expression) || typeScript.isNumericLiteral(expression)) {
    return [createTextVariant(expression.text)];
  }
  if (isExpressionWrapper(expression)) return expand(expression.expression);
  if (typeScript.isTemplateExpression(expression)) {
    let variants = [createTextVariant(expression.head.text)];
    for (const span of expression.templateSpans) {
      variants = combineTextVariants(variants, expand(span.expression)).map(variant => ({
        ...variant,
        text: variant.text + span.literal.text
      }));
    }
    return variants;
  }
  if (typeScript.isConditionalExpression(expression)) {
    const knownValue = getStaticBoolean(expression.condition);
    if (knownValue !== undefined)
      return expand(knownValue ? expression.whenTrue : expression.whenFalse);
    return normalizeTextVariants([
      ...constrainTextVariants(expand(expression.whenTrue), expression.condition, true),
      ...constrainTextVariants(expand(expression.whenFalse), expression.condition, false)
    ]);
  }
  if (
    typeScript.isBinaryExpression(expression) &&
    expression.operatorToken.kind === typeScript.SyntaxKind.PlusToken
  ) {
    return combineTextVariants(expand(expression.left), expand(expression.right));
  }
  if (typeScript.isIdentifier(expression)) {
    const binding = findBinding(expression);
    let variants = expand(binding);
    if (binding && typeScript.isVariableDeclaration(binding)) {
      const scope = binding.parent.parent.parent;
      visitNodes(scope, mutation => {
        if (
          !typeScript.isBinaryExpression(mutation) ||
          ![typeScript.SyntaxKind.EqualsToken, typeScript.SyntaxKind.PlusEqualsToken].includes(
            mutation.operatorToken.kind
          ) ||
          !typeScript.isIdentifier(mutation.left) ||
          findBinding(mutation.left) !== binding ||
          mutation.end >= expression.getStart() ||
          mutation.getStart() < binding.end
        )
          return;
        const {guards, unsupported, nestedFunction} = getExecutionGuards(mutation, scope);
        if (nestedFunction) return;
        if (unsupported) {
          // Unknown control flow may write the string, but cannot establish that prior text
          // disappeared. Retain the unchanged path as well as the opaque possibility.
          variants = normalizeTextVariants([...variants, createTextVariant(UNKNOWN_WGSL_TEXT)]);
          return;
        }
        let matching = variants;
        const unchanged = [];
        for (const [condition, value, kind] of guards) {
          unchanged.push(...constrainTextVariants(matching, condition, !value, kind));
          matching = constrainTextVariants(matching, condition, value, kind);
        }
        variants = normalizeTextVariants([
          ...unchanged,
          ...combineTextVariants(
            matching,
            expand(mutation.right),
            mutation.operatorToken.kind === typeScript.SyntaxKind.PlusEqualsToken
          )
        ]);
      });
    }
    return variants;
  }
  if (typeScript.isVariableDeclaration(expression)) return expand(expression.initializer);
  if (typeScript.isImportSpecifier(expression)) {
    const declaration = resolveDeclaration(expression);
    if (declaration !== expression) return expand(declaration);
  }
  return [createTextVariant(UNKNOWN_WGSL_TEXT)];
};
// Numeric dispatch sizes must be fully known. Partial text such as `rowCount + 1` must never
// collapse to the trailing literal and be mistaken for a one-invocation dispatch.
const getStaticNumber = (expression, visiting = new Set()) => {
  if (!expression || visiting.has(expression)) return undefined;
  const nextVisiting = new Set(visiting).add(expression);
  const evaluate = node => getStaticNumber(node, nextVisiting);
  if (typeScript.isParameter(expression) && kernelParameterBindings.has(expression)) {
    return evaluate(kernelParameterBindings.get(expression));
  }
  if (typeScript.isNumericLiteral(expression)) return Number(expression.text);
  if (isExpressionWrapper(expression)) return evaluate(expression.expression);
  if (typeScript.isIdentifier(expression)) return evaluate(findBinding(expression));
  if (
    typeScript.isVariableDeclaration(expression) &&
    expression.parent.flags & typeScript.NodeFlags.Const
  ) {
    return evaluate(expression.initializer);
  }
  if (typeScript.isImportSpecifier(expression)) return evaluate(resolveDeclaration(expression));
  if (typeScript.isBinaryExpression(expression)) {
    const left = evaluate(expression.left);
    const right = evaluate(expression.right);
    if (left === undefined || right === undefined) return undefined;
    switch (expression.operatorToken.kind) {
      case typeScript.SyntaxKind.PlusToken:
        return left + right;
      case typeScript.SyntaxKind.MinusToken:
        return left - right;
      case typeScript.SyntaxKind.AsteriskToken:
        return left * right;
      case typeScript.SyntaxKind.SlashToken:
        return left / right;
      default:
        return undefined;
    }
  }
  return undefined;
};
const getProperty = (object, name) =>
  object?.properties?.find(
    property =>
      property.name &&
      (typeScript.isIdentifier(property.name) || typeScript.isStringLiteral(property.name)) &&
      property.name.text === name
  );
const getPropertyValue = property =>
  property &&
  (typeScript.isShorthandPropertyAssignment(property) ? property.name : property.initializer);

// One local function/arrow layer, including aliases, generic calls, defaults, and lexical
// captures. This deliberately does not attempt arbitrary interprocedural TypeScript execution.
const getEnclosingFunction = node => {
  for (let parent = node.parent; parent; parent = parent.parent) {
    if (typeScript.isFunctionLike(parent)) return parent;
  }
  return null;
};
const getKernelCallSites = sourceTree => {
  const calls = [];
  visitNodes(sourceTree, node => {
    if (typeScript.isCallExpression(node)) calls.push(node);
  });
  const kernelCalls = calls.filter(call =>
    isNamedReference(call.expression, 'createWGSLKernelNode')
  );
  const wrappers = new Map();
  for (const call of kernelCalls) {
    const owner = getEnclosingFunction(call);
    if (!owner) continue;
    const properties = resolveDeclaration(call.arguments[1]);
    if (!properties || !typeScript.isObjectLiteralExpression(properties)) continue;
    const count = resolveDeclaration(getPropertyValue(getProperty(properties, 'invocationCount')));
    const body = resolveDeclaration(getPropertyValue(getProperty(properties, 'body')));
    // Recognize a forwarding wrapper, not every larger contributor factory that happens
    // to contain a kernel. Keeping count and body tied to the same owner avoids expansion
    // of unrelated graph builders and duplicate reports for their fixed kernels.
    if (!owner.parameters.includes(count) || !owner.parameters.includes(body)) continue;
    const owned = wrappers.get(owner) ?? [];
    owned.push(call);
    wrappers.set(owner, owned);
  }
  const expandedCalls = new Set();
  const sites = [];
  for (const callSite of calls) {
    if (callSite.arguments.some(typeScript.isSpreadElement)) continue;
    const owner = resolveDeclaration(callSite.expression);
    const owned = wrappers.get(owner);
    if (!owned || owner.getSourceFile() !== sourceTree) continue;
    if (getEnclosingFunction(callSite) === owner) continue;
    if (getStableConditionText(callSite.expression) === null) continue;
    if (
      owner.parameters.some(
        parameter => !typeScript.isIdentifier(parameter.name) || parameter.dotDotDotToken
      )
    )
      continue;
    const parameterBindings = new Map();
    for (const [index, parameter] of owner.parameters.entries()) {
      // A parameter overwritten inside the factory cannot be replaced by its incoming value.
      if (getStableConditionText(parameter.name) === null) continue;
      const argument = callSite.arguments[index];
      const omitted =
        !argument ||
        (typeScript.isIdentifier(argument) &&
          argument.text === 'undefined' &&
          !findBinding(argument));
      parameterBindings.set(parameter, omitted ? parameter.initializer : argument);
    }
    for (const call of owned) {
      expandedCalls.add(call);
      sites.push({call, callSite, parameterBindings});
    }
  }
  for (const call of kernelCalls) {
    if (!expandedCalls.has(call)) {
      sites.push({call, callSite: call, parameterBindings: new Map()});
    }
  }
  return sites;
};

// WGSL has nested block comments. Keep offsets/newlines intact for scans and diagnostics.
const maskWGSLComments = text => {
  let commentDepth = 0;
  let lineComment = false;
  let result = '';
  for (let offset = 0; offset < text.length; offset++) {
    const character = text[offset];
    const pair = text.slice(offset, offset + 2);
    if (lineComment) {
      if (character === '\n' || character === '\r') lineComment = false;
      result += lineComment ? ' ' : character;
    } else if (commentDepth && pair === '*/') {
      commentDepth--;
      result += '  ';
      offset++;
    } else if (pair === '/*') {
      commentDepth++;
      result += '  ';
      offset++;
    } else if (!commentDepth && pair === '//') {
      lineComment = true;
      result += '  ';
      offset++;
    } else {
      result += commentDepth && character !== '\n' && character !== '\r' ? ' ' : character;
    }
  }
  return result;
};
// A shared declarations string often includes bounded helpers the kernel never calls. Follow
// the call graph from the body so an unused helper's loop does not create a serial-work finding.
const getReachableWGSLText = (declarations, body, includeReferencedDeclarations = false) => {
  const cleanDeclarations = maskWGSLComments(declarations);
  const functions = new Map();
  const moduleDeclarations = new Map();
  if (includeReferencedDeclarations) {
    let depth = 0;
    let previousEnd = 0;
    for (const match of cleanDeclarations.matchAll(
      /\b(?:const|override|alias)\s+([A-Za-z_]\w*)[^;]*;/g
    )) {
      for (const character of cleanDeclarations.slice(previousEnd, match.index)) {
        if (character === '{') depth++;
        if (character === '}') depth--;
      }
      previousEnd = match.index;
      if (depth === 0) moduleDeclarations.set(match[1], match[0]);
    }
  }
  for (const match of cleanDeclarations.matchAll(/(?<![\w\u0000])fn\s+([A-Za-z_][\w]*)\s*\(/g)) {
    const start = cleanDeclarations.indexOf('{', match.index);
    if (start < 0) continue;
    let depth = 1;
    let end = start + 1;
    for (; end < cleanDeclarations.length && depth; end++) {
      if (cleanDeclarations[end] === '{') depth++;
      if (cleanDeclarations[end] === '}') depth--;
    }
    functions.set(match[1], cleanDeclarations.slice(start, end));
  }
  const pieces = [maskWGSLComments(body)];
  const visited = new Set();
  for (let index = 0; index < pieces.length; index++) {
    if (includeReferencedDeclarations) {
      for (const match of pieces[index].matchAll(/\b[A-Za-z_]\w*\b/g)) {
        const key = `declaration:${match[0]}`;
        if (moduleDeclarations.has(match[0]) && !visited.has(key)) {
          visited.add(key);
          pieces.push(moduleDeclarations.get(match[0]));
        }
      }
    }
    for (const match of pieces[index].matchAll(/(?<![\w\u0000])([A-Za-z_][\w]*)\s*\(/g)) {
      if (functions.has(match[1]) && !visited.has(match[1])) {
        visited.add(match[1]);
        pieces.push(functions.get(match[1]));
      }
    }
  }
  return pieces.join('\n');
};
const getDuplicateWGSLNames = text => {
  const scopes = [{names: new Set(), kind: 'block'}];
  const duplicates = new Set();
  const listOpeners = new Map();
  const bodyHeaderNames = new Map();
  // Include punctuation and opaque fragments so unknown name pieces taint the whole token.
  const tokens = maskWGSLComments(text).match(/[\w\u0000]+|[^\s]/g) ?? [];
  const registerName = name => {
    if (!/^[A-Za-z_][\w]*$/.test(name ?? '')) return;
    const scope = scopes.at(-1);
    if (scope?.names.has(name)) duplicates.add(name);
    scope?.names.add(name);
  };
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index];
    if (token === '{' || token === '(') {
      const kind =
        token === '(' && tokens[index - 1] === 'for'
          ? 'for-header'
          : (listOpeners.get(index) ?? 'block');
      scopes.push({names: new Set(bodyHeaderNames.get(index)), kind});
    } else if (token === '}' || token === ')') {
      const scope = scopes.pop();
      if (token === ')' && ['parameters', 'for-header'].includes(scope?.kind)) {
        const bodyIndex = tokens.indexOf('{', index + 1);
        if (bodyIndex >= 0) bodyHeaderNames.set(bodyIndex, scope.names);
      }
    } else if (token === ':' && ['parameters', 'members'].includes(scopes.at(-1)?.kind)) {
      // Attributes have their own parentheses; generic type commas cannot declare a name.
      registerName(tokens[index - 1]);
    } else if (/^(fn|struct|alias|const|override|let|var)$/.test(token)) {
      if (tokens[index + 1] === '<') {
        while (++index < tokens.length && tokens[index] !== '>') {
          /* address space */
        }
      }
      const name = tokens[++index];
      if (token === 'fn' && tokens[index + 1] === '(') listOpeners.set(index + 1, 'parameters');
      if (token === 'struct' && tokens[index + 1] === '{') listOpeners.set(index + 1, 'members');
      registerName(name);
    }
  }
  return duplicates;
};

// 11. storage-binding-preflight: a conservative, syntax-only declaration-order audit.
// Follow local/imported helpers and this-methods, but never execute deferred callbacks. A check
// must dominate declarations on every syntactic path. This checks presence/order, not whether
// every scratch formula or aligned view range was covered; reviewers still verify those sizes.
const STORAGE_PREFLIGHT_HELPERS = new Set([
  'validateGPUStorageBindingByteLength',
  'validateGraphChunkBindingRanges',
  'validateGraphChunkDispatchBindings'
]);
const STORAGE_DECLARATION_HELPERS = new Set([
  'createTransientView',
  'createTransientVectorView',
  'createWGSLKernelNode',
  'createGraphChunkGroupKernelNode',
  'createGPUComputeCommandNode',
  'createGPUCopyCommandNode'
]);
const getReferenceName = expression => {
  const declaration = resolveDeclaration(expression);
  return typeScript.isImportSpecifier(declaration ?? expression)
    ? (declaration.propertyName ?? declaration.name).text
    : (declaration?.name?.text ?? expression.name?.text ?? expression.text);
};
// Named-capacity wrappers are preflights too when their accepted maximum derives from the
// device storage binding. Follow local/imported returns and const aliases, without typechecking.
const referencesStorageBindingLimit = (node, visiting = new Set()) => {
  if (!node || visiting.has(node)) return false;
  const nextVisiting = new Set(visiting).add(node);
  if (
    typeScript.isPropertyAccessExpression(node) &&
    node.name.text === 'maxStorageBufferBindingSize'
  )
    return true;
  if (
    typeScript.isCallExpression(node) &&
    ['getStorageBindingElementLimit', 'getAnalysisMaximumRowCount'].includes(
      getReferenceName(node.expression)
    )
  )
    return true;
  const resolved = resolveDeclaration(node);
  if (resolved && resolved !== node && referencesStorageBindingLimit(resolved, nextVisiting))
    return true;
  if (typeScript.isCallExpression(node)) {
    const helper = resolveDeclaration(node.expression);
    if (helper?.body && referencesStorageBindingLimit(helper.body, nextVisiting)) return true;
  }
  let found = false;
  typeScript.forEachChild(node, child => {
    found ||= referencesStorageBindingLimit(child, nextVisiting);
  });
  return found;
};
const isStorageCapacityCheck = call => {
  const name = getReferenceName(call.expression);
  const limit =
    name === 'validateCapacity'
      ? call.arguments[4]
      : name === 'validateMaximumCount'
        ? call.arguments[3]
        : name === 'validateRequirement'
          ? call.arguments[1]
          : null;
  return referencesStorageBindingLimit(limit);
};

// A loop that checks the aligned range of every present view is a collection preflight even
// for an empty collection. Optional-view guards are allowed; unrelated conditional checks and
// any declarations inside the loop are not accepted as a summary.
const isCollectionBindingPreflight = loop => {
  if (!typeScript.isForOfStatement(loop)) return false;
  const names = new Set();
  visitNodes(loop.initializer, node => {
    if (typeScript.isIdentifier(node)) names.add(node.text);
  });
  let foundRangeCheck = false;
  let safe = true;
  const inspect = node => {
    if (typeScript.isFunctionLike(node)) return;
    if (typeScript.isIfStatement(node)) {
      let condition = node.expression;
      while (isExpressionWrapper(condition)) condition = condition.expression;
      if (
        typeScript.isPrefixUnaryExpression(condition) &&
        condition.operator === typeScript.SyntaxKind.ExclamationToken
      ) {
        condition = condition.operand;
      }
      if (!typeScript.isIdentifier(condition) || !names.has(condition.text)) safe = false;
    }
    if (typeScript.isCallExpression(node)) {
      const name = getReferenceName(node.expression);
      if (
        STORAGE_DECLARATION_HELPERS.has(name) ||
        /^(createTransientBuffer|createBuffer|createDataView|add)$/.test(name)
      )
        safe = false;
      if (name === 'validateGPUStorageBindingByteLength') {
        visitNodes(node.arguments[3], value => {
          if (
            typeScript.isCallExpression(value) &&
            getReferenceName(value.expression) === 'getViewBindingRange'
          )
            foundRangeCheck = true;
        });
      }
    }
    typeScript.forEachChild(node, child => {
      inspect(child);
    });
  };
  inspect(loop.statement);
  return safe && foundRangeCheck;
};

const getStoragePreflightFinding = declaration => {
  const members = getClassMembers(declaration);
  const entry = members.find(
    member => member.name?.getText().replace(/['"]/g, '') === 'getCommandNodes'
  );
  const entryBody = entry?.body ?? entry?.initializer?.body;
  if (!entryBody) return null;
  let hasRowStorage = false;
  let firstUncheckedDeclaration = null;
  const visiting = new Set();
  const parameterValues = new Map();
  const isNonemptyIterable = (expression, seen = new Set()) => {
    if (!expression || seen.has(expression)) return false;
    const nextSeen = new Set(seen).add(expression);
    if (typeScript.isArrayLiteralExpression(expression)) {
      return expression.elements.some(element => !typeScript.isSpreadElement(element));
    }
    if (
      typeScript.isCallExpression(expression) &&
      typeScript.isPropertyAccessExpression(expression.expression) &&
      expression.expression.name.text === 'entries'
    ) {
      return isNonemptyIterable(expression.expression.expression, nextSeen);
    }
    const resolved = resolveDeclaration(expression);
    if (parameterValues.has(resolved))
      return isNonemptyIterable(parameterValues.get(resolved), nextSeen);
    if (resolved && resolved !== expression) return isNonemptyIterable(resolved, nextSeen);
    return false;
  };
  // Metadata visitors often use Set.add. Only the contributor graph (including forwarded
  // helper parameters and typed graph aliases) has a graph-mutating add method.
  const isCommandGraphReceiver = (expression, seen = new Set()) => {
    if (!expression || seen.has(expression)) return false;
    const nextSeen = new Set(seen).add(expression);
    const resolved = resolveDeclaration(expression);
    if (resolved === entry.parameters?.[0]) return true;
    if (resolved?.type && /\bGPUCommandGraph\b/.test(resolved.type.getText())) return true;
    if (parameterValues.has(resolved))
      return isCommandGraphReceiver(parameterValues.get(resolved), nextSeen);
    if (resolved && typeScript.isNewExpression(resolved)) {
      return getReferenceName(resolved.expression) === 'GPUCommandGraph';
    }
    return typeScript.isPropertyAccessExpression(expression) && expression.name.text === 'graph';
  };
  const analyze = (node, states) => {
    if (!node || !states.size) return states;
    if (typeScript.isFunctionLike(node)) return states;
    if (typeScript.isReturnStatement(node)) {
      analyze(node.expression, states);
      return new Set();
    }
    if (typeScript.isThrowStatement(node)) return new Set();
    if (typeScript.isIfStatement(node)) {
      const conditions = analyze(node.expression, states);
      return new Set([
        ...analyzeBody(node.thenStatement, new Set(conditions)),
        ...analyzeBody(node.elseStatement, new Set(conditions))
      ]);
    }
    if (typeScript.isConditionalExpression(node)) {
      const conditions = analyze(node.condition, states);
      return new Set([
        ...analyze(node.whenTrue, new Set(conditions)),
        ...analyze(node.whenFalse, new Set(conditions))
      ]);
    }
    if (
      typeScript.isBinaryExpression(node) &&
      [
        typeScript.SyntaxKind.AmpersandAmpersandToken,
        typeScript.SyntaxKind.BarBarToken,
        typeScript.SyntaxKind.QuestionQuestionToken
      ].includes(node.operatorToken.kind)
    ) {
      const leftStates = analyze(node.left, states);
      return new Set([...leftStates, ...analyze(node.right, new Set(leftStates))]);
    }
    if (typeScript.isSwitchStatement(node)) {
      const conditions = analyze(node.expression, states);
      return new Set([
        ...conditions,
        ...node.caseBlock.clauses.flatMap(clause => [...analyzeBody(clause, new Set(conditions))])
      ]);
    }
    if (typeScript.isTryStatement(node)) {
      const outcomes = new Set([
        ...analyzeBody(node.tryBlock, new Set(states)),
        ...analyzeBody(node.catchClause?.block, new Set(states))
      ]);
      return analyzeBody(node.finallyBlock, outcomes);
    }
    if (typeScript.isIterationStatement(node, false)) {
      // The loop may run zero times; checks inside it do not dominate subsequent declarations.
      let iterationStates = new Set(states);
      typeScript.forEachChild(node, child => {
        iterationStates = analyzeBody(child, iterationStates);
      });
      // A literal nonempty list (also through a shared helper parameter) runs at least once.
      // Unknown row/chunk/round loops retain the zero-iteration path.
      return isCollectionBindingPreflight(node)
        ? new Set([true])
        : typeScript.isForOfStatement(node) && isNonemptyIterable(node.expression)
          ? iterationStates
          : states;
    }
    if (typeScript.isCallExpression(node)) {
      let nextStates = states;
      for (const argument of node.arguments) nextStates = analyze(argument, nextStates);
      const name = getReferenceName(node.expression);
      if (STORAGE_PREFLIGHT_HELPERS.has(name) || isStorageCapacityCheck(node))
        return new Set([true]);
      const isGraphDeclaration =
        typeScript.isPropertyAccessExpression(node.expression) &&
        (/^(createTransientBuffer|createBuffer|createDataView)$/.test(name) ||
          (name === 'add' && isCommandGraphReceiver(node.expression.expression)));
      if (STORAGE_DECLARATION_HELPERS.has(name) || isGraphDeclaration) {
        if (nextStates.has(false)) firstUncheckedDeclaration ??= node;
        if (name === 'createTransientView') {
          hasRowStorage ||= getStaticNumber(node.arguments[3]) === undefined;
        } else if (name === 'createTransientVectorView' || name === 'createTransientBuffer') {
          hasRowStorage = true;
        } else if (/KernelNode|ComputeCommandNode/.test(name)) {
          const properties = resolveDeclaration(node.arguments[1] ?? node.arguments[0]);
          const invocation =
            properties && typeScript.isObjectLiteralExpression(properties)
              ? getPropertyValue(getProperty(properties, 'invocationCount'))
              : null;
          // A one-invocation reference kernel may still bind a row-sized input. Without a
          // type checker, any caller-provided binding array is potentially row-proportional.
          hasRowStorage ||=
            getStaticNumber(invocation) === undefined ||
            Boolean(
              properties &&
                typeScript.isObjectLiteralExpression(properties) &&
                getProperty(properties, 'bindings')
            );
        }
        return nextStates;
      }
      let helper = resolveDeclaration(node.expression);
      if (
        typeScript.isPropertyAccessExpression(node.expression) &&
        node.expression.expression.kind === typeScript.SyntaxKind.ThisKeyword
      ) {
        helper = members.find(member => member.name?.getText() === node.expression.name.text);
      }
      const body = helper?.body ?? helper?.initializer?.body;
      if (body && !visiting.has(body)) {
        visiting.add(body);
        const previousParameters = new Map(parameterValues);
        for (const [index, parameter] of (helper.parameters ?? []).entries()) {
          parameterValues.set(parameter, node.arguments[index]);
        }
        // Returns end the helper's path, not the caller's. Preserve incoming unchecked paths
        // unless the helper demonstrably checks them before returning.
        const returns = [];
        const previousReturns = returnStates;
        returnStates = returns;
        const remaining = analyzeBody(body, nextStates);
        returnStates = previousReturns;
        visiting.delete(body);
        parameterValues.clear();
        for (const [parameter, value] of previousParameters) parameterValues.set(parameter, value);
        return new Set([...remaining, ...returns]);
      }
      return nextStates;
    }
    let nextStates = states;
    typeScript.forEachChild(node, child => {
      nextStates = analyzeBody(child, nextStates);
    });
    return nextStates;
  };
  let returnStates = [];
  const analyzeBody = (node, states) => {
    if (node && typeScript.isReturnStatement(node)) {
      const nextStates = analyze(node.expression, states);
      returnStates.push(...nextStates);
      return new Set();
    }
    return analyze(node, states);
  };
  analyzeBody(entryBody, new Set([false]));
  return hasRowStorage ? firstUncheckedDeclaration : null;
};

for (const file of checkedFiles) {
  const text = fileTexts.get(file);
  const lines = text.split('\n');

  // 1. bare-assert: assert imported from @luma.gl/core and called.
  if (/import\s*\{[^}]*\bassert\b[^}]*\}\s*from\s*'@luma\.gl\/core'/.test(text)) {
    const assertLines = lines
      .map((line, index) => (!isCommentLine(line) && /\bassert\(/.test(line) ? index + 1 : 0))
      .filter(Boolean);
    if (assertLines.length) {
      addFinding('bare-assert', file, {count: assertLines.length, lines: assertLines});
    }
  }

  // Classes exported directly or through a named export; inspect members, not following text.
  const sourceTree = sourceTrees.get(file);
  const exportedOrigins = new Set(getExports(file).values());
  const classMatches = sourceTree.statements.filter(
    node =>
      typeScript.isClassDeclaration(node) &&
      node.name &&
      (node.modifiers?.some(modifier => modifier.kind === typeScript.SyntaxKind.ExportKeyword) ||
        exportedOrigins.has(`${file}#${node.name.text}`))
  );
  const contributorClasses = classMatches.filter(
    declaration =>
      /^GPU/.test(declaration.name.text) &&
      getClassMembers(declaration).some(
        member => member.name?.getText().replace(/['"]/g, '') === 'getCommandNodes'
      )
  );

  // 2. missing-producer-declaration.
  for (const declaration of contributorClasses) {
    if (!isProducerClass(declaration)) {
      addFinding('missing-producer-declaration', file, {
        symbol: declaration.name.text,
        lines: [getLineNumber(text, declaration.getStart())]
      });
    }
  }

  for (const declaration of contributorClasses) {
    const unchecked = getStoragePreflightFinding(declaration);
    if (unchecked) {
      addFinding('storage-binding-preflight', file, {
        symbol: declaration.name.text,
        lines: [getLineNumber(text, declaration.getStart())]
      });
    }
  }

  // 3. single-invocation-data-loop: all WGSL loop forms, including referenced helper snippets.
  // Bounded dense loops remain reviewable heuristics with an explicit contract-allow comment.
  const invocationLines = [];
  const invocationDescriptions = [];
  const analysisLimitLines = [];
  const duplicateLinesByName = new Map();
  const checkKernelCallSite = ({call, callSite}) => {
    const properties = resolveDeclaration(call.arguments[1]);
    if (!properties || !typeScript.isObjectLiteralExpression(properties)) return;
    const {guards: callGuards} = getExecutionGuards(call);
    const siteGuards = callSite === call ? [] : getExecutionGuards(callSite).guards;
    const callPaths = applyExecutionGuards([createTextVariant('')], [...siteGuards, ...callGuards]);
    if (!callPaths.length) return;
    const invocation = getProperty(properties, 'invocationCount');
    const limitedProperties = [];
    const getPropertyVariants = name => {
      const property = getProperty(properties, name);
      if (!property) return callPaths;
      try {
        return combineTextVariants(
          callPaths,
          getStaticTextVariants(getPropertyValue(property)),
          false
        );
      } catch (error) {
        if (error !== SHADER_ANALYSIS_LIMIT) throw error;
        limitedProperties.push(name);
        return callPaths.map(variant => ({...variant, text: UNKNOWN_WGSL_TEXT}));
      }
    };
    const declarations = getPropertyVariants('declarations');
    const bodies = getPropertyVariants('body');
    const kernelIds = getProperty(properties, 'id')
      ? getPropertyVariants('id')
          .map(variant => variant.text)
          .sort()
      : ['<anonymous>'];
    const constants = resolveDeclaration(getPropertyValue(getProperty(properties, 'constants')));
    const kernelConstants =
      constants && typeScript.isObjectLiteralExpression(constants)
        ? constants.properties
            .map(property => {
              if (!typeScript.isPropertyAssignment(property))
                return normalizeSite(property.getText());
              const value = getPropertyValue(property);
              const numericValue = getStaticNumber(value);
              return `${property.name.getText()}:${numericValue ?? normalizeSite((resolveDeclaration(value) ?? value).getText())}`;
            })
            .sort()
        : constants
          ? [normalizeSite(constants.getText())]
          : [];
    if (limitedProperties.length) analysisLimitLines.push(getLineNumber(text, callSite.getStart()));
    if (getStaticNumber(getPropertyValue(invocation)) === 1) {
      const line = getLineNumber(
        text,
        callSite === call ? invocation.getStart() : callSite.getStart()
      );
      const hasLoop = declarations.some(declaration =>
        bodies.some(
          body =>
            combineTextVariants([declaration], [body]).length &&
            /(?<![\w\u0000])(?:for|while)\s*\(|(?<![\w\u0000])loop\s*\{/.test(
              getReachableWGSLText(declaration.text, body.text)
            )
        )
      );
      if (hasLoop && !/contract-allow:\s*single-invocation/.test(lines[line - 2] ?? '')) {
        invocationLines.push(line);
        const context = [];
        for (let parent = callSite.parent; parent; parent = parent.parent) {
          if (
            (typeScript.isFunctionDeclaration(parent) ||
              typeScript.isMethodDeclaration(parent) ||
              typeScript.isClassDeclaration(parent) ||
              typeScript.isVariableDeclaration(parent)) &&
            parent.name
          ) {
            context.unshift(parent.name.getText());
          }
        }
        const shaders = declarations.flatMap(declaration =>
          bodies.flatMap(body =>
            combineTextVariants([declaration], [body]).length
              ? [normalizeSite(getReachableWGSLText(declaration.text, body.text, true))]
              : []
          )
        );
        invocationDescriptions.push(
          `${context.join('.') || '<module>'}: kernel ${JSON.stringify(kernelIds)}${kernelConstants.length ? ` constants ${JSON.stringify(kernelConstants)}` : ''} shader ${hashSite([...new Set(shaders)].sort().join('\n'))}`
        );
      }
    }
    // Module declarations and the main body have different lexical scopes. Analyze them
    // separately so independent choices do not multiply entire shader strings.
    const duplicateNames = new Set([
      ...declarations.flatMap(variant => [...getDuplicateWGSLNames(variant.text)]),
      ...bodies.flatMap(variant => [...getDuplicateWGSLNames(`{${variant.text}}`)])
    ]);
    for (const name of duplicateNames) {
      const linesForName = duplicateLinesByName.get(name) ?? [];
      linesForName.push(getLineNumber(text, callSite.getStart()));
      duplicateLinesByName.set(name, linesForName);
    }
  };
  for (const site of getKernelCallSites(sourceTree)) {
    withKernelParameterBindings(site.parameterBindings, () => checkKernelCallSite(site));
  }
  if (invocationLines.length) {
    addFinding('single-invocation-data-loop', file, {
      count: invocationLines.length,
      lines: invocationLines,
      siteDescriptions: invocationDescriptions
    });
  }
  if (analysisLimitLines.length) {
    addFinding('shader-analysis-limit', file, {
      count: analysisLimitLines.length,
      lines: analysisLimitLines
    });
  }
  for (const [symbol, duplicateLines] of duplicateLinesByName) {
    addFinding('duplicate-wgsl-declaration', file, {symbol, lines: duplicateLines});
  }

  // 5. luma-reexport.
  const reexportLines = [
    ...text.matchAll(
      /^export\s+(?:type\s+)?(?:\*(?:\s+as\s+\w+)?|\{[^}]*\})\s*from\s*['"]@luma\.gl\//gm
    )
  ].map(match => getLineNumber(text, match.index));
  if (reexportLines.length) {
    addFinding('luma-reexport', file, {count: reexportLines.length, lines: reexportLines});
  }

  // 6. multiple-classes.
  if (classMatches.length > 1) {
    addFinding('multiple-classes', file, {
      count: classMatches.length,
      lines: classMatches.map(declaration => getLineNumber(text, declaration.getStart()))
    });
  }

  // 8. unregistered-contributor: exported contributor classes and add*Recipe functions.
  const registrableNames = [
    ...contributorClasses.map(declaration => ({
      name: declaration.name.text,
      offset: declaration.getStart()
    })),
    ...[...text.matchAll(/^export function (add\w+Recipe)\b/gm)].map(match => ({
      name: match[1],
      offset: match.index
    }))
  ];
  for (const {name, offset} of registrableNames) {
    if (internalSymbols.has(`${file}#${name}`)) continue;
    const identities =
      publicIdentities.get(`${file}#${name}`) ??
      new Set([`${packageName}/${getTopDirectory(file)}:${name}`]);
    for (const identity of identities) {
      if (!registeredIdentities.has(identity))
        addFinding('unregistered-contributor', file, {
          symbol: name,
          entryPoint: identity.slice(0, identity.lastIndexOf(':')),
          lines: [getLineNumber(text, offset)]
        });
    }
  }
}

// Version 2: { _version: 2, rule: { file: [{id, description, reason}] } }.
// Legacy counts are accepted only by the explicit migration, never by the checking path.
const buildBaseline = (foundList, previous, reason, migrate) => {
  const baseline = {_version: 2};
  for (const finding of [...foundList].sort(compareFindings)) {
    if (STRICT_RULES.has(finding.rule) && !hasReviewedLegacyInternalAllowance(finding, previous))
      continue;
    const recorded = previous[finding.rule]?.[finding.file];
    if (migrate) {
      const permitted = COUNT_RULES.has(finding.rule)
        ? typeof recorded === 'number' && finding.count <= recorded
        : Array.isArray(recorded) && recorded.includes(finding.symbol);
      if (!permitted) {
        process.stdout.write(
          `NOT MIGRATED ${finding.rule} ${finding.file}: not covered by the legacy baseline\n`
        );
        continue;
      }
    }
    const entries = finding.sites.map(site => {
      const prior =
        previous._version === 2 && Array.isArray(recorded)
          ? recorded.find(entry => entry.id === site.id)
          : null;
      const explanation =
        prior?.reason ||
        (migrate ? previous._reasons?.[finding.rule]?.[finding.file] : '') ||
        reason;
      if (typeof explanation !== 'string' || !explanation.trim()) {
        throw new Error(
          `A nonempty reason is required for ${finding.rule} ${finding.file} ${site.id}; pass --reason <explanation>`
        );
      }
      if (!prior)
        process.stdout.write(
          `ADDED ${finding.rule} ${finding.file} ${site.id} ${site.description}: ${explanation}\n`
        );
      return {...prior, ...site, reason: explanation};
    });
    baseline[finding.rule] ??= {};
    baseline[finding.rule][finding.file] ??= [];
    baseline[finding.rule][finding.file].push(...entries);
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
const loadBaseline = () =>
  fs.existsSync(BASELINE_PATH) ? JSON.parse(fs.readFileSync(BASELINE_PATH, 'utf8')) : {};
function hasReviewedLegacyInternalAllowance(finding, baseline) {
  if (finding.rule !== 'internal-export' || baseline._version !== 2) return false;
  const entries = baseline[finding.rule]?.[finding.file];
  return (
    Array.isArray(entries) &&
    finding.sites.every(site =>
      entries.some(
        entry =>
          entry.id === site.id &&
          entry.description === site.description &&
          entry.legacy === 'count-baseline-2026-11-17' &&
          typeof entry.owner === 'string' &&
          entry.owner.trim() &&
          typeof entry.reason === 'string' &&
          entry.reason.trim()
      )
    )
  );
}
const isNewFinding = (finding, baseline) => {
  if (STRICT_RULES.has(finding.rule)) return !hasReviewedLegacyInternalAllowance(finding, baseline);
  if (baseline._version !== 2) return true;
  const recorded = baseline[finding.rule]?.[finding.file];
  return (
    !Array.isArray(recorded) ||
    finding.sites.some(
      site =>
        !recorded.some(
          entry => entry.id === site.id && typeof entry.reason === 'string' && entry.reason.trim()
        )
    )
  );
};

// Command line.
const arguments_ = process.argv.slice(2);
const ruleIndex = arguments_.indexOf('--rule');
const selectedRule = ruleIndex >= 0 ? arguments_[ruleIndex + 1] : null;
const reasonIndex = arguments_.indexOf('--reason');
const reason = reasonIndex >= 0 ? arguments_[reasonIndex + 1] : null;
if (reasonIndex >= 0 && (!reason?.trim() || reason.startsWith('--'))) {
  process.stderr.write('--reason requires a nonempty explanation\n');
  process.exit(2);
}
const migrate = arguments_.includes('--migrate-baseline');
const selectedPaths = [];
for (let index = 0; index < arguments_.length; index++) {
  if (arguments_[index] !== '--paths') continue;
  const firstPathIndex = index + 1;
  while (arguments_[index + 1] && !arguments_[index + 1].startsWith('--')) {
    selectedPaths.push(path.relative(process.cwd(), path.resolve(arguments_[++index])));
  }
  if (index < firstPathIndex) {
    process.stderr.write('--paths requires at least one file or directory\n');
    process.exit(2);
  }
}
if (selectedPaths.length && (arguments_.includes('--update-baseline') || migrate)) {
  process.stderr.write(
    '--paths cannot be combined with --update-baseline; update the complete baseline explicitly\n'
  );
  process.exit(2);
}
const matchesSelectedPath = file =>
  !selectedPaths.length ||
  selectedPaths.some(
    selected => selected === '' || file === selected || file.startsWith(`${selected}${path.sep}`)
  );

if (selectedRule && !RULE_IDENTIFIERS.includes(selectedRule)) {
  process.stderr.write(`Unknown rule '${selectedRule}'. Rules: ${RULE_IDENTIFIERS.join(', ')}\n`);
  process.exit(2);
}

if (arguments_.includes('--update-baseline') || migrate) {
  const previous = loadBaseline();
  if (migrate && previous._version === 2) {
    process.stderr.write('Baseline is already version 2; use --update-baseline\n');
    process.exit(2);
  }
  const strictFindings = findings.filter(
    finding =>
      STRICT_RULES.has(finding.rule) && !hasReviewedLegacyInternalAllowance(finding, previous)
  );
  if (strictFindings.length && !migrate) {
    process.stderr.write(
      `Rules with zero baselines cannot be baselined: ${[...new Set(strictFindings.map(finding => finding.rule))].join(', ')}\n`
    );
    process.exit(1);
  }
  let baseline;
  try {
    baseline = buildBaseline(findings, previous, reason, migrate);
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exit(1);
  }
  fs.writeFileSync(BASELINE_PATH, `${JSON.stringify(baseline, null, 2)}\n`);
  process.stdout.write(`Wrote ${BASELINE_PATH} with ${findings.length} findings.\n`);
  process.exit(0);
}

const baseline = loadBaseline();
const shownFindings = findings
  .filter(
    finding => (!selectedRule || finding.rule === selectedRule) && matchesSelectedPath(finding.file)
  )
  .sort(compareFindings);
const newFindings = shownFindings.filter(finding => isNewFinding(finding, baseline));
const printedFindings = arguments_.includes('--all') ? shownFindings : newFindings;
let staleCount = 0;
if (arguments_.includes('--check-stale')) {
  for (const [rule, files] of Object.entries(baseline)) {
    if (rule.startsWith('_') || (selectedRule && rule !== selectedRule)) continue;
    for (const [file, sites] of Object.entries(files)) {
      if (!matchesSelectedPath(file) || !Array.isArray(sites)) continue;
      const current = findings
        .filter(finding => finding.rule === rule && finding.file === file)
        .flatMap(finding => finding.sites);
      for (const site of sites) {
        if (!current.some(entry => entry.id === site.id)) {
          process.stdout.write(`STALE ${rule} ${file} ${site.id} ${site.description}\n`);
          staleCount++;
        }
      }
    }
  }
}

process.stdout.write('rule                           total  new\n');
for (const rule of selectedRule ? [selectedRule] : RULE_IDENTIFIERS) {
  const ruleFindings = shownFindings.filter(finding => finding.rule === rule);
  const total = COUNT_RULES.has(rule)
    ? ruleFindings.reduce((sum, finding) => sum + finding.count, 0)
    : ruleFindings.length;
  const newCount = newFindings.filter(finding => finding.rule === rule).length;
  process.stdout.write(
    `${rule.padEnd(30)} ${String(total).padStart(5)} ${String(newCount).padStart(4)}\n`
  );
}
for (const finding of printedFindings) {
  const detail = finding.symbol
    ? ` ${finding.symbol}`
    : finding.count > 1
      ? ` x${finding.count}`
      : '';
  const marker = isNewFinding(finding, baseline) ? 'NEW ' : '';
  const qualification = finding.entryPoint ? ` (${finding.entryPoint})` : '';
  process.stdout.write(
    `${marker}${finding.rule} ${finding.file}:${finding.lines[0]}${qualification}${detail}\n`
  );
}
if (newFindings.length || staleCount) {
  process.stdout.write(`${newFindings.length} new contributor-contract findings\n`);
  process.exitCode = 1;
} else {
  process.stdout.write('contributor-contract: no new findings\n');
}
