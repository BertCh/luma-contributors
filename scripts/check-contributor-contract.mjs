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
import {fileURLToPath} from 'node:url';
import {findLumaReExportSites} from './luma-binding-origins.mjs';
import {collectCatalogIdentities} from './contributor-contract-catalog.mjs';
import {findInternalPublicMembers} from './contributor-contract-public-surface.mjs';
import {
  createPublicSourceInventory,
  createSourceResolvers
} from './contributor-contract-source-inventory.mjs';

function runContributorContract() {
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

  // Root coverage also inventories declaration files and wildcard target directories.
  function listExportSourceFiles(directory) {
    return fs.readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
      if (entry.name === 'node_modules' || entry.name === 'dist') return [];
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) return listExportSourceFiles(file);
      return entry.isFile() && SOURCE_EXTENSION_PATTERN.test(entry.name) ? [file] : [];
    });
  }

  function createSourceTargetPattern(target) {
    const [prefix, ...suffixes] = target
      .slice('./dist/'.length)
      .replace(SOURCE_SUFFIX_PATTERN, '')
      .split('*')
      .map(part => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    return new RegExp(
      suffixes.length ? `^${prefix}(?<subpath>.*)${suffixes.join('\\k<subpath>')}$` : `^${prefix}$`
    );
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
        const pattern = createSourceTargetPattern(target);
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
    'root-file',
    'self-format-validation'
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
    'storage-binding-preflight',
    'self-format-validation'
  ];

  // Generic baseline updates cannot excuse strict-rule regressions. Final reconciliation can retain
  // explicitly reviewed pre-pass internal exports; their site, owner and revision must all match.
  const STRICT_RULES = new Set([
    'bare-assert',
    'missing-producer-declaration',
    'internal-export',
    'luma-reexport',
    'unregistered-contributor',
    'duplicate-wgsl-declaration',
    'self-format-validation'
  ]);

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
  if (ruleIndex >= 0 && (arguments_.includes('--update-baseline') || migrate)) {
    process.stderr.write(
      '--rule cannot be combined with --update-baseline or --migrate-baseline; baseline writes require the complete inventory\n'
    );
    process.exit(2);
  }

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

  if (ruleIndex >= 0 && !RULE_IDENTIFIERS.includes(selectedRule)) {
    process.stderr.write(`Unknown rule '${selectedRule}'. Rules: ${RULE_IDENTIFIERS.join(', ')}\n`);
    process.exit(2);
  }

  const walk = directory =>
    fs.readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
      if (entry.name === 'node_modules' || entry.name === 'dist') return [];
      const fullPath = path.join(directory, entry.name);
      if (entry.isDirectory()) return walk(fullPath);
      return SOURCE_EXTENSION_PATTERN.test(entry.name) &&
        !DECLARATION_SOURCE_PATTERN.test(entry.name)
        ? [fullPath]
        : [];
    });

  const getLineNumber = (text, offset) => text.slice(0, offset).split('\n').length;

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

  const sourceFiles = listExportSourceFiles(SOURCE_ROOT);
  const fileTexts = new Map(sourceFiles.map(file => [file, fs.readFileSync(file, 'utf8')]));
  const getTopDirectory = file => path.relative(SOURCE_ROOT, file).split(path.sep)[0];
  const isCheckedFile = file => !SKIPPED_DIRECTORIES.has(getTopDirectory(file));
  const checkedFiles = sourceFiles.filter(
    file => isCheckedFile(file) && !DECLARATION_SOURCE_PATTERN.test(file)
  );

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

  const {resolveSourceFile, resolveModule} = createSourceResolvers(sourceTrees);

  // Evaluate the values emitted by the public catalog, rather than crediting every read dependency.
  const catalogTrees = [
    ...walk(ONTOLOGY_DIRECTORY),
    `${SOURCE_ROOT}/gpu-spatial-analysis/spatial-validation/spatial-validation-descriptors.ts`
  ]
    .filter(file => sourceTrees.has(file))
    .map(file => sourceTrees.get(file));
  const registeredIdentities = collectCatalogIdentities(catalogTrees, resolveModule);

  // 4. internal-export: resolve export chains from each entry barrel by reading files.
  const internalSymbols = new Set(); // `${file}#${topLevelDeclarationName}`
  for (const [file, tree] of sourceTrees) {
    for (const statement of tree.statements) {
      const declarations = typeScript.isVariableStatement(statement)
        ? statement.declarationList.declarations
        : [statement];
      for (const declaration of declarations) {
        // Bind documentation to the actual top-level declaration. A nested namesake cannot
        // mark an unrelated public declaration, and default modifiers do not hide its tag.
        const internal = [statement, declaration].some(node =>
          typeScript.getJSDocTags(node).some(tag => tag.tagName.text === 'internal')
        );
        if (!internal) continue;
        const name =
          declaration.name && typeScript.isIdentifier(declaration.name)
            ? declaration.name.text
            : statement.modifiers?.some(
                  modifier => modifier.kind === typeScript.SyntaxKind.DefaultKeyword
                )
              ? 'default'
              : undefined;
        if (name) internalSymbols.add(`${file}#${name}`);
      }
    }
  }
  const {
    inventory: publicInventory,
    barrelFiles,
    getExports
  } = createPublicSourceInventory({
    analysisManifest,
    sourceRoot: SOURCE_ROOT,
    sourceTrees,
    resolveModule,
    resolveSourceFile
  });
  const packageName = analysisManifest.name ?? '@luma-contributors/analysis';
  const publicIdentities = new Map();
  for (const {entryPoint, exportName, declaration} of publicInventory) {
    if (!publicIdentities.has(declaration)) publicIdentities.set(declaration, new Set());
    publicIdentities.get(declaration).add(`${entryPoint}:${exportName}`);
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
        for (const declaration of statement.declarationList.declarations)
          addDeclaration(declaration);
      } else if (typeScript.isImportDeclaration(statement)) {
        const clause = statement.importClause;
        if (clause?.name) bindings.set(clause.name.text, clause);
        if (clause?.namedBindings && typeScript.isNamespaceImport(clause.namedBindings)) {
          addDeclaration(clause.namedBindings);
        }
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
  const valueScopeBindings = new WeakMap();
  const getValueScopeBindings = scope => {
    if (valueScopeBindings.has(scope)) return valueScopeBindings.get(scope);
    const bindings = new Map();
    const add = declaration => {
      if (declaration.name && typeScript.isIdentifier(declaration.name))
        bindings.set(declaration.name.text, declaration);
      else
        for (const element of declaration.name?.elements ?? [])
          if (typeScript.isBindingElement(element)) add(element);
    };
    for (const parameter of scope.parameters ?? []) add(parameter);
    if (typeScript.isCatchClause(scope) && scope.variableDeclaration)
      add(scope.variableDeclaration);
    if (
      typeScript.isForStatement(scope) ||
      typeScript.isForOfStatement(scope) ||
      typeScript.isForInStatement(scope)
    ) {
      if (scope.initializer && typeScript.isVariableDeclarationList(scope.initializer))
        for (const declaration of scope.initializer.declarations) add(declaration);
    }
    for (const statement of scope.statements ?? []) {
      if (typeScript.isVariableStatement(statement))
        for (const declaration of statement.declarationList.declarations) add(declaration);
      else if (typeScript.isImportDeclaration(statement)) {
        const clause = statement.importClause;
        if (!clause || clause.isTypeOnly) continue;
        if (clause.name) bindings.set(clause.name.text, clause);
        const named = clause.namedBindings;
        if (named && typeScript.isNamespaceImport(named)) add(named);
        else if (named)
          for (const specifier of named.elements) if (!specifier.isTypeOnly) add(specifier);
      } else if (
        typeScript.isFunctionDeclaration(statement) ||
        typeScript.isClassDeclaration(statement) ||
        typeScript.isEnumDeclaration(statement) ||
        typeScript.isModuleDeclaration(statement)
      )
        add(statement);
    }
    valueScopeBindings.set(scope, bindings);
    return bindings;
  };
  const findBinding = (identifier, valueOnly = false) => {
    for (let scope = identifier.parent; scope; scope = scope.parent) {
      const binding = (valueOnly ? getValueScopeBindings(scope) : getScopeBindings(scope)).get(
        identifier.text
      );
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
    if (typeScript.isImportSpecifier(expression) || typeScript.isImportClause(expression)) {
      const declaration = typeScript.isImportClause(expression)
        ? expression.parent
        : expression.parent.parent.parent;
      const target = resolveModule(
        expression.getSourceFile().fileName,
        declaration.moduleSpecifier.text
      );
      if (!target) return expression;
      const name = typeScript.isImportClause(expression)
        ? 'default'
        : (expression.propertyName ?? expression.name).text;
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
                [
                  typeScript.SyntaxKind.PlusPlusToken,
                  typeScript.SyntaxKind.MinusMinusToken
                ].includes(node.operator)
              ? node.operand
              : null;
        let root = target;
        while (
          root &&
          (typeScript.isPropertyAccessExpression(root) ||
            typeScript.isElementAccessExpression(root))
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
    const knownValue =
      kind === 'nullish' ? getStaticNullish(condition) : getStaticBoolean(condition);
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
        if (child !== parent.condition)
          guards.unshift([parent.condition, child === parent.whenTrue]);
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
    if (
      typeScript.isCallExpression(expression) &&
      expression.arguments.length === 1 &&
      typeScript.isPropertyAccessExpression(expression.expression) &&
      typeScript.isIdentifier(expression.expression.expression) &&
      expression.expression.expression.text === 'Math' &&
      expression.expression.name.text === 'floor' &&
      !findBinding(expression.expression.expression)
    ) {
      const value = evaluate(expression.arguments[0]);
      return Number.isFinite(value) ? Math.floor(value) : undefined;
    }
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
        case typeScript.SyntaxKind.AsteriskAsteriskToken: {
          // Host bounds such as 2 ** 24 are exact integers. Never certify an overflowed or
          // rounded power as an enforcing bound; declaration/import resolution stays shared.
          const value = left ** right;
          return Number.isSafeInteger(value) ? value : undefined;
        }
        case typeScript.SyntaxKind.SlashToken:
          return left / right;
        default:
          return undefined;
      }
    }
    return undefined;
  };

  // CORE-118: a loop allowance (baseline reason or inline contract-allow comment) that names a host
  // bound such as `MAXIMUM_TILE_COUNT = 4096` stays valid only while that bound does. A cited
  // CAPITAL_CASE name with an underscore must still be visible in the file (a top-level binding or
  // import, or a WGSL constant in its shader text); when the citation gives a numeric value and the
  // name resolves to a static host number, the two must agree. Changing the bound's name or value
  // therefore invalidates the allowance; a site's own shader text is already part of its identity.
  const CITED_BOUND_PATTERN = /\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b/g;
  const CITED_BOUND_VALUE_PATTERN =
    /\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\s*=\s*(0x[0-9a-fA-F_]+|\d[\d_,]*(?:\.\d+)?)(?![\w(]|\.\d)/g;
  const parseCitedNumber = text => Number(text.replace(/[_,]/g, ''));
  let topLevelConstants;
  // Top-level `const` declarations of every analysis source, by name.
  const getTopLevelConstants = () => {
    if (topLevelConstants) return topLevelConstants;
    topLevelConstants = new Map();
    for (const [file, sourceTree] of sourceTrees) {
      for (const statement of sourceTree.statements) {
        if (
          !typeScript.isVariableStatement(statement) ||
          !(statement.declarationList.flags & typeScript.NodeFlags.Const)
        )
          continue;
        for (const declaration of statement.declarationList.declarations) {
          if (!typeScript.isIdentifier(declaration.name)) continue;
          const declarations = topLevelConstants.get(declaration.name.text) ?? [];
          declarations.push({file, declaration});
          topLevelConstants.set(declaration.name.text, declarations);
        }
      }
    }
    return topLevelConstants;
  };
  // The declaration a citation in `file` names: a binding in the file's scope, else the one
  // top-level constant of that name in the file's directory, else the only one in the package.
  // `found` is false when no declaration or WGSL use of the name exists for the file.
  const resolveCitedBound = (file, name) => {
    const binding = getScopeBindings(sourceTrees.get(file)).get(name);
    if (binding) return {found: true, declaration: binding};
    const declarations = getTopLevelConstants().get(name) ?? [];
    const local = declarations.filter(
      ({file: other}) => path.dirname(other) === path.dirname(file)
    );
    const candidates = local.length ? local : declarations;
    if (candidates.length === 1) return {found: true, declaration: candidates[0].declaration};
    // Comments, including the allowance itself, do not count as a use.
    const code = fileTexts.get(file).replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
    return {found: candidates.length > 1 || new RegExp(`\\b${name}\\b`).test(code)};
  };
  const getCitedBoundProblems = (file, text) => {
    if (!sourceTrees.has(file) || typeof text !== 'string') return [];
    const problems = [];
    for (const name of new Set(text.match(CITED_BOUND_PATTERN) ?? [])) {
      if (!resolveCitedBound(file, name).found) {
        problems.push(`cites ${name}, which no analysis source declares and ${file} does not use`);
      }
    }
    for (const [, name, cited] of text.matchAll(CITED_BOUND_VALUE_PATTERN)) {
      const {declaration} = resolveCitedBound(file, name);
      const value = declaration && getStaticNumber(declaration);
      if (declaration && value === undefined) {
        problems.push(`cites ${name} = ${cited}, but cannot statically resolve its host value`);
      } else if (value !== undefined && value !== parseCitedNumber(cited)) {
        problems.push(`cites ${name} = ${cited}, but the source value is ${value}`);
      }
    }
    return problems;
  };
  // An inline allowance needs a stated bound: text after the marker on its line, or comment lines
  // directly above it. A bare marker with no explanation does not suppress the finding.
  const hasQualifiedLoopAllowance = (file, lines, line) => {
    const markerLine = lines[line - 2] ?? '';
    const marker = /contract-allow:\s*single-invocation\b(.*)$/.exec(markerLine);
    if (!marker) return false;
    const reasonLines = [marker[1].replace(/^[\s\-—:;,.]+/, '')];
    for (let index = line - 3; index >= 0 && /^\s*\/\//.test(lines[index]); index--) {
      reasonLines.unshift(lines[index].replace(/^\s*\/\/\s*/, ''));
    }
    const reason = reasonLines.join(' ').trim();
    return (
      (/\b\d+(?:[_,.]\d+)*(?:\s*\^\s*\d+)?\b/.test(reason) ||
        /\b(?:one|two|three|four|five|six|seven|eight|nine|ten|sixteen|thirty-two|sixty-four)\b(?:[ -]\w+){0,3}\s+(?:words?|steps?|projections?|iterations?|rows?|slots?|columns?|bisections?)\b/i.test(
          reason
        ) ||
        [...reason.matchAll(CITED_BOUND_PATTERN)].some(
          ([name]) => resolveCitedBound(file, name).found
        )) &&
      !getCitedBoundProblems(file, reason).length
    );
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
      const count = resolveDeclaration(
        getPropertyValue(getProperty(properties, 'invocationCount'))
      );
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

  // VAL-01: checking a port only against its own format proves layout, not its element type.
  // This is deliberately a source heuristic: require a dominating explicit format guard, or a
  // generic parameter whose every in-repository call has such a guard. Unknown callers are not proof.
  const unwrapFormatExpression = expression => {
    while (expression && isExpressionWrapper(expression)) expression = expression.expression;
    return expression;
  };
  const getFormatOwner = expression => {
    expression = unwrapFormatExpression(expression);
    if (
      expression &&
      typeScript.isPropertyAccessExpression(expression) &&
      expression.name.text === 'format'
    )
      return expression.expression;
    if (
      expression &&
      typeScript.isElementAccessExpression(expression) &&
      expression.argumentExpression &&
      typeScript.isStringLiteralLike(expression.argumentExpression) &&
      expression.argumentExpression.text === 'format'
    )
      return expression.expression;
    return null;
  };
  const validatorName = expression => {
    const resolved = resolveDeclaration(expression);
    return resolved && typeScript.isImportSpecifier(resolved)
      ? (resolved.propertyName ?? resolved.name).text
      : getReferenceName(expression);
  };
  const formatReference = value => {
    if (value?.isFormatReference) return value;
    value = unwrapFormatExpression(value);
    if (!value) return null;
    if (typeScript.isIdentifier(value))
      return {
        isFormatReference: true,
        root: findBinding(value, true) ?? value.getSourceFile(),
        parts: findBinding(value, true) ? [] : [value.text]
      };
    if (value.kind === typeScript.SyntaxKind.ThisKeyword)
      return {
        isFormatReference: true,
        root: getEnclosingFunction(value),
        parts: ['this']
      };
    if (typeScript.isPropertyAccessExpression(value)) {
      const owner = formatReference(value.expression);
      return owner && {...owner, parts: [...owner.parts, value.name.text]};
    }
    if (
      typeScript.isElementAccessExpression(value) &&
      value.argumentExpression &&
      (typeScript.isStringLiteralLike(value.argumentExpression) ||
        typeScript.isNumericLiteral(value.argumentExpression))
    ) {
      const owner = formatReference(value.expression);
      return owner && {...owner, parts: [...owner.parts, value.argumentExpression.text]};
    }
    return null;
  };
  const sameFormatReference = (left, right) => {
    left = formatReference(left);
    right = formatReference(right);
    return Boolean(
      left &&
        right &&
        left.root === right.root &&
        left.parts.length === right.parts.length &&
        left.parts.every((part, index) => part === right.parts[index])
    );
  };
  const getFormatIteration = value => {
    const reference = formatReference(value);
    if (!reference || reference.parts.length) return null;
    for (let node = reference.root; node && !typeScript.isFunctionLike(node); node = node.parent) {
      if (typeScript.isForOfStatement(node)) return node;
    }
    return null;
  };
  const getSelfFormatSource = call => {
    if (
      !['validateSinglePackedView', 'validatePackedView'].includes(validatorName(call.expression))
    )
      return null;
    const formats = unwrapFormatExpression(call.arguments[1]);
    if (!formats || !typeScript.isArrayLiteralExpression(formats) || formats.elements.length !== 1)
      return null;
    const owner = getFormatOwner(formats.elements[0]);
    if (!owner) return null;
    if (sameFormatReference(owner, call.arguments[0])) return owner;
    const loop = getFormatIteration(call.arguments[0]);
    const source = loop && unwrapFormatExpression(loop.expression);
    return source &&
      typeScript.isCallExpression(source) &&
      validatorName(source.expression) === 'getGraphViewChunks' &&
      sameFormatReference(source.arguments[0], owner)
      ? owner
      : null;
  };
  const formatCallsByTarget = new Map();
  const checksSelfFormats = !selectedRule || selectedRule === 'self-format-validation';
  if (checksSelfFormats)
    for (const tree of sourceTrees.values())
      visitNodes(tree, node => {
        if (!typeScript.isCallExpression(node)) return;
        const target = resolveDeclaration(node.expression);
        if (!target || !typeScript.isFunctionLike(target)) return;
        const calls = formatCallsByTarget.get(target) ?? [];
        calls.push(node);
        formatCallsByTarget.set(target, calls);
      });
  const containsFormatReference = (node, value) => {
    let found = false;
    visitNodes(node, child => {
      const owner = getFormatOwner(child);
      if (owner && sameFormatReference(owner, value)) found = true;
    });
    return found;
  };
  const isExplicitFormatList = expression => {
    const list = resolveDeclaration(expression) ?? unwrapFormatExpression(expression);
    return (
      list &&
      typeScript.isArrayLiteralExpression(list) &&
      list.elements.length > 0 &&
      list.elements.every(element =>
        typeScript.isStringLiteralLike(unwrapFormatExpression(element))
      )
    );
  };
  const writesFormatReference = (statement, value) => {
    const expected = formatReference(value);
    if (!expected) return true;
    let written = false;
    const inspect = node => {
      if (typeScript.isFunctionLike(node)) return;
      let target;
      if (
        typeScript.isBinaryExpression(node) &&
        node.operatorToken.kind >= typeScript.SyntaxKind.FirstAssignment &&
        node.operatorToken.kind <= typeScript.SyntaxKind.LastAssignment
      )
        target = node.left;
      if (
        (typeScript.isPrefixUnaryExpression(node) || typeScript.isPostfixUnaryExpression(node)) &&
        [typeScript.SyntaxKind.PlusPlusToken, typeScript.SyntaxKind.MinusMinusToken].includes(
          node.operator
        )
      )
        target = node.operand;
      if (typeScript.isDeleteExpression(node)) target = node.expression;
      if (target) {
        const actual = formatReference(target);
        if (
          actual &&
          actual.root === expected.root &&
          actual.parts
            .slice(0, Math.min(actual.parts.length, expected.parts.length))
            .every((part, index) => part === expected.parts[index])
        )
          written = true;
      }
      typeScript.forEachChild(node, inspect);
    };
    inspect(statement);
    return written;
  };
  const getTupleFormatChecks = statement => {
    if (
      !typeScript.isForOfStatement(statement) ||
      !typeScript.isVariableDeclarationList(statement.initializer)
    )
      return [];
    const declaration = statement.initializer.declarations[0];
    if (!declaration || !typeScript.isArrayBindingPattern(declaration.name)) return [];
    const [viewBinding, formatBinding] = declaration.name.elements;
    if (
      !viewBinding?.name ||
      !formatBinding?.name ||
      !typeScript.isIdentifier(viewBinding.name) ||
      !typeScript.isIdentifier(formatBinding.name)
    )
      return [];
    const rows = unwrapFormatExpression(statement.expression);
    if (!rows || !typeScript.isArrayLiteralExpression(rows)) return [];
    const body = typeScript.isBlock(statement.statement)
      ? statement.statement.statements
      : [statement.statement];
    const first = body[0];
    if (!first || !typeScript.isExpressionStatement(first)) return [];
    const check = unwrapFormatExpression(first.expression);
    if (
      !typeScript.isCallExpression(check) ||
      !['validateSinglePackedView', 'validatePackedView'].includes(
        validatorName(check.expression)
      ) ||
      !sameFormatReference(check.arguments[0], viewBinding.name)
    )
      return [];
    const formats = unwrapFormatExpression(check.arguments[1]);
    if (
      !formats ||
      !typeScript.isArrayLiteralExpression(formats) ||
      formats.elements.length !== 1 ||
      !sameFormatReference(formats.elements[0], formatBinding.name)
    )
      return [];
    // A break/continue before the next row can leave the remaining collection unchecked.
    let incomplete = false;
    visitNodes(statement.statement, node => {
      if (
        typeScript.isBreakStatement(node) ||
        typeScript.isContinueStatement(node) ||
        typeScript.isReturnStatement(node)
      )
        incomplete = true;
    });
    if (incomplete) return [];
    return rows.elements.flatMap(row => {
      row = unwrapFormatExpression(row);
      return typeScript.isArrayLiteralExpression(row) &&
        row.elements[0] &&
        row.elements[1] &&
        typeScript.isStringLiteralLike(unwrapFormatExpression(row.elements[1]))
        ? [row.elements[0]]
        : [];
    });
  };
  const rejectsWrongFormat = (condition, value) => {
    condition = unwrapFormatExpression(condition);
    if (typeScript.isBinaryExpression(condition)) {
      if (condition.operatorToken.kind === typeScript.SyntaxKind.AmpersandAmpersandToken)
        return (
          rejectsWrongFormat(condition.left, value) && rejectsWrongFormat(condition.right, value)
        );
      if (
        [
          typeScript.SyntaxKind.ExclamationEqualsEqualsToken,
          typeScript.SyntaxKind.ExclamationEqualsToken
        ].includes(condition.operatorToken.kind)
      )
        return (
          (sameFormatReference(getFormatOwner(condition.left), value) &&
            typeScript.isStringLiteralLike(unwrapFormatExpression(condition.right))) ||
          (sameFormatReference(getFormatOwner(condition.right), value) &&
            typeScript.isStringLiteralLike(unwrapFormatExpression(condition.left)))
        );
    }
    if (
      typeScript.isPrefixUnaryExpression(condition) &&
      condition.operator === typeScript.SyntaxKind.ExclamationToken
    ) {
      const call = unwrapFormatExpression(condition.operand);
      return (
        typeScript.isCallExpression(call) &&
        typeScript.isPropertyAccessExpression(call.expression) &&
        call.expression.name.text === 'includes' &&
        isExplicitFormatList(call.expression.expression) &&
        containsFormatReference(call, value)
      );
    }
    return false;
  };
  const checksFormat = (statement, value) => {
    if (getTupleFormatChecks(statement).some(checked => sameFormatReference(checked, value)))
      return true;
    if (typeScript.isIfStatement(statement)) {
      const refusal = typeScript.isBlock(statement.thenStatement)
        ? statement.thenStatement.statements.at(-1)
        : statement.thenStatement;
      return (
        refusal &&
        typeScript.isThrowStatement(refusal) &&
        rejectsWrongFormat(statement.expression, value)
      );
    }
    if (!typeScript.isExpressionStatement(statement)) return false;
    const call = unwrapFormatExpression(statement.expression);
    if (!typeScript.isCallExpression(call)) return false;
    const name = validatorName(call.expression);
    if (['validateSinglePackedView', 'validatePackedView'].includes(name))
      return (
        sameFormatReference(call.arguments[0], value) && isExplicitFormatList(call.arguments[1])
      );
    if (name === 'validatePackedUint32View') return sameFormatReference(call.arguments[0], value);
    if (name === 'validateRequirement' && call.arguments[1]) {
      const condition = unwrapFormatExpression(call.arguments[1]);
      if (
        !typeScript.isCallExpression(condition) ||
        !typeScript.isPropertyAccessExpression(condition.expression)
      )
        return false;
      const predicate = condition.expression;
      if (
        predicate.name.text === 'test' &&
        typeScript.isRegularExpressionLiteral(predicate.expression)
      )
        return containsFormatReference(condition, value);
      if (predicate.name.text === 'includes' && isExplicitFormatList(predicate.expression))
        return containsFormatReference(condition, value);
    }
    if (name === 'validateGraphPorts') {
      const ports = resolveDeclaration(call.arguments[1]);
      const descriptors = resolveDeclaration(call.arguments[2]);
      if (
        !ports ||
        !descriptors ||
        !typeScript.isObjectLiteralExpression(ports) ||
        !typeScript.isObjectLiteralExpression(descriptors)
      )
        return false;
      return ports.properties.some(port => {
        if (!typeScript.isPropertyAssignment(port) || !sameFormatReference(port.initializer, value))
          return false;
        const descriptor = resolveDeclaration(
          getPropertyValue(getProperty(descriptors, port.name.getText().replace(/['"]/g, '')))
        );
        return (
          descriptor &&
          typeScript.isObjectLiteralExpression(descriptor) &&
          isExplicitFormatList(getPropertyValue(getProperty(descriptor, 'formats')))
        );
      });
    }
    return false;
  };
  const getDeclaredCollectionKeys = expression => {
    const reference = formatReference(expression);
    if (!reference) return null;
    let type = reference.root.type;
    const resolveShape = node => {
      const seen = new Set();
      while (node && !seen.has(node)) {
        seen.add(node);
        if (typeScript.isTypeLiteralNode(node) || typeScript.isInterfaceDeclaration(node))
          return node;
        if (typeScript.isTypeAliasDeclaration(node) || typeScript.isParenthesizedTypeNode(node))
          node = node.type;
        else if (typeScript.isTypeReferenceNode(node)) node = resolvePublicType(node.typeName);
        else return null;
      }
      return null;
    };
    for (const part of reference.parts) {
      const shape = resolveShape(type);
      type = shape?.members.find(
        member => member.name?.getText().replace(/['"]/g, '') === part
      )?.type;
    }
    const shape = resolveShape(type);
    if (
      !shape ||
      shape.members.some(member => !typeScript.isPropertySignature(member) || !member.name)
    )
      return null;
    return shape.members.map(member => member.name.getText().replace(/['"]/g, ''));
  };
  const forEveryFormatCaller = (site, parameter, check, visiting) => {
    const function_ = getEnclosingFunction(site);
    const index = function_?.parameters.indexOf(parameter) ?? -1;
    const callers = (function_ && formatCallsByTarget.get(function_)) || [];
    return (
      index >= 0 &&
      callers.length > 0 &&
      callers.every(call => call.arguments[index] && check(call, call.arguments[index], visiting))
    );
  };
  const pendingFormatCollections = new Set();
  const hasCheckedFormatCollection = (site, collection, visiting) => {
    collection = unwrapFormatExpression(collection);
    if (!collection || pendingFormatCollections.has(collection)) return false;
    pendingFormatCollections.add(collection);
    try {
      if (typeScript.isArrayLiteralExpression(collection))
        return (
          collection.elements.length > 0 &&
          collection.elements.every(element =>
            typeScript.isSpreadElement(element)
              ? hasCheckedFormatCollection(site, element.expression, visiting)
              : hasPriorFormatCheck(site, element, visiting)
          )
        );
      if (
        typeScript.isCallExpression(collection) &&
        typeScript.isPropertyAccessExpression(collection.expression) &&
        collection.expression.expression.getText() === 'Object' &&
        collection.expression.name.text === 'values' &&
        collection.arguments[0]
      ) {
        const object = collection.arguments[0];
        const keys = getDeclaredCollectionKeys(object);
        const reference = formatReference(object);
        return (
          keys &&
          keys.length > 0 &&
          keys.every(key =>
            hasPriorFormatCheck(site, {...reference, parts: [...reference.parts, key]}, visiting)
          )
        );
      }
      const reference = formatReference(collection);
      if (!reference || reference.parts.length || !typeScript.isParameter(reference.root))
        return false;
      return forEveryFormatCaller(site, reference.root, hasCheckedFormatCollection, visiting);
    } finally {
      pendingFormatCollections.delete(collection);
    }
  };
  const hasPriorFormatCheck = (site, value, visiting = new Set()) => {
    if (!site || visiting.has(site)) return false;
    const nextVisiting = new Set(visiting).add(site);
    for (let parent = site.parent, child = site; parent; child = parent, parent = parent.parent) {
      if (typeScript.isBlock(parent) || typeScript.isSourceFile(parent)) {
        const previous = [...parent.statements].filter(
          statement => statement !== child && statement.getStart() < child.getStart()
        );
        for (const statement of previous.reverse()) {
          if (writesFormatReference(statement, value)) return false;
          if (checksFormat(statement, value)) return true;
        }
      }
      if (typeScript.isFunctionLike(parent)) break;
    }
    const reference = formatReference(value);
    if (!reference) return false;
    const loop = getFormatIteration(value);
    if (loop && !sameFormatReference(loop.expression, value))
      return hasCheckedFormatCollection(site, loop.expression, nextVisiting);
    if (!typeScript.isParameter(reference.root)) return false;
    return forEveryFormatCaller(
      site,
      reference.root,
      (call, argument, seen) => {
        const actual = formatReference(argument);
        return (
          actual &&
          hasPriorFormatCheck(call, {...actual, parts: [...actual.parts, ...reference.parts]}, seen)
        );
      },
      nextVisiting
    );
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
          for (const [parameter, value] of previousParameters)
            parameterValues.set(parameter, value);
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
    if (!matchesSelectedPath(file)) continue;
    const text = fileTexts.get(file);
    const lines = text.split('\n');

    // 1. bare-assert: assert imported from @luma.gl/core, under any local name or through a
    // namespace import, and called (TOOL-117: `import {assert as check}` used to pass).
    const assertBindings = new Set();
    const assertNamespaces = new Set();
    for (const statement of sourceTrees.get(file).statements) {
      if (
        !typeScript.isImportDeclaration(statement) ||
        !typeScript.isStringLiteral(statement.moduleSpecifier) ||
        statement.moduleSpecifier.text !== '@luma.gl/core'
      )
        continue;
      const bindings = statement.importClause?.namedBindings;
      if (bindings && typeScript.isNamespaceImport(bindings)) assertNamespaces.add(bindings);
      else if (bindings)
        for (const element of bindings.elements) {
          if ((element.propertyName ?? element.name).text === 'assert') assertBindings.add(element);
        }
    }
    const assertLines = [];
    visitNodes(sourceTrees.get(file), node => {
      if (!typeScript.isCallExpression(node)) return;
      let callee = node.expression;
      while (isExpressionWrapper(callee)) callee = callee.expression;
      const named =
        typeScript.isIdentifier(callee) && assertBindings.has(findBinding(callee, true));
      const namespace =
        (typeScript.isPropertyAccessExpression(callee) ||
          typeScript.isElementAccessExpression(callee)) &&
        (typeScript.isPropertyAccessExpression(callee)
          ? callee.name.text === 'assert'
          : callee.argumentExpression &&
            typeScript.isStringLiteralLike(callee.argumentExpression) &&
            callee.argumentExpression.text === 'assert') &&
        typeScript.isIdentifier(callee.expression) &&
        assertNamespaces.has(findBinding(callee.expression, true));
      if (named || namespace) assertLines.push(getLineNumber(text, callee.getStart()));
    });
    if (assertLines.length)
      addFinding('bare-assert', file, {count: assertLines.length, lines: assertLines});

    const selfFormatLines = [];
    if (checksSelfFormats)
      visitNodes(sourceTrees.get(file), node => {
        if (
          typeScript.isCallExpression(node) &&
          getSelfFormatSource(node) &&
          !hasPriorFormatCheck(node, getSelfFormatSource(node))
        )
          selfFormatLines.push(getLineNumber(text, node.getStart()));
      });
    if (selfFormatLines.length)
      addFinding('self-format-validation', file, {
        count: selfFormatLines.length,
        lines: selfFormatLines
      });

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

    // An explicitly selected rule needs only its own diagnostics. Cross-file source trees and
    // resolvers stay complete above, so selected files still follow imported kernel helpers.
    if (
      !selectedRule ||
      [
        'single-invocation-data-loop',
        'shader-analysis-limit',
        'duplicate-wgsl-declaration'
      ].includes(selectedRule)
    ) {
      // 3. single-invocation-data-loop: all WGSL loop forms, including referenced helper snippets.
      // Bounded dense loops remain reviewable heuristics with an explicit contract-allow comment.
      // The comment must give its bound on the marker line or the comment lines directly above it,
      // and every named bound it cites must still hold (see getCitedBoundProblems, CORE-118).
      const invocationLines = [];
      const invocationDescriptions = [];
      const analysisLimitLines = [];
      const duplicateLinesByName = new Map();
      const checkKernelCallSite = ({call, callSite}) => {
        const properties = resolveDeclaration(call.arguments[1]);
        if (!properties || !typeScript.isObjectLiteralExpression(properties)) return;
        const {guards: callGuards} = getExecutionGuards(call);
        const siteGuards = callSite === call ? [] : getExecutionGuards(callSite).guards;
        const callPaths = applyExecutionGuards(
          [createTextVariant('')],
          [...siteGuards, ...callGuards]
        );
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
        const constants = resolveDeclaration(
          getPropertyValue(getProperty(properties, 'constants'))
        );
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
        if (limitedProperties.length)
          analysisLimitLines.push(getLineNumber(text, callSite.getStart()));
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
          if (hasLoop && !hasQualifiedLoopAllowance(file, lines, line)) {
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
    }

    // 5. luma-reexport.
    const reexportLines = findLumaReExportSites(text, file).map(site => site.line);
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
    if (STRICT_RULES.has(finding.rule))
      return !hasReviewedLegacyInternalAllowance(finding, baseline);
    if (baseline._version !== 2) return true;
    const recorded = baseline[finding.rule]?.[finding.file];
    return (
      !Array.isArray(recorded) ||
      finding.sites.some(
        site =>
          !recorded.some(
            entry =>
              entry.id === site.id &&
              typeof entry.reason === 'string' &&
              entry.reason.trim() &&
              !getAllowanceBoundProblems(finding, entry).length
          )
      )
    );
  };
  // A loop allowance is renewed only with the bounds its reason cites (CORE-118).
  const getAllowanceBoundProblems = (finding, entry) =>
    finding.rule === 'single-invocation-data-loop'
      ? getCitedBoundProblems(finding.file, entry.reason)
      : [];

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
      finding =>
        (!selectedRule || finding.rule === selectedRule) && matchesSelectedPath(finding.file)
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
    const recorded = baseline[finding.rule]?.[finding.file];
    for (const entry of Array.isArray(recorded) ? recorded : []) {
      if (!finding.sites.some(site => site.id === entry.id)) continue;
      for (const problem of getAllowanceBoundProblems(finding, entry)) {
        process.stdout.write(`  STALE BOUND ${entry.id}: ${problem}\n`);
      }
    }
  }
  if (newFindings.length || staleCount) {
    process.stdout.write(`${newFindings.length} new contributor-contract findings\n`);
    process.exitCode = 1;
  } else {
    process.stdout.write('contributor-contract: no new findings\n');
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runContributorContract();
}
