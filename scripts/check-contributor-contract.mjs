// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Report departures from dev-docs/contributor-contract.md across modules/analysis/src.
//
//   node scripts/check-contributor-contract.mjs                  new findings only; exit 1 if any
//   node scripts/check-contributor-contract.mjs --all            every finding
//   node scripts/check-contributor-contract.mjs --rule <id>      one rule only
//   node scripts/check-contributor-contract.mjs --all --paths <file-or-directory> [...]
//   node scripts/check-contributor-contract.mjs --update-baseline
//
// The baseline (scripts/contributor-contract-baseline.json) records existing findings by rule and
// file (a count, or a list of symbols), never by line number, so unrelated edits do not churn it.
// TypeScript is parsed without typechecking. WGSL checks inspect statically resolvable snippets;
// they are heuristics, not a replacement for compiling and testing the shaders.
import fs from 'node:fs';
import path from 'node:path';
import typeScript from 'typescript';

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

// These rules reached zero. Updating or hand-editing a baseline must never excuse a regression.
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

const sourceTrees = new Map(
  sourceFiles.map(file => [
    file,
    typeScript.createSourceFile(file, fileTexts.get(file), typeScript.ScriptTarget.Latest, true)
  ])
);

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
].filter(file => fileTexts.has(file));
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
const isExpressionWrapper = expression =>
  typeScript.isParenthesizedExpression(expression) ||
  typeScript.isAsExpression(expression) ||
  typeScript.isSatisfiesExpression(expression) ||
  typeScript.isTypeAssertionExpression(expression) ||
  typeScript.isNonNullExpression(expression);
const resolveDeclaration = (expression, visiting = new Set()) => {
  if (!expression || visiting.has(expression)) return null;
  visiting.add(expression);
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
const getStableConditionText = expression => {
  if (typeScript.isParenthesizedExpression(expression))
    return getStableConditionText(expression.expression);
  if (typeScript.isStringLiteralLike(expression) || typeScript.isNumericLiteral(expression)) {
    return `${expression.kind}:${expression.text}`;
  }
  if (typeScript.isPropertyAccessExpression(expression)) {
    const owner = getStableConditionText(expression.expression);
    return owner === null ? null : `${owner}.${expression.name.text}`;
  }
  if (
    typeScript.isBinaryExpression(expression) &&
    expression.operatorToken.kind < typeScript.SyntaxKind.FirstAssignment
  ) {
    const left = getStableConditionText(expression.left);
    const right = getStableConditionText(expression.right);
    return left === null || right === null
      ? null
      : `(${left} ${expression.operatorToken.kind} ${right})`;
  }
  if (!typeScript.isIdentifier(expression)) return null;
  const binding = findBinding(expression);
  if (!binding) return null;
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
const getReachableWGSLText = (declarations, body) => {
  const cleanDeclarations = maskWGSLComments(declarations);
  const functions = new Map();
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
    : declaration?.name?.text ?? expression.name?.text ?? expression.text;
};
// Named-capacity wrappers are preflights too when their accepted maximum derives from the
// device storage binding. Follow local/imported returns and const aliases, without typechecking.
const referencesStorageBindingLimit = (node, visiting = new Set()) => {
  if (!node || visiting.has(node)) return false;
  const nextVisiting = new Set(visiting).add(node);
  if (typeScript.isPropertyAccessExpression(node) && node.name.text === 'maxStorageBufferBindingSize') return true;
  if (typeScript.isCallExpression(node) &&
    ['getStorageBindingElementLimit', 'getAnalysisMaximumRowCount'].includes(getReferenceName(node.expression))) return true;
  const resolved = resolveDeclaration(node);
  if (resolved && resolved !== node && referencesStorageBindingLimit(resolved, nextVisiting)) return true;
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
  const limit = name === 'validateCapacity' ? call.arguments[4]
    : name === 'validateMaximumCount' ? call.arguments[3]
      : name === 'validateRequirement' ? call.arguments[1] : null;
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
      if (typeScript.isPrefixUnaryExpression(condition) && condition.operator === typeScript.SyntaxKind.ExclamationToken) {
        condition = condition.operand;
      }
      if (!typeScript.isIdentifier(condition) || !names.has(condition.text)) safe = false;
    }
    if (typeScript.isCallExpression(node)) {
      const name = getReferenceName(node.expression);
      if (STORAGE_DECLARATION_HELPERS.has(name) || /^(createTransientBuffer|createBuffer|createDataView|add)$/.test(name)) safe = false;
      if (name === 'validateGPUStorageBindingByteLength') {
        visitNodes(node.arguments[3], value => {
          if (typeScript.isCallExpression(value) && getReferenceName(value.expression) === 'getViewBindingRange') foundRangeCheck = true;
        });
      }
    }
    typeScript.forEachChild(node, child => { inspect(child); });
  };
  inspect(loop.statement);
  return safe && foundRangeCheck;
};

const getStoragePreflightFinding = declaration => {
  const members = getClassMembers(declaration);
  const entry = members.find(member => member.name?.getText().replace(/['"]/g, '') === 'getCommandNodes');
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
    if (typeScript.isCallExpression(expression) && typeScript.isPropertyAccessExpression(expression.expression) &&
      expression.expression.name.text === 'entries') {
      return isNonemptyIterable(expression.expression.expression, nextSeen);
    }
    const resolved = resolveDeclaration(expression);
    if (parameterValues.has(resolved)) return isNonemptyIterable(parameterValues.get(resolved), nextSeen);
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
    if (parameterValues.has(resolved)) return isCommandGraphReceiver(parameterValues.get(resolved), nextSeen);
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
    if (typeScript.isBinaryExpression(node) && [
      typeScript.SyntaxKind.AmpersandAmpersandToken,
      typeScript.SyntaxKind.BarBarToken,
      typeScript.SyntaxKind.QuestionQuestionToken
    ].includes(node.operatorToken.kind)) {
      const leftStates = analyze(node.left, states);
      return new Set([...leftStates, ...analyze(node.right, new Set(leftStates))]);
    }
    if (typeScript.isSwitchStatement(node)) {
      const conditions = analyze(node.expression, states);
      return new Set([...conditions, ...node.caseBlock.clauses.flatMap(clause =>
        [...analyzeBody(clause, new Set(conditions))])]);
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
      return isCollectionBindingPreflight(node) ? new Set([true])
        : typeScript.isForOfStatement(node) && isNonemptyIterable(node.expression)
          ? iterationStates : states;
    }
    if (typeScript.isCallExpression(node)) {
      let nextStates = states;
      for (const argument of node.arguments) nextStates = analyze(argument, nextStates);
      const name = getReferenceName(node.expression);
      if (STORAGE_PREFLIGHT_HELPERS.has(name) || isStorageCapacityCheck(node)) return new Set([true]);
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
          const invocation = properties && typeScript.isObjectLiteralExpression(properties)
            ? getPropertyValue(getProperty(properties, 'invocationCount'))
            : null;
          // A one-invocation reference kernel may still bind a row-sized input. Without a
          // type checker, any caller-provided binding array is potentially row-proportional.
          hasRowStorage ||= getStaticNumber(invocation) === undefined ||
            Boolean(properties && typeScript.isObjectLiteralExpression(properties) &&
              getProperty(properties, 'bindings'));
        }
        return nextStates;
      }
      let helper = resolveDeclaration(node.expression);
      if (typeScript.isPropertyAccessExpression(node.expression) &&
        node.expression.expression.kind === typeScript.SyntaxKind.ThisKeyword) {
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
    typeScript.forEachChild(node, child => { nextStates = analyzeBody(child, nextStates); });
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
  const analysisLimitLines = [];
  const duplicateLinesByName = new Map();
  visitNodes(sourceTree, call => {
    if (
      !typeScript.isCallExpression(call) ||
      !isNamedReference(call.expression, 'createWGSLKernelNode')
    )
      return;
    const properties = resolveDeclaration(call.arguments[1]);
    if (!properties || !typeScript.isObjectLiteralExpression(properties)) return;
    const {guards: callGuards} = getExecutionGuards(call);
    const callPaths = applyExecutionGuards([createTextVariant('')], callGuards);
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
    if (limitedProperties.length) analysisLimitLines.push(getLineNumber(text, call.getStart()));
    if (getStaticNumber(getPropertyValue(invocation)) === 1) {
      const line = getLineNumber(text, invocation.getStart());
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
      linesForName.push(getLineNumber(text, call.getStart()));
      duplicateLinesByName.set(name, linesForName);
    }
  });
  if (invocationLines.length) {
    addFinding('single-invocation-data-loop', file, {
      count: invocationLines.length,
      lines: invocationLines
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
    if (
      !internalSymbols.has(`${file}#${name}`) &&
      !new RegExp(`\\b${name}\\b`).test(ontologyText)
    ) {
      addFinding('unregistered-contributor', file, {
        symbol: name,
        lines: [getLineNumber(text, offset)]
      });
    }
  }
}

// Baseline handling: { rule: { file: count | [symbols] }, _reasons?: { rule: { file: reason } }.
// Reasons document newly discovered legacy findings and survive baseline regeneration.
const buildBaseline = foundList => {
  const baseline = {};
  for (const finding of [...foundList].sort(compareFindings)) {
    baseline[finding.rule] ??= {};
    if (COUNT_RULES.has(finding.rule)) baseline[finding.rule][finding.file] = finding.count;
    else {
      baseline[finding.rule][finding.file] ??= [];
      baseline[finding.rule][finding.file].push(finding.symbol);
    }
  }
  const previousReasons = loadBaseline()._reasons ?? {};
  for (const [rule, entries] of Object.entries(previousReasons)) {
    for (const [file, reason] of Object.entries(entries)) {
      if (baseline[rule]?.[file] !== undefined) {
        baseline._reasons ??= {};
        baseline._reasons[rule] ??= {};
        baseline._reasons[rule][file] = reason;
      }
    }
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
const isNewFinding = (finding, baseline) => {
  if (STRICT_RULES.has(finding.rule)) return true;
  const recorded = baseline[finding.rule]?.[finding.file];
  if (recorded === undefined) return true;
  return COUNT_RULES.has(finding.rule)
    ? finding.count > recorded
    : !recorded.includes(finding.symbol);
};

// Command line.
const arguments_ = process.argv.slice(2);
const ruleIndex = arguments_.indexOf('--rule');
const selectedRule = ruleIndex >= 0 ? arguments_[ruleIndex + 1] : null;
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
if (selectedPaths.length && arguments_.includes('--update-baseline')) {
  process.stderr.write('--paths cannot be combined with --update-baseline; update the complete baseline explicitly\n');
  process.exit(2);
}
const matchesSelectedPath = file => !selectedPaths.length || selectedPaths.some(selected =>
  selected === '' || file === selected || file.startsWith(`${selected}${path.sep}`));

if (selectedRule && !RULE_IDENTIFIERS.includes(selectedRule)) {
  process.stderr.write(`Unknown rule '${selectedRule}'. Rules: ${RULE_IDENTIFIERS.join(', ')}\n`);
  process.exit(2);
}

if (arguments_.includes('--update-baseline')) {
  const strictFindings = findings.filter(finding => STRICT_RULES.has(finding.rule));
  if (strictFindings.length) {
    process.stderr.write(
      `Rules with zero baselines cannot be baselined: ${[...new Set(strictFindings.map(finding => finding.rule))].join(', ')}\n`
    );
    process.exit(1);
  }
  const baseline = buildBaseline(findings);
  fs.writeFileSync(BASELINE_PATH, `${JSON.stringify(baseline, null, 2)}\n`);
  process.stdout.write(`Wrote ${BASELINE_PATH} with ${findings.length} findings.\n`);
  process.exit(0);
}

const baseline = loadBaseline();
const shownFindings = findings
  .filter(finding => (!selectedRule || finding.rule === selectedRule) && matchesSelectedPath(finding.file))
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
  process.stdout.write(`${marker}${finding.rule} ${finding.file}:${finding.lines[0]}${detail}\n`);
}
if (newFindings.length) {
  process.stdout.write(`${newFindings.length} new contributor-contract findings\n`);
  process.exitCode = 1;
} else {
  process.stdout.write('contributor-contract: no new findings\n');
}
