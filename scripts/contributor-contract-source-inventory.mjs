// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import path from 'node:path';
import typeScript from 'typescript';
import {createPublicDeclarationInventory} from './public-declaration-inventory.mjs';

/** Resolve source modules without loading dependencies or creating a TypeScript program. */
export function createSourceResolvers(sourceTrees) {
  // Preserve authored module formats; .ts is the repository's generated CommonJS fallback.
  const resolveSourceFile = base => {
    const suffix = /(?:\.d)?\.[cm]?[jt]sx?$/.exec(base)?.[0];
    const stem = suffix ? base.slice(0, -suffix.length) : base;
    const extensions = {
      '.cjs': ['.cts', '.d.cts', '.cjs', '.ts', '.tsx', '.d.ts', '.js'],
      '.mjs': ['.mts', '.d.mts', '.mjs', '.ts', '.tsx', '.d.ts', '.js'],
      '.js': ['.ts', '.tsx', '.d.ts', '.js'],
      '.d.ts': ['.d.ts', '.ts', '.tsx', '.js'],
      '.d.cts': ['.d.cts', '.cts', '.ts'],
      '.d.mts': ['.d.mts', '.mts', '.ts']
    };
    if (suffix)
      return (
        (extensions[suffix] ?? [suffix])
          .map(extension => stem + extension)
          .find(candidate => sourceTrees.has(candidate)) ?? null
      );
    const candidates = ['.ts', '.tsx', '.mts', '.cts', '.d.ts', '.js', '.mjs', '.cjs'];
    return (
      candidates
        .flatMap(extension => [base + extension, path.join(base, `index${extension}`)])
        .find(candidate => sourceTrees.has(candidate)) ?? null
    );
  };
  const resolveModule = (fromFile, specifier) =>
    specifier.startsWith('.')
      ? resolveSourceFile(path.join(path.dirname(fromFile), specifier))
      : null;

  return {resolveSourceFile, resolveModule};
}

/**
 * Development-only source inventory. All consumers share manifest selection, export spelling
 * and defining origin; admission (props, packers, contributors, internal members) stays local.
 * Resolvers are supplied so fixtures and compiler-backed docs can use the very same walk.
 */
export function createPublicSourceInventory({
  analysisManifest,
  sourceRoot: SOURCE_ROOT,
  sourceTrees,
  resolveModule,
  resolveSourceFile
}) {
  const sourceFiles = [...sourceTrees.keys()];
  const packageName = analysisManifest.name ?? '@luma-contributors/analysis';
  const SOURCE_SUFFIX_PATTERN = /\.(?:d\.[cm]?ts|[cm]?[jt]s|[jt]sx)$/;
  const collectExportTargets = (_entry, target) =>
    typeof target === 'string'
      ? [target]
      : target && typeof target === 'object'
        ? Object.values(target).flatMap(value => collectExportTargets(_entry, value))
        : [];
  const createSourceTargetPattern = target => {
    const [prefix, ...suffixes] = target
      .slice('./dist/'.length)
      .replace(SOURCE_SUFFIX_PATTERN, '')
      .split('*')
      .map(part => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    return new RegExp(
      suffixes.length ? `^${prefix}(?<subpath>.*)${suffixes.join('\\k<subpath>')}$` : `^${prefix}$`
    );
  };
  // Resolve one exported spelling at a time, as ECMAScript ResolveExport does. A recursive
  // cycle has no answer on that path; it must not become a cached module-wide partial map.
  const AMBIGUOUS_EXPORT = Symbol('ambiguous export');
  const modules = new Map();
  const getModule = file => {
    if (modules.has(file)) return modules.get(file);
    const imports = new Map();
    const explicit = new Map();
    const stars = [];
    for (const statement of sourceTrees.get(file).statements) {
      if (
        typeScript.isImportDeclaration(statement) &&
        typeScript.isStringLiteral(statement.moduleSpecifier)
      ) {
        const specifier = statement.moduleSpecifier.text;
        const target = resolveModule(file, specifier);
        const clause = statement.importClause;
        if (clause?.name) imports.set(clause.name.text, {target, specifier, name: 'default'});
        const bindings = clause?.namedBindings;
        if (bindings && typeScript.isNamespaceImport(bindings))
          imports.set(bindings.name.text, {target, specifier});
        else if (bindings)
          for (const item of bindings.elements)
            imports.set(item.name.text, {
              target,
              specifier,
              name: (item.propertyName ?? item.name).text
            });
      }
      if (
        statement.modifiers?.some(modifier => modifier.kind === typeScript.SyntaxKind.ExportKeyword)
      ) {
        const isDefault = statement.modifiers.some(
          modifier => modifier.kind === typeScript.SyntaxKind.DefaultKeyword
        );
        const declarations = typeScript.isVariableStatement(statement)
          ? statement.declarationList.declarations
          : [statement];
        for (const declaration of declarations) {
          const name =
            declaration.name && typeScript.isIdentifier(declaration.name)
              ? declaration.name.text
              : isDefault
                ? 'default'
                : undefined;
          if (name)
            explicit.set(isDefault ? 'default' : name, {kind: 'origin', origin: `${file}#${name}`});
        }
      }
      if (typeScript.isExportAssignment(statement))
        explicit.set('default', {kind: 'expression', expression: statement.expression});
      if (!typeScript.isExportDeclaration(statement)) continue;
      const specifier = statement.moduleSpecifier?.text;
      const target = specifier ? resolveModule(file, specifier) : undefined;
      if (!statement.exportClause) {
        if (target) stars.push(target);
      } else if (typeScript.isNamespaceExport(statement.exportClause)) {
        explicit.set(statement.exportClause.name.text, {kind: 'namespace', target, specifier});
      } else
        for (const item of statement.exportClause.elements) {
          explicit.set(item.name.text, {
            kind: 'reference',
            name: (item.propertyName ?? item.name).text,
            target,
            specifier
          });
        }
    }
    const module = {imports, explicit, stars};
    modules.set(file, module);
    return module;
  };
  const unwrapExpression = expression => {
    let current = expression;
    while (
      current &&
      (typeScript.isParenthesizedExpression(current) ||
        typeScript.isAsExpression(current) ||
        typeScript.isSatisfiesExpression(current) ||
        typeScript.isTypeAssertionExpression(current) ||
        typeScript.isNonNullExpression(current))
    )
      current = current.expression;
    return current;
  };
  const getExpressionMember = expression =>
    typeScript.isPropertyAccessExpression(expression)
      ? expression.name.text
      : typeScript.isElementAccessExpression(expression) &&
          expression.argumentExpression &&
          typeScript.isStringLiteralLike(expression.argumentExpression)
        ? expression.argumentExpression.text
        : undefined;
  const selectNames = (names, name) =>
    [...names].flatMap(candidate =>
      candidate === name
        ? ['']
        : candidate.startsWith(`${name}.`)
          ? [candidate.slice(name.length)]
          : []
    );
  const getLocalNames = (file, name, visiting) => {
    const imported = getModule(file).imports.get(name);
    if (!imported) return [''];
    if (!imported.target) return imported.name ? [''] : [''];
    const names = getExportNames(imported.target, visiting);
    return imported.name
      ? selectNames(names, imported.name)
      : [...names].map(candidate => `.${candidate}`);
  };
  const getExpressionNames = (file, input, visiting) => {
    const expression = unwrapExpression(input);
    if (expression && typeScript.isIdentifier(expression))
      return getLocalNames(file, expression.text, visiting);
    const member = expression && getExpressionMember(expression);
    return member === undefined
      ? []
      : selectNames(getExpressionNames(file, expression.expression, visiting), `.${member}`);
  };
  const getExportNames = (file, visiting = new Set()) => {
    if (visiting.has(file)) return new Set();
    const next = new Set(visiting).add(file);
    const {explicit, stars} = getModule(file);
    const names = new Set();
    for (const [name, descriptor] of explicit) {
      let suffixes = [''];
      if (descriptor.kind === 'namespace')
        suffixes = descriptor.target
          ? [...getExportNames(descriptor.target, next)].map(candidate => `.${candidate}`)
          : [''];
      if (descriptor.kind === 'reference')
        suffixes = descriptor.specifier
          ? descriptor.target
            ? selectNames(getExportNames(descriptor.target, next), descriptor.name)
            : ['']
          : getLocalNames(file, descriptor.name, next);
      if (descriptor.kind === 'expression')
        suffixes = getExpressionNames(file, descriptor.expression, next);
      for (const suffix of suffixes) names.add(name + suffix);
    }
    for (const target of stars)
      for (const name of getExportNames(target, next))
        if (name !== 'default' && !name.startsWith('default.')) names.add(name);
    return names;
  };
  const resolveLocal = (file, name, suffix, resolving) => {
    const imported = getModule(file).imports.get(name);
    if (!imported) return suffix ? undefined : `${file}#${name}`;
    if (!imported.target)
      return suffix ? undefined : imported.name ? `${file}#${name}` : `${imported.specifier}#*`;
    return imported.name
      ? resolveExport(imported.target, imported.name + suffix, resolving)
      : suffix
        ? resolveExport(imported.target, suffix.slice(1), resolving)
        : undefined;
  };
  const resolveExpression = (file, input, suffix, resolving) => {
    const expression = unwrapExpression(input);
    if (expression && typeScript.isIdentifier(expression))
      return resolveLocal(file, expression.text, suffix, resolving);
    const member = expression && getExpressionMember(expression);
    return member === undefined
      ? undefined
      : resolveExpression(file, expression.expression, `.${member}${suffix}`, resolving);
  };
  const resolveExport = (file, name, resolving = new Set()) => {
    const key = `${file}\0${name}`;
    if (resolving.has(key)) return undefined;
    const next = new Set(resolving).add(key);
    const {explicit, stars} = getModule(file);
    const explicitName = [...explicit.keys()]
      .filter(candidate => candidate === name || name.startsWith(`${candidate}.`))
      .sort((left, right) => right.length - left.length)[0];
    if (explicitName !== undefined) {
      const descriptor = explicit.get(explicitName);
      const suffix = name.slice(explicitName.length);
      if (descriptor.kind === 'origin') return suffix ? undefined : descriptor.origin;
      if (descriptor.kind === 'expression')
        return resolveExpression(file, descriptor.expression, suffix, next);
      if (descriptor.kind === 'namespace')
        return descriptor.target
          ? suffix
            ? resolveExport(descriptor.target, suffix.slice(1), next)
            : undefined
          : suffix
            ? undefined
            : `${descriptor.specifier}#*`;
      if (descriptor.specifier)
        return descriptor.target
          ? resolveExport(descriptor.target, descriptor.name + suffix, next)
          : suffix
            ? undefined
            : `${descriptor.specifier}#${descriptor.name}`;
      return resolveLocal(file, descriptor.name, suffix, next);
    }
    if (name === 'default' || name.startsWith('default.')) return undefined;
    let origin;
    for (const target of stars) {
      const candidate = resolveExport(target, name, next);
      if (
        candidate === AMBIGUOUS_EXPORT ||
        (candidate !== undefined && origin !== undefined && candidate !== origin)
      )
        return AMBIGUOUS_EXPORT;
      if (candidate !== undefined) origin = candidate;
    }
    return origin;
  };
  const exportCache = new Map();
  const getExports = file => {
    if (!exportCache.has(file)) {
      const exports = new Map();
      for (const name of getExportNames(file)) {
        const origin = resolveExport(file, name);
        if (typeof origin === 'string') exports.set(name, origin);
      }
      exportCache.set(file, exports);
    }
    return exportCache.get(file);
  };
  // Every concrete manifest target is a public surface, including nested conditional patterns.
  // Root-file exceptions apply only to placement; they do not hide public exports from this rule.
  const getWinningExport = subpath => {
    if (Object.hasOwn(analysisManifest.exports, subpath)) return subpath;
    return Object.keys(analysisManifest.exports)
      .filter(entry => {
        if (!entry.includes('*')) return false;
        const [prefix, suffix] = entry.split('*');
        return (
          subpath.startsWith(prefix) &&
          subpath.endsWith(suffix) &&
          subpath.length >= prefix.length + suffix.length
        );
      })
      .sort(
        (left, right) => right.indexOf('*') - left.indexOf('*') || right.length - left.length
      )[0];
  };
  const publicSourceTargets = new Map();
  for (const [entry, exported] of Object.entries(analysisManifest.exports)) {
    for (const target of collectExportTargets(entry, exported)) {
      if (!target.startsWith('./dist/') || !SOURCE_SUFFIX_PATTERN.test(target)) continue;
      const pattern = createSourceTargetPattern(target);
      for (const file of sourceFiles) {
        const match = pattern.exec(
          path
            .relative(SOURCE_ROOT, file)
            .split(path.sep)
            .join('/')
            .replace(SOURCE_SUFFIX_PATTERN, '')
        );
        if (!match) continue;
        const subpath = entry.replaceAll('*', match.groups?.subpath ?? '');
        if (getWinningExport(subpath) !== entry) continue;
        const concreteTarget = target.replaceAll('*', match.groups?.subpath ?? '');
        if (
          resolveSourceFile(path.join(SOURCE_ROOT, concreteTarget.slice('./dist/'.length))) !== file
        )
          continue;
        const entryPoint =
          packageName +
          (entry === '.' ? '' : entry.slice(1).replaceAll('*', match.groups?.subpath ?? ''));
        if (!publicSourceTargets.has(file)) publicSourceTargets.set(file, new Set());
        publicSourceTargets.get(file).add(entryPoint);
      }
    }
  }
  const barrelFiles = [...publicSourceTargets.keys()].sort();

  const inventory = createPublicDeclarationInventory(
    [...publicSourceTargets].flatMap(([file, entries]) =>
      [...entries].map(entryPoint => ({entryPoint, sourceTarget: file, exports: getExports(file)}))
    )
  );
  return {inventory, publicSourceTargets, barrelFiles, getExports};
}
