// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import typeScript from 'typescript';

// A deliberately bounded static evaluator for catalog construction, never an execution of source
// modules. Only values returned/emitted by public catalog roots register identities. Reading an
// array (notably via find) does not emit its other elements. Unsupported values remain unknown.
export function collectCatalogIdentities(trees, resolveModule) {
  const unknown = Symbol('unknown catalog value');
  const invalidClone = Symbol('missing cloned catalog record');
  const contexts = new Map();
  const cache = new Map();
  const active = new Set();
  const spatialPackage = '@luma-contributors/analysis/gpu-spatial-analysis';
  const families = new Set([
    'GPU_SPATIAL_ANALYSIS_CAPABILITIES',
    'ENTRY_POINT_CAPABILITIES',
    'REUSED_ANALYSIS_OPERATORS',
    'SPATIOTEMPORAL_REVIEW_OPERATORS',
    'FITTING_KERNEL_OPERATORS'
  ]);
  const unwrap = expression => {
    let node = expression;
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
  for (const tree of trees) {
    const declarations = new Map();
    const imports = new Map();
    const exported = new Set();
    for (const statement of tree.statements) {
      if (
        typeScript.isImportDeclaration(statement) &&
        typeScript.isStringLiteral(statement.moduleSpecifier)
      ) {
        const bindings = statement.importClause?.namedBindings;
        if (bindings && typeScript.isNamedImports(bindings))
          for (const binding of bindings.elements)
            imports.set(binding.name.text, {
              file: resolveModule(tree.fileName, statement.moduleSpecifier.text),
              name: (binding.propertyName ?? binding.name).text
            });
      }
      const entries = typeScript.isVariableStatement(statement)
        ? statement.declarationList.declarations
        : [statement];
      for (const declaration of entries)
        if (declaration.name && typeScript.isIdentifier(declaration.name)) {
          declarations.set(declaration.name.text, declaration);
          if (
            statement.modifiers?.some(
              modifier => modifier.kind === typeScript.SyntaxKind.ExportKeyword
            )
          )
            exported.add(declaration.name.text);
        }
      if (
        typeScript.isExportDeclaration(statement) &&
        !statement.moduleSpecifier &&
        statement.exportClause &&
        typeScript.isNamedExports(statement.exportClause)
      )
        for (const binding of statement.exportClause.elements)
          exported.add((binding.propertyName ?? binding.name).text);
    }
    contexts.set(tree.fileName, {tree, declarations, imports, exported});
  }
  const property = (value, key) =>
    value !== unknown &&
    value !== null &&
    value !== undefined &&
    typeof key === 'string' &&
    Object.hasOwn(Object(value), key)
      ? value[key]
      : undefined;
  const bind = (name, value, environment) => {
    if (typeScript.isIdentifier(name)) environment.set(name.text, value);
    else if (typeScript.isArrayBindingPattern(name))
      name.elements.forEach((element, index) => {
        if (typeScript.isBindingElement(element))
          bind(element.name, Array.isArray(value) ? value[index] : unknown, environment);
      });
    else if (typeScript.isObjectBindingPattern(name))
      for (const element of name.elements)
        bind(
          element.name,
          property(value, (element.propertyName ?? element.name).getText()),
          environment
        );
  };
  const lookup = (name, context, environment) => {
    if (environment.has(name)) return environment.get(name);
    const declaration = context.declarations.get(name);
    if (declaration) {
      if (typeScript.isFunctionDeclaration(declaration))
        return {functionNode: declaration, context, environment};
      if (cache.has(declaration)) return cache.get(declaration);
      if (active.has(declaration)) return unknown;
      active.add(declaration);
      const value = evaluate(declaration.initializer, context, new Map());
      active.delete(declaration);
      cache.set(declaration, value);
      return value;
    }
    const imported = context.imports.get(name);
    const target = imported && contexts.get(imported.file);
    return target
      ? lookup(imported.name, target, new Map())
      : name === 'undefined'
        ? undefined
        : unknown;
  };
  const invoke = (callable, arguments_, depth = 0) => {
    if (!callable?.functionNode || depth > 64) return unknown;
    const {functionNode, context} = callable;
    // These repository helpers preserve identity. Their extra fields/freezing are immaterial to
    // registration. The allowlist is tied to the defining module, not an arbitrary local name.
    if (
      (context.tree.fileName.endsWith('/freeze-catalog.ts') &&
        functionNode.name?.text === 'freezeCatalog') ||
      (context.tree.fileName.endsWith('/resolve-operator-descriptor.ts') &&
        ['resolveOperatorDescriptor', 'projectSpatialOperator'].includes(functionNode.name?.text))
    )
      return arguments_[0];
    const environment = new Map(callable.environment);
    functionNode.parameters.forEach((parameter, index) =>
      bind(
        parameter.name,
        arguments_[index] === undefined && parameter.initializer
          ? evaluate(parameter.initializer, context, environment, depth + 1)
          : arguments_[index],
        environment
      )
    );
    if (!typeScript.isBlock(functionNode.body))
      return evaluate(functionNode.body, context, environment, depth + 1);
    for (const statement of functionNode.body.statements) {
      if (typeScript.isVariableStatement(statement))
        for (const declaration of statement.declarationList.declarations)
          bind(
            declaration.name,
            evaluate(declaration.initializer, context, environment, depth + 1),
            environment
          );
      if (typeScript.isReturnStatement(statement))
        return evaluate(statement.expression, context, environment, depth + 1);
      if (
        typeScript.isIfStatement(statement) &&
        typeScript.isReturnStatement(statement.thenStatement)
      ) {
        const condition = evaluate(statement.expression, context, environment, depth + 1);
        if (condition === unknown) return unknown;
        if (condition)
          return evaluate(statement.thenStatement.expression, context, environment, depth + 1);
      }
    }
    return unknown;
  };
  const evaluate = (expression, context, environment, depth = 0) => {
    const node = unwrap(expression);
    if (!node || depth > 64) return unknown;
    const read = child => evaluate(child, context, environment, depth + 1);
    if (typeScript.isStringLiteralLike(node) || typeScript.isNumericLiteral(node))
      return typeScript.isNumericLiteral(node) ? Number(node.text) : node.text;
    if (node.kind === typeScript.SyntaxKind.TrueKeyword) return true;
    if (node.kind === typeScript.SyntaxKind.FalseKeyword) return false;
    if (node.kind === typeScript.SyntaxKind.NullKeyword) return null;
    if (typeScript.isIdentifier(node)) return lookup(node.text, context, environment);
    if (typeScript.isArrowFunction(node) || typeScript.isFunctionExpression(node))
      return {functionNode: node, context, environment};
    if (typeScript.isArrayLiteralExpression(node)) {
      const result = [];
      for (const element of node.elements) {
        if (typeScript.isSpreadElement(element)) {
          const value = read(element.expression);
          if (!Array.isArray(value)) return unknown;
          result.push(...value);
        } else result.push(read(element));
      }
      return result;
    }
    if (typeScript.isObjectLiteralExpression(node)) {
      const result = Object.create(null);
      for (const entry of node.properties) {
        if (typeScript.isSpreadAssignment(entry)) {
          const value = read(entry.expression);
          const spread = unwrap(entry.expression);
          if (
            (value === undefined || value === unknown) &&
            typeScript.isCallExpression(spread) &&
            typeScript.isPropertyAccessExpression(spread.expression) &&
            spread.expression.name.text === 'find'
          )
            result[invalidClone] = true;
          if (value === unknown) {
            result.exportName = unknown;
            result.package = unknown;
          } else if (value && typeof value === 'object') Object.assign(result, value);
        } else if (entry.name) {
          const key = typeScript.isComputedPropertyName(entry.name)
            ? read(entry.name.expression)
            : entry.name.text;
          if (typeof key !== 'string' && typeof key !== 'number') {
            result.exportName = unknown;
            result.package = unknown;
            continue;
          }
          result[key] = typeScript.isPropertyAssignment(entry)
            ? read(entry.initializer)
            : typeScript.isShorthandPropertyAssignment(entry)
              ? read(entry.name)
              : unknown;
        }
      }
      return result;
    }
    if (typeScript.isPropertyAccessExpression(node) || typeScript.isElementAccessExpression(node)) {
      const owner = read(node.expression);
      const key = typeScript.isPropertyAccessExpression(node)
        ? node.name.text
        : read(node.argumentExpression);
      return owner === unknown ? unknown : property(owner, String(key));
    }
    if (typeScript.isTemplateExpression(node)) {
      let result = node.head.text;
      for (const span of node.templateSpans) {
        const value = read(span.expression);
        if (value === unknown) return unknown;
        result += String(value) + span.literal.text;
      }
      return result;
    }
    if (typeScript.isConditionalExpression(node)) {
      const condition = read(node.condition);
      return condition === unknown ? unknown : read(condition ? node.whenTrue : node.whenFalse);
    }
    if (typeScript.isBinaryExpression(node)) {
      const left = read(node.left);
      if (left === unknown) return unknown;
      if (node.operatorToken.kind === typeScript.SyntaxKind.QuestionQuestionToken)
        return left ?? read(node.right);
      if (node.operatorToken.kind === typeScript.SyntaxKind.BarBarToken)
        return left || read(node.right);
      if (node.operatorToken.kind === typeScript.SyntaxKind.AmpersandAmpersandToken)
        return left && read(node.right);
      const right = read(node.right);
      if (right === unknown) return unknown;
      if (node.operatorToken.kind === typeScript.SyntaxKind.EqualsEqualsEqualsToken)
        return left === right;
      if (node.operatorToken.kind === typeScript.SyntaxKind.ExclamationEqualsEqualsToken)
        return left !== right;
      if (node.operatorToken.kind === typeScript.SyntaxKind.PlusToken) return left + right;
      return unknown;
    }
    if (typeScript.isCallExpression(node)) {
      const callee = unwrap(node.expression);
      if (typeScript.isPropertyAccessExpression(callee)) {
        const method = callee.name.text;
        const arguments_ = node.arguments.map(read);
        if (typeScript.isIdentifier(callee.expression) && callee.expression.text === 'Object') {
          const value = arguments_[0];
          if (method === 'freeze') return value;
          if (value === unknown || value === undefined || value === null) return unknown;
          if (method === 'entries') return Object.entries(value);
          if (method === 'values') return Object.values(value);
          if (method === 'fromEntries' && Array.isArray(value))
            return Object.fromEntries(value.filter(Array.isArray));
        }
        const owner = read(callee.expression);
        if (Array.isArray(owner)) {
          const callback = (value, index) => invoke(arguments_[0], [value, index], depth + 1);
          if (method === 'map') return owner.map(callback);
          if (method === 'flatMap') return owner.flatMap(callback);
          if (method === 'find')
            return owner.find((value, index) => callback(value, index) === true);
          if (method === 'filter')
            return owner.filter((value, index) => callback(value, index) === true);
          if (method === 'includes') return owner.includes(arguments_[0]);
          if (method === 'concat') return owner.concat(...arguments_);
        }
        return unknown;
      }
      return invoke(read(callee), node.arguments.map(read), depth + 1);
    }
    return unknown;
  };
  const roots = [];
  const getters = [...contexts.values()].filter(context =>
    context.exported.has('getGPUAnalysisOperatorCatalog')
  );
  const exportedAssemblies = [...contexts.values()].filter(context =>
    context.exported.has('ALL_OPERATORS')
  );
  if (getters.length)
    for (const context of getters)
      roots.push(invoke(lookup('getGPUAnalysisOperatorCatalog', context, new Map()), []));
  else if (exportedAssemblies.length)
    for (const context of exportedAssemblies)
      roots.push(lookup('ALL_OPERATORS', context, new Map()));
  else
    for (const context of contexts.values())
      for (const name of families)
        if (context.exported.has(name)) {
          const value = lookup(name, context, new Map());
          if (!Array.isArray(value)) continue;
          if (name.endsWith('CAPABILITIES')) {
            const details = lookup('ENTRY_POINT_OPERATOR_DETAILS', context, new Map());
            for (const family of value)
              for (const exportName of Array.isArray(family?.operators) ? family.operators : [])
                roots.push([
                  {
                    exportName,
                    package:
                      family.package ??
                      property(property(details, exportName), 'package') ??
                      (name === 'ENTRY_POINT_CAPABILITIES' ? undefined : spatialPackage)
                  }
                ]);
          } else roots.push(value);
        }
  const identities = new Set();
  for (const rows of roots)
    if (Array.isArray(rows))
      for (const row of rows)
        if (
          row &&
          !row[invalidClone] &&
          typeof row.exportName === 'string' &&
          typeof row.package === 'string'
        )
          identities.add(`${row.package}:${row.exportName}`);
  return identities;
}
