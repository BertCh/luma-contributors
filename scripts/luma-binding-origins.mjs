// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import typeScript from 'typescript';

// Shared AST origin check; importing this module never runs a repository scan.
export const findLumaReExportSites = (sourceText, fileName) => {
  const sourceFile = typeScript.createSourceFile(
    fileName,
    sourceText,
    typeScript.ScriptTarget.Latest,
    true
  );
  const isLumaSpecifier = node =>
    node && typeScript.isStringLiteral(node) && /^@luma\.gl\//.test(node.text);
  const found = [];
  const report = (node, message) =>
    found.push({
      line: sourceFile.getLineAndCharacterOfPosition(node.getStart()).line + 1,
      message
    });
  // Local name -> luma.gl specifier of every imported binding, then of value aliases of them.
  const lumaBindings = new Map();
  for (const statement of sourceFile.statements) {
    if (!typeScript.isImportDeclaration(statement)) continue;
    if (!isLumaSpecifier(statement.moduleSpecifier)) continue;
    const specifier = statement.moduleSpecifier.text;
    const clause = statement.importClause;
    if (!clause) continue;
    if (clause.name) lumaBindings.set(clause.name.text, {specifier, member: 'default'});
    const bindings = clause.namedBindings;
    if (bindings && typeScript.isNamespaceImport(bindings)) {
      lumaBindings.set(bindings.name.text, {specifier, namespace: true});
    } else if (bindings) {
      for (const element of bindings.elements)
        lumaBindings.set(element.name.text, {
          specifier,
          member: (element.propertyName ?? element.name).text
        });
    }
  }
  const getLumaOrigin = expression => {
    let current = expression;
    while (
      typeScript.isParenthesizedExpression(current) ||
      typeScript.isAsExpression(current) ||
      typeScript.isTypeAssertionExpression(current) ||
      typeScript.isSatisfiesExpression?.(current) ||
      typeScript.isNonNullExpression(current)
    ) {
      current = current.expression;
    }
    if (typeScript.isIdentifier(current)) return lumaBindings.get(current.text);
    if (
      typeScript.isPropertyAccessExpression(current) ||
      typeScript.isElementAccessExpression(current)
    ) {
      const owner = getLumaOrigin(current.expression);
      const member = typeScript.isPropertyAccessExpression(current)
        ? current.name.text
        : current.argumentExpression && typeScript.isStringLiteralLike(current.argumentExpression)
          ? current.argumentExpression.text
          : undefined;
      if (owner?.namespace && member) return {specifier: owner.specifier, member};
    }
    return undefined;
  };
  const hasModifier = (statement, kind) =>
    statement.modifiers?.some(modifier => modifier.kind === kind) ?? false;
  // Value aliases, in source order (an alias of an alias resolves to the original specifier).
  for (const statement of sourceFile.statements) {
    if (!typeScript.isVariableStatement(statement)) continue;
    const exported = hasModifier(statement, typeScript.SyntaxKind.ExportKeyword);
    for (const declaration of statement.declarationList.declarations) {
      if (!typeScript.isIdentifier(declaration.name) || !declaration.initializer) continue;
      const origin = getLumaOrigin(declaration.initializer);
      if (!origin) continue;
      const isConstant = name => /^[A-Z][A-Z0-9_]*$/.test(name);
      const domainLimit =
        !origin.namespace && isConstant(declaration.name.text) && isConstant(origin.member ?? '');
      lumaBindings.set(declaration.name.text, {...origin, domainLimit});
      if (exported && !domainLimit) {
        report(
          statement,
          `${fileName}: re-export of luma.gl binding through alias '${declaration.name.text}' ('${origin.specifier}')`
        );
      }
    }
  }
  for (const statement of sourceFile.statements) {
    if (typeScript.isExportDeclaration(statement)) {
      if (statement.moduleSpecifier) {
        if (isLumaSpecifier(statement.moduleSpecifier)) {
          report(
            statement,
            `${fileName}: re-export from luma.gl '${statement.moduleSpecifier.text}'`
          );
        }
        continue;
      }
      const clause = statement.exportClause;
      if (!clause || !typeScript.isNamedExports(clause)) continue;
      for (const element of clause.elements) {
        const local = (element.propertyName ?? element.name).text;
        const origin = lumaBindings.get(local);
        if (origin && !origin.domainLimit) {
          report(
            statement,
            `${fileName}: re-export of luma.gl binding '${local}' ('${origin.specifier}')`
          );
        }
      }
    } else if (typeScript.isExportAssignment(statement)) {
      const origin = getLumaOrigin(statement.expression);
      if (origin && !origin.domainLimit)
        report(
          statement,
          `${fileName}: default export of a luma.gl binding ('${origin.specifier}')`
        );
    }
  }
  return found;
};
