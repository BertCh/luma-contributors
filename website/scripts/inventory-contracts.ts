// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import ts from 'typescript';
import type {InventoryContract} from './extract-inventory';

/** Select a concrete contract for the mode and required options in an inline example call. */
export function selectInventoryContract<
  Value extends {props: unknown; outputs?: unknown; overloads?: InventoryContract[]}
>(inventory: Value, symbol: string, example: string): Value {
  if (!inventory.overloads?.some(contract => contract.mode)) return inventory;
  const source = ts.createSourceFile('catalog-example.ts', example, ts.ScriptTarget.Latest, true);
  let mode: string | undefined;
  let found = false;
  const provided = new Set<string>();
  const visit = (node: ts.Node): void => {
    if (
      ts.isCallExpression(node) &&
      ((ts.isIdentifier(node.expression) && node.expression.text === symbol) ||
        (ts.isPropertyAccessExpression(node.expression) && node.expression.name.text === symbol))
    ) {
      const options = node.arguments.find(argument => ts.isObjectLiteralExpression(argument));
      if (!options || !ts.isObjectLiteralExpression(options))
        throw new Error(`${symbol}: overload selection requires an inline options object`);
      if (
        options.properties.some(
          property =>
            ts.isSpreadAssignment(property) ||
            (property.name && ts.isComputedPropertyName(property.name))
        )
      )
        throw new Error(
          `${symbol}: overload selection requires options without spreads or computed keys`
        );
      found = true;
      for (const property of options.properties)
        if (property.name) provided.add(property.name.getText(source).replace(/["']/g, ''));
      const property = options.properties.find(
        property => property.name && property.name.getText(source).replace(/["']/g, '') === 'mode'
      );
      if (property) {
        if (!ts.isPropertyAssignment(property) || !ts.isStringLiteral(property.initializer))
          throw new Error(`${symbol}: mode must be a string literal for overload validation`);
        mode = property.initializer.text;
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  if (!found) throw new Error(`${symbol}: no example call found for overload validation`);
  const modeMatches = inventory.overloads.flatMap(contract => {
    const declaredMode = contract.mode;
    if (
      !declaredMode ||
      !(mode === undefined ? declaredMode.optional : declaredMode.values.includes(mode))
    )
      return [];
    return [{contract, modeCount: declaredMode.values.length + Number(declaredMode.optional)}];
  });
  // The union overload serves callers with union-typed options. Its merged props cannot validate
  // a concrete example because fields required by only one route become optional in the union.
  const narrowestModeCount = Math.min(...modeMatches.map(match => match.modeCount));
  const matches = modeMatches.filter(
    ({contract, modeCount}) =>
      modeCount === narrowestModeCount &&
      contract.props.every(property => property.optional || provided.has(property.name))
  );
  if (matches.length !== 1)
    throw new Error(
      `${symbol}: no unique overload for mode ${JSON.stringify(mode ?? '(omitted)')}`
    );
  return {...inventory, ...matches[0].contract};
}

/** Output checks shared by the CLI and its regression: nested names refer to their root field. */
export function getInventoryOutputIssues(
  contract: {props: {name: string}[]; outputs?: {name: string; optional: boolean}[]},
  names: readonly string[]
) {
  const fields = contract.outputs ?? [];
  const listed = new Set<string>();
  const unknown: string[] = [];
  for (const name of names) {
    for (const part of name.split(/\s*[\/,]\s*/)) {
      const root = part.split(/[.\[]/)[0];
      if (fields.some(field => field.name === root)) listed.add(root);
      else if (!contract.props.some(prop => prop.name === root)) unknown.push(part);
    }
  }
  return {
    unknown,
    missing: fields
      .filter(field => !field.optional && !listed.has(field.name))
      .map(field => field.name),
    optional: fields
      .filter(field => field.optional && !listed.has(field.name))
      .map(field => field.name),
    listedCount: listed.size
  };
}
