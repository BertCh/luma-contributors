// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readdirSync, readFileSync} from 'node:fs';
import {join} from 'node:path';
import ts from 'typescript';
import {expect, test} from 'vitest';

const sourceDirectory = join(__dirname, '../../../examples/deck/kepler-compute/src');

/** Host-added UI: every panel plus resident legend/table/chart/drawing/tooltip and source controls. */
const HOST_UI_FILE =
  /(-panel|-legend|-control|-data-table|-tooltip|-notice|-host)\.tsx$|^(remote-source-control|native-sql-explorer|native-export-image-tiled|native-workflows|native-image-export|resident-table-column-(edits|tools)|resident-select-geometry-control)\.tsx$/;

type Violation = {file: string; line: number; rule: string};

function getAttributes(node: ts.JsxOpeningLikeElement): Map<string, ts.JsxAttribute> {
  const attributes = new Map<string, ts.JsxAttribute>();
  for (const attribute of node.attributes.properties)
    if (ts.isJsxAttribute(attribute)) attributes.set(attribute.name.getText(), attribute);
  return attributes;
}
function hasSpread(node: ts.JsxOpeningLikeElement): boolean {
  return node.attributes.properties.some(attribute => ts.isJsxSpreadAttribute(attribute));
}
function getLiteral(attribute?: ts.JsxAttribute): string | undefined {
  const initializer = attribute?.initializer;
  if (initializer && ts.isStringLiteral(initializer)) return initializer.text;
  if (initializer && ts.isJsxExpression(initializer) && initializer.expression) {
    const expression = initializer.expression;
    if (ts.isStringLiteralLike(expression)) return expression.text;
  }
  return undefined;
}
function hasContent(element: ts.JsxElement): boolean {
  return element.children.some(child => {
    if (ts.isJsxText(child)) return child.text.trim().length > 0;
    if (ts.isJsxExpression(child)) return Boolean(child.expression);
    if (ts.isJsxElement(child)) {
      const tag = child.openingElement.tagName.getText();
      if (tag === 'svg' || tag === 'img') return false;
      return (
        hasContent(child) ||
        child.openingElement.attributes.properties.some(
          attribute => ts.isJsxAttribute(attribute) && attribute.name.getText() === 'aria-label'
        )
      );
    }
    return false; // Self-closing children (icons) carry no text.
  });
}

/** Scans one TSX source for missing accessible names, roles and relationships. */
export function scanAccessibility(file: string, text: string): Violation[] {
  const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const violations: Violation[] = [];
  const hasLabelFor = /htmlFor=/.test(text);
  const visit = (node: ts.Node, insideLabel: boolean) => {
    const opening = ts.isJsxElement(node)
      ? node.openingElement
      : ts.isJsxSelfClosingElement(node)
        ? node
        : undefined;
    let nextInsideLabel = insideLabel;
    if (opening) {
      const tag = opening.tagName.getText();
      const attributes = getAttributes(opening);
      const spread = hasSpread(opening);
      const line = source.getLineAndCharacterOfPosition(opening.getStart()).line + 1;
      const fail = (rule: string) => violations.push({file, line, rule});
      const named =
        attributes.has('aria-label') ||
        attributes.has('aria-labelledby') ||
        attributes.has('title');
      const role = getLiteral(attributes.get('role'));
      if (tag === 'label') nextInsideLabel = true;
      if (tag === 'input' || tag === 'select' || tag === 'textarea') {
        const type = getLiteral(attributes.get('type'));
        const exempt = type === 'hidden' || type === 'submit' || type === 'button';
        if (!exempt && !spread && !named && !insideLabel && !(attributes.has('id') && hasLabelFor))
          fail(`${tag} has no accessible name`);
      }
      if (tag === 'button' && !spread && !named) {
        const nameless =
          ts.isJsxSelfClosingElement(node) || (ts.isJsxElement(node) && !hasContent(node));
        if (nameless) fail('button has no accessible name');
      }
      if (tag === 'table' && !spread && !named) {
        const hasCaption = ts.isJsxElement(node) && /<caption/.test(node.getText());
        if (!hasCaption) fail('table has no name or caption');
      }
      if (
        (tag === 'canvas' || tag === 'svg') &&
        !spread &&
        !named &&
        role !== 'presentation' &&
        attributes.get('aria-hidden') === undefined
      )
        fail(`${tag} needs aria-label or aria-hidden`);
      if (tag === 'img' && !attributes.has('alt') && !spread) fail('img has no alt');
      if ((role === 'dialog' || role === 'alertdialog' || attributes.has('aria-modal')) && !named)
        fail('dialog has no name');
      if ((tag === 'progress' || role === 'progressbar') && !spread && !named && !insideLabel)
        fail('progress has no name');
      if (role === 'tablist' && !named) fail('tablist has no name');
      if (role === 'listbox' && !named) fail('listbox has no name');
      if (attributes.has('aria-expanded') && !attributes.has('aria-controls') && tag === 'button')
        fail('aria-expanded button has no aria-controls');
      if (getLiteral(attributes.get('aria-live')) === 'assertive' && role !== 'alert')
        fail('aria-live assertive outside an alert (reserve for errors)');
      if (
        role === 'alert' &&
        /(progress|loading|running|reading|busy)/i.test(node.getText().slice(0, 400)) &&
        !/(error|fail|unavailable|stale)/i.test(node.getText().slice(0, 400))
      )
        fail('role=alert on progress text (use role=status, polite)');
    }
    ts.forEachChild(node, child => visit(child, nextInsideLabel));
  };
  visit(source, false);
  return violations;
}

const hostFiles = readdirSync(sourceDirectory).filter(name => HOST_UI_FILE.test(name));

test('host panel scanner covers the expected panels', () => {
  for (const name of [
    'analysis-catalog-panel.tsx',
    'memory-manager-panel.tsx',
    'memory-budget-panel.tsx',
    'assistant-panel.tsx',
    'sandbox-share-panel.tsx',
    'source-workspace-panel.tsx',
    'resident-data-table.tsx',
    'resident-style-legend.tsx',
    'resident-drawing-panel.tsx',
    'resident-chart-panel.tsx'
  ])
    expect(hostFiles).toContain(name);
});

test('scanner detects nameless controls', () => {
  const violations = scanAccessibility(
    'fixture.tsx',
    `export const A = () => <div><input type="text" /><button><svg /></button><table><tbody /></table>
     <label>Name <input /></label><button aria-label="Close" /><div role="dialog" /></div>;`
  );
  expect(violations.map(violation => violation.rule).sort()).toEqual([
    'button has no accessible name',
    'dialog has no name',
    'input has no accessible name',
    'svg needs aria-label or aria-hidden',
    'table has no name or caption'
  ]);
});

test('host panels give every control, table, dialog and live region an accessible name', () => {
  const violations = hostFiles.flatMap(name =>
    scanAccessibility(name, readFileSync(join(sourceDirectory, name), 'utf8'))
  );
  expect(violations.map(({file, line, rule}) => `${file}:${line} ${rule}`)).toEqual([]);
});

test('interpolation surface tooltip is a dismissible tooltip role that does not take the pointer', () => {
  const text = readFileSync(join(sourceDirectory, 'interpolation-surface-tooltip.tsx'), 'utf8');
  expect(text).toContain('role="tooltip"');
  expect(text).toContain("event.key === 'Escape'");
  expect(text).toContain("pointerEvents: 'none'");
});
