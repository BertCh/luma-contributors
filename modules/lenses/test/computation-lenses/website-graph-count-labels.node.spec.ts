// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import typescript from 'typescript';
import {expect, test} from 'vitest';
import {formatCount} from '../../../../website/src/lens/format';
import {createElement, type ReactElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import * as jsxRuntime from 'react/jsx-runtime';
import type {KernelSpread} from '../../../../website/src/components/meta/pages/lenses/record-parts';
import {fromRecordedLensFile} from '../../src/computation-lenses/registry';
import {extractLensViewModel} from '../../../../website/src/lens/extract-lens-view-model';

test('review 2 R2 time-window page and figure describe eight compute plus one copy as nine graph nodes', () => {
  const id = 'gpu-dataframe.time-window-filter';
  const record = fromRecordedLensFile(
    JSON.parse(
      readFileSync(new URL(`../../../../docs/lens-records/${id}.json`, import.meta.url), 'utf8')
    )
  ).record;
  const graph = record.graphs.find(graph => graph.form === 'compiled')!;
  expect(graph.nodes.filter(node => node.kind === 'compute')).toHaveLength(8);
  expect(graph.nodes.filter(node => node.kind === 'copy')).toHaveLength(1);
  const model = extractLensViewModel(id, record);
  const snippets: string[] = [];
  for (const file of ['index.tsx', 'declare-figure.tsx']) {
    const source = typescript.createSourceFile(
      file,
      readFileSync(
        new URL(
          `../../../../website/src/components/meta/pages/command-graph/${file}`,
          import.meta.url
        ),
        'utf8'
      ),
      typescript.ScriptTarget.Latest,
      true,
      typescript.ScriptKind.TSX
    );
    function visit(node: typescript.Node) {
      if (
        typescript.isObjectLiteralExpression(node) &&
        node.properties.some(
          property =>
            typescript.isPropertyAssignment(property) &&
            property.name.getText(source) === 'id' &&
            property.initializer.getText(source) === "'time-window-filter'"
        )
      ) {
        const note = node.properties.find(
          property =>
            typescript.isPropertyAssignment(property) && property.name.getText(source) === 'note'
        ) as typescript.PropertyAssignment;
        snippets.push(note.initializer.getText(source));
      }
      if (
        typescript.isJsxElement(node) &&
        ((node.openingElement.tagName.getText(source) === 'Prose' &&
          node.getText(source).includes('rowCount')) ||
          (node.openingElement.tagName.getText(source) === 'li' &&
            /model\.(nodes\.length|recordedNodeCount)/.test(node.getText(source))))
      )
        snippets.push(node.getText(source));
      if (
        typescript.isJsxAttribute(node) &&
        node.name.getText(source) === 'aria-label' &&
        node.initializer &&
        typescript.isJsxExpression(node.initializer) &&
        node.initializer.expression?.getText(source).includes('schedule order')
      )
        snippets.push(node.initializer.expression.getText(source));
      typescript.forEachChild(node, visit);
    }
    visit(source);
  }
  expect(snippets).toHaveLength(4);
  for (const snippet of snippets) {
    const code = typescript.transpileModule(`export const text = (${snippet});`, {
      fileName: 'snippet.tsx',
      compilerOptions: {module: typescript.ModuleKind.CommonJS, jsx: typescript.JsxEmit.ReactJSX}
    }).outputText;
    const exports: {text?: string} = {};
    const render = (_tag: unknown, props: {children?: unknown}) =>
      Array.isArray(props.children)
        ? props.children.flat(Infinity).join('')
        : String(props.children ?? '');
    runInNewContext(code, {
      exports,
      model,
      timeWindow: model,
      rowCount: 3,
      formatCount,
      Prose: 'prose',
      Term: 'term',
      selectedRow: {label: 'fixture'},
      sources: [],
      fromInside: [],
      instances: [],
      selected: 0,
      require: () => ({jsx: render, jsxs: render, Fragment: 'fragment'})
    });
    expect(exports.text?.replace(/\s+/g, ' ').trim()).toMatch(/9 graph nodes/);
    expect(exports.text).not.toMatch(/nine-kernel|9 kernels/);
  }
});

test('review 2 R1 empty histogram renders finite zero-height bars', () => {
  const file = 'record-parts.tsx';
  const source = typescript.createSourceFile(
    file,
    readFileSync(
      new URL(
        '../../../../website/src/components/meta/pages/lenses/record-parts.tsx',
        import.meta.url
      ),
      'utf8'
    ),
    typescript.ScriptTarget.Latest,
    true,
    typescript.ScriptKind.TSX
  );
  const declaration = source.statements.find(
    statement =>
      typescript.isFunctionDeclaration(statement) && statement.name?.text === 'KernelSpreadChart'
  );
  if (!declaration) throw new Error('missing histogram component');
  const code = typescript.transpileModule(declaration.getText(source), {
    fileName: file,
    compilerOptions: {module: typescript.ModuleKind.CommonJS, jsx: typescript.JsxEmit.ReactJSX}
  }).outputText;
  const exports: {KernelSpreadChart?: (props: {spread: KernelSpread}) => ReactElement} = {};
  runInNewContext(code, {exports, require: () => jsxRuntime});
  const spread = {
    bins: [
      {label: '0', count: 0},
      {label: '1', count: 0}
    ],
    median: {id: 'empty', kernels: 0},
    smallest: {id: 'empty', kernels: 0},
    largest: {id: 'empty', kernels: 0}
  };
  const html = renderToStaticMarkup(createElement(exports.KernelSpreadChart!, {spread}));
  expect(html).not.toMatch(/NaN|Infinity/);
  expect(html.match(/height:0%/g)).toHaveLength(2);
});

test.each([
  'lens-graph-panel.tsx',
  'pages/command-graph/index.tsx',
  'pages/building-blocks/grid-index-figure.tsx',
  'pages/building-blocks/compaction-figure.tsx'
])('R2 renders one compute and one copy as two graph nodes in %s', file => {
  const source = typescript.createSourceFile(
    file,
    readFileSync(
      new URL(`../../../../website/src/components/meta/${file}`, import.meta.url),
      'utf8'
    ),
    typescript.ScriptTarget.Latest,
    true,
    typescript.ScriptKind.TSX
  );
  let expression: typescript.Node | undefined;
  function visit(node: typescript.Node) {
    if (typescript.isPropertyAccessExpression(node) && node.name.text === 'recordedNodeCount') {
      for (let parent: typescript.Node | undefined = node.parent; parent; parent = parent.parent) {
        if (
          typescript.isJsxFragment(parent) ||
          (typescript.isJsxElement(parent) &&
            parent.openingElement.tagName.getText(source) === 'li')
        ) {
          expression = parent;
          break;
        }
        if (
          typescript.isJsxAttribute(parent) &&
          parent.name.getText(source) === 'notice' &&
          parent.initializer &&
          typescript.isJsxExpression(parent.initializer)
        ) {
          expression = parent.initializer.expression;
          break;
        }
      }
    }
    typescript.forEachChild(node, visit);
  }
  visit(source);
  if (!expression) throw new Error(`missing recorded graph label in ${file}`);
  const code = typescript.transpileModule(`export const label = (${expression.getText(source)});`, {
    fileName: 'label.tsx',
    compilerOptions: {module: typescript.ModuleKind.CommonJS, jsx: typescript.JsxEmit.ReactJSX}
  }).outputText;
  const model = {
    recordedNodeCount: 2,
    recordedEdgeCount: 1,
    collapsed: false,
    nodes: [{kind: 'compute'}, {kind: 'copy'}]
  };
  const exports: {label?: string} = {};
  const render = (_tag: unknown, props: {children?: unknown}) =>
    Array.isArray(props.children)
      ? props.children.flat(Infinity).join('')
      : String(props.children ?? '');
  runInNewContext(code, {
    exports,
    model,
    moran: model,
    rowCount: 3,
    formatCount,
    require: () => ({jsx: render, jsxs: render, Fragment: 'fragment'})
  });
  expect(exports.label?.replace(/\s+/g, ' ').trim()).toMatch(/2 graph nodes/);
  expect(exports.label).not.toMatch(/2 kernels/);
});
