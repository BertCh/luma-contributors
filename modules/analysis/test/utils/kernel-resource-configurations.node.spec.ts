// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, readdirSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {expect, it} from 'vitest';

function findLossyResourceLoops(code: string): string[] {
  const source = typescript.createSourceFile('gate.ts', code, typescript.ScriptTarget.Latest, true);
  const recordedNames = new Set<string>();
  const declarations: typescript.VariableDeclaration[] = [];
  const violations: string[] = [];
  const visitDeclarations = (node: typescript.Node) => {
    if (typescript.isVariableDeclaration(node)) declarations.push(node);
    if (
      typescript.isVariableDeclaration(node) &&
      node.initializer &&
      typescript.isCallExpression(node.initializer) &&
      node.initializer.expression.getText(source) === 'getGPUCommandNodeKernelSources'
    ) {
      recordedNames.add(node.name.getText(source));
    }
    typescript.forEachChild(node, visitDeclarations);
  };
  visitDeclarations(source);
  const getScope = (node: typescript.Node): typescript.Node | undefined => {
    for (let parent = node.parent; parent; parent = parent.parent) {
      if (
        typescript.isBlock(parent) ||
        typescript.isSourceFile(parent) ||
        typescript.isFunctionLike(parent) ||
        typescript.isForStatement(parent) ||
        typescript.isForOfStatement(parent)
      )
        return parent;
    }
    return undefined;
  };
  const getRecorder = (name: string, use: typescript.Node): string | undefined => {
    for (let scope = getScope(use); scope; scope = getScope(scope)) {
      const declaration = declarations
        .filter(
          candidate =>
            candidate.name.getText(source) === name &&
            getScope(candidate) === scope &&
            candidate.getStart() < use.getStart()
        )
        .at(-1);
      if (declaration)
        return declaration.initializer && typescript.isCallExpression(declaration.initializer)
          ? declaration.initializer.expression.getText(source)
          : undefined;
      if (
        typescript.isFunctionLike(scope) &&
        scope.parameters.some(parameter => parameter.name.getText(source) === name)
      )
        return undefined;
    }
    return undefined;
  };
  const visitLoops = (node: typescript.Node) => {
    if (
      typescript.isCallExpression(node) &&
      /^get(?:StorageBindingCount|PrivateMemoryBytes|WorkgroupMemoryBytes)$/.test(
        node.expression.getText(source)
      )
    ) {
      const argument = node.arguments[0]?.getText(source) ?? '';
      const lookup = /^(\w+)\.get\(/.exec(argument);
      if (lookup && getRecorder(lookup[1], node) === 'getParsedKernelSources') {
        violations.push(
          `${source.getLineAndCharacterOfPosition(node.getStart()).line + 1}: ${argument}`
        );
      }
    }
    if (typescript.isForOfStatement(node)) {
      const expression = node.expression.getText(source);
      const root = /^(\w+)/.exec(expression)?.[1];
      const body = node.statement.getText(source);
      const checksResources =
        /get(?:StorageBindingCount|PrivateMemoryBytes|WorkgroupMemoryBytes)\s*\(|auditWGSLModule\s*\(|\.storage\.length|\.match\([^\n]*(?:@binding|storage)/.test(
          body
        );
      const parsedMap =
        root &&
        getRecorder(root, node) === 'getParsedKernelSources' &&
        !expression.startsWith(`${root}.configurations`) &&
        !expression.startsWith(`${root}.getConfigurations(`);
      const directMap =
        /^getParsedKernelSources\s*\(/.test(expression) &&
        !/\.(?:configurations\b|getConfigurations\s*\()/.test(expression);
      const sourceMap =
        /^new Map\(/.test(expression) &&
        [...recordedNames].some(
          name =>
            expression.includes(`${name}.map(`) &&
            getRecorder(name, node) === 'getGPUCommandNodeKernelSources'
        );
      if (checksResources && (parsedMap || directMap || sourceMap)) {
        violations.push(
          `${source.getLineAndCharacterOfPosition(node.getStart()).line + 1}: ${expression}`
        );
      }
    }
    typescript.forEachChild(node, visitLoops);
  };
  visitLoops(source);
  return violations;
}

function getSpecFiles(directory: string): string[] {
  return readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    const path = join(directory, entry.name);
    return entry.isDirectory()
      ? getSpecFiles(path)
      : /\.spec\.[jt]s$/.test(entry.name)
        ? [path]
        : [];
  });
}

it('rejects last-per-ID map budget loops and accepts configuration-preserving loops', () => {
  const declaration = 'const sources = getParsedKernelSources(nodes);';
  expect(
    findLossyResourceLoops(`${declaration}
    for (const source of sources.values()) expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);`)
  ).toHaveLength(1);
  expect(
    findLossyResourceLoops(`${declaration}
    for (const [id, source] of sources) expect(getWorkgroupMemoryBytes(source)).toBeLessThanOrEqual(16384);`)
  ).toHaveLength(1);
  expect(
    findLossyResourceLoops(`${declaration}
    expect(getStorageBindingCount(sources.get('shared'))).toBe(8);`)
  ).toHaveLength(1);
  expect(
    findLossyResourceLoops(`for (const source of getParsedKernelSources(nodes).values()) {
    expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
  }`)
  ).toHaveLength(1);
  expect(
    findLossyResourceLoops(`${declaration}
    for (const source of sources.values()) expect((source.match(/var<storage/g) ?? []).length).toBeLessThanOrEqual(8);`)
  ).toHaveLength(1);
  expect(
    findLossyResourceLoops(`const sources = getGPUCommandNodeKernelSources(nodes);
    for (const source of new Map(sources.map(source => [source.source, source])).values()) {
      expect(auditWGSLModule(source.source, source.constants)).toEqual([]);
    }`)
  ).toHaveLength(1);
  expect(
    findLossyResourceLoops(`function first() { const sources = getParsedKernelSources(nodes); }
    function second() { const sources = getGPUCommandNodeKernelSources(nodes);
      for (const kernel of sources) expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    }`)
  ).toEqual([]);
  expect(
    findLossyResourceLoops(`${declaration}
    for (const {source, constants} of sources.configurations) expect(getWorkgroupMemoryBytes(source, constants)).toBeLessThanOrEqual(16384);`)
  ).toEqual([]);
  expect(
    findLossyResourceLoops(`${declaration}
    for (const {source} of sources.getConfigurations('shared')) expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);`)
  ).toEqual([]);
});

it('checks every recorded configuration in resource-budget loops across analysis and lenses specs', () => {
  const root = fileURLToPath(new URL('../../../../', import.meta.url));
  const files = ['modules/analysis/test', 'modules/lenses/test'].flatMap(directory =>
    getSpecFiles(resolve(root, directory))
  );
  const violations = files.flatMap(file => {
    const code = readFileSync(file, 'utf8');
    if (!/getParsedKernelSources|getGPUCommandNodeKernelSources/.test(code)) return [];
    return findLossyResourceLoops(code).map(violation => `${file.slice(root.length)}:${violation}`);
  });
  expect(files.length).toBeGreaterThan(0);
  expect(violations).toEqual([]);
});
