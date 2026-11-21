// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, readdirSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {expect, it} from 'vitest';

function findLossyResourceLoops(code: string): string[] {
  const source = typescript.createSourceFile('gate.ts', code, typescript.ScriptTarget.Latest, true);
  type Origin = {
    kind:
      | 'parsed'
      | 'configurations'
      | 'recorded'
      | 'configuration'
      | 'source'
      | 'constants'
      | 'parsed-source';
    identity: string;
  };
  const declarations: typescript.VariableDeclaration[] = [];
  const violations: string[] = [];
  const visitDeclarations = (node: typescript.Node) => {
    if (typescript.isVariableDeclaration(node)) declarations.push(node);
    typescript.forEachChild(node, visitDeclarations);
  };
  visitDeclarations(source);
  const unwrap = (expression: typescript.Expression): typescript.Expression => {
    while (
      typescript.isParenthesizedExpression(expression) ||
      typescript.isAsExpression(expression) ||
      typescript.isTypeAssertionExpression(expression) ||
      typescript.isNonNullExpression(expression) ||
      typescript.isSatisfiesExpression(expression)
    )
      expression = expression.expression;
    return expression;
  };
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
  const containsBinding = (binding: typescript.BindingName, name: string): boolean =>
    typescript.isIdentifier(binding)
      ? binding.text === name
      : binding.elements.some(
          element => typescript.isBindingElement(element) && containsBinding(element.name, name)
        );
  const projectProperty = (origin: Origin | undefined, property: string): Origin | undefined => {
    if (origin?.kind === 'parsed-source') return origin;
    if (origin?.kind === 'configuration' && (property === 'source' || property === 'constants'))
      return {kind: property, identity: origin.identity};
    if (origin?.kind === 'parsed' && property === 'configurations')
      return {kind: 'configurations', identity: origin.identity};
    return undefined;
  };
  const projectBinding = (
    binding: typescript.BindingName,
    name: string,
    origin: Origin | undefined
  ): Origin | undefined => {
    if (typescript.isIdentifier(binding)) return binding.text === name ? origin : undefined;
    for (const [index, element] of binding.elements.entries()) {
      if (!typescript.isBindingElement(element) || !containsBinding(element.name, name)) continue;
      const projected = typescript.isObjectBindingPattern(binding)
        ? projectProperty(
            origin,
            element.propertyName?.getText(source) ?? element.name.getText(source)
          )
        : index === 1 && origin?.kind === 'parsed-source'
          ? origin
          : undefined;
      return projectBinding(element.name, name, projected);
    }
    return undefined;
  };
  const getElement = (
    origin: Origin | undefined,
    identity: typescript.Node
  ): Origin | undefined => {
    if (origin?.kind === 'parsed')
      return {kind: 'parsed-source', identity: String(identity.getStart())};
    if (origin?.kind === 'configurations' || origin?.kind === 'recorded')
      return {kind: 'configuration', identity: String(identity.getStart())};
    return undefined;
  };
  const resolveIdentifier = (
    name: string,
    use: typescript.Node,
    visited: Set<typescript.Node>
  ): Origin | undefined => {
    for (let scope = getScope(use); scope; scope = getScope(scope)) {
      const declaration = declarations
        .filter(
          candidate =>
            containsBinding(candidate.name, name) &&
            getScope(candidate) === scope &&
            candidate.getStart() < use.getStart()
        )
        .at(-1);
      if (declaration) {
        if (visited.has(declaration)) return undefined;
        const nextVisited = new Set([...visited, declaration]);
        if (declaration.initializer)
          return projectBinding(
            declaration.name,
            name,
            resolveExpression(declaration.initializer, nextVisited)
          );
        const loop = declaration.parent.parent;
        if (typescript.isForOfStatement(loop))
          return projectBinding(
            declaration.name,
            name,
            getElement(resolveExpression(loop.expression, nextVisited), loop)
          );
        return undefined;
      }
      if (typescript.isFunctionLike(scope)) {
        const parameter = scope.parameters.find(candidate => containsBinding(candidate.name, name));
        if (!parameter) continue;
        const call = scope.parent;
        if (
          typescript.isCallExpression(call) &&
          call.arguments[0] === scope &&
          typescript.isPropertyAccessExpression(call.expression) &&
          ['forEach', 'map', 'flatMap', 'every', 'some', 'filter', 'find'].includes(
            call.expression.name.text
          )
        ) {
          return projectBinding(
            parameter.name,
            name,
            getElement(resolveExpression(call.expression.expression, visited), parameter)
          );
        }
        return undefined;
      }
    }
    return undefined;
  };
  const resolveExpression = (
    expression: typescript.Expression,
    visited = new Set<typescript.Node>()
  ): Origin | undefined => {
    expression = unwrap(expression);
    if (typescript.isIdentifier(expression))
      return resolveIdentifier(expression.text, expression, visited);
    if (typescript.isPropertyAccessExpression(expression))
      return projectProperty(
        resolveExpression(expression.expression, visited),
        expression.name.text
      );
    if (typescript.isElementAccessExpression(expression)) {
      const receiver = resolveExpression(expression.expression, visited);
      if (typescript.isStringLiteral(expression.argumentExpression))
        return projectProperty(receiver, expression.argumentExpression.text);
      if (receiver?.kind === 'recorded' || receiver?.kind === 'configurations')
        return {
          kind: 'configuration',
          identity: `${receiver.identity}[${expression.argumentExpression.getText(source)}]`
        };
      return getElement(receiver, expression);
    }
    if (typescript.isCallExpression(expression)) {
      const callee = unwrap(expression.expression);
      if (typescript.isIdentifier(callee)) {
        if (callee.text === 'getParsedKernelSources')
          return {kind: 'parsed', identity: String(expression.getStart())};
        if (callee.text === 'getGPUCommandNodeKernelSources')
          return {kind: 'recorded', identity: String(expression.getStart())};
      }
      if (typescript.isPropertyAccessExpression(callee)) {
        const receiver = resolveExpression(callee.expression, visited);
        if (callee.name.text === 'values') return receiver;
        if (receiver?.kind === 'parsed' && callee.name.text === 'getConfigurations')
          return {kind: 'configurations', identity: String(expression.getStart())};
        if (receiver?.kind === 'parsed' && callee.name.text === 'get')
          return {kind: 'parsed-source', identity: String(expression.getStart())};
      }
    }
    if (typescript.isNewExpression(expression) && expression.expression.getText(source) === 'Map') {
      const argument = expression.arguments?.[0];
      if (
        argument &&
        typescript.isCallExpression(argument) &&
        typescript.isPropertyAccessExpression(argument.expression) &&
        argument.expression.name.text === 'map' &&
        resolveExpression(argument.expression.expression, visited)?.kind === 'recorded'
      ) {
        const callback = argument.arguments[0];
        const body =
          callback &&
          (typescript.isArrowFunction(callback) || typescript.isFunctionExpression(callback))
            ? callback.body
            : undefined;
        const key =
          body && typescript.isArrayLiteralExpression(body)
            ? body.elements[0]?.getText(source)
            : undefined;
        return {
          kind: key?.includes('.source') && key.includes('.constants') ? 'recorded' : 'parsed',
          identity: String(expression.getStart())
        };
      }
    }
    return undefined;
  };
  const visitCalls = (node: typescript.Node) => {
    if (typescript.isCallExpression(node)) {
      const callee = unwrap(node.expression);
      const name = callee.getText(source);
      const argument = node.arguments[0];
      const origin = argument ? resolveExpression(argument) : undefined;
      if (
        /^(?:getStorageBindingCount|getPrivateMemoryBytes|getWorkgroupMemoryBytes|auditWGSLModule)$/.test(
          name
        )
      ) {
        const constants = node.arguments[1] ? resolveExpression(node.arguments[1]) : undefined;
        const missingConstants =
          origin?.kind === 'source' &&
          name !== 'getStorageBindingCount' &&
          (constants?.kind !== 'constants' || constants.identity !== origin.identity);
        if (origin?.kind === 'parsed-source' || missingConstants)
          violations.push(
            `${source.getLineAndCharacterOfPosition(node.getStart()).line + 1}: lost configuration or constants: ${node.getText(source)}`
          );
      }
      if (
        typescript.isPropertyAccessExpression(callee) &&
        callee.name.text === 'match' &&
        resolveExpression(callee.expression)?.kind === 'parsed-source' &&
        /@binding|storage/.test(node.getText(source))
      )
        violations.push(
          `${source.getLineAndCharacterOfPosition(node.getStart()).line + 1}: lossy source binding count`
        );
    }
    typescript.forEachChild(node, visitCalls);
  };
  visitCalls(source);
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
  // An alias of the parsed-source map is the same lossy map.
  expect(
    findLossyResourceLoops(`${declaration}
    const alias = sources;
    for (const source of alias.values()) expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);`)
  ).toHaveLength(1);
  expect(
    findLossyResourceLoops(`${declaration}
    const alias = sources;
    const second = alias;
    expect(getStorageBindingCount(second.get('shared'))).toBe(8);`)
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

it('rejects omitted or substituted constants in configuration memory checks (TOOL-118)', () => {
  const declaration = 'const sources = getParsedKernelSources(nodes);';
  for (const constants of ['', ', {}', ', other.constants']) {
    expect(
      findLossyResourceLoops(`${declaration}
      for (const kernel of sources.configurations) {
        expect(getWorkgroupMemoryBytes(kernel.source${constants})).toBeLessThanOrEqual(16384);
      }`)
    ).toHaveLength(1);
  }
  expect(
    findLossyResourceLoops(`${declaration}
    for (const {source, constants} of sources.getConfigurations('shared')) {
      expect(getPrivateMemoryBytes(source, {})).toBeLessThanOrEqual(8192);
    }`)
  ).toHaveLength(1);
  expect(
    findLossyResourceLoops(`${declaration}
    for (const kernel of sources.configurations) {
      expect(getWorkgroupMemoryBytes(kernel.source, kernel.constants)).toBeLessThanOrEqual(16384);
      expect(getStorageBindingCount(kernel.source)).toBeLessThanOrEqual(8);
    }`)
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
  expect(violations, violations.join('\n')).toEqual([]);
});

it.each([
  [
    'configuration collection alias',
    `const batch = parsed.configurations; for (const kernel of batch) getWorkgroupMemoryBytes(kernel.source);`
  ],
  [
    'local source alias',
    `for (const kernel of parsed.configurations) { const shader = kernel.source; getPrivateMemoryBytes(shader); }`
  ],
  ['map callback', `parsed.configurations.map(kernel => getWorkgroupMemoryBytes(kernel.source));`],
  [
    'forEach callback',
    `parsed.configurations.forEach(kernel => getWorkgroupMemoryBytes(kernel.source));`
  ],
  [
    'forEach destructured callback',
    `parsed.getConfigurations('id').forEach(({source, constants}) => getPrivateMemoryBytes(source, {}));`
  ],
  [
    'full-module audit',
    `for (const kernel of parsed.configurations) expect(auditWGSLModule(kernel.source)).toEqual([]);`
  ],
  [
    'parenthesized parsed-map alias',
    `const copy = (parsed); for (const shader of copy.values()) getWorkgroupMemoryBytes(shader);`
  ],
  [
    'asserted parsed-map alias',
    `const copy = parsed as typeof parsed; for (const shader of copy.values()) getWorkgroupMemoryBytes(shader);`
  ]
])('rejects dropped configurations/constants through %s (B4)', (_name, body) => {
  expect(
    findLossyResourceLoops(`const parsed = getParsedKernelSources(nodes); ${body}`)
  ).toHaveLength(1);
});

it('accepts constants/source aliases and source-only binding counts through both iteration forms (B4)', () => {
  const declaration = 'const parsed = getParsedKernelSources(nodes);';
  for (const body of [
    `const batch = (parsed.configurations); for (const kernel of batch) { const shader = kernel.source; const overrides = kernel.constants; getWorkgroupMemoryBytes(shader, overrides); }`,
    `for (const kernel of parsed.configurations) { const overrides = kernel.constants; getPrivateMemoryBytes(kernel.source, overrides); }`,
    `parsed.configurations.forEach(kernel => { const shader = kernel.source; const overrides = kernel.constants; auditWGSLModule(shader, overrides); getStorageBindingCount(shader); });`,
    `const kernels = getGPUCommandNodeKernelSources(nodes); const unique = new Map(kernels.map(kernel => [kernel.source + JSON.stringify(kernel.constants), kernel])); for (const kernel of unique.values()) auditWGSLModule(kernel.source, kernel.constants);`,
    `parsed.getConfigurations('id').forEach(({source: shader, constants: overrides}) => getWorkgroupMemoryBytes(shader, overrides));`
  ])
    expect(findLossyResourceLoops(`${declaration} ${body}`)).toEqual([]);
});
