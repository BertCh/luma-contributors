// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {join} from 'node:path';
import typescript from 'typescript';
import {
  DOCS_ROOT,
  createVirtualProgram,
  formatDiagnostic,
  getRepositoryCompilerOptions,
  readDocPage
} from './doc-program';

export type DocSnippet = {
  /** Doc page relative to `DOCS_ROOT`. */
  page: string;
  /** 1-based line of the opening fence. */
  fenceLine: number;
  /** Block text between the fences. */
  code: string;
  /** Reason given by a `no-typecheck` opt-out comment, if any. */
  optOutReason?: string;
  markdownLines?: number[];
};

const OPT_OUT =
  /^(?:<!--\s*no-typecheck(?::\s*(.*?))?\s*-->|\{\/\*\s*no-typecheck(?::\s*(.*?))?\s*\*\/\})$/;

/** Free names with real types. Values are `[import specifier, type expression]`. */
const TYPED_FREE_NAMES: Record<string, [string, string]> = {
  device: ['@luma.gl/core', 'Device'],
  graph: ['@luma.gl/graph/gpu-core', 'GPUCommandGraph'],
  commandEncoder: ['@luma.gl/core', 'CommandEncoder']
};

/** Diagnostics that report a name with no declaration. */
const MISSING_NAME_CODES = new Set([2304, 2552, 2749, 18004, 2503]);

/** Extracts TypeScript fragments; complete walkthroughs also have stricter, no-implicit-input specs. */
export function extractSnippets(page: string, text = readDocPage(page)): DocSnippet[] {
  const lines = text.split('\n');
  const snippets: DocSnippet[] = [];
  for (let index = 0; index < lines.length; index++) {
    if (!/^```(?:ts|typescript)\s*$/.test(lines[index].trim())) {
      continue;
    }
    const fenceLine = index + 1;
    let previous = index - 1;
    while (previous >= 0 && lines[previous].trim() === '') {
      previous--;
    }
    const optOut = previous >= 0 ? OPT_OUT.exec(lines[previous].trim()) : null;
    const body: string[] = [];
    for (index++; index < lines.length && lines[index].trim() !== '```'; index++) {
      body.push(lines[index]);
    }
    snippets.push({
      page,
      fenceLine,
      code: body.join('\n'),
      ...(optOut ? {optOutReason: optOut[1] ?? optOut[2] ?? ''} : {})
    });
  }
  return snippets;
}

export function getSnippetPath(snippet: DocSnippet): string {
  return join(DOCS_ROOT, '__doc_snippets__', `${snippet.page}-${snippet.fenceLine}.ts`);
}

/** Returns the identifier a missing-name diagnostic points at and whether it is a type. */
function getMissingName(diagnostic: typescript.Diagnostic): {name: string; isType: boolean} | null {
  if (!diagnostic.file || diagnostic.start === undefined || !diagnostic.length) {
    return null;
  }
  const name = diagnostic.file.text.slice(diagnostic.start, diagnostic.start + diagnostic.length);
  if (!/^[A-Za-z_$][\w$]*$/.test(name)) {
    return null;
  }
  // Start above the identifier itself, which is an expression node in every position.
  let node: typescript.Node | undefined = findNodeAt(diagnostic.file, diagnostic.start).parent;
  let isType = false;
  while (node && !typescript.isSourceFile(node)) {
    if (typescript.isTypeNode(node)) {
      isType = true;
      break;
    }
    if (typescript.isExpression(node) || typescript.isStatement(node)) {
      break;
    }
    node = node.parent;
  }
  return {name, isType};
}

function findNodeAt(sourceFile: typescript.SourceFile, position: number): typescript.Node {
  let found: typescript.Node = sourceFile;
  const visit = (node: typescript.Node): void => {
    if (node.getStart(sourceFile) <= position && position < node.getEnd()) {
      found = node;
      node.forEachChild(visit);
    }
  };
  sourceFile.forEachChild(visit);
  return found;
}

// These are deliberate platform services. Shadowable DOM values such as status, origin and
// window must be declared when a fragment uses them as application inputs.
const PLATFORM_GLOBALS = new Set([
  'console',
  'document',
  'navigator',
  'globalThis',
  'performance',
  'crypto',
  'customElements',
  'indexedDB',
  'localStorage',
  'sessionStorage',
  'caches',
  'Math',
  'JSON',
  'Reflect',
  'Atomics',
  'WebAssembly',
  'NaN',
  'Infinity'
]);

function getLibraryGlobalIdentifiers(
  program: typescript.Program,
  sourceFile: typescript.SourceFile | undefined
): typescript.Identifier[] {
  if (!sourceFile) return [];
  const identifiers: typescript.Identifier[] = [];
  const checker = program.getTypeChecker();
  const visit = (node: typescript.Node): void => {
    if (typescript.isIdentifier(node) && !PLATFORM_GLOBALS.has(node.text)) {
      const symbol = typescript.isShorthandPropertyAssignment(node.parent)
        ? checker.getShorthandAssignmentValueSymbol(node.parent)
        : checker.getSymbolAtLocation(node);
      const declarations = symbol?.declarations ?? [];
      if (
        symbol &&
        declarations.length &&
        declarations.every(
          declaration =>
            program.isSourceFileDefaultLibrary(declaration.getSourceFile()) &&
            typescript.isVariableDeclaration(declaration)
        )
      ) {
        const type = checker.getTypeOfSymbolAtLocation(symbol, node);
        if (!type.getCallSignatures().length && !type.getConstructSignatures().length)
          identifiers.push(node);
      }
    }
    node.forEachChild(visit);
  };
  sourceFile.forEachChild(visit);
  return identifiers;
}

export function getLibraryGlobalNames(
  program: typescript.Program,
  sourceFile: typescript.SourceFile | undefined
): Set<string> {
  return new Set(
    getLibraryGlobalIdentifiers(program, sourceFile).map(identifier => identifier.text)
  );
}

function getFragmentInputDiagnostics(
  program: typescript.Program,
  sourceFile: typescript.SourceFile
): typescript.Diagnostic[] {
  const diagnostics: typescript.Diagnostic[] = getLibraryGlobalIdentifiers(program, sourceFile).map(
    identifier => ({
      file: sourceFile,
      start: identifier.getStart(sourceFile),
      length: identifier.getWidth(sourceFile),
      category: typescript.DiagnosticCategory.Error,
      code: 90002,
      messageText: `Undeclared fragment input '${identifier.text}'; declare its type explicitly.`
    })
  );
  const visit = (node: typescript.Node) => {
    if (typescript.isCallExpression(node)) {
      let expression: typescript.Expression = node.expression;
      while (typescript.isParenthesizedExpression(expression)) expression = expression.expression;
      if (typescript.isIdentifier(expression) && expression.text === 'require') {
        const argument = node.arguments[0];
        const specifier =
          argument && typescript.isStringLiteralLike(argument)
            ? argument.text
            : node.getText(sourceFile);
        diagnostics.push({
          file: sourceFile,
          start: node.getStart(sourceFile),
          length: node.getWidth(sourceFile),
          category: typescript.DiagnosticCategory.Error,
          code: 90003,
          messageText: `Untyped CommonJS require(${JSON.stringify(specifier)}); use an explicit typed public import.`
        });
      }
      const checker = program.getTypeChecker();
      const signature = checker.getResolvedSignature(node);
      // Resolve aliases and namespace accesses through the signature, rather than relying on
      // the local import spelling. Follow wrappers and control-flow expression results to
      // distinguish a discarded array from an array consumed by an owner write.
      const declaration = signature?.declaration;
      const name = declaration && 'name' in declaration ? declaration.name?.getText() : '';
      const isPackedArray = (type: typescript.Type): boolean =>
        type.isUnion()
          ? type.types.every(isPackedArray)
          : /^(?:Float32Array|Uint32Array|Int32Array)$/.test(type.getSymbol()?.getName() ?? '');
      const hasDestination = signature?.parameters.some((parameter, index) => {
        const argument = node.arguments[index];
        const destinationType = checker
          .getTypeOfSymbolAtLocation(parameter, declaration ?? node)
          .getNonNullableType();
        return (
          index > 0 &&
          argument !== undefined &&
          /^(?:target|destination|output|out|values)$/.test(parameter.name) &&
          isPackedArray(destinationType) &&
          checker.isTypeAssignableTo(checker.getTypeAtLocation(argument), destinationType)
        );
      });
      if (
        /^get\w*ParameterValues$/.test(name ?? '') &&
        signature &&
        isPackedArray(signature.getReturnType()) &&
        isDiscardedExpressionResult(node) &&
        !hasDestination
      ) {
        diagnostics.push({
          file: sourceFile,
          start: node.getStart(sourceFile),
          length: node.getWidth(sourceFile),
          category: typescript.DiagnosticCategory.Error,
          code: 90004,
          messageText: `Allocating parameter packer result discarded; write the returned array to its parameter buffer, initialize values from it, or supply a supported destination.`
        });
      }
    }
    node.forEachChild(visit);
  };
  sourceFile.forEachChild(visit);
  return diagnostics;
}

function isDiscardedExpressionResult(expression: typescript.Expression): boolean {
  let current: typescript.Node = expression;
  while (current.parent) {
    const parent = current.parent;
    if (typescript.isVoidExpression(parent) || typescript.isExpressionStatement(parent))
      return true;
    if (
      typescript.isParenthesizedExpression(parent) ||
      typescript.isAsExpression(parent) ||
      typescript.isTypeAssertionExpression(parent) ||
      typescript.isNonNullExpression(parent) ||
      typescript.isSatisfiesExpression(parent) ||
      typescript.isAwaitExpression(parent)
    ) {
      current = parent;
      continue;
    }
    if (typescript.isConditionalExpression(parent) && current !== parent.condition) {
      current = parent;
      continue;
    }
    if (typescript.isBinaryExpression(parent)) {
      const operator = parent.operatorToken.kind;
      if (operator === typescript.SyntaxKind.CommaToken) {
        if (current === parent.left) return true;
        current = parent;
        continue;
      }
      if (
        current === parent.right &&
        (operator === typescript.SyntaxKind.AmpersandAmpersandToken ||
          operator === typescript.SyntaxKind.BarBarToken ||
          operator === typescript.SyntaxKind.QuestionQuestionToken)
      ) {
        current = parent;
        continue;
      }
    }
    return false;
  }
  return false;
}

/** Explicitly typed host inputs; API names must be bound by the fence itself. */
function getPrelude(missingNames: Map<string, boolean>): string[] {
  return [...missingNames].flatMap(([name, isType]) => {
    const typed = TYPED_FREE_NAMES[name];
    return typed && !isType
      ? [
          `import type {${typed[1]} as Snippet${typed[1]}} from '${typed[0]}';`,
          `declare const ${name}: Snippet${typed[1]};`
        ]
      : [];
  });
}

export function getSnippetDiagnostics(checked: DocSnippet[]) {
  const options = getRepositoryCompilerOptions();
  // Pass 1: find the free names of each block.
  const firstFiles = new Map(
    checked.map(snippet => [getSnippetPath(snippet), `${snippet.code}\nexport {};\n`])
  );
  const firstProgram = createVirtualProgram(firstFiles, options);
  const missingNamesByPath = new Map<string, Map<string, boolean>>();
  for (const path of firstFiles.keys()) {
    const sourceFile = firstProgram.getSourceFile(path);
    const missingNames = new Map<string, boolean>();
    for (const diagnostic of firstProgram.getSemanticDiagnostics(sourceFile)) {
      const missing = MISSING_NAME_CODES.has(diagnostic.code) ? getMissingName(diagnostic) : null;
      if (missing) {
        missingNames.set(missing.name, (missingNames.get(missing.name) ?? false) || missing.isType);
      }
    }
    for (const name of getLibraryGlobalNames(firstProgram, sourceFile)) {
      missingNames.set(name, missingNames.get(name) ?? false);
    }
    missingNamesByPath.set(path, missingNames);
  }

  // Pass 2: declare them and report every remaining diagnostic at its doc line.
  const preludeLengths = new Map<string, number>();
  const secondFiles = new Map<string, string>();
  for (const [path, text] of firstFiles) {
    const snippet = checked.find(candidate => getSnippetPath(candidate) === path)!;
    const prelude = snippet.markdownLines
      ? []
      : getPrelude(missingNamesByPath.get(path) ?? new Map());
    preludeLengths.set(path, prelude.length);
    secondFiles.set(path, [...prelude, text].join('\n'));
  }
  const secondProgram = createVirtualProgram(secondFiles, options, firstProgram);
  const failures: {page: string; line: number; code: number; message: string}[] = [];
  for (const snippet of checked) {
    const path = getSnippetPath(snippet);
    const sourceFile = secondProgram.getSourceFile(path);
    const diagnostics = [
      ...secondProgram.getSyntacticDiagnostics(sourceFile),
      ...secondProgram.getSemanticDiagnostics(sourceFile),
      ...getFragmentInputDiagnostics(secondProgram, sourceFile!)
    ];
    for (const diagnostic of diagnostics) {
      const formatted = formatDiagnostic(diagnostic);
      const [, line, rest] = /^(\d+):\d+ (.*)$/s.exec(formatted) ?? [undefined, '0', formatted];
      const bodyLine = Number(line) - (preludeLengths.get(path) ?? 0);
      const docLine = snippet.markdownLines?.[bodyLine - 1] ?? snippet.fenceLine + bodyLine;
      failures.push({
        page: snippet.page,
        line: docLine,
        code: diagnostic.code,
        message: rest.split('\n')[0]
      });
    }
  }
  return failures;
}
