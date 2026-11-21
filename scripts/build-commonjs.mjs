// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {basename, dirname, resolve} from 'node:path';
import {build, transform} from 'esbuild';
import typescript from 'typescript';

const RELATIVE_JAVASCRIPT_SPECIFIER = /^(\.[^'"]+)\.js$/;

/**
 * Points relative `.js` module specifiers at their `.cjs` siblings. Only parsed specifiers of
 * static `import`/`export ... from` declarations and of `import()` calls change; string and
 * template values that merely look like imports (generated source, documentation, WGSL) keep
 * their bytes.
 */
export function rewriteRelativeModuleSpecifiers(source, sourceFile) {
  const parsed = typescript.createSourceFile(
    sourceFile,
    source,
    typescript.ScriptTarget.Latest,
    true,
    typescript.ScriptKind.JS
  );
  const replacements = [];
  function visit(node) {
    let specifier;
    if (typescript.isImportDeclaration(node) || typescript.isExportDeclaration(node)) {
      specifier = node.moduleSpecifier;
    } else if (
      typescript.isCallExpression(node) &&
      node.expression.kind === typescript.SyntaxKind.ImportKeyword
    ) {
      specifier = node.arguments[0];
    }
    if (specifier && typescript.isStringLiteral(specifier)) {
      // The literal's own characters between its quotes, so escapes and quote style survive.
      const start = specifier.getStart(parsed) + 1;
      const end = specifier.getEnd() - 1;
      const match = source.slice(start, end).match(RELATIVE_JAVASCRIPT_SPECIFIER);
      if (match) replacements.push({start, end, text: `${match[1]}.cjs`});
    }
    typescript.forEachChild(node, visit);
  }
  visit(parsed);
  let rewritten = source;
  for (const {start, end, text} of replacements.reverse()) {
    rewritten = `${rewritten.slice(0, start)}${text}${rewritten.slice(end)}`;
  }
  return rewritten;
}

/**
 * Transforms one emitted ESM file while preserving the package's shared module identities.
 * @param {string} source
 * @param {string} sourceFile
 * @param {string} [sourceMap] The compiler's map for the emitted JavaScript.
 */
export async function transformCommonJSModule(source, sourceFile, sourceMap) {
  const options = {
    format: 'cjs',
    platform: 'node',
    target: 'node16',
    // JSON imports remain native in CommonJS. Node 22 requires their type attribute,
    // even though the surrounding JavaScript can retain the older syntax target.
    supported: {'import-attributes': true}
  };
  if (sourceMap === undefined) {
    return transform(rewriteRelativeModuleSpecifiers(source, sourceFile), {
      ...options,
      sourcefile: sourceFile
    });
  }

  const sourcePath = resolve(sourceFile);
  const directory = dirname(sourcePath);
  const outputPath = sourcePath.replace(/\.js$/, '.cjs');
  const inlineMap = Buffer.from(sourceMap).toString('base64');
  const contents =
    source.replace(/(?:^|\n)\/\/# sourceMappingURL=[^\r\n]*(?:\r?\n)?$/, '') +
    `\n//# sourceMappingURL=data:application/json;base64,${inlineMap}\n`;
  const {outputFiles} = await build({
    ...options,
    absWorkingDir: directory,
    stdin: {contents, sourcefile: basename(sourcePath), resolveDir: directory},
    outfile: outputPath,
    sourcemap: 'linked',
    write: false,
    // Resolution hooks require bundle mode. Every import stays external, and keeping the source
    // text unchanged here lets esbuild compose the map without shifting its original columns.
    bundle: true,
    treeShaking: false,
    legalComments: 'inline',
    logLevel: 'silent',
    plugins: [
      {
        name: 'commonjs-relative-specifiers',
        setup(context) {
          context.onResolve({filter: /.*/}, ({path, kind}) => {
            const rewrite =
              (kind === 'import-statement' || kind === 'dynamic-import') &&
              RELATIVE_JAVASCRIPT_SPECIFIER.test(path);
            return {path: rewrite ? path.replace(/\.js$/, '.cjs') : path, external: true};
          });
        }
      }
    ]
  });
  return {
    code: outputFiles.find(file => file.path === outputPath).text,
    map: outputFiles.find(file => file.path === `${outputPath}.map`).text
  };
}
