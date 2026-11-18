// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {transform} from 'esbuild';

/** Transforms one emitted ESM file while preserving the package's shared module identities. */
export async function transformCommonJSModule(source, sourceFile) {
  const rewrittenSource = source.replace(
    /((?:from\s+|import\s*(?:\(\s*)?)['"])(\.[^'"]+)\.js(['"])/g,
    '$1$2.cjs$3'
  );
  return transform(rewrittenSource, {
    format: 'cjs',
    platform: 'node',
    target: 'node16',
    // JSON imports remain native in CommonJS. Node 22 requires their type attribute,
    // even though the surrounding JavaScript can retain the older syntax target.
    supported: {'import-attributes': true},
    sourcefile: sourceFile
  });
}
