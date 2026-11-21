// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {Worker} from 'node:worker_threads';
import {ModuleKind, transpileModule} from 'typescript';
import {expect, it} from 'vitest';

it('R3 rank-deficient and wide singular-value ranges terminate before the worker deadline', async () => {
  // Transpile the actual pure helper into an isolated worker. A timeout terminates the worker,
  // so a recurrence of the old zero-tangent loop cannot hang Vitest's own execution thread.
  const source = readFileSync(
    new URL(
      '../../../src/gpu-spatial-analysis/spatial-regression/cpu-regression-linear-algebra.ts',
      import.meta.url
    ),
    'utf8'
  );
  const code = transpileModule(source, {compilerOptions: {module: ModuleKind.CommonJS}}).outputText;
  const worker = new Worker(
    `const {parentPort}=require('node:worker_threads');\n${code}\nparentPort.postMessage([exports.getCPURegressionSingularValueRange([[1,0,1],[0,1,1],[0,0,0]]),exports.getCPURegressionSingularValueRange([[1,1]])]);`,
    {eval: true}
  );
  try {
    const result = await new Promise<number[][]>((resolve, reject) => {
      const deadline = setTimeout(() => {
        void worker.terminate();
        reject(new Error('Singular-value helper exceeded 2000 ms deadline'));
      }, 2000);
      worker.once('message', values => {
        clearTimeout(deadline);
        resolve(values);
      });
      worker.once('error', error => {
        clearTimeout(deadline);
        reject(error);
      });
    });
    // AAᵀ has eigenvalues 0,1,3; the one-row matrix has its sole singular value sqrt(2).
    const expected = [
      [0, Math.sqrt(3)],
      [Math.sqrt(2), Math.sqrt(2)]
    ];
    result.forEach((range, index) =>
      range.forEach((value, slot) => {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeCloseTo(expected[index][slot], 13);
      })
    );
  } finally {
    await worker.terminate();
  }
});
