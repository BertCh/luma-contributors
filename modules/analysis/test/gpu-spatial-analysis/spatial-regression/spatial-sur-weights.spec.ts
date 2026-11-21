// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getSpatialSURWeightValidationNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-sur-nodes';
import {readFloat32} from '../../utils/gpu-contributor-test-utils';
import {createRegressionTestContext} from './math-review-regression-test-utils';

for (const scenario of [
  'valid',
  'empty-row',
  'self-link',
  'duplicate',
  'unsorted',
  'out-of-range',
  'negative',
  'nonfinite',
  'not-standardized',
  'bad-offset',
  'leading-slot',
  'overflow',
  'candidate-overflow'
] as const) {
  it(`tiled spatial SUR weights preserve strict validation and ignore unused slots: ${scenario}`, async () => {
    const context = await createRegressionTestContext();
    const {graph, input, output} = context;
    try {
      const rowCount = 129;
      const offsets = Uint32Array.from({length: rowCount + 1}, (_, row) => row * 2);
      const neighbors = new Uint32Array(rowCount * 2 + 3).fill(0xffffffff);
      const weights = new Float32Array(neighbors.length).fill(NaN);
      for (let row = 0; row < rowCount; row++) {
        const candidates = [(row + 1) % rowCount, (row + 2) % rowCount].sort(
          (first, second) => first - second
        );
        neighbors.set(candidates, row * 2);
        weights.set([0.5, 0.5], row * 2);
      }
      // Corrupt the partial final tile so an error cannot be lost at a tile boundary.
      const slot = (rowCount - 1) * 2;
      switch (scenario) {
        case 'leading-slot':
          neighbors.copyWithin(1, 0, rowCount * 2);
          weights.copyWithin(1, 0, rowCount * 2);
          for (let row = 0; row <= rowCount; row++) offsets[row]++;
          break;
        case 'empty-row':
          offsets[rowCount] = slot;
          break;
        case 'self-link':
          neighbors[slot + 1] = rowCount - 1;
          break;
        case 'duplicate':
          neighbors[slot + 1] = neighbors[slot];
          break;
        case 'unsorted':
          neighbors.set([1, 0], slot);
          break;
        case 'out-of-range':
          neighbors[slot + 1] = rowCount;
          break;
        case 'negative':
          weights.set([-0.5, 1.5], slot);
          break;
        case 'nonfinite':
          weights[slot] = NaN;
          break;
        case 'not-standardized':
          weights[slot] = 0.25;
          break;
        case 'bad-offset':
          offsets[rowCount] = neighbors.length + 1;
          break;
      }
      const sliceInput = <Format extends 'float32' | 'uint32'>(
        format: Format,
        values: ArrayLike<number>
      ) => {
        const padded =
          format === 'uint32'
            ? new Uint32Array(values.length + 2)
            : new Float32Array(values.length + 2);
        padded.set(Array.from(values), 1);
        const source = input(format, padded);
        return graph.createDataView(source.buffer, {
          format,
          byteOffset: source.byteOffset + 4,
          length: values.length
        });
      };
      const invalid = output('float32', 3);
      invalid.buffer.write(Float32Array.of(99, 99, 99));
      const invalidView = graph.createDataView(invalid.view.buffer, {
        format: 'float32',
        byteOffset: invalid.view.byteOffset + 4,
        length: 1
      });
      for (const node of getSpatialSURWeightValidationNodes(graph, {
        id: 'validate-sur',
        rowCount,
        weights: {
          offsets: sliceInput('uint32', offsets),
          neighbors: sliceInput('uint32', neighbors),
          weights: sliceInput('float32', weights),
          status:
            scenario === 'overflow'
              ? {overflow: sliceInput('uint32', Uint32Array.of(1))}
              : scenario === 'candidate-overflow'
                ? {candidateOverflow: sliceInput('uint32', Uint32Array.of(1))}
                : undefined
        },
        output: invalidView
      }))
        graph.add(node);
      context.run();
      expect(Array.from(await readFloat32(invalid.buffer, 3))).toEqual([
        99,
        Number(scenario !== 'valid'),
        99
      ]);
    } finally {
      context.destroy();
    }
  });
}
