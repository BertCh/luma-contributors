// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-lag';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';

export const LAG_AUDIT_EXPECTED = {
  masked: {
    sum: [37, 50, 0, 0, 105, 118, 8, 16, 22, 24, 27, 30, 0, 0, 0, 0],
    normalized: [
      2.8461538461538463, 3.8461538461538463, 0, 0, 8.076923076923077, 9.076923076923077, 1, 2, 11,
      12, 9, 10, 0, 0, 0, 0
    ]
  },
  shifted: {
    sum: [176, 215, 0, 0, 380, 419, 64, 88, 76, 82, 96, 105, 0, 0, 0, 0],
    normalized: [
      13.538461538461538, 16.53846153846154, 0, 0, 29.23076923076923, 32.23076923076923, 8, 11, 38,
      41, 32, 35, 0, 0, 0, 0
    ]
  },
  zeroWeight: {
    sum: [37, 50, 0, 0, 105, 118, 8, 16, 22, 24, 0, 0, 0, 0, 0, 0],
    normalized: [
      2.8461538461538463, 3.8461538461538463, 0, 0, 8.076923076923077, 9.076923076923077, 1, 2, 11,
      12, 0, 0, 0, 0, 0, 0
    ]
  },
  rectangular: {
    sum: [75, 90, 0, 0, 105, 118, 8, 16, 22, 24, 27, 30, 0, 0, 0, 0],
    normalized: [5, 6, 0, 0, 8.076923076923077, 9.076923076923077, 1, 2, 11, 12, 9, 10, 0, 0, 0, 0]
  }
};

export type LagAuditOptions = {
  scenario: keyof typeof LAG_AUDIT_EXPECTED;
  normalize: boolean;
  columnCount: 1 | 2;
  valueFormat: 'float32' | 'uint32';
  chunked: boolean;
  lossLocation?: 'outer' | 'chunk';
  lossKind?: 'overflow' | 'candidateOverflow';
};
type InputFormat = 'uint32' | 'float32';
type LagAuditPorts = {
  input: <Format extends InputFormat>(
    data: Uint32Array | Float32Array,
    format: Format
  ) => GraphDataView<Format>;
  output: (length: number) => GraphDataView<'float32'>;
};

/** Inputs and literal expectations from audit-weights-a-lag.oracle.py; no implementation readback. */
export function createLagAuditCase(ports: LagAuditPorts, options: LagAuditOptions) {
  const rectangular = options.scenario === 'rectangular';
  const offsets = rectangular ? [0, 4, 4, 6, 7, 8, 9, 9, 9] : [0, 3, 5, 7, 8, 9, 10, 10, 10];
  const neighbors = rectangular ? [0, 3, 9, 11, 3, 4, 0, 5, 4] : [0, 1, 3, 0, 2, 3, 4, 0, 5, 4];
  const weights = rectangular
    ? [9, 4, 2, 999, 6, 7, 8, 2, 3]
    : [9, 2, 4, 3, 5, 6, 7, 8, 2, options.scenario === 'zeroWeight' ? 0 : 3];
  const valueRows = rectangular ? 10 : 8;
  const values = Array.from({length: valueRows * 2}, (_, index) =>
    options.scenario === 'shifted' ? 3 * (index + 1) + 5 : index + 1
  ).filter((_, index) => options.columnCount === 2 || index % 2 === 0);
  const expected = LAG_AUDIT_EXPECTED[options.scenario][
    options.normalize ? 'normalized' : 'sum'
  ].filter((_, index) => options.columnCount === 2 || index % 2 === 0);
  const valueView = ports.input(
    options.valueFormat === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values),
    options.valueFormat
  );
  const loss = options.lossLocation ? ports.input(Uint32Array.of(0), 'uint32') : undefined;
  const status = loss ? {[options.lossKind ?? 'overflow']: loss} : undefined;
  const upload = (rowStart: number, rowCount: number): GPUSpatialWeights => {
    const begin = offsets[rowStart];
    const end = offsets[rowStart + rowCount];
    return {
      offsets: ports.input(
        Uint32Array.from(offsets.slice(rowStart, rowStart + rowCount + 1), value => value - begin),
        'uint32'
      ),
      // Unused slots are plausible neighbors with large weights: a capacity loop must fail.
      neighbors: ports.input(Uint32Array.from([...neighbors.slice(begin, end), 0]), 'uint32'),
      weights: ports.input(Float32Array.from([...weights.slice(begin, end), 777]), 'float32'),
      status: options.lossLocation === 'chunk' && rowStart === 2 ? status : undefined
    };
  };
  const weightsInput = options.chunked
    ? {
        rowCount: 8,
        status: options.lossLocation === 'outer' ? status : undefined,
        // Final chunk is wholly empty; middle chunk starts at a masked global row boundary.
        chunks: [
          {rowStart: 0, rowCount: 2},
          {rowStart: 2, rowCount: 3},
          {rowStart: 5, rowCount: 1},
          {rowStart: 6, rowCount: 2}
        ].map(range => ({...range, weights: upload(range.rowStart, range.rowCount)}))
      }
    : {...upload(0, 8), status};
  const output = ports.output(8 * options.columnCount);
  return {
    contributor: new GPUSpatialLag({
      weights: weightsInput,
      values: valueView,
      columnCount: options.columnCount,
      normalize: options.normalize,
      mask: rectangular ? undefined : ports.input(Uint32Array.of(1, 0, 2, 1, 1, 1, 1, 1), 'uint32'),
      output
    }),
    output,
    expected,
    loss
  };
}
