// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {GPUEmergingHotSpots} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import type {AnalysisRig} from '../catchment-accessibility/rig';

export function createCompositionOutputs(rig: AnalysisRig, slices = 2) {
  return {
    giZScores: rig.output('float32', 2 * slices),
    trendZ: rig.output('float32', 2),
    trendP: rig.output('float32', 2),
    trendS: rig.output('sint32', 2),
    category: rig.output('uint32', 2),
    hotSliceCount: rig.output('uint32', 2),
    coldSliceCount: rig.output('uint32', 2),
    globalStatistics: rig.output('float32', 4),
    includedCount: rig.output('uint32', 1)
  };
}

export function createSparseGatherComposition(rig: AnalysisRig, unknown: boolean) {
  // Fixed dictionary [10,20] in weights order, periods [0,1]. Composite keys make
  // identity AND time explicit; the missing (10,0) bin is either observed empty or unknown.
  const gathered = unknown ? rig.output('float32', 4) : rig.output('uint32', 4);
  const column = unknown
    ? rig.input(Float32Array.of(3, 1, 2), 'float32')
    : rig.input(Uint32Array.of(3, 1, 2), 'uint32');
  const join = new GPUKeyLookup({
    id: 'sparse-cube-join',
    leftKeys: rig.input(Uint32Array.of(100, 101, 200, 201), 'uint32'),
    rightKeys: rig.input(Uint32Array.of(200, 101, 201), 'uint32'),
    gather: [{column, output: gathered, unmatchedValue: unknown ? NaN : 0}],
    output: {rightRows: rig.output('uint32', 4)}
  });
  const sliceValues = unknown ? rig.output('float32', 2) : rig.output('uint32', 2);
  const sliceJoin = new GPUKeyLookup({
    id: 'sparse-slice-join',
    leftKeys: rig.input(Uint32Array.of(100, 200), 'uint32'),
    rightKeys: rig.input(Uint32Array.of(200, 101, 201), 'uint32'),
    gather: [{column, output: sliceValues, unmatchedValue: unknown ? NaN : 0}],
    output: {}
  });
  const sliceOutput = createCompositionOutputs(rig, 1);
  const output = createCompositionOutputs(rig);
  const explicitOutput = createCompositionOutputs(rig);
  const parameters = rig.input(Float32Array.of(0, 0, 1.96, 0.05, 0.9, 2, 0.05, 0), 'float32');
  const weights = rig.weights({offsets: [0, 0, 0], neighbors: [], weights: []});
  const producers = [
    join,
    sliceJoin,
    new GPUEmergingHotSpots({
      id: 'slice-hot-spots',
      values: sliceValues,
      weights,
      sliceCount: 1,
      multipleTesting: 'none',
      parameters,
      output: sliceOutput
    }),
    new GPUEmergingHotSpots({
      id: 'gathered-hot-spots',
      values: gathered,
      weights,
      sliceCount: 2,
      multipleTesting: 'none',
      parameters,
      output
    }),
    new GPUEmergingHotSpots({
      id: 'explicit-hot-spots',
      values: rig.input(Float32Array.of(unknown ? NaN : 0, 1, 3, 2), 'float32'),
      weights,
      sliceCount: 2,
      multipleTesting: 'none',
      parameters,
      output: explicitOutput
    })
  ];
  return {producers, gathered, sliceOutput, output, explicitOutput};
}
