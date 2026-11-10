// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Companion rasters for the index runs, so one picture can swipe an index against its relative.

import {
  getGPURasterArithmeticParameterValues,
  GPURasterArithmetic,
  type GPURasterBinaryOperation
} from '@luma-contributors/analysis/raster-analysis';
import type {RunBuilder, RunBuffer} from '../../../define-run';
import {sampleNdviSeries} from '../../../../data/ndvi';

/** (first - second) / (first + second) per cell, computed by GPURasterArithmetic. */
export function addNormalizedDifference(
  builder: RunBuilder,
  id: string,
  first: RunBuffer<'float32'>,
  second: RunBuffer<'float32'>
): RunBuffer<'float32'> {
  return addArithmetic(builder, id, 'normalizedDifference', first, second);
}

/** One GPURasterArithmetic pass over two float rasters. */
export function addArithmetic(
  builder: RunBuilder,
  id: string,
  operation: GPURasterBinaryOperation,
  first: RunBuffer<'float32'>,
  second: RunBuffer<'float32'>
): RunBuffer<'float32'> {
  const parameters = builder.upload(
    `${id}-parameters`,
    'float32',
    getGPURasterArithmeticParameterValues({
      operation,
      scaleB: 1,
      clampMaximum: 1
    })
  );
  const output = builder.allocate(id, 'float32', first.length);
  builder.graph.add(
    new GPURasterArithmetic({
      cellCount: first.length,
      a: first.view,
      b: second.view,
      parameters: parameters.view,
      output: {values: output.view}
    })
  );
  return output;
}

/**
 * The same scene after the Dixie Fire, as a scenario: cells whose measured NDVI fell between the
 * July and September 2021 dates lose near-infrared and gain shortwave-infrared 2 in proportion.
 */
export function createPostFireBands(
  columns: number,
  rows: number,
  nearInfrared: Float32Array,
  shortwaveInfrared2: Float32Array
) {
  const series = sampleNdviSeries(columns, rows, [7, 8]);
  const burn = Float32Array.from(nearInfrared, (_, cell) => {
    const drop = series[cell * 2] - series[cell * 2 + 1];
    return Number.isFinite(drop) ? Math.min(1, Math.max(0, drop / 0.4)) : 0;
  });
  return {
    burn,
    nearInfrared: nearInfrared.map((value, cell) => value * (1 - 0.6 * burn[cell])),
    shortwaveInfrared2: shortwaveInfrared2.map(
      (value, cell) => value * (1 + 1.5 * burn[cell]) + 0.04 * burn[cell]
    )
  };
}
