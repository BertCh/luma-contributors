// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Some recipes keep their results in graph-owned views. GPUElementwise copies such a scalar view
// into a storage buffer the picture can draw; values pass through unchanged.

import {GPUElementwise, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';

export function addReadout(
  builder: RunBuilder,
  id: string,
  source: GraphDataView<'float32'>
): RunBuffer<'float32'>;
export function addReadout(
  builder: RunBuilder,
  id: string,
  source: GraphDataView<'uint32'>
): RunBuffer<'uint32'>;
export function addReadout(
  builder: RunBuilder,
  id: string,
  source: GraphDataView<'float32'> | GraphDataView<'uint32'>
): RunBuffer<'float32'> | RunBuffer<'uint32'> {
  if (source.format === 'uint32') {
    const copy = builder.allocate(`${id}-readout`, 'uint32', source.length);
    builder.graph.add(
      new GPUElementwise({
        id: `${id}-copy`,
        operation: 'copy',
        input: source as GraphDataView<'uint32'>,
        output: copy.view
      })
    );
    return copy;
  }
  const copy = builder.allocate(`${id}-readout`, 'float32', source.length);
  builder.graph.add(
    new GPUElementwise({
      id: `${id}-copy`,
      operation: 'copy',
      input: source as GraphDataView<'float32'>,
      output: copy.view
    })
  );
  return copy;
}
