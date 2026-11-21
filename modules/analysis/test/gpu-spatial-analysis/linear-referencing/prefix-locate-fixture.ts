// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPULineLocate} from '../../../src/gpu-spatial-analysis/linear-referencing/index';
import type {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';

/** Shared by null-device declaration checks and the actual device consumer regression. */
export function addPrefixLocateConsumers(
  fixture: ReturnType<typeof createOutputOracleFixture>,
  positions: GraphDataView<'float32x2'>,
  lineOffsets: GraphDataView<'uint32'>,
  rowCount: number,
  total: number
) {
  const {graph, input, output} = fixture;
  return ['distance', 'fraction'].map(mode => {
    const position = output(`${mode}-position`, 'float32x2', 1);
    const vertexMeasures = output(`${mode}-prefix`, 'float32', rowCount);
    const statuses = output(`${mode}-status`, 'uint32', 1);
    graph.add(
      new GPULineLocate({
        id: `locate-${mode}`,
        positions,
        lineOffsets,
        eventLineIndices: input(`${mode}-line`, 'uint32', Uint32Array.of(0)),
        eventMeasures: input(
          `${mode}-measure`,
          'float32',
          Float32Array.of(mode === 'fraction' ? 1 : total)
        ),
        measureMode: mode === 'fraction' ? 'fraction' : 'distance',
        output: {positions: position, vertexMeasures, statuses}
      })
    );
    return {position, vertexMeasures, statuses};
  });
}
