// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// biome-ignore-all format: the shown example is hand-formatted to fit the 25-line card limit

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUScalarScale, prepareGPUScalarScaleControls} from '@luma-contributors/analysis/dataframe-analysis';

/** Tract columns the lens scenario reads from `f-chicago-tracts.json`; suppressed estimates are NaN. */
export type ScalarScaleExampleInput = {
  graph: GPUCommandGraph<undefined>;
  tracts: {poverty150Pct: Float32Array};
};

/** Poverty share mapped to a point radius in pixels with a square-root scale. */
export async function runExample(device: Device, {graph, tracts}: ScalarScaleExampleInput) {
  const buffers: Buffer[] = [];
  const importColumn = <Format extends 'float32' | 'uint32'>(
    id: string,
    data: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = device.createBuffer({
      id,
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format, length: data.length});
  };
  // #region example
  const poverty = importColumn('poverty150Pct', tracts.poverty150Pct, 'float32');
  const controls = prepareGPUScalarScaleControls({scale: 'sqrt', domain: [0, 100], range: [1, 20]});
  const radius = createTransientView(graph, 'radius', 'float32', poverty.length, Buffer.STORAGE | Buffer.COPY_SRC);
  const radiusValidity = createTransientView(graph, 'radiusValidity', 'uint32', poverty.length, Buffer.STORAGE | Buffer.COPY_SRC);
  graph.add(new GPUScalarScale({
    values: poverty,
    domain: importColumn('domain', controls.domain, 'float32'),
    range: importColumn('range', controls.range, 'float32'),
    parameters: importColumn('parameters', controls.parameters, 'float32'),
    maximumDomainCount: 2,
    output: {values: radius, validity: radiusValidity}
  }));
  // #endregion example
  return {
    poverty,
    radius,
    radiusValidity,
    destroy() {
      for (const buffer of buffers) buffer.destroy();
    }
  };
}