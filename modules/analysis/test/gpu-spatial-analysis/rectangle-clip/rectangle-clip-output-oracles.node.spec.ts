// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPURectangleClip,
  type GPURectangleClipOutput,
  type GPURectangleClipProps
} from '../../../src/gpu-spatial-analysis/rectangle-clip/gpu-rectangle-clip';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();

function checkRejectedClipOutputs(
  output: GPURectangleClipOutput,
  scalar: GraphDataView<'uint32'>,
  measures: GraphDataView<'float32'>
) {
  // @ts-expect-error Clipped vertices have no supported input-row output.
  output.sourceRows = scalar;
  // @ts-expect-error Clip has no measure input/output interpolation contract.
  output.measures = measures;
  // @ts-expect-error Clip does not write this diagnostic.
  output.status.approximation = scalar;
}
void checkRejectedClipOutputs;

function checkPreparedClipOutputs(
  props: GPURectangleClipProps,
  scalar: GraphDataView<'uint32'>,
  measures: GraphDataView<'float32'>
) {
  const supportedOutput = {...props.output, status: {...props.output.status}};
  new GPURectangleClip({...props, output: supportedOutput}); // Prepared supported shape remains assignable.
  const rowOutput = {...props.output, sourceRows: scalar};
  // @ts-expect-error Prepared output variables cannot supply a refused sourceRows view.
  new GPURectangleClip({...props, output: rowOutput});
  const measuredOutput = {...props.output, measures};
  // @ts-expect-error Prepared output variables cannot supply a refused measures view.
  new GPURectangleClip({...props, output: measuredOutput});
  const approximateOutput = {
    ...props.output,
    status: {...props.output.status, approximation: scalar}
  };
  // @ts-expect-error Prepared nested status variables cannot supply approximation.
  new GPURectangleClip({...props, output: approximateOutput});
}
void checkPreparedClipOutputs;
it('refuses unwritten row, measure and approximation outputs and clips exactly with them absent', () => {
  const device = createNullWebGPUDevice();
  const fixture = createOutputOracleFixture(device);
  const {graph, column, output} = fixture;
  const props = {
    geometryType: 'lines' as const,
    positions: column('input', 'float32x2', new Float32Array([-2, 0, 6, 0])),
    lineOffsets: column('lines', 'uint32', new Uint32Array([0, 2])),
    parameters: column('parameters', 'float32', new Float32Array([0, -1, 4, 1])),
    output: {
      positions: output('positions', 'float32x2', 2),
      lineOffsets: output('offsets', 'uint32', 2),
      status: {count: output('count', 'uint32', 1), overflow: output('overflow', 'uint32', 1)}
    }
  };
  try {
    for (const name of ['sourceRows', 'measures', 'approximation'] as const) {
      const view = output(
        `unsupported-${name}`,
        name === 'measures' ? 'float32' : 'uint32',
        name === 'approximation' ? 1 : 2
      );
      const result =
        name === 'approximation'
          ? {...props.output, status: {...props.output.status, approximation: view}}
          : {...props.output, [name]: view};
      expect(() => new GPURectangleClip({...props, output: result} as never)).toThrow(
        new RegExp(`output\\.${name === 'approximation' ? 'status\\.' : ''}${name}`)
      );
    }
    const memory = createCPUGraphMemory();
    for (const entry of fixture.columns) memory.write(entry.view, entry.values);
    executeCommandNodesOnCPU(new GPURectangleClip(props).getCommandNodes(graph), memory);
    // x=-2+8t: rectangle retains t=1/4 to 3/4, endpoints (0,0) and (4,0).
    expect(Array.from(memory.readFloat32(props.output.positions))).toEqual([0, 0, 4, 0]);
    expect(Array.from(memory.readUint32(props.output.lineOffsets))).toEqual([0, 2]);
    expect(Array.from(memory.readUint32(props.output.status.count))).toEqual([1]);
    expect(Array.from(memory.readUint32(props.output.status.overflow))).toEqual([0]);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
