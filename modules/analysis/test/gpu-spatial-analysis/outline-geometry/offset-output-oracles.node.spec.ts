// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPUOffsetCurve,
  type GPUOffsetCurveCompactOutput,
  type GPUOffsetCurveProps
} from '../../../src/gpu-spatial-analysis/outline-geometry/gpu-offset-curve';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createOffsetOutputOracle} from './offset-output-oracles';

useCPUExecutorTestHooks();

function checkRejectedCompactOffsetOutput(
  output: GPUOffsetCurveCompactOutput,
  scalar: GPUOffsetCurveCompactOutput['status']['count']
) {
  // @ts-expect-error Compact offset never writes approximation; untyped callers are refused too.
  output.status.approximation = scalar;
}
void checkRejectedCompactOffsetOutput;

function checkPreparedCompactOffsetOutput(
  props: GPUOffsetCurveProps & {output: GPUOffsetCurveCompactOutput},
  scalar: GPUOffsetCurveCompactOutput['status']['count']
) {
  const supportedOutput = {...props.output, status: {...props.output.status}};
  new GPUOffsetCurve({...props, output: supportedOutput}); // Prepared supported shape remains assignable.
  const output = {...props.output, status: {...props.output.status, approximation: scalar}};
  // @ts-expect-error Prepared variables cannot supply a refused approximation view.
  new GPUOffsetCurve({...props, output});
}
void checkPreparedCompactOffsetOutput;
it('publishes hand-derived compact requirements and refuses an unwritten approximation view', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const optional of [undefined, 'vertexCount', 'requiredVertexCount'] as const) {
      const fixture = createOffsetOutputOracle(device, optional);
      try {
        const memory = createCPUGraphMemory();
        for (const entry of fixture.columns) memory.write(entry.view, entry.values);
        for (const distance of [1, -2]) {
          memory.write(fixture.parameters, new Float32Array([distance, 5, 0, 0]));
          executeCommandNodesOnCPU(fixture.nodes, memory);
          const actual = Object.fromEntries(
            fixture.columns
              .filter(entry => entry.output)
              .map(entry => [
                entry.name,
                Array.from(
                  entry.values instanceof Uint32Array
                    ? memory.readUint32(entry.view)
                    : memory.readFloat32(entry.view)
                )
              ])
          );
          expect(actual).toEqual({...fixture.expected, positions: [0, distance]});
        }
        const props = fixture.contributor.props;
        expect(
          () =>
            new GPUOffsetCurve({
              ...props,
              output: {
                ...props.output,
                status: {
                  ...('status' in props.output ? props.output.status : {}),
                  approximation: fixture.output('approximation', 'uint32', 1)
                }
              }
            } as never)
        ).toThrow(/output.status.approximation/);
      } finally {
        fixture.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});
