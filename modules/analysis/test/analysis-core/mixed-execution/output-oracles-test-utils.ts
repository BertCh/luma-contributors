// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import type {
  GPUCommandGraph,
  GPUCommandGraphCopyNode,
  GraphDataView,
  GPUCPUVector,
  GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import {expect} from 'vitest';
import type {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';

/** CPU callback rows (distinct from the CPUVector transport object used by compiled graphs). */
export function makeCPUInputRows(
  format: GPUGraphVectorFormat,
  values: Float32Array | Uint32Array
): GPUCPUVector {
  const length = values.length / getGPUVectorFormatInfo(format).components;
  return {format, length, chunks: [{values, length}], pack: () => values};
}

/** Initialize every declared output to nonzero bytes before the contributor runs. */
export function poisonOperation<Operation extends GPUContributorOperation>(
  operation: Operation
): Operation {
  const createContributor = operation.createContributor.bind(operation);
  operation.createContributor = getView => {
    const contributor = createContributor(getView);
    const outputs = Object.keys(operation.outputs).map(port => getView(port));
    return {
      getCommandNodes: <Parameters>(graph: GPUCommandGraph<Parameters>) => [
        ...outputs.map((output, index) => createPoisonNode<Parameters>(output, `poison-${index}`)),
        ...contributor.getCommandNodes!(graph)
      ],
      destroy: () => (contributor as {destroy?: () => void}).destroy?.()
    };
  };
  return operation;
}

export function createPoisonNode<Parameters = void>(
  output: GraphDataView,
  id: string
): GPUCommandGraphCopyNode<Parameters> {
  return {
    id,
    type: 'copy',
    resources: [{buffer: output, usage: 'copy-destination'}],
    compile: ({device}) => {
      const size = output.length * output.rowByteLength;
      const poison = device.createBuffer({
        data: new Uint32Array(size / 4).fill(0x42f60000),
        usage: Buffer.COPY_SRC | Buffer.COPY_DST
      });
      return {
        encode: ({commandEncoder, getBuffer}) =>
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: poison,
            sourceOffset: 0,
            destinationBuffer: getBuffer(output),
            destinationOffset: output.byteOffset,
            size
          }),
        destroy: () => poison.destroy()
      };
    }
  };
}

/** Float32 tolerance includes NaN rejection; NaN literals explicitly denote missing results. */
export function expectLiteralRows(
  actual: ArrayLike<number>,
  expected: readonly number[],
  tolerance = 2e-5
) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, index) => {
    if (Number.isNaN(value)) expect(actual[index]).toBeNaN();
    else {
      expect(Number.isFinite(actual[index]), `row ${index}`).toBe(true);
      expect(Math.abs(actual[index] - value), `row ${index}`).toBeLessThanOrEqual(
        tolerance * Math.max(1, Math.abs(value))
      );
    }
  });
}
