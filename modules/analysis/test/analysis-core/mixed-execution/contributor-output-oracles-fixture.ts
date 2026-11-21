// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraphCopyNode} from '@luma.gl/graph/gpu-core';
import {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';

export function createCopyOperation() {
  return new GPUContributorOperation({
    id: 'literal-copy',
    inputs: {left: 'leftInput', right: 'rightInput', poison: 'poisonInput'},
    outputs: {
      first: {value: 'firstResult', format: 'uint32', length: 3},
      second: {value: 'secondResult', format: 'uint32', length: 3}
    },
    createContributor: getView => {
      const copy = <Parameters>(
        source: string,
        destination: string,
        id: string
      ): GPUCommandGraphCopyNode<Parameters> => {
        const input = getView<'uint32'>(source);
        const output = getView<'uint32'>(destination);
        return {
          id,
          type: 'copy',
          resources: [
            {buffer: input, usage: 'copy-source'},
            {buffer: output, usage: 'copy-destination'}
          ],
          compile: () => ({
            encode: ({commandEncoder, getBuffer}) =>
              commandEncoder.copyBufferToBuffer({
                sourceBuffer: getBuffer(input),
                sourceOffset: input.byteOffset,
                destinationBuffer: getBuffer(output),
                destinationOffset: output.byteOffset,
                size: 12
              })
          })
        };
      };
      return {
        getCommandNodes: <Parameters>() => [
          copy<Parameters>('poison', 'first', 'poison-first'),
          copy<Parameters>('poison', 'second', 'poison-second'),
          copy<Parameters>('left', 'first', 'copy-first'),
          copy<Parameters>('right', 'second', 'copy-second')
        ]
      };
    }
  });
}
