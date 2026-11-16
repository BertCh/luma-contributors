// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  createTransientView,
  GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getGraphViewChunks} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createDensityPointPass} from './density-point-passes';

/** Fixed-width packed point tiles keep the published bin/group pipelines independent of tails. */
export function createPointDensityTiles<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  positions: GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>,
  weights: GraphDataView<'float32'> | GraphVectorView<'float32'> | undefined,
  mask: GraphDataView<'uint32'> | undefined,
  tileLength: number
): {
  positions: GraphVectorView<'float32x2'>;
  weights?: GraphVectorView<'float32'>;
  nodes: GPUCommandNode<Parameters>[];
} {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const pointTiles: GraphDataView<'float32x2'>[] = [];
  const weightTiles: GraphDataView<'float32'>[] = [];
  for (let tileIndex = 0; tileIndex < Math.ceil(positions.length / tileLength); tileIndex++) {
    const tileId = `${id}-tile-${tileIndex}`;
    const begin = tileIndex * tileLength;
    const end = Math.min(begin + tileLength, positions.length);
    const inputs: GraphDataView[] = [];
    for (const [name, source, padding] of [
      ['positions', positions, 0x7fc00000],
      ['weights', weights, 0],
      ['mask', mask, 0]
    ] as const) {
      if (!source) continue;
      const destination = createTransientView(
        graph,
        `${tileId}-${name}`,
        source.format,
        tileLength
      );
      if (name === 'positions') pointTiles.push(destination as GraphDataView<'float32x2'>);
      if (name === 'weights') weightTiles.push(destination as GraphDataView<'float32'>);
      inputs.push(destination);
      const components = source.rowByteLength / 4;
      nodes.push(
        createDensityPointPass(graph, {
          id: `${tileId}-${name}-pad`,
          operation: 'GPUPointDensity',
          invocationCount: tileLength,
          bindings: [{name: 'destination', view: destination, type: 'u32', access: 'read_write'}],
          body: Array.from(
            {length: components},
            (_, component) =>
              `destination[destinationOffset + ${components}u * index + ${component}u] = ${padding}u;`
          ).join('\n')
        })
      );
      let sourceBegin = 0;
      for (const [chunkIndex, chunk] of getGraphViewChunks(source).entries()) {
        const sourceEnd = sourceBegin + chunk.length;
        const first = Math.max(begin, sourceBegin);
        const last = Math.min(end, sourceEnd);
        if (first < last) {
          const span = graph.createDataView(chunk.buffer, {
            format: chunk.format,
            length: last - first,
            byteOffset: chunk.byteOffset + (first - sourceBegin) * chunk.rowByteLength
          });
          nodes.push(
            createDensityPointPass(graph, {
              id: `${tileId}-${name}-gather-${chunkIndex}`,
              operation: 'GPUPointDensity',
              invocationCount: last - first,
              bindings: [
                {name: 'source', view: span, type: 'u32', access: 'read'},
                {name: 'destination', view: destination, type: 'u32', access: 'read_write'}
              ],
              runtimeConstants: {DESTINATION_ROW_BASE: first - begin},
              body: Array.from(
                {length: components},
                (_, component) =>
                  `destination[destinationOffset + ${components}u * (DESTINATION_ROW_BASE + index) + ${component}u] = source[sourceOffset + ${components}u * index + ${component}u];`
              ).join('\n')
            })
          );
        }
        sourceBegin = sourceEnd;
      }
    }
    if (mask)
      nodes.push(
        createDensityPointPass(graph, {
          id: `${tileId}-mask`,
          operation: 'GPUPointDensity',
          invocationCount: tileLength,
          bindings: [
            {name: 'positions', view: pointTiles[tileIndex], type: 'u32', access: 'read_write'},
            {name: 'mask', view: inputs[inputs.length - 1], type: 'u32', access: 'read'}
          ],
          body: 'if (mask[maskOffset + index] == 0u) { positions[positionsOffset + 2u * index] = 0x7fc00000u; positions[positionsOffset + 2u * index + 1u] = 0x7fc00000u; }'
        })
      );
  }
  const length = pointTiles.length * tileLength;
  return {
    positions: new GraphVectorView({
      id: `${id}-position-tiles`,
      name: 'density point tiles',
      format: 'float32x2',
      length,
      valueLength: length * 2,
      stride: 2,
      byteStride: 8,
      rowByteLength: 8,
      data: pointTiles
    }),
    weights: weights
      ? new GraphVectorView({
          id: `${id}-weight-tiles`,
          name: 'density weight tiles',
          format: 'float32',
          length,
          valueLength: length,
          stride: 1,
          byteStride: 4,
          rowByteLength: 4,
          data: weightTiles
        })
      : undefined,
    nodes
  };
}
