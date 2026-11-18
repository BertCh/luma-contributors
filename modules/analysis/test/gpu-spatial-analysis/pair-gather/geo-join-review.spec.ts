// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUBufferSelection} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {
  GPUNearestFeatureJoin,
  GPUNearestFeatureWeights,
  GPUSpatialPredicateJoin
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';

for (const metric of ['planar', 'great-circle'] as const) {
  for (const sourceId of [100, 0xffffffff]) {
    it(`R5 ${metric} selection uses row zero for feature ID ${sourceId} and clears on reuse`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createGeometryFixture(device, {
        inputs: {
          points: {format: 'float32x2', values: Float32Array.of(0, 0)},
          features: {format: 'float32x2', values: Float32Array.of(0, 0)},
          queryIds: {format: 'uint32', values: Uint32Array.of(42)},
          featureIds: {format: 'uint32', values: Uint32Array.of(sourceId)},
          radius: {format: 'float32', values: Float32Array.of(1)}
        },
        outputs: {
          rows: {format: 'uint32', length: 1},
          mask: {format: 'uint32', length: 1},
          selected: {format: 'uint32', length: 1},
          count: {format: 'uint32', length: 1},
          overflow: {format: 'uint32', length: 1}
        },
        create: ({inputs, outputs}) =>
          new GPUBufferSelection({
            spatialContext:
              metric === 'planar'
                ? {coordinateSpace: 'planar', metric: 'native'}
                : {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
            points: {
              kind: 'points',
              positions: inputs['points'] as GraphDataView<'float32x2'>,
              sourceIds: inputs['queryIds'] as GraphDataView<'uint32'>
            },
            features: {
              kind: 'points',
              positions: inputs['features'] as GraphDataView<'float32x2'>,
              sourceIds: inputs['featureIds'] as GraphDataView<'uint32'>
            },
            distance: inputs['radius'] as GraphDataView<'float32'>,
            candidateCapacity: 1,
            output: {
              nearestFeatureRows: outputs['rows'] as GraphDataView<'uint32'>,
              outputMask: outputs['mask'] as GraphDataView<'uint32'>,
              matches: {
                ids: outputs['selected'] as GraphDataView<'uint32'>,
                status: {
                  count: outputs['count'] as GraphDataView<'uint32'>,
                  overflow: outputs['overflow'] as GraphDataView<'uint32'>
                }
              }
            }
          })
      });
      try {
        const matched = await fixture.run();
        expect(matched['rows']).toEqual([0]);
        expect(matched['mask']).toEqual([1]);
        expect(matched['count']).toEqual([1]);
        expect(matched['selected']).toEqual([42]);
        expect(matched['overflow']).toEqual([0]);
        fixture.writeInput('features', Float32Array.of(10, 0));
        const empty = await fixture.run();
        expect(empty['rows']).toEqual([0xffffffff]);
        expect(empty['mask']).toEqual([0]);
        expect(empty['count']).toEqual([0]);
        expect(empty['overflow']).toEqual([0]);
      } finally {
        fixture.destroy();
      }
    });
  }
}

it('R6 two winning segment rows with source IDs [7,7] produce distinct CSR columns [0,1]', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      queries: {format: 'float32x2', values: Float32Array.of(0, 0)},
      starts: {format: 'float32x2', values: Float32Array.of(-1, 0, 0, -1)},
      ends: {format: 'float32x2', values: Float32Array.of(1, 0, 0, 1)},
      sourceIds: {format: 'uint32', values: Uint32Array.of(7, 7)}
    },
    outputs: {
      neighborIds: {format: 'uint32', length: 2},
      neighborRows: {format: 'uint32', length: 2},
      counts: {format: 'uint32', length: 1},
      joinOverflow: {format: 'uint32', length: 1},
      offsets: {format: 'uint32', length: 2},
      neighbors: {format: 'uint32', length: 2},
      weights: {format: 'float32', length: 2},
      overflow: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs}) => ({
      getCommandNodes(graph) {
        const neighborRows = outputs['neighborRows'] as GraphDataView<'uint32'>;
        const neighborCounts = outputs['counts'] as GraphDataView<'uint32'>;
        const join = new GPUNearestFeatureJoin({
          queries: {kind: 'points', positions: inputs['queries'] as GraphDataView<'float32x2'>},
          features: {
            kind: 'segments',
            starts: inputs['starts'] as GraphDataView<'float32x2'>,
            ends: inputs['ends'] as GraphDataView<'float32x2'>,
            sourceIds: inputs['sourceIds'] as GraphDataView<'uint32'>
          },
          k: 2,
          output: {
            neighborIds: outputs['neighborIds'] as GraphDataView<'uint32'>,
            neighborRows,
            neighborCounts,
            status: {overflow: outputs['joinOverflow'] as GraphDataView<'uint32'>}
          }
        });
        const weights = new GPUNearestFeatureWeights({
          neighborRows,
          neighborCounts,
          slotCapacity: 2,
          output: {
            offsets: outputs['offsets'] as GraphDataView<'uint32'>,
            neighbors: outputs['neighbors'] as GraphDataView<'uint32'>,
            weights: outputs['weights'] as GraphDataView<'float32'>,
            status: {overflow: outputs['overflow'] as GraphDataView<'uint32'>}
          }
        });
        return [...join.getCommandNodes(graph), ...weights.getCommandNodes(graph)];
      }
    })
  });
  try {
    const result = await fixture.run();
    expect(result['neighborIds']).toEqual([7, 7]);
    expect(result['neighborRows']).toEqual([0, 1]);
    expect(result['counts']).toEqual([2]);
    expect(result['offsets']).toEqual([0, 2]);
    expect(result['neighbors']).toEqual([0, 1]);
    expect(result['weights']).toEqual([1, 1]);
    expect(result['joinOverflow']).toEqual([0]);
    expect(result['overflow']).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

for (const fixtureName of [
  'both road parts hit',
  'polygon in road gap',
  'second multipoint hits'
] as const) {
  it(`R2 joins multipart input as feature rows: ${fixtureName}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: ReturnType<typeof createInputBuffer>[] = [];
    let serial = 0;
    const input = <Format extends 'float32x2' | 'uint32'>(
      values: Float32Array | Uint32Array,
      format: Format
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(
        graph,
        `input-${serial++}`,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      );
    };
    const geometry =
      fixtureName === 'second multipoint hits'
        ? {
            kind: 'points' as const,
            positions: input(Float32Array.of(-5, 0, 0, 0), 'float32x2'),
            featureOffsets: input(Uint32Array.of(0, 2), 'uint32')
          }
        : {
            kind: 'lines' as const,
            positions: input(Float32Array.of(-3, 0, -2, 0, 2, 0, 3, 0), 'float32x2'),
            lineOffsets: input(Uint32Array.of(0, 2, 4), 'uint32'),
            featureOffsets: input(Uint32Array.of(0, 2), 'uint32')
          };
    const extent = fixtureName === 'both road parts hit' ? 4 : 1;
    const polygon = {
      kind: 'polygons' as const,
      positions: input(
        Float32Array.of(-extent, -1, extent, -1, extent, 1, -extent, 1),
        'float32x2'
      ),
      featureOffsets: input(Uint32Array.of(0, 1), 'uint32'),
      polygonOffsets: input(Uint32Array.of(0, 1), 'uint32'),
      ringOffsets: input(Uint32Array.of(0, 4), 'uint32')
    };
    const outputBuffers: ReturnType<typeof createOutputBuffer>[] = [];
    const scalar = () => {
      const buffer = createOutputBuffer(device, 1);
      outputBuffers.push(buffer);
      return importGraphBuffer(graph, `output-${serial++}`, buffer, 'uint32', 1);
    };
    const leftRows = scalar(),
      rightRows = scalar(),
      count = scalar();
    let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
    try {
      graph.add(
        new GPUSpatialPredicateJoin({
          left: geometry,
          right: polygon,
          predicate: 'intersects',
          output: {
            pairs: {
              leftRows,
              rightRows,
              status: {count, overflow: scalar()}
            }
          }
        })
      );
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      const expectedCount = fixtureName === 'polygon in road gap' ? 0 : 1;
      expect(Array.from(await readUint32(outputBuffers[2], 1))).toEqual([expectedCount]);
      if (expectedCount) {
        expect(Array.from(await readUint32(outputBuffers[0], 1))).toEqual([0]);
        expect(Array.from(await readUint32(outputBuffers[1], 1))).toEqual([0]);
      }
    } finally {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
      for (const buffer of outputBuffers) buffer.destroy();
    }
  });
}

for (const byteOffset of [4, 256]) {
  for (const duplicateImport of [false, true]) {
    it(`R3 refuses source-ID/row physical aliases before dispatch at ${byteOffset}, duplicate=${duplicateImport}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const graph = new GPUCommandGraph(device);
      const backing = createInputBuffer(device, new Uint32Array(128));
      const sourceIds = importGraphBuffer(graph, 'feature-ids', backing, 'uint32', 1);
      const destination = duplicateImport
        ? importGraphBuffer(graph, 'destination', backing, 'uint32', 128)
        : sourceIds;
      const neighborRows = graph.createDataView(destination.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset
      });
      const scalar = (id: string) => createTransientView(graph, id, 'uint32', 1);
      const props = {
        queries: {
          kind: 'points' as const,
          positions: createTransientView(graph, 'queries', 'float32x2', 1)
        },
        features: {
          kind: 'points' as const,
          positions: createTransientView(graph, 'features', 'float32x2', 1),
          sourceIds
        },
        output: {
          neighborIds: scalar('ids'),
          neighborRows,
          neighborCounts: scalar('counts'),
          status: {overflow: scalar('overflow')}
        }
      };
      try {
        expect(() => new GPUNearestFeatureJoin(props).getCommandNodes(graph)).toThrow(
          /physical buffer.*read-only and writable/
        );
      } finally {
        backing.destroy();
      }
    });
  }
}
