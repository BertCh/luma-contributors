// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUPairGather} from '../../src/gpu-spatial-analysis/pair-gather/gpu-pair-gather';
import {GPUSpatialPredicateJoin} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {GPUPointInPolygonJoin} from '../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {
  createFillNode,
  createWGSLKernelNode
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createGeometryFixture} from '../gpu-spatial-analysis/outline-geometry/geometry-fixture';
import {RecipeTestFixture} from '../gpu-spatial-analysis/recipes/recipe-harness';
import {addFleetDwellZoneEventsRecipe} from '../../src/gpu-spatial-analysis/recipes/fleet-dwell-recipe';
import {
  GPURandomPointsOnLine,
  GPURandomPointsInPolygon,
  getGPUDotDensityParameterValues
} from '../../src/gpu-spatial-analysis/dot-density/index';
import {runExample} from '../../../lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/pair-gather.example';

it('final2 G9 getter-backed weights requiredCount replaces the caller prefill', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'weights-required-getter');
  const required = fixture.output('required', 'uint32', 1);
  required.buffer.write(new Uint32Array([77]));
  const count = fixture.output('count', 'uint32', 1);
  const status = Object.defineProperty({count: count.view}, 'requiredCount', {
    get: () => required.view
  });
  const join = new GPUSpatialPredicateJoin({
    left: {
      kind: 'points',
      positions: fixture.input('left', new Float32Array([0, 0]), 'float32x2', 1)
    },
    right: {
      kind: 'points',
      positions: fixture.input('right', new Float32Array([0, 0]), 'float32x2', 1)
    },
    predicate: 'intersects',
    output: {
      weights: {
        offsets: fixture.output('offsets', 'uint32', 2).view,
        neighbors: fixture.output('neighbors', 'uint32', 2).view,
        weights: fixture.output('weights', 'float32', 2).view,
        status
      },
      status: {overflow: fixture.output('overflow', 'uint32', 1).view}
    }
  });
  fixture.graph.add(join);
  try {
    fixture.run();
    expect(join.outputWeights?.status.requiredCount).toBe(required.view);
    expect(await fixture.readUint32(required, 1)).toEqual([1]);
    expect(await fixture.readUint32(count, 1)).toEqual([1]);
  } finally {
    fixture.destroy();
  }
});
import {requireTestDevice} from './require-test-device';
import {submitGraph} from './gpu-contributor-test-utils';
import {getZoneEventLossCaptures} from '../../../lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/zone-event-loss-captures';

it('R1 anti join through left gather rewrites prefilled candidate flags on graph reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      left: {format: 'float32x2', values: new Float32Array([0, 0])},
      right: {format: 'float32x2', values: new Float32Array([0, 0, 0, 0])}
    },
    outputs: {
      leftRows: {format: 'uint32', length: 2},
      rightRows: {format: 'uint32', length: 2},
      rowCount: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      candidate: {format: 'uint32', length: 1},
      antiOverflow: {format: 'uint32', length: 1},
      antiCandidate: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs}) => ({
      getCommandNodes(graph) {
        const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
        const pairs = {
          leftRows: createTransientView(graph, 'pair-left', 'uint32', 2),
          rightRows: createTransientView(graph, 'pair-right', 'uint32', 2),
          status: {
            count: scalar('pair-count'),
            overflow: scalar('pair-overflow'),
            candidateOverflow: scalar('pair-candidate')
          }
        };
        const unmatched = {
          ids: createTransientView(graph, 'anti-ids', 'uint32', 2),
          status: {
            count: scalar('anti-count'),
            overflow: outputs['antiOverflow'] as GraphDataView<'uint32'>,
            candidateOverflow: outputs['antiCandidate'] as GraphDataView<'uint32'>
          }
        };
        const common = {
          left: {kind: 'points' as const, positions: inputs['left'] as GraphDataView<'float32x2'>},
          right: {
            kind: 'points' as const,
            positions: inputs['right'] as GraphDataView<'float32x2'>
          },
          predicate: 'intersects' as const,
          candidateCapacity: 1
        };
        return [
          createFillNode(graph, {
            id: 'prefill-anti',
            operation: 'Review',
            view: unmatched.status.candidateOverflow,
            type: 'u32',
            value: '37u'
          }),
          ...new GPUSpatialPredicateJoin({...common, id: 'inner', output: {pairs}}).getCommandNodes(
            graph
          ),
          ...new GPUSpatialPredicateJoin({
            ...common,
            id: 'anti',
            how: 'anti',
            output: {unmatched}
          }).getCommandNodes(graph),
          ...new GPUPairGather({
            how: 'left',
            pairs,
            unmatchedLeft: unmatched,
            output: {
              leftRows: outputs['leftRows'] as never,
              rightRows: outputs['rightRows'] as never,
              rowCount: outputs['rowCount'] as never,
              status: {
                overflow: outputs['overflow'] as never,
                candidateOverflow: outputs['candidate'] as never
              }
            }
          }).getCommandNodes(graph)
        ];
      }
    })
  });
  try {
    const first = await fixture.run();
    expect([first['rowCount'][0], first['leftRows'][0], first['rightRows'][0]]).toEqual([1, 0, 0]);
    expect([
      first['overflow'][0],
      first['candidate'][0],
      first['antiOverflow'][0],
      first['antiCandidate'][0]
    ]).toEqual([0, 1, 0, 1]);
    fixture.writeInput('right', new Float32Array([0, 0, 10, 10]));
    const second = await fixture.run();
    expect([second['rowCount'][0], second['leftRows'][0], second['rightRows'][0]]).toEqual([
      1, 0, 0
    ]);
    expect([
      second['overflow'][0],
      second['candidate'][0],
      second['antiOverflow'][0],
      second['antiCandidate'][0]
    ]).toEqual([0, 0, 0, 0]);
    expect(fixture.getCompileCount()).toBe(0);
  } finally {
    fixture.destroy();
  }
});

it('R1 point-in-polygon compact matches split candidate loss and clear it on reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      points: {format: 'float32x2', values: new Float32Array([1.5, 1])},
      positions: {
        format: 'float32x2',
        values: new Float32Array([0, 0, 2, 0, 2, 2, 0, 2, 1, 0, 3, 0, 3, 2, 1, 2])
      },
      featureOffsets: {format: 'uint32', values: new Uint32Array([0, 1, 2])},
      polygonOffsets: {format: 'uint32', values: new Uint32Array([0, 1, 2])},
      ringOffsets: {format: 'uint32', values: new Uint32Array([0, 4, 8])}
    },
    outputs: {
      ids: {format: 'uint32', length: 2},
      count: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      candidate: {format: 'uint32', length: 1},
      aggregate: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs}) => ({
      getCommandNodes(graph) {
        const candidate = outputs['candidate'] as GraphDataView<'uint32'>;
        return [
          createFillNode(graph, {
            id: 'prefill',
            operation: 'Review',
            view: candidate,
            type: 'u32',
            value: '37u'
          }),
          ...new GPUPointInPolygonJoin({
            points: {kind: 'points', positions: inputs['points'] as never},
            polygons: {
              kind: 'polygons',
              positions: inputs['positions'] as never,
              featureOffsets: inputs['featureOffsets'] as never,
              polygonOffsets: inputs['polygonOffsets'] as never,
              ringOffsets: inputs['ringOffsets'] as never
            },
            candidateCapacity: 1,
            spatialSort: false,
            querySort: false,
            output: {
              pointFeatureIds: createTransientView(graph, 'point-feature-ids', 'uint32', 1),
              status: {overflow: outputs['aggregate'] as never},
              matches: {
                ids: outputs['ids'] as never,
                status: {
                  count: outputs['count'] as never,
                  overflow: outputs['overflow'] as never,
                  candidateOverflow: candidate
                }
              }
            }
          }).getCommandNodes(graph)
        ];
      }
    })
  });
  try {
    const first = await fixture.run();
    expect([
      first['count'][0],
      first['ids'][0],
      first['overflow'][0],
      first['candidate'][0]
    ]).toEqual([1, 0, 0, 1]);
    fixture.writeInput('points', new Float32Array([0.5, 1]));
    const second = await fixture.run();
    expect([
      second['count'][0],
      second['ids'][0],
      second['overflow'][0],
      second['candidate'][0]
    ]).toEqual([1, 0, 0, 0]);
    expect(fixture.getCompileCount()).toBe(0);
  } finally {
    fixture.destroy();
  }
});

it('R4 gather writes count and candidate in adjacent words in separate GPU passes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'adjacent-gather');
  try {
    const storage = fixture.output('status', 'uint32', 65);
    const count = fixture.graph.createDataView(storage.view.buffer, {format: 'uint32', length: 1});
    const candidate = fixture.graph.createDataView(storage.view.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    fixture.graph.add(
      new GPUPairGather({
        pairs: {
          leftRows: fixture.input('left', new Uint32Array([0]), 'uint32', 1),
          rightRows: fixture.input('right', new Uint32Array([0]), 'uint32', 1),
          status: {
            count: fixture.input('pair-count', new Uint32Array([1]), 'uint32', 1),
            overflow: fixture.input('pair-overflow', new Uint32Array([0]), 'uint32', 1),
            candidateOverflow: fixture.input('pair-candidate', new Uint32Array([1]), 'uint32', 1)
          }
        },
        output: {
          leftRows: fixture.output('left-out', 'uint32', 2).view,
          rightRows: fixture.output('right-out', 'uint32', 2).view,
          rowCount: count,
          status: {
            overflow: fixture.output('overflow', 'uint32', 1).view,
            candidateOverflow: candidate
          }
        }
      })
    );
    fixture.run();
    expect(await fixture.readUint32(storage, 2)).toEqual([1, 1]);
  } finally {
    fixture.destroy();
  }
});

for (const kind of ['line', 'polygon'] as const)
  it(`R5 ${kind} sampling distinguishes slot clamping from final capacity`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = new RecipeTestFixture(device, `sampling-clamp-${kind}`);
    try {
      const requests = new Uint32Array(65536);
      requests[0] = 65536;
      const requestBuffer = fixture.output('counts', 'uint32', 65536);
      requestBuffer.buffer.write(requests);
      const counts = requestBuffer.view;
      const parameters = fixture.parameters(
        'parameters',
        'uint32',
        getGPUDotDensityParameterValues({seed: 1})
      );
      const count = fixture.output('count', 'uint32', 1),
        overflow = fixture.output('overflow', 'uint32', 1),
        candidate = fixture.output('candidate', 'uint32', 1);
      const compact = {
        ids: fixture.output('ids', 'uint32', 65536).view,
        status: {count: count.view, overflow: overflow.view, candidateOverflow: candidate.view}
      };
      const positions = fixture.output('sample-positions', 'float32x2', 65536).view;
      if (kind === 'line') {
        const offsets = new Uint32Array(65537).fill(2);
        offsets[0] = 0;
        fixture.graph.add(
          new GPURandomPointsOnLine({
            positions: fixture.input('vertices', new Float32Array([0, 0, 1, 0]), 'float32x2', 2),
            lineOffsets: fixture.input('offsets', offsets, 'uint32', offsets.length),
            counts,
            parameters,
            output: {positions, points: compact}
          })
        );
      } else {
        const offsets = new Uint32Array(65537).fill(1);
        offsets[0] = 0;
        fixture.graph.add(
          new GPURandomPointsInPolygon({
            positions: fixture.input(
              'vertices',
              new Float32Array([0, 0, 2, 0, 0, 2]),
              'float32x2',
              3
            ),
            featureOffsets: fixture.input('features', offsets, 'uint32', offsets.length),
            polygonOffsets: fixture.input('polygons', new Uint32Array([0, 1]), 'uint32', 2),
            ringOffsets: fixture.input('rings', new Uint32Array([0, 3]), 'uint32', 2),
            counts,
            parameters,
            output: {positions, dots: compact}
          })
        );
      }
      const compiled = fixture.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await fixture.readUint32(count, 1)).toEqual([65535]);
        expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
        expect(await fixture.readUint32(candidate, 1)).toEqual([1]);
        const smaller = new Uint32Array(65536);
        smaller[0] = 1;
        requestBuffer.buffer.write(smaller);
        submitGraph(device, compiled, undefined);
        expect(await fixture.readUint32(count, 1)).toEqual([1]);
        expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
        expect(await fixture.readUint32(candidate, 1)).toEqual([0]);
      } finally {
        compiled.destroy();
      }
    } finally {
      fixture.destroy();
    }
  });

for (const candidateCapacity of [32, 1])
  it(`R6 fleet-dwell distinguishes event and dwell completeness (candidate capacity ${candidateCapacity})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = new RecipeTestFixture(device, 'recipe-candidate');
    try {
      const count = fixture.output('count', 'uint32', 1),
        overflow = fixture.output('overflow', 'uint32', 1),
        candidate = fixture.output('candidate', 'uint32', 1);
      const recipe = addFleetDwellZoneEventsRecipe(fixture.graph, {
        trajectories: {
          kind: 'trajectories',
          positions: fixture.input('track', new Float32Array([-1, 0.5, 2, 0.5]), 'float32x2', 2),
          timestamps: fixture.input('times', new Float32Array([0, 3]), 'float32', 2),
          trackOffsets: fixture.input('tracks', new Uint32Array([0, 2]), 'uint32', 2)
        },
        zones: {
          kind: 'polygons',
          positions: fixture.input(
            'polygon',
            new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
            'float32x2',
            4
          ),
          featureOffsets: fixture.input('features', new Uint32Array([0, 1]), 'uint32', 2),
          polygonOffsets: fixture.input('polygons', new Uint32Array([0, 1]), 'uint32', 2),
          ringOffsets: fixture.input('rings', new Uint32Array([0, 4]), 'uint32', 2)
        },
        candidateCapacity,
        trackEventCapacity: 1,
        eventCapacity: 2,
        outputs: {
          events: {
            status: {count: count.view, overflow: overflow.view, candidateOverflow: candidate.view}
          }
        }
      });
      expect(recipe.outputs.events.status.candidateOverflow).toBe(candidate.view);
      expect(
        recipe.status.stages.find(stage => stage.stage === 'zone-events')?.status.candidateOverflow
      ).toBe(candidate.view);
      // The lens captures the combined event word and scratch-only dwell evidence separately.
      const scratch = fixture.output('scratch-capture', 'uint32', 1);
      fixture.graph.add(
        createWGSLKernelNode(fixture.graph, {
          id: 'capture-scratch-loss',
          operation: 'Review',
          invocationCount: 1,
          bindings: [
            {name: 'source', view: recipe.status.candidateOverflow, type: 'u32', access: 'read'},
            {name: 'destination', view: scratch.view, type: 'u32', access: 'read_write'}
          ],
          body: 'destination[destinationOffset] = source[sourceOffset];'
        })
      );
      fixture.run();
      if (candidateCapacity === 32) expect(await fixture.readUint32(count, 1)).toEqual([1]);
      expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
      expect(await fixture.readUint32(candidate, 1)).toEqual([1]);
      const captures = getZoneEventLossCaptures(
        {
          overflow: (await fixture.readUint32(overflow, 1))[0],
          candidateOverflow: (await fixture.readUint32(candidate, 1))[0]
        },
        (await fixture.readUint32(scratch, 1))[0]
      );
      expect(!captures.eventOverflow.view && !captures.eventCandidateOverflow.view).toBe(false);
      expect(!captures.scratchCandidateOverflow.view).toBe(candidateCapacity === 32);
    } finally {
      fixture.destroy();
    }
  });

for (const supplyCandidateFlag of [true, false])
  it(`R9 pair-gather example preserves candidate loss (caller flag ${supplyCandidateFlag})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = new RecipeTestFixture(device, 'lens-candidate');
    try {
      const polygon = (name: string, count: number) => ({
        kind: 'polygons' as const,
        positions: fixture.input(
          `${name}-positions`,
          new Float32Array(Array.from({length: count}, () => [0, 0, 1, 0, 1, 1, 0, 1]).flat()),
          'float32x2',
          4 * count
        ),
        featureOffsets: fixture.input(
          `${name}-features`,
          Uint32Array.from({length: count + 1}, (_, index) => index),
          'uint32',
          count + 1
        ),
        polygonOffsets: fixture.input(
          `${name}-polygons`,
          Uint32Array.from({length: count + 1}, (_, index) => index),
          'uint32',
          count + 1
        ),
        ringOffsets: fixture.input(
          `${name}-rings`,
          Uint32Array.from({length: count + 1}, (_, index) => index * 4),
          'uint32',
          count + 1
        )
      });
      const pairCandidate = fixture.output('pair-candidate', 'uint32', 1);
      const result = await runExample(fixture.graph, {
        parks: polygon('parks', 1),
        tracts: polygon('tracts', 2),
        pairs: {
          leftRows: fixture.output('left', 'uint32', 2).view,
          rightRows: fixture.output('right', 'uint32', 2).view,
          status: {
            count: fixture.output('pair-count', 'uint32', 1).view,
            overflow: fixture.output('pair-overflow', 'uint32', 1).view,
            ...(supplyCandidateFlag ? {candidateOverflow: pairCandidate.view} : {})
          }
        },
        spatialContext: {coordinateSpace: 'planar', metric: 'native'},
        parkAreas: fixture.input('areas', new Float32Array([1]), 'float32', 1),
        candidateCapacity: 1
      });
      const capture = fixture.output('capture', 'uint32', 3);
      fixture.graph.add(
        createWGSLKernelNode(fixture.graph, {
          id: 'capture-candidate',
          operation: 'Review',
          invocationCount: 1,
          bindings: [
            {name: 'source', view: result.candidateOverflow, type: 'u32', access: 'read'},
            {name: 'count', view: result.rowCount, type: 'u32', access: 'read'},
            {name: 'overflow', view: result.overflow, type: 'u32', access: 'read'},
            {name: 'destination', view: capture.view, type: 'u32', access: 'read_write'}
          ],
          body: 'destination[destinationOffset] = count[countOffset]; destination[destinationOffset + 1u] = overflow[overflowOffset]; destination[destinationOffset + 2u] = source[sourceOffset];'
        })
      );
      fixture.run();
      if (supplyCandidateFlag) expect(await fixture.readUint32(pairCandidate, 1)).toEqual([1]);
      expect(await fixture.readUint32(capture, 3)).toEqual([1, 0, 1]);
    } finally {
      fixture.destroy();
    }
  });
