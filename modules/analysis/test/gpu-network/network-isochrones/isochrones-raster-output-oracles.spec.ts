// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUNetworkIsochrones,
  getGPUNetworkIsochronesParameterValues
} from '../../../src/gpu-network/network-isochrones/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {RASTER_SQUARE, rotateOracleRing} from './isochrones-output-oracle-fixtures';
import {
  createIsochroneOracleHarness,
  createOracleRingOutput,
  expectOracleSquareRings,
  expectOracleValues,
  NONE,
  POISON,
  RING_PORTS,
  RASTER_ORACLE_SELECTIONS,
  type RingSelection
} from './isochrones-output-oracle-utils';

const selections = RASTER_ORACLE_SELECTIONS;

it('isochrone raster output oracle: second submission removes the square from the selected band', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const costs = harness.input('float32', [3]);
  const ringOutput = createOracleRingOutput(harness, 'all', 4);
  const contributor = new GPUNetworkIsochrones({
    topology: {
      offsets: harness.input('uint32', [0, 1]),
      neighbors: harness.input('uint32', [0]),
      weights: harness.input('float32', [0]),
      nodePositions: harness.input('float32x2', [1, 1])
    },
    costs,
    assignments: harness.input('uint32', [7]),
    costBreaks: harness.input('float32', [2, 4]),
    parameters: harness.input(
      'float32',
      Array.from(
        getGPUNetworkIsochronesParameterValues({
          breakCount: 2,
          extent: [0, 0, 2, 2],
          bufferRadius: 2,
          firstBand: 1,
          lastBand: 1
        })
      )
    ),
    raster: {
      width: 2,
      height: 2,
      output: {
        values: harness.output('values', 'float32', 4),
        bandClasses: harness.output('classes', 'uint32', 4),
        pixelFacilities: harness.output('pixelFacilities', 'uint32', 4),
        triangleFacilities: harness.output('triangleFacilities', 'uint32', 3),
        triangles: harness.output('triangles', 'float32x2', 9),
        triangleBands: harness.output('triangleBands', 'uint32', 3),
        edges: harness.output('edges', 'float32x4', 5),
        edgeBands: harness.output('edgeBands', 'uint32', 5),
        status: {
          count: harness.output('count', 'uint32', 1),
          overflow: harness.output('overflow', 'uint32', 1),
          requiredCount: harness.output('requiredCount', 'uint32', 1),
          vertexCount: harness.output('vertexCount', 'uint32', 1),
          edgeCount: harness.output('edgeCount', 'uint32', 1),
          edgeOverflow: harness.output('edgeOverflow', 'uint32', 1),
          edgeRequiredCount: harness.output('edgeRequiredCount', 'uint32', 1)
        }
      },
      rings: {
        edgeCapacity: 4,
        output: {
          ...ringOutput,
          status: {
            ...ringOutput.status,
            edgeOverflow: harness.output('ringEdgeOverflow', 'uint32', 1),
            edgeCount: harness.output('ringEdgeCount', 'uint32', 1)
          }
        }
      }
    }
  });
  const result = await harness.run(contributor, first => {
    expectOracleValues(first['values'], RASTER_SQUARE.values);
    expectOracleValues(first['count'], [2]);
    expectOracleValues(first['edgeCount'], [4]);
    expectOracleSquareRings(first, RASTER_SQUARE.corners, 1, 1, [0, 1, 2, 3], RASTER_SQUARE.edges);
    harness.write(costs, [9]);
  });
  // Cost 9 is above both breaks. Band 1 becomes empty on the same compiled graph.
  expectOracleValues(result['values'], [9, 9, 9, 9]);
  expectOracleValues(result['classes'], [2, 2, 2, 2]);
  for (const name of [
    'count',
    'requiredCount',
    'overflow',
    'vertexCount',
    'edgeCount',
    'edgeRequiredCount',
    'edgeOverflow',
    'ringCount',
    'ringRequiredCount',
    'ringOverflow',
    'ringEdgeCount',
    'ringEdgeOverflow',
    'openSegmentCount',
    'touchingSegmentCount'
  ])
    expectOracleValues(result[name], [0]);
  expectOracleValues(result['triangleFacilities'], [NONE, NONE, NONE]);
  expectOracleValues(result['ringOffsets'], [0, 0, 0]);
  expectOracleValues(result['ringAreas'], [0, 0]);
  expectOracleValues(result['ringIsHole'], [0, 0]);
  expectOracleValues(result['ringShells'], [NONE, NONE]);
  expectOracleValues(result['ringGroups'], [0, 0]);
  expectOracleValues(result['segmentRings'], [NONE, NONE, NONE, NONE]);
  expectOracleValues(result['segmentVertices'], [NONE, NONE, NONE, NONE]);
  expectOracleValues(result['segmentFlags'], [0, 0, 0, 0]);
  expectOracleValues(result['polygonRingOffsets'], [0, 0, 0]);
  expectOracleValues(result['polygonOffsets'], [0, 0, 0]);
  expectOracleValues(result['featureOffsets'], [0, 0, 0]);
  expectOracleValues(result['sourceIds'], [NONE, NONE]);
});

for (const selection of selections) {
  it(`isochrone raster output oracle: constant square, ${selection}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createIsochroneOracleHarness(device);
    const triangles = [
      'all',
      'triangles',
      'triangleFacilities',
      'triangleRequiredCount',
      'vertexCount',
      'no-rings'
    ].includes(selection);
    const edges = ['all', 'edges', 'edgeRequiredCount', 'no-rings'].includes(selection);
    const facilities = selection === 'all' || selection === 'triangleFacilities';
    const rings = selection !== 'no-rings' && selection !== 'triangleRequiredCount';
    const ringSelection: RingSelection =
      selection === 'all' || selection === 'minimal' || RING_PORTS.some(port => port === selection)
        ? (selection as RingSelection)
        : 'minimal';
    const ringOutput = rings ? createOracleRingOutput(harness, ringSelection, 4) : undefined;
    const result = await harness.run(
      new GPUNetworkIsochrones({
        topology: {
          offsets: harness.input('uint32', [0, 1]),
          neighbors: harness.input('uint32', [0]),
          weights: harness.input('float32', [0]),
          nodePositions: harness.input('float32x2', [1, 1])
        },
        costs: harness.input('float32', [3]),
        ...(facilities ? {assignments: harness.input('uint32', [7])} : {}),
        costBreaks: harness.input('float32', [2, 4]),
        parameters: harness.input(
          'float32',
          Array.from(
            getGPUNetworkIsochronesParameterValues({
              breakCount: 2,
              extent: [0, 0, 2, 2],
              bufferRadius: 2,
              firstBand: 1,
              lastBand: 1
            })
          )
        ),
        raster: {
          width: 2,
          height: 2,
          output: {
            values: harness.output('values', 'float32', 4),
            bandClasses: harness.output('bandClasses', 'uint32', 4),
            ...(facilities
              ? {
                  pixelFacilities: harness.output('pixelFacilities', 'uint32', 4),
                  triangleFacilities: harness.output('triangleFacilities', 'uint32', 3)
                }
              : {}),
            ...(triangles
              ? {
                  triangles: harness.output('triangles', 'float32x2', 9),
                  triangleBands: harness.output('triangleBands', 'uint32', 3)
                }
              : {}),
            ...(edges
              ? {
                  edges: harness.output('edges', 'float32x4', 5),
                  edgeBands: harness.output('edgeBands', 'uint32', 5)
                }
              : {}),
            status: {
              ...(triangles
                ? {
                    count: harness.output('count', 'uint32', 1),
                    overflow: harness.output('overflow', 'uint32', 1)
                  }
                : {}),
              ...(selection === 'all' || selection === 'triangleRequiredCount'
                ? {requiredCount: harness.output('requiredCount', 'uint32', 1)}
                : {}),
              ...(selection === 'all' || selection === 'vertexCount'
                ? {vertexCount: harness.output('vertexCount', 'uint32', 1)}
                : {}),
              ...(edges
                ? {
                    edgeCount: harness.output('edgeCount', 'uint32', 1),
                    edgeOverflow: harness.output('edgeOverflow', 'uint32', 1)
                  }
                : {}),
              ...(selection === 'all' || selection === 'edgeRequiredCount'
                ? {edgeRequiredCount: harness.output('edgeRequiredCount', 'uint32', 1)}
                : {})
            }
          },
          ...(ringOutput
            ? {
                rings: {
                  edgeCapacity: 4,
                  output: {
                    ...ringOutput,
                    status: {
                      ...ringOutput.status,
                      edgeOverflow: harness.output('ringEdgeOverflow', 'uint32', 1),
                      ...(selection === 'all' || selection === 'edgeCount'
                        ? {edgeCount: harness.output('ringEdgeCount', 'uint32', 1)}
                        : {})
                    }
                  }
                }
              }
            : {})
        }
      })
    );
    // All inputs and vertices are dyadic: exact float32 arithmetic, no tolerance needed.
    expectOracleValues(result['values'], RASTER_SQUARE.values);
    expectOracleValues(result['bandClasses'], RASTER_SQUARE.bandClasses);
    if (triangles) {
      expectOracleValues(result['count'], [2]);
      expectOracleValues(result['overflow'], [0]);
      expectOracleValues(result['triangles'], [
        ...RASTER_SQUARE.triangles,
        ...new Array(6).fill(POISON)
      ]);
      expectOracleValues(result['triangleBands'], [...RASTER_SQUARE.triangleBands, POISON]);
      if (result['requiredCount']) expectOracleValues(result['requiredCount'], [2]);
      if (result['vertexCount']) expectOracleValues(result['vertexCount'], [6]);
    }
    if (edges) {
      expectOracleValues(result['edges'], [...RASTER_SQUARE.edges, POISON, POISON, POISON, POISON]);
      expectOracleValues(result['edgeBands'], [1, 1, 1, 1, POISON]);
      expectOracleValues(result['edgeCount'], [4]);
      expectOracleValues(result['edgeOverflow'], [0]);
      if (result['edgeRequiredCount']) expectOracleValues(result['edgeRequiredCount'], [4]);
    }
    if (facilities) {
      expectOracleValues(result['pixelFacilities'], [7, 7, 7, 7]);
      expectOracleValues(result['triangleFacilities'], [7, 7, NONE]);
    }
    if (rings) {
      expectOracleSquareRings(
        result,
        RASTER_SQUARE.corners,
        1,
        1,
        [0, 1, 2, 3],
        RASTER_SQUARE.edges
      );
      expectOracleValues(result['ringEdgeOverflow'], [0]);
      if (result['ringEdgeCount']) expectOracleValues(result['ringEdgeCount'], [4]);
    }
    if (selection === 'triangleRequiredCount') {
      expectOracleValues(result['requiredCount'], [2]);
      expect(result).not.toHaveProperty('ringRequiredCount');
    }
    if (selection === 'ringRequiredCount') {
      expectOracleValues(result['ringRequiredCount'], [1]);
      expect(result).not.toHaveProperty('requiredCount');
    }
  });
}

it('isochrone raster output oracle: triangle and edge capacities preserve unclamped demand', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const result = await harness.run(
    new GPUNetworkIsochrones({
      topology: {
        offsets: harness.input('uint32', [0, 1]),
        neighbors: harness.input('uint32', [0]),
        weights: harness.input('float32', [0]),
        nodePositions: harness.input('float32x2', [1, 1])
      },
      costs: harness.input('float32', [3]),
      costBreaks: harness.input('float32', [2, 4]),
      parameters: harness.input(
        'float32',
        Array.from(
          getGPUNetworkIsochronesParameterValues({
            breakCount: 2,
            extent: [0, 0, 2, 2],
            bufferRadius: 2,
            firstBand: 1,
            lastBand: 1
          })
        )
      ),
      raster: {
        width: 2,
        height: 2,
        output: {
          triangles: harness.output('triangles', 'float32x2', 3),
          triangleBands: harness.output('triangleBands', 'uint32', 1),
          edges: harness.output('edges', 'float32x4', 2),
          edgeBands: harness.output('edgeBands', 'uint32', 2),
          status: {
            count: harness.output('count', 'uint32', 1),
            overflow: harness.output('overflow', 'uint32', 1),
            requiredCount: harness.output('requiredCount', 'uint32', 1),
            vertexCount: harness.output('vertexCount', 'uint32', 1),
            edgeCount: harness.output('edgeCount', 'uint32', 1),
            edgeOverflow: harness.output('edgeOverflow', 'uint32', 1),
            edgeRequiredCount: harness.output('edgeRequiredCount', 'uint32', 1)
          }
        }
      }
    })
  );
  // The same square needs two triangles and four edges; retain just the first one/two.
  expectOracleValues(result['triangles'], RASTER_SQUARE.triangles.slice(0, 6));
  expectOracleValues(result['triangleBands'], [1]);
  expectOracleValues(result['count'], [1]);
  expectOracleValues(result['requiredCount'], [2]);
  expectOracleValues(result['overflow'], [1]);
  expectOracleValues(result['vertexCount'], [3]);
  expectOracleValues(result['edges'], RASTER_SQUARE.edges.slice(0, 8));
  expectOracleValues(result['edgeBands'], [1, 1]);
  expectOracleValues(result['edgeCount'], [2]);
  expectOracleValues(result['edgeRequiredCount'], [4]);
  expectOracleValues(result['edgeOverflow'], [1]);
});

for (const edgeCapacity of [2, 4]) {
  it(`isochrone raster ring output oracle: incomplete edges, capacity ${edgeCapacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createIsochroneOracleHarness(device);
    const output = createOracleRingOutput(harness, 'all', edgeCapacity, 4);
    // Four vertices cannot hold the five-row closed square when edges are complete.
    const result = await harness.run(
      new GPUNetworkIsochrones({
        topology: {
          offsets: harness.input('uint32', [0, 1]),
          neighbors: harness.input('uint32', [0]),
          weights: harness.input('float32', [0]),
          nodePositions: harness.input('float32x2', [1, 1])
        },
        costs: harness.input('float32', [3]),
        costBreaks: harness.input('float32', [2, 4]),
        parameters: harness.input(
          'float32',
          Array.from(
            getGPUNetworkIsochronesParameterValues({
              breakCount: 2,
              extent: [0, 0, 2, 2],
              bufferRadius: 2,
              firstBand: 1,
              lastBand: 1
            })
          )
        ),
        raster: {
          width: 2,
          height: 2,
          output: {bandClasses: harness.output('classes', 'uint32', 4)},
          rings: {
            edgeCapacity,
            output: {
              ...output,
              status: {
                ...output.status,
                edgeCount: harness.output('ringEdgeCount', 'uint32', 1),
                edgeOverflow: harness.output('ringEdgeOverflow', 'uint32', 1)
              }
            }
          }
        }
      })
    );
    expectOracleValues(result['classes'], [1, 1, 1, 1]);
    expectOracleValues(result['ringEdgeCount'], [edgeCapacity]);
    expectOracleValues(result['ringEdgeOverflow'], [edgeCapacity === 2 ? 1 : 0]);
    expectOracleValues(result['ringCount'], [0]);
    expectOracleValues(result['ringRequiredCount'], [edgeCapacity === 2 ? 0 : 1]);
    expectOracleValues(result['ringOverflow'], [edgeCapacity === 2 ? 0 : 1]);
    expectOracleValues(result['openSegmentCount'], [edgeCapacity === 2 ? 2 : 0]);
    expectOracleValues(result['touchingSegmentCount'], [0]);
    expectOracleValues(result['segmentRings'], new Array(edgeCapacity).fill(NONE));
    expectOracleValues(result['segmentVertices'], new Array(edgeCapacity).fill(NONE));
    if (edgeCapacity === 2) expectOracleValues(result['segmentFlags'], [0, 2]);
  });
}

for (const facilities of [false, true]) {
  it(`isochrone raster output oracle: two facility patches, labels ${facilities}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createIsochroneOracleHarness(device);
    const result = await harness.run(
      new GPUNetworkIsochrones({
        topology: {
          offsets: harness.input('uint32', [0, 1, 2]),
          neighbors: harness.input('uint32', [0, 1]),
          weights: harness.input('float32', [0, 0]),
          nodePositions: harness.input('float32x2', [1, 1, 3, 1])
        },
        costs: harness.input('float32', [3, 3]),
        assignments: harness.input('uint32', [5, 2]),
        costBreaks: harness.input('float32', [2, 4]),
        parameters: harness.input(
          'float32',
          Array.from(
            getGPUNetworkIsochronesParameterValues({
              breakCount: 2,
              extent: [0, 0, 4, 2],
              bufferRadius: 0.75,
              firstBand: 1,
              lastBand: 1
            })
          )
        ),
        raster: {
          width: 4,
          height: 2,
          output: {
            values: harness.output('values', 'float32', 8),
            triangles: harness.output('triangles', 'float32x2', 21),
            triangleBands: harness.output('triangleBands', 'uint32', 7),
            ...(facilities
              ? {
                  pixelFacilities: harness.output('pixelFacilities', 'uint32', 8),
                  triangleFacilities: harness.output('triangleFacilities', 'uint32', 7)
                }
              : {}),
            status: {
              count: harness.output('count', 'uint32', 1),
              overflow: harness.output('overflow', 'uint32', 1)
            }
          }
        }
      })
    );
    // The two radius-.75 disks cover separate 2x2 blocks of centers (distance sqrt(.5)).
    // Three unit squares each emit two fan triangles. The right/left fan centroids choose
    // the right/left sample columns respectively, including the middle cell across facilities.
    expectOracleValues(result['values'], [3, 3, 3, 3, 3, 3, 3, 3]);
    expectOracleValues(result['count'], [6]);
    expectOracleValues(result['overflow'], [0]);
    expectOracleValues(result['triangleBands'], [1, 1, 1, 1, 1, 1, POISON]);
    expectOracleValues(result['triangles'], [
      0.5,
      0.5,
      1.5,
      0.5,
      1.5,
      1.5,
      0.5,
      0.5,
      1.5,
      1.5,
      0.5,
      1.5,
      1.5,
      0.5,
      2.5,
      0.5,
      2.5,
      1.5,
      1.5,
      0.5,
      2.5,
      1.5,
      1.5,
      1.5,
      2.5,
      0.5,
      3.5,
      0.5,
      3.5,
      1.5,
      2.5,
      0.5,
      3.5,
      1.5,
      2.5,
      1.5,
      POISON,
      POISON,
      POISON,
      POISON,
      POISON,
      POISON
    ]);
    if (facilities) {
      expectOracleValues(result['pixelFacilities'], [5, 5, 2, 2, 5, 5, 2, 2]);
      expectOracleValues(result['triangleFacilities'], [5, 5, 2, 5, 2, 2, NONE]);
    }
  });
}

it('R2 rectangular pixels label the triangle by world distance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const result = await harness.run(
    new GPUNetworkIsochrones({
      topology: {
        offsets: harness.input('uint32', [0, 1, 2]),
        neighbors: harness.input('uint32', [0, 1]),
        weights: harness.input('float32', [0, 0]),
        nodePositions: harness.input('float32x2', [2, 0.5, 7.75, 1.75])
      },
      costs: harness.input('float32', [3, 1]),
      assignments: harness.input('uint32', [2, 5]),
      costBreaks: harness.input('float32', [5]),
      parameters: harness.input(
        'float32',
        Array.from(
          getGPUNetworkIsochronesParameterValues({
            breakCount: 1,
            extent: [0, 0, 8, 2],
            firstBand: 1,
            lastBand: 1
          })
        )
      ),
      raster: {
        width: 2,
        height: 2,
        unreachedCost: 9,
        output: {
          values: harness.output('values', 'float32', 4),
          pixelFacilities: harness.output('pixelFacilities', 'uint32', 4),
          triangleFacilities: harness.output('triangleFacilities', 'uint32', 2),
          triangles: harness.output('triangles', 'float32x2', 6),
          triangleBands: harness.output('bands', 'uint32', 2),
          status: {
            count: harness.output('count', 'uint32', 1),
            overflow: harness.output('overflow', 'uint32', 1)
          }
        }
      }
    })
  );
  // Independent linear interpolation (oracle.py): vertices (10/3,.5),(6,.5),(6,1),
  // centroid (46/9,2/3). Sample distances in world units: 9.70679 vs 1.484568.
  // Moving the break one f32 ulp changes a vertex by <4e-7; 2e-6 covers interpolation.
  expectOracleValues(result['values'], [3, 9, 3, 1]);
  expectOracleValues(result['pixelFacilities'], [2, NONE, 2, 5]);
  expectOracleValues(result['count'], [1]);
  expectOracleValues(result['overflow'], [0]);
  expectOracleValues(result['triangles'].slice(0, 6), [10 / 3, 0.5, 6, 0.5, 6, 1], 2e-6);
  expectOracleValues(result['bands'], [1, POISON]);
  expectOracleValues(result['triangleFacilities'], [5, NONE]);
});

for (const metadata of [false, true]) {
  it(`R6 raster ring oracle distinguishes a shell and hole, metadata ${metadata}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createIsochroneOracleHarness(device);
    const result = await harness.run(
      new GPUNetworkIsochrones({
        topology: {
          offsets: harness.input('uint32', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
          neighbors: harness.input('uint32', [0, 1, 2, 3, 4, 5, 6, 7, 8]),
          weights: harness.input('float32', [0, 0, 0, 0, 0, 0, 0, 0, 0]),
          nodePositions: harness.input(
            'float32x2',
            [
              0.5, 0.5, 1.5, 0.5, 2.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 1.5, 0.5, 2.5, 1.5, 2.5, 2.5,
              2.5
            ]
          )
        },
        costs: harness.input('float32', [3, 3, 3, 3, 5, 3, 3, 3, 3]),
        costBreaks: harness.input('float32', [4]),
        parameters: harness.input(
          'float32',
          Array.from(
            getGPUNetworkIsochronesParameterValues({
              breakCount: 1,
              extent: [0, 0, 3, 3],
              firstBand: 0,
              lastBand: 0
            })
          )
        ),
        raster: {
          width: 3,
          height: 3,
          output: {
            values: harness.output('values', 'float32', 9),
            bandClasses: harness.output('classes', 'uint32', 9)
          },
          rings: {
            edgeCapacity: 12,
            output: {
              ringOffsets: harness.output('ringOffsets', 'uint32', 4),
              positions: harness.output('positions', 'float32x2', 14),
              ...(metadata
                ? {
                    ringAreas: harness.output('areas', 'float32', 3),
                    ringIsHole: harness.output('isHole', 'uint32', 3),
                    ringShells: harness.output('shells', 'uint32', 3),
                    polygons: {
                      kind: 'polygons' as const,
                      positions: harness.output('polygonPositions', 'float32x2', 14),
                      ringOffsets: harness.output('polygonRingOffsets', 'uint32', 4),
                      polygonOffsets: harness.output('polygonOffsets', 'uint32', 4),
                      featureOffsets: harness.output('featureOffsets', 'uint32', 4)
                    }
                  }
                : {}),
              status: {
                count: harness.output('count', 'uint32', 1),
                overflow: harness.output('overflow', 'uint32', 1),
                edgeCount: harness.output('edgeCount', 'uint32', 1),
                edgeOverflow: harness.output('edgeOverflow', 'uint32', 1)
              }
            }
          }
        }
      })
    );
    // Unit samples with center cost 5 and neighbors 3 cross level 4 at edge midpoints.
    // The domain is [.5,2.5]^2 minus a diamond with diagonals 1: areas 4 and -.5.
    // Isochrone upper-inclusivity moves crossings one f32 ulp; use 1e-6 in coordinates.
    const shell = [0.5, 0.5, 1.5, 0.5, 2.5, 0.5, 2.5, 1.5, 2.5, 2.5, 1.5, 2.5, 0.5, 2.5, 0.5, 1.5];
    const hole = [1, 1.5, 1.5, 2, 2, 1.5, 1.5, 1];
    expectOracleValues(result['values'], [3, 3, 3, 3, 5, 3, 3, 3, 3]);
    expectOracleValues(result['classes'], [0, 0, 0, 0, 1, 0, 0, 0, 0]);
    expectOracleValues(result['count'], [2]);
    expectOracleValues(result['overflow'], [0]);
    expectOracleValues(result['edgeCount'], [12]);
    expectOracleValues(result['edgeOverflow'], [0]);
    expectOracleValues(result['ringOffsets'], [0, 9, 14, 14]);
    expectOracleValues(rotateOracleRing(result['positions'].slice(0, 16)), shell, 1e-6);
    expectOracleValues(rotateOracleRing(result['positions'].slice(18, 26)), hole, 1e-6);
    expectOracleValues(result['positions'].slice(16, 18), result['positions'].slice(0, 2));
    expectOracleValues(result['positions'].slice(26, 28), result['positions'].slice(18, 20));
    if (metadata) {
      expectOracleValues(result['isHole'], [0, 1, 0]);
      expectOracleValues(result['shells'], [0, 0, NONE]);
      expectOracleValues(result['areas'], [4, -0.5, 0], 1e-5);
      expectOracleValues(result['polygonRingOffsets'], [0, 8, 12, 12]);
      expectOracleValues(result['polygonOffsets'], [0, 2, 2, 2]);
      expectOracleValues(result['featureOffsets'], [0, 1, 1, 1]);
      expectOracleValues(rotateOracleRing(result['polygonPositions'].slice(0, 16)), shell, 1e-6);
      expectOracleValues(rotateOracleRing(result['polygonPositions'].slice(16, 24)), hole, 1e-6);
    }
  });
}

for (const mode of ['min', 'max'] as const) {
  for (const [label, cost, weight, expected] of [
    ['negative cost', -1, 0, 9],
    ['infinite weight', 3, Infinity, 9],
    ['negative zero', -0, 0, 0]
  ] as const) {
    it(`R3 GPU ${mode} raster handles ${label}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createIsochroneOracleHarness(device);
      const result = await harness.run(
        new GPUNetworkIsochrones({
          topology: {
            offsets: harness.input('uint32', [0, 1]),
            neighbors: harness.input('uint32', [0]),
            weights: harness.input('float32', [weight]),
            nodePositions: harness.input('float32x2', [1, 1])
          },
          costs: harness.input('float32', [cost]),
          costBreaks: harness.input('float32', [5]),
          parameters: harness.input(
            'float32',
            Array.from(
              getGPUNetworkIsochronesParameterValues({
                breakCount: 1,
                extent: [0, 0, 2, 2],
                bufferRadius: 2
              })
            )
          ),
          raster: {
            width: 2,
            height: 2,
            mode,
            unreachedCost: 9,
            output: {
              values: harness.output('values', 'float32', 4),
              bandClasses: harness.output('classes', 'uint32', 4)
            }
          }
        })
      );
      // Invalid street inputs leave unreachedCost untouched; -0 covers every center at zero.
      expectOracleValues(result['values'], [expected, expected, expected, expected]);
      expectOracleValues(result['classes'], expected === 9 ? [1, 1, 1, 1] : [0, 0, 0, 0]);
    });
  }
}
