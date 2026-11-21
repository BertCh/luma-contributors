// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {
  GPUNetworkIsochrones,
  getGPUNetworkIsochronesParameterValues
} from '../../../src/gpu-network/network-isochrones/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {CELL_HOLE, CELL_SQUARE, rotateOracleRing} from './isochrones-output-oracle-fixtures';
import {
  createIsochroneOracleHarness,
  createOracleRingOutput,
  expectOracleSquareRings,
  expectOracleValues,
  expectOracleSegmentStarts,
  NONE,
  RING_PORTS,
  type RingSelection
} from './isochrones-output-oracle-utils';

const selections = [
  'all',
  'minimal',
  ...RING_PORTS,
  'no-rings',
  'no-table',
  'tableRequiredCount',
  'candidateOverflow',
  'outlineRequiredCount'
] as const;

for (const selection of selections) {
  it(`isochrone cell output oracle: one Quadbin tile, ${selection}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createIsochroneOracleHarness(device);
    const table = selection !== 'no-table';
    const rings = selection !== 'no-rings';
    const ringSelection: RingSelection =
      selection === 'all' || selection === 'minimal' || RING_PORTS.some(port => port === selection)
        ? (selection as RingSelection)
        : 'minimal';
    const result = await harness.run(
      new GPUNetworkIsochrones({
        topology: {
          offsets: harness.input('uint32', [0, 1, 1, 1, 1]),
          neighbors: harness.input('uint32', [0]),
          weights: harness.input('float32', [0]),
          nodePositions: harness.input('float32x2', CELL_SQUARE.positions)
        },
        costs: harness.input('float32', [3, 2, 2, 4]),
        // Lowest cost ties choose facility 7, rather than 9. No search is run.
        assignments: harness.input('uint32', [8, 9, 7, 6]),
        costBreaks: harness.input('float32', [10]),
        parameters: harness.input(
          'float32',
          Array.from(
            getGPUNetworkIsochronesParameterValues({
              breakCount: 1,
              extent: [-180, -80, 180, 80],
              cellCostLimit: 5
            })
          )
        ),
        cellOutline: {
          family: 'quadbin',
          resolution: 2,
          byFacility: true,
          ...(table
            ? {
                table: {
                  cells: harness.output('tableCells', 'uint32x2', 2),
                  counts: harness.output('tableCounts', 'uint32', 2),
                  status: {
                    count: harness.output('tableCount', 'uint32', 1),
                    overflow: harness.output('tableOverflow', 'uint32', 1),
                    ...(selection === 'all' || selection === 'tableRequiredCount'
                      ? {requiredCount: harness.output('tableRequiredCount', 'uint32', 1)}
                      : {}),
                    ...(selection === 'all' || selection === 'candidateOverflow'
                      ? {candidateOverflow: harness.output('candidateOverflow', 'uint32', 1)}
                      : {})
                  }
                }
              }
            : {}),
          output: {
            rows: harness.output('outlineRows', 'uint32', 4),
            cells: harness.output('outlineCells', 'uint32x2', 4),
            edgeIndices: harness.output('edgeIndices', 'uint32', 4),
            endpoints: harness.output('endpoints', 'float32x4', 4),
            groups: harness.output('groups', 'uint32', 4),
            status: {
              count: harness.output('outlineCount', 'uint32', 1),
              overflow: harness.output('outlineOverflow', 'uint32', 1),
              ...(selection === 'all' || selection === 'outlineRequiredCount'
                ? {requiredCount: harness.output('outlineRequiredCount', 'uint32', 1)}
                : {})
            }
          },
          ...(rings
            ? {
                rings: {
                  normalizeWinding: true,
                  output: createOracleRingOutput(harness, ringSelection, 4)
                }
              }
            : {})
        }
      })
    );
    // Four points inside the independently identified tile, four exposed cell edges, one shell.
    // 3e-5 degrees allows the float32 Mercator transcendental; area error is at most 0.0054 deg².
    expectOracleValues(result['outlineRows'], [0, 0, 0, 0]);
    expectOracleValues(result['edgeIndices'], [0, 1, 2, 3]);
    expectOracleValues(result['outlineCells'], [
      ...CELL_SQUARE.key,
      ...CELL_SQUARE.key,
      ...CELL_SQUARE.key,
      ...CELL_SQUARE.key
    ]);
    expectOracleValues(result['endpoints'], CELL_SQUARE.endpoints, 3e-5);
    expectOracleValues(result['groups'], [7, 7, 7, 7]);
    expectOracleValues(result['outlineCount'], [4]);
    expectOracleValues(result['outlineOverflow'], [0]);
    if (result['outlineRequiredCount']) expectOracleValues(result['outlineRequiredCount'], [4]);
    if (table) {
      expectOracleValues(result['tableCells'], [...CELL_SQUARE.key, NONE, NONE]);
      expectOracleValues(result['tableCounts'], [4, 0]);
      expectOracleValues(result['tableCount'], [1]);
      expectOracleValues(result['tableOverflow'], [0]);
      if (result['tableRequiredCount']) expectOracleValues(result['tableRequiredCount'], [1]);
      if (result['candidateOverflow']) expectOracleValues(result['candidateOverflow'], [0]);
    }
    if (rings)
      expectOracleSquareRings(
        result,
        CELL_SQUARE.corners,
        CELL_SQUARE.area,
        7,
        [0, 3, 2, 1],
        CELL_SQUARE.endpoints,
        3e-5
      );
  });
}

for (const optionalMetadata of [false, true]) {
  it(`isochrone cell output oracle: detached shell and annulus, metadata ${optionalMetadata}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createIsochroneOracleHarness(device);
    const result = await harness.run(
      new GPUNetworkIsochrones({
        topology: {
          offsets: harness.input('uint32', [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
          neighbors: harness.input('uint32', [0]),
          weights: harness.input('float32', [0]),
          nodePositions: harness.input('float32x2', CELL_HOLE.positions)
        },
        costs: harness.input('float32', [1, 1, 1, 1, 1, 1, 1, 1, 1]),
        assignments: harness.input('uint32', [3, 7, 7, 7, 7, 7, 7, 7, 7]),
        costBreaks: harness.input('float32', [10]),
        parameters: harness.input(
          'float32',
          Array.from(
            getGPUNetworkIsochronesParameterValues({
              breakCount: 1,
              extent: [-180, -85, 180, 85],
              cellCostLimit: 5
            })
          )
        ),
        cellOutline: {
          family: 'quadbin',
          resolution: 4,
          byFacility: true,
          table: {
            cells: harness.output('tableCells', 'uint32x2', 9),
            counts: harness.output('tableCounts', 'uint32', 9),
            status: {
              count: harness.output('tableCount', 'uint32', 1),
              overflow: harness.output('tableOverflow', 'uint32', 1)
            }
          },
          output: {
            rows: harness.output('rows', 'uint32', 20),
            cells: harness.output('cells', 'uint32x2', 20),
            edgeIndices: harness.output('edgeIndices', 'uint32', 20),
            endpoints: harness.output('endpoints', 'float32x4', 20),
            groups: harness.output('groups', 'uint32', 20),
            status: {
              count: harness.output('count', 'uint32', 1),
              overflow: harness.output('overflow', 'uint32', 1),
              requiredCount: harness.output('requiredCount', 'uint32', 1)
            }
          },
          rings: {
            normalizeWinding: true,
            output: {
              ringOffsets: harness.output('ringOffsets', 'uint32', 5),
              positions: harness.output('positions', 'float32x2', 23),
              ...(optionalMetadata
                ? {
                    ringAreas: harness.output('areas', 'float32', 4),
                    ringIsHole: harness.output('isHole', 'uint32', 4),
                    ringShells: harness.output('shells', 'uint32', 4),
                    ringGroups: harness.output('ringGroups', 'uint32', 4),
                    segmentRings: harness.output('segmentRings', 'uint32', 20),
                    segmentVertices: harness.output('segmentVertices', 'uint32', 20),
                    segmentFlags: harness.output('segmentFlags', 'uint32', 20),
                    polygons: {
                      kind: 'polygons' as const,
                      positions: harness.output('polygonPositions', 'float32x2', 23),
                      ringOffsets: harness.output('polygonRingOffsets', 'uint32', 5),
                      polygonOffsets: harness.output('polygonOffsets', 'uint32', 5),
                      featureOffsets: harness.output('featureOffsets', 'uint32', 5),
                      sourceIds: harness.output('sourceIds', 'uint32', 4)
                    }
                  }
                : {}),
              status: {
                count: harness.output('ringCount', 'uint32', 1),
                overflow: harness.output('ringOverflow', 'uint32', 1),
                requiredCount: harness.output('ringRequiredCount', 'uint32', 1),
                openSegmentCount: harness.output('open', 'uint32', 1),
                touchingSegmentCount: harness.output('touching', 'uint32', 1)
              }
            }
          }
        }
      })
    );
    // Literal Morton rows distinguish row 1+ from both caller IDs and a zero-filled port.
    expectOracleValues(result['tableCells'], CELL_HOLE.keys);
    expectOracleValues(result['tableCounts'], [1, 1, 1, 1, 1, 1, 1, 1, 1]);
    expectOracleValues(result['tableCount'], [9]);
    expectOracleValues(result['tableOverflow'], [0]);
    expectOracleValues(result['rows'], CELL_HOLE.rows);
    expectOracleValues(result['edgeIndices'], CELL_HOLE.edgeIndices);
    expectOracleValues(
      result['cells'],
      CELL_HOLE.rows.flatMap(row => CELL_HOLE.keys.slice(2 * row, 2 * row + 2))
    );
    expectOracleValues(
      result['groups'],
      [3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    );
    expectOracleValues(result['count'], [20]);
    expectOracleValues(result['requiredCount'], [20]);
    expectOracleValues(result['overflow'], [0]);
    expectOracleValues(result['ringCount'], [3]);
    expectOracleValues(result['ringRequiredCount'], [3]);
    expectOracleValues(result['ringOverflow'], [0]);
    expectOracleValues(result['open'], [0]);
    expectOracleValues(result['touching'], [0]);
    expectOracleValues(result['ringOffsets'], CELL_HOLE.ringOffsets);
    // Transcendental Mercator decoding gets 3e-5 degrees; no relative/self comparison.
    for (let ring = 0; ring < 3; ring++) {
      const first = CELL_HOLE.ringOffsets[ring];
      const last = CELL_HOLE.ringOffsets[ring + 1];
      expectOracleValues(
        rotateOracleRing(result['positions'].slice(2 * first, 2 * (last - 1))),
        CELL_HOLE.rings[ring],
        3e-5
      );
      expectOracleValues(
        result['positions'].slice(2 * (last - 1), 2 * last),
        result['positions'].slice(2 * first, 2 * first + 2)
      );
    }
    if (optionalMetadata) {
      // 0.01 deg² bounds float32 shoelace products/sums at these longitudes.
      expectOracleValues(result['areas'], CELL_HOLE.areas, 0.01);
      expectOracleValues(result['isHole'], [0, 0, 1, 0]);
      // The hole belongs to shell 1, not the earlier disconnected shell 0.
      expectOracleValues(result['shells'], [0, 1, 1, NONE]);
      expectOracleValues(result['ringGroups'], [3, 7, 7, 0]);
      expectOracleValues(result['segmentFlags'], new Array(20).fill(0));
      expectOracleValues(
        result['segmentRings'],
        [0, 0, 0, 0, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1]
      );
      // Trace the three rectangles from their first emitted edge, reversing the clockwise
      // cell outline. These literal slots pin each segment's start, including the hole.
      expectOracleValues(
        result['segmentVertices'],
        [0, 3, 2, 1, 5, 6, 16, 18, 15, 14, 21, 7, 9, 8, 13, 19, 20, 10, 12, 11]
      );
      expectOracleSegmentStarts(
        result['positions'],
        result['segmentVertices'],
        CELL_HOLE.segmentStarts,
        3e-5
      );
      expectOracleValues(result['polygonRingOffsets'], [0, 4, 16, 20, 20]);
      expectOracleValues(result['polygonOffsets'], [0, 1, 3, 3, 3]);
      expectOracleValues(result['featureOffsets'], [0, 1, 2, 2, 2]);
      expectOracleValues(result['sourceIds'], [3, 7, NONE, NONE]);
      expectOracleValues(
        result['polygonPositions'].slice(40),
        [12345, 12345, 12345, 12345, 12345, 12345]
      );
      for (const [ring, first, last] of [
        [0, 0, 4],
        [1, 4, 16],
        [2, 16, 20]
      ]) {
        expectOracleValues(
          rotateOracleRing(result['polygonPositions'].slice(2 * first, 2 * last)),
          CELL_HOLE.rings[ring],
          3e-5
        );
      }
    }
  });
}

for (const diagnostic of ['none', 'segmentFlags', 'touchingSegmentCount'] as const) {
  it(`isochrone cell output oracle: two tiles touch at one corner, ${diagnostic}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createIsochroneOracleHarness(device);
    const result = await harness.run(
      new GPUNetworkIsochrones({
        topology: {
          offsets: harness.input('uint32', [0, 1, 1]),
          neighbors: harness.input('uint32', [0]),
          weights: harness.input('float32', [0]),
          nodePositions: harness.input('float32x2', [-45, 30, 45, -30])
        },
        costs: harness.input('float32', [1, 1]),
        costBreaks: harness.input('float32', [5]),
        parameters: harness.input(
          'float32',
          Array.from(
            getGPUNetworkIsochronesParameterValues({breakCount: 1, extent: [-180, -85, 180, 85]})
          )
        ),
        cellOutline: {
          family: 'quadbin',
          resolution: 2,
          output: {
            rows: harness.output('rows', 'uint32', 8),
            cells: harness.output('cells', 'uint32x2', 8),
            edgeIndices: harness.output('edgeIndices', 'uint32', 8),
            endpoints: harness.output('endpoints', 'float32x4', 8),
            status: {
              count: harness.output('count', 'uint32', 1),
              overflow: harness.output('overflow', 'uint32', 1)
            }
          },
          rings: {
            normalizeWinding: true,
            output: {
              ringOffsets: harness.output('ringOffsets', 'uint32', 3),
              positions: harness.output('positions', 'float32x2', 10),
              ...(diagnostic === 'segmentFlags'
                ? {segmentFlags: harness.output('flags', 'uint32', 8)}
                : {}),
              status: {
                count: harness.output('ringCount', 'uint32', 1),
                overflow: harness.output('ringOverflow', 'uint32', 1),
                ...(diagnostic === 'touchingSegmentCount'
                  ? {touchingSegmentCount: harness.output('touching', 'uint32', 1)}
                  : {})
              }
            }
          }
        }
      })
    );
    // Tiles (1,1) and (2,2) at z2 share only (0,0). Two incoming segments end there.
    expectOracleValues(result['count'], [8]);
    expectOracleValues(result['overflow'], [0]);
    expectOracleValues(result['rows'], [0, 0, 0, 0, 1, 1, 1, 1]);
    expectOracleValues(result['edgeIndices'], [0, 1, 2, 3, 0, 1, 2, 3]);
    expectOracleValues(result['ringCount'], [2]);
    expectOracleValues(result['ringOverflow'], [0]);
    expectOracleValues(result['ringOffsets'], [0, 5, 10]);
    expectOracleValues(
      rotateOracleRing(result['positions'].slice(0, 8)),
      CELL_SQUARE.corners,
      3e-5
    );
    expectOracleValues(
      rotateOracleRing(result['positions'].slice(10, 18)),
      [0, -66.51326044311186, 90, -66.51326044311186, 90, 0, 0, 0],
      3e-5
    );
    if (diagnostic === 'segmentFlags')
      expectOracleValues(result['flags'], [0, 1, 0, 0, 0, 0, 0, 1]);
    if (diagnostic === 'touchingSegmentCount') expectOracleValues(result['touching'], [2]);
  });
}

for (const cost of [-1, -0]) {
  it(`R3 Quadbin cell mask handles cost ${Object.is(cost, -0) ? '-0' : cost}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createIsochroneOracleHarness(device);
    const result = await harness.run(
      new GPUNetworkIsochrones({
        topology: {
          offsets: harness.input('uint32', [0, 1]),
          neighbors: harness.input('uint32', [0]),
          weights: harness.input('float32', [0]),
          nodePositions: harness.input('float32x2', [-45, 30])
        },
        costs: harness.input('float32', [cost]),
        assignments: harness.input('uint32', [7]),
        costBreaks: harness.input('float32', [5]),
        parameters: harness.input(
          'float32',
          Array.from(
            getGPUNetworkIsochronesParameterValues({breakCount: 1, extent: [-180, -80, 180, 80]})
          )
        ),
        cellOutline: {
          family: 'quadbin',
          resolution: 2,
          byFacility: true,
          cellFacilities: harness.output('facilities', 'uint32', 1),
          table: {
            cells: harness.output('tableCells', 'uint32x2', 1),
            counts: harness.output('tableCounts', 'uint32', 1),
            status: {
              count: harness.output('tableCount', 'uint32', 1),
              overflow: harness.output('tableOverflow', 'uint32', 1)
            }
          },
          output: {
            rows: harness.output('rows', 'uint32', 4),
            cells: harness.output('cells', 'uint32x2', 4),
            edgeIndices: harness.output('edges', 'uint32', 4),
            endpoints: harness.output('endpoints', 'float32x4', 4),
            groups: harness.output('groups', 'uint32', 4),
            status: {
              count: harness.output('count', 'uint32', 1),
              overflow: harness.output('overflow', 'uint32', 1)
            }
          }
        }
      })
    );
    const reached = cost >= 0;
    expectOracleValues(result['tableCount'], [reached ? 1 : 0]);
    expectOracleValues(result['tableOverflow'], [0]);
    expectOracleValues(result['tableCounts'], [reached ? 1 : 0]);
    expectOracleValues(result['count'], [reached ? 4 : 0]);
    expectOracleValues(result['overflow'], [0]);
    expectOracleValues(result['facilities'], [reached ? 7 : NONE]);
    if (reached) expectOracleValues(result['groups'], [7, 7, 7, 7]);
    else expectOracleValues(result['tableCells'], [NONE, NONE]);
  });
}

for (const bottleneck of ['table', 'outline', 'ring'] as const) {
  it(`R4 ${bottleneck} requiredCount measures demand before truncation`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createIsochroneOracleHarness(device);
    const twoTiles = bottleneck === 'table';
    const outlineCapacity = bottleneck === 'outline' ? 2 : 4;
    const result = await harness.run(
      new GPUNetworkIsochrones({
        topology: {
          offsets: harness.input('uint32', twoTiles ? [0, 1, 1] : [0, 1]),
          neighbors: harness.input('uint32', [0]),
          weights: harness.input('float32', [0]),
          nodePositions: harness.input('float32x2', twoTiles ? [-45, 30, 45, -30] : [-45, 30])
        },
        costs: harness.input('float32', twoTiles ? [1, 1] : [1]),
        costBreaks: harness.input('float32', [5]),
        parameters: harness.input(
          'float32',
          Array.from(
            getGPUNetworkIsochronesParameterValues({breakCount: 1, extent: [-180, -80, 180, 80]})
          )
        ),
        cellOutline: {
          family: 'quadbin',
          resolution: 2,
          table: {
            cells: harness.output('tableCells', 'uint32x2', 1),
            counts: harness.output('tableCounts', 'uint32', 1),
            status: {
              count: harness.output('tableCount', 'uint32', 1),
              requiredCount: harness.output('tableRequiredCount', 'uint32', 1),
              overflow: harness.output('tableOverflow', 'uint32', 1)
            }
          },
          output: {
            rows: harness.output('rows', 'uint32', outlineCapacity),
            cells: harness.output('cells', 'uint32x2', outlineCapacity),
            edgeIndices: harness.output('edges', 'uint32', outlineCapacity),
            endpoints: harness.output('endpoints', 'float32x4', outlineCapacity),
            status: {
              count: harness.output('count', 'uint32', 1),
              requiredCount: harness.output('requiredCount', 'uint32', 1),
              overflow: harness.output('overflow', 'uint32', 1),
              candidateOverflow: harness.output('candidateOverflow', 'uint32', 1)
            }
          },
          ...(bottleneck === 'ring'
            ? {
                rings: {
                  output: {
                    ringOffsets: harness.output('ringOffsets', 'uint32', 3),
                    positions: harness.output('positions', 'float32x2', 4),
                    status: {
                      count: harness.output('ringCount', 'uint32', 1),
                      requiredCount: harness.output('ringRequiredCount', 'uint32', 1),
                      overflow: harness.output('ringOverflow', 'uint32', 1)
                    }
                  }
                }
              }
            : {})
        }
      })
    );
    // At z2, (-45,30) and (45,-30) occupy distinct tiles (columns 1 and 2).
    // Two keys cannot fit in one table row; the retained tile has four edges. Requesting
    // candidateOverflow keeps that retained prefix readable while explicitly reporting loss;
    // without it the incomplete outline count is suppressed to zero. A closed four-edge
    // ring needs five position rows and is dropped whole when only four fit.
    expectOracleValues(result['tableCount'], [1]);
    expectOracleValues(result['tableRequiredCount'], [twoTiles ? 2 : 1]);
    expectOracleValues(result['tableOverflow'], [twoTiles ? 1 : 0]);
    expectOracleValues(result['count'], [outlineCapacity]);
    expectOracleValues(result['requiredCount'], [4]);
    expectOracleValues(result['overflow'], [bottleneck === 'outline' ? 1 : 0]);
    expectOracleValues(result['candidateOverflow'], [twoTiles ? 1 : 0]);
    if (bottleneck === 'ring') {
      expectOracleValues(result['ringCount'], [0]);
      expectOracleValues(result['ringRequiredCount'], [1]);
      expectOracleValues(result['ringOverflow'], [1]);
    }
  });
}
