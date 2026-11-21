// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  GPUPolygonOverlay,
  type GPUPolygonOverlayProps
} from '../../../src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';

export function createPolygonOverlayOutputOracle(
  device: Device,
  uncertainty = false,
  doubleTraceFillRule?: 'own' | 'inherited'
) {
  const fixture = createOutputOracleFixture(device);
  const {graph, column, output} = fixture;
  const left = {
    kind: 'polygons' as const,
    positions: column(
      'left',
      'float32x2',
      new Float32Array(
        doubleTraceFillRule
          ? [0, 0, 4, 0, 4, 4, 0, 4, 0, 0, 4, 0, 4, 4, 0, 4]
          : [0, 0, 4, 0, 4, 4, 0, 4]
      )
    ),
    featureOffsets: column('leftFeatures', 'uint32', new Uint32Array([0, 0, 1])),
    polygonOffsets: column('leftPolygons', 'uint32', new Uint32Array([0, 1])),
    ringOffsets: column('leftRings', 'uint32', new Uint32Array([0, doubleTraceFillRule ? 8 : 4]))
  };
  const right = {
    kind: 'polygons' as const,
    positions: column('right', 'float32x2', new Float32Array([2, 2, 6, 2, 6, 6, 2, 6])),
    featureOffsets: column('rightFeatures', 'uint32', new Uint32Array([0, 1])),
    polygonOffsets: column('rightPolygons', 'uint32', new Uint32Array([0, 1])),
    ringOffsets: column('rightRings', 'uint32', new Uint32Array([0, 4]))
  };
  const props = {
    left,
    right: doubleTraceFillRule ? undefined : right,
    operation: doubleTraceFillRule ? ('dissolve' as const) : ('intersection' as const),
    vertexTolerance: 0.25,
    capacity: {
      intersections: doubleTraceFillRule ? 64 : 16,
      nodedSegments: doubleTraceFillRule ? 64 : 32
    },
    output: {
      geometry: {
        kind: 'polygons' as const,
        // Four boundary vertices plus one closing row for the nested ring assembly.
        positions: output('positions', 'float32x2', 5),
        featureOffsets: output('features', 'uint32', 2),
        polygonOffsets: output('polygons', 'uint32', 2),
        ringOffsets: output('rings', 'uint32', 2)
      },
      status: {
        count: output('count', 'uint32', 1),
        overflow: output('overflow', 'uint32', 1),
        candidateOverflow: output('candidateOverflow', 'uint32', 1),
        ...(uncertainty ? {uncertainCount: output('uncertainCount', 'uint32', 1)} : {})
      },
      boundary: {
        endpoints: output('endpoints', 'float32x4', 4),
        operandIds: output('operandIds', 'uint32', 4),
        sourceFeatureIds: output('sourceFeatureIds', 'uint32', 4),
        status: {
          count: output('boundaryCount', 'uint32', 1),
          requiredCount: output('boundaryRequiredCount', 'uint32', 1),
          overflow: output('boundaryOverflow', 'uint32', 1),
          candidateOverflow: output('boundaryCandidateOverflow', 'uint32', 1)
        }
      }
    }
  };
  const supplied: GPUPolygonOverlayProps =
    doubleTraceFillRule === 'inherited'
      ? Object.assign(
          Object.create({
            get fillRule() {
              return 'nonzero';
            }
          }),
          props
        )
      : {...props, ...(doubleTraceFillRule ? {fillRule: 'nonzero' as const} : {})};
  const nodes = new GPUPolygonOverlay(supplied).getCommandNodes(graph);
  for (const node of nodes) graph.add(node);
  return {...fixture, props, nodes};
}
