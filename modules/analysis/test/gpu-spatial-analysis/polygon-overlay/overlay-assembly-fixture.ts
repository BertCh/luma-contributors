// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Device} from '@luma.gl/core';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUPolygonOverlay} from '../../../src/gpu-spatial-analysis/polygon-overlay';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';

/** Real overlay nodes, with a controlled assembler handoff to exercise loss that valid rings hide. */
export function createOverlayAssemblyFixture(device: Device) {
  const fixture = createOutputOracleFixture(device);
  const {graph, column, output} = fixture;
  const words = (name: string, length: number) => output(name, 'uint32', length);
  const boundaryStatus = {
    count: column('boundary-count', 'uint32', Uint32Array.of(3)),
    requiredCount: words('boundary-required', 1),
    overflow: column('boundary-overflow', 'uint32', Uint32Array.of(0)),
    candidateOverflow: column('boundary-candidate', 'uint32', Uint32Array.of(0))
  };
  const status = {
    count: words('count', 1),
    overflow: words('overflow', 1),
    candidateOverflow: words('candidate', 1)
  };
  const geometry = {
    kind: 'polygons' as const,
    positions: output('result', 'float32x2', 32),
    ringOffsets: words('ring-offsets', 9),
    polygonOffsets: words('polygon-offsets', 9),
    featureOffsets: words('feature-offsets', 9)
  };
  const producer = new GPUPolygonOverlay({
    id: 'overlay-review',
    operation: 'dissolve',
    vertexTolerance: 1,
    left: {
      kind: 'polygons',
      positions: column('input', 'float32x2', Float32Array.of(0, 0, 1, 0, 0, 1)),
      ringOffsets: column('input-rings', 'uint32', Uint32Array.of(0, 3)),
      polygonOffsets: column('input-polygons', 'uint32', Uint32Array.of(0, 1)),
      featureOffsets: column('input-features', 'uint32', Uint32Array.of(0, 1))
    },
    capacity: {intersections: 8, nodedSegments: 16},
    output: {
      geometry,
      status,
      boundary: {
        endpoints: output('boundary', 'float32x4', 5),
        operandIds: words('operands', 5),
        sourceFeatureIds: words('sources', 5),
        status: boundaryStatus
      }
    }
  });
  const nodes = producer.getCommandNodes(graph);
  const views = new Map<string, GraphDataView>();
  for (const node of nodes)
    for (const resource of node.resources ?? []) {
      if ('buffer' in resource && 'buffer' in resource.buffer) {
        const view = resource.buffer;
        views.set(view.buffer.id, view);
      }
    }
  const internal = (name: string) => {
    const view = views.get(`overlay-review-${name}`);
    if (!view) throw new Error(`missing overlay view ${name}`);
    return view;
  };
  return {...fixture, nodes, internal, status, boundaryStatus, geometry};
}
