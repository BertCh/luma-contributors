// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUPolygonOverlay} from '../../../src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function publish(uncertainty: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const scalar = () => createTransientView(graph, `status-${serial++}`, 'uint32', 1);
  const view = (length: number) =>
    createTransientView(graph, `offset-${serial++}`, 'uint32', length);
  const status = {
    count: scalar(),
    overflow: scalar(),
    candidateOverflow: scalar(),
    uncertainCount: scalar()
  };
  const boundaryStatus = {
    count: scalar(),
    requiredCount: scalar(),
    overflow: scalar(),
    candidateOverflow: scalar()
  };
  const nodes = new GPUPolygonOverlay({
    id: 'overlay-probe',
    operation: 'dissolve',
    vertexTolerance: 1,
    left: {
      kind: 'polygons',
      positions: createTransientView(graph, 'input', 'float32x2', 4),
      featureOffsets: view(2),
      polygonOffsets: view(2),
      ringOffsets: view(2)
    },
    capacity: {intersections: 4, nodedSegments: 16},
    output: {
      geometry: {
        kind: 'polygons',
        positions: createTransientView(graph, 'output', 'float32x2', 32),
        featureOffsets: view(9),
        polygonOffsets: view(9),
        ringOffsets: view(9)
      },
      status,
      boundary: {
        endpoints: createTransientView(graph, 'boundary', 'float32x4', 16),
        operandIds: view(16),
        sourceFeatureIds: view(16),
        status: boundaryStatus
      }
    }
  }).getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(status.uncertainCount, Uint32Array.of(uncertainty));
  executeCommandNodesOnCPU(
    nodes.filter(node =>
      [
        'overlay-probe-noding-status-candidates',
        'overlay-probe-noding-status',
        'overlay-probe-boundary-status',
        'overlay-probe-publish'
      ].includes(node.id)
    ),
    memory
  );
  const result = {
    overflow: memory.readUint32(status.overflow)[0],
    candidateOverflow: memory.readUint32(status.candidateOverflow)[0]
  };
  expect(memory.readUint32(boundaryStatus.overflow)[0]).toBe(0);
  expect(memory.readUint32(boundaryStatus.candidateOverflow)[0]).toBe(uncertainty);
  expect(result).toEqual({overflow: 0, candidateOverflow: uncertainty});
  return result;
}
const source = () =>
  readFileSync(
    new URL(
      '../../../src/gpu-spatial-analysis/polygon-overlay/polygon-overlay-types.ts',
      import.meta.url
    ),
    'utf8'
  );

it('the documented uncertainty flag is the flag written by the production publishers', () => {
  const flags = publish(1);
  const documentedFlag = source().match(
    /Optional one-row count of segment pairs[\s\S]*?They set `(\w+)`/u
  )![1];
  expect(flags[documentedFlag as keyof typeof flags]).toBe(1);
  expect(publish(0)).toEqual({overflow: 0, candidateOverflow: 0});
});

it('documented provenance eligibility refuses candidate-only uncertainty', () => {
  const flags = publish(1);
  const documentation = source();
  const eligibility = documentation.slice(
    documentation.indexOf('/** Optional per-output-feature'),
    documentation.indexOf('featureSources?:')
  );
  const checkedFlags = [
    ...eligibility.matchAll(/output.status\.(overflow|candidateOverflow)/gu)
  ].map(match => match[1] as keyof typeof flags);
  expect(checkedFlags.every(flag => flags[flag] === 0)).toBe(false);
});
