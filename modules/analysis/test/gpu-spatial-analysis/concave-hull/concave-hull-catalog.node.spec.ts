// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {transpileModule, ScriptTarget, ModuleKind} from 'typescript';
import {entries} from '../../../../../website/src/catalog/entries/geometry-construction-overlay';
import {getGPUSpatialAnalysisOperator} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-operators';
import {GPUGroupConcaveHull} from '../../../src/gpu-spatial-analysis/concave-hull/gpu-concave-hull';
import {getDelaunayMaximumTriangleCount} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('R7 publishes groupCount convergence rows and executes the packed catalog example with two groups', () => {
  const entry = entries.find(candidate => candidate.symbol === 'GPUGroupConcaveHull')!;
  expect(entry.params.some(parameter => parameter.name === 'layout')).toBe(true);
  expect(entry.outputs.find(output => output.name === 'status.converged')?.description).toContain(
    'groupCount'
  );
  const groups = entry.outputs.find(output => output.name === 'groups');
  expect(groups).toBeDefined();
  for (const field of ['triangleOffsets', 'triangleCounts', 'edgeOffsets', 'edgeCounts'])
    expect(groups!.type).toMatch(new RegExp(`\\b${field}\\b`));
  expect(entry.outputs.flatMap(output => output.name.split(/\s*,\s*/))).toEqual(
    expect.arrayContaining(['status.count', 'status.requiredCount', 'status.overflow'])
  );
  expect(entry.outputs.length).toBeLessThanOrEqual(8);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 6);
    const labels = createTransientView(graph, 'labels', 'uint32', 6);
    const add = vi.spyOn(graph, 'add');
    // Execute the published allocation and construction verbatim, stopping before device compilation.
    const source = entry.example.split('const compiled')[0].replace(/^import[^\n]*\n/gm, '');
    const javascript = transpileModule(source, {
      compilerOptions: {target: ScriptTarget.ES2022, module: ModuleKind.ESNext}
    }).outputText;
    new Function(
      'graph',
      'positions',
      'labels',
      'createTransientView',
      'GPUGroupConcaveHull',
      'getDelaunayMaximumTriangleCount',
      javascript
    )(
      graph,
      positions,
      labels,
      createTransientView,
      GPUGroupConcaveHull,
      getDelaunayMaximumTriangleCount
    );
    const contributor = add.mock.calls[0][0] as GPUGroupConcaveHull;
    expect(contributor.props.groupCount).toBe(2);
    expect(contributor.props.output.status.converged.length).toBe(2);
    expect(contributor.props.layout).toBe('packed');
    if (contributor.props.layout === 'packed') {
      for (const view of Object.values(contributor.props.output.groups))
        expect(view.length).toBe(2);
      expect(contributor.props.output.status.count.length).toBe(1);
      expect(contributor.props.output.triangles.length).toBe(7);
    }
    add.mockRestore();
  } finally {
    device.destroy();
  }
});

it('R7 operator metadata lists packed boundary-edge status and fixed/packed reservation limits', () => {
  const operator = getGPUSpatialAnalysisOperator('GPUGroupConcaveHull')!;
  expect(operator.status).toEqual(
    expect.arrayContaining(['converged', 'count', 'requiredCount', 'overflow'])
  );
  expect(operator.limitation).toContain('groupCount');
  expect(operator.limitation).toContain('packed');
  expect(operator.limits?.find(limit => limit.name === 'triangle reservation')?.maximum).toBe(
    715827882
  );
});
