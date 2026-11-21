// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import type {GPUComputationCapture} from '@luma-contributors/analysis/computation-record';
import {GPUDelaunayTessellation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {expect, test, vi} from 'vitest';
import {requireTestDevice} from '../../../analysis/src/testing/require-test-device';
import {getLensScenario} from '../../src/computation-lenses';
import {
  LensCommandGraph,
  addLensViewSnapshot,
  submitLensGraph
} from '../../src/computation-lenses/lens-view-snapshot';
import {
  runS3Lens,
  type S3Uploads
} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s3-lens-run';
import {runExample as runCover} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/cell-cover.example';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/cell-set-outline.lens';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/voronoi-diagram.lens';

vi.mock(
  import('../../src/computation-lenses/scenarios/gpu-spatial-analysis/s3-lens-run'),
  async importOriginal => {
    const actual = await importOriginal();
    return {...actual, runS3Lens: vi.fn(actual.runS3Lens)};
  }
);

vi.mock(
  import('../../src/computation-lenses/scenarios/gpu-spatial-analysis/cell-cover.example'),
  async importOriginal => {
    const actual = await importOriginal();
    return {...actual, runExample: vi.fn(actual.runExample)};
  }
);

function expectCaptureWord(
  captures: readonly GPUComputationCapture[],
  name: string,
  frame: number,
  expected: number
): void {
  const capture = captures.find(entry => entry.name === name && entry.iteration === frame);
  expect(capture?.values, `${name} at frame ${frame}`).toEqual(Uint32Array.of(expected));
}

test('W6 actual outline scenario suppresses candidate, final and non-capacity cover loss and recovers', {
  timeout: 300000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new LensCommandGraph<undefined>(device);
  const owned: {destroy(): void}[] = [];
  const buffers = new Map<string, Buffer>();
  const uploads: S3Uploads = {
    own: resource => {
      owned.push(resource);
      return resource;
    },
    view(id, data, format, length) {
      const buffer = device.createBuffer({
        id,
        data,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      owned.push(buffer);
      buffers.set(id, buffer);
      const handle = graph.importBuffer(
        {id, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      const components = Number(format.match(/x([234])$/)?.[1] ?? 1);
      return graph.createDataView(handle, {format, length: length ?? data.length / components});
    }
  };
  const cover = vi.mocked(runCover).mockImplementation(async () => ({
    // H3 8828308281fffff is one resolution-8 hexagon: six exposed edges, one ring.
    // A second valid cell beyond count=1 must never enter the table.
    cells: uploads.view(
      'controlled-cells',
      Uint32Array.of(0x281fffff, 0x08828308, 0x01fffff, 0x08808000),
      'uint32x2'
    ),
    featureIds: uploads.view('controlled-features', Uint32Array.of(0, 0), 'uint32'),
    status: {
      count: uploads.view('controlled-count', Uint32Array.of(1), 'uint32'),
      requiredCount: uploads.view('controlled-required', Uint32Array.of(1), 'uint32'),
      overflow: uploads.view('controlled-overflow', Uint32Array.of(0), 'uint32'),
      candidateOverflow: uploads.view('controlled-candidate', Uint32Array.of(0), 'uint32'),
      complete: uploads.view('controlled-complete', Uint32Array.of(1), 'uint32')
    }
  }));
  const boundary = new Error('actual scenario tested before record assembly');
  const runner = vi.mocked(runS3Lens).mockImplementation(async (_device, props) => {
    const specifications = await props.build(graph, uploads);
    const frames = [
      [0, 0, 1],
      [0, 1, 1],
      [1, 0, 1],
      [0, 0, 0],
      [0, 0, 1]
    ] as const;
    const snapshot = addLensViewSnapshot(graph, device, {
      id: 'outline-w6',
      frameCapacity: frames.length,
      views: {
        segmentCount: specifications['segmentCount'].view,
        ringCount: specifications['ringCount'].view
      }
    });
    owned.push(snapshot.arena);
    const compiled = graph.compile();
    try {
      for (const [overflow, candidateOverflow, complete] of frames) {
        buffers.get('controlled-overflow')!.write(Uint32Array.of(overflow));
        buffers.get('controlled-candidate')!.write(Uint32Array.of(candidateOverflow));
        buffers.get('controlled-complete')!.write(Uint32Array.of(complete));
        submitLensGraph(device, compiled, {parameters: undefined});
      }
      const captures = await snapshot.arena.read({
        graph: graph.id,
        resources: snapshot.resources,
        runId: 'outline-w6'
      });
      expect(captures).toHaveLength(frames.length * 2);
      for (const [frame, [overflow, candidateOverflow, complete]] of frames.entries()) {
        const lost = overflow || candidateOverflow || !complete;
        expectCaptureWord(captures, 'segmentCount', frame, lost ? 0 : 6);
        expectCaptureWord(captures, 'ringCount', frame, lost ? 0 : 1);
      }
    } finally {
      compiled.destroy();
    }
    throw boundary;
  });
  try {
    const scenario = getLensScenario('gpu-spatial-analysis.cell-set-outline')!;
    await expect((await scenario.load()).run(device)).rejects.toBe(boundary);
  } finally {
    runner.mockReset();
    cover.mockReset();
    for (const resource of owned) resource.destroy();
  }
});

test('W6 actual Voronoi capture distinguishes solver suppression from a successful empty diagram', {
  timeout: 300000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new LensCommandGraph<undefined>(device);
  const owned: {destroy(): void}[] = [];
  const buffers = new Map<string, Buffer>();
  const uploads: S3Uploads = {
    own: resource => {
      owned.push(resource);
      return resource;
    },
    view(id, data, format, length) {
      // A single right triangle has three hull bisector rays that intersect this clip box.
      const values =
        id === 'places'
          ? Float32Array.of(0, 0, 1, 0, 0, 1)
          : id === 'clipBounds'
            ? Float32Array.of(-1, -1, 2, 2)
            : data;
      const buffer = device.createBuffer({
        id,
        data: values,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      owned.push(buffer);
      buffers.set(id, buffer);
      const handle = graph.importBuffer(
        {id, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      const components = Number(format.match(/x([234])$/)?.[1] ?? 1);
      return graph.createDataView(handle, {format, length: length ?? values.length / components});
    }
  };
  const boundary = new Error('actual Voronoi capture checked');
  const runner = vi.mocked(runS3Lens).mockImplementation(async (_device, props) => {
    const control = uploads.view('controlled-triangle-status', Uint32Array.of(0, 1), 'uint32');
    // Replace only the solver. The example still connects its real Delaunay ports to Voronoi.
    const addNode = graph.add.bind(graph);
    const replaceSolver = vi.spyOn(graph, 'add').mockImplementation(node => {
      if (!(node instanceof GPUDelaunayTessellation)) {
        addNode(node);
        return;
      }
      const {triangles, status} = node.props.output;
      if (!status.converged) throw new Error('The Voronoi example needs a solver convergence port');
      addNode(
        createWGSLKernelNode(graph, {
          id: 'controlled-delaunay',
          operation: 'test-delaunay',
          invocationCount: 1,
          bindings: [
            {name: 'control', view: control, type: 'u32', access: 'read'},
            {name: 'triangles', view: triangles, type: 'u32', access: 'read_write'},
            {name: 'count', view: status.count, type: 'u32', access: 'read_write'},
            {name: 'required', view: status.requiredCount, type: 'u32', access: 'read_write'},
            {name: 'overflow', view: status.overflow, type: 'u32', access: 'read_write'},
            {name: 'converged', view: status.converged, type: 'u32', access: 'read_write'}
          ],
          body: `
  triangles[trianglesOffset] = 0u;
  triangles[trianglesOffset + 1u] = 1u;
  triangles[trianglesOffset + 2u] = 2u;
  count[countOffset] = control[controlOffset];
  required[requiredOffset] = control[controlOffset];
  overflow[overflowOffset] = 0u;
  converged[convergedOffset] = control[controlOffset + 1u];`
        })
      );
    });
    const specifications = await Promise.resolve()
      .then(() => props.build(graph, uploads))
      .finally(() => replaceSolver.mockRestore());
    const frames = [
      [0, 1, 0],
      [1, 1, 3],
      [1, 0, 0],
      [1, 1, 3]
    ] as const;
    const names = ['count', 'requiredCount', 'overflow', 'converged'] as const;
    const snapshot = addLensViewSnapshot(graph, device, {
      id: 'voronoi-w6',
      frameCapacity: frames.length,
      views: Object.fromEntries(names.map(name => [name, specifications[name].view]))
    });
    owned.push(snapshot.arena);
    const compiled = graph.compile();
    try {
      for (const [triangleCount, converged] of frames) {
        buffers.get('controlled-triangle-status')!.write(Uint32Array.of(triangleCount, converged));
        submitLensGraph(device, compiled, {parameters: undefined});
      }
      const captures = await snapshot.arena.read({
        graph: graph.id,
        resources: snapshot.resources,
        runId: 'voronoi-w6'
      });
      expect(captures).toHaveLength(frames.length * names.length);
      for (const [frame, [, converged, segmentCount]] of frames.entries()) {
        const expected = {count: segmentCount, requiredCount: segmentCount, overflow: 0, converged};
        for (const name of names) expectCaptureWord(captures, name, frame, expected[name]);
      }
    } finally {
      compiled.destroy();
    }
    throw boundary;
  });
  try {
    await expect(
      (await getLensScenario('gpu-spatial-analysis.voronoi-diagram')!.load()).run(device)
    ).rejects.toBe(boundary);
  } finally {
    runner.mockReset();
    for (const resource of owned) resource.destroy();
  }
});
