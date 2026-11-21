// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {createNullWebGPUDevice} from '../../../analysis/test/utils/gpu-contributor-test-utils';
import {
  LensCommandGraph,
  addLensViewSnapshot
} from '../../src/computation-lenses/lens-view-snapshot';
import {getLensScenario} from '../../src/computation-lenses/registry-implementation';
import * as runner from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s3-lens-run';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/polygonize.lens';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/make-valid.lens';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/geometry-validity.lens';
import * as simpleRunner from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s2-lens-run';
import {entries} from '../../../../website/src/catalog/entries/geometry-processing';
import ts from 'typescript';
import {resolve} from 'node:path';

function createUploads(graph: LensCommandGraph<undefined>): simpleRunner.S2Uploads {
  let serial = 0;
  return {
    view<Format extends GPUVectorFormat>(
      _name: string,
      data: Float32Array | Uint32Array | Int32Array,
      format: Format,
      length?: number
    ) {
      const components = /x([234])$/.exec(format);
      const buffer = graph.createTransientBuffer({
        id: `review-upload-${serial++}`,
        byteLength: data.byteLength,
        usage: Buffer.STORAGE | Buffer.COPY_SRC
      });
      return graph.createDataView(buffer, {
        format,
        length: length ?? data.length / (components ? Number(components[1]) : 1)
      });
    },
    own: resource => resource
  };
}

it.each([
  'polygonize',
  'make-valid'
])('%s builds snapshots with the required-ring capture', async name => {
  const device = createNullWebGPUDevice();
  let captures: Record<string, runner.S3CaptureSpec> = {};
  const spy = vi.spyOn(runner, 'runS3Lens').mockImplementation(async (_device, props) => {
    const graph = new LensCommandGraph<undefined>(device);
    captures = await props.build(graph, createUploads(graph));
    const snapshot = addLensViewSnapshot(graph, device, {
      id: 'capture-review',
      views: Object.fromEntries(
        Object.entries(captures).map(([key, capture]) => [key, capture.view])
      )
    });
    snapshot.arena.destroy();
    return {} as never;
  });
  try {
    const scenario = await getLensScenario(`gpu-spatial-analysis.${name}`)!.load();
    await scenario.run(device);
    expect(captures).toHaveProperty('polygonRequiredRingCount');
    expect(captures).not.toHaveProperty('polygonRequiredCount');
    expect(captures['polygonRequiredRingCount'].view.length).toBe(1);
    // LENS-108: after candidate loss the assembled requirement is only a lower bound, so the
    // label must not present it as the exact capacity to allocate.
    const meaning = captures['polygonRequiredRingCount'].meaning ?? '';
    expect(meaning).toContain('only a lower bound when it is 1');
    expect(meaning).toContain('polygonCandidateOverflow');
    expect(captures).toHaveProperty('polygonCandidateOverflow');
    expect(meaning).not.toMatch(/^face rings needed \(ring capacity to allocate\)$/);
  } finally {
    spy.mockRestore();
    device.destroy();
  }
});

it('geometry validity captures diagnostics as issueCode instead of a filter-compatible mask', async () => {
  const device = createNullWebGPUDevice();
  let captures: Record<string, simpleRunner.S2CaptureSpec> = {};
  const spy = vi.spyOn(simpleRunner, 'runS2Lens').mockImplementation(async (_device, props) => {
    const graph = new LensCommandGraph<undefined>(device);
    captures = await props.build(graph, createUploads(graph));
    expect(captures).toHaveProperty('issueCode');
    expect(captures).not.toHaveProperty('mask');
    const geometry = captures['resultParks'].geometry;
    expect(geometry && 'valuesCapture' in geometry ? geometry.valuesCapture : undefined).toBe(
      'issueCode'
    );
    const snapshot = addLensViewSnapshot(graph, device, {
      id: 'validity-snapshot',
      views: Object.fromEntries(
        Object.entries(captures).map(([name, capture]) => [name, capture.view])
      )
    });
    snapshot.arena.destroy();
    return {} as never;
  });
  try {
    await (await getLensScenario('gpu-spatial-analysis.geometry-validity')!.load()).run(device);
  } finally {
    spy.mockRestore();
    device.destroy();
  }
});

it('typechecks the displayed geometry validity and lon-lat preparation examples', () => {
  const selected = entries.filter(entry =>
    ['GPUGeometryValidity', 'GPULonLatPolygonPreparation'].includes(entry.symbol)
  );
  expect(selected).toHaveLength(2);
  const prelude = `import type {Device} from '@luma.gl/core';
    import type {GraphDataView} from '@luma.gl/graph/gpu-core';
    import type {GPULonLatPolygonPreparationOutput} from '@luma-contributors/analysis/gpu-spatial-analysis';
    declare const device: Device;
    declare const positions: GraphDataView<'float32x2'>;
    declare const featureOffsets: GraphDataView<'uint32'>, polygonOffsets: GraphDataView<'uint32'>, ringOffsets: GraphDataView<'uint32'>;
    declare const issueCode: GraphDataView<'uint32'>, validity: GraphDataView<'uint32'>, overflow: GraphDataView<'uint32'>;
    declare const parameters: GraphDataView<'float32'>;
    declare const output: GPULonLatPolygonPreparationOutput;`;
  const files = new Map(
    selected.map(entry => [
      resolve(import.meta.dirname, `geo-ports-${entry.symbol}.virtual.ts`),
      `${prelude}\n${entry.example}`
    ])
  );
  const parsed = ts.readConfigFile(resolve('tsconfig.json'), ts.sys.readFile);
  const configuration = ts.parseJsonConfigFileContent(parsed.config, ts.sys, process.cwd());
  const options = {...configuration.options, noEmit: true};
  const host = ts.createCompilerHost(options);
  const original = host.getSourceFile.bind(host);
  host.getSourceFile = (file, version, onError, shouldCreate) =>
    files.has(file)
      ? ts.createSourceFile(file, files.get(file)!, version, true)
      : original(file, version, onError, shouldCreate);
  const program = ts.createProgram([...files.keys()], options, host);
  const diagnostics = program
    .getSemanticDiagnostics()
    .filter(diagnostic => diagnostic.file && files.has(diagnostic.file.fileName));
  expect(
    diagnostics.map(diagnostic => ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'))
  ).toEqual([]);
  expect(selected.find(entry => entry.symbol === 'GPUGeometryValidity')!.example).toContain(
    'intersectionCapacity:'
  );
});
