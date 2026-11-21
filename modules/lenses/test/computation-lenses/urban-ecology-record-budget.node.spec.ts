// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {Buffer} from '@luma.gl/core';
import {expect, test} from 'vitest';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {describeGPUCommandGraph} from '@luma-contributors/analysis/computation-record';
import {
  LensCommandGraph,
  addLensViewSnapshot
} from '../../src/computation-lenses/lens-view-snapshot';
import {fromRecordedLensFile, toRecordedLensFile} from '../../src/computation-lenses/registry';
import {buildUrbanEcologyScenario} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/urban-ecology-examples';
import type {S2Uploads} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s2-lens-run';

for (const [scenario, element] of [
  ['building-adjacency-recipe', 'addBuildingAdjacencyRecipe'],
  ['building-tessellation-recipe', 'addBuildingTessellationRecipe']
] as const) {
  test(`${scenario} compiles a docs-sized record without dropping any captured evidence`, async () => {
    const scenarioId = `gpu-spatial-analysis.${scenario}`;
    const archived = fromRecordedLensFile(
      JSON.parse(
        readFileSync(
          new URL(`../../../../docs/lens-records/${scenarioId}.json`, import.meta.url),
          'utf8'
        )
      )
    );
    const device = createNullWebGPUDevice();
    Object.defineProperty(device, 'props', {
      value: {...device.props, _cacheShaders: false, _cachePipelines: false}
    });
    Object.defineProperty(device, 'createComputePipeline', {
      value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy() {}})
    });
    const graph = new LensCommandGraph<undefined>(device, {id: scenarioId.replace(/\./g, '-')});
    const owned: {destroy(): void}[] = [];
    const uploads: S2Uploads = {
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
        const handle = graph.importBuffer(
          {id, byteLength: buffer.byteLength, usage: buffer.usage},
          buffer
        );
        const components = Number(format.match(/x([234])$/)?.[1] ?? 1);
        return graph.createDataView(handle, {format, length: length ?? data.length / components});
      }
    };
    try {
      const specifications = await buildUrbanEcologyScenario(element, graph, uploads);
      expect(Object.keys(specifications).sort()).toEqual(
        archived.record.captures!.snapshots.map(capture => capture.name).sort()
      );
      const snapshot = addLensViewSnapshot(graph, device, {
        id: graph.id,
        views: Object.fromEntries(
          Object.entries(specifications).map(([name, capture]) => [name, capture.view])
        )
      });
      const compiled = graph.compile();
      try {
        const description = describeGPUCommandGraph(compiled, {
          domain: 'gpu-spatial-analysis',
          graph,
          sourceText: true
        });
        // No submission or GPU readback: distinct synthetic words bound the payload size without
        // depending on zero/sentinel compression. Actual numeric results are checked headlessly.
        const captures = archived.record.captures!.snapshots.map((capture, captureIndex) => {
          const view = specifications[capture.name].view;
          const components = Number(view.format.match(/x([234])$/)?.[1] ?? 1);
          const Values = view.format.startsWith('uint') ? Uint32Array : Float32Array;
          return {
            ...capture,
            values: new Values(
              Array.from(
                {length: view.length * components},
                (_, index) => (captureIndex + 1) * 100_000 + index
              )
            )
          };
        });
        const projected = toRecordedLensFile(
          {
            ...archived.record,
            graphs: [description.graph],
            refs: {...archived.record.refs, sources: description.sources},
            captures: {...archived.record.captures!, snapshots: captures}
          },
          archived
        );
        const bytes = new TextEncoder().encode(`${JSON.stringify(projected)}\n`).length;
        console.info(`${scenarioId}: ${bytes} projected bytes with synthetic capture words`);
        // The public record budget is 512 KiB (docs/api/computation-views.md). Retain 4 KiB
        // for timing/adapter metadata, matching the network envelope qualification fixture.
        expect(bytes, `${scenarioId}: ${bytes} projected bytes`).toBeLessThan(512 * 1024 - 4096);
      } finally {
        compiled.destroy();
        snapshot.arena.destroy();
      }
    } finally {
      for (const resource of owned) resource.destroy();
      device.destroy();
    }
  });
}
