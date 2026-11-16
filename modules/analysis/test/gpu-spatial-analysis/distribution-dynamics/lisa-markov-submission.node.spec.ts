// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  getViewBindingRange,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {GPULISAMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-lisa-markov';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import * as giddy from './markov-giddy-fixture';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-parameters';

for (const diagnostics of [false, true]) {
  it(`audits the 37-row giddy LISA submission and executes its state kernel (diagnostics ${diagnostics})`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `lisa-${serial++}`, format, length);
    try {
      const rows = 37;
      const periods = 4;
      const output = {
        counts: view('uint32', 25),
        quadrants: view('uint32', rows * periods),
        ignored: view('uint32', 1)
      };
      const props = {
        values: view('float32', rows * periods),
        weights: {
          offsets: view('uint32', rows + 1),
          neighbors: view('uint32', 123),
          weights: view('float32', 123),
          information: {rowCount: rows, transform: 'row' as const, symmetric: false}
        },
        periodCount: periods,
        parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
        output: {
          ...output,
          ...(diagnostics
            ? {steadyStates: view('float32', 5), stateSparsity: view('uint32', 2)}
            : {})
        }
      };
      const nodes = new GPULISAMarkov(props).getCommandNodes(graph);
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
        if (kernel.source.includes('workgroupBarrier()')) {
          expect(kernel.source, kernel.id).not.toMatch(
            /if\s*\(index\s*>=\s*INVOCATION_COUNT\)\s*\{\s*return/
          );
        }
      }
      for (const node of nodes) {
        const reads = new Set(
          (node.resources ?? [])
            .filter(resource => 'buffer' in resource && resource.usage === 'storage-read')
            .map(resource =>
              'buffer' in resource
                ? 'buffer' in resource.buffer
                  ? resource.buffer.buffer
                  : resource.buffer
                : undefined
            )
        );
        for (const resource of node.resources ?? []) {
          if ('buffer' in resource && resource.usage === 'storage-read-write')
            expect(
              reads.has('buffer' in resource.buffer ? resource.buffer.buffer : resource.buffer),
              node.id
            ).toBe(false);
        }
      }
      // Independently committed giddy states supply the upstream Moran stage's output. Execute
      // the actual unpack/exclusion WGSL, including untouched alignment padding. Cooperative
      // histogram barriers require real WebGPU and are covered by the companion headless spec.
      for (const [values, states] of [
        [giddy.C1_VALUES, giddy.C1_STATES],
        [giddy.C2_VALUES, giddy.C2_STATES]
      ] as const) {
        const expected = states.map(state => (state < 0 ? 0xffffffff : state));
        const memory = new Map<GraphDataView['buffer'], ArrayBuffer>();
        const getMemory = (field: GraphDataView) => {
          let bytes = memory.get(field.buffer);
          if (!bytes) {
            bytes = new ArrayBuffer(field.buffer.byteLength);
            new Uint32Array(bytes).fill(0x7f7f7f7f);
            memory.set(field.buffer, bytes);
          }
          return bytes;
        };
        new Float32Array(getMemory(props.values)).set(values);
        const stateRecord = getWGSLKernelNodeRecord(graph, 'lisa-markov-states')!;
        const moran = stateRecord.bindings.find(binding => binding.name === 'moranQuadrants')!.view;
        const neighbors = stateRecord.bindings.find(
          binding => binding.name === 'neighborCounts'
        )!.view;
        const moranValues = new Uint32Array(getMemory(moran));
        const neighborValues = new Uint32Array(getMemory(neighbors));
        for (let period = 0; period < periods; period++)
          for (let row = 0; row < rows; row++) {
            const state = expected[period * rows + row];
            moranValues[period * 64 + row] = state === 0xffffffff ? 0 : state;
            neighborValues[period * 64 + row] = state === 0xffffffff ? 0 : 1;
          }
        for (const id of ['lisa-markov-states']) {
          const record = getWGSLKernelNodeRecord(graph, id)!;
          const bindings = Object.fromEntries(
            record.bindings.map((binding, index) => {
              const bytes = getMemory(binding.view);
              const range = getViewBindingRange(binding.view);
              return [
                index,
                binding.type === 'f32'
                  ? new Float32Array(bytes, range.offset, range.size / 4)
                  : new Uint32Array(bytes, range.offset, range.size / 4)
              ];
            })
          );
          // WgslExec needs the equivalent three-axis attribute; it treats a scalar string as characters.
          new WgslExec(
            new WgslParser().parse(
              record.source
                .replace(/^\s*_\s*=[^;]*;/gm, '')
                .replace(/@workgroup_size\((\d+)\)/g, '@workgroup_size($1, 1, 1)')
            )
          ).dispatchWorkgroups(
            'main',
            [...record.workgroups],
            {0: bindings},
            {constants: {...record.constants}}
          );
        }
        expect([...new Uint32Array(getMemory(output.quadrants))]).toEqual(expected);
      }
    } finally {
      device.destroy();
    }
  });
}
