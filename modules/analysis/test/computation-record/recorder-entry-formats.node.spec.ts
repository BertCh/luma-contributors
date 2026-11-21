// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {execFileSync} from 'node:child_process';
import {existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {buildSync} from 'esbuild';
import {expect, test} from 'vitest';

test('ENT-03 / B2: recorder describes selected compilations across ESM/CommonJS, including reduction', () => {
  const scratchRoot = resolve(
    '.typecheck/r19/scratch/dataframe-pointcloud-record-w5-pointcloud-and-record-integration'
  );
  mkdirSync(scratchRoot, {recursive: true});
  const directory = mkdtempSync(resolve(scratchRoot, 'entry-formats-'));
  try {
    const contents = `
    export {GPUComputationRecorder, describeGPUCommandGraph} from './modules/analysis/src/computation-record/index.ts';
    export {createWGSLKernelNode, getWGSLKernelNodeRecord, getCompiledWGSLKernelNodeRecord} from './modules/analysis/src/analysis-core/kernels/wgsl-kernel-nodes.ts';
    export {createSegmentSumNode} from './modules/analysis/src/analysis-core/kernels/sorted-segment-sums.ts';
  `;
    for (const format of ['esm', 'cjs'] as const) {
      buildSync({
        stdin: {contents, resolveDir: process.cwd(), loader: 'ts'},
        bundle: true,
        packages: 'external',
        platform: 'node',
        format,
        outfile: resolve(directory, format === 'esm' ? 'records.mjs' : 'records.cjs')
      });
    }
    // Native Node loads both formats; Vite must not collapse the independent module instances.
    const probe = resolve(directory, 'recorder-entry-probe.mjs');
    writeFileSync(
      probe,
      `
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {NullDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, createTransientView, createGPUComputeCommandNode} from '@luma.gl/graph/gpu-core';
const esm = await import('./records.mjs');
const cjs = createRequire(import.meta.url)('./records.cjs');
for (const [reader, writer] of [[esm, cjs], [cjs, esm]]) {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {value: {...device.limits,
    maxComputeWorkgroupsPerDimension: 65535, maxStorageBuffersPerShaderStage: 8,
    maxStorageBufferBindingSize: 134217728}});
  Object.defineProperty(device, 'props', {value: {...device.props, _cacheShaders: false, _cachePipelines: false}});
  const pipelineSources = [];
  Object.defineProperty(device, 'createComputePipeline', {value: props => {
    pipelineSources.push(props.shader.source);
    return {id: props.id ?? '', hash: '', destroy() {}};
  }});
  const recorder = new reader.GPUComputationRecorder({sourceText: true});
  const unhandled = [];
  const onUnhandled = reason => unhandled.push(reason.message);
  process.on('unhandledRejection', onUnhandled);
  try {
    assert.throws(() => recorder.withSynchronousScope({}, () => Promise.reject(new Error('callback failure'))), /withAsyncScope/);
    assert.throws(() => recorder.withSynchronousScope({}, () => ({then(_resolve, reject) {reject(new Error('thenable failure'));}})), /withAsyncScope/);
    await new Promise(resolve => setTimeout(resolve, 0));
    assert.deepEqual(unhandled, [], 'boundary must observe rejection of already-started callbacks');
  } finally {process.off('unhandledRejection', onUnhandled);}
  const graph = recorder.recordKernels(new GPUCommandGraph(device));
  const inputStorage = createTransientView(graph, 'input', 'float32', 68);
  const outputStorage = createTransientView(graph, 'output', 'float32', 135);
  const offsetsStorage = createTransientView(graph, 'offsets', 'uint32', 72);
  const input = graph.createDataView(inputStorage.buffer, {format: 'float32', length: 3, byteOffset: 260});
  const output = graph.createDataView(outputStorage.buffer, {format: 'float32', length: 5, byteOffset: 520});
  const offsets = graph.createDataView(offsetsStorage.buffer, {format: 'uint32', length: 6, byteOffset: 264});
  const ordinary = writer.createWGSLKernelNode(graph, {
    id: 'scale', operation: 'Scale', invocationCount: 3, workgroupSize: 2,
    bindings: [{name: 'input', view: input, type: 'f32', access: 'read'},
      {name: 'output', view: output, type: 'f32', access: 'read_write'}],
    body: 'let executed = index; output[outputOffset + executed] = input[inputOffset + executed] * 2.0;'
  });
  const reduction = writer.createSegmentSumNode(graph, {
    id: 'sum', operation: 'RecordedSum', segmentCount: 2,
    input, output, segmentOffsets: offsets
  });
  graph.add([ordinary, reduction]);
  // An unselected same-ID declaration must never replace the selected execution facts.
  const declareUnusedScale = () => writer.createWGSLKernelNode(graph, {
    id: 'scale', operation: 'Unused', invocationCount: 1, workgroupSize: 2,
    bindings: [{name: 'unusedOutput', view: offsets, type: 'u32', access: 'read_write'}],
    body: 'let unused = index; unusedOutput[unusedOutputOffset + unused] = 0u;'
  });
  declareUnusedScale();
  assert.ok(writer.getWGSLKernelNodeRecord(graph, 'scale').body.includes('let unused = index;'));
  graph.add(createGPUComputeCommandNode({
    id: 'custom', workload: {operation: 'Custom', commandCount: 1},
    kernel: {source: '@compute @workgroup_size(1) fn customEntry() {}', entryPoint: 'customEntry',
      workgroupSize: [1, 1, 1], bindings: []},
    compile: () => ({encode() {}})
  }));
  writer.createWGSLKernelNode(graph, {
    id: 'custom', operation: 'UnselectedCustomCandidate', invocationCount: 1,
    bindings: [{name: 'unusedOutput', view: offsets, type: 'u32', access: 'read_write'}],
    body: 'unusedOutput[unusedOutputOffset + index] = 0u;'
  });
  const compiled = graph.compile({kernelMetadata: 'source'});
  declareUnusedScale(); // Later unused declarations cannot overwrite the captured compilation either.
  try {
    const description = reader.describeGPUCommandGraph(compiled, {recorder});
    const custom = description.graph.nodes.find(node => node.id === 'custom');
    assert.equal(description.sources.find(source => source.id === custom.source).text,
      '@compute @workgroup_size(1) fn customEntry() {}');
    assert.equal(custom.dispatch, undefined);
    assert.equal(custom.bindings, undefined);
    assert.deepEqual(custom.declaredBindings, []);
    for (const [index, node] of [ordinary, reduction].entries()) {
      const kernel = writer.getCompiledWGSLKernelNodeRecord(graph, node.id);
      assert.ok(kernel);
      const serialized = description.graph.nodes.find(entry => entry.id === node.id);
      assert.ok(serialized);
      assert.equal(serialized.id, node.id);
      if (node.id === 'scale') {
        assert.ok(description.sources.find(source => source.id === serialized.source).text.includes('let executed = index;'));
        assert.deepEqual(serialized.dispatch.workgroups, [2, 1, 1]);
        assert.deepEqual(serialized.bindings.map(binding => [binding.name, binding.resource, binding.elementOffset, binding.length]),
          [['input', 'input', 1, 3], ['output', 'output', 2, 5]]);
      }
      assert.equal(serialized.op, node.workload.operation);
      assert.equal(description.sources.find(source => source.id === serialized.source).text, pipelineSources[index]);
      assert.equal(kernel.source, pipelineSources[index]);
      assert.deepEqual(serialized.dispatch.workgroups, kernel.workgroups);
      // Aligned binding windows: input 260=256+4 (4+3*4 bytes), output 520=512+8,
      // and offsets 264=256+8. These ranges are independent of either record implementation.
      assert.deepEqual(serialized.bindings.map(binding => [binding.bindingByteOffset, binding.bindingByteLength]),
        node.id === 'scale' ? [[256, 16], [512, 28]] : [[256, 16], [256, 32], [512, 28]]);
      assert.deepEqual(serialized.bindings.map(binding => binding.length),
        node.resources.map(resource => resource.buffer.length));
      assert.deepEqual(serialized.bindings.map(binding => binding.resource),
        node.resources.map(resource => resource.buffer.buffer.id));
      // Run the returned node's actual encode callback against a recording compute pass.
      const bindings = new Map(node.resources.map(resource => [resource.buffer.buffer.id,
        device.createBuffer({id: resource.buffer.buffer.id, byteLength: resource.buffer.buffer.byteLength})]));
      let bound;
      let dispatched;
      const executable = node.compile({device});
      executable.encode({getBuffer: view => {
        const buffer = bindings.get(view.buffer.id);
        assert.ok(buffer, 'unresolved view ' + view.buffer.id);
        return buffer;
      }, computePass: {
        setPipeline() {}, setBindings(values) {bound = values;},
        dispatch(x, y, z) {dispatched = [x, y, z];}
      }});
      assert.deepEqual(dispatched, serialized.dispatch.workgroups);
      assert.deepEqual(Object.keys(bound), serialized.bindings.map(binding => binding.name));
      for (const binding of serialized.bindings) {
        assert.ok(bound[binding.name].buffer, 'missing executed buffer for ' + binding.name);
        assert.equal(bound[binding.name].buffer, bindings.get(binding.resource));
        assert.equal(bound[binding.name].offset, binding.bindingByteOffset);
        assert.equal(bound[binding.name].size, binding.bindingByteLength);
      }
      executable.destroy?.();
      for (const buffer of bindings.values()) buffer.destroy();
    }
    assert.equal(writer.getWGSLKernelNodeRecord(new GPUCommandGraph(device), 'scale'), undefined);
  } finally {compiled.destroy(); device.destroy();}
}
console.log('ENT-03 recorder/description and B3 reduction PASS both format directions');
`
    );
    expect(execFileSync(process.execPath, [probe], {encoding: 'utf8'})).toContain(
      'PASS both format directions'
    );
  } finally {
    rmSync(directory, {recursive: true, force: true});
  }
  expect(existsSync(directory)).toBe(false);
});
