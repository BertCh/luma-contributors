// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

// Arithmetic diagnostic only: run with the shared GPU slot, never concurrently with another suite.
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputFile = process.argv[3] ?? '/tmp/convex-wavefront-diagnostic.json';
const reference = JSON.parse(await readFile('/tmp/footprint-shape-independent-reference.json', 'utf8'));
const source = reference.references.find(row => row.sourceId === 814558);
const points = source.inputPolygons[0][0].slice(0, -1);
assert.equal(points.length, 4);
const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
try {
  const page = await browser.newPage();
  // This static JSON page supplies a secure localhost origin without starting a showcase scene.
  await page.goto(new URL('build-evidence.json', baseUrl).href);
  const result = await page.evaluate(async sourcePoints => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const input = device.createBuffer({size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    const output = device.createBuffer({size: 88, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
    const readback = device.createBuffer({size: 88, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    device.queue.writeBuffer(input, 0, Float32Array.from(sourcePoints.flat()));
    const module = device.createShaderModule({code: `
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
fn point(vertex: u32) -> vec2f { return vec2f(positions[vertex * 2u], positions[vertex * 2u + 1u]); }
fn cross(left: vec2f, right: vec2f) -> f32 { return left.x * right.y - left.y * right.x; }
@compute @workgroup_size(1) fn main() {
  let orientation = sign(cross(point(1u) - point(0u), point(2u) - point(1u)));
  var originalUnsupported = 0.0;
  var excludedEndpointsUnsupported = 0.0;
  for (var vertex = 0u; vertex < 4u; vertex++) {
    let current = point(vertex);
    let next = point((vertex + 1u) % 4u);
    let tangent = next - current;
    for (var other = 0u; other < 4u; other++) {
      let determinant = cross(tangent, point(other) - current) * orientation;
      result[4u * vertex + other] = determinant;
      if (determinant < 0.0) {
        originalUnsupported = 1.0;
        if (other != vertex && other != (vertex + 1u) % 4u) { excludedEndpointsUnsupported = 1.0; }
      }
    }
    result[16u + vertex] = cross(tangent, point((vertex + 2u) % 4u) - next) * orientation;
  }
  result[20u] = originalUnsupported;
  result[21u] = excludedEndpointsUnsupported;
}`});
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter(message => message.type === 'error').map(message => message.message);
    if (errors.length) throw new Error(errors.join('\n'));
    const pipeline = device.createComputePipeline({layout: 'auto', compute: {module, entryPoint: 'main'}});
    const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [
      {binding: 0, resource: {buffer: input}}, {binding: 1, resource: {buffer: output}}
    ]});
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, 88);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
    readback.unmap(); input.destroy(); output.destroy(); readback.destroy(); device.destroy();
    return {determinants: values.slice(0, 16), cornerDeterminants: values.slice(16, 20), originalUnsupported: values[20], excludedEndpointsUnsupported: values[21], errors};
  }, points);
  const evidence = await (await fetch(new URL('build-evidence.json', baseUrl), {headers: {connection: 'close'}})).json();
  await writeFile(outputFile, JSON.stringify({sourceId: source.sourceId, points, bundleSha256: evidence.bundleSha256, ...result}, null, 2) + '\n');
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
}
