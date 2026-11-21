// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUPointCloudCoverageValidity} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-coverage-validity';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  getRasterMetadataFromPointCloudGrid,
  validatePointCloudGrid
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-surfaces';
import {
  GPUPointCloudChange,
  getGPUPointCloudChangeParameterValues,
  decodeGPUPointCloudChangeRow,
  decodeGPUPointCloudChangeStatus
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-change';
import {
  decodeGPUPointCloudNormalScaleStatus,
  GPU_POINT_CLOUD_NORMAL_SCALE_STATUS_NO_USABLE_SCALE,
  GPU_POINT_CLOUD_NORMAL_SCALE_STATUS_CANDIDATE_OVERFLOW,
  GPU_POINT_CLOUD_NORMAL_SCALE_STATUS_EIGENSOLVE_EXHAUSTED
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-normal-scale';
import {createNullWebGPUDevice, getParsedKernelSources} from '../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';

it('UNI-04: aligns arrays to the lower-left grid footprint in both row directions', () => {
  const grid = {origin: [100, 200] as const, width: 2, height: 2, cellSize: 10};
  // Footprint is [100,120] x [200,220]; south rows start at its northern edge.
  expect(getRasterMetadataFromPointCloudGrid(grid).affine).toEqual([10, 0, 100, 0, -10, 220]);
  expect(getRasterMetadataFromPointCloudGrid({...grid, rowDirection: 'north'}).affine).toEqual([
    10, 0, 100, 0, 10, 200
  ]);
});

it('SCL-15: enforces the inclusive product limit independently of the axes', () => {
  const grid = {origin: [0, 0] as const, width: 65535, height: 65536, cellSize: 1};
  expect(() => validatePointCloudGrid(grid)).not.toThrow();
  expect(() => validatePointCloudGrid({...grid, height: 65537})).toThrow(/4294967294/);
});

it('VAL-07: rejects positive doubles that round to float32 zero in packer and constructor', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32x3' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `change-w3-${serial++}`, format, length);
    const props = {
      reference: view('float32x3', 3),
      comparison: view('float32x3', 3),
      cores: view('float32x3', 1),
      normals: view('float32x3', 1),
      alignment: 'already-aligned' as const,
      radius: 1,
      halfDepth: 1,
      output: {
        changes: view('float32x3', 1),
        roughness: view('float32x3', 1),
        counts: view('uint32', 2),
        status: view('uint32', 1)
      }
    };
    expect(Math.fround(1e-100)).toBe(0);
    expect(() => getGPUPointCloudChangeParameterValues({confidenceMultiplier: 1e-100})).toThrow(
      /confidenceMultiplier.*1e-100.*0/
    );
    expect(() => new GPUPointCloudChange({...props, confidenceMultiplier: 1e-100})).toThrow(
      /confidenceMultiplier.*1e-100.*0/
    );
    expect(() => new GPUPointCloudChange({...props, halfDepth: 1e-100})).toThrow(
      /halfDepth.*1e-100.*0/
    );
    expect(
      () => new GPUPointCloudChange({...props, halfDepth: 1e-40, confidenceMultiplier: 1e-40})
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('EGR-15: decodes independent outcome and grid flag, retaining the word', () => {
  for (const [word, code, outcome, gridIncomplete] of [
    [0, 0, 'valid', false],
    [1, 1, 'invalid-core', false],
    [2, 2, 'insufficient-support', false],
    [3, 3, 'nonfinite-uncertainty', false],
    [256, 0, 'valid', true],
    [258, 2, 'insufficient-support', true],
    [259, 3, 'nonfinite-uncertainty', true]
  ] as const) {
    expect(decodeGPUPointCloudChangeStatus(word)).toEqual({word, code, outcome, gridIncomplete});
  }
  expect(decodeGPUPointCloudChangeStatus(new Uint32Array([258])).outcome).toBe(
    'insufficient-support'
  );
  expect(() => decodeGPUPointCloudChangeStatus(new Uint8Array(3))).toThrow();
  expect(() => decodeGPUPointCloudChangeStatus(new Uint32Array(0))).toThrow();
  const bytes = (values: Float32Array | Uint32Array) => {
    const result = new Uint8Array(values.byteLength + 5);
    result.set(new Uint8Array(values.buffer), 1);
    return result.subarray(1, 1 + values.byteLength);
  };
  const output = {
    changes: bytes(new Float32Array([99, 99, 99, -2, -3, -1])),
    roughness: bytes(new Float32Array([99, 99, 99, 0.5, 0.75, 1])),
    counts: bytes(new Uint32Array([99, 99, 4, 5])),
    status: bytes(new Uint32Array([99, 256]))
  };
  expect(decodeGPUPointCloudChangeRow(output, 1)).toEqual({
    displacement: -2,
    lowerBound: -3,
    upperBound: -1,
    referenceRoughness: 0.5,
    comparisonRoughness: 0.75,
    detectionLimit: 1,
    referenceCount: 4,
    comparisonCount: 5,
    status: {word: 256, code: 0, outcome: 'valid', gridIncomplete: true}
  });
  for (const port of ['changes', 'roughness', 'counts', 'status'] as const) {
    expect(() => decodeGPUPointCloudChangeRow({...output, [port]: new Uint8Array(0)}, 1)).toThrow();
    expect(() => decodeGPUPointCloudChangeRow({...output, [port]: new Uint8Array(3)}, 1)).toThrow();
  }
  expect(() => decodeGPUPointCloudChangeRow(output, -1)).toThrow(/row/);
});

it('STA-41: names and decodes all normal-scale flag combinations', () => {
  expect([
    GPU_POINT_CLOUD_NORMAL_SCALE_STATUS_NO_USABLE_SCALE,
    GPU_POINT_CLOUD_NORMAL_SCALE_STATUS_CANDIDATE_OVERFLOW,
    GPU_POINT_CLOUD_NORMAL_SCALE_STATUS_EIGENSOLVE_EXHAUSTED
  ]).toEqual([1, 2, 4]);
  expect(decodeGPUPointCloudNormalScaleStatus(new Uint32Array([7])).candidateOverflow).toBe(true);
  expect(() => decodeGPUPointCloudNormalScaleStatus(new Uint8Array(3))).toThrow();
  expect(() => decodeGPUPointCloudNormalScaleStatus(new Uint32Array(0))).toThrow();
  for (let word = 0; word < 8; word++)
    expect(decodeGPUPointCloudNormalScaleStatus(word)).toEqual({
      word,
      noUsableScale: Boolean(word & 1),
      candidateOverflow: Boolean(word & 2),
      eigensolveExhausted: Boolean(word & 4)
    });
});

// GPU semantics are pinned by pointcloud-coverage-validity-w3.spec.ts. This host audit catches
// binding/type errors without claiming a shader execution in the Node project.
it('UNI-01: validates, snapshots and audits the reusable coverage conversion', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const coverage = createTransientView(graph, 'coverage', 'uint32', 2);
    const output = {
      surfaceValidity: createTransientView(graph, 'surface-validity', 'uint32', 2),
      terrainValidity: createTransientView(graph, 'terrain-validity', 'uint32', 2),
      canopyValidity: createTransientView(graph, 'canopy-validity', 'uint32', 2)
    };
    const contributor = new GPUPointCloudCoverageValidity({coverage, output});
    const captured = output.terrainValidity;
    output.terrainValidity = output.surfaceValidity;
    expect(contributor.props.output.terrainValidity).toBe(captured);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes).toHaveLength(1);
    for (const {source, constants} of getParsedKernelSources(nodes).configurations)
      expect(auditWGSLModule(source, constants)).toEqual([]);
    expect(() => new GPUPointCloudCoverageValidity({coverage, output})).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});
