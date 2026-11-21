// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWGSLKernelNodeRecord} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  createSphericalReviewFixture,
  createPlanarIdentityFixture
} from './spherical-overlay-review-fixture';

it('planar directed boundary IDs use caller identities while representative provenance keeps row one', () => {
  const device = createNullWebGPUDevice();
  const fixture = createPlanarIdentityFixture(device);
  try {
    const memory = createCPUGraphMemory();
    const seed = (suffix: string, name: string, values: Uint32Array | Float32Array) => {
      const node = fixture.nodes.find(candidate => candidate.id.endsWith(suffix))!;
      const binding = getWGSLKernelNodeRecord(fixture.graph, node.id)!.bindings.find(
        candidate => candidate.name === name
      )!;
      memory.write(binding.view, values);
    };
    seed(
      '-emit-boundary-provenance',
      'selected',
      Uint32Array.from({length: 32}, (_, row) => Number(row < 3))
    );
    seed(
      '-emit-boundary-provenance',
      'scan',
      Uint32Array.from({length: 32}, (_, row) => Math.min(row + 1, 3))
    );
    seed(
      '-emit-boundary-provenance',
      'heads',
      Uint32Array.from({length: 32}, (_, row) => row)
    );
    seed(
      '-emit-boundary-provenance',
      'sortedIndices',
      Uint32Array.from({length: 32}, (_, row) => row)
    );
    seed(
      '-emit-boundary-provenance',
      'packedSources',
      Uint32Array.of(1, 0x80000001, 1, ...Array(29).fill(0))
    );
    seed('-source-shells', 'shells', Uint32Array.of(0, 0xffffffff));
    seed('-source-shells', 'areas', Float32Array.of(1, 0));
    seed('-source-shells', 'ringCount', Uint32Array.of(1));
    seed(
      '-reduce-feature-sources',
      'segmentRings',
      Uint32Array.of(0, 0, 0, ...Array(5).fill(0xffffffff))
    );
    seed('-reduce-feature-sources', 'ranks', Uint32Array.of(0, 1));
    memory.write(fixture.outputs.boundaryCount.view, Uint32Array.of(3));
    memory.write(fixture.props.left.sourceIds!, Uint32Array.of(999, 101));
    memory.write(fixture.props.right!.sourceIds!, Uint32Array.of(888, 202));
    const selected = fixture.nodes.filter(node =>
      [
        '-emit-boundary-provenance',
        '-source-shells',
        '-reduce-feature-sources',
        '-publish-feature-sources',
        '-source-feature-ids',
        '-publish-source-feature-ids'
      ].some(suffix => node.id.endsWith(suffix))
    );
    executeCommandNodesOnCPU(selected, memory);
    expect(Array.from(memory.readUint32(fixture.outputs.owners.view)).slice(0, 3)).toEqual([
      101, 202, 101
    ]);
    expect(memory.readUint32(fixture.outputs.sourceRows.view)[0]).toBe(1);
    expect(memory.readUint32(fixture.outputs.sourceOperands.view)[0]).toBe(0);
    expect(memory.readUint32(fixture.outputs.mixed.view)[0]).toBe(1);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it.each([
  false,
  true
])('complemented intersection reduces caller-numbered owners (mixed=%s)', mixed => {
  const device = createNullWebGPUDevice();
  const fixture = createSphericalReviewFixture(device, {leftComplement: true});
  try {
    const memory = createCPUGraphMemory();
    const kernels = fixture.nodes.filter(
      node =>
        node.id.endsWith('-source-shells') ||
        node.id.endsWith('-reduce-feature-sources') ||
        node.id.endsWith('-publish-feature-sources') ||
        node.id.endsWith('-operand-provenance')
    );
    const seed = (suffix: string, name: string, values: Uint32Array | Float32Array) => {
      const node = kernels.find(candidate => candidate.id.endsWith(suffix))!;
      const binding = getWGSLKernelNodeRecord(fixture.graph, node.id)!.bindings.find(
        candidate => candidate.name === name
      )!;
      memory.write(binding.view, values);
    };
    // One shell owns three edges. Internal operand zero is the original right operand.
    // On a mixed shell the caller's operand zero (row 2) must beat operand one (row 1).
    seed('-source-shells', 'shells', Uint32Array.of(0, 0xffffffff));
    seed('-source-shells', 'areas', Float32Array.of(1, 0));
    seed('-source-shells', 'ringCount', Uint32Array.of(1));
    seed(
      '-reduce-feature-sources',
      'segmentRings',
      Uint32Array.of(0, 0, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff)
    );
    seed('-reduce-feature-sources', 'ranks', Uint32Array.of(0, 1));
    memory.write(fixture.outputs.boundaryCount.view, Uint32Array.of(3));
    memory.write(fixture.outputs.operands.view, Uint32Array.of(0, mixed ? 1 : 0, 0, 0, 0, 0, 0, 0));
    memory.write(
      fixture.outputs.owners.view,
      Uint32Array.of(mixed ? 1 : 0, mixed ? 2 : 0, mixed ? 1 : 0, 0, 0, 0, 0, 0)
    );
    executeCommandNodesOnCPU(kernels, memory);
    expect(Array.from(memory.readUint32(fixture.outputs.operands.view)).slice(0, 3)).toEqual(
      mixed ? [1, 0, 1] : [1, 1, 1]
    );
    expect(memory.readUint32(fixture.outputs.sourceOperands.view)[0]).toBe(mixed ? 0 : 1);
    expect(memory.readUint32(fixture.outputs.sourceRows.view)[0]).toBe(mixed ? 2 : 0);
    expect(memory.readUint32(fixture.outputs.mixed.view)[0]).toBe(Number(mixed));
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('spherical boundary IDs read caller identities while feature sources retain rows', () => {
  const device = createNullWebGPUDevice();
  const fixture = createSphericalReviewFixture(device, {
    leftRow: 1,
    rightRow: 1,
    leftIds: [999, 101],
    rightIds: [888, 202]
  });
  try {
    const node = fixture.nodes.find(candidate => candidate.id.endsWith('-source-feature-ids'));
    expect(node, 'caller IDs need a publisher after representative row reduction').toBeDefined();
    const memory = createCPUGraphMemory();
    memory.write(fixture.props.left.sourceIds!, Uint32Array.of(999, 101));
    memory.write(fixture.props.right.sourceIds!, Uint32Array.of(888, 202));
    memory.write(fixture.outputs.boundaryCount.view, Uint32Array.of(3));
    memory.write(fixture.outputs.operands.view, Uint32Array.of(0, 1, 0, 0, 0, 0, 0, 0));
    memory.write(fixture.outputs.owners.view, Uint32Array.of(1, 1, 1, 0, 0, 0, 0, 0));
    memory.write(fixture.outputs.sourceRows.view, Uint32Array.of(1, 0xffffffff));
    executeCommandNodesOnCPU(
      fixture.nodes.filter(candidate => candidate.id.endsWith('-source-feature-ids')),
      memory
    );
    expect(Array.from(memory.readUint32(fixture.outputs.owners.view)).slice(0, 3)).toEqual([
      101, 202, 101
    ]);
    expect(Array.from(memory.readUint32(fixture.outputs.sourceRows.view))).toEqual([1, 0xffffffff]);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('bounded boundary publisher separates written two from required three', () => {
  const device = createNullWebGPUDevice();
  const fixture = createSphericalReviewFixture(device, {boundaryCapacity: 2});
  try {
    const node = fixture.nodes.find(
      candidate => candidate.id === 'spherical-review-arrangement-boundary-status'
    )!;
    const bindings = getWGSLKernelNodeRecord(fixture.graph, node.id)!.bindings;
    const memory = createCPUGraphMemory();
    const scan = bindings.find(binding => binding.name === 'scan')!.view;
    memory.write(
      scan,
      Uint32Array.from({length: scan.length}, () => 3)
    );
    for (const name of ['nodingOverflow', 'nodingCandidateOverflow'])
      memory.write(bindings.find(binding => binding.name === name)!.view, Uint32Array.of(0));
    executeCommandNodesOnCPU([node], memory);
    expect(Array.from(memory.readUint32(fixture.outputs.boundaryCount.view))).toEqual([2]);
    expect(Array.from(memory.readUint32(fixture.outputs.boundaryRequired.view))).toEqual([3]);
    expect(Array.from(memory.readUint32(fixture.outputs.boundaryOverflow.view))).toEqual([1]);
    expect(Array.from(memory.readUint32(fixture.outputs.boundaryCandidate.view))).toEqual([0]);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('tagged packed boundary columns do not mix physical storage access modes', () => {
  const device = createNullWebGPUDevice();
  const fixture = createSphericalReviewFixture(device, {
    leftIds: [101],
    rightIds: [202],
    packBoundary: true
  });
  try {
    for (const node of fixture.nodes.filter(candidate =>
      candidate.id.endsWith('-source-feature-ids')
    )) {
      const bindings = getWGSLKernelNodeRecord(fixture.graph, node.id)!.bindings;
      const readBuffers = new Set(
        bindings.filter(binding => binding.access === 'read').map(binding => binding.view.buffer)
      );
      for (const binding of bindings.filter(binding => binding.access === 'read_write'))
        expect(readBuffers.has(binding.view.buffer)).toBe(false);
    }
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
