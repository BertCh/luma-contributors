// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {Buffer, Texture} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {AnalysisValidationError} from '../../../src/analysis-core/kernels/analysis-validation-error';
import {validateMaximumCount} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {auditKernelCase} from './kernel-audit-runner';
import {createKernelAuditContext, getAuditResourceCounts} from './kernel-audit-context';
import type {KernelAuditRejection} from './kernel-audit-rejection-type';

const identity = {id: 'ui-instance', contributor: 'SyntheticContributor'} as const;

const expected: KernelAuditRejection = {
  exportName: 'SyntheticRefusal',
  name: 'above limit',
  kind: 'binding',
  phase: 'construction',
  limit: 7,
  rejects: /^Error: ui-instance count must be at most 7$/,
  validation: {
    code: 'INTEGER_RANGE',
    path: 'count',
    received: 8,
    expected: {maximum: 7},
    id: 'ui-instance',
    contributor: 'SyntheticContributor'
  }
};

class SyntheticContributor {
  constructor(props: {graph: GPUCommandGraph; count: number; mutateFirst?: boolean}) {
    const declare = () => {
      props.graph.createTransientBuffer({
        id: 'same-id-scratch',
        byteLength: 4,
        usage: Buffer.STORAGE
      });
      props.graph.add({
        id: 'same-id-node',
        type: 'compute',
        resources: [],
        compile: () => ({encode: () => {}})
      });
    };
    if (props.mutateFirst) declare();
    validateMaximumCount(identity, 'count', props.count, 7);
    if (!props.mutateFirst) declare();
  }
}

it('VAL-13 does not certify matching refusal text after node and resource mutation', () => {
  const result = auditKernelCase(
    {
      exportName: expected.exportName,
      name: expected.name,
      build: ({graph, view, construct}) => {
        view('uint32', 1, 'fixture');
        construct(SyntheticContributor, {graph, count: 8, mutateFirst: true});
      }
    },
    expected
  );
  expect(result.nodeCount).toBe(1);
  expect(result.rejected).toBe(false);
  expect(result.findings.map(finding => finding.check)).toEqual(['refusal mutated graph']);
  const observation = result.rejectionObservation!;
  expect(observation.before.buffers).toBe(1); // fixture excluded from the measured delta
  expect(observation.after.buffers - observation.before.buffers).toBe(1);
  expect(observation.after.nodes - observation.before.nodes).toBe(1);
});

it('VAL-13 corrected construction rejects before mutation and permits a same-ID retry', () => {
  const context = createKernelAuditContext('same-id-retry');
  try {
    context.view('uint32', 1, 'fixture');
    const before = getAuditResourceCounts(context);
    expect(() => context.construct(SyntheticContributor, {graph: context.graph, count: 8})).toThrow(
      /count must be at most 7/
    );
    expect(context.rejectionObservation).toMatchObject({
      phase: 'construction',
      before,
      after: before
    });
    context.construct(SyntheticContributor, {graph: context.graph, count: 7});
    expect(getAuditResourceCounts(context).nodes - before.nodes).toBe(1);
    expect(getAuditResourceCounts(context).buffers - before.buffers).toBe(1);
  } finally {
    context.device.destroy();
  }
});

it('VAL-13 detects resource-only mutation and a refusal in the wrong phase', () => {
  const result = auditKernelCase(
    {
      exportName: expected.exportName,
      name: expected.name,
      build: ({graph, declare}) => {
        declare(() => {
          graph.device.createBuffer({byteLength: 4});
          graph.createTransientTexture({
            id: 'leaked-texture',
            width: 1,
            height: 1,
            format: 'rgba8unorm',
            usage: Texture.STORAGE
          });
          validateMaximumCount(identity, 'count', 8, 7);
        });
      }
    },
    expected
  );
  expect(result.nodeCount).toBe(0);
  expect(result.rejected).toBe(false);
  expect(result.findings.map(finding => finding.check)).toEqual([
    'refusal phase',
    'refusal mutated graph'
  ]);
  expect(
    result.rejectionObservation!.after.deviceResources -
      result.rejectionObservation!.before.deviceResources
  ).toBe(1);
  expect(
    result.rejectionObservation!.after.textures - result.rejectionObservation!.before.textures
  ).toBe(1);
});

it('VAL-12 requires structured fields even when representative message text matches', () => {
  const result = auditKernelCase(
    {
      exportName: expected.exportName,
      name: expected.name,
      build: ({declare}) =>
        declare(() => {
          throw new Error('ui-instance count must be at most 7');
        })
    },
    {...expected, phase: 'declaration'}
  );
  expect(result.rejected).toBe(false);
  expect(result.findings.some(finding => finding.check === 'structured refusal class')).toBe(true);
  expect(result.findings.filter(finding => finding.check === 'structured refusal')).toHaveLength(6);
});

it('review B2 retains the operation snapshot when fixture cleanup destroys a leaked resource', () => {
  const result = auditKernelCase(
    {
      exportName: expected.exportName,
      name: expected.name,
      build: ({device, declare}) => {
        let buffer: Buffer | undefined;
        try {
          declare(() => {
            buffer = device.createBuffer({byteLength: 4});
            validateMaximumCount(identity, 'count', 8, 7);
          });
        } finally {
          buffer?.destroy();
        }
      }
    },
    {...expected, phase: 'declaration'}
  );
  expect(
    result.rejectionObservation!.after.deviceResources -
      result.rejectionObservation!.before.deviceResources
  ).toBe(1);
  expect(result.rejected).toBe(false);
  expect(result.findings.map(finding => finding.check)).toEqual(['refusal mutated graph']);
});

it('review B2 excludes fixture teardown after a clean refusal', () => {
  const result = auditKernelCase(
    {
      exportName: expected.exportName,
      name: expected.name,
      build: ({device, declare}) => {
        const fixture = device.createBuffer({byteLength: 4});
        try {
          declare(() => validateMaximumCount(identity, 'count', 8, 7));
        } finally {
          fixture.destroy();
        }
      }
    },
    {...expected, phase: 'declaration'}
  );
  expect(result.rejectionObservation!.after).toEqual(result.rejectionObservation!.before);
  expect(result.findings).toEqual([]);
  expect(result.rejected).toBe(true);
});

it('review B3 does not attach a recovered observation to a different escaping error', () => {
  const result = auditKernelCase(
    {
      exportName: expected.exportName,
      name: expected.name,
      build: ({declare}) => {
        try {
          declare(() => {
            throw new Error('recoverable');
          });
        } catch {
          /* fixture recovers */
        }
        validateMaximumCount(identity, 'count', 8, 7);
      }
    },
    {...expected, phase: 'declaration'}
  );
  expect(result.rejected).toBe(false);
  expect(result.findings.map(finding => finding.check)).toEqual(['missing refusal boundary']);
  expect(result.rejectionObservation).toBeUndefined();
});

it('review B5 unexpected build errors retain their single diagnostic', () => {
  const result = auditKernelCase({
    exportName: 'unregistered-error-fixture',
    name: 'unexpected',
    build: () => {
      throw new Error('build failure');
    }
  });
  expect(result.rejected).toBe(false);
  expect(result.findings.map(finding => finding.check)).toEqual(['build throws']);
});

const resourceFactories: ReadonlyArray<
  readonly [string, (device: import('@luma.gl/core').Device) => {destroy: () => void}]
> = [
  ['buffer', device => device.createBuffer({byteLength: 4})],
  [
    'texture',
    device =>
      device.createTexture({format: 'rgba8unorm', width: 1, height: 1, usage: Texture.STORAGE})
  ],
  [
    'shader',
    device =>
      device.createShader({stage: 'compute', source: '@compute @workgroup_size(1) fn main() {}'})
  ],
  ['sampler', device => device.createSampler({})],
  ['framebuffer', device => device.createFramebuffer({colorAttachments: []})],
  [
    'vertex-array',
    device =>
      device.createVertexArray({shaderLayout: {attributes: [], bindings: []}, bufferLayout: []})
  ],
  [
    'transform-feedback',
    device => device.createTransformFeedback({layout: {attributes: [], bindings: []}, buffers: {}})
  ],
  ['query-set', device => device.createQuerySet({type: 'occlusion', count: 1})],
  ['fence', device => device.createFence()],
  [
    'render-pipeline',
    device => device.createRenderPipeline({shaderLayout: {attributes: [], bindings: []}})
  ],
  ['command-encoder', device => device.createCommandEncoder()]
];
for (const [name, create] of resourceFactories) {
  it(`review B4 detects resource-only ${name} mutation and tracks destruction`, () => {
    const context = createKernelAuditContext('factory-count');
    try {
      const before = getAuditResourceCounts(context).deviceResources;
      const resource = create(context.device);
      expect(getAuditResourceCounts(context).deviceResources - before).toBe(1);
      resource.destroy();
      expect(getAuditResourceCounts(context).deviceResources).toBe(before);
    } finally {
      context.device.destroy();
    }
    const result = auditKernelCase(
      {
        exportName: expected.exportName,
        name: expected.name,
        build: ({device, declare}) =>
          declare(() => {
            create(device);
            validateMaximumCount(identity, 'count', 8, 7);
          })
      },
      {...expected, phase: 'declaration'}
    );
    expect(result.nodeCount).toBe(0);
    expect(result.rejected).toBe(false);
    expect(result.findings.map(finding => finding.check)).toEqual(['refusal mutated graph']);
  });
}

it('review S3 rejects swapped custom instance and contributor identities independently', () => {
  const result = auditKernelCase(
    {
      exportName: expected.exportName,
      name: expected.name,
      build: ({declare}) =>
        declare(() => {
          throw new AnalysisValidationError(
            {id: identity.contributor, contributor: identity.id, messagePrefix: identity.id},
            'count must be at most 7',
            {code: 'INTEGER_RANGE', path: 'count', received: 8, expected: {maximum: 7}}
          );
        })
    },
    {...expected, phase: 'declaration'}
  );
  expect(result.rejected).toBe(false);
  expect(result.findings.map(finding => finding.check)).toEqual([
    'structured refusal',
    'structured refusal'
  ]);
  expect(result.findings[0].detail).toContain(
    'id: expected "ui-instance", received "SyntheticContributor"'
  );
  expect(result.findings[1].detail).toContain(
    'contributor: expected "SyntheticContributor", received "ui-instance"'
  );
});
