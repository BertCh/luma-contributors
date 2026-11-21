// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {GPUNearestFeatureWeights} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const UNSUPPORTED_STATUS_FIELDS = [
  'count',
  'uncertainCount',
  'invalidCount',
  'iterationCount',
  'converged',
  'iterationLimitReached',
  'approximation',
  'candidateCount',
  'candidateTruncatedCount',
  'antimeridianUncertainCount',
  'poleUncertainCount'
] as const;

for (const field of UNSUPPORTED_STATUS_FIELDS) {
  for (const declaration of ['own', 'non-enumerable', 'inherited'] as const) {
    it(`refuses unwritten ${field} from ${declaration} status properties`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
        const status = {overflow: scalar('overflow')};
        const descriptor = {value: scalar('unsupported'), enumerable: declaration === 'own'};
        if (declaration === 'inherited')
          Object.setPrototypeOf(status, Object.create(null, {[field]: descriptor}));
        else Object.defineProperty(status, field, descriptor);
        // This is the declared producer contract, not an expectation derived from its shaders:
        // only final-capacity and candidate-loss flags exist; unrelated diagnostics must refuse.
        expect(
          () =>
            new GPUNearestFeatureWeights({
              id: 'nearest-status-oracle',
              slotCapacity: 1,
              neighborRows: scalar('neighborRows'),
              neighborCounts: scalar('neighborCounts'),
              output: {
                offsets: createTransientView(graph, 'offsets', 'uint32', 2),
                neighbors: scalar('neighbors'),
                weights: createTransientView(graph, 'weights', 'float32', 1),
                status
              }
            })
        ).toThrow(
          `nearest-status-oracle output.status.${field} is unsupported because this adapter publishes only overflow and candidateOverflow; omit ${field} and use those completeness flags`
        );
      } finally {
        device.destroy();
      }
    });
  }
}

it('exposes only the status destinations the nearest adapter writes', () => {
  expectTypeOf<
    keyof import('../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights').GPUNearestFeatureWeightsProps['output']['status']
  >().toEqualTypeOf<'overflow' | 'candidateOverflow'>();
  expectTypeOf<
    keyof NonNullable<GPUNearestFeatureWeights['outputWeights']['status']>
  >().toEqualTypeOf<'overflow' | 'candidateOverflow'>();
});

it('explicitly refuses the unsupported requiredCount destination', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    const status = {overflow: scalar('overflow'), requiredCount: scalar('requiredCount')};
    expect(
      () =>
        new GPUNearestFeatureWeights({
          neighborRows: scalar('neighborRows'),
          neighborCounts: scalar('neighborCounts'),
          slotCapacity: 1,
          output: {
            offsets: createTransientView(graph, 'offsets', 'uint32', 2),
            neighbors: scalar('neighbors'),
            weights: createTransientView(graph, 'weights', 'float32', 1),
            status
          }
        })
    ).toThrow(
      'nearest-feature-weights output.status.requiredCount is unsupported because this adapter publishes only overflow and candidateOverflow; omit requiredCount and use those completeness flags'
    );
  } finally {
    device.destroy();
  }
});

for (const missing of ['status', 'empty', 'candidate-only'] as const) {
  it(`refuses ${missing} missing required overflow at construction`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
      const output = {
        offsets: createTransientView(graph, 'offsets', 'uint32', 2),
        neighbors: scalar('neighbors'),
        weights: createTransientView(graph, 'weights', 'float32', 1),
        status: {overflow: scalar('overflow')}
      };
      // Model JavaScript callers, which can omit TypeScript's required fields.
      if (missing === 'status') Reflect.deleteProperty(output, 'status');
      else
        Reflect.set(
          output,
          'status',
          missing === 'empty' ? {} : {candidateOverflow: scalar('candidate')}
        );
      expect(
        () =>
          new GPUNearestFeatureWeights({
            id: 'required-status-oracle',
            slotCapacity: 1,
            neighborRows: scalar('inputRows'),
            neighborCounts: scalar('inputCounts'),
            output
          })
      ).toThrow('required-status-oracle output.status.overflow is required');
    } finally {
      device.destroy();
    }
  });
}

it('nearest weights ignore undeclared geometry metadata instead of advertising a status destination', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    const status = {overflow: scalar('overflow'), vertexCount: scalar('vertexCount')};
    const contributor = new GPUNearestFeatureWeights({
      neighborRows: scalar('rows'),
      neighborCounts: scalar('counts'),
      slotCapacity: 1,
      output: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 2),
        neighbors: scalar('neighbors'),
        weights: createTransientView(graph, 'weights', 'float32', 1),
        status
      }
    });
    expect(contributor.outputWeights.status).not.toHaveProperty('vertexCount');
    expect(contributor.outputWeights.status?.overflow).toBe(status.overflow);
  } finally {
    device.destroy();
  }
});
