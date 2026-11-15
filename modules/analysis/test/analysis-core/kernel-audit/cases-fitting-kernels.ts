// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUAdamStep,
  GPUFittingCoordinateEncoder,
  GPUFittingGradientSums,
  GPUFittingMinibatchOrder
} from '../../../src/analysis-core/index';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

/** Caller-owned persistent Adam state, every status present. */
export function createAdamAuditProps(
  {view}: KernelAuditContext,
  parameterCount = 100_000,
  maximumIterations = 10_000
) {
  return {
    gradients: view('float32', parameterCount),
    loss: view('float32', 1),
    parameters: view('float32', 6),
    maximumIterations,
    output: {
      weights: view('float32', parameterCount),
      firstMoment: view('float32', parameterCount),
      secondMoment: view('float32', parameterCount),
      lossHistory: view('float32', maximumIterations + 1),
      status: {
        iterationCount: view('uint32', 1),
        converged: view('uint32', 1),
        iterationLimitReached: view('uint32', 1),
        invalidCount: view('uint32', 1)
      }
    }
  };
}

/** Maximal compatible shapes, including default WebGPU binding/work limits. */
export const FITTING_KERNEL_AUDIT_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUFittingGradientSums',
    name: 'maximum parameter count, odd tiles and mask',
    build: context =>
      context.graph.add(
        new GPUFittingGradientSums({
          rowCount: 257,
          parameterCount: 100_000,
          gradients: context.view('float32', 257 * 100_000),
          mask: context.view('uint32', 257),
          output: {sums: context.view('float32', 100_000)}
        })
      )
  },
  {
    exportName: 'GPUFittingGradientSums',
    name: 'maximum rows and depth',
    build: context =>
      context.graph.add(
        new GPUFittingGradientSums({
          rowCount: 2 ** 24,
          parameterCount: 1,
          gradients: context.view('float32', 2 ** 24),
          output: {sums: context.view('float32', 1)}
        })
      )
  },
  {
    exportName: 'GPUAdamStep',
    name: 'maximum parameters and history with eight control bindings',
    build: context => context.graph.add(new GPUAdamStep(createAdamAuditProps(context)))
  },
  {
    exportName: 'GPUFittingMinibatchOrder',
    name: 'maximum rows and sort scratch',
    build: context =>
      context.graph.add(
        new GPUFittingMinibatchOrder({
          rowCount: 2 ** 24,
          parameters: context.view('uint32', 3),
          output: {order: context.view('uint32', 2 ** 24)}
        })
      )
  },
  {
    exportName: 'GPUFittingCoordinateEncoder',
    name: 'maximum Fourier width, three dimensions and mask',
    build: context =>
      context.graph.add(
        new GPUFittingCoordinateEncoder({
          encoding: 'fourier',
          dimensions: 3,
          rowCount: 65536,
          frequencyCount: 256,
          coordinates: context.view('float32', 65536 * 3),
          frequencies: context.view('float32', 256 * 3),
          mask: context.view('uint32', 65536),
          output: {
            features: context.view('float32', 65536 * 512),
            validity: context.view('uint32', 65536)
          }
        })
      )
  },
  {
    exportName: 'GPUFittingCoordinateEncoder',
    name: 'degree 39 and binding-limited rows',
    build: context =>
      context.graph.add(
        new GPUFittingCoordinateEncoder({
          encoding: 'spherical-harmonics',
          maximumDegree: 39,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
          rowCount: 20971,
          coordinates: context.view('float32', 20971 * 2),
          mask: context.view('uint32', 20971),
          output: {
            features: context.view('float32', 20971 * 1600),
            validity: context.view('uint32', 20971)
          }
        })
      )
  }
];
