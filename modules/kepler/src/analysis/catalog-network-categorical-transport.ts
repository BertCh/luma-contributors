// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUSinkhornDivergence,
  getGPUSinkhornDivergenceParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogOperationBinding} from './catalog-binding-types';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {
  makeInputSpec,
  makeOutputSpec,
  makeNumberParameter,
  makeStringParameter
} from './catalog-platform3-b-common';
import {
  getNumber,
  getString,
  makeBinding,
  makeResult,
  readNumbers
} from './catalog-network-categorical-common';

export const TRANSPORT_BINDING: CatalogOperationBinding = makeBinding({
  id: 'sinkhorn-divergence',
  title: 'Sinkhorn divergence',
  domain: 'analysis',
  summary: 'Compare two nonnegative mass fields on one explicit row-major planar lattice.',
  contributorNames: ['GPUSinkhornDivergence'],
  inputs: [makeInputSpec('cells', 'Mass grid cells (row-major)', ['point'], 16384, 16384)],
  parameters: [
    makeStringParameter('sourceProperty', 'Source mass property', 'source'),
    makeStringParameter('targetProperty', 'Target mass property', 'target'),
    makeNumberParameter('width', 'Grid width', 2, 1, 128, undefined, 'integer'),
    makeNumberParameter('height', 'Grid height', 2, 1, 128, undefined, 'integer'),
    makeNumberParameter('cellWidth', 'Cell width', 1, 1e-6, 1e6, 'planar'),
    makeNumberParameter('cellHeight', 'Cell height', 1, 1e-6, 1e6, 'planar'),
    makeNumberParameter('epsilon', 'Entropic regularization', 1, 1e-6, 1e12, 'squared planar'),
    makeNumberParameter('tolerance', 'Marginal error tolerance', 1e-4, 1e-8, 1),
    makeNumberParameter(
      'maximumIterations',
      'Maximum iterations',
      100,
      1,
      1000,
      undefined,
      'integer'
    )
  ],
  outputs: [
    makeOutputSpec(
      'sinkhorn-divergence',
      'Sinkhorn divergence report',
      'none',
      ['comparison_id'],
      1
    )
  ],
  declaredBehavior: [
    'Grid width * height must equal source rows, in row-major order. Geometry is not used to infer a lattice: the explicit cell width/height define squared planar transport cost.',
    'Both mass fields must be finite and nonnegative and have positive total mass. Each is normalized to one, so total mass differences are not compared.',
    'Full row/column blur windows are used (no truncation approximation). The result includes divergence, three entropic OT components, marginal error, convergence, iteration limit, iteration count, invalid count and approximation status.',
    'An exhausted iteration budget publishes the current estimate with iterationLimitReached=1. Tiny negative divergence can occur from Float32 rounding.'
  ],
  limitation:
    'Balanced planar transport only; no geographic distance, unbalanced transport or transport plan. Dimensions at most 128 by 128; epsilon should be at least about 1e-3 of the squared grid diagonal.',
  runGpu: async (context, inputs, values) => {
    const width = getNumber(values, 'width');
    const height = getNumber(values, 'height');
    if (width * height !== inputs['cells'].features.length)
      throw new Error('Grid width * height must equal the mass row count');
    const source = readNumbers(inputs['cells'], getString(values, 'sourceProperty'), true);
    const target = readNumbers(inputs['cells'], getString(values, 'targetProperty'), true);
    if (!source.some(value => value > 0) || !target.some(value => value > 0))
      throw new Error('Each mass field must have positive total mass');
    return withCatalogGpuSession(context.device, 'catalog-sinkhorn', async session => {
      const divergence = session.output('divergence', 'float32', 1);
      const components = session.output('components', 'float32', 3);
      const error = session.output('marginal-error', 'float32', 1);
      const statusNames = [
        'converged',
        'iterationLimitReached',
        'iterationCount',
        'invalidCount',
        'approximation'
      ] as const;
      const statuses = statusNames.map(name => session.output(name, 'uint32', 1));
      session.graph.add(
        new GPUSinkhornDivergence({
          grid: {width, height, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'point'},
          source: session.input('source', source, 'float32'),
          target: session.input('target', target, 'float32'),
          parameters: session.input(
            'parameters',
            getGPUSinkhornDivergenceParameterValues({
              cellWidth: getNumber(values, 'cellWidth'),
              cellHeight: getNumber(values, 'cellHeight'),
              epsilon: getNumber(values, 'epsilon'),
              tolerance: getNumber(values, 'tolerance')
            }),
            'float32'
          ),
          maximumIterations: getNumber(values, 'maximumIterations'),
          blurRadius: Math.max(1, width - 1, height - 1),
          output: {
            divergence: divergence.view,
            components: components.view,
            marginalError: error.view,
            status: Object.fromEntries(
              statusNames.map((name, index) => [name, statuses[index].view])
            )
          }
        })
      );
      session.run();
      const costs = await session.readFloat32(components);
      const statusValues = await Promise.all(statuses.map(status => session.readScalar(status)));
      return makeResult(
        'sinkhorn-divergence',
        'Sinkhorn divergence report',
        {
          comparison_id: 'string',
          divergence: 'float64',
          cross_cost: 'float64',
          source_self_cost: 'float64',
          target_self_cost: 'float64',
          marginal_error: 'float64',
          ...Object.fromEntries(statusNames.map(name => [name, 'int32' as const]))
        },
        [
          {
            comparison_id: 'source-target',
            divergence: (await session.readFloat32(divergence))[0],
            cross_cost: costs[0],
            source_self_cost: costs[1],
            target_self_cost: costs[2],
            marginal_error: (await session.readFloat32(error))[0],
            ...Object.fromEntries(statusNames.map((name, index) => [name, statusValues[index]]))
          }
        ]
      );
    });
  }
});
