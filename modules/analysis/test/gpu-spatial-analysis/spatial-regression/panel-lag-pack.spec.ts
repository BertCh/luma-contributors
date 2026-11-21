// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {
  createPanelLagPackInput,
  expectPanelLagPackFit,
  createPanelLagMomentInput,
  createPanelLagMomentProbe,
  expectPanelLagMoments,
  getPanelLagMomentRecord
} from './panel-lag-pack.test-utils';

it('native tiled moments retain small within squares against the integer oracle', async () => {
  const context = await createRegressionTestContext();
  try {
    recordWGSLKernelNodes(context.graph);
    const input = createPanelLagMomentInput('unit');
    const record = getPanelLagMomentRecord(context.graph, 'unit', true);
    const output = context.output('float32', 4);
    const node = createWGSLKernelNode(context.graph, {
      ...record,
      id: 'native-tiled-moment-probe',
      workgroupSize: record.workgroupSize[0],
      bindings: record.bindings.map(binding => {
        if (binding.name === 'partial') return {...binding, view: output.view};
        const column = binding.name === 'predictors' ? 1 : binding.name === 'response' ? 2 : 3;
        const values =
          binding.name === 'validity'
            ? new Float32Array(4)
            : Float32Array.from({length: 32}, (_, row) => input.data[row * 8 + column]);
        return {...binding, view: context.input('float32', values)};
      })
    });
    context.graph.add(node);
    context.run();
    const values = await readFloat32(output.buffer, 3);
    expectPanelLagMoments(values, 'unit', true);
  } finally {
    context.destroy();
  }
});

for (const effects of ['unit', 'random'] as const)
  it(`native moments retain small within squares against the integer oracle: ${effects}`, async () => {
    const context = await createRegressionTestContext();
    try {
      recordWGSLKernelNodes(context.graph);
      const input = createPanelLagMomentInput(effects);
      const output = context.output('float32', effects === 'unit' ? 9 : 24);
      const data = context.input('float32', input.data);
      context.graph.add(
        createPanelLagMomentProbe(
          context.graph,
          getPanelLagMomentRecord(context.graph, effects),
          output.view,
          effects,
          data
        )
      );
      context.run();
      expectPanelLagMoments(await readFloat32(output.buffer, output.view.length), effects);
    } finally {
      context.destroy();
    }
  });

for (const scale of [1, 2])
  it(`native unit fit matches the original-input SVD oracle with large effects, scale=${scale}`, async () => {
    const input = createPanelLagPackInput(scale);
    const context = await createRegressionTestContext();
    try {
      const output = {
        coefficients: context.output('float32', 4),
        covariance: context.output('float32', 9),
        summary: context.output('float32', 6),
        status: context.output('uint32', 1),
        fitted: context.output('float32', 8),
        residuals: context.output('float32', 8)
      };
      context.graph.add(
        new GPUSpatialPanelLagMaximumLikelihood({
          effects: 'unit',
          periodCount: 2,
          predictorCount: 1,
          predictors: context.input('float32', input.predictors),
          response: context.input('float32', input.response),
          weights: {
            offsets: context.input('uint32', input.weights.offsets),
            neighbors: context.input('uint32', input.weights.neighbors),
            weights: context.input('float32', input.weights.weights)
          },
          output: {
            coefficients: output.coefficients.view,
            covariance: output.covariance.view,
            summary: output.summary.view,
            status: output.status.view,
            fitted: output.fitted.view,
            residuals: output.residuals.view
          }
        })
      );
      context.run();
      expectPanelLagPackFit(
        {
          coefficients: await readFloat32(output.coefficients.buffer, 4),
          covariance: await readFloat32(output.covariance.buffer, 9),
          summary: await readFloat32(output.summary.buffer, 6),
          status: await readUint32(output.status.buffer, 1),
          fitted: await readFloat32(output.fitted.buffer, 8),
          residuals: await readFloat32(output.residuals.buffer, 8)
        },
        scale
      );
    } finally {
      context.destroy();
    }
  });

for (const effects of ['unit', 'random'] as const)
  for (const scale of [1, 2])
    it(`native pack retains exact within-lag half steps: ${effects}, scale=${scale}`, async () => {
      const input = createPanelLagPackInput(scale);
      const context = await createRegressionTestContext();
      try {
        recordWGSLKernelNodes(context.graph);
        const contributor = new GPUSpatialPanelLagMaximumLikelihood({
          effects,
          periodCount: 2,
          predictorCount: 1,
          predictors: context.input('float32', input.predictors),
          response: context.input('float32', input.response),
          weights: {
            offsets: context.input('uint32', input.weights.offsets),
            neighbors: context.input('uint32', input.weights.neighbors),
            weights: context.input('float32', input.weights.weights)
          },
          output: {
            coefficients: context.output('float32', 4).view,
            covariance: context.output('float32', 9).view,
            summary: context.output('float32', 6).view,
            status: context.output('uint32', 1).view,
            fitted: context.output('float32', 8).view,
            residuals: context.output('float32', 8).view
          }
        });
        const nodes = contributor
          .getCommandNodes(context.graph)
          .filter(
            node =>
              node.id.includes('-within-pack-') ||
              ['-pooled-means', '-dense-weights', '-pack'].some(suffix => node.id.endsWith(suffix))
          );
        const record = getWGSLKernelNodeRecord(
          context.graph,
          nodes.find(node => node.id.endsWith('-pack'))!.id
        )!;
        const data = record.bindings.find(binding => binding.name === 'data')!.view;
        context.graph.add(nodes);
        const output = context.output('float32', 16);
        context.graph.add(
          createWGSLKernelNode(context.graph, {
            id: 'read-native-within-lag',
            operation: 'PanelPackOracle',
            invocationCount: 8,
            bindings: [
              {name: 'data', view: data, type: 'f32', access: 'read'},
              {name: 'output', view: output.view, type: 'f32', access: 'read_write'}
            ],
            body: 'output[outputOffset+2u*index]=data[dataOffset+8u*index+2u];output[outputOffset+2u*index+1u]=data[dataOffset+8u*index+3u];'
          })
        );
        context.run();
        const actual = await readFloat32(output.buffer, 16);
        // All values are dyadic and exactly representable, so no float32 tolerance is needed.
        for (let row = 0; row < 8; row++) {
          expect(Number.isFinite(actual[2 * row + 1])).toBe(true);
          expect(actual[2 * row]).toBe(input.withinResponse[row]);
          expect(actual[2 * row + 1]).toBe(input.withinLag[row]);
        }
      } finally {
        context.destroy();
      }
    });
