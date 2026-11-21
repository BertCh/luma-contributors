// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUProbitRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression';
import {GPUSpatialPanelRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {getSpatialSURWeightValidationNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-sur-nodes';
import {GPUInstrumentalVariablesCovariance} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance';
import {GPULocalCollinearity} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-local-collinearity';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import spatialSURFixture from './seemingly-unrelated-regression-spatial-operation-fixtures.json';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

function expectFiniteValues(actual: number[], expected: readonly number[], precision = 4) {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((value, index) => {
    expect(Number.isFinite(value), `entry ${index}`).toBe(true);
    expect(value, `entry ${index}`).toBeCloseTo(expected[index], precision);
  });
}

function ring(rig: WeightsRig, rows: number) {
  const loss = rig.input(Uint32Array.of(0), 'uint32', 1);
  return {
    loss,
    weights: {
      ...rig.uploadWeights({
        offsets: Array.from({length: rows + 1}, (_, row) => row),
        neighbors: Array.from({length: rows}, (_, row) => (row + 1) % rows),
        weights: Array(rows).fill(1),
        distances: []
      }),
      status: {overflow: rig.input(Uint32Array.of(0), 'uint32', 1), candidateOverflow: loss}
    }
  };
}

it('SSR-125 whole-CSR local collinearity recovers orthogonal-neighbor diagnostics after candidate loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  let compiled: ReturnType<typeof rig.graph.compile> | undefined;
  try {
    const loss = rig.input(Uint32Array.of(0), 'uint32', 1);
    const weights = {
      ...rig.uploadWeights({
        offsets: [0, 4, 4, 4, 4, 4],
        neighbors: [1, 2, 3, 4],
        weights: [1, 1, 1, 1],
        distances: []
      }),
      status: {candidateOverflow: loss}
    };
    const factors = rig.output('float32', 10),
      condition = rig.output('float32', 5),
      status = rig.output('uint32', 5);
    rig.graph.add(
      new GPULocalCollinearity({
        weights,
        predictorCount: 2,
        predictors: rig.input(Float32Array.of(0, 0, -1, -1, -1, 1, 1, -1, 1, 1), 'float32', 10),
        output: {
          varianceInflationFactors: factors.view,
          conditionNumbers: condition.view,
          status: status.view
        }
      })
    );
    compiled = rig.graph.compile();
    for (const flag of [0, 1, 0]) {
      rig.bufferOf(loss).write(Uint32Array.of(flag));
      submitGraph(device, compiled, undefined);
      const factorValues = await readFloat32(factors.buffer, 10),
        conditionValues = await readFloat32(condition.buffer, 5),
        statusValues = await readUint32(status.buffer, 5);
      if (flag) {
        expect(factorValues.every(Number.isNaN)).toBe(true);
        expect(conditionValues.every(Number.isNaN)).toBe(true);
        expect(statusValues.every(value => value !== 0)).toBe(true);
      } else {
        // Focus 0 has four neighbors forming the corners (+/-1,+/-1): means zero,
        // variances one, covariance zero. Correlation=I, VIFs=1 and condition number=1.
        expect(statusValues[0]).toBe(0);
        expectFiniteValues(factorValues.slice(0, 2), [1, 1]);
        expectFiniteValues(conditionValues.slice(0, 1), [1]);
      }
    }
  } finally {
    compiled?.destroy();
    rig.destroy();
  }
});

it('SSR-125 probit diagnostics distinguish candidate-only loss and recover on the same graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  let compiled: ReturnType<typeof rig.graph.compile> | undefined;
  try {
    const {weights, loss} = ring(rig, 4);
    const tests = rig.output('float32', 6),
      diagnosticStatus = rig.output('uint32', 1),
      modelStatus = rig.output('uint32', 1);
    rig.graph.add(
      new GPUProbitRegression({
        predictors: rig.input(Float32Array.of(-1, -1, 1, 1), 'float32', 4),
        predictorCount: 1,
        response: rig.input(Float32Array.of(0, 1, 0, 1), 'float32', 4),
        spatialDiagnostics: {weights, output: {tests: tests.view, status: diagnosticStatus.view}},
        output: {
          coefficients: rig.output('float32', 2).view,
          standardErrors: rig.output('float32', 2).view,
          fitted: rig.output('float32', 4).view,
          marginalEffects: rig.output('float32', 1).view,
          summary: rig.output('float32', 4).view,
          status: modelStatus.view
        }
      })
    );
    compiled = rig.graph.compile();
    for (const flag of [0, 1, 0]) {
      rig.bufferOf(loss).write(Uint32Array.of(flag));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(modelStatus.buffer, 1)).toEqual([0]);
      expect(await readUint32(diagnosticStatus.buffer, 1)).toEqual([flag ? 2 : 0]);
      const values = await readFloat32(tests.buffer, 6);
      if (flag) expect(values.every(Number.isNaN)).toBe(true);
      else {
        // beta=0 and p=.5. Alternating residuals on a directed 4-cycle give trace=4,
        // standardizedProduct=-4, hence Pinkse=Slade=16/4=4 and KP z=-2.
        expectFiniteValues([values[0], values[2], values[4]], [4, -2, 4]);
      }
    }
  } finally {
    compiled?.destroy();
    rig.destroy();
  }
});

it('SSR-125 spatial SUR validation rejects candidate-only loss then clears its dependent gate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  let compiled: ReturnType<typeof rig.graph.compile> | undefined;
  try {
    const {weights, loss} = ring(rig, 4),
      invalid = rig.output('float32', 1);
    rig.graph.add(
      getSpatialSURWeightValidationNodes(rig.graph, {
        id: 'sur-r19',
        rowCount: 4,
        weights,
        output: invalid.view
      })
    );
    compiled = rig.graph.compile();
    for (const flag of [0, 1, 0]) {
      rig.bufferOf(loss).write(Uint32Array.of(flag));
      submitGraph(device, compiled, undefined);
      // Every row has exactly one positive unit edge to a distinct row; only the loss flag
      // can make this invalid. Spatial SUR uses this gate to publish spatial status 4.
      expectFiniteValues(await readFloat32(invalid.buffer, 1), [flag]);
    }
  } finally {
    compiled?.destroy();
    rig.destroy();
  }
});

it('SSR-125 spatial SUR publishes invalid-weights status and restores NumPy spatial parameters', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  let compiled: ReturnType<typeof rig.graph.compile> | undefined;
  try {
    const fixture = spatialSURFixture,
      loss = rig.input(Uint32Array.of(0), 'uint32', 1);
    const weights = {
      ...rig.uploadWeights({...fixture.weights, distances: []}),
      status: {overflow: rig.input(Uint32Array.of(0), 'uint32', 1), candidateOverflow: loss}
    };
    const parameters = rig.output('float32', 2),
      spatialStatus = rig.output('uint32', 2),
      status = rig.output('uint32', 1);
    rig.graph.add(
      new GPUSeeminglyUnrelatedRegression({
        equationCount: 2,
        predictorCount: 2,
        rowCount: fixture.rowCount,
        predictors: rig.input(
          Float32Array.from(fixture.predictors),
          'float32',
          fixture.predictors.length
        ),
        responses: rig.input(
          Float32Array.from(fixture.responses),
          'float32',
          fixture.responses.length
        ),
        spatial: {weights, output: {parameters: parameters.view, status: spatialStatus.view}},
        output: {
          coefficients: rig.output('float32', 6).view,
          covariance: rig.output('float32', 36).view,
          residualCovariance: rig.output('float32', 4).view,
          residuals: rig.output('float32', fixture.rowCount * 2).view,
          status: status.view
        }
      })
    );
    compiled = rig.graph.compile();
    for (const flag of [0, 1, 0]) {
      rig.bufferOf(loss).write(Uint32Array.of(flag));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(spatialStatus.buffer, 2)).toEqual(flag ? [4, 4] : [0, 0]);
      const values = await readFloat32(parameters.buffer, 2);
      if (flag) expect(values.every(Number.isNaN)).toBe(true);
      // Independent NumPy moment minimization, filtering and joint GLS are recorded in
      // seemingly-unrelated-regression-operation-fixtures.py:create_spatial_fixture.
      else {
        expect(await readUint32(status.buffer, 1)).toEqual([0]);
        expectFiniteValues(values, [0.5411583952369895, 0.5450196831334901], 3);
      }
    }
  } finally {
    compiled?.destroy();
    rig.destroy();
  }
});

// Independent NumPy derivation (float64, not a contributor/twin):
// W=kron(I3, roll(I4,1,axis=1)); H=[1,x,Wx];
// noise=[1,-1,2,-2,3,-1,1,-3,2,1,-1,0]; u=noise-H solve(H'H,H'noise);
// y=solve(I-.25W,2+3x+u). Orthogonality H'u=0 gives exact IV coefficients [2,3,.25].
const PANEL_PREDICTORS = [-2, -1, 1, 2, 0, 2, -2, 1, 3, 0, 2, -1];
const PANEL_RESPONSE = [
  -2.3965210638710928, -0.5590215492137447, 7.540811492913997, 6.556645311589983,
  5.8228693457581056, 7.523820617355854, -2.5278198408076102, 3.5045622209279754,
  13.941650165016501, 3.8933333333333335, 7.805676567656765, 0.8939933993399338
];

for (const effects of ['pooled', 'unit'] as const) {
  for (const lossKind of ['candidate', 'required-count'] as const) {
    it(`SSR-125 ${effects} ${lossKind} panel regression suppresses candidate loss and recovers known IV coefficients`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new WeightsRig(device);
      let compiled: ReturnType<typeof rig.graph.compile> | undefined;
      try {
        const {weights, loss} = ring(rig, 4),
          table = rig.output('float32', 12),
          status = rig.output('uint32', 1);
        rig.graph.add(
          new GPUSpatialPanelRegression({
            weights:
              lossKind === 'candidate' ? weights : {...weights, status: {requiredCount: loss}},
            periodCount: 3,
            predictorCount: 1,
            effects,
            predictors: rig.input(Float32Array.from(PANEL_PREDICTORS), 'float32', 12),
            response: rig.input(Float32Array.from(PANEL_RESPONSE), 'float32', 12),
            output: {
              table: table.view,
              summary: rig.output('float32', 16).view,
              status: {code: status.view}
            }
          })
        );
        compiled = rig.graph.compile();
        for (const flag of [0, 1, 0]) {
          rig.bufferOf(loss).write(Uint32Array.of(lossKind === 'candidate' ? flag : 4 + flag));
          submitGraph(device, compiled, undefined);
          const code = (await readUint32(status.buffer, 1))[0],
            values = await readFloat32(table.buffer, 12);
          if (flag) {
            expect(code).toBe(3);
            expect(values.every(Number.isNaN)).toBe(true);
          } else {
            expect(code).toBe(0);
            if (effects === 'pooled') expectFiniteValues([values[0]], [2]);
            else expect(Number.isNaN(values[0])).toBe(true);
            // Independent float64 IV solve on the uploaded f32 values:
            // H=[x,Wx], X=[x,Wy] after entity demeaning; solve(H'X,H'y).
            expectFiniteValues(
              [values[4], values[8]],
              effects === 'pooled' ? [3.00000002, 0.25000001] : [3.06600232, 0.25662286]
            );
          }
        }
      } finally {
        compiled?.destroy();
        rig.destroy();
      }
    });
  }
}

it('SSR-125 panel-error diagnostics recover independent pooled OLS tests after candidate loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  let compiled: ReturnType<typeof rig.graph.compile> | undefined;
  try {
    const {weights, loss} = ring(rig, 4),
      tests = rig.output('float32', 6),
      status = rig.output('uint32', 1);
    const contributor = new GPUSpatialPanelErrorMaximumLikelihood({
      weights,
      periodCount: 3,
      predictorCount: 1,
      predictors: rig.input(Float32Array.from(PANEL_PREDICTORS), 'float32', 12),
      response: rig.input(Float32Array.from(PANEL_RESPONSE), 'float32', 12),
      output: {
        table: rig.output('float32', 12).view,
        summary: rig.output('float32', 22).view,
        status: {code: rig.output('uint32', 1).view, converged: rig.output('uint32', 1).view}
      },
      diagnostics: {tests: tests.view, status: status.view}
    });
    rig.graph.add(contributor);
    compiled = rig.graph.compile();
    for (const flag of [0, 1, 0]) {
      rig.bufferOf(loss).write(Uint32Array.of(flag));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(status.buffer, 1)).toEqual([flag ? 1 : 0]);
      const values = await readFloat32(tests.buffer, 6);
      if (flag) expect(values.every(Number.isNaN)).toBe(true);
      // NumPy: e=y-X lstsq(X,y); s²=e'e/(12-2); t=tr(W'W+WW)=12;
      // fitted=X lstsq(X,y); q=(I-P_X)W*fitted; information=q'q/s²+t.
      // Evaluate pooled LM formulas with scipy chi2.sf (the projection uses fitted, not y).
      else
        expectFiniteValues(
          values,
          [
            1.1466961875008301, 0.28424182269679044, 4.083256552505403, 0.04331011061212494,
            3.2372744922823418, 0.07198028892768155
          ],
          3
        );
    }
  } finally {
    compiled?.destroy();
    rig.destroy();
  }
});

it('SSR-125 IV HAC covariance recovers an independently inverted information matrix', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  let compiled: ReturnType<typeof rig.graph.compile> | undefined;
  try {
    const loss = rig.input(Uint32Array.of(0), 'uint32', 1);
    const kernel = {
      ...rig.uploadWeights({
        offsets: [0, 1, 2, 3, 4],
        neighbors: [0, 1, 2, 3],
        weights: [1, 1, 1, 1],
        distances: []
      }),
      status: {overflow: rig.input(Uint32Array.of(0), 'uint32', 1), candidateOverflow: loss}
    };
    const design = rig.input(Float32Array.of(-1, 0, 1, 2), 'float32', 4),
      covariance = rig.output('float32', 4),
      status = rig.output('uint32', 1);
    rig.graph.add(
      new GPUInstrumentalVariablesCovariance({
        design,
        coefficientCount: 1,
        instruments: design,
        instrumentCount: 1,
        residuals: rig.input(Float32Array.of(1, -1, -1, 1), 'float32', 4),
        method: 'HAC',
        kernel,
        output: {covariance: covariance.view, status: {code: status.view}}
      })
    );
    compiled = rig.graph.compile();
    for (const flag of [0, 1, 0]) {
      rig.bufferOf(loss).write(Uint32Array.of(flag));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(status.buffer, 1)).toEqual([flag ? 4 : 0]);
      const values = await readFloat32(covariance.buffer, 4);
      if (flag) expect(values.every(Number.isNaN)).toBe(true);
      // K=I and u²=1 give inverse(X'X); mean(x)=.5, sum (x-.5)²=5.
      else expectFiniteValues(values, [0.3, -0.1, -0.1, 0.2]);
    }
  } finally {
    compiled?.destroy();
    rig.destroy();
  }
});
