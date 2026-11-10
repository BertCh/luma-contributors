// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The spatial panel models share one run: a small tract map observed over several periods (period
// major), a response with a spatial process and entity effects, the operation added by the caller,
// a residual map of the first period and the coefficient bars with the true values.

import type {GPUSpatialWeights} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  defineOperationRun,
  type ParameterValues,
  type RunBuffer,
  type RunBuilder
} from '../../../define-run';
import {
  createCoefficientPanel,
  createRegressionInputs,
  createResponse,
  drawCoefficientPanel,
  drawResidualMap,
  TRUE_COEFFICIENTS,
  uploadRegressionMap,
  uploadStandardizedWeights,
  type RegressionInputs
} from './regression';
import {addKernelPass} from './kernel';
import {bindAxes, getFractionOf} from '../../../interaction';
import type {GpuSceneParameter} from '../../../../types';

export const PANEL_PERIOD_COUNT = 6;

export type PanelContext = {
  builder: RunBuilder;
  values: ParameterValues;
  weights: GPUSpatialWeights;
  predictors: RunBuffer<'float32'>;
  response: RunBuffer<'float32'>;
  entityCount: number;
  periodCount: number;
  residuals: RunBuffer<'float32'>;
};

/** Where the coefficients of the operation live and what the true values are. */
export type PanelCoefficients = {
  /** The buffer the operation wrote: a four-column table or a plain coefficient vector. */
  source: RunBuffer<'float32'>;
  /** Floats per coefficient row in `source`. */
  columnCount: number;
  /** True values from the spatial strength slider. */
  truthOf(strength: number): number[];
};

export type PanelSpec = {
  parameters: readonly GpuSceneParameter[];
  caption: string;
  process: 'lag' | 'error';
  rebuildOn?: readonly string[];
  /** Parameters must include `period`: the period whose residuals the map shows. */
  /** Adds the real operation and returns where its coefficients are. */
  addOperation(context: PanelContext): PanelCoefficients;
};

/** Predictors of every period: the entity values with a period-specific shift. */
function createPanelPredictors(inputs: RegressionInputs, periodCount: number): Float32Array {
  const entityCount = inputs.tracts.count;
  const predictors = new Float32Array(periodCount * entityCount * 2);
  for (let period = 0; period < periodCount; period++) {
    for (let entity = 0; entity < entityCount; entity++) {
      const shift = 0.35 * inputs.normals[(entity * 5 + period * 11) % entityCount];
      const row = (period * entityCount + entity) * 2;
      predictors[row] = inputs.predictors[entity * 2] + shift;
      // A different draw for the second predictor, so the within-entity variation is not collinear.
      const otherShift = 0.35 * inputs.normals[(entity * 3 + period * 7 + 1) % entityCount];
      predictors[row + 1] = inputs.predictors[entity * 2 + 1] + otherShift;
    }
  }
  return predictors;
}

type PanelInputs = {inputs: RegressionInputs; panelPredictors: Float32Array};
type PanelOutputs = {
  mesh: ReturnType<typeof uploadRegressionMap>;
  residuals: RunBuffer<'float32'>;
  panel: ReturnType<typeof createCoefficientPanel>;
};

/** Defines a run for a spatial panel model. */
export function definePanelRun(spec: PanelSpec) {
  return defineOperationRun({
    parameters: spec.parameters,
    caption: spec.caption,
    rebuildOn: spec.rebuildOn,
    // The pointer steps through the periods: the map shows the residuals of that period.
    interaction: bindAxes<PanelInputs, PanelOutputs>(spec.parameters, {
      kind: 'time',
      hint: 'step through periods',
      x: 'period',
      decorate: (values, _pointer, {outputs, peek, world}) => {
        const period = spec.parameters.find(parameter => parameter.id === 'period');
        const estimates = peek(outputs.panel.estimates);
        const ticks = Array.from({length: PANEL_PERIOD_COUNT}, (_, index) => {
          const x = period ? getFractionOf(period, index + 1) * world.width : 0;
          const isCurrent = index + 1 === values.period;
          return {
            shape: 'line' as const,
            from: [x, 0.02 * world.height] as const,
            to: [x, (isCurrent ? 0.08 : 0.045) * world.height] as const
          };
        });
        const label = spec.process === 'lag' ? 'rho' : 'lambda';
        return {
          glyphs: ticks,
          readout: `period ${values.period} of ${PANEL_PERIOD_COUNT} · ${label} ${estimates ? estimates[3].toFixed(2) : '-'} (true ${values.strength})`
        };
      }
    }),
    createInputs: ({random, world}) => {
      const inputs = createRegressionInputs(random, world, 5);
      return {inputs, panelPredictors: createPanelPredictors(inputs, PANEL_PERIOD_COUNT)};
    },
    build(builder, {inputs, panelPredictors}, values) {
      const entityCount = inputs.tracts.count;
      const rowCount = entityCount * PANEL_PERIOD_COUNT;
      const weights = uploadStandardizedWeights(builder, inputs.weights);
      const predictors = builder.upload('predictors', 'float32', panelPredictors);
      const response = builder.allocate('response', 'float32', rowCount);
      const residuals = builder.allocate('residuals', 'float32', rowCount);
      const coefficients = spec.addOperation({
        builder,
        values,
        weights: weights.input,
        predictors,
        response,
        entityCount,
        periodCount: PANEL_PERIOD_COUNT,
        residuals
      });
      // The residuals of the period shown, copied out of the period-major buffer the model wrote.
      const periodIndex = builder.allocate('period-index', 'uint32', 1);
      const shown = builder.allocate('period-residuals', 'float32', entityCount);
      addKernelPass(builder, {
        id: 'period-residuals',
        invocationCount: entityCount,
        bindings: [
          {name: 'residuals', buffer: residuals, access: 'read'},
          {name: 'periodIndex', buffer: periodIndex, access: 'read'},
          {name: 'shown', buffer: shown, access: 'read_write'}
        ],
        declarations: `const ENTITY_COUNT: u32 = ${entityCount}u;`,
        body: 'shown[shownOffset + index] = residuals[residualsOffset + periodIndex[periodIndexOffset] * ENTITY_COUNT + index];'
      });
      const panel = createCoefficientPanel(
        builder,
        coefficients.source,
        coefficients.columnCount,
        coefficients.truthOf(0)
      );
      return {
        outputs: {mesh: uploadRegressionMap(builder, inputs), residuals: shown, panel},
        update: current => {
          periodIndex.write(
            Uint32Array.of(Math.max(0, Math.min(PANEL_PERIOD_COUNT, current.period) - 1))
          );
          const strength = current.strength;
          const effects = current.unitEffects ?? 0;
          const entityEffects = Float32Array.from(
            {length: entityCount},
            (_, entity) => effects * (inputs.tracts.values[entity] - 0.5) * 3
          );
          const all = new Float32Array(rowCount);
          for (let period = 0; period < PANEL_PERIOD_COUNT; period++) {
            const periodInputs: RegressionInputs = {
              ...inputs,
              predictors: panelPredictors.subarray(
                period * entityCount * 2,
                (period + 1) * entityCount * 2
              ),
              normals: inputs.normals.map(
                (_, entity) => inputs.normals[(entity + period * 17) % entityCount]
              )
            };
            const periodResponse = createResponse(periodInputs, {
              noise: current.noise ?? 0.4,
              ...(spec.process === 'lag' ? {lagRho: strength} : {errorLambda: strength}),
              coefficients: entity => {
                const effect = entityEffects[entity];
                return [TRUE_COEFFICIENTS[0] + effect, TRUE_COEFFICIENTS[1], TRUE_COEFFICIENTS[2]];
              }
            });
            all.set(periodResponse, period * entityCount);
          }
          response.write(all);
          panel.truth.write(Float32Array.from(coefficients.truthOf(strength)));
        }
      };
    },
    draw: ({inputs, outputs}) => [
      ...drawResidualMap(inputs.inputs, outputs.mesh, outputs.residuals),
      ...drawCoefficientPanel(outputs.panel)
    ]
  });
}
