// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUMapColoring, type GPUMapColoringPriority} from './gpu-map-coloring';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateRasterIterations} from '../../raster-analysis/cost-distance/raster-relaxation';

/** Packed, unchunked CSR graph values. Distances and weights metadata are unused and omitted. */
export type MapColoringOperationProps = {
  id?: string;
  weights: {
    offsets: string;
    neighbors: string;
    weights: string;
    /** Optional scalar loss inputs, preserved through both operation forms. */
    status?: {overflow?: string; candidateOverflow?: string};
  };
  /** Explicit row count, otherwise inferred by the contributor from offsets. */
  rowCount: number;
  /** uint32 words from `getGPUMapColoringParameterValues`. */
  parameters: string;
  maximumIterations?: number;
  priority?: GPUMapColoringPriority;
  output: {
    colors: string;
    colorCount?: string;
    conflictCount?: string;
    status?: {converged?: string; iterationCount?: string};
  };
};

/** @internal Capture configuration and port names once for both execution forms. */
export function snapshotMapColoringOperationProps(
  props: MapColoringOperationProps
): MapColoringOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'weights',
    'parameters',
    'rowCount',
    'maximumIterations',
    'priority',
    'output'
  ]);
  const output = snapshotConstructionFields(captured.output, [
    'colors',
    'colorCount',
    'conflictCount',
    'status'
  ]);
  const weights = snapshotConstructionFields(captured.weights, [
    'offsets',
    'neighbors',
    'weights',
    'status'
  ]);
  return Object.freeze({
    ...captured,
    weights: Object.freeze({
      ...weights,
      ...(weights.status
        ? {status: snapshotConstructionFields(weights.status, ['overflow', 'candidateOverflow'])}
        : {})
    }),
    output: Object.freeze({
      ...output,
      ...(output.status
        ? {status: snapshotConstructionFields(output.status, ['converged', 'iterationCount'])}
        : {})
    })
  });
}

/** Flatten captured loss names into logical operation inputs. @internal */
export function getMapColoringOperationInputs(
  props: MapColoringOperationProps
): Record<string, string> {
  const {weights} = props;
  return {
    offsets: weights.offsets,
    neighbors: weights.neighbors,
    weights: weights.weights,
    parameters: props.parameters,
    ...(weights.status?.overflow ? {overflow: weights.status.overflow} : {}),
    ...(weights.status?.candidateOverflow
      ? {candidateOverflow: weights.status.candidateOverflow}
      : {})
  };
}

/** Map coloring operation; only whole CSR input is supported by the mixed executor. */
export class GPUMapColoringOperation extends GPUContributorOperation {
  constructor(props: MapColoringOperationProps) {
    props = snapshotMapColoringOperationProps(props);
    const id = props.id ?? 'map-coloring';
    if (props.maximumIterations !== undefined)
      validateRasterIterations(id, 'maximumIterations', props.maximumIterations);
    validateRequirement(
      id,
      ['largest-degree-first', 'random'].includes(props.priority ?? 'largest-degree-first'),
      "priority must be 'largest-degree-first' or 'random'"
    );
    const {output} = props;
    super({
      id,
      inputs: getMapColoringOperationInputs(props),
      outputs: {
        colors: {value: output.colors, format: 'uint32', length: props.rowCount},
        ...(output.colorCount
          ? {colorCount: {value: output.colorCount, format: 'uint32' as const, length: 1}}
          : {}),
        ...(output.conflictCount
          ? {conflictCount: {value: output.conflictCount, format: 'uint32' as const, length: 1}}
          : {}),
        ...(output.status?.converged
          ? {converged: {value: output.status.converged, format: 'uint32' as const, length: 1}}
          : {}),
        ...(output.status?.iterationCount
          ? {
              iterationCount: {
                value: output.status.iterationCount,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUMapColoring({
          id,
          weights: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            status: {
              overflow: props.weights.status?.overflow ? getView<'uint32'>('overflow') : undefined,
              candidateOverflow: props.weights.status?.candidateOverflow
                ? getView<'uint32'>('candidateOverflow')
                : undefined
            }
          },
          parameters: getView<'uint32'>('parameters'),
          maximumIterations: props.maximumIterations,
          priority: props.priority,
          output: {
            colors: getView<'uint32'>('colors'),
            colorCount: output.colorCount ? getView<'uint32'>('colorCount') : undefined,
            conflictCount: output.conflictCount ? getView<'uint32'>('conflictCount') : undefined,
            status: {
              converged: output.status?.converged ? getView<'uint32'>('converged') : undefined,
              iterationCount: output.status?.iterationCount
                ? getView<'uint32'>('iterationCount')
                : undefined
            }
          }
        })
    });
  }
}
