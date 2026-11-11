// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUContributorOperationOutput} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUInequality, type GPUInequalityOutput} from './gpu-inequality';
import {GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH} from './inequality-parameters';

/** Packed graph values shared by both inequality operations; no oversized output tails. */
export type InequalityOperationProps = {
  id?: string;
  values: string;
  zoneIds: string;
  zoneCount: number;
  weights?: string;
  mask?: string;
  /** Layout written by getGPUInequalityParameterValues. */
  parameters: string;
  lorenzKnotCount?: number;
  output: Partial<Record<keyof GPUInequalityOutput, string>>;
};

/** Declares only requested outputs at their exact logical lengths. @internal */
export function getInequalityOperationOutputs(
  props: InequalityOperationProps
): Record<string, GPUContributorOperationOutput> {
  return Object.fromEntries(
    Object.entries(props.output)
      .filter(([, value]) => value !== undefined)
      .map(([name, value]) => [
        name,
        {
          value: value!,
          format: name === 'counts' ? 'uint32' : 'float32',
          length:
            name === 'globalSummary'
              ? GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH
              : name === 'lorenzKnots'
                ? props.zoneCount * (props.lorenzKnotCount ?? 11)
                : props.zoneCount
        }
      ])
  );
}

/** Operation form. Input and parameter contents are per-frame; other props are compile-time. */
export class GPUInequalityOperation extends GPUContributorOperation {
  constructor(props: InequalityOperationProps) {
    const id = props.id ?? 'inequality';
    super({
      id,
      inputs: {
        values: props.values,
        zoneIds: props.zoneIds,
        parameters: props.parameters,
        ...(props.weights ? {weights: props.weights} : {}),
        ...(props.mask ? {mask: props.mask} : {})
      },
      outputs: getInequalityOperationOutputs(props),
      createContributor: getView =>
        new GPUInequality({
          id,
          values: getView<'float32'>('values'),
          zoneIds: getView<'uint32'>('zoneIds'),
          parameters: getView<'float32'>('parameters'),
          weights: props.weights ? getView<'float32'>('weights') : undefined,
          mask: props.mask ? getView<'uint32'>('mask') : undefined,
          zoneCount: props.zoneCount,
          lorenzKnotCount: props.lorenzKnotCount,
          output: Object.fromEntries(
            Object.keys(getInequalityOperationOutputs(props)).map(name => [name, getView(name)])
          ) as GPUInequalityOutput
        })
    });
  }
}
