// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerDatasetReference} from '../recipes/analysis-recipe';

export type KeplerResidentStyleDomain =
  /** Continuous endpoints, quantile edges including endpoints, threshold boundaries, or ordered
   * point category codes. Optional selectionRevision records explicitly supplied selected controls;
   * it never requests domain computation or a source readback. */
  | {kind: 'fixed'; values: readonly number[]; selectionRevision?: number}
  /** Domain derives from all valid finite source rows, independent of display filtering.
   * Log domains follow native extent semantics: zero minimum becomes Float32(1e-5), with no
   * positive-only population filtering. Invalid zero/cross-sign extents are reported by the legend.
   */
  | {kind: 'source'}
  /** Only domain calculation consumes the borrowed selection mask; row output still spans the source. */
  | {kind: 'selection'; revision: number};

type StyleBase = {
  id: string;
  source: KeplerDatasetReference;
  column: string;
  domain: KeplerResidentStyleDomain;
  /** Native category order is explicit; source dictionary code order is never assumed to match it. */
  legend?: {labels: readonly string[]; codes?: readonly number[]};
};

export type KeplerResidentStylePlan = StyleBase &
  (
    | {
        kind: 'color';
        scale: 'linear' | 'sqrt' | 'log' | 'quantile' | 'equal-interval' | 'ordinal' | 'threshold';
        /** Packed RGBA8 colors. Ordinal palette entries are already remapped into source-code order. */
        palette: readonly number[];
        noDataColor?: number;
        interpolation?: 'linear' | 'step';
        outsideDomain: 'extrapolate' | 'clamp' | 'reject';
      }
    | {
        kind: 'scalar';
        scale: 'linear' | 'sqrt' | 'log' | 'identity' | 'point';
        range: readonly [number, number];
        nullValue?: number;
        /** Reverse the identity domain mapping, including GPU-derived endpoints. */
        invert?: boolean;
      }
  );

/** Detached semantic cache key; GPU handles and native layer instances never enter the plan. */
export function getKeplerResidentStyleKey(plan: KeplerResidentStylePlan): string {
  return JSON.stringify(plan);
}

/** Changes that require new graph/source topology and a fenced replacement of the old outputs. */
export function getKeplerResidentStyleTopologyKey(plan: KeplerResidentStylePlan): string {
  return JSON.stringify([
    plan.id,
    plan.source.datasetId,
    plan.source.revision,
    plan.column,
    plan.kind,
    plan.kind === 'color' && plan.scale === 'ordinal',
    plan.kind === 'scalar' && plan.scale === 'point',
    plan.domain.kind
  ]);
}

export function validateKeplerResidentStylePlan(plan: KeplerResidentStylePlan): void {
  if (
    !plan.id ||
    !plan.column ||
    !plan.source.datasetId ||
    !Number.isSafeInteger(plan.source.revision) ||
    plan.source.revision < 1
  )
    throw new Error('Invalid resident style reference');
  if (
    (plan.domain.kind === 'selection' &&
      (!Number.isSafeInteger(plan.domain.revision) || plan.domain.revision < 1)) ||
    (plan.domain.kind === 'fixed' &&
      plan.domain.selectionRevision !== undefined &&
      (!Number.isSafeInteger(plan.domain.selectionRevision) || plan.domain.selectionRevision < 1))
  )
    throw new Error('Invalid style selection revision');
  if (
    plan.domain.kind === 'fixed' &&
    plan.domain.values.some(value => !Number.isFinite(value) || Math.fround(value) !== value)
  )
    throw new Error('Style domains require explicit Float32 normalization');
  if (
    plan.legend &&
    (plan.legend.labels.some(label => typeof label !== 'string') ||
      (plan.legend.codes && plan.legend.codes.length !== plan.legend.labels.length))
  )
    throw new Error('Invalid style legend metadata');
  if (plan.kind === 'color') {
    if (
      !['linear', 'sqrt', 'log', 'quantile', 'equal-interval', 'ordinal', 'threshold'].includes(
        plan.scale
      ) ||
      !['extrapolate', 'clamp', 'reject'].includes(plan.outsideDomain)
    )
      throw new Error('Unsupported resident color scale');
    if (
      !plan.palette.length ||
      plan.palette.some(value => !Number.isInteger(value) || value < 0 || value > 0xffffffff)
    )
      throw new Error('Resident palettes require packed RGBA8 integers');
    if (
      plan.noDataColor !== undefined &&
      (!Number.isInteger(plan.noDataColor) || plan.noDataColor < 0 || plan.noDataColor > 0xffffffff)
    )
      throw new Error('Invalid resident null color');
    if (
      plan.scale === 'threshold' &&
      (plan.domain.kind !== 'fixed' || plan.palette.length !== plan.domain.values.length + 1)
    )
      throw new Error('Threshold styles require explicit fixed boundaries and one more color');
    if (plan.scale === 'ordinal' && plan.domain.kind !== 'fixed')
      throw new Error('Ordinal styles require an explicit native category order');
    if (
      plan.scale === 'log' &&
      plan.domain.kind === 'fixed' &&
      !plan.domain.values.every(value => value > 0) &&
      !plan.domain.values.every(value => value < 0)
    )
      throw new Error('Color log styles require an explicitly verified same-sign domain');
    if (plan.domain.kind === 'fixed' && plan.scale !== 'ordinal') {
      const values = plan.domain.values;
      const descending =
        (plan.scale === 'linear' || plan.scale === 'sqrt' || plan.scale === 'log') &&
        values[0] > values[values.length - 1];
      if (
        (plan.scale !== 'threshold' && values.length < 2) ||
        values.some(
          (value, index) =>
            index > 0 && (descending ? value > values[index - 1] : value < values[index - 1])
        )
      )
        throw new Error('Color domains require ordered stops or quantile edges');
      if (plan.scale === 'quantile' && plan.domain.values.length !== plan.palette.length + 1)
        throw new Error('Quantile domains require one more edge than palette colors');
    }
  } else {
    if (!['linear', 'sqrt', 'log', 'identity', 'point'].includes(plan.scale))
      throw new Error('Unsupported resident scalar scale');
    if (plan.invert && plan.scale !== 'identity')
      throw new Error('Scalar inversion requires an identity scale');
    if (
      plan.range.length !== 2 ||
      plan.range.some(value => !Number.isFinite(value) || Math.fround(value) !== value)
    )
      throw new Error('Scalar ranges require two explicitly normalized Float32 values');
    if (!Number.isFinite(Math.fround(plan.nullValue ?? 0)))
      throw new Error('Invalid scalar null value');
    if (plan.scale === 'point' && plan.domain.kind !== 'fixed')
      throw new Error('Point styles require an explicit native category order');
    if (
      plan.domain.kind === 'fixed' &&
      (plan.scale === 'point'
        ? new Set(plan.domain.values).size !== plan.domain.values.length
        : plan.domain.values.length !== 2)
    )
      throw new Error('Invalid scalar domain');
    if (
      plan.scale === 'log' &&
      plan.domain.kind === 'fixed' &&
      !plan.domain.values.every(value => value > 0) &&
      !plan.domain.values.every(value => value < 0)
    )
      throw new Error('Log styles require an explicitly verified same-sign domain');
  }
}
