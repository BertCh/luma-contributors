// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createCaptureValueInterpreter} from './capture-missing-values';
import {resolveCaptureCompanion, resolvePointCaptureWindow} from './capture-scope';
import {resolveGPUComputationCaptureGraph} from '@luma-contributors/analysis/computation-record';
import {getRecordOperator} from '../registry/record-operator';
import {clamp} from '@math.gl/core';
import {renderRasterImage} from '../encoding/raster-image';
import type {GPUComputationCapture} from '@luma-contributors/analysis/computation-record';

import {CARD_PALETTE_SOURCES} from '../encoding/card-tokens';
import {getSelectionInk} from '../encoding/selection-ink';
import {crispnessAttributes} from '../encoding/crispness';
import {provenanceMarkAttributes} from '../encoding/provenance';
import {rgbToHex} from '../scales/class-table';
import {NO_DATA_COLOR} from '../scales/hue-registry';
import {createQuantileScaleSpec, type ResolvedScale, ScaleRegistry} from '../scales/scale-registry';
import type {MarkProvenance, ProvisionalState, VariableKind, ViewGround} from '../types';
import {h, type ViewElementNode} from '../view-node';
import type {LensRecord, ViewContext} from '../registry/view-registration';
import {
  getAllCaptures,
  getCaptureFirstKey,
  getCaptureRowCount,
  getCaptureValue,
  getRecordedOperatorPackKey
} from './record-queries';

const RUN_SCALE_REGISTRIES = new WeakMap<LensRecord, ScaleRegistry>();
const AUTOMATIC_SCALE_REGISTRIES = new WeakMap<ScaleRegistry, Map<string, ScaleRegistry>>();
const SHARED_AUTOMATIC_SCOPES = new WeakMap<ScaleRegistry, Map<string, string>>();

/**
 * The run's scale registry: `context.scales` when the host supplies one, otherwise one registry
 * shared by every view of the same record object, so all views of a lens page agree (§E rule 1).
 */
export function getRunScaleRegistry(context: ViewContext): ScaleRegistry {
  if (context.scales) {
    return context.scales;
  }
  let registry = RUN_SCALE_REGISTRIES.get(context.record);
  if (!registry) {
    registry = new ScaleRegistry({runId: context.record.runId});
    RUN_SCALE_REGISTRIES.set(context.record, registry);
  }
  return registry;
}

/**
 * Final-state values of a captured variable: the capture with that name at the highest
 * iteration (or without iteration). Iteration domains come from the final state (§E rule 2).
 */
export function getFinalStateCapture(
  record: LensRecord,
  name: string
): GPUComputationCapture | undefined {
  let finalCapture: GPUComputationCapture | undefined;
  for (const capture of getAllCaptures(record)) {
    if (capture.name !== name) {
      continue;
    }
    if (!finalCapture || (capture.iteration ?? Infinity) >= (finalCapture.iteration ?? Infinity)) {
      finalCapture = capture;
    }
  }
  return finalCapture;
}

/** Declared variable kind of a capture from the domain packs, or undefined (never guessed). */
export function getDeclaredVariableKind(
  context: ViewContext,
  name: string
): VariableKind | undefined {
  const operator = getRecordOperator(context, name);
  const port = name.startsWith('result.') ? name.slice('result.'.length) : name;
  // Operator metadata wins even when another domain supplies a same-named fallback.
  if (operator) {
    const currentOperator = getRecordedOperatorPackKey(operator);
    for (const pack of context.packs ?? []) {
      const descriptor = pack.operators?.[operator] ?? pack.operators?.[currentOperator];
      const kinds = descriptor?.variableKinds;
      const kind = kinds?.[name] ?? kinds?.[port];
      if (kind) return kind;
    }
  }
  for (const pack of context.packs ?? []) {
    const kind = pack.variableKinds?.[name] ?? pack.variableKinds?.[port];
    if (kind) return kind;
  }
  return undefined;
}

/** Automatic point scales use only final captured keys in the published active prefix. */
function getAutomaticScaleValues(
  record: LensRecord,
  values: GPUComputationCapture
): ArrayLike<number> {
  let counted = false;
  const keys = new Set<number>();
  for (const points of getAllCaptures(record)) {
    const geometry = points.geometry;
    if (
      geometry?.kind !== 'points' ||
      geometry.countCapture === undefined ||
      (geometry.valuesCapture !== values.name && geometry.valuesCapture !== values.id) ||
      getAllCaptures(record).some(
        candidate =>
          candidate.name === points.name &&
          resolveGPUComputationCaptureGraph(record, candidate) ===
            resolveGPUComputationCaptureGraph(record, points) &&
          (candidate.iteration ?? Infinity) > (points.iteration ?? Infinity)
      )
    )
      continue;
    const pointGraph = resolveGPUComputationCaptureGraph(record, points);
    if (pointGraph && pointGraph !== resolveGPUComputationCaptureGraph(record, values)) continue;
    counted = true;
    const companion = resolveCaptureCompanion(record, points, geometry.valuesCapture);
    if (companion !== values) continue;
    const window = resolvePointCaptureWindow(record, points);
    if (!window) continue;
    const first = Math.max(window.first, getCaptureFirstKey(values));
    const end = Math.min(window.end, getCaptureFirstKey(values) + getCaptureRowCount(values));
    for (let key = first; key < end; key++) keys.add(key);
  }
  const automaticValues = counted
    ? [...keys].sort((first, second) => first - second).map(key => getCaptureValue(values, key)!)
    : values.values;
  return Array.from(automaticValues, createCaptureValueInterpreter(record, values).interpret);
}

/**
 * Resolves the one value scale of a captured variable for this run: an existing registration,
 * else a final-state quantile scale with lightness ramp and a stated beyond-range class. Returns
 * undefined when the final state has no finite values.
 *
 * When the measurement level is not declared the scale is ordered (lightness) and the legend says
 * so; hue is never chosen without a declared nominal level.
 */
export function resolveValueScale(
  context: ViewContext,
  name: string,
  capture?: GPUComputationCapture
): {scale: ResolvedScale; levelDeclared: boolean} | undefined {
  const registry = getRunScaleRegistry(context);
  const existing = registry.getByVariable(name);
  const declaredKind = getDeclaredVariableKind(context, name);
  const sharedScopes = SHARED_AUTOMATIC_SCOPES.get(registry);
  if (existing && !sharedScopes?.has(name)) {
    return {scale: registry.get(existing), levelDeclared: Boolean(declaredKind)};
  }
  const candidates = getAllCaptures(context.record).filter(candidate => candidate.name === name);
  const graphs = new Set(
    candidates.map(candidate => resolveGPUComputationCaptureGraph(context.record, candidate)?.id)
  );
  const graph = capture
    ? resolveGPUComputationCaptureGraph(context.record, capture)?.id
    : graphs.size === 1
      ? [...graphs][0]
      : undefined;
  if (!graph) return undefined;
  const scoped = candidates.filter(
    candidate => resolveGPUComputationCaptureGraph(context.record, candidate)?.id === graph
  );
  const latest = Math.max(...scoped.map(candidate => candidate.iteration ?? Infinity));
  const finalCaptures = scoped.filter(candidate => (candidate.iteration ?? Infinity) === latest);
  const finalCapture = finalCaptures.length === 1 ? finalCaptures[0] : undefined;
  const finalValues = finalCapture && getAutomaticScaleValues(context.record, finalCapture);
  if (!finalValues || !hasFiniteValue(finalValues)) {
    return undefined;
  }
  let automaticRegistries = AUTOMATIC_SCALE_REGISTRIES.get(registry);
  if (!automaticRegistries) {
    automaticRegistries = new Map();
    AUTOMATIC_SCALE_REGISTRIES.set(registry, automaticRegistries);
  }
  const scope = `${graph}:${context.ground}`;
  if (existing && sharedScopes?.get(name) === scope)
    return {scale: registry.get(existing), levelDeclared: Boolean(declaredKind)};
  let automaticRegistry = automaticRegistries.get(scope);
  if (!automaticRegistry) {
    automaticRegistry = new ScaleRegistry({runId: context.record.runId});
    automaticRegistries.set(scope, automaticRegistry);
  }
  const automatic = automaticRegistry.getByVariable(name);
  if (automatic)
    return {scale: automaticRegistry.get(automatic), levelDeclared: Boolean(declaredKind)};
  const spec = createQuantileScaleSpec({
    id: `${context.record.runId}:${graph}:${name}`,
    variable: name,
    variableKind: declaredKind ?? {level: 'interval'},
    ramp: CARD_PALETTE_SOURCES.seq[context.ground].ramp,
    noDataColor: rgbToHex(NO_DATA_COLOR[context.ground]),
    ground: context.ground,
    finalStateValues: finalValues,
    classCount: 5
  });
  // A globally unambiguous variable keeps the public run-registry registration. Only names
  // shared across graphs (or another ground) need the separate automatic scope registry.
  if (graphs.size === 1 && !existing) {
    const reference = registry.register(spec);
    const scopes = sharedScopes ?? new Map<string, string>();
    scopes.set(name, scope);
    SHARED_AUTOMATIC_SCOPES.set(registry, scopes);
    return {scale: registry.get(reference), levelDeclared: Boolean(declaredKind)};
  }
  return {
    scale: automaticRegistry.get(automaticRegistry.register(spec)),
    levelDeclared: Boolean(declaredKind)
  };
}

/** Props of {@link renderValueStrip}. */
export type ValueStripProps = {
  values: ArrayLike<number>;
  scale: ResolvedScale | undefined;
  provenance: MarkProvenance;
  idPrefix: string;
  ground: ViewGround;
  width?: number;
  height?: number;
  /** Largest number of drawn cells; longer buffers are drawn as means per bin. */
  maxCells?: number;
  /** Key of `values[0]` (for windows into a larger buffer). */
  keyOffset?: number;
  /** Identity domain of the represented absolute keys, when declared by the capture. */
  keySpace?: string;
  /** Pinned key, drawn with the selection core and halo. */
  pinnedKey?: number;
  /** Highlighted keys (selection image), drawn with an achromatic outline. */
  highlightedKeys?: ReadonlySet<number>;
  /** Convergence state, encoded by crispness only. */
  provisional?: ProvisionalState;
  /** Accessible label of the strip. */
  label: string;
};

/**
 * A 1-D index-space strip: one cell per key (or a mean per bin when the buffer is
 * longer than `maxCells`). Unaggregated small strips carry `data-action="pick" data-axis="entity" data-key`.
 * Value is lightness from the run's scale; lineage is texture; provisional state is blur.
 */
export function renderValueStrip(props: ValueStripProps): ViewElementNode {
  const width = props.width ?? 640;
  const height = props.height ?? 28;
  const keyOffset = props.keyOffset ?? 0;
  const count = props.values.length;
  const maxCells = clamp(props.maxCells ?? 160, 1, 160);
  const binSize = Math.max(1, Math.ceil(count / maxCells));
  const binCount = Math.min(count, maxCells);
  const cellWidth = binCount > 0 ? width / binCount : width;
  const ink = getSelectionInk(props.ground);
  const noData = props.scale?.noDataColor ?? rgbToHex(NO_DATA_COLOR[props.ground]);
  const colorFor = (value: number) =>
    props.scale && Number.isFinite(value) ? props.scale.colorFor(value) : noData;
  const cells: ViewElementNode[] = [];
  const outlines: ViewElementNode[] = [];
  for (let bin = 0; bin < binCount; bin++) {
    const first = Math.floor((bin * count) / binCount);
    const last = Math.floor(((bin + 1) * count) / binCount) - 1;
    const x = bin * cellWidth;
    const firstKey = keyOffset + first;
    const lastKey = keyOffset + last;
    const markAttributes = {
      'data-action': 'pick',
      'data-axis': 'entity',
      'data-key': firstKey,
      'data-key-space': props.keySpace,
      'data-key-end': binSize > 1 ? lastKey : undefined
    };
    if (binSize === 1 && count <= 32) {
      const value = Number(props.values[first]);
      cells.push(
        h(
          'rect',
          {
            x: round(x),
            y: 0,
            width: round(Math.max(cellWidth, 0.5)),
            height,
            ...markAttributes,
            ...provenanceMarkAttributes(props.provenance, colorFor(value), props.idPrefix)
          },
          h('title', null, `${firstKey}: ${formatCellValue(value)}`)
        )
      );
    }
    const highlighted = rangeIntersects(props.highlightedKeys, firstKey, lastKey);
    const pinned =
      props.pinnedKey !== undefined && props.pinnedKey >= firstKey && props.pinnedKey <= lastKey;
    if (pinned) {
      outlines.push(
        h('rect', {
          x: round(x - 1.5),
          y: -1.5,
          width: round(Math.max(cellWidth, 0.5) + 3),
          height: height + 3,
          fill: 'none',
          stroke: ink.halo,
          'stroke-width': 3,
          'data-state': 'selection-halo'
        }),
        h('rect', {
          x: round(x),
          y: 0,
          width: round(Math.max(cellWidth, 0.5)),
          height,
          fill: 'none',
          stroke: ink.core,
          'stroke-width': 1.5,
          'data-state': 'selection'
        })
      );
    } else if (highlighted) {
      outlines.push(
        h('rect', {
          x: round(x),
          y: 0,
          width: round(Math.max(cellWidth, 0.5)),
          height,
          fill: 'none',
          stroke: ink.core,
          'stroke-width': 1,
          'data-state': 'highlight'
        })
      );
    }
  }
  if (count > 32 || binSize > 1)
    cells.push(
      renderRasterImage({
        columns: count,
        rows: 1,
        limit: Math.min(160, maxCells),
        valueAt: index => Number(props.values[index]),
        categorical: props.scale?.variableKind.level === 'nominal',
        colorFor,
        attributes: {width, height, 'data-provenance': props.provenance}
      })
    );
  return h(
    'svg',
    {
      class: 'lens-value-strip',
      width,
      height: height + 4,
      viewBox: `0 -2 ${width} ${height + 4}`,
      role: 'img',
      'aria-label': props.label,
      'data-aggregate': binSize > 1 ? `bins of ${binSize}` : undefined
    },
    h('g', crispnessAttributes(props.provisional ?? 'final', props.idPrefix), cells),
    outlines
  );
}

function hasFiniteValue(values: ArrayLike<number>): boolean {
  for (let index = 0; index < values.length; index++) {
    if (Number.isFinite(Number(values[index]))) {
      return true;
    }
  }
  return false;
}

function rangeIntersects(
  keys: ReadonlySet<number> | undefined,
  firstKey: number,
  lastKey: number
): boolean {
  if (!keys || keys.size === 0) {
    return false;
  }
  if (lastKey - firstKey < keys.size) {
    for (let key = firstKey; key <= lastKey; key++) {
      if (keys.has(key)) {
        return true;
      }
    }
    return false;
  }
  for (const key of keys) {
    if (key >= firstKey && key <= lastKey) {
      return true;
    }
  }
  return false;
}

function formatCellValue(value: number): string {
  if (Number.isNaN(value)) {
    return 'NaN';
  }
  if (!Number.isFinite(value)) {
    return value > 0 ? '∞' : '−∞';
  }
  return Number.isInteger(value) ? String(value) : value.toPrecision(4);
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}
