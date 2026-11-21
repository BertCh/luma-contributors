// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogParameterSpec} from './catalog-binding-types';

export type CatalogNumericConstraints = Pick<
  CatalogParameterSpec,
  'minimum' | 'maximum' | 'nonzero' | 'nonzeroFloat32Scale'
>;

export function getCatalogValidationIdentity(id: string) {
  return {id, contributor: id, messagePrefix: `${id}:`};
}

export function getCatalogParameterRenameMessage(
  specs: readonly Pick<CatalogParameterSpec, 'name' | 'renamedFrom'>[],
  name: string
): string | undefined {
  const spec = specs.find(candidate => candidate.renamedFrom === name);
  return spec ? `${name} was removed (use ${spec.name})` : undefined;
}

const NUMBER_PATTERN = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

export function parseCatalogParameterNumber(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value !== 'string' || !NUMBER_PATTERN.test(value.trim())) return null;
  const parsed = Number(value.trim());
  return Number.isFinite(parsed) ? parsed : null;
}

/** Resolve only explicit spellings; defaults must be injected after this step. */
export function resolveCatalogParameterAliases<Value>(
  specs: readonly ({name: string; kind: string; aliasFor?: string} & CatalogNumericConstraints)[],
  raw: Readonly<Record<string, Value>>,
  reject: (name: string, problem: string) => never
): Record<string, Value> {
  const supplied = {...raw};
  const resolved = {...supplied};
  for (const spec of specs) {
    if (!spec.aliasFor) continue;
    const canonical = supplied[spec.aliasFor];
    const legacy = supplied[spec.name];
    const numeric = spec.kind === 'number' || spec.kind === 'integer';
    // Validate both original spellings before one can overwrite the other. Booleans and arrays
    // are not numeric text, even when Number() happens to coerce them to the same value.
    if (numeric) {
      for (const name of [spec.aliasFor, spec.name]) {
        if (supplied[name] === undefined) continue;
        const declaration = specs.find(candidate => candidate.name === name) ?? spec;
        const parsed = parseCatalogParameterNumber(supplied[name]);
        if (parsed === null)
          reject(name, `must be a finite decimal number; received ${String(supplied[name])}`);
        if (declaration.kind === 'integer' && !Number.isInteger(parsed))
          reject(name, `must be a whole number; received ${String(supplied[name])}`);
        const problem = getCatalogNumberProblem(declaration, parsed);
        if (problem) reject(name, problem);
      }
    }
    if (canonical !== undefined && legacy !== undefined) {
      if (
        numeric
          ? parseCatalogParameterNumber(canonical) !== parseCatalogParameterNumber(legacy)
          : canonical !== legacy
      )
        reject(
          spec.name,
          `must agree with ${spec.aliasFor}; received ${String(legacy)} and ${String(canonical)}; use ${spec.aliasFor}`
        );
    }
    const explicit = canonical !== undefined ? canonical : legacy;
    if (explicit !== undefined) {
      resolved[spec.name] = explicit;
      resolved[spec.aliasFor] = explicit;
    }
  }
  return resolved;
}

/** One numeric constraint description for catalog admission and both parameter forms. */
export function getCatalogNumberProblem(
  spec: CatalogNumericConstraints,
  value: number,
  received = 'received'
): string | null {
  if (!Number.isFinite(value)) return `must be finite; ${received} ${value}`;
  if (spec.nonzero && value === 0) return `must be nonzero; ${received} ${value}`;
  if (spec.nonzeroFloat32Scale !== undefined) {
    const encodedValue = Math.fround(value);
    if (!Number.isFinite(encodedValue) || encodedValue === 0)
      return `must remain finite and nonzero as Float32 before duration scaling; ${received} ${value}`;
    const encoded = Math.fround(value * spec.nonzeroFloat32Scale);
    if (!Number.isFinite(encoded) || encoded === 0)
      return `must give a finite nonzero Float32 value when multiplied by ${spec.nonzeroFloat32Scale}; ${received} ${value}`;
  }
  if (spec.minimum !== undefined && value < spec.minimum)
    return `must be at least ${spec.minimum}; ${received} ${value}`;
  if (spec.maximum !== undefined && value > spec.maximum)
    return `must be at most ${spec.maximum}; ${received} ${value}`;
  return null;
}

/** Explicit source units remain distinct from the metric frame used by geographic grids. */
export function resolveCatalogParameterUnit(
  spec: Pick<CatalogParameterSpec, 'unit' | 'coordinateModeUnits'>,
  coordinateSystem: unknown
): string | undefined {
  const unit = spec.coordinateModeUnits
    ? coordinateSystem === 'planar'
      ? spec.coordinateModeUnits.planar
      : spec.coordinateModeUnits.geographic
    : spec.unit;
  if (unit === 'coordinate-units') return 'coordinate units';
  if (unit === 'crs') return 'CRS units';
  return unit === 'source-coordinate-units' ? 'source coordinate units' : unit;
}
