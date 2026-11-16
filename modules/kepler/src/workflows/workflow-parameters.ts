// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import {WORKFLOW_LIMITS, WorkflowError, type WorkflowParameterSpec} from './workflow-types';

const NUMBER_PATTERN = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

function parseNumber(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value !== 'string' || !NUMBER_PATTERN.test(value.trim())) return null;
  const number = Number(value.trim());
  return Number.isFinite(number) ? number : null;
}

function formatProblemValue(value: unknown): string {
  if (typeof value === 'string') {
    return JSON.stringify(value.length > 40 ? `${value.slice(0, 39)}…` : value);
  }
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'an array';
  if (typeof value === 'object') return 'an object';
  return String(value);
}

function getNumberRangeProblem(spec: WorkflowParameterSpec, value: number): string | null {
  if (
    (spec.minimum === undefined || value >= spec.minimum) &&
    (spec.maximum === undefined || value <= spec.maximum)
  ) {
    return null;
  }
  if (spec.minimum !== undefined && spec.maximum !== undefined) {
    return `must be between ${spec.minimum} and ${spec.maximum}; got ${value}`;
  }
  if (spec.minimum !== undefined) return `must be at least ${spec.minimum}; got ${value}`;
  return `must be at most ${spec.maximum}; got ${value}`;
}

function countNumberListEntries(value: string): number {
  if (value.trim() === '') return 0;
  let count = 1;
  for (let index = 0; index < value.length; index++) {
    if (value[index] === ',') count++;
  }
  return count;
}

function getNumberListEntries(value: string | readonly unknown[]): readonly unknown[] {
  if (typeof value !== 'string') return value;
  if (value.trim() === '') return [];
  return value.split(',').map(entry => parseNumber(entry) ?? entry.trim());
}

function getMaximumStringLength(spec: WorkflowParameterSpec): number {
  if (spec.maximumLength !== undefined) return spec.maximumLength;
  switch (spec.kind) {
    case 'column-list':
      return 256;
    case 'expression':
      return 2000;
    case 'text':
      return WORKFLOW_LIMITS.maximumTextCharacters;
    default:
      return 64;
  }
}

/** Static defaults by name; input-derived parameters remain absent until resolved. */
export function getWorkflowParameterDefaults(
  specs: readonly WorkflowParameterSpec[]
): Record<string, CatalogParameterValue> {
  return Object.fromEntries(
    specs.flatMap(spec => {
      const value = spec.defaultValue;
      return value === undefined ? [] : [[spec.name, Array.isArray(value) ? [...value] : value]];
    })
  );
}

/**
 * Why a raw value is not acceptable for a parameter, or null when it is. The sentence names the
 * limit and the value, without the parameter label: `must be between 0 and 100; got 250`,
 * `must be a whole number; got 1.5`, `must be one of knn, distance; got "x"`.
 */
export function getWorkflowParameterProblem(
  spec: WorkflowParameterSpec,
  rawValue: unknown
): string | null {
  switch (spec.kind) {
    case 'number':
    case 'integer': {
      const value = parseNumber(rawValue);
      if (value === null) return `must be a number; got ${formatProblemValue(rawValue)}`;
      if (spec.kind === 'integer' && !Number.isInteger(value)) {
        return `must be a whole number; got ${value}`;
      }
      return getNumberRangeProblem(spec, value);
    }
    case 'boolean':
      return typeof rawValue === 'boolean'
        ? null
        : `must be a boolean; got ${formatProblemValue(rawValue)}`;
    case 'enum': {
      if (typeof rawValue === 'string' && spec.options?.includes(rawValue)) return null;
      const options = spec.options ?? [];
      const choices = [...options.slice(0, 8), ...(options.length > 8 ? ['…'] : [])];
      return `must be one of ${choices.join(', ')}; got ${formatProblemValue(rawValue)}`;
    }
    case 'number-list': {
      if (typeof rawValue !== 'string' && !Array.isArray(rawValue)) {
        return `must be a list of numbers; got ${formatProblemValue(rawValue)}`;
      }
      const maximumLength = spec.maximumLength ?? 8;
      const entryCount =
        typeof rawValue === 'string' ? countNumberListEntries(rawValue) : rawValue.length;
      if (entryCount < 1 || entryCount > maximumLength) {
        return `must have 1 to ${maximumLength} numbers; got ${entryCount}`;
      }
      const entries = getNumberListEntries(rawValue);
      for (let index = 0; index < entries.length; index++) {
        const entry = entries[index];
        const problem =
          typeof entry !== 'number' || !Number.isFinite(entry)
            ? `must be a number; got ${formatProblemValue(entry)}`
            : getNumberRangeProblem(spec, entry);
        if (problem) return `entry ${index + 1} ${problem}`;
      }
      return null;
    }
    default: {
      if (typeof rawValue !== 'string') {
        return `must be a string; got ${formatProblemValue(rawValue)}`;
      }
      const maximumLength = getMaximumStringLength(spec);
      if (rawValue.length > maximumLength) {
        return `must be at most ${maximumLength} characters; got ${rawValue.length}`;
      }
      const length =
        spec.kind === 'column-list'
          ? parseWorkflowColumnList(rawValue).join(', ').length
          : rawValue.length;
      return length > maximumLength
        ? `must be at most ${maximumLength} characters; got ${length}`
        : null;
    }
  }
}

/**
 * Coerces and range-checks raw values (form text, saved JSON) against parameter specs of any
 * kind; absent names take any static default and unknown names are rejected. Parameters with
 * no static default remain absent for AX or the binding to derive from its input. Throws
 * `WorkflowError('parameter-out-of-range', message, {parameter})` (or `unknown-parameter`) where
 * the message is `${operationTitle}: ${label}${unit} ${problem}.`
 *
 * Coercion: numbers and integers accept a numeric string; `number-list` accepts a
 * comma-separated string or an array; `column-list` is kept as one comma-separated string with
 * names trimmed; `string`, `text`, `column`, `dataset` and `expression` must be strings within
 * `maximumLength` (default 64 for `string`, `column` and `dataset`, 256 for `column-list`, 2000
 * for `expression`, `WORKFLOW_LIMITS.maximumTextCharacters` for `text`).
 */
export function normalizeWorkflowParameters(
  specs: readonly WorkflowParameterSpec[],
  raw: Readonly<Record<string, unknown>>,
  context: {operationTitle: string}
): Record<string, CatalogParameterValue> {
  const parameterNames = new Set(specs.map(spec => spec.name));
  for (const name of Object.keys(raw)) {
    if (!parameterNames.has(name)) {
      throw new WorkflowError(
        'unknown-parameter',
        `${context.operationTitle}: Unknown parameter ${formatProblemValue(name)}.`,
        {parameter: name}
      );
    }
  }
  return Object.fromEntries(
    specs.flatMap(spec => {
      const supplied = Object.hasOwn(raw, spec.name);
      if (!supplied && spec.defaultValue === undefined) return [];
      const rawValue = supplied ? raw[spec.name] : spec.defaultValue;
      const problem = getWorkflowParameterProblem(spec, rawValue);
      if (problem !== null) {
        const unit = spec.unit ? ` (${spec.unit === 'crs' ? 'CRS units' : spec.unit})` : '';
        throw new WorkflowError(
          'parameter-out-of-range',
          `${context.operationTitle}: ${spec.label}${unit} ${problem}.`,
          {parameter: spec.name}
        );
      }
      let value: CatalogParameterValue;
      switch (spec.kind) {
        case 'number':
        case 'integer':
          value = Number(rawValue);
          break;
        case 'number-list':
          // The validity check above guarantees a finite number at every index.
          value = [
            ...(getNumberListEntries(rawValue as string | readonly number[]) as readonly number[])
          ];
          break;
        case 'column-list':
          value = parseWorkflowColumnList(rawValue as string).join(', ');
          break;
        default:
          value = rawValue as string | boolean;
      }
      return [[spec.name, value]];
    })
  );
}

/** Text shown in a form control or an export for a parameter value (`1, 2, 3` for a list). */
export function formatWorkflowParameterValue(
  _spec: WorkflowParameterSpec,
  value: CatalogParameterValue
): string {
  return Array.isArray(value) ? value.join(', ') : String(value);
}

/** Whether `visibleWhen` lets the form show the parameter for the current values. */
export function isWorkflowParameterVisible(
  spec: WorkflowParameterSpec,
  values: Readonly<Record<string, CatalogParameterValue>>
): boolean {
  if (!spec.visibleWhen) return true;
  const value = values[spec.visibleWhen.parameter];
  return spec.visibleWhen.equals.some(candidate => {
    if (Array.isArray(candidate) && Array.isArray(value)) {
      return (
        candidate.length === value.length &&
        candidate.every((entry, index) => entry === value[index])
      );
    }
    return candidate === value;
  });
}

/** Names in a `column-list` value, trimmed, empty entries dropped. */
export function parseWorkflowColumnList(value: string): string[] {
  return value
    .split(',')
    .map(name => name.trim())
    .filter(name => name.length > 0);
}
