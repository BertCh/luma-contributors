// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  isGraphDataView,
  isGraphVectorView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** All declared branch keys, preserving the constructor's independently inferred row views. @internal */
export type DataframePropKeys<Props> = Props extends unknown ? keyof Props : never;

/** At least one actual terminal output; undefined alone does not select a writer. @internal */
export type DataframeOutputSelection<Output> = {
  [Key in keyof Output]-?: Output & Required<Pick<Output, Key>>;
}[keyof Output];

/** Reject misspellings, then capture option values and nested port maps without cloning resources. @internal */
export function captureDataframeConstructorProps<Props extends object>(
  id: string,
  props: Props,
  allowedKeys: readonly string[]
): Props {
  let prototype: object | null = props;
  while (prototype !== null && !isDataframeObjectPrototype(prototype)) {
    for (const key of Reflect.ownKeys(prototype)) {
      if (prototype !== props && key === 'constructor') continue;
      validateRequirement(
        id,
        typeof key === 'string' && allowedKeys.includes(key),
        `unknown prop ${String(key)}; accepted props: ${allowedKeys.join(', ')}`
      );
    }
    prototype = Object.getPrototypeOf(prototype);
  }
  const captured = snapshotConstructionFields(props, allowedKeys as (keyof Props)[]);
  return captureOptions(captured) as Props;
}

// Declared nested fields are read through property access, including inherited accessors and
// non-enumerable properties. Remaining own option metadata is retained without cloning resources.
const NESTED_OPTION_FIELDS: Readonly<Record<string, readonly string[]>> = {
  temporalContext: ['unit', 'epoch', 'encoding', 'timeZone', 'axis'],
  axis: ['kind', 'step'],
  step: ['calendarUnit', 'interval'],
  periods: ['kind', 'step', 'origin', 'count', 'width'],
  origin: ['year', 'month', 'day', 'hour'],
  columns: ['year', 'month', 'dayOfMonth', 'hour'],
  timeZone: ['transitions', 'offsets'],
  model: [
    'featureCount',
    'treeCount',
    'nodeCount',
    'outputCount',
    'maximumDepth',
    'task',
    'transform'
  ],
  output: [
    'year',
    'month',
    'dayOfMonth',
    'hour',
    'minute',
    'weekday',
    'dayOfYear',
    'isoWeek',
    'isoWeekYear',
    'quarter',
    'hourWeekdayCounts',
    'periods',
    'keys',
    'predictions',
    'validity',
    'scores',
    'leafIndices',
    'status'
  ],
  status: ['invalidCount'],
  rasterStack: ['bands', 'grid'],
  bands: ['id', 'format', 'storage', 'validity'],
  storage: ['kind', 'values', 'texture'],
  grid: ['width', 'height', 'affine', 'pixelInterpretation']
};

function captureOptions(value: unknown, name = ''): unknown {
  if (!value || typeof value !== 'object' || isGraphDataView(value) || isGraphVectorView(value))
    return value;
  if (Array.isArray(value)) return Object.freeze(value.map(child => captureOptions(child, name)));
  const prototype = Object.getPrototypeOf(value);
  const declaredFields = Object.prototype.hasOwnProperty.call(NESTED_OPTION_FIELDS, name)
    ? NESTED_OPTION_FIELDS[name]
    : undefined;
  if (!declaredFields && prototype !== null && !isDataframeObjectPrototype(prototype)) return value;
  const fields = new Set<PropertyKey>(Reflect.ownKeys(value));
  for (const field of declaredFields ?? []) if (field in value) fields.add(field);
  return Object.freeze(
    Object.fromEntries(
      [...fields].map(field => [field, captureOptions(Reflect.get(value, field), String(field))])
    )
  );
}

/** Recognize the intrinsic Object prototype by identity with its own native constructor's prototype. */
function isDataframeObjectPrototype(prototype: object): boolean {
  if (Object.getPrototypeOf(prototype) !== null) return false;
  const objectConstructor = Object.getOwnPropertyDescriptor(prototype, 'constructor')?.value;
  return (
    typeof objectConstructor === 'function' &&
    objectConstructor.prototype === prototype &&
    Function.prototype.toString.call(objectConstructor) === Function.prototype.toString.call(Object)
  );
}
