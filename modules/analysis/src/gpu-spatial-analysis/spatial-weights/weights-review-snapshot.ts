// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {snapshotWeightsFields, snapshotWeightsDescription} from './weights-construction-snapshot';

/** @internal Capture inherited props once, retaining all requested status fields and certificates. */
export function snapshotAlgebraReviewProps<Props extends object>(value: Props): Props {
  const captured = snapshotWeightsFields(value, [
    'id',
    'operation',
    'output',
    'weights',
    'left',
    'right',
    'selfWeight',
    'mask',
    'groupIds',
    'groupCount',
    'order',
    'cumulative',
    'workCapacity',
    'periodCount',
    'layout',
    'lags',
    'weightRule',
    'rowCount',
    'capacity',
    'distances'
  ]) as Props & {
    output?: object;
    weights?: object;
    left?: object;
    right?: object;
    lags?: readonly object[];
  };
  return Object.freeze({
    ...captured,
    ...Object.fromEntries(
      (['output', 'weights', 'left', 'right'] as const).flatMap(name =>
        captured[name] ? [[name, snapshotWeightsDescription(captured[name]!)]] : []
      )
    ),
    ...(captured.lags
      ? {
          lags: Object.freeze(
            captured.lags.map(lag => snapshotWeightsFields(lag, ['lag', 'weight', 'includeSelf']))
          )
        }
      : {})
  }) as Props;
}

/** @internal Capture polygon producers, including profile port maps, without dropping inherited fields. */
export function snapshotPolygonReviewProps<Props extends object>(value: Props): Props {
  const captured = snapshotWeightsFields(value, [
    'id',
    'criterion',
    'polygons',
    'featureCount',
    'capacity',
    'snapTolerance',
    'coordinateExtent',
    'pairCapacity',
    'tolerance',
    'output'
  ]) as Props & {polygons: object; output: object & {weights?: object}};
  const output = snapshotWeightsDescription(captured.output);
  return Object.freeze({
    ...captured,
    polygons: snapshotWeightsFields(captured.polygons, [
      'kind',
      'positions',
      'ringOffsets',
      'polygonOffsets',
      'featureOffsets'
    ]),
    output: Object.freeze({
      ...output,
      ...(output.weights && typeof output.weights === 'object' && 'offsets' in output.weights
        ? {weights: snapshotWeightsDescription(output.weights)}
        : {})
    })
  }) as Props;
}
