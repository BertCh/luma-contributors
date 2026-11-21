// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import type {GPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';

/** Capture caller frame getters before passing plain fields to the shared deep-copy helper. */
function snapshotPairSpatialContext(context: GPUSpatialContext): GPUSpatialContext {
  const captured = snapshotConstructionFields(context, [
    'coordinateSpace',
    'metric',
    'units',
    'sphereRadius',
    'ellipsoid',
    'edgeModel',
    'datum',
    'gridFamily',
    'gridResolution',
    'frame'
  ]);
  return snapshotConstructionSpatialContext({
    ...captured,
    ...(captured.frame
      ? {
          frame: snapshotConstructionFields(captured.frame, [
            'id',
            'kind',
            'origin',
            'crs',
            'signature',
            'unit',
            'axisOrder',
            'areaOfUse',
            'anchor',
            'precision'
          ])
        }
      : {})
  });
}

/** Capture temporal metadata, including calendar steps, without copying any live data. */
function snapshotSpaceTimeTemporalContext(context: GPUTemporalContext): GPUTemporalContext {
  const captured = snapshotConstructionFields(context, [
    'unit',
    'epoch',
    'encoding',
    'timeZone',
    'axis'
  ]);
  const axis = captured.axis;
  // Preserve invalid declarations for the named temporal-context validator to reject.
  if (!axis) return captured;
  const capturedKind = snapshotConstructionFields(axis, ['kind']);
  const kind = capturedKind.kind;
  if (kind !== 'regular' && kind !== 'irregular') return captured;
  if (kind === 'irregular') return Object.freeze({...captured, axis: Object.freeze({kind})});
  const capturedStep = snapshotConstructionFields(
    axis as Extract<GPUTemporalContext['axis'], {kind: 'regular'}>,
    ['step']
  );
  const step = capturedStep.step;
  return Object.freeze({
    ...captured,
    axis: Object.freeze({
      kind,
      step:
        step && typeof step === 'object'
          ? snapshotConstructionFields(step, ['calendarUnit', 'interval'])
          : step
    })
  });
}

/** @internal Capture topology and metadata, retaining the live graph views and their buffers. */
export function snapshotSpaceTimePairProps<
  Props extends {
    gridSize: readonly [number, number];
    spatialContext?: GPUSpatialContext;
    temporalContext?: GPUTemporalContext;
    output: object;
  }
>(
  props: Props,
  fields: readonly (keyof Props)[],
  outputFields: readonly (keyof Props['output'])[]
): Props {
  const captured = snapshotConstructionFields(props, fields);
  return Object.freeze({
    ...captured,
    gridSize: Object.freeze([...captured.gridSize]) as readonly [number, number],
    spatialContext: captured.spatialContext && snapshotPairSpatialContext(captured.spatialContext),
    temporalContext:
      captured.temporalContext && snapshotSpaceTimeTemporalContext(captured.temporalContext),
    output: snapshotConstructionFields<Props['output']>(captured.output, outputFields)
  });
}
