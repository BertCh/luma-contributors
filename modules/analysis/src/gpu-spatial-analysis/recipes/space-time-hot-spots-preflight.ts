// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  resolveGPUMultipleTestingMethod,
  GPU_MULTIPLE_TESTING_MAXIMUM_ROWS
} from '../multiple-testing/index';
import {GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH} from '../../dataframe-analysis/calendar-buckets/index';
import {
  validateGPUPeriodIndexStepsPerPeriod,
  validateGPUPeriodIndexOriginStep
} from '../../dataframe-analysis/period-index/period-index-calendar';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {GPU_COLOR_SCALE_PARAMETER_LENGTH} from '../../dataframe-analysis/column-classification/index';
import {
  getGPUEmergingHotSpotsParameterValues,
  GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT,
  GPU_EMERGING_HOT_SPOTS_MAXIMUM_RADIUS,
  GPU_EMERGING_HOT_SPOTS_MAXIMUM_CORRECTED_TREND_SLICE_COUNT,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
  type GPUEmergingHotSpotsSettings
} from '../emerging-hot-spots/index';
import {validateGPUSpatialWeightsOrChunks} from '../spatial-weights/index';
import {resolveRecipeDestinations} from './recipe-utils';
import type {GPUSpaceTimeHotSpotsRecipeProps} from './space-time-hot-spots-recipe';

// The emerging-hot-spot contributor uses exact float32 cell indices and signed bin indices.
const MAXIMUM_CELL_COUNT = 2 ** 24 - 1;
const MAXIMUM_BIN_COUNT = 2 ** 31 - 1;

/** Capture every declared recipe port, including inherited status getters, before validation. @internal */
export function snapshotSpaceTimeHotSpotsRecipe(
  props: GPUSpaceTimeHotSpotsRecipeProps
): GPUSpaceTimeHotSpotsRecipeProps {
  const destinations = resolveRecipeDestinations(props.id ?? 'space-time-hot-spots', props);
  const captured = snapshotConstructionFields(props, [
    'id',
    'timestamps',
    'mask',
    'utcOffsets',
    'timeZone',
    'calendarParameters',
    'cells',
    'slices',
    'cellMask',
    'settings',
    'parameters',
    'multipleTesting',
    'finalSlice',
    'trendAutocorrelationCorrection',
    'trendCorrectionMaximumLag',
    'trendCorrectionCriticalZ',
    'color',
    'scratch'
  ]);
  const output = snapshotConstructionFields(destinations ?? {}, [
    'cube',
    'giZScores',
    'trendZ',
    'trendP',
    'trendS',
    'category',
    'hotSliceCount',
    'coldSliceCount',
    'pValues',
    'adjustedPValues',
    'lastValidSlice',
    'colors',
    'status'
  ]);
  const slices = snapshotConstructionFields(
    captured.slices as GPUSpaceTimeHotSpotsRecipeProps['slices'] & {field?: unknown},
    ['step', 'interval', 'origin', 'count', 'field']
  );
  validateRequirement(
    'addSpaceTimeHotSpotsRecipe',
    !('field' in slices),
    'slices.field is no longer accepted: cyclic calendar fields cannot carry a trend, use slices.step with an origin'
  );
  let cells = snapshotConstructionFields(captured.cells, [
    'kind',
    'positions',
    'spatialContext',
    'width',
    'height',
    'bounds',
    'maximumRadius',
    'cellIds',
    'cellCount',
    'weights',
    'selfWeight'
  ] as (keyof typeof captured.cells)[]);
  if (cells.kind === 'ids') {
    const weights = snapshotConstructionFields(cells.weights, [
      'offsets',
      'neighbors',
      'weights',
      'distances',
      'spatialContext',
      'information',
      'status'
    ]);
    cells = {
      ...cells,
      weights: {
        ...weights,
        ...(weights.status
          ? {
              status: snapshotConstructionFields(weights.status, [
                'count',
                'requiredCount',
                'overflow',
                'candidateOverflow',
                'complete',
                'invalidCount',
                'uncertainCount',
                'iterationCount',
                'converged',
                'iterationLimitReached',
                'approximation'
              ])
            }
          : {})
      }
    };
  } else cells = {...cells, bounds: Object.freeze([...cells.bounds]) as typeof cells.bounds};
  return Object.freeze({
    ...captured,
    cells,
    slices: {
      ...slices,
      origin: snapshotConstructionFields(slices.origin, ['year', 'month', 'day', 'hour'])
    },
    settings: snapshotConstructionFields(captured.settings, [
      'radius',
      'temporalWindow',
      'confidenceLevel',
      'criticalZ',
      'trendSignificanceLevel',
      'persistentFraction',
      'minimumTrendSliceCount',
      'sporadicRequiresFinalSlice',
      'multipleTesting'
    ] as (keyof typeof captured.settings)[]),
    output: {
      ...output,
      ...(output.status
        ? {
            status: snapshotConstructionFields(output.status, [
              'skippedEventCount',
              'groupOverflow',
              'groupRequiredCount'
            ])
          }
        : {})
    },
    scratch: snapshotConstructionFields(captured.scratch ?? {}, [
      'calendarYear',
      'calendarMonth',
      'calendarDayOfMonth',
      'calendarHour',
      'periods',
      'keys',
      'keyMask',
      'groupKeys',
      'groupCounts',
      'groupCount',
      'colorDomain'
    ]),
    ...(captured.timeZone
      ? {timeZone: snapshotConstructionFields(captured.timeZone, ['transitions', 'offsets'])}
      : {}),
    ...(captured.color
      ? {
          color: snapshotConstructionFields(captured.color, [
            'palette',
            'parameters',
            'maximumPaletteCount'
          ])
        }
      : {})
  });
}

/** Checks caller preconditions without registering any recipe resources or writing settings. @internal */
export function validateSpaceTimeHotSpotsRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUSpaceTimeHotSpotsRecipeProps
): Float32Array {
  validateRequirement(
    id,
    props.parameters.length >= GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
    `parameters needs ${GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH} values`
  );
  const multipleTesting = resolveGPUMultipleTestingMethod(
    id,
    props.multipleTesting ?? 'benjamini-hochberg'
  );
  const settings = props.settings as GPUEmergingHotSpotsSettings;
  validateRequirement(
    id,
    settings.multipleTesting === undefined || settings.multipleTesting === multipleTesting,
    'settings.multipleTesting must match the recipe'
  );
  const packedSettings = getGPUEmergingHotSpotsParameterValues({...settings, multipleTesting});
  validateRequirement(
    id,
    props.finalSlice === undefined ||
      props.finalSlice === 'require' ||
      props.finalSlice === 'last-valid',
    'finalSlice must be require or last-valid'
  );
  const {cells, slices} = props;
  validateGPUPeriodIndexStepsPerPeriod(id, {calendarUnit: slices.step, interval: slices.interval});
  validateGPUPeriodIndexOriginStep(id, slices.step, slices.origin);
  const eventCount = props.timestamps.length;
  const cellCount = cells.kind === 'lattice' ? cells.width * cells.height : cells.cellCount;
  const binCount = cellCount * slices.count;
  validateCapacity(id, 'timestamps.length', eventCount);
  validateCapacity(id, 'cells count', cellCount, 1, MAXIMUM_CELL_COUNT);
  validateCapacity(id, 'cellCount * slices.count', binCount, 1, MAXIMUM_BIN_COUNT);
  validateCapacity(id, 'slices.count', slices.count, 1, GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT);
  if (cells.kind === 'lattice') {
    validateCapacity(id, 'cells.width', cells.width);
    validateCapacity(id, 'cells.height', cells.height);
    validateCapacity(
      id,
      'cells.maximumRadius',
      cells.maximumRadius ?? 4,
      0,
      GPU_EMERGING_HOT_SPOTS_MAXIMUM_RADIUS
    );
    for (const coordinate of cells.bounds) validateFiniteNumber(id, 'cells.bounds', coordinate);
    validateRequirement(
      id,
      cells.bounds[2] > cells.bounds[0] && cells.bounds[3] > cells.bounds[1],
      'cells.bounds must have positive width and height'
    );
  } else {
    validateRequirement(
      id,
      validateGPUSpatialWeightsOrChunks(id, cells.weights) === cellCount,
      'cells.weights row count must equal cells.cellCount'
    );
    validateFiniteNumber(id, 'cells.selfWeight float32', Math.fround(cells.selfWeight ?? 1), 0);
  }
  const correction = props.trendAutocorrelationCorrection ?? 'none';
  validateRequirement(
    id,
    ['none', 'hamed-rao', 'yue-wang'].includes(correction),
    'trendAutocorrelationCorrection must be none, hamed-rao or yue-wang'
  );
  if (correction !== 'none')
    validateCapacity(
      id,
      'slices.count with trendAutocorrelationCorrection',
      slices.count,
      1,
      GPU_EMERGING_HOT_SPOTS_MAXIMUM_CORRECTED_TREND_SLICE_COUNT
    );
  if (props.trendCorrectionMaximumLag !== undefined)
    validateCapacity(id, 'trendCorrectionMaximumLag', props.trendCorrectionMaximumLag);
  if (props.trendCorrectionCriticalZ !== undefined)
    validateFiniteNumber(
      id,
      'trendCorrectionCriticalZ float32',
      Math.fround(props.trendCorrectionCriticalZ),
      0
    );
  if (multipleTesting !== 'none' || props.output?.pValues || props.output?.adjustedPValues) {
    validateCapacity(
      id,
      'multiple-testing bin count',
      binCount,
      1,
      GPU_MULTIPLE_TESTING_MAXIMUM_ROWS - 1
    );
  }

  const output = props.output ?? {};
  const scratch = props.scratch ?? {};
  const inputs: Record<string, GraphDataView | undefined> = {
    timestamps: props.timestamps,
    calendarParameters: props.calendarParameters,
    mask: props.mask,
    utcOffsets: props.utcOffsets,
    cellMask: props.cellMask,
    'cells.positions': cells.kind === 'lattice' ? cells.positions : undefined,
    'cells.cellIds': cells.kind === 'ids' ? cells.cellIds : undefined,
    'timeZone.transitions': props.timeZone?.transitions,
    'timeZone.offsets': props.timeZone?.offsets,
    'color.palette': props.color?.palette,
    'color.parameters': props.color?.parameters,
    'color.domain': scratch.colorDomain,
    ...(cells.kind === 'ids'
      ? {
          'weights.offsets': cells.weights.offsets,
          'weights.neighbors': cells.weights.neighbors,
          'weights.weights': cells.weights.weights,
          'weights.distances': cells.weights.distances,
          ...Object.fromEntries(
            Object.entries(cells.weights.status ?? {}).map(([name, view]) => [
              `weights.status.${name}`,
              view
            ])
          )
        }
      : {})
  };
  const writable: Record<string, GraphDataView | undefined> = {
    ...Object.fromEntries(Object.entries(output).filter(([name]) => name !== 'status')),
    ...output.status,
    ...Object.fromEntries(Object.entries(scratch).filter(([name]) => name !== 'colorDomain'))
  };
  validateGraphChunkBindingRanges(graph, id, {...inputs, ...writable});
  const calendarOutputs = {
    calendarYear: props.scratch?.calendarYear,
    calendarMonth: props.scratch?.calendarMonth,
    calendarDayOfMonth: props.scratch?.calendarDayOfMonth,
    calendarHour: props.scratch?.calendarHour
  };
  validateGraphOutputsDisjointFromInputs(id, calendarOutputs, {
    timestamps: props.timestamps,
    mask: props.mask,
    utcOffsets: props.utcOffsets,
    calendarParameters: props.calendarParameters,
    transitions: props.timeZone?.transitions,
    offsets: props.timeZone?.offsets
  });
  const emergingOutputs = {
    giZScores: output.giZScores,
    trendZ: output.trendZ,
    trendP: output.trendP,
    trendS: output.trendS,
    category: output.category,
    hotSliceCount: output.hotSliceCount,
    coldSliceCount: output.coldSliceCount,
    pValues: output.pValues,
    adjustedPValues: output.adjustedPValues,
    lastValidSlice: output.lastValidSlice
  };
  validateDisjointWritableViews(id, emergingOutputs);
  validateGraphOutputsDisjointFromInputs(id, emergingOutputs, {
    cube: output.cube,
    cellMask: props.cellMask,
    ...(cells.kind === 'ids'
      ? {
          offsets: cells.weights.offsets,
          neighbors: cells.weights.neighbors,
          weights: cells.weights.weights
        }
      : {})
  });
  const parameterBuffer = getCoreDefaultBuffer(props.parameters.buffer);
  for (const [name, view] of Object.entries(writable)) {
    validateRequirement(
      id,
      !view?.buffer.defaultBuffer ||
        getCoreDefaultBuffer(view.buffer.defaultBuffer) !== parameterBuffer,
      `${name} must not share the parameters buffer`
    );
  }
  validateGraphOutputsDisjointFromInputs(
    id,
    {colors: output.colors},
    {
      category: output.category,
      palette: props.color?.palette,
      colorParameters: props.color?.parameters,
      colorDomain: props.scratch?.colorDomain
    }
  );
  validateGraphOutputsDisjointFromInputs(
    id,
    {
      groupKeys: props.scratch?.groupKeys,
      groupCounts: props.scratch?.groupCounts,
      groupCount: props.scratch?.groupCount,
      groupOverflow: output.status?.groupOverflow,
      groupRequiredCount: output.status?.groupRequiredCount
    },
    {keys: props.scratch?.keys, keyMask: props.scratch?.keyMask}
  );
  // Cross-stage writes must not destroy caller inputs or other published values. Distinct passes
  // may share a physical buffer only when their logical write ranges are disjoint.
  validateDisjointWritableViews(id, writable, {separateDispatches: true});
  for (const [name, input] of Object.entries(inputs)) {
    if (input)
      validateDisjointWritableViews(
        id,
        {...writable, [`input.${name}`]: input},
        {separateDispatches: true}
      );
  }
  const fields: [string, GraphDataView | undefined, GraphDataView['format'], number, boolean?][] = [
    ['timestamps', props.timestamps, 'uint32x2', eventCount],
    [
      'calendarParameters',
      props.calendarParameters,
      'sint32',
      GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH,
      true
    ],
    ['mask', props.mask, 'uint32', eventCount],
    ['utcOffsets', props.utcOffsets, 'sint32', eventCount],
    ['cellMask', props.cellMask, 'uint32', cellCount],
    [
      'cells.positions',
      cells.kind === 'lattice' ? cells.positions : undefined,
      'float32x2',
      eventCount
    ],
    ['cells.cellIds', cells.kind === 'ids' ? cells.cellIds : undefined, 'uint32', eventCount],
    ['cube', output.cube, 'uint32', binCount],
    ['giZScores', output.giZScores, 'float32', binCount],
    ['pValues', output.pValues, 'float32', binCount],
    ['adjustedPValues', output.adjustedPValues, 'float32', binCount],
    ['trendZ', output.trendZ, 'float32', cellCount],
    ['trendP', output.trendP, 'float32', cellCount],
    ['trendS', output.trendS, 'sint32', cellCount],
    ['category', output.category, 'uint32', cellCount],
    ['hotSliceCount', output.hotSliceCount, 'uint32', cellCount],
    ['coldSliceCount', output.coldSliceCount, 'uint32', cellCount],
    ['lastValidSlice', output.lastValidSlice, 'uint32', cellCount],
    ['colors', output.colors, 'uint32', cellCount, true],
    ['calendarYear', scratch.calendarYear, 'sint32', eventCount],
    ['calendarMonth', scratch.calendarMonth, 'uint32', eventCount],
    ['calendarDayOfMonth', scratch.calendarDayOfMonth, 'uint32', eventCount],
    ['calendarHour', scratch.calendarHour, 'uint32', eventCount],
    ['periods', scratch.periods, 'uint32', eventCount],
    ['keys', scratch.keys, 'uint32', eventCount],
    ['keyMask', scratch.keyMask, 'uint32', eventCount],
    ['groupKeys', scratch.groupKeys, 'uint32', 1, true],
    ['groupCounts', scratch.groupCounts, 'uint32', scratch.groupKeys?.length ?? binCount],
    ['groupCount', scratch.groupCount, 'uint32', 1],
    ['colorDomain', scratch.colorDomain, 'float32', 2, true],
    ...Object.entries(output.status ?? {}).map(
      ([name, view]) =>
        [name, view, 'uint32', 1, name === 'groupOverflow' || name === 'groupRequiredCount'] as [
          string,
          GraphDataView | undefined,
          'uint32',
          number,
          boolean
        ]
    )
  ];
  if (scratch.groupKeys && !scratch.groupCounts)
    validateRequirement(
      id,
      scratch.groupKeys.length === binCount,
      'groupKeys length must match the default groupCounts length'
    );
  if (props.timeZone) {
    fields.push(
      ['timeZone.transitions', props.timeZone.transitions, 'uint32x2', 1, true],
      ['timeZone.offsets', props.timeZone.offsets, 'sint32', props.timeZone.transitions.length + 1]
    );
  }
  if (props.color) {
    const maximumPaletteCount = props.color.maximumPaletteCount ?? 18;
    validateCapacity(id, 'color.maximumPaletteCount', maximumPaletteCount);
    fields.push(
      ['color.palette', props.color.palette, 'uint32', maximumPaletteCount, true],
      [
        'color.parameters',
        props.color.parameters,
        'float32',
        GPU_COLOR_SCALE_PARAMETER_LENGTH,
        true
      ]
    );
  }
  for (const [name, view, format, length, allowSpare] of fields) {
    if (!view) continue;
    validateSinglePackedView(view, [format], `${id} ${name}`);
    if (allowSpare)
      validateRequirement(id, view.length >= length, `${name} must hold at least ${length} rows`);
    else validateRequirement(id, view.length === length, `${name} length must equal ${length}`);
  }
  // Check the actual co-bound glue subsets (aligned ranges include subview prefixes).
  const eventInputs = {
    'cells.positions': inputs['cells.positions'],
    'cells.cellIds': inputs['cells.cellIds'],
    periods: scratch.periods
  };
  const checkBindings = (
    writes: Record<string, GraphDataView | undefined>,
    reads: Record<string, GraphDataView | undefined>
  ) => {
    validateDisjointWritableViews(id, writes);
    for (const [name, view] of Object.entries(reads))
      if (view) validateDisjointWritableViews(id, {...writes, [`read.${name}`]: view});
  };
  checkBindings({skippedEventCount: output.status?.skippedEventCount}, eventInputs);
  checkBindings(
    {cube: output.cube},
    {
      ...eventInputs,
      groupKeys: scratch.groupKeys,
      groupCounts: scratch.groupCounts,
      groupCount: scratch.groupCount
    }
  );
  checkBindings({keys: scratch.keys, keyMask: scratch.keyMask}, eventInputs);
  return packedSettings;
}
