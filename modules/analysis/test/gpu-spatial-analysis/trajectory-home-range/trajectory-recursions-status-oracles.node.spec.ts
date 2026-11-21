// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeTrajectoryRecursionsOnCPU,
  CPUTrajectoryRecursionsOperation
} from '../../../src/gpu-spatial-analysis/trajectory-recursion/cpu-trajectory-recursions-operation';
import {
  GPUTrajectoryRecursionsOperation,
  snapshotTrajectoryRecursionsOperationProps,
  type TrajectoryRecursionsOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-recursion/gpu-trajectory-recursions-operation';
import type {GPUTrajectoryRecursions} from '../../../src/gpu-spatial-analysis/trajectory-recursion/gpu-trajectory-recursions';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/temporal-context';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';

function createStructuralContributor(operation: GPUTrajectoryRecursionsOperation) {
  // Real graph views exercise the contributor factory without device submission.
  const definitions = {
    positions: ['float32x2', 4, 8],
    timestamps: ['float32', 4, 4],
    trackOffsets: ['uint32', 2, 4],
    radii: ['float32', 1, 4],
    residenceTimes: ['float32', 4, 4],
    candidateOverflow: ['uint32', 1, 4]
  } as const;
  const context = createKernelAuditContext('recursion-snapshot');
  try {
    return operation.createContributor(port => {
      const [format, length] = definitions[port as keyof typeof definitions];
      return context.view(format, length) as never;
    }) as GPUTrajectoryRecursions;
  } finally {
    context.device.destroy();
  }
}

function executeStationaryRecursions(operation: CPUTrajectoryRecursionsOperation) {
  return operation.execute({
    inputs: {
      positions: createInput('float32x2', new Float32Array(8)),
      timestamps: createInput('float32', Float32Array.from([0, 1, 2, 3])),
      trackOffsets: createInput('uint32', Uint32Array.from([0, 4])),
      radii: createInput('float32', Float32Array.from([1]))
    }
  });
}

// The independent oracle is a stationary four-fix track with three unit segments:
// cap one retains one unit of residence at each row and necessarily omits two segments.
for (const Operation of [GPUTrajectoryRecursionsOperation, CPUTrajectoryRecursionsOperation]) {
  for (const inheritedResidence of [false, true]) {
    it(`${Operation.name} preserves prototype output getters with inherited residence ${inheritedResidence}`, async () => {
      const reads = {residence: 0, status: 0, candidate: 0};
      class Status {
        get candidateOverflow() {
          reads.candidate++;
          return 'overflow';
        }
      }
      const status = new Status();
      class Output {
        get residenceTimes() {
          reads.residence++;
          return 'residence';
        }
        get status() {
          reads.status++;
          return status;
        }
      }
      const props = createMutationProps();
      props.output = inheritedResidence ? new Output() : {residenceTimes: 'residence', status};
      const operation = new Operation(props);
      if (operation instanceof GPUTrajectoryRecursionsOperation) {
        expect(
          Object.values(operation.outputs)
            .map(output => output.value)
            .sort()
        ).toEqual(['overflow', 'residence']);
        const contributor = createStructuralContributor(operation);
        expect(contributor.props.output.status?.candidateOverflow).toBeDefined();
        expect(contributor.props.output.residenceTimes).toBeDefined();
      } else {
        expect(Object.keys(operation.outputs).sort()).toEqual(['overflow', 'residence']);
        const result = await executeStationaryRecursions(operation);
        expect(Array.from(result['overflow'] as Uint32Array)).toEqual([1]);
        expect(Array.from(result['residence'] as Float32Array)).toEqual([1, 1, 1, 1]);
      }
      expect(reads).toEqual({
        residence: inheritedResidence ? 1 : 0,
        status: inheritedResidence ? 1 : 0,
        candidate: 1
      });
    });
  }
  for (const field of [
    'maximumTrackSegments',
    'rowCount',
    'radiusCount',
    'timestampFormat'
  ] as const) {
    it(`${Operation.name} reads changing ${field} getter once`, async () => {
      const props = createMutationProps();
      let reads = 0;
      const first = props[field];
      Object.defineProperty(props, field, {
        get() {
          reads++;
          return reads === 1
            ? first
            : field === 'timestampFormat'
              ? 'uint32x2'
              : field === 'radiusCount'
                ? 2
                : 3;
        }
      });
      const operation = new Operation(props);
      if (operation instanceof GPUTrajectoryRecursionsOperation) {
        const contributor = createStructuralContributor(operation);
        expect(contributor.maximumTrackSegments).toBe(1);
        expect(contributor.rowCount).toBe(4);
        expect(contributor.radiusCount).toBe(1);
      } else {
        const result = await executeStationaryRecursions(operation);
        expect(Array.from(result['overflow'] as Uint32Array)).toEqual([1]);
        expect(Array.from(result['residence'] as Float32Array)).toEqual([1, 1, 1, 1]);
      }
      expect(reads).toBe(1);
    });
  }
  for (const mutation of ['axis', 'spatial', 'temporal descriptor'] as const) {
    it(`${Operation.name} retains nested contexts after ${mutation} mutation`, async () => {
      const spatialContext: GPUSpatialContext = {
        coordinateSpace: 'planar',
        metric: 'native',
        units: 'meters'
      };
      const temporalContext: GPUTemporalContext = {
        encoding: 'float32-relative',
        unit: 'second',
        epoch: 0n,
        timeZone: 'UTC',
        axis: {kind: 'irregular'}
      };
      const props = {...createMutationProps(), spatialContext, temporalContext};
      const operation = new Operation(props);
      if (mutation === 'axis') Object.assign(temporalContext.axis, {kind: 'regular', step: 1});
      if (mutation === 'spatial')
        Object.assign(spatialContext, {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle'
        });
      if (mutation === 'temporal descriptor')
        Object.assign(temporalContext, {
          encoding: 'int64-words',
          unit: 'nanosecond',
          epoch: 9n,
          timeZone: 'changed'
        });
      if (operation instanceof GPUTrajectoryRecursionsOperation) {
        const contributor = createStructuralContributor(operation);
        expect(contributor.props.spatialContext).toEqual({
          coordinateSpace: 'planar',
          metric: 'native',
          units: 'meters'
        });
        expect(contributor.props.temporalContext).toEqual({
          encoding: 'float32-relative',
          unit: 'second',
          epoch: 0n,
          timeZone: 'UTC',
          axis: {kind: 'irregular'}
        });
        expect(contributor.maximumTrackSegments).toBe(1);
      } else {
        const result = await executeStationaryRecursions(operation);
        expect(Array.from(result['overflow'] as Uint32Array)).toEqual([1]);
        expect(Array.from(result['residence'] as Float32Array)).toEqual([1, 1, 1, 1]);
      }
    });
  }
}

function createMutationProps(): TrajectoryRecursionsOperationProps {
  return {
    positions: 'positions',
    timestamps: 'timestamps',
    trackOffsets: 'offsets',
    radii: 'radii',
    rowCount: 4,
    radiusCount: 1,
    timestampFormat: 'float32',
    maximumTrackSegments: 1,
    output: {residenceTimes: 'residence', status: {candidateOverflow: 'overflow'}}
  };
}

for (const Operation of [GPUTrajectoryRecursionsOperation, CPUTrajectoryRecursionsOperation]) {
  it(`${Operation.name} never reads an irregular axis step getter`, async () => {
    let kindReads = 0;
    const axis = {
      get kind(): 'irregular' {
        kindReads++;
        return 'irregular';
      },
      get step(): number {
        throw new Error('irrelevant step getter');
      }
    };
    const operation = new Operation({
      ...createMutationProps(),
      temporalContext: {encoding: 'float32-relative', unit: 'second', epoch: 0n, axis}
    });
    expect(kindReads).toBe(1);
    if (operation instanceof GPUTrajectoryRecursionsOperation) {
      expect(createStructuralContributor(operation).props.temporalContext?.axis).toEqual({
        kind: 'irregular'
      });
    } else {
      const result = await executeStationaryRecursions(operation);
      expect(Array.from(result['residence'] as Float32Array)).toEqual([1, 1, 1, 1]);
    }
  });
  it(`${Operation.name} rejects a missing axis with the named temporal contract`, () => {
    const props = {
      ...createMutationProps(),
      id: 'recursion-missing-axis',
      temporalContext: {
        encoding: 'float32-relative',
        unit: 'second',
        epoch: 0n
      }
    };
    // Reflect exercises an untyped JavaScript call without weakening the public props type.
    expect(() => Reflect.construct(Operation, [props])).toThrow(
      /^recursion-missing-axis axis must be regular or irregular$/
    );
  });
}

it('recursion snapshots retain numeric and calendar regular steps after caller mutation', () => {
  // A regular descriptor is valid temporal metadata, but recursion execution supports only
  // irregular axes. Test capture directly without changing that execution contract.
  const numeric = {kind: 'regular', step: 2} as const;
  const numericAxis: {kind: 'regular'; step: number} = {...numeric};
  const calendarAxis = {kind: 'regular', step: {calendarUnit: 'day', interval: 2}} as const;
  const mutableCalendar: {kind: 'regular'; step: {calendarUnit: 'day'; interval: number}} = {
    kind: calendarAxis.kind,
    step: {...calendarAxis.step}
  };
  for (const axis of [numericAxis, mutableCalendar]) {
    const captured = snapshotTrajectoryRecursionsOperationProps({
      ...createMutationProps(),
      temporalContext: {encoding: 'float32-relative', unit: 'second', epoch: 0n, axis}
    });
    if (typeof axis.step === 'number') axis.step = 9;
    else axis.step.interval = 9;
    expect(captured.temporalContext?.axis).toEqual(
      typeof axis.step === 'number' ? numeric : calendarAxis
    );
  }
});

for (const Operation of [GPUTrajectoryRecursionsOperation, CPUTrajectoryRecursionsOperation]) {
  it(`${Operation.name} rejects regular temporal axes before lowering`, () => {
    expect(
      () =>
        new Operation({
          ...createMutationProps(),
          id: 'review-temporal',
          temporalContext: {
            encoding: 'float32-relative',
            unit: 'second',
            epoch: 0n,
            axis: {kind: 'regular', step: 2}
          }
        })
    ).toThrow(/^review-temporal does not support a regular temporal axis$/);
  });
  it(`${Operation.name} rejects an unknown timestampFormat at construction`, () => {
    expect(() =>
      Reflect.construct(Operation, [{...createMutationProps(), timestampFormat: 'float64'}])
    ).toThrow(/timestampFormat must be float32 or uint32x2/);
  });
  for (const timestampFormat of ['float32', 'uint32x2'] as const) {
    it(`${Operation.name} rejects mismatched ${timestampFormat} encoding before lowering`, () => {
      expect(
        () =>
          new Operation({
            ...createMutationProps(),
            id: 'review-temporal',
            timestampFormat,
            temporalContext: {
              encoding: timestampFormat === 'float32' ? 'int64-words' : 'float32-relative',
              unit: 'second',
              epoch: 0n,
              axis: {kind: 'irregular'}
            }
          })
      ).toThrow(/^review-temporal does not support .* times/);
    });
    it(`${Operation.name} accepts matching ${timestampFormat} temporal metadata`, () => {
      expect(
        () =>
          new Operation({
            ...createMutationProps(),
            timestampFormat,
            temporalContext: {
              encoding: timestampFormat === 'float32' ? 'float32-relative' : 'int64-words',
              unit: 'second',
              epoch: 0n,
              axis: {kind: 'irregular'}
            }
          })
      ).not.toThrow();
    });
  }
}

for (const field of [
  'maximumTrackSegments',
  'rowCount',
  'radiusCount',
  'timestampFormat'
] as const) {
  it(`CPU recursion retains constructed ${field} after caller mutation`, async () => {
    const props = createMutationProps();
    const operation = new CPUTrajectoryRecursionsOperation(props);
    if (field === 'timestampFormat') props.timestampFormat = 'uint32x2';
    else props[field] = field === 'radiusCount' ? 2 : 3;
    const result = await operation.execute({
      inputs: {
        positions: createInput('float32x2', new Float32Array(8)),
        timestamps: createInput('float32', Float32Array.from([0, 1, 2, 3])),
        trackOffsets: createInput('uint32', Uint32Array.from([0, 4])),
        radii: createInput('float32', Float32Array.from([1]))
      }
    });
    // Three unit-duration segments, but the constructed cap retains only one per row.
    expect(Array.from(result['overflow'] as Uint32Array)).toEqual([1]);
    expect(Array.from(result['residence'] as Float32Array)).toEqual([1, 1, 1, 1]);
  });
}

it('GPU recursion factory retains constructed cap after caller mutation', () => {
  const props = createMutationProps();
  const operation = new GPUTrajectoryRecursionsOperation(props);
  props.maximumTrackSegments = 3;
  const contributor = createStructuralContributor(operation);
  expect(contributor.maximumTrackSegments).toBe(1);
});

function createInput(
  format: 'float32x2' | 'float32' | 'uint32',
  values: Float32Array | Uint32Array
) {
  const length = values.length / (format === 'float32x2' ? 2 : 1);
  return {format, length, chunks: [{values, length}], pack: () => values};
}

it('CPU recursion counts crossings in centred two-segment windows', () => {
  // x=[0,2,0,2], t=[0,2,4,6], radius=1: every examined segment spends
  // exactly one unit inside the focal disc. Windows are [0,2), [0,2),
  // [1,3), [1,3); endpoint-centred discs have two visits, others have one.
  const result = computeTrajectoryRecursionsOnCPU(
    {
      positions: new Float32Array([0, 0, 2, 0, 0, 0, 2, 0]),
      timestamps: new Float32Array([0, 2, 4, 6]),
      trackOffsets: new Uint32Array([0, 4]),
      radii: new Float32Array([1])
    },
    {maximumTrackSegments: 2}
  );
  expect(Array.from(result.candidateOverflow)).toEqual([1]);
  expect(Array.from(result.revisits)).toEqual([1, 0, 0, 1]);
  expect(Array.from(result.residenceTimes)).toEqual([2, 2, 2, 2]);
  expect(Array.from(result.firstPassageTimes)).toEqual([1, 1, 1, 0]);
  expect(Array.from(result.firstPassageCensored)).toEqual([0, 0, 0, 1]);
});

// Four stationary fixes have three unit-duration segments. A cap of 1 leaves
// two segments outside each row's window, so overflow=1 and residence=1.
// Caps of 3/4 cover all segments: overflow=0, residence=3. Passage is time to
// window end (all censored), [1,1,1,0] with cap 1, [3,2,1,0] otherwise.
for (const scenario of [
  {budget: 1, overflow: [1], residence: [1, 1, 1, 1], passage: [1, 1, 1, 0]},
  {budget: 3, overflow: [0], residence: [3, 3, 3, 3], passage: [3, 2, 1, 0]},
  {budget: 4, overflow: [0], residence: [3, 3, 3, 3], passage: [3, 2, 1, 0]}
]) {
  it(`CPU recursion honors the explicit segment window ${scenario.budget}`, async () => {
    const operation = new CPUTrajectoryRecursionsOperation({
      positions: 'positions',
      timestamps: 'timestamps',
      trackOffsets: 'offsets',
      radii: 'radii',
      rowCount: 4,
      radiusCount: 1,
      maximumTrackSegments: scenario.budget,
      output: {
        residenceTimes: 'residence',
        firstPassageTimes: 'passage',
        firstPassageCensored: 'censored',
        status: {candidateOverflow: 'overflow'}
      }
    });
    const result = await operation.execute({
      inputs: {
        positions: createInput('float32x2', new Float32Array(8)),
        timestamps: createInput('float32', new Float32Array([0, 1, 2, 3])),
        trackOffsets: createInput('uint32', new Uint32Array([0, 4])),
        radii: createInput('float32', new Float32Array([1]))
      }
    });
    expect(Array.from(result['overflow'] as Uint32Array)).toEqual(scenario.overflow);
    expect(Array.from(result['residence'] as Float32Array)).toEqual(scenario.residence);
    expect(Array.from(result['passage'] as Float32Array)).toEqual(scenario.passage);
    expect(Array.from(result['censored'] as Uint32Array)).toEqual([1, 1, 1, 1]);
  });
}
