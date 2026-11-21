// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {COVERAGE_TERRAIN_CASES} from '../../analysis-core/kernel-audit/cases-coverage-terrain';
import {COVERAGE_RASTER_KERNEL_CASES} from '../../analysis-core/kernel-audit/cases-coverage-raster';
import {RTS_TIME_SERIES_AUDIT_CASES} from '../../analysis-core/kernel-audit/cases-rts-time-series';
import {RTS04_RTS07_MONITORING_EOF_AUDIT_CASES} from '../../analysis-core/kernel-audit/cases-rts04-rts07-monitoring-eof';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {getTerrainBindingViews} from '../../../src/gpu-terrain/terrain-processing/terrain-binding-preflight';

const REQUIRED_PORTS: Record<string, string[]> = {
  GPURasterBFASTMonitor: [
    'values',
    'parameters',
    'dateAxis.acquisitionTimes',
    'output.coefficients',
    'output.scores',
    'output.cellSummary'
  ],
  GPURasterCOLDMonitor: [
    'values',
    'parameters',
    'dateAxis.acquisitionTimes',
    'output.segments',
    'output.coefficients',
    'output.segmentStatistics',
    'output.cellSummary'
  ],
  GPURasterClimateAnomaly: ['state.counts', 'state.sums', 'accumulatorParameters'],
  GPURasterHANTS: [
    'values',
    'parameters',
    'output.smoothed',
    'output.coefficients',
    'output.status'
  ],
  GPURasterHarmonicRegression: [
    'values',
    'parameters',
    'dateAxis.acquisitionTimes',
    'output.coefficients',
    'output.summary'
  ],
  GPURasterLaggedCorrelation: [
    'values',
    'referenceIndex',
    'parameters',
    'output.correlations',
    'output.status'
  ],
  GPURasterWhittakerSmoother: ['values', 'parameters', 'output.smoothed', 'output.status'],
  GPURasterConservativeRemapping: [
    'values',
    'output.values',
    'output.coveredArea',
    'output.status'
  ],
  GPURasterRegistration: ['reference', 'moving', 'output.shift', 'output.peak', 'output.status'],
  GPURasterSeasonalBreak: [
    'values',
    'output.coefficients',
    'output.components',
    'output.breakIndex',
    'output.improvement',
    'output.status'
  ],
  GPURasterSpectralUnmixing: [
    'values',
    'endmembers',
    'output.abundances',
    'output.residual',
    'output.status',
    'output.iterationCounts'
  ],
  GPURasterTemporal: [
    'values',
    'output.values',
    'output.counts',
    'output.chosenSlice',
    'output.status'
  ],
  GPURasterTexture: ['values', 'features', 'pairCount'],
  GPUTerrainBranchingFlow: [
    'fractions',
    'values',
    'output.values',
    'output.status',
    'output.unresolvedCount'
  ],
  GPUTerrainBreach: [
    'elevation',
    'pitCells',
    'output.elevation',
    'output.cutDepth',
    'output.modificationMask',
    'output.pathCells',
    'output.pathLengths',
    'output.pitStatus',
    'output.unresolvedCount'
  ],
  GPUTerrainCatchmentSummary: [
    'elevation',
    'basinIds',
    'streams',
    'flowDirections',
    'output.statistics',
    'output.hypsometry',
    'output.profiles',
    'output.profileStatus'
  ],
  GPUTerrainDEMCoregistration: [
    'reference',
    'moving',
    'stableMask',
    'differenceVariance',
    'output.fit',
    'output.differences',
    'output.validity',
    'output.volume'
  ],
  GPUTerrainDepressionHierarchy: [
    'elevation',
    'outlets',
    'stages',
    'output.nodes',
    'output.levels',
    'output.cellNodes',
    'output.curves',
    'output.status.count',
    'output.status.requiredCount',
    'output.status.overflow'
  ],
  GPUTerrainFlowFractions: [
    'elevation',
    'flowDirections',
    'parameters',
    'output.fractions',
    'output.edgeLengths'
  ]
};

function replaceField(fields: object, path: string, replacement: unknown): object {
  const [name, ...rest] = path.split('.');
  const values = fields as Record<string, unknown>;
  return {
    ...values,
    [name]: rest.length
      ? replaceField(values[name] as object, rest.join('.'), replacement)
      : replacement
  };
}

for (const [name, requiredPorts] of Object.entries(REQUIRED_PORTS)) {
  it(`${name} rejects wrong formats and identifies missing required ports before dereferencing`, () => {
    const context = createKernelAuditContext(`ports-${name}`);
    let producer: (GPUCommandNodeProducer & {props: object}) | undefined;
    const capture = vi.spyOn(context.graph, 'add').mockImplementation(node => {
      if ('getCommandNodes' in node) producer = node as GPUCommandNodeProducer & {props: object};
    });
    try {
      const fixture = [
        ...COVERAGE_TERRAIN_CASES,
        ...COVERAGE_RASTER_KERNEL_CASES,
        ...RTS_TIME_SERIES_AUDIT_CASES,
        ...RTS04_RTS07_MONITORING_EOF_AUDIT_CASES
      ].find(candidate => candidate.exportName === name);
      expect(fixture, name).toBeDefined();
      fixture!.build(context);
      expect(producer, name).toBeDefined();
      const props = {...producer!.props, id: 'custom-layer'};
      const Producer = producer!.constructor as new (props: object) => object;
      expect(() => new Producer(props)).not.toThrow();
      for (const [path, view] of Object.entries(getTerrainBindingViews(props))) {
        for (const format of [
          'unorm8x4',
          view.format === 'float32' ? 'uint32' : 'float32'
        ] as const) {
          const replacement = context.view(format, view.length);
          expect(
            () => new Producer(replaceField(props, path, replacement)),
            `${name} ${path} ${format}`
          ).toThrow(/custom-layer.*format/);
        }
      }
      for (const path of requiredPorts) {
        expect(() => new Producer(replaceField(props, path, undefined)), `${name} ${path}`).toThrow(
          new RegExp(`custom-layer.*${path.replaceAll('.', '\\.')}.*received undefined`)
        );
      }
      // Masks are optional even when all result fields are mandatory.
      if ('mask' in props) expect(() => new Producer({...props, mask: undefined})).not.toThrow();
    } finally {
      capture.mockRestore();
      context.device.destroy();
    }
  });
}

for (const [name, path, invalidValue, accepted] of [
  ['GPURasterClimateAnomaly', 'output', undefined, 'map'],
  ['GPUTerrainDepressionHierarchy', 'output.ports', {}, 'output\\.status\\.count'],
  ['GPUTerrainBranchingFlow', 'mode', 'expectd-distance', 'expected-distance'],
  ['GPUTerrainFlowFractions', 'routing', 'mfd-freemen', 'mfd-freeman'],
  ['GPURasterSeasonalBreak', 'seasonality', 'dumym', 'harmonic'],
  ['GPURasterSeasonalBreak', 'fitMethod', 'jont', 'joint'],
  ['GPURasterSeasonalBreak', 'missingPolicy', 'interpolte', 'interpolate']
] as const) {
  it(`review rejects ${name} ${path} before native dereferences or algorithm selection`, () => {
    const context = createKernelAuditContext(`review-${name}`);
    let producer: (GPUCommandNodeProducer & {props: object}) | undefined;
    const capture = vi.spyOn(context.graph, 'add').mockImplementation(node => {
      if ('getCommandNodes' in node) producer = node as GPUCommandNodeProducer & {props: object};
    });
    try {
      const fixture = [...COVERAGE_TERRAIN_CASES, ...RTS_TIME_SERIES_AUDIT_CASES].find(
        candidate => candidate.exportName === name
      )!;
      fixture.build(context);
      const Producer = producer!.constructor as new (props: object) => object;
      const props = replaceField({...producer!.props, id: 'review-layer'}, path, invalidValue);
      expect(() => new Producer(props)).toThrow(new RegExp(`review-layer.*${path}.*${accepted}`));
    } finally {
      capture.mockRestore();
      context.device.destroy();
    }
  });
}
