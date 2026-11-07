// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  createNativeStylePlan,
  type NativeStyleLayer,
  type NativeStylePlanContext
} from '../../../examples/deck/kepler-compute/src/native-style-plan';

/**
 * NATIVEUI5 qualification matrix: every native scalar scale mode (SCALE_TYPES) against every
 * scalar channel (size, radius, height, elevation offset, coverage). Numeric channels accept
 * linear, sqrt and log; point is allowed only for dictionary strings; everything else must refuse
 * with a stated reason rather than silently mapping to another scale.
 */
const source = {datasetId: 'sample', revision: 1};
const context: NativeStylePlanContext = {
  source,
  columns: {
    value: {format: 'float32'},
    category: {format: 'uint32', dictionary: ['a', 'b', 'c']}
  },
  zoom: 12,
  h3Branch: 'columns',
  // GeoJSON layers need explicit source-property presence metadata.
  geojsonProperties: {
    fillColor: 'absent',
    lineColor: 'absent',
    lineWidth: 'absent',
    radius: 'absent',
    elevation: 'absent',
    elevationOffset: 'absent'
  },
  geojsonPointFeatures: true,
  geojsonPolygonFeatures: true
};
const NATIVE_SCALES = [
  'ordinal',
  'quantile',
  'quantize',
  'jenks',
  'linear',
  'sqrt',
  'log',
  'threshold',
  'custom',
  'customOrdinal',
  'point'
] as const;
const NUMERIC_SUPPORTED = new Set(['linear', 'sqrt', 'log']);

type ChannelCase = {
  name: string;
  layerType: string;
  channel: string;
  fieldKey: string;
  scaleKey: string;
  domainKey: string;
  visConfig: Record<string, unknown>;
};
const CASES: ChannelCase[] = [
  {
    name: 'point radius',
    layerType: 'point',
    channel: 'size',
    fieldKey: 'sizeField',
    scaleKey: 'sizeScale',
    domainKey: 'sizeDomain',
    visConfig: {radiusRange: [0, 50]}
  },
  {
    name: 'geojson height',
    layerType: 'geojson',
    channel: 'height',
    fieldKey: 'heightField',
    scaleKey: 'heightScale',
    domainKey: 'heightDomain',
    visConfig: {enable3d: true, heightRange: [0, 500]}
  },
  {
    name: 'geojson elevation offset',
    layerType: 'geojson',
    channel: 'elevationOffset',
    fieldKey: 'elevationOffsetField',
    scaleKey: 'elevationOffsetScale',
    domainKey: 'elevationOffsetDomain',
    visConfig: {enable3d: true, elevationOffsetRange: [0, 500]}
  },
  {
    name: 'h3 elevation',
    layerType: 'hexagonId',
    channel: 'size',
    fieldKey: 'sizeField',
    scaleKey: 'sizeScale',
    domainKey: 'sizeDomain',
    visConfig: {enable3d: true, sizeRange: [0, 500]}
  },
  {
    name: 'h3 coverage',
    layerType: 'hexagonId',
    channel: 'coverage',
    fieldKey: 'coverageField',
    scaleKey: 'coverageScale',
    domainKey: 'coverageDomain',
    visConfig: {coverageRange: [0, 1]}
  }
];

function makeLayer(testCase: ChannelCase, scale: string, field: string): NativeStyleLayer {
  return {
    id: 'native',
    type: testCase.layerType,
    config: {
      dataId: source.datasetId,
      color: [5, 10, 15],
      [testCase.fieldKey]: {name: field},
      [testCase.scaleKey]: scale,
      [testCase.domainKey]: scale === 'point' ? ['a', 'b'] : [1, 100],
      visConfig: {
        filled: true,
        colorRange: {colors: ['#ff0000', '#00ff00']},
        ...testCase.visConfig
      }
    }
  } as unknown as NativeStyleLayer;
}

describe('scalar scale mode matrix', () => {
  for (const testCase of CASES) {
    for (const scale of NATIVE_SCALES) {
      test(`${testCase.name} with ${scale}`, () => {
        const numeric = createNativeStylePlan(makeLayer(testCase, scale, 'value'), context);
        const channel = numeric.supported ? numeric.plan.channels[testCase.channel] : undefined;
        if (NUMERIC_SUPPORTED.has(scale)) {
          expect(numeric.supported ? '' : numeric.reasons.join(' ')).toBe('');
          expect(channel).toMatchObject({kind: 'resident', plan: {scale}});
        } else {
          // Unsupported modes refuse the whole layer plan with a reason; they never remap silently.
          expect(numeric.supported).toBe(false);
          if (!numeric.supported) expect(numeric.reasons.join(' ')).toMatch(/\S/);
        }
        const categorical = createNativeStylePlan(makeLayer(testCase, scale, 'category'), context);
        if (scale === 'point') {
          expect(categorical.supported).toBe(true);
          if (categorical.supported)
            expect(categorical.plan.channels[testCase.channel]).toMatchObject({
              kind: 'resident',
              plan: {scale: 'point'}
            });
        } else expect(categorical.supported).toBe(false);
      });
    }
  }
});
