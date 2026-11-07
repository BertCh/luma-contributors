// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {scaleOrdinal, scaleQuantile, scaleThreshold} from 'd3-scale';
import {packGPUColor} from '@luma-contributors/graph/gpu-dataframe';
import {GPUVector} from '@luma-contributors/graph/gpu-data';
import {
  createNativeStylePlan,
  type NativeStyleLayer,
  type NativeStylePlanContext,
  type NativeStyleChannel,
  type NativeStylePlan
} from '../../../examples/deck/kepler-compute/src/native-style-plan';

const source = {datasetId: 'sample', revision: 3};
const context: NativeStylePlanContext = {
  source,
  columns: {
    value: {format: 'float32'},
    category: {format: 'uint32', dictionary: ['zebra', 'ant', 'moth']},
    unsafeInteger: {format: 'uint32'}
  },
  zoom: 12
};
function layer(type = 'point'): NativeStyleLayer {
  return {
    id: 'native',
    type,
    config: {
      dataId: source.datasetId,
      color: [5, 10, 15],
      colorDomain: [0, 1],
      colorScale: 'quantile',
      sizeDomain: [0, 100],
      sizeScale: 'linear',
      visConfig: {
        filled: true,
        stroked: true,
        colorRange: {colors: ['#ff0000', '#00ff00', '#0000ff']},
        strokeColorRange: {colors: ['#ffffff', '#000000']},
        radius: 10,
        radiusRange: [0, 50],
        thickness: 2,
        sizeRange: [0, 10],
        opacity: 0.8,
        strokeOpacity: 0.5
      }
    }
  };
}
function plan(native: NativeStyleLayer, options = context): NativeStylePlan {
  const result = createNativeStylePlan(native, options);
  if (!result.supported) throw new Error(result.reasons.join('\n'));
  return result.plan;
}
function resident(result: NativeStylePlan, channel: NativeStyleChannel) {
  const value = result.channels[channel];
  if (value?.kind !== 'resident') throw new Error(`Expected resident ${channel}`);
  return value.plan;
}

describe('pinned native Kepler resident style planning', () => {
  test('normalizes pinned PropertyKey scales and nullable restored defaults without host casts', () => {
    const native = layer();
    native.config.colorField = native.config.sizeField = {name: 'value'};
    native.config.colorScale = native.config.sizeScale = null;
    expect(resident(plan(native), 'fillColor')).toMatchObject({scale: 'quantile'});
    expect(resident(plan(native), 'size')).toMatchObject({scale: 'linear'});
    for (const scale of [42, Symbol('unsupported')]) {
      native.config.colorScale = scale;
      expect(createNativeStylePlan(native, context)).toMatchObject({supported: false});
    }
  });
  test('keeps constants and native zoom/wrapper multipliers outside resident output buffers', () => {
    const native = layer();
    expect(plan(native).channels).toEqual({
      fillColor: {kind: 'constant', value: [5, 10, 15]},
      size: {kind: 'constant', value: 1}
    });
    expect(plan(native).render).toMatchObject({radiusScale: 40, radiusMaxPixels: 500});
    native.config.visConfig.fixedRadius = true;
    expect(plan(native).render.radiusScale).toBe(40);
    for (const type of ['line', 'arc']) {
      native.type = type;
      native.config.visConfig.targetColor = [20, 30, 40, 90];
      expect(plan(native).channels.targetColor).toEqual({
        kind: 'constant',
        value: [20, 30, 40, 90]
      });
      expect(plan(native).render).toMatchObject({widthUnits: 'pixels', widthScale: 4 / 3});
    }
  });

  test('translates fixed quantile samples into native d3 quantile edges, retaining ties', () => {
    const native = layer('arc');
    native.config.colorField = {name: 'value'};
    const samples = [100, 0, 4, 4, 4, 10, 30];
    native.config.colorDomain = samples;
    const colors = [packGPUColor(255, 0, 0), packGPUColor(0, 255, 0), packGPUColor(0, 0, 255)];
    const oracle = scaleQuantile<number>().domain(samples).range(colors);
    const translated = plan(native);
    expect(resident(translated, 'fillColor')).toMatchObject({
      kind: 'color',
      scale: 'quantile',
      source,
      column: 'value',
      palette: colors,
      domain: {kind: 'fixed', values: [0, ...oracle.quantiles().map(Math.fround), 100]},
      noDataColor: 0,
      interpolation: 'step',
      outsideDomain: 'clamp'
    });
    const {id: sourceId, ...sourcePlan} = resident(translated, 'fillColor');
    const {id: targetId, ...targetPlan} = resident(translated, 'targetColor');
    expect(sourceId).not.toBe(targetId);
    expect(sourcePlan).toEqual(targetPlan);
    native.config.colorDomain = [4, 4, 4];
    expect(resident(plan(native), 'fillColor').domain).toEqual({
      kind: 'fixed',
      values: [4, 4, 4, 4]
    });
    expect(samples).toEqual([100, 0, 4, 4, 4, 10, 30]);
  });

  test('retains source/selection domain scope and revision without copying masks into recipes', () => {
    const native = layer();
    native.config.colorField = native.config.sizeField = {name: 'value'};
    native.config.sizeScale = 'sqrt';
    const translated = plan(native, {
      ...context,
      domains: {
        fillColor: {kind: 'selection', revision: 8},
        size: {kind: 'source'}
      }
    });
    expect(resident(translated, 'fillColor').domain).toEqual({kind: 'selection', revision: 8});
    expect(resident(translated, 'size')).toMatchObject({
      scale: 'sqrt',
      domain: {kind: 'source'},
      range: [0, 50],
      nullValue: 0
    });
    expect(translated.render.radiusScale).toBe(4);
    native.config.visConfig.fixedRadius = true;
    expect(resident(plan(native), 'size')).toMatchObject({scale: 'identity'});
    expect(plan(native).render.radiusScale).toBe(1);
    native.config.sizeDomain = [7, 7];
    expect(resident(plan(native), 'size').domain).toEqual({kind: 'fixed', values: [7, 7]});
  });

  test('preserves precomputed resident quantile edges and ignores disabled native color channels', () => {
    const native = layer();
    native.config.colorField = {name: 'value'};
    const edges = [0, 10, 30, 100];
    expect(
      resident(
        plan(native, {...context, domains: {fillColor: {kind: 'fixed', values: edges}}}),
        'fillColor'
      ).domain
    ).toEqual({kind: 'fixed', values: edges});
    native.config.colorField = {name: 'missing'};
    native.config.visConfig.filled = false;
    expect(plan(native).channels.fillColor).toBeUndefined();
  });

  test('honors d3 continuous palette length and explicit native sqrt/quantize scales', () => {
    const native = layer();
    native.config.colorField = {name: 'value'};
    for (const scale of ['linear', 'sqrt']) {
      native.config.colorScale = scale;
      expect(resident(plan(native), 'fillColor')).toMatchObject({
        scale,
        palette: [packGPUColor(255, 0, 0), packGPUColor(0, 255, 0)],
        outsideDomain: 'extrapolate'
      });
    }
    native.config.colorScale = 'quantize';
    expect(resident(plan(native), 'fillColor')).toMatchObject({
      scale: 'equal-interval',
      outsideDomain: 'clamp'
    });
  });

  test('translates fixed and GPU-derived log controls while rejecting fixed zero-crossing domains', () => {
    const native = layer();
    native.config.colorField = {name: 'value'};
    native.config.colorScale = 'log';
    for (const values of [
      [1, 100],
      [-100, -1],
      [100, 1],
      [-1, -100],
      [7, 7]
    ]) {
      native.config.colorDomain = values;
      expect(resident(plan(native), 'fillColor')).toMatchObject({
        scale: 'log',
        domain: {kind: 'fixed', values},
        interpolation: 'linear',
        outsideDomain: 'extrapolate',
        palette: [packGPUColor(255, 0, 0), packGPUColor(0, 255, 0)]
      });
    }
    for (const values of [
      [-1, 1],
      [0, 100]
    ]) {
      native.config.colorDomain = values;
      expect(createNativeStylePlan(native, context)).toMatchObject({
        supported: false,
        reasons: [expect.stringContaining('same-sign')]
      });
    }
    native.config.colorDomain = [1, 100];
    native.config.sizeField = {name: 'value'};
    native.config.sizeScale = 'log';
    for (const domain of [{kind: 'source'} as const, {kind: 'selection', revision: 2} as const]) {
      // The native fixed defaults do not stand in for the selected GPU population.
      native.config.colorDomain = native.config.sizeDomain = [-1, 1];
      const translated = plan(native, {...context, domains: {fillColor: domain, size: domain}});
      expect(resident(translated, 'fillColor')).toMatchObject({
        scale: 'log',
        domain,
        noDataColor: 0
      });
      expect(resident(translated, 'size')).toMatchObject({scale: 'log', domain, nullValue: 0});
    }
  });

  test('preserves native custom threshold map order, duplicate keys, tails and exact break ownership', () => {
    const native = layer();
    native.config.colorField = {name: 'value'};
    native.config.colorScale = 'custom';
    const colors = [packGPUColor(0, 0, 255), packGPUColor(0, 255, 0), packGPUColor(255, 255, 255)];
    native.config.visConfig.colorRange = {
      colors: [],
      colorMap: [
        [10, '#ff0000'],
        [20, '#00ff00'],
        [10, '#0000ff'],
        [null, '#ffffff']
      ]
    };
    const translated = resident(
      plan(native, {...context, domains: {fillColor: {kind: 'selection', revision: 9}}}),
      'fillColor'
    );
    expect(translated).toMatchObject({
      scale: 'quantile',
      domain: {kind: 'fixed', values: [10, 10, 20, 20]},
      palette: colors,
      interpolation: 'step',
      outsideDomain: 'clamp',
      noDataColor: 0
    });
    if (translated.kind !== 'color' || translated.domain.kind !== 'fixed')
      throw new Error('Expected fixed colors');
    const thresholds = translated.domain.values.slice(1, -1);
    const oracle = scaleThreshold<number, number>().domain([10, 20]).range(colors);
    for (const value of [-Infinity, -1000, 0, 9.5, 10, 19.5, 20, 1000, Infinity]) {
      const index = thresholds.filter(threshold => threshold <= value).length;
      expect(translated.palette[index]).toBe(oracle(value));
    }
  });

  test('uses only active native threshold controls, including tied boundaries and constant palettes', () => {
    const native = layer();
    native.config.colorField = {name: 'value'};
    native.config.colorScale = 'custom';
    native.config.colorDomain = [4, 4, 999];
    expect(resident(plan(native), 'fillColor')).toMatchObject({
      domain: {kind: 'fixed', values: [4, 4, 4, 4]}
    });
    native.config.colorDomain = [4];
    expect(resident(plan(native), 'fillColor')).toMatchObject({
      domain: {kind: 'fixed', values: [4, 4, 4]},
      palette: [packGPUColor(255, 0, 0), packGPUColor(0, 255, 0)]
    });
    native.config.visConfig.colorRange = {colors: [], colorMap: [[null, [1, 2, 3, 90]]]};
    expect(resident(plan(native), 'fillColor')).toMatchObject({
      domain: {kind: 'fixed', values: [0, 0]},
      palette: [packGPUColor(1, 2, 3, 90)]
    });
  });

  test('rejects custom threshold approximations and distinct NaN/null colors explicitly', () => {
    const native = layer();
    native.config.colorField = {name: 'value'};
    native.config.colorScale = 'custom';
    for (const [domain, reason] of [
      [[0.1, 10], 'exactly representable'],
      [[20, 10], 'ascending'],
      [[NaN, 10], 'finite numbers'],
      [[Infinity, 10], 'finite numbers']
    ] as const) {
      native.config.colorDomain = domain;
      expect(createNativeStylePlan(native, context)).toMatchObject({
        supported: false,
        reasons: [expect.stringContaining(reason)]
      });
    }
    native.config.visConfig.colorRange = {
      colors: [],
      colorMap: [
        [10, '#ff0000'],
        ['__unknownColor__', '#ffffff']
      ]
    };
    expect(createNativeStylePlan(native, context)).toMatchObject({
      supported: false,
      reasons: [expect.stringContaining('separate NaN and null')]
    });
    native.config.visConfig.colorRange.colorMap = [
      [10, '#ff0000'],
      ['__unknownColor__', [0, 0, 0, 0]]
    ];
    expect(resident(plan(native), 'fillColor')).toMatchObject({
      domain: {kind: 'fixed', values: [10, 10, 10]},
      palette: [packGPUColor(255, 0, 0), 0],
      noDataColor: 0
    });
  });

  test('remaps native category order into dictionary-code colors and point-scale domain order', () => {
    const native = layer();
    native.config.colorField = native.config.sizeField = {name: 'category'};
    native.config.colorScale = 'ordinal';
    native.config.sizeScale = 'point';
    native.config.colorDomain = native.config.sizeDomain = ['ant', 'moth', 'zebra'];
    const colors = [packGPUColor(255, 0, 0), packGPUColor(0, 255, 0), packGPUColor(0, 0, 255)];
    const oracle = scaleOrdinal<string, number>().domain(['ant', 'moth', 'zebra']).range(colors);
    expect(resident(plan(native), 'fillColor')).toMatchObject({
      palette: context.columns['category'].dictionary!.map(label => oracle(String(label))),
      legend: {labels: ['ant', 'moth', 'zebra'], codes: [1, 2, 0]}
    });
    expect(resident(plan(native), 'size')).toMatchObject({
      scale: 'point',
      domain: {kind: 'fixed', values: [1, 2, 0]},
      range: [0, 50]
    });
  });

  test('preserves customOrdinal overflow grouping, explicit grouped maps and unknown transparency', () => {
    const native = layer();
    native.config.colorField = {name: 'category'};
    native.config.colorScale = 'customOrdinal';
    native.config.colorDomain = ['zebra', 'ant', 'moth'];
    native.config.visConfig.colorRange = {colors: ['#ff0000', '#0000ff']};
    expect(resident(plan(native), 'fillColor')).toMatchObject({
      palette: [packGPUColor(0, 0, 255), packGPUColor(255, 0, 0), packGPUColor(0, 0, 255)],
      legend: {labels: ['ant', 'moth', 'zebra'], codes: [1, 2, 0]}
    });
    native.config.visConfig.colorRange.colorMap = [[['zebra', 'ant'], '#010203']];
    expect(resident(plan(native), 'fillColor')).toMatchObject({
      palette: [packGPUColor(1, 2, 3), packGPUColor(1, 2, 3), 0],
      legend: {labels: ['zebra', 'ant'], codes: [0, 1]}
    });
  });

  test('plans GeoJSON fill, stroke, width and point radius independently', () => {
    const native = layer('geojson');
    native.config.colorField =
      native.config.strokeColorField =
      native.config.sizeField =
      native.config.radiusField =
        {name: 'value'};
    native.config.strokeColorDomain = [0, 100];
    native.config.strokeColorScale = 'quantize';
    native.config.radiusDomain = [0, 100];
    native.config.radiusScale = 'sqrt';
    const translated = plan(native, {...context, geojsonPointFeatures: true});
    expect(resident(translated, 'fillColor')).toMatchObject({scale: 'quantile'});
    expect(resident(translated, 'strokeColor')).toMatchObject({scale: 'equal-interval'});
    expect(resident(translated, 'size')).toMatchObject({scale: 'linear', range: [0, 10]});
    expect(resident(translated, 'pointSize')).toMatchObject({scale: 'sqrt', range: [0, 50]});
    expect(translated.render).toMatchObject({
      widthUnits: 'meters',
      widthScale: 64,
      radiusScale: 4,
      strokeOpacity: 0.5
    });
  });

  test('translates native Trip field channels and meter widths without implied GeoJSON properties', () => {
    const native = layer('trip');
    // Trips remain lines regardless of unrelated fill/stroke/outline controls in a saved config.
    native.config.visConfig.filled = native.config.visConfig.stroked = false;
    native.config.visConfig.outline = true;
    native.config.visConfig.targetColor = [99, 88, 77];
    const constants = plan(native);
    expect(constants.channels).toEqual({
      fillColor: {kind: 'constant', value: [5, 10, 15]},
      size: {kind: 'constant', value: 1}
    });
    expect(constants.render).toMatchObject({widthUnits: 'meters', widthScale: 64});
    native.config.visConfig.billboard = true;
    expect(plan(native).render.widthScale).toBe((64 * 2) / 3);
    native.config.visConfig.thickness = undefined;
    expect(plan(native).render.widthScale).toBe((16 * 2) / 3);
    native.config.colorField = native.config.sizeField = {name: 'value'};
    native.config.sizeScale = 'sqrt';
    const selected = plan(native, {
      ...context,
      domains: {
        fillColor: {kind: 'selection', revision: 4},
        size: {kind: 'source'}
      }
    });
    expect(Object.keys(selected.channels)).toEqual(['fillColor', 'size']);
    expect(resident(selected, 'fillColor')).toMatchObject({
      scale: 'quantile',
      noDataColor: 0,
      domain: {kind: 'selection', revision: 4}
    });
    expect(resident(selected, 'size')).toMatchObject({
      scale: 'sqrt',
      nullValue: 0,
      range: [0, 10],
      domain: {kind: 'source'}
    });
  });

  test('translates native Icon radius ranges and distinct fixedRadius wrapper semantics', () => {
    const native = layer('icon');
    native.config.visConfig.filled = false;
    native.config.visConfig.outline = true;
    expect(plan(native).channels).toEqual({
      fillColor: {kind: 'constant', value: [5, 10, 15]},
      size: {kind: 'constant', value: 1}
    });
    expect(plan(native).render).toMatchObject({
      radiusUnits: 'meters',
      radiusScale: 40,
      radiusMaxPixels: 500
    });
    native.config.visConfig.fixedRadius = true;
    // Icon calls getRadiusScaleByZoom directly, without Point's Boolean(sizeField) override.
    expect(plan(native).render).toMatchObject({
      radiusScale: 1,
      radiusMaxPixels: Number.MAX_SAFE_INTEGER
    });
    native.config.colorField = native.config.sizeField = {name: 'value'};
    native.config.sizeScale = 'sqrt';
    native.config.visConfig.radiusRange = [2, 24];
    const translated = plan(native, {
      ...context,
      domains: {fillColor: {kind: 'source'}, size: {kind: 'selection', revision: 3}}
    });
    expect(Object.keys(translated.channels)).toEqual(['fillColor', 'size']);
    expect(resident(translated, 'fillColor')).toMatchObject({scale: 'quantile', noDataColor: 0});
    // Icon's size visual channel does not declare fixed:'fixedRadius': field encoding remains sqrt.
    expect(resident(translated, 'size')).toMatchObject({
      scale: 'sqrt',
      range: [2, 24],
      nullValue: 0,
      domain: {kind: 'selection', revision: 3}
    });
    native.config.visConfig.fixedRadius = false;
    expect(plan(native).render.radiusScale).toBe(4);
    expect(plan(native, {...context, zoom: 16}).render.radiusScale).toBe(0.25);
    expect(plan(native, {...context, zoom: 16, zoomOffset: 1}).render.radiusScale).toBe(0.5);
    native.config.sizeField = null;
    expect(plan(native, {...context, zoom: 16}).render.radiusScale).toBe(2.5);
    native.config.visConfig.fixedRadius = true;
    expect(plan(native, {...context, zoom: 16}).render.radiusScale).toBe(1);
    // Base Point zoom behavior remains clamped at its zoom-14 radius.
    expect(plan(layer('point'), {...context, zoom: 16}).render.radiusScale).toBe(10);
  });

  test('preserves native H3 constants, meter outline widths and independent elevation zoom', () => {
    const native = layer('hexagonId');
    native.config.visConfig.outline = true;
    native.config.visConfig.coverage = 0.4;
    native.config.visConfig.thickness = 3;
    native.config.visConfig.elevationScale = 2;
    native.config.visConfig.strokeColor = [9, 8, 7];
    const result = plan(native, {...context, zoom: 6, h3Branch: 'columns'});
    expect(result.channels).toEqual({
      fillColor: {kind: 'constant', value: [5, 10, 15]},
      strokeColor: {kind: 'constant', value: [9, 8, 7]},
      size: {kind: 'constant', value: 500},
      coverage: {kind: 'constant', value: 1}
    });
    expect(result.render).toMatchObject({
      widthUnits: 'meters',
      widthScale: 3,
      coverage: 0.4,
      elevationScale: 8,
      extruded: false,
      filled: true,
      stroked: true,
      strokeOpacity: 0.5
    });
    expect(
      plan(native, {...context, zoom: 6, zoomOffset: 1, h3Branch: 'columns'}).render.elevationScale
    ).toBe(16);
    native.config.visConfig.enableElevationZoomFactor = false;
    expect(plan(native, {...context, zoom: 6, h3Branch: 'columns'}).render.elevationScale).toBe(2);
    native.config.visConfig.enableElevationZoomFactor = true;
    native.config.visConfig.fixedHeight = true;
    expect(plan(native, {...context, zoom: 6, h3Branch: 'columns'}).render.elevationScale).toBe(2);
  });

  test('maps native H3 coverage and elevation fields independently, including fixed heights', () => {
    const native = layer('hexagonId');
    native.config.sizeField = native.config.coverageField = {name: 'value'};
    native.config.coverageScale = 'sqrt';
    native.config.coverageDomain = [0, 100];
    native.config.visConfig.coverageRange = [0, 1];
    native.config.visConfig.sizeRange = undefined;
    const options: NativeStylePlanContext = {
      ...context,
      h3Branch: 'columns',
      domains: {
        size: {kind: 'selection', revision: 5},
        coverage: {kind: 'source'}
      }
    };
    const result = plan(native, options);
    expect(resident(result, 'coverage')).toMatchObject({
      kind: 'scalar',
      column: 'value',
      scale: 'sqrt',
      range: [0, 1],
      nullValue: 0,
      domain: {kind: 'source'}
    });
    expect(resident(result, 'size')).toMatchObject({
      kind: 'scalar',
      column: 'value',
      scale: 'linear',
      range: [0, 500],
      nullValue: 0,
      domain: {kind: 'selection', revision: 5}
    });
    expect(result.render).toMatchObject({coverage: 1, extruded: false});
    // Base getAttributeAccessors maps configured sizeField even when its UI condition is false.
    native.config.visConfig.enable3d = true;
    native.config.visConfig.fixedHeight = true;
    expect(resident(plan(native, options), 'size')).toMatchObject({scale: 'identity'});
    expect(plan(native, options).render).toMatchObject({extruded: true, elevationScale: 5});
  });

  test('native H3 polygon branch ignores per-row coverage and separate stroke opacity', () => {
    const native = layer('hexagonId');
    native.config.coverageField = {name: 'not-uploaded'};
    native.config.coverageScale = 'not-a-scale';
    native.config.visConfig.coverage = 0.25;
    native.config.visConfig.outline = true;
    native.config.visConfig.strokeOpacity = 0.1;
    native.config.visConfig.opacity = 0.7;
    const high = plan(native, {...context, h3Branch: 'polygons'});
    expect(high.channels.coverage).toEqual({kind: 'constant', value: 1});
    expect(high.render).toMatchObject({coverage: 1, strokeOpacity: 0.7});
    expect(createNativeStylePlan(native, {...context, h3Branch: 'columns'})).toMatchObject({
      supported: false,
      reasons: [expect.stringContaining('No resident source column')]
    });
    native.config.coverageField = null;
    expect(plan(native, {...context, h3Branch: 'polygons'}).render.coverage).toBe(0.25);
    expect(createNativeStylePlan(native, context)).toMatchObject({
      supported: false,
      reasons: [expect.stringContaining('current selected-population rendering branch')]
    });
  });

  test('H3 render visibility excludes unused fill/outline fields and 3D outlines', () => {
    const native = layer('hexagonId');
    native.config.visConfig.filled = false;
    native.config.visConfig.outline = true;
    native.config.visConfig.enable3d = true;
    native.config.colorField = native.config.strokeColorField = {name: 'not-uploaded'};
    const result = plan(native, {...context, h3Branch: 'columns'});
    expect(Object.keys(result.channels)).toEqual(['size', 'coverage']);
    expect(result.render).toMatchObject({filled: false, stroked: false, extruded: true});
    native.config.visConfig.enable3d = false;
    expect(createNativeStylePlan(native, {...context, h3Branch: 'columns'})).toMatchObject({
      supported: false,
      reasons: [expect.stringContaining('strokeColor')]
    });
  });

  test('uses Trip line properties only when explicitly bound, with native field precedence', () => {
    const native = layer('trip');
    const colors = new GPUVector({
      type: 'data',
      name: 'property',
      format: 'unorm8x4',
      data: [],
      ownsData: false
    });
    const widths = new GPUVector({
      type: 'data',
      name: 'property',
      format: 'float32',
      data: [],
      ownsData: false
    });
    const validity = new GPUVector({
      type: 'data',
      name: 'property',
      format: 'uint32',
      data: [],
      ownsData: false
    });
    const tripProperties = {
      lineColor: {values: colors, validity},
      lineWidth: {values: widths, validity}
    };
    const result = plan(native, {...context, tripProperties});
    expect(result.channels.fillColor).toEqual({
      kind: 'property',
      property: 'lineColor',
      source: tripProperties.lineColor,
      fallback: [5, 10, 15]
    });
    expect(result.channels.size).toEqual({
      kind: 'property',
      property: 'lineWidth',
      source: tripProperties.lineWidth,
      fallback: 1
    });
    native.config.colorField = native.config.sizeField = {name: 'value'};
    const selected = plan(native, {
      ...context,
      tripProperties: {
        get lineColor(): never {
          throw new Error('Unexpected trip property read');
        },
        get lineWidth(): never {
          throw new Error('Unexpected trip property read');
        }
      }
    });
    expect(selected.channels.fillColor?.kind).toBe('resident');
    expect(selected.channels.size?.kind).toBe('resident');
  });

  test('matches pinned Trip width accessors for default, false and true stroked controls', () => {
    const native = layer('trip');
    const values = new GPUVector({
      type: 'data',
      name: 'property',
      format: 'float32',
      data: [],
      ownsData: false
    });
    const validity = new GPUVector({
      type: 'data',
      name: 'property',
      format: 'uint32',
      data: [],
      ownsData: false
    });
    const source = {values, validity};
    // Trip's visualChannel.condition is a UI control. Pinned BaseLayer.getAttributeAccessors
    // checks config[field] directly and never invokes condition before mapping getWidth.
    for (const stroked of [undefined, false, true]) {
      native.config.visConfig.stroked = stroked;
      native.config.sizeField = null;
      const defaults = plan(native);
      expect(defaults.channels.size).toEqual({kind: 'constant', value: 1});
      const sourceDefault = plan(native, {
        ...context,
        tripProperties: {lineColor: 'absent', lineWidth: source}
      });
      expect(sourceDefault.channels.size).toEqual({
        kind: 'property',
        property: 'lineWidth',
        source,
        fallback: 1
      });
      native.config.sizeField = {name: 'value'};
      const mapped = plan(native, {
        ...context,
        tripProperties: {
          lineColor: 'absent',
          get lineWidth(): never {
            throw new Error('Field must bypass lineWidth');
          }
        }
      });
      expect(resident(mapped, 'size')).toMatchObject({
        column: 'value',
        scale: 'linear',
        nullValue: 0,
        range: [0, 10]
      });
    }
  });

  test('rejects observed Trip properties lacking uploads instead of assuming absence', () => {
    const native = layer('trip');
    expect(
      createNativeStylePlan(native, {...context, tripProperties: {lineColor: 'absent'}})
    ).toMatchObject({
      supported: false,
      reasons: ['size: Trip lineWidth property presence requires upload metadata.']
    });
    expect(
      createNativeStylePlan(native, {...context, tripProperties: {lineWidth: 'absent'}})
    ).toMatchObject({
      supported: false,
      reasons: ['fillColor: Trip lineColor property presence requires upload metadata.']
    });
  });

  test('requires explicit source-property presence metadata and retains uploaded property vectors', () => {
    const native = layer('geojson');
    expect(createNativeStylePlan(native, context)).toMatchObject({supported: false});
    const properties = {
      fillColor: 'absent',
      lineColor: 'absent',
      lineWidth: 'absent',
      radius: 'absent'
    } as const;
    const translated = plan(native, {
      ...context,
      geojsonProperties: properties,
      geojsonPointFeatures: true
    });
    expect(translated.channels.pointSize).toEqual({kind: 'constant', value: 1});
    const width = new GPUVector({
      type: 'data',
      name: 'property',
      format: 'float32',
      data: [],
      ownsData: false
    });
    const validity = new GPUVector({
      type: 'data',
      name: 'property',
      format: 'uint32',
      data: [],
      ownsData: false
    });
    const propertySource = {values: width, validity};
    expect(
      plan(native, {...context, geojsonProperties: {...properties, lineWidth: propertySource}})
        .channels.size
    ).toEqual({kind: 'property', property: 'lineWidth', source: propertySource, fallback: 1});
    native.config.visConfig.filled = native.config.visConfig.stroked = false;
    expect(plan(native).channels).toEqual({});
  });

  test('keeps immutable source properties separate from live native fallbacks and field plans', () => {
    const native = layer('geojson');
    const colors = new GPUVector({
      type: 'data',
      name: 'property',
      format: 'unorm8x4',
      data: [],
      ownsData: false
    });
    const values = new GPUVector({
      type: 'data',
      name: 'property',
      format: 'float32',
      data: [],
      ownsData: false
    });
    const validity = new GPUVector({
      type: 'data',
      name: 'property',
      format: 'uint32',
      data: [],
      ownsData: false
    });
    const colorSource = {values: colors, validity};
    const scalarSource = {values, validity};
    const options: NativeStylePlanContext = {
      ...context,
      geojsonPointFeatures: true,
      geojsonFixedRadius: true,
      geojsonProperties: {
        fillColor: colorSource,
        lineColor: colorSource,
        lineWidth: scalarSource,
        radius: scalarSource
      }
    };
    const first = plan(native, options);
    expect(first.channels.fillColor).toEqual({
      kind: 'property',
      property: 'fillColor',
      source: colorSource,
      fallback: [5, 10, 15]
    });
    expect(first.channels.pointSize).toEqual({
      kind: 'property',
      property: 'radius',
      source: scalarSource,
      fallback: 1
    });
    expect(first.render.radiusScale).toBe(1);
    native.config.color = [50, 60, 70];
    native.config.visConfig.strokeColor = [80, 90, 100];
    native.config.visConfig.thickness = 3;
    const changed = plan(native, options);
    expect(changed.channels.fillColor).toEqual({
      kind: 'property',
      property: 'fillColor',
      source: colorSource,
      fallback: [50, 60, 70]
    });
    expect(changed.channels.strokeColor).toEqual({
      kind: 'property',
      property: 'lineColor',
      source: colorSource,
      fallback: [80, 90, 100]
    });
    expect(changed.channels.size).toEqual({
      kind: 'property',
      property: 'lineWidth',
      source: scalarSource,
      fallback: 1
    });
    expect(changed.render.widthScale).toBe(96);
    native.config.colorField =
      native.config.strokeColorField =
      native.config.sizeField =
      native.config.radiusField =
        {name: 'value'};
    native.config.strokeColorDomain = native.config.radiusDomain = [0, 100];
    // A native field owns its channel; source properties must not be read in this path.
    const selected = plan(native, {
      ...options,
      geojsonProperties: {
        get fillColor(): never {
          throw new Error('Unexpected property read');
        },
        get lineColor(): never {
          throw new Error('Unexpected property read');
        },
        get lineWidth(): never {
          throw new Error('Unexpected property read');
        },
        get radius(): never {
          throw new Error('Unexpected property read');
        }
      }
    });
    for (const channel of ['fillColor', 'strokeColor', 'size', 'pointSize'] as const)
      expect(selected.channels[channel]?.kind).toBe('resident');
  });

  test('rejects missing sources, unsupported scales, implicit ordinal growth and zoom domains atomically', () => {
    const native = layer();
    native.config.colorField = {name: 'unsafeInteger'};
    native.config.sizeField = {name: 'missing'};
    const result = createNativeStylePlan(native, context);
    expect(result.supported).toBe(false);
    if (!result.supported) expect(result.reasons).toHaveLength(2);
    native.config.colorField = {name: 'category'};
    native.config.colorScale = 'ordinal';
    native.config.colorDomain = ['ant'];
    native.config.sizeField = null;
    expect(createNativeStylePlan(native, context)).toMatchObject({
      supported: false,
      reasons: [expect.stringContaining('complete native domain')]
    });
    native.config.colorField = {name: 'value'};
    native.config.colorScale = 'jenks';
    expect(createNativeStylePlan(native, context)).toMatchObject({supported: false});
    native.config.colorScale = 'linear';
    native.config.colorDomain = {stops: [0, 1], z: [0, 14]};
    expect(createNativeStylePlan(native, context)).toMatchObject({
      supported: false,
      reasons: [expect.stringContaining('Zoom-dependent')]
    });
  });
});

const polygonContext: NativeStylePlanContext = {
  ...context,
  zoom: 6,
  geojsonPolygonFeatures: true,
  geojsonProperties: {
    fillColor: 'absent',
    lineColor: 'absent',
    lineWidth: 'absent',
    elevation: 'absent',
    elevationOffset: 'absent'
  }
};

describe('native GeoJSON height and independent elevation offset', () => {
  test('maps height and offset independently from stroke width and their domain scopes', () => {
    const native = layer('geojson');
    native.config.sizeField =
      native.config.heightField =
      native.config.elevationOffsetField =
        {name: 'value'};
    native.config.visConfig.enable3d = true;
    native.config.heightScale = 'sqrt';
    native.config.heightDomain = [10, 20];
    native.config.elevationOffsetScale = 'linear';
    native.config.elevationOffsetDomain = [-5, 5];
    native.config.visConfig.heightRange = [10, 1000];
    native.config.visConfig.elevationOffsetRange = [-30, 30];
    native.config.visConfig.elevationScale = 3;
    // GeoJSON ignores this H3-specific toggle; its own fixedHeight controls the zoom factor.
    native.config.visConfig.enableElevationZoomFactor = false;
    const translated = plan(native, {
      ...polygonContext,
      zoomOffset: 1,
      domains: {height: {kind: 'selection', revision: 12}, elevationOffset: {kind: 'source'}}
    });
    expect(resident(translated, 'size')).toMatchObject({
      scale: 'linear',
      range: [0, 10],
      domain: {kind: 'fixed', values: [0, 100]}
    });
    expect(resident(translated, 'height')).toMatchObject({
      scale: 'sqrt',
      range: [10, 1000],
      domain: {kind: 'selection', revision: 12},
      nullValue: 0
    });
    expect(resident(translated, 'elevationOffset')).toMatchObject({
      scale: 'linear',
      range: [-30, 30],
      domain: {kind: 'source'},
      nullValue: 0
    });
    expect(translated.render).toMatchObject({
      elevationScale: 24,
      elevationOffsetActive: true,
      extruded: true,
      stroked: false
    });
    expect(resident(translated, 'elevationOffset')).not.toHaveProperty('elevationScale');
  });

  test('keeps embedded height fallback and finite offset presence separate from layer defaults', () => {
    const native = layer('geojson');
    native.config.visConfig.enable3d = true;
    const values = new GPUVector({
      type: 'data',
      name: 'property',
      format: 'float32',
      data: [],
      ownsData: false
    });
    const validity = new GPUVector({
      type: 'data',
      name: 'property',
      format: 'uint32',
      data: [],
      ownsData: false
    });
    const property = {values, validity};
    const options = {
      ...polygonContext,
      geojsonProperties: {
        ...polygonContext.geojsonProperties,
        elevation: property,
        elevationOffset: property
      }
    };
    const first = plan(native, options);
    expect(first.channels.height).toEqual({
      kind: 'property',
      property: 'elevation',
      source: property,
      fallback: 500
    });
    expect(first.channels.elevationOffset).toEqual({
      kind: 'property',
      property: 'elevationOffset',
      source: property,
      fallback: 0
    });
    native.config.visConfig.elevationOffset = -35;
    native.config.visConfig.fixedHeight =
      native.config.visConfig.fixedElevation =
      native.config.visConfig.invertHeight =
        true;
    const changed = plan(native, options);
    expect(changed.channels.height).toEqual(first.channels.height);
    expect(changed.channels.elevationOffset).toEqual({
      kind: 'property',
      property: 'elevationOffset',
      source: property,
      fallback: -35
    });
    expect(changed.render).toMatchObject({
      elevationScale: 1,
      elevationOffsetActive: false,
      extruded: true
    });
    // Embedded offsets (including valid zero and negative values) remain active in flat polygons.
    native.config.visConfig.enable3d = false;
    const flat = plan(native, options);
    expect(flat.channels.height).toEqual({kind: 'constant', value: 0});
    expect(flat.channels.elevationOffset).toEqual(changed.channels.elevationOffset);
    expect(flat.render).toMatchObject({
      elevationOffsetActive: false,
      extruded: false,
      stroked: true
    });
  });

  test('offset-only extrusion forces zero height without resolving unused height properties or fields', () => {
    const native = layer('geojson');
    native.config.heightField = {name: 'unsupported-height'};
    native.config.visConfig.elevationOffset = 4;
    const translated = plan(native, {
      ...polygonContext,
      geojsonProperties: {
        ...polygonContext.geojsonProperties,
        get elevation(): never {
          throw new Error('Unused height must not resolve');
        }
      }
    });
    expect(translated.channels.height).toEqual({kind: 'constant', value: 0});
    expect(translated.channels.elevationOffset).toEqual({kind: 'constant', value: 4});
    expect(translated.render).toMatchObject({
      elevationOffsetActive: true,
      extruded: true,
      stroked: false
    });
    native.config.visConfig.elevationOffset = -4;
    expect(plan(native, polygonContext).render).toMatchObject({
      elevationOffsetActive: false,
      extruded: false,
      stroked: true
    });
    native.config.visConfig.elevationOffset = 0;
    native.config.elevationOffsetField = {name: 'value'};
    expect(plan(native, polygonContext).render).toMatchObject({
      elevationOffsetActive: true,
      extruded: true
    });
  });

  test('inverts output ranges and fixed identity around domain endpoints instead of negating height', () => {
    const native = layer('geojson');
    native.config.heightField = native.config.elevationOffsetField = {name: 'value'};
    native.config.heightDomain = [10, 30];
    native.config.elevationOffsetDomain = [-10, 40];
    native.config.heightScale = native.config.elevationOffsetScale = 'sqrt';
    native.config.visConfig.enable3d = true;
    native.config.visConfig.heightRange = [100, 800];
    native.config.visConfig.invertHeight = true;
    expect(resident(plan(native, polygonContext), 'height')).toMatchObject({
      scale: 'sqrt',
      range: [800, 100]
    });
    native.config.visConfig.fixedHeight = true;
    const fixed = plan(native, polygonContext);
    expect(resident(fixed, 'height')).toMatchObject({
      scale: 'identity',
      invert: true,
      domain: {kind: 'fixed', values: [10, 30]},
      nullValue: 0
    });
    expect(resident(fixed, 'elevationOffset')).toMatchObject({scale: 'sqrt'});
    expect(fixed.render.elevationScale).toBe(1);
    native.config.visConfig.fixedElevation = true;
    expect(resident(plan(native, polygonContext), 'elevationOffset')).toMatchObject({
      scale: 'identity',
      domain: {kind: 'fixed', values: [-10, 40]}
    });
    expect(resident(plan(native, polygonContext), 'elevationOffset')).not.toHaveProperty('invert');
    native.config.heightDomain = [7, 7];
    expect(resident(plan(native, polygonContext), 'height')).toMatchObject({
      scale: 'identity',
      invert: true,
      domain: {kind: 'fixed', values: [7, 7]}
    });
    expect(
      resident(
        plan(native, {...polygonContext, domains: {height: {kind: 'selection', revision: 9}}}),
        'height'
      )
    ).toMatchObject({scale: 'identity', invert: true, domain: {kind: 'selection', revision: 9}});
  });

  test('does not request vertical source metadata for lines, points, or disabled polygon fills', () => {
    const native = layer('geojson');
    native.config.visConfig.enable3d = true;
    native.config.heightField = native.config.elevationOffsetField = {name: 'unavailable'};
    const nonPolygon = plan(native, {...polygonContext, geojsonPolygonFeatures: false});
    expect(nonPolygon.channels.height).toBeUndefined();
    expect(nonPolygon.channels.elevationOffset).toBeUndefined();
    native.config.visConfig.filled = false;
    expect(plan(native, polygonContext).channels.height).toBeUndefined();
    expect(plan(native, polygonContext).channels.elevationOffset).toBeUndefined();
    native.config.visConfig.filled = true;
    const failure = createNativeStylePlan(native, polygonContext);
    expect(failure).toMatchObject({
      supported: false,
      reasons: [expect.stringContaining('height:'), expect.stringContaining('elevationOffset:')]
    });
  });
});
