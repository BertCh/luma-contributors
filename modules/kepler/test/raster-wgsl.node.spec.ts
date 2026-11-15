// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WgslReflect} from 'wgsl_reflect';
import {color, project32, phongMaterial, WebMercatorViewport} from '@deck.gl/core';
import {WGSLShaderAssembler} from '@luma.gl/shaders';
import {describe, expect, it} from 'vitest';
import {
  mergeRasterZoneColumns,
  RASTER_ZONE_NONE,
  rasterizeZoneIds
} from '../../../examples/deck/kepler-compute/src/raster-gpu-zonal-zones';
import {
  CompensatedSum,
  estimateRasterZoneSumErrorBound,
  getRasterZonalShift,
  shiftValuesToFloat32,
  unshiftRasterZoneColumns
} from '../../../examples/deck/kepler-compute/src/raster-gpu-zonal-precision';
import {createXyzTileSource} from '../../../examples/deck/kepler-compute/src/raster-tile-ingest';
import {computeRasterZonalStatistics} from '../../../examples/deck/kepler-compute/src/raster-zonal';
import {
  buildRasterTileWgsl,
  createRasterTileMesh,
  createRasterTerrainMesh,
  getRasterRescaleFlags,
  planRasterRender,
  planRasterRescale,
  RASTER_WGSL_MODE_CODES,
  RASTER_WGSL_MODULE_SOURCE,
  type RasterWgslSubLayerProps
} from '../../../examples/deck/kepler-compute/src/raster-wgsl-shader';
import {
  expandWmsTimeExtent,
  findWmsTimeExtentText,
  parseWmsExtentInstants
} from '../../../examples/deck/kepler-compute/src/wms-time-extent';
import {parseWmsTimeExtent} from '../../../examples/deck/kepler-compute/src/wms-identify';

const baseProps: RasterWgslSubLayerProps = {
  imageBandCount: 3,
  hasColormap: true,
  hasRgba: false,
  hasMask: false,
  minPixelValue: 0,
  maxPixelValue: 1000,
  linearRescalingFactor: [0, 1]
};

describe('raster WGSL plan and shader', () => {
  it('plans single band colormap, rgb and index presets', () => {
    expect(planRasterRender({...baseProps, bandCombination: 'single'})).toMatchObject({
      mode: 'colormap',
      expression: 'b1',
      usedBandMask: 1,
      domain: [0, 1000]
    });
    expect(planRasterRender({...baseProps, bandCombination: 'rgb'})).toMatchObject({
      mode: 'rgb',
      usedBandMask: 7
    });
    const index = planRasterRender({
      ...baseProps,
      bandCombination: 'normalizedDifference',
      renderBandIndexes: [2, 0]
    });
    expect(index).toMatchObject({
      mode: 'colormap',
      expression: '((b1 - b2) / (b1 + b2))',
      usedBandMask: 3,
      isSignedColormap: true,
      domain: [-1, 1]
    });
    expect(index?.bandPositions.slice(0, 2)).toEqual([2, 0]);
  });

  it('keeps Kepler drawing combinations it cannot express', () => {
    expect(planRasterRender({...baseProps, bandCombination: 'unknownPreset'})).toBeNull();
  });

  it('plans plain RGBA tiles without bands', () => {
    expect(
      planRasterRender({...baseProps, imageBandCount: 0, hasRgba: true, hasColormap: false})
    ).toMatchObject({mode: 'rgba'});
  });

  it('generates parsable WGSL that embeds the band math expression', () => {
    const plan = planRasterRender({...baseProps, bandCombination: 'enhancedVegetationIndex'});
    const source = buildRasterTileWgsl(plan?.expression);
    expect(source).toContain('let result = ((2.5 *');
    const reflection = new WgslReflect(
      `${RASTER_WGSL_MODULE_SOURCE}\n${source}`.replace(
        /project_position_to_clipspace_and_commonspace\([^;]*;/,
        'vec4<f32>(0.0);'
      )
    );
    expect(reflection.entry.fragment.map(entry => entry.name)).toContain('fragmentMain');
  });

  it('tessellates tiles with Mercator-linear texture rows', () => {
    const mesh = createRasterTileMesh([0, 0, 10, 60], 4);
    expect(mesh.vertexCount).toBe(24);
    const rowBoundaries = [
      ...new Set(Array.from(mesh.texCoords.filter((_, index) => index % 2 === 1)))
    ];
    expect(rowBoundaries[0]).toBe(0);
    expect(rowBoundaries.at(-1)).toBeCloseTo(1, 6);
    // Mercator stretches the north, so the latitude midpoint lies below the pixel midpoint.
    const middleLatitudeVertex = Array.from({length: mesh.vertexCount}, (_, i) => i).find(
      i => mesh.positions[i * 3 + 1] === 30
    ) as number;
    expect(mesh.texCoords[middleLatitudeVertex * 2 + 1]).toBeGreaterThan(0.5);
  });
});

describe('raster WGSL Kepler rescale parity', () => {
  it('uses Kepler linear rescale value * scaler + offset', () => {
    const rescale = planRasterRescale({
      ...baseProps,
      bandCombination: 'single',
      minPixelValue: 0,
      maxPixelValue: 1000,
      linearRescalingFactor: [0.2, 0.8]
    });
    expect(rescale.isEnabled).toBe(true);
    expect(rescale.scaler).toBeCloseTo(1 / (0.6 * 1000), 12);
    expect(rescale.offset).toBeCloseTo(-0.2, 12);
    expect(getRasterRescaleFlags(rescale)).toBe(1);
  });

  it('uses 1 / maxPixelValue with gamma, sigmoidal contrast and saturation when non-linear', () => {
    const rescale = planRasterRescale({
      ...baseProps,
      bandCombination: 'rgb',
      nonLinearRescaling: true,
      gammaContrastFactor: 2,
      sigmoidalContrastFactor: 5,
      sigmoidalBiasFactor: 0.4,
      saturationValue: 1.5
    });
    expect(rescale).toMatchObject({
      scaler: 1 / 1000,
      offset: 0,
      gamma: 2,
      sigmoidalContrast: 5,
      sigmoidalBias: 0.4,
      saturation: 1.5
    });
    expect(getRasterRescaleFlags(rescale)).toBe(1 | 2 | 4 | 8);
  });

  it('does not rescale index combinations and skips neutral saturation', () => {
    const rescale = planRasterRescale({
      ...baseProps,
      bandCombination: 'normalizedDifference',
      saturationValue: 2
    });
    expect(rescale.isEnabled).toBe(false);
    expect(rescale.saturation).toBe(1);
    expect(
      planRasterRescale({...baseProps, bandCombination: 'single', saturationValue: 1}).saturation
    ).toBe(1);
  });

  it('steps categorical colormaps with integer maxPixelValue over the category range', () => {
    const rescale = planRasterRescale({
      ...baseProps,
      bandCombination: 'single',
      hasCategoricalColorMap: true,
      dataType: 'uint8',
      maxPixelValue: 255,
      minCategoricalBandValue: 1,
      maxCategoricalBandValue: 11
    });
    expect(rescale).toMatchObject({isEnabled: true, isCategorical: true, scaler: 1 / 255});
    expect(rescale.categoricalStep).toBeCloseTo(25.5, 6);
    expect(getRasterRescaleFlags(rescale) & 16).toBe(16);
    // Float data has no finite dtype maximum, so Kepler does not step categorically.
    expect(
      planRasterRescale({
        ...baseProps,
        hasCategoricalColorMap: true,
        dataType: 'float32',
        minCategoricalBandValue: 0,
        maxCategoricalBandValue: 9
      }).isCategorical
    ).toBe(false);
  });

  it('does not read identity-filled bands for saturation in single mode', () => {
    const plan = planRasterRender({
      ...baseProps,
      bandCombination: 'single',
      renderBandIndexes: [2],
      saturationValue: 0.5
    });
    expect(plan?.bandPositions).toEqual([2, 0, 0, 0]);
    expect(plan?.rescale.saturation).toBe(1);
    expect(plan?.usedBandMask).toBe(1);
  });

  it('fills identity bands for rgb saturation like the Kepler reorder matrix', () => {
    const plan = planRasterRender({
      ...baseProps,
      imageBandCount: 4,
      bandCombination: 'rgb',
      renderBandIndexes: [2],
      saturationValue: 0.5
    });
    expect(plan?.bandPositions.slice(0, 3)).toEqual([2, 1, 2]);
    expect(plan?.rescale.saturation).toBe(0.5);
  });

  it('draws spectral indices without a colormap as the red channel like Kepler', () => {
    const plan = planRasterRender({
      ...baseProps,
      hasColormap: false,
      bandCombination: 'normalizedDifference'
    });
    expect(plan?.mode).toBe('index');
    expect(RASTER_WGSL_MODE_CODES.index).toBe(4);
    expect(buildRasterTileWgsl(plan?.expression)).toContain('mode == 4');
    // single stays gray; an index with a colormap still colormaps
    expect(
      planRasterRender({...baseProps, hasColormap: false, bandCombination: 'single'})?.mode
    ).toBe('grayscale');
    expect(planRasterRender({...baseProps, bandCombination: 'normalizedDifference'})?.mode).toBe(
      'colormap'
    );
  });

  it('plans custom band math over up to eight bands', () => {
    const plan = planRasterRender({
      ...baseProps,
      imageBandCount: 8,
      bandCombination: 'custom',
      bandMathExpression: '(b5 - b8) / (b5 + b8 + b1)'
    });
    expect(plan?.bandPositions).toHaveLength(8);
    expect(plan?.bandPositions[4]).toBe(4);
    expect(plan?.bandPositions[7]).toBe(7);
    // unused extended slots stay on position 0 so they are not made resident
    expect(plan?.bandPositions[5]).toBe(0);
    expect(plan?.usedBandMask).toBe(0b10010001);
    expect(plan?.mode).toBe('colormap');
    const source = buildRasterTileWgsl(plan?.expression);
    expect(source).toContain('let b5 = rescaleChannel(loadValue(valueTexture5, uv));');
    expect(source).toContain('let b8 = rescaleChannel(loadValue(valueTexture8, uv));');
    expect(source).not.toContain('let b6 =');
    const reflection = new WgslReflect(`${RASTER_WGSL_MODULE_SOURCE}\n${source}`);
    expect(reflection.entry.fragment).toHaveLength(1);
  });

  it('treats an unknown preset that parses as band math as custom math', () => {
    const plan = planRasterRender({
      ...baseProps,
      hasColormap: false,
      bandCombination: 'b1 * 2 - b2'
    });
    expect(plan?.mode).toBe('grayscale');
    expect(plan?.usedBandMask).toBe(3);
  });

  it('refuses expressions past b8 and unparsable expressions', () => {
    expect(
      planRasterRender({...baseProps, bandCombination: 'custom', bandMathExpression: 'b9 + b1'})
    ).toBeNull();
    expect(
      planRasterRender({...baseProps, bandCombination: 'custom', bandMathExpression: 'b1 +'})
    ).toBeNull();
  });

  it('emits parsable WGSL with the rescale, saturation and colormap functions', () => {
    const source = buildRasterTileWgsl('b1');
    for (const name of ['rescaleChannel', 'saturateColor', 'sampleColormap'])
      expect(source).toContain(`fn ${name}`);
    const reflection = new WgslReflect(
      `${RASTER_WGSL_MODULE_SOURCE}\n${source}`.replace(
        /project_position_to_clipspace_and_commonspace\([^;]*;/,
        'vec4<f32>(0.0);'
      )
    );
    expect(reflection.entry.fragment.map(entry => entry.name)).toContain('fragmentMain');
  });
});

describe('GPU zonal zone rasterization', () => {
  const source = createXyzTileSource('zones', [
    {
      index: {x: 1, y: 1, z: 1},
      width: 8,
      height: 8,
      bands: [
        {
          values: Float32Array.from({length: 64}, (_, index) => (index % 7 === 0 ? -9999 : index)),
          noDataValue: -9999
        }
      ]
    }
  ]);
  const squareA = [
    [
      [10, -50],
      [60, -50],
      [60, -10],
      [10, -10],
      [10, -50]
    ]
  ] as const;
  const squareB = [
    [
      [100, -70],
      [150, -70],
      [150, -30],
      [100, -30],
      [100, -70]
    ]
  ] as const;

  it('reproduces the CPU reference when merged from per-cell zones', () => {
    const tile = source.tiles[0];
    const zonePolygons = [[squareA], [squareB]] as const;
    const raster = rasterizeZoneIds(tile, source.crs, zonePolygons);
    expect(raster.hasOverlap).toBe(false);
    // Emulate the GPU reduction on the CPU from the zone raster.
    const band = tile.bands[0];
    const columns = {
      cellCounts: [0, 0],
      valueCounts: [0, 0],
      sums: [0, 0],
      minimums: [Infinity, Infinity],
      maximums: [-Infinity, -Infinity]
    };
    raster.zones.forEach((zone, cell) => {
      if (zone === RASTER_ZONE_NONE) return;
      columns.cellCounts[zone]++;
      if (!band.validity[cell]) return;
      const value = band.values[cell];
      columns.valueCounts[zone]++;
      columns.sums[zone] += value;
      columns.minimums[zone] = Math.min(columns.minimums[zone], value);
      columns.maximums[zone] = Math.max(columns.maximums[zone], value);
    });
    const merged = mergeRasterZoneColumns(2, [columns]);
    zonePolygons.forEach((polygons, zoneIndex) => {
      const reference = computeRasterZonalStatistics(source, polygons);
      expect(merged[zoneIndex]).toMatchObject({
        count: reference.count,
        sum: reference.sum,
        minimum: reference.minimum,
        maximum: reference.maximum,
        nodataCount: reference.nodataCount,
        scannedCount: reference.scannedCount
      });
    });
  });

  it('flags overlapping zones so the caller falls back to the CPU path', () => {
    const raster = rasterizeZoneIds(source.tiles[0], source.crs, [[squareA], [squareA]]);
    expect(raster.hasOverlap).toBe(true);
  });
});

describe('WMS time extent', () => {
  const capabilities = `<Capability><Layer><Name>root</Name>
    <Layer><Name>sst</Name><Dimension name="time" units="ISO8601" default="2024-03-01">2024-01-01/2024-03-01/P1M</Dimension></Layer>
    <Layer><Name>plain</Name></Layer></Layer></Capability>`;

  it('finds a layer dimension without leaking into siblings', () => {
    expect(findWmsTimeExtentText(capabilities, 'sst')).toEqual({
      extentText: '2024-01-01/2024-03-01/P1M',
      defaultValue: '2024-03-01'
    });
    expect(findWmsTimeExtentText(capabilities, 'plain')).toBeUndefined();
  });

  it('expands periods, lists and sorts instants', () => {
    expect(parseWmsExtentInstants('2024-01-01/2024-03-01/P1M')).toEqual([
      '2024-01-01T00:00:00Z',
      '2024-02-01T00:00:00Z',
      '2024-03-01T00:00:00Z'
    ]);
    expect(
      expandWmsTimeExtent(parseWmsTimeExtent('2024-05-01T00:00:00Z,2024-04-01T00:00:00Z'))
    ).toEqual(['2024-04-01T00:00:00Z', '2024-05-01T00:00:00Z']);
  });
});

describe('raster WGSL wrapper', () => {
  it('assembles raster textures alongside published phong lighting without binding collisions', () => {
    const shader = new WGSLShaderAssembler().assembleWGSLShader({
      platformInfo: {
        type: 'webgpu',
        shaderLanguage: 'wgsl',
        shaderLanguageVersion: 300,
        gpu: 'test',
        features: new Set<string>()
      },
      source: buildRasterTileWgsl('b1', true, true),
      modules: [
        project32,
        color,
        phongMaterial,
        {name: 'rasterWgsl', source: RASTER_WGSL_MODULE_SOURCE}
      ]
    });
    const reflection = new WgslReflect(shader.source);
    const bindings = [...reflection.uniforms, ...reflection.textures, ...reflection.samplers];
    const slots = bindings.map(binding => `${binding.group}/${binding.binding}`);
    expect(new Set(slots).size).toBe(slots.length);
    expect(reflection.textures).toHaveLength(14);
  });
  it('preserves indexed terrain positions, UVs and smooth normals', () => {
    const mesh = createRasterTerrainMesh({
      attributes: {
        POSITION: {value: [0, 0, 10, 1, 0, 20, 0, 1, 30]},
        TEXCOORD_0: {value: [0, 1, 1, 1, 0, 0]},
        NORMAL: {value: [0, 0, 1, 0, 1, 0, 1, 0, 0]}
      },
      indices: {value: [2, 0, 1]}
    });
    expect(Array.from(mesh.positions)).toEqual([0, 1, 30, 0, 0, 10, 1, 0, 20]);
    expect(Array.from(mesh.texCoords)).toEqual([0, 0, 0, 1, 1, 1]);
    expect(Array.from(mesh.normals)).toEqual([1, 0, 0, 0, 0, 1, 0, 1, 0]);
    new WgslReflect(buildRasterTileWgsl('b1', true, true));
  });
  it('uses Kepler global range overrides before tile statistics', async () => {
    const {getRasterSubLayerPlanProps} = await import(
      '../../../examples/deck/kepler-compute/src/raster-wgsl-wrap'
    );
    const images = {imageBands: [{data: new Float32Array([50]), width: 1, height: 1}]};
    const argumentsWithRange = {
      bandCombination: 'single',
      data: {minPixelValue: 0, maxPixelValue: 100},
      minPixelValue: NaN,
      maxPixelValue: undefined
    };
    const tilePlan = planRasterRender(getRasterSubLayerPlanProps(argumentsWithRange, images))!;
    expect(tilePlan.domain).toEqual([0, 100]);
    expect(tilePlan.rescale.scaler).toBe(0.01);
    const override = getRasterSubLayerPlanProps(
      {...argumentsWithRange, minPixelValue: 0, maxPixelValue: 200},
      images
    );
    expect(planRasterRender(override)!.domain).toEqual([0, 200]);
    expect(planRasterRender(override)!.rescale.scaler).toBe(0.005);
    expect(
      planRasterRender(
        getRasterSubLayerPlanProps(
          {...argumentsWithRange, minPixelValue: -100, maxPixelValue: 200},
          images
        )
      )!.domain
    ).toEqual([-100, 200]);
    const otherTile = getRasterSubLayerPlanProps(
      {
        ...argumentsWithRange,
        minPixelValue: 0,
        maxPixelValue: 200,
        data: {minPixelValue: 20, maxPixelValue: 50}
      },
      images
    );
    expect(planRasterRender(otherTile)!.rescale).toEqual(planRasterRender(override)!.rescale);
    expect(
      planRasterRender(
        getRasterSubLayerPlanProps(
          {...argumentsWithRange, minPixelValue: -Infinity, maxPixelValue: Infinity},
          images
        )
      )!.domain
    ).toEqual([0, 100]);
  });

  it('tessellates east-west globe edges as well as north-south edges', () => {
    const mesh = createRasterTileMesh([-180, -80, 180, 80], 32, 72);
    expect(mesh.vertexCount).toBe(32 * 72 * 6);
    for (let vertex = 0; vertex < mesh.vertexCount; vertex += 3) {
      const longitudes = [0, 1, 2].map(offset => mesh.positions[(vertex + offset) * 3]);
      expect(Math.max(...longitudes) - Math.min(...longitudes)).toBeLessThanOrEqual(5);
    }
    expect(buildRasterTileWgsl()).toContain('gpuGlobeIsCommonHidden(varyings.globePosition)');
  });

  it('replaces RasterLayer sublayers, keeps others, and stays idempotent', async () => {
    const {RasterWgslWrapper} = await import(
      '../../../examples/deck/kepler-compute/src/raster-wgsl-wrap'
    );
    const requested: unknown[] = [];
    const controller = {
      ensureTileBandsResident: (...args: unknown[]) => requested.push(args),
      peekTileBandTextures: () => undefined
    };
    const wrapper = new RasterWgslWrapper(controller as never);
    const makeSublayer = (layerName: string) => ({
      id: `${layerName}-1`,
      constructor: {layerName},
      props: {
        id: `${layerName}-1`,
        bounds: [0, 0, 1, 1],
        opacity: 1,
        // Match the inherited public Deck defaults of a constructed native raster sublayer.
        visible: true,
        images: {
          imageBands: [{data: new Float32Array(4), width: 2, height: 2}],
          imageColormap: {data: new Uint8Array(8), width: 2, height: 1}
        }
      },
      clone: () => undefined
    });
    const tileLayer = {
      id: 'kepler-raster',
      constructor: {layerName: 'TileLayer'},
      props: {
        renderSubLayers: () => [makeSublayer('RasterLayer'), makeSublayer('PathLayer')]
      },
      clone(props: Record<string, unknown>) {
        return {...this, props: {...this.props, ...props}};
      }
    };
    const wrapped = wrapper.wrapLayer(tileLayer);
    expect(wrapped).not.toBe(tileLayer);
    const rendered = (wrapped.props.renderSubLayers as (props: unknown) => unknown[])({
      bandCombination: 'single',
      minPixelValue: 0,
      maxPixelValue: 10,
      tile: {index: {x: 1, y: 2, z: 3}}
    });
    expect((rendered[0] as {constructor: {layerName: string}}).constructor.layerName).toBe(
      'RasterWgslLayer'
    );
    expect((rendered[1] as {constructor: {layerName: string}}).constructor.layerName).toBe(
      'PathLayer'
    );
    expect(requested).toHaveLength(1);
    expect(wrapper.statistics.replacedCount).toBe(1);
    // A second wrap of the already wrapped layer must not double wrap.
    expect(wrapper.wrapLayer(wrapped)).toBe(wrapped);
    const secondTileLayer = wrapper.wrapLayer({...tileLayer, id: 'second-raster'});
    (secondTileLayer.props.renderSubLayers as (props: object) => unknown)({
      bandCombination: 'single',
      tile: {index: {x: 1, y: 2, z: 3}}
    });
    expect(requested[1]).toEqual(['second-raster', {x: 1, y: 2, z: 3}, [0, 0, 0, 0]]);
    // Deck renderSubLayers also accepts a single layer or nested arrays. These must never reach
    // native RasterLayer.initializeState, which calls the removed luma 9 ShaderAssembler API.
    for (const result of [
      makeSublayer('RasterLayer'),
      makeSublayer('RasterMeshLayer'),
      [[makeSublayer('RasterLayer')], null]
    ]) {
      const direct = wrapper.wrapLayer({
        ...tileLayer,
        props: {renderSubLayers: () => result}
      });
      const replaced = (direct.props.renderSubLayers as (props: object) => unknown)({
        bandCombination: 'single'
      });
      const layers = [replaced].flat(Infinity).filter(Boolean);
      expect(layers).toHaveLength(1);
      expect((layers[0] as {constructor: {layerName: string}}).constructor.layerName).toBe(
        'RasterWgslLayer'
      );
    }
    const terrain = makeSublayer('RasterMeshLayer');
    const mesh = {
      attributes: {
        POSITION: {value: [0, 0, 1, 1, 0, 1, 0, 1, 1]},
        TEXCOORD_0: {value: [0, 0, 1, 0, 0, 1]}
      }
    };
    const terrainTile = wrapper.wrapLayer({
      ...tileLayer,
      props: {
        renderSubLayers: () => ({...terrain, props: {...terrain.props, bounds: undefined, mesh}})
      }
    });
    const renderedTerrain = (terrainTile.props.renderSubLayers as (props: object) => unknown)({
      bandCombination: 'single',
      tile: {bbox: {west: 0, south: 0, east: 1, north: 1}}
    }) as import('../../../examples/deck/kepler-compute/src/raster-wgsl-layer').RasterWgslLayer;
    expect(renderedTerrain.props.mesh).toBe(mesh);
    expect(renderedTerrain.props.bounds).toEqual([0, 0, 1, 1]);
    expect(
      renderedTerrain.props.visible,
      'native terrain fixture must carry the inherited Deck visible default'
    ).toBe(true);
    // Native quantized-mesh output has no NORMAL. Native RasterMeshLayer derives face normals
    // in that case; inspect the replacement's actual shader, not a manually flat-shaded builder.
    Object.assign(renderedTerrain, {
      context: {
        viewport: new WebMercatorViewport({width: 128, height: 128}),
        defaultShaderModules: [],
        defaultShaderPlugins: []
      }
    });
    const shaders = renderedTerrain.getShaders();
    expect(shaders.source).toContain(
      'normalize(cross(dpdy(input.globePosition), dpdx(input.globePosition)))'
    );
    expect(shaders.modules).toContain(phongMaterial);
    const smoothMesh = {
      ...mesh,
      attributes: {...mesh.attributes, NORMAL: {value: [0, 0, 1, 0, 0, 1, 0, 0, 1]}}
    };
    const smooth = renderedTerrain.clone({mesh: smoothMesh});
    Object.assign(smooth, {context: renderedTerrain.context});
    expect(smooth.getShaders().source).not.toContain('cross(dpdy(');
    const explicitFlat = smooth.clone({flatShading: true});
    Object.assign(explicitFlat, {context: renderedTerrain.context});
    expect(explicitFlat.getShaders().source).toContain('cross(dpdy(');
    const hiddenTile = wrapper.wrapLayer({
      ...tileLayer,
      props: {renderSubLayers: () => ({...terrain, props: {...terrain.props, visible: false}})}
    });
    const hiddenTerrain = (hiddenTile.props.renderSubLayers as (props: object) => unknown)(
      {}
    ) as typeof renderedTerrain;
    expect(hiddenTerrain.props.visible, 'intentional native visibility is preserved').toBe(false);
  });
});

describe('GPU zonal precision safeguards', () => {
  it('compensated addition keeps small terms next to a large one', () => {
    const accumulator = new CompensatedSum();
    accumulator.add(1e16);
    for (let index = 0; index < 1000; index++) accumulator.add(1);
    accumulator.add(-1e16);
    expect(accumulator.value).toBe(1000);
    expect(1e16 + 1 - 1e16).not.toBe(1);
  });

  it('centering turns a float32 sum with a large offset into an accurate one', () => {
    const count = 4096;
    const values = Float64Array.from({length: count}, (_, index) => 123456789 + (index % 7));
    const validity = new Uint8Array(count).fill(1);
    const exact = values.reduce((total, value) => total + value, 0);
    const naive = Float32Array.from(values).reduce((total, value) => Math.fround(total + value), 0);
    const shift = getRasterZonalShift(values, validity);
    const centered = shiftValuesToFloat32(values, shift).reduce(
      (total, value) => Math.fround(total + value),
      0
    );
    const restored = unshiftRasterZoneColumns(
      {
        cellCounts: [count],
        valueCounts: [count],
        sums: [centered],
        minimums: [-3],
        maximums: [3]
      },
      shift
    );
    expect(Math.abs(restored.sums[0] - exact)).toBeLessThan(Math.abs(naive - exact) / 100);
    expect(restored.minimums[0]).toBeCloseTo(shift - 3, 6);
  });

  it('ignores invalid samples when choosing the shift and keeps zero shifts untouched', () => {
    expect(getRasterZonalShift([5, 1000, 7], [1, 0, 1])).toBe(6);
    expect(getRasterZonalShift([NaN], [1])).toBe(0);
    const columns = {
      cellCounts: [1],
      valueCounts: [0],
      sums: [0],
      minimums: [Infinity],
      maximums: [-Infinity]
    };
    expect(unshiftRasterZoneColumns(columns, 0)).toBe(columns);
    expect(unshiftRasterZoneColumns(columns, 5).minimums[0]).toBe(Infinity);
  });

  it('reports a worst-case error bound on merged statistics', () => {
    const merged = mergeRasterZoneColumns(1, [
      {cellCounts: [100], valueCounts: [100], sums: [500], minimums: [1], maximums: [9]}
    ]);
    expect(merged[0]?.sumErrorBound).toBeCloseTo(estimateRasterZoneSumErrorBound(100, 9), 12);
  });
});
