// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Density surfaces, kriging and interpolation, point processes, sampling, Markov dynamics, region
// statistics and regionalization.

import type {CatalogEntry} from '../schema';

const ANALYSIS = '@luma-contributors/analysis/gpu-spatial-analysis';
const RASTER = '@luma-contributors/analysis/raster-analysis';
const GRAPH = '@luma.gl/graph/gpu-core';

const F32 = "GraphDataView<'float32'>";
const F32X2 = "GraphDataView<'float32x2'>";
const F32X4 = "GraphDataView<'float32x4'>";
const U32 = "GraphDataView<'uint32'>";
const U32X4 = "GraphDataView<'uint32x4'>";
const WEIGHTS = 'GPUSpatialWeights';
const GRID = 'GPURasterMetadata';
const INDEX_SIZE = 'readonly [number, number]';
const INDEX_BOUNDS = 'readonly [number, number, number, number]';

export const entries: CatalogEntry[] = [
  // ---------------------------------------------------------------- density surfaces
  {
    id: 'point-density',
    symbol: 'GPUPointDensity',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Point density',
    summary: 'Bins points into square or hexagon cells and writes a count, sum or mean raster.',
    explanation:
      'Use it for fast heat maps of millions of points. This is binned counting, not a kernel estimate, so values change with `gridSize`. Optional `smoothing` convolves the field in cell units.',
    dataTypes: {input: ['points'], output: ['raster']},
    useCases: ['summarize', 'find-patterns'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      {
        name: 'positions',
        type: 'GPUFloat32Positions',
        required: true,
        description: 'Point locations.'
      },
      {
        name: 'bounds',
        type: 'GPUPointDensityBounds',
        required: true,
        description: 'Viewport [minX, minY, maxX, maxY], static or GPU view.'
      },
      {
        name: 'gridSize',
        type: INDEX_SIZE,
        required: true,
        description: 'Columns and rows of the output grid.'
      },
      {name: 'binning', type: "'grid' | 'hexagon'", description: 'Cell shape.', default: "'grid'"},
      {
        name: 'statistic',
        type: "'count' | 'sum' | 'mean'",
        description: 'What each cell holds.',
        default: "'count'"
      },
      {name: 'weights', type: F32, description: 'Per-point weight for sum and mean.'},
      {
        name: 'smoothing',
        type: 'GPUPointDensitySmoothing',
        description: 'Convolution kernel applied to the binned field.'
      },
      {
        name: 'output',
        type: 'GPUPointDensityOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'values', type: F32, description: 'Count, sum or mean per cell.'},
      {name: 'extent', type: F32, description: 'Field [min, max] for color scaling.'},
      {name: 'histogram', type: U32, description: 'Distribution of cell values.'},
      {
        name: 'texture',
        type: "GraphTextureView<'r32float'>",
        description: 'Optional field as a texture.'
      }
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUPointDensity} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'heat'});
// positions, values, extent: graph views over caller-owned buffers
graph.add(
  new GPUPointDensity({
    positions,
    bounds: [minX, minY, maxX, maxY],
    gridSize: [256, 256],
    statistic: 'count',
    output: {values, extent}
  })
);
const compiled = graph.compile();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'point-clusters',
      seed: 11,
      caption: 'Points binned into cells; brighter means more.'
    },
    related: ['kernel-density', 'line-density', 'dot-density'],
    lensScenario: 'gpu-spatial-analysis.point-density',
    reference: 'api/gpu-spatial-analysis/operators-density',
    maturity: 'stable'
  },
  {
    id: 'kernel-density',
    symbol: 'GPUKernelDensity',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Kernel density',
    summary:
      'Estimates event intensity on a raster with a planar kernel, a ground-unit bandwidth and edge correction.',
    explanation:
      'Use it when you need intensity per unit area, not counts. Each cell sums `K_h(distance)` over nearby points; pick gaussian, quartic or epanechnikov. Bandwidth and window are per-frame parameters, so changing them never recompiles.',
    dataTypes: {input: ['points'], output: ['raster']},
    useCases: ['find-patterns', 'interpolate'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {
        name: 'positions',
        type: F32X2,
        required: true,
        description: 'Planar points with equal axis units.'
      },
      {name: 'grid', type: GRID, required: true, description: 'Axis-aligned output raster.'},
      {
        name: 'parameters',
        type: F32,
        required: true,
        description: 'Bandwidth and window, from getGPUKernelDensityParameterValues.'
      },
      {
        name: 'indexGridSize',
        type: INDEX_SIZE,
        required: true,
        description: 'Internal point index resolution.'
      },
      {
        name: 'indexBounds',
        type: INDEX_BOUNDS,
        required: true,
        description: 'Must cover every point that should count.'
      },
      {
        name: 'kernel',
        type: "'gaussian' | 'quartic' | 'epanechnikov'",
        description: 'Kernel shape.',
        default: "'gaussian'"
      },
      {
        name: 'edgeCorrection',
        type: "'none' | 'uniform' | 'diggle'",
        description: 'Compensates mass lost at the window edge.',
        default: "'uniform'"
      },
      {
        name: 'weights',
        type: F32,
        description: 'Optional per-point weight; NaN or negative excludes.'
      }
    ],
    outputs: [
      {name: 'values', type: F32, description: 'Intensity per cell.'},
      {name: 'weightSum', type: F32, description: 'Optional total point weight.'},
      {
        name: 'status',
        type: '{invalidCount?, insufficientPadding?}',
        description: 'Optional diagnostics.'
      }
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUKernelDensity, GPUParameterBuffer, GPU_KERNEL_DENSITY_PARAMETER_LENGTH,
  getGPUKernelDensityParameterValues} from '${ANALYSIS}';
import {makeGPURasterMetadata} from '${RASTER}';

const graph = new GPUCommandGraph(device, {id: 'kde'});
const grid = makeGPURasterMetadata({width: 128, height: 128, cellSize: 125, origin: [x0, y0]});
const kdeParameters = new GPUParameterBuffer(device, {
  id: 'kde-parameters', format: 'float32', length: GPU_KERNEL_DENSITY_PARAMETER_LENGTH
});
// positions, values: graph views over caller-owned buffers
graph.add(new GPUKernelDensity({
  positions, grid, parameters: kdeParameters.importToGraph(graph),
  indexGridSize: [64, 64], indexBounds: [x0, y0, x1, y1],
  kernel: 'quartic', output: {values}
}));
const compiled = graph.compile();
kdeParameters.write(getGPUKernelDensityParameterValues({bandwidth: 250}, grid)); // meters
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'density-surface',
      seed: 12,
      caption: 'Smooth intensity surface from scattered events.'
    },
    related: ['point-density', 'inhomogeneous-point-pattern', 'line-density'],
    lensScenario: 'gpu-spatial-analysis.kernel-density',
    reference: 'api/gpu-spatial-analysis/operators-density',
    maturity: 'stable'
  },
  {
    id: 'line-density',
    symbol: 'GPULineDensity',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Line density',
    summary:
      'Accumulates the length of line segments falling in each grid cell and divides by cell area.',
    explanation:
      'Use it for road, river or track density maps. Segments are clipped to cells exactly, so length is conserved. Grid origin and cell size are per-frame parameters.',
    dataTypes: {input: ['lines'], output: ['raster']},
    useCases: ['summarize', 'measure'],
    domains: ['urban', 'transport'],
    params: [
      {name: 'positions', type: F32X2, required: true, description: 'Packed line vertices.'},
      {name: 'lineOffsets', type: U32, required: true, description: 'Vertex offsets per line.'},
      {name: 'columns', type: 'number', required: true, description: 'Grid columns.'},
      {name: 'rows', type: 'number', required: true, description: 'Grid rows.'},
      {
        name: 'parameters',
        type: F32,
        required: true,
        description: 'minX, minY, cellWidth, cellHeight.'
      },
      {name: 'recordCapacity', type: 'number', description: 'Upper bound on segment-cell records.'},
      {
        name: 'spatialContext',
        type: 'GPUSpatialContext',
        description: 'Planar frame of the coordinates.'
      },
      {
        name: 'output',
        type: 'GPULineDensityOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'lengths', type: F32, description: 'Total line length per cell.'},
      {name: 'densities', type: F32, description: 'Optional length per unit area.'},
      {
        name: 'status',
        type: 'GPUOverflowStatusPort',
        description: 'Overflow flag and required capacity.'
      }
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPULineDensity, GPUParameterBuffer, GPU_LINE_DENSITY_PARAMETER_LENGTH,
  getGPULineDensityParameterValues} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'roads'});
const parameters = new GPUParameterBuffer(device, {
  id: 'line-density-parameters', format: 'float32', length: GPU_LINE_DENSITY_PARAMETER_LENGTH,
  values: getGPULineDensityParameterValues({minX, minY, cellWidth: 100, cellHeight: 100})
});
// positions, lineOffsets, lengths, densities, overflow: graph views
graph.add(new GPULineDensity({
  positions, lineOffsets, columns: 200, rows: 200,
  parameters: parameters.importToGraph(graph),
  output: {lengths, densities, status: {overflow}}
}));
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'lines', seed: 13, caption: 'Line length per cell, drawn as a heat grid.'},
    related: ['line-length-per-polygon', 'kernel-density', 'point-density'],
    lensScenario: 'gpu-spatial-analysis.line-density',
    reference: 'api/gpu-spatial-analysis/operators-geometry',
    maturity: 'stable'
  },
  {
    id: 'line-length-per-polygon',
    symbol: 'GPULineLengthPerPolygon',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Line length per polygon',
    summary: 'Sums the length of lines that fall inside each polygon, optionally weighted.',
    explanation:
      'Use it for road length per tract or river length per watershed. Each line is split at polygon edges and measured exactly. Weights give length-weighted attributes such as lanes.',
    dataTypes: {input: ['lines', 'polygons'], output: ['table']},
    useCases: ['measure', 'summarize'],
    domains: ['urban', 'transport'],
    params: [
      {name: 'positions', type: F32X2, required: true, description: 'Packed line vertices.'},
      {name: 'lineOffsets', type: U32, required: true, description: 'Vertex offsets per line.'},
      {
        name: 'polygons',
        type: 'GPUPackedGeometryPort<GPUPolygonGeometryPort>',
        required: true,
        description: 'Packed polygon geometry.'
      },
      {name: 'lineWeights', type: F32, description: 'Per-line weight for weightedLengths.'},
      {
        name: 'pairCapacity',
        type: 'number',
        description: 'Upper bound on line-polygon candidate pairs.'
      },
      {
        name: 'maximumCrossingCount',
        type: 'number',
        description: 'Upper bound on edge crossings per segment.'
      },
      {name: 'spatialContext', type: 'GPUSpatialContext', description: 'Frame of the lines.'},
      {
        name: 'output',
        type: 'GPULineLengthPerPolygonOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'lengths', type: F32, description: 'Line length per polygon.'},
      {name: 'weightedLengths', type: F32, description: 'Optional weight-multiplied length.'},
      {name: 'segmentCounts', type: U32, description: 'Optional segments counted per polygon.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPULineLengthPerPolygon} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'road-length'});
// roads: positions + lineOffsets; tracts: packed polygon geometry (graph views)
graph.add(
  new GPULineLengthPerPolygon({
    positions: roadPositions,
    lineOffsets: roadOffsets,
    polygons: {kind: 'polygons', positions: tractPositions,
      featureOffsets, polygonOffsets, ringOffsets},
    output: {lengths, status: {overflow}}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'polygon-overlay',
      seed: 14,
      caption: 'Lines clipped to polygons and measured.'
    },
    related: ['line-density', 'zonal-statistics'],
    lensScenario: 'gpu-spatial-analysis.line-length-per-polygon',
    reference: 'api/gpu-spatial-analysis/operators-geometry',
    maturity: 'stable'
  },
  {
    id: 'dot-density',
    symbol: 'GPUDotDensity',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Dot density',
    summary: 'Scatters one dot per unit of a polygon attribute, with a category per dot.',
    explanation:
      'Use it for dot maps of population or land use. Dots are placed uniformly inside each polygon by rejection sampling, seeded so results repeat. Output is compacted so it renders directly.',
    dataTypes: {input: ['polygons', 'table'], output: ['points']},
    useCases: ['transform', 'classify'],
    domains: ['demographics', 'cartography'],
    params: [
      {name: 'positions', type: F32X2, required: true, description: 'Packed polygon vertices.'},
      {
        name: 'featureOffsets',
        type: U32,
        required: true,
        description: 'Polygon offsets per feature.'
      },
      {name: 'polygonOffsets', type: U32, required: true, description: 'Ring offsets per polygon.'},
      {name: 'ringOffsets', type: U32, required: true, description: 'Vertex offsets per ring.'},
      {name: 'values', type: F32, required: true, description: 'Counts per category per feature.'},
      {name: 'categoryCount', type: 'number', description: 'Categories per feature.'},
      {name: 'parameters', type: U32, required: true, description: 'Seed and dots per unit.'},
      {
        name: 'output',
        type: 'GPUDotDensityOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'positions', type: F32X2, description: 'Dot locations.'},
      {name: 'dots', type: 'GPUCompactOutput', description: 'Compacted dot count.'},
      {name: 'categories', type: U32, description: 'Category index per dot.'},
      {name: 'status', type: '{failedCount?}', description: 'Dots that exhausted attempts.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUDotDensity, GPUParameterBuffer, GPU_DOT_DENSITY_PARAMETER_LENGTH,
  getGPUDotDensityParameterValues} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'dots'});
const parameters = new GPUParameterBuffer(device, {
  id: 'dot-parameters', format: 'uint32', length: GPU_DOT_DENSITY_PARAMETER_LENGTH,
  values: getGPUDotDensityParameterValues({seed: 7, dotsPerUnit: 1 / 100})
});
// polygon geometry views, per-tract counts, and outputs are graph views
graph.add(new GPUDotDensity({
  positions, featureOffsets, polygonOffsets, ringOffsets,
  values: populationByCategory, categoryCount: 3,
  parameters: parameters.importToGraph(graph),
  output: {positions: dotPositions, dots, categories}
}));
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'points',
      seed: 15,
      caption: 'One colored dot per 100 people, inside each tract.'
    },
    related: ['random-points-in-polygon', 'random-points-on-line', 'point-density'],
    lensScenario: 'gpu-spatial-analysis.dot-density',
    reference: 'api/gpu-spatial-analysis/operators-density',
    maturity: 'stable'
  },
  {
    id: 'random-points-in-polygon',
    symbol: 'GPURandomPointsInPolygon',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Random points in polygon',
    summary: 'Draws a requested number of uniform random points inside each polygon.',
    explanation:
      'Use it to seed simulations, null models or synthetic populations. Counts are per polygon; sampling is by seeded rejection, so reruns repeat. Output is a compact point list.',
    dataTypes: {input: ['polygons'], output: ['points']},
    useCases: ['simulate', 'transform'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {name: 'positions', type: F32X2, required: true, description: 'Packed polygon vertices.'},
      {
        name: 'featureOffsets',
        type: U32,
        required: true,
        description: 'Polygon offsets per feature.'
      },
      {name: 'polygonOffsets', type: U32, required: true, description: 'Ring offsets per polygon.'},
      {name: 'ringOffsets', type: U32, required: true, description: 'Vertex offsets per ring.'},
      {name: 'counts', type: U32, required: true, description: 'Points to draw per feature.'},
      {
        name: 'parameters',
        type: U32,
        required: true,
        description: 'Seed, from getGPUDotDensityParameterValues.'
      },
      {
        name: 'maximumAttempts',
        type: 'number',
        description: 'Rejection tries per point.',
        default: '32'
      },
      {
        name: 'output',
        type: "Omit<GPUDotDensityOutput, 'categories'>",
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'positions', type: F32X2, description: 'Sampled locations.'},
      {name: 'dots', type: 'GPUCompactOutput', description: 'Compacted point count.'},
      {name: 'status', type: '{failedCount?}', description: 'Points that exhausted attempts.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPURandomPointsInPolygon, GPUParameterBuffer, GPU_DOT_DENSITY_PARAMETER_LENGTH,
  getGPUDotDensityParameterValues} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'random-points'});
const parameters = new GPUParameterBuffer(device, {
  id: 'seed', format: 'uint32', length: GPU_DOT_DENSITY_PARAMETER_LENGTH,
  values: getGPUDotDensityParameterValues({seed: 42})
});
// geometry views, counts (uint32 per polygon), and outputs are graph views
graph.add(new GPURandomPointsInPolygon({
  positions, featureOffsets, polygonOffsets, ringOffsets, counts,
  parameters: parameters.importToGraph(graph),
  output: {positions: samples, dots}
}));
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'points',
      seed: 16,
      caption: 'Uniform random points inside irregular polygons.'
    },
    related: ['dot-density', 'random-points-on-line', 'continuous-spatial-sample'],
    lensScenario: 'gpu-spatial-analysis.random-points-in-polygon',
    reference: 'api/gpu-spatial-analysis/operators-density',
    maturity: 'stable'
  },
  {
    id: 'random-points-on-line',
    symbol: 'GPURandomPointsOnLine',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Random points on line',
    summary: 'Draws a requested number of uniform random points along each line.',
    explanation:
      'Use it to sample roads, rivers or tracks. Points are uniform by length, not by vertex. Optional `fractions` returns each point as a position along its line.',
    dataTypes: {input: ['lines'], output: ['points']},
    useCases: ['simulate', 'transform'],
    domains: ['transport', 'ecology'],
    params: [
      {name: 'positions', type: F32X2, required: true, description: 'Packed line vertices.'},
      {name: 'lineOffsets', type: U32, required: true, description: 'Vertex offsets per line.'},
      {name: 'counts', type: U32, required: true, description: 'Points to draw per line.'},
      {
        name: 'parameters',
        type: U32,
        required: true,
        description: 'Seed, from getGPUDotDensityParameterValues.'
      },
      {name: 'spatialContext', type: 'GPUSpatialContext', description: 'Frame of the coordinates.'},
      {
        name: 'output',
        type: 'GPURandomPointsOnLineOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'positions', type: F32X2, description: 'Sampled locations.'},
      {name: 'points', type: 'GPUCompactOutput', description: 'Compacted point count.'},
      {name: 'fractions', type: F32, description: 'Optional 0..1 position along the line.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPURandomPointsOnLine, GPUParameterBuffer, GPU_DOT_DENSITY_PARAMETER_LENGTH,
  getGPUDotDensityParameterValues} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'points-on-roads'});
const parameters = new GPUParameterBuffer(device, {
  id: 'seed', format: 'uint32', length: GPU_DOT_DENSITY_PARAMETER_LENGTH,
  values: getGPUDotDensityParameterValues({seed: 3})
});
// positions, lineOffsets, counts, outputs: graph views
graph.add(new GPURandomPointsOnLine({
  positions, lineOffsets, counts,
  parameters: parameters.importToGraph(graph),
  output: {positions: samples, points, fractions}
}));
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'lines', seed: 17, caption: 'Points placed uniformly by length along roads.'},
    related: ['random-points-in-polygon', 'dot-density'],
    lensScenario: 'gpu-spatial-analysis.random-points-on-line',
    reference: 'api/gpu-spatial-analysis/operators-density',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- interpolation
  {
    id: 'inverse-distance-weighting',
    symbol: 'GPUInverseDistanceWeighting',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Inverse distance weighting',
    summary:
      'Interpolates a raster from scattered samples, weighting neighbors by inverse distance.',
    explanation:
      'Use it for a quick, assumption-light surface from stations or soundings. Nearer samples count more, controlled by `power`; a Shepard taper smooths the search edge. It cannot predict beyond the sample range.',
    dataTypes: {input: ['points'], output: ['raster']},
    useCases: ['interpolate'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      {name: 'positions', type: F32X2, required: true, description: 'Sample locations.'},
      {name: 'values', type: F32, required: true, description: 'Sample values.'},
      {name: 'grid', type: GRID, required: true, description: 'Output raster.'},
      {
        name: 'parameters',
        type: F32,
        required: true,
        description: 'Search radius, power, taper, neighbor limits.'
      },
      {
        name: 'indexGridSize',
        type: INDEX_SIZE,
        required: true,
        description: 'Internal sample index resolution.'
      },
      {
        name: 'indexBounds',
        type: INDEX_BOUNDS,
        required: true,
        description: 'Must cover all samples.'
      },
      {name: 'maximumNeighborCount', type: 'number', description: 'Compile-time neighbor cap.'},
      {name: 'mask', type: U32, description: 'Zero excludes a sample.'}
    ],
    outputs: [
      {name: 'values', type: F32, description: 'Interpolated value per cell.'},
      {
        name: 'counts',
        type: "GraphDataView<'uint32'>",
        description: 'Optional neighbors used per cell.'
      }
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUInverseDistanceWeighting, GPUParameterBuffer, GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH,
  getGPUInverseDistanceWeightingParameterValues} from '${ANALYSIS}';
import {makeGPURasterMetadata} from '${RASTER}';

const graph = new GPUCommandGraph(device, {id: 'idw'});
const grid = makeGPURasterMetadata({width: 256, height: 256, cellSize: 50, origin: [x0, y0]});
const parameters = new GPUParameterBuffer(device, {
  id: 'idw-parameters', format: 'float32', length: GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH,
  values: getGPUInverseDistanceWeightingParameterValues({searchRadius: 2000, power: 2}, grid)
});
// positions, values, surface: graph views
graph.add(new GPUInverseDistanceWeighting({
  positions, values, grid, parameters: parameters.importToGraph(graph),
  indexGridSize: [32, 32], indexBounds: [x0, y0, x1, y1], output: {values: surface}
}));
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'interpolation',
      seed: 21,
      caption: 'A continuous surface through scattered stations.'
    },
    related: ['kriging', 'natural-neighbor', 'local-rbf'],
    lensScenario: 'gpu-spatial-analysis.inverse-distance-weighting',
    reference: 'api/gpu-spatial-analysis/operators-density',
    maturity: 'stable'
  },
  {
    id: 'kriging',
    symbol: 'GPUKriging',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Kriging',
    summary: 'Ordinary kriging of scattered samples onto a raster, with prediction variance.',
    explanation:
      'Use it when you can fit a variogram and want uncertainty with the surface. Each cell solves a local kriging system over its nearest samples using a spherical, exponential or gaussian model. Optional leave-one-out outputs validate the fit.',
    dataTypes: {input: ['points'], output: ['raster']},
    useCases: ['interpolate', 'model'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {name: 'positions', type: F32X2, required: true, description: 'Sample locations.'},
      {name: 'values', type: F32, required: true, description: 'Sample values.'},
      {name: 'grid', type: GRID, required: true, description: 'Output raster.'},
      {
        name: 'parameters',
        type: F32,
        required: true,
        description: 'Variogram, search radius and neighbor count.'
      },
      {
        name: 'indexGridSize',
        type: INDEX_SIZE,
        required: true,
        description: 'Internal sample index resolution.'
      },
      {
        name: 'indexBounds',
        type: INDEX_BOUNDS,
        required: true,
        description: 'Must cover all samples.'
      },
      {name: 'maximumNeighborCount', type: 'number', description: 'Compile-time neighbor cap.'},
      {name: 'invocationTileSize', type: 'number', description: 'Cells solved per dispatch tile.'}
    ],
    outputs: [
      {name: 'values', type: F32, description: 'Predicted value per cell.'},
      {name: 'variance', type: F32, description: 'Optional kriging variance.'},
      {
        name: 'crossValidationPrediction',
        type: F32,
        description: 'Optional leave-one-out prediction per sample.'
      },
      {
        name: 'status',
        type: '{invalidCount?, uncertainCount?}',
        description: 'Optional diagnostics.'
      }
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUKriging, GPUParameterBuffer, GPU_KRIGING_PARAMETER_LENGTH,
  getGPUKrigingParameterValues} from '${ANALYSIS}';
import {makeGPURasterMetadata} from '${RASTER}';

const graph = new GPUCommandGraph(device, {id: 'kriging'});
const grid = makeGPURasterMetadata({width: 128, height: 128, cellSize: 100, origin: [x0, y0]});
const parameters = new GPUParameterBuffer(device, {
  id: 'kriging-parameters', format: 'float32', length: GPU_KRIGING_PARAMETER_LENGTH,
  values: getGPUKrigingParameterValues({
    searchRadius: 3000, neighborCount: 16,
    variogram: {model: 'spherical', nugget: 0.1, sill: 1, range: 1500}
  }, grid)
});
// positions, values, prediction, variance: graph views
graph.add(new GPUKriging({
  positions, values, grid, parameters: parameters.importToGraph(graph),
  indexGridSize: [32, 32], indexBounds: [x0, y0, x1, y1],
  output: {values: prediction, variance}
}));
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'interpolation',
      seed: 22,
      caption: 'Predicted surface; uncertainty grows away from samples.'
    },
    related: ['local-kriging', 'conditional-gaussian-simulation', 'inverse-distance-weighting'],
    lensScenario: 'gpu-spatial-analysis.kriging',
    reference: 'api/gpu-spatial-analysis/operators-density',
    maturity: 'stable'
  },
  {
    id: 'local-kriging',
    symbol: 'GPULocalKriging',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Local kriging',
    summary:
      'Simple, ordinary, universal, external-drift or co-kriging at arbitrary query supports.',
    explanation:
      'Use it for prediction at points or blocks rather than a raster, or when you need drift terms or a second variable. Each query solves its own neighborhood system with per-observation error variance. Supports carry quadrature weights for block averages.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['interpolate', 'model'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {name: 'positions', type: F32X2, required: true, description: 'Sample locations.'},
      {
        name: 'observations',
        type: F32X4,
        required: true,
        description: 'Rows of [value, error variance, variable, validity].'
      },
      {
        name: 'supports',
        type: "GraphDataView<'float32x2' | 'float32x4'>",
        required: true,
        description: 'Query points or [x, y, weight] quadrature.'
      },
      {
        name: 'mode',
        type: "'simple' | 'ordinary' | 'universal' | 'external-drift' | 'co-kriging'",
        required: true,
        description: 'Kriging system.'
      },
      {
        name: 'covariance',
        type: 'GPUPredictionCovariance',
        required: true,
        description: 'Model, range, sill and nugget.'
      },
      {name: 'neighborCount', type: 'number', description: 'Nearest samples per query.'},
      {name: 'searchRadius', type: 'number', description: 'Maximum neighbor distance.'},
      {name: 'output', type: 'object', required: true, description: 'Destination views.'}
    ],
    outputs: [
      {
        name: 'values',
        type: F32X4,
        description: 'Prediction, variance and related terms per query.'
      },
      {name: 'status', type: U32, description: 'Per-query solve status.'},
      {name: 'weights', type: F32, description: 'Optional kriging weights.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPULocalKriging} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'local-kriging'});
// positions, observations [value, errorVariance, 0, 1], supports, outputs: graph views
graph.add(
  new GPULocalKriging({
    positions,
    observations,
    supports: queries,
    mode: 'ordinary',
    covariance: {model: 'exponential', range: 1200, sill: 1, nugget: 0.05},
    neighborCount: 16,
    output: {values, status}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'interpolation', seed: 23, caption: 'Predictions at chosen query locations.'},
    related: ['kriging', 'local-rbf', 'conditional-gaussian-simulation'],
    lensScenario: 'gpu-spatial-analysis.local-kriging',
    reference: 'api/gpu-spatial-analysis/interpolation-expansion',
    maturity: 'stable'
  },
  {
    id: 'conditional-gaussian-simulation',
    symbol: 'GPUConditionalGaussianSimulation',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Conditional Gaussian simulation',
    summary: 'Draws equally likely surfaces that honor the samples and a covariance model.',
    explanation:
      'Use it to carry spatial uncertainty into downstream results: run many replicates and summarize. Each replicate is an exact joint draw conditioned on the observations, seeded for repeatability. Output includes per-query mean and variance.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['simulate', 'interpolate'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {name: 'positions', type: F32X2, required: true, description: 'Sample locations.'},
      {
        name: 'observations',
        type: F32X4,
        required: true,
        description: 'Rows of [value, error variance, reserved, validity].'
      },
      {name: 'queries', type: F32X2, required: true, description: 'Locations to simulate.'},
      {
        name: 'covariance',
        type: 'GPUPredictionCovariance',
        required: true,
        description: 'Model, range, sill and nugget.'
      },
      {name: 'seed', type: 'number', required: true, description: 'Random seed.'},
      {
        name: 'replicateCount',
        type: 'number',
        required: true,
        description: 'Realizations to draw.'
      },
      {name: 'mean', type: 'number', description: 'Known process mean.'},
      {
        name: 'reusePreparation',
        type: 'boolean',
        description: 'Reuse the factorization across calls.'
      }
    ],
    outputs: [
      {name: 'values', type: F32, description: 'Simulated value per replicate and query.'},
      {name: 'moments', type: F32X2, description: 'Mean and variance across replicates.'},
      {name: 'status', type: U32, description: 'Solve status.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUConditionalGaussianSimulation} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'simulation'});
// positions, observations, queries, outputs: graph views
graph.add(
  new GPUConditionalGaussianSimulation({
    positions,
    observations,
    queries,
    covariance: {model: 'gaussian', range: 800, sill: 1, nugget: 0},
    mean: 0,
    seed: 2024,
    replicateCount: 32,
    output: {values, moments, status}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'interpolation',
      seed: 24,
      caption: 'One of many surfaces consistent with the data.'
    },
    related: ['kriging', 'local-kriging'],
    lensScenario: 'gpu-spatial-analysis.conditional-gaussian-simulation',
    reference: 'api/gpu-spatial-analysis/interpolation-expansion',
    maturity: 'stable'
  },
  {
    id: 'local-rbf',
    symbol: 'GPULocalRBF',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Local radial basis functions',
    summary: 'Interpolates scattered values with local radial basis functions at query points.',
    explanation:
      'Use it for smooth surfaces through exact or lightly smoothed samples when no variogram is available. Each query fits a small RBF system with a gaussian, thin-plate or polynomial kernel. Smoothing relaxes the exact fit.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['interpolate'],
    domains: ['spatial-statistics', 'three-d'],
    params: [
      {name: 'positions', type: F32X2, required: true, description: 'Sample locations.'},
      {name: 'values', type: F32, required: true, description: 'Sample values.'},
      {name: 'queries', type: F32X2, required: true, description: 'Prediction locations.'},
      {
        name: 'kernel',
        type: "'gaussian' | 'thin-plate-spline' | 'cubic' | 'linear' | 'quintic'",
        description: 'Basis function.'
      },
      {name: 'degree', type: '-1 | 0 | 1 | 2', description: 'Polynomial trend degree.'},
      {name: 'epsilon', type: 'number', description: 'Kernel shape parameter.'},
      {name: 'smoothing', type: 'number', description: 'Regularization; 0 interpolates exactly.'},
      {name: 'neighborCount', type: 'number', description: 'Nearest samples per query.'}
    ],
    outputs: [
      {name: 'values', type: F32X4, description: 'Prediction and related terms per query.'},
      {name: 'status', type: U32, description: 'Per-query solve status.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPULocalRBF} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'rbf'});
// positions, values, queries, outputs: graph views
graph.add(
  new GPULocalRBF({
    positions,
    values,
    queries,
    kernel: 'thin-plate-spline',
    degree: 1,
    smoothing: 0.01,
    neighborCount: 24,
    output: {values: predictions, status}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'interpolation',
      seed: 25,
      caption: 'Smooth surface fit through the sample points.'
    },
    related: ['local-kriging', 'natural-neighbor', 'inverse-distance-weighting'],
    lensScenario: 'gpu-spatial-analysis.local-rbf',
    reference: 'api/gpu-spatial-analysis/interpolation-expansion',
    maturity: 'stable'
  },
  {
    id: 'natural-neighbor',
    symbol: 'GPUNaturalNeighbor',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Natural neighbor',
    summary:
      'Interpolates at query points using Voronoi-area (Sibson) weights of neighboring samples.',
    explanation:
      'Use it for a parameter-free surface that adapts to uneven sample spacing. Weights come from how much of each sample’s Voronoi cell a new point would steal. Results stay inside the data range.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['interpolate'],
    domains: ['spatial-statistics', 'hydrology'],
    params: [
      {name: 'positions', type: F32X2, required: true, description: 'Sample locations.'},
      {name: 'values', type: F32, required: true, description: 'Sample values.'},
      {name: 'queries', type: F32X2, required: true, description: 'Prediction locations.'},
      {name: 'triangles', type: U32, description: 'Precomputed Delaunay triangles.'},
      {name: 'queryTriangles', type: U32, description: 'Containing triangle per query.'},
      {
        name: 'maximumVertexCount',
        type: 'number',
        description: 'Capacity for the internal triangulation.'
      },
      {name: 'maximumNeighborCount', type: 'number', description: 'Compile-time neighbor cap.'},
      {name: 'output', type: 'object', required: true, description: 'Destination views.'}
    ],
    outputs: [
      {name: 'values', type: F32X4, description: 'Prediction and related terms per query.'},
      {name: 'status', type: U32, description: 'Per-query status.'},
      {
        name: 'weights',
        type: "GraphDataView<'float32' | 'float32x2'>",
        description: 'Optional neighbor weights.'
      }
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUNaturalNeighbor} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'natural-neighbor'});
// positions, values, queries, outputs: graph views
graph.add(
  new GPUNaturalNeighbor({
    positions,
    values,
    queries,
    maximumNeighborCount: 32,
    output: {values: predictions, status}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'tessellation',
      seed: 26,
      caption: 'Each sample owns a Voronoi cell; queries blend neighbors.'
    },
    related: ['inverse-distance-weighting', 'local-rbf', 'kriging'],
    lensScenario: 'gpu-spatial-analysis.natural-neighbor',
    reference: 'api/gpu-spatial-analysis/interpolation-expansion',
    maturity: 'stable'
  },
  {
    id: 'focal-statistics',
    symbol: 'GPUFocalStatistics',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Focal statistics',
    summary:
      'Computes moving-window mean, min, max, range, deviation and count for every raster cell.',
    explanation:
      'Use it to smooth rasters, find local relief or fill gaps. The window is a square or circle up to radius 64; nodata cells are skipped. Radius and shape are parameters, so they change without recompiling.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['summarize', 'transform'],
    domains: ['remote-sensing', 'hydrology'],
    params: [
      {name: 'values', type: F32, required: true, description: 'Row-major raster; NaN is nodata.'},
      {name: 'grid', type: GRID, required: true, description: 'Raster dimensions and affine.'},
      {
        name: 'maximumRadius',
        type: 'number',
        required: true,
        description: 'Compile-time radius cap, at most 64.'
      },
      {
        name: 'parameters',
        type: F32,
        required: true,
        description: 'Radius, shape and minimum count.'
      },
      {name: 'validity', type: U32, description: 'Zero marks nodata.'},
      {name: 'noDataValue', type: 'number', description: 'Sentinel treated as nodata.'},
      {
        name: 'output',
        type: 'GPUFocalStatisticsOutput',
        required: true,
        description: 'Any non-empty subset of statistics.'
      }
    ],
    outputs: [
      {name: 'mean', type: F32, description: 'Window mean per cell.'},
      {name: 'standardDeviation', type: F32, description: 'Window deviation per cell.'},
      {name: 'min / max / range / sum', type: F32, description: 'Other window statistics.'},
      {name: 'counts', type: U32, description: 'Valid cells in each window.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUFocalStatistics, GPUParameterBuffer, GPU_FOCAL_STATISTICS_PARAMETER_LENGTH,
  getGPUFocalStatisticsParameterValues} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'focal'});
const parameters = new GPUParameterBuffer(device, {
  id: 'focal-parameters', format: 'float32', length: GPU_FOCAL_STATISTICS_PARAMETER_LENGTH,
  values: getGPUFocalStatisticsParameterValues({radius: 3, shape: 'circle', minimumCount: 5})
});
// raster values, grid metadata, outputs: graph views and GPURasterMetadata
graph.add(new GPUFocalStatistics({
  values: raster, noDataValue: -9999, grid, maximumRadius: 16,
  parameters: parameters.importToGraph(graph),
  output: {mean, standardDeviation, counts}
}));
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'raster', seed: 27, caption: 'A circular window slides over every cell.'},
    related: ['zonal-statistics', 'kernel-density'],
    lensScenario: 'gpu-spatial-analysis.focal-statistics',
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  {
    id: 'areal-interpolation',
    symbol: 'GPUArealInterpolation',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Areal interpolation',
    summary: 'Transfers values between two zone systems by overlap area, extensive or intensive.',
    explanation:
      'Use it to move census counts or rates onto new boundaries. Both zone sets are rasterized on a fine grid, and each source-target pair gets an area share. Optional `cellWeights` make it dasymetric.',
    dataTypes: {input: ['polygons', 'raster'], output: ['spatial-weights']},
    useCases: ['relate', 'summarize'],
    domains: ['demographics', 'urban'],
    params: [
      {name: 'grid', type: GRID, required: true, description: 'Fine common raster.'},
      {name: 'sourceZones', type: U32, required: true, description: 'Source zone id per cell.'},
      {name: 'targetZones', type: U32, required: true, description: 'Target zone id per cell.'},
      {
        name: 'sourceZoneCount',
        type: 'number',
        required: true,
        description: 'Number of source zones.'
      },
      {
        name: 'targetZoneCount',
        type: 'number',
        required: true,
        description: 'Number of target zones.'
      },
      {name: 'cellWeights', type: F32, description: 'Ancillary weight per cell (dasymetric).'},
      {
        name: 'mode',
        type: "'extensive' | 'intensive'",
        description: 'Sum-preserving or average-preserving.'
      },
      {
        name: 'output',
        type: 'GPUArealInterpolationOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {
        name: 'weights',
        type: 'GPUSpatialWeights',
        description: 'Source-to-target share matrix; apply with GPUSpatialLag.'
      },
      {name: 'status', type: 'GPUOverflowStatusPort', description: 'Overflow and invalid counts.'},
      {name: 'categoryShares', type: F32, description: 'Optional per-category shares.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUArealInterpolation, GPUSpatialLag} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'areal'});
// zone rasters, weights buffers, outputs: graph views; grid is GPURasterMetadata
graph.add(new GPUArealInterpolation({
  grid, sourceZones, targetZones, sourceZoneCount: 120, targetZoneCount: 48,
  mode: 'extensive',
  output: {weights: {offsets, neighbors, weights}, status: {overflow}}
}));
graph.add(new GPUSpatialLag({
  weights: {offsets, neighbors, weights},
  values: populationBySource,
  output: populationByTarget
}));
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'polygon-overlay',
      seed: 28,
      caption: 'Source zones re-aggregated onto target zones.'
    },
    related: ['pycnophylactic', 'zonal-statistics'],
    lensScenario: 'gpu-spatial-analysis.areal-interpolation',
    reference: 'api/gpu-spatial-analysis/operators-accessibility',
    maturity: 'stable'
  },
  {
    id: 'pycnophylactic',
    symbol: 'GPUPycnophylactic',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Pycnophylactic interpolation',
    summary:
      'Smooths zone totals into a continuous density surface that preserves each zone’s total.',
    explanation:
      'Use it to turn polygon counts into a surface without sharp zone jumps (Tobler 1979). It alternates neighbor smoothing with a mass correction until the change drops below `tolerance`. The result sums back to each zone total.',
    dataTypes: {input: ['polygons', 'table'], output: ['raster']},
    useCases: ['interpolate', 'transform'],
    domains: ['demographics', 'urban'],
    params: [
      {name: 'grid', type: GRID, required: true, description: 'Output raster.'},
      {name: 'zones', type: U32, required: true, description: 'Zone id per cell.'},
      {name: 'zoneCount', type: 'number', required: true, description: 'Number of zones.'},
      {name: 'totals', type: F32, required: true, description: 'Value to preserve per zone.'},
      {name: 'maximumIterations', type: 'number', required: true, description: 'Iteration cap.'},
      {name: 'tolerance', type: F32, description: 'Convergence threshold.'},
      {name: 'kernel', type: "'rook' | 'box'", description: 'Smoothing neighborhood.'},
      {
        name: 'output',
        type: 'GPUPycnophylacticOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'mass', type: F32, description: 'Mass per cell; sums to each zone total.'},
      {name: 'density', type: F32, description: 'Optional mass per unit area.'},
      {
        name: 'status',
        type: '{converged?, iterationCount?}',
        description: 'Optional convergence info.'
      }
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUPycnophylactic} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'pycno'});
// zones, totals, tolerance, outputs: graph views; grid is GPURasterMetadata
graph.add(
  new GPUPycnophylactic({
    grid,
    zones,
    zoneCount: 64,
    totals: populationByZone,
    maximumIterations: 200,
    tolerance,
    kernel: 'rook',
    output: {mass, density, status: {converged, iterationCount}}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'density-surface',
      seed: 29,
      caption: 'Zone totals relaxed into a smooth, mass-true surface.'
    },
    related: ['areal-interpolation', 'kernel-density'],
    lensScenario: 'gpu-spatial-analysis.pycnophylactic',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- point processes and sampling
  {
    id: 'inhomogeneous-point-pattern',
    symbol: 'GPUInhomogeneousPointPattern',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Inhomogeneous point pattern',
    summary:
      'Estimates intensity, K and pair correlation for a point pattern with varying density.',
    explanation:
      'Use it to test clustering after accounting for a non-uniform background such as population. Intensity comes from a kernel estimate or your own surface; K and g are edge-corrected over a polygon window. Compare against simulated envelopes.',
    dataTypes: {input: ['points', 'polygons'], output: ['table']},
    useCases: ['find-patterns', 'model'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {name: 'positions', type: F32X2, required: true, description: 'Event locations.'},
      {
        name: 'studyWindow',
        type: 'GPUPolygonStudyWindow',
        required: true,
        description: 'Polygon observation window.'
      },
      {
        name: 'radii',
        type: F32,
        required: true,
        description: 'Distances at which K and g are evaluated.'
      },
      {
        name: 'suppliedIntensity',
        type: F32,
        description: 'Intensity per point; skips the kernel estimate.'
      },
      {name: 'bandwidth', type: 'number', description: 'Kernel bandwidth for intensity.'},
      {
        name: 'edgeCorrection',
        type: "'border' | 'translation'",
        description: 'Edge correction for K.'
      },
      {
        name: 'selfContribution',
        type: "'exclude' | 'include'",
        description: 'Whether a point counts toward its own intensity.'
      },
      {name: 'output', type: 'object', required: true, description: 'Destination views.'}
    ],
    outputs: [
      {name: 'intensity', type: F32, description: 'Intensity at each point.'},
      {name: 'k', type: F32, description: 'Inhomogeneous K at each radius.'},
      {name: 'pairCorrelation', type: F32, description: 'Pair correlation g at each radius.'},
      {name: 'status', type: U32, description: 'Capacity and validity flags.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUInhomogeneousPointPattern} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'kinhom'});
// positions, studyWindow, radii, outputs: graph views and window geometry
graph.add(
  new GPUInhomogeneousPointPattern({
    positions,
    studyWindow,
    radii,
    bandwidth: 500,
    edgeCorrection: 'translation',
    selfContribution: 'exclude',
    output: {intensity, k, pairCorrelation, status}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'point-clusters',
      seed: 31,
      caption: 'Clustering measured against a changing background.'
    },
    related: ['point-pattern-envelope', 'kernel-density'],
    lensScenario: 'gpu-spatial-analysis.inhomogeneous-point-pattern',
    reference: 'api/gpu-spatial-analysis/point-process',
    maturity: 'stable'
  },
  {
    id: 'point-pattern-envelope',
    symbol: 'GPUPointPatternEnvelope',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Point pattern envelope',
    summary: 'Builds simulation envelopes and a Monte Carlo p-value for a point pattern function.',
    explanation:
      'Use it to judge whether an observed K or g curve is more clustered than chance. Rank the simulated curves to get lower and upper bands, and a deviation statistic for a global test. State the null model and whether parameters were refit.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {
        name: 'observed',
        type: F32,
        required: true,
        description: 'Observed function values per radius.'
      },
      {
        name: 'simulations',
        type: F32,
        required: true,
        description: 'Simulated values, replicate-major.'
      },
      {
        name: 'reference',
        type: F32,
        required: true,
        description: 'Theoretical reference per radius.'
      },
      {
        name: 'integrationWeights',
        type: F32,
        required: true,
        description: 'Radius weights for global statistics.'
      },
      {
        name: 'simulationCount',
        type: 'number',
        required: true,
        description: 'Number of replicates.'
      },
      {name: 'rank', type: 'number', description: 'Envelope rank from each end.'},
      {
        name: 'statistic',
        type: "'maximum-deviation' | 'integrated-squared-deviation' | ...",
        description: 'Global deviation statistic.'
      },
      {
        name: 'nullModel',
        type: 'string',
        required: true,
        description: 'Label of the simulated null.'
      }
    ],
    outputs: [
      {name: 'lower', type: F32, description: 'Lower envelope per radius.'},
      {name: 'upper', type: F32, description: 'Upper envelope per radius.'},
      {name: 'statistics', type: F32, description: 'Global statistics for observed and simulated.'},
      {name: 'pValueInterval', type: F32, description: 'Optional Monte Carlo p-value bounds.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUPointPatternEnvelope} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'envelope'});
// observed, simulations (99 x radii), reference, weights, outputs: graph views
graph.add(
  new GPUPointPatternEnvelope({
    observed,
    simulations,
    reference,
    integrationWeights,
    simulationCount: 99,
    rank: 5,
    nullModel: 'complete spatial randomness',
    fittedParameters: 'none',
    output: {lower, upper, statistics, status}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'histogram', seed: 32, caption: 'Observed curve against the simulated band.'},
    related: ['inhomogeneous-point-pattern'],
    lensScenario: 'gpu-spatial-analysis.point-pattern-envelope',
    reference: 'api/gpu-spatial-analysis/point-process',
    maturity: 'stable'
  },
  {
    id: 'continuous-spatial-sample',
    symbol: 'GPUContinuousSpatialSample',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Continuous spatial sample',
    summary:
      'Draws a well-spread random sample of sites on a line or in an area, with design weights.',
    explanation:
      'Use it to pick field sites that cover a region evenly yet randomly. Candidates are accepted in waves when they respect a minimum `spacing`. Each site carries a design weight for unbiased estimates.',
    dataTypes: {input: ['lines', 'polygons'], output: ['points']},
    useCases: ['simulate', 'summarize'],
    domains: ['ecology', 'spatial-statistics'],
    params: [
      {
        name: 'frame',
        type: 'GPUContinuousSpatialFrame',
        required: true,
        description: 'Line or area to sample.'
      },
      {
        name: 'spacing',
        type: 'number',
        required: true,
        description: 'Minimum distance between sites.'
      },
      {name: 'seed', type: 'number', required: true, description: 'Random seed.'},
      {name: 'capacity', type: 'number', required: true, description: 'Maximum sites returned.'},
      {name: 'replicate', type: 'number', description: 'Independent replicate index.'},
      {name: 'waves', type: 'number', description: 'Candidate rounds.'},
      {name: 'maximumCandidates', type: 'number', description: 'Candidate cap per wave.'},
      {name: 'output', type: 'object', required: true, description: 'Destination views.'}
    ],
    outputs: [
      {name: 'positions', type: F32X2, description: 'Selected sites.'},
      {name: 'designWeights', type: F32, description: 'Inverse inclusion weight per site.'},
      {name: 'provenance', type: U32X4, description: 'How each site was drawn.'},
      {name: 'status', type: U32X4, description: 'Completeness counters.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUContinuousSpatialSample} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'sites'});
// frame polygon/line vertices and outputs are graph views
graph.add(
  new GPUContinuousSpatialSample({
    frame: {type: 'area', positions: boundary, bounds: [x0, y0, x1, y1]},
    spacing: 250,
    seed: 11,
    capacity: 200,
    output: {positions: sites, designWeights, provenance, status}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'points', seed: 33, caption: 'Random yet evenly spread sample sites.'},
    related: ['spatial-balanced-sample', 'random-points-in-polygon'],
    lensScenario: 'gpu-spatial-analysis.continuous-spatial-sample',
    reference: 'api/gpu-spatial-analysis/point-process',
    maturity: 'stable'
  },
  {
    id: 'spatial-balanced-sample',
    symbol: 'GPUSpatialBalancedSample',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Spatially balanced sample',
    summary:
      'Selects units from a finite frame so the sample is spread across space within each stratum.',
    explanation:
      'Use it to sample parcels, plots or addresses with an even footprint. Allocations set the sample size per stratum, and inclusion probabilities are returned for design-based estimates. A seed and replicate index make draws reproducible.',
    dataTypes: {input: ['points', 'table'], output: ['table']},
    useCases: ['simulate', 'summarize'],
    domains: ['ecology', 'demographics'],
    params: [
      {name: 'positions', type: F32X2, required: true, description: 'Frame unit locations.'},
      {name: 'frameIds', type: U32, required: true, description: 'Stable id per frame unit.'},
      {
        name: 'allocations',
        type: 'readonly GPUSpatialSampleAllocation[]',
        required: true,
        description: 'Sample count per stratum.'
      },
      {
        name: 'bounds',
        type: INDEX_BOUNDS,
        required: true,
        description: 'Extent used for spatial ordering.'
      },
      {name: 'seed', type: 'number', required: true, description: 'Random seed.'},
      {name: 'strata', type: U32, description: 'Stratum id per unit.'},
      {name: 'replicate', type: 'number', description: 'Independent replicate index.'},
      {name: 'output', type: 'object', required: true, description: 'Destination views.'}
    ],
    outputs: [
      {name: 'frameIds', type: U32, description: 'Selected unit ids.'},
      {name: 'inclusionProbabilities', type: F32, description: 'Chance each unit was selected.'},
      {name: 'designWeights', type: F32, description: 'Inverse inclusion probability.'},
      {name: 'status', type: U32, description: 'Selection status.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUSpatialBalancedSample} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'balanced-sample'});
// positions, frameIds, strata, outputs: graph views
graph.add(
  new GPUSpatialBalancedSample({
    positions,
    frameIds,
    strata,
    allocations: [
      {stratumId: 0, sampleCount: 30},
      {stratumId: 1, sampleCount: 20}
    ],
    bounds: [x0, y0, x1, y1],
    seed: 5,
    output: {frameIds: selected, inclusionProbabilities, designWeights, status}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'point-clusters',
      seed: 34,
      caption: 'A sample drawn evenly across clusters and strata.'
    },
    related: ['continuous-spatial-sample', 'random-points-in-polygon'],
    lensScenario: 'gpu-spatial-analysis.spatial-balanced-sample',
    reference: 'api/gpu-spatial-analysis/point-process',
    maturity: 'stable'
  },
  {
    id: 'geographic-distribution',
    symbol: 'GPUGeographicDistribution',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Geographic distribution',
    summary:
      'Computes mean center, median center, standard distance and directional ellipse per group.',
    explanation:
      'Use it to summarize where a set of events sits and how it is stretched, such as crime by type or species by year. Weights and group ids allow many groups in one pass. Ellipse vertices are ready to draw.',
    dataTypes: {input: ['points'], output: ['table', 'polygons']},
    useCases: ['summarize', 'measure'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      {name: 'positions', type: F32X2, required: true, description: 'Point locations.'},
      {
        name: 'parameters',
        type: F32,
        required: true,
        description: 'Origin and ellipse deviations.'
      },
      {name: 'weights', type: F32, description: 'Per-point weight.'},
      {name: 'groupIds', type: U32, description: 'Group id per point.'},
      {name: 'groupCount', type: 'number', description: 'Number of groups.'},
      {name: 'mask', type: U32, description: 'Zero excludes a point.'},
      {name: 'maximumMedianIterations', type: 'number', description: 'Weiszfeld iteration cap.'},
      {
        name: 'reduction',
        type: "'auto' | 'chunked' | 'sorted'",
        description: 'Group reduction strategy.',
        default: "'auto'"
      }
    ],
    outputs: [
      {name: 'meanCenters', type: F32X2, description: 'Weighted mean center per group.'},
      {name: 'medianCenters', type: F32X2, description: 'Geometric median per group.'},
      {name: 'standardDistances', type: F32, description: 'Standard distance per group.'},
      {name: 'ellipses', type: F32, description: 'Directional ellipse parameters.'},
      {name: 'ellipseVertices', type: F32X2, description: 'Outline vertices for drawing.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUGeographicDistribution, GPUParameterBuffer, GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
  getGPUGeographicDistributionParameterValues} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'distribution'});
const parameters = new GPUParameterBuffer(device, {
  id: 'distribution-parameters', format: 'float32', length: GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
  values: getGPUGeographicDistributionParameterValues({origin: [x0, y0], standardDeviations: 2})
});
// positions, weights, groupIds, outputs: graph views
graph.add(new GPUGeographicDistribution({
  positions, weights, groupIds, groupCount: 8,
  parameters: parameters.importToGraph(graph),
  output: {meanCenters, standardDistances, ellipses, ellipseVertices}
}));
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'point-clusters',
      seed: 35,
      caption: 'Mean center and standard-deviation ellipse per group.'
    },
    related: ['rose-statistic', 'point-density'],
    lensScenario: 'gpu-spatial-analysis.geographic-distribution',
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  {
    id: 'rose-statistic',
    symbol: 'GPURoseStatistic',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Rose statistic',
    summary:
      'Summarizes the directions of line segments as a rose diagram with circular statistics.',
    explanation:
      'Use it for street orientation, fault trends or movement headings. Bearings are binned into a rose and reduced to a mean direction and concentration. Choose directional (0..360) or axial (0..180) mode.',
    dataTypes: {input: ['lines', 'points'], output: ['table']},
    useCases: ['measure', 'summarize'],
    domains: ['urban', 'transport'],
    params: [
      {name: 'starts', type: F32X2, required: true, description: 'Segment start points.'},
      {name: 'ends', type: F32X2, required: true, description: 'Segment end points.'},
      {
        name: 'circular',
        type: 'GPUCircularStatisticsContract',
        required: true,
        description: 'Mode and bin count.'
      },
      {name: 'weights', type: F32, description: 'Per-segment weight, such as length.'},
      {name: 'groupIds', type: U32, description: 'Group id per segment.'},
      {name: 'groupCount', type: 'number', description: 'Number of groups.'},
      {name: 'mask', type: U32, description: 'Zero excludes a segment.'},
      {
        name: 'output',
        type: 'GPURoseStatisticOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'bins', type: F32, description: 'Weighted count per angular bin.'},
      {name: 'summary', type: F32, description: 'Mean direction and concentration per group.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPURoseStatistic} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'street-rose'});
// starts, ends, lengths, bins, summary: graph views
graph.add(
  new GPURoseStatistic({
    starts,
    ends,
    weights: lengths,
    circular: {
      mode: 'axial',
      angleUnit: 'radians',
      origin: 'positive-x-counter-clockwise',
      binCount: 36
    },
    output: {bins, summary}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'histogram', seed: 36, caption: 'Street bearings binned into a rose.'},
    related: ['geographic-distribution'],
    lensScenario: 'gpu-spatial-analysis.rose-statistic',
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- distribution dynamics
  {
    id: 'class-assignment',
    symbol: 'GPUClassAssignment',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Class assignment',
    summary: 'Assigns every value to a class of shared breaks, one legend across all periods.',
    explanation:
      'Use it before a transition matrix: classify each period with the same pooled breaks. A value gets the number of inner breaks it meets or exceeds. NaN and masked rows get a no-class sentinel.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['classify'],
    domains: ['demographics', 'spatial-statistics'],
    params: [
      {name: 'values', type: F32, required: true, description: 'Column to classify.'},
      {
        name: 'breaks',
        type: F32,
        required: true,
        description: 'Class edges from a classification.'
      },
      {name: 'classCount', type: U32, required: true, description: 'Single row holding k.'},
      {name: 'mask', type: U32, description: 'Zero excludes a row.'},
      {name: 'output', type: U32, required: true, description: 'Class per row, 0..k-1.'}
    ],
    outputs: [
      {name: 'output', type: U32, description: 'Class per row; sentinel for NaN or masked rows.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUClassAssignment} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'classes'});
// values (all periods stacked), breaks (k + 1), classCount, classes: graph views
graph.add(
  new GPUClassAssignment({
    values,
    breaks,
    classCount,
    output: classes
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'classification', seed: 41, caption: 'Values mapped to five shared classes.'},
    related: ['classification-fit', 'transition-matrix'],
    lensScenario: 'gpu-spatial-analysis.class-assignment',
    reference: 'api/gpu-spatial-analysis/operators-dynamics',
    maturity: 'stable'
  },
  {
    id: 'classification-fit',
    symbol: 'GPUClassificationFit',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Classification fit',
    summary:
      'Scores how well a classification describes a column: class counts, medians, deviations and GADF.',
    explanation:
      'Use it to compare class schemes (quantile, natural breaks, equal interval) on one footing. The goodness of absolute deviation fit compares within-class to overall deviation around the median. Higher is better.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['classify', 'model'],
    domains: ['demographics', 'spatial-statistics'],
    params: [
      {name: 'values', type: F32, required: true, description: 'Column that was classified.'},
      {name: 'classes', type: U32, required: true, description: 'Class per row.'},
      {name: 'classCount', type: 'number', required: true, description: 'Number of classes.'},
      {name: 'mask', type: U32, description: 'Zero excludes a row.'},
      {
        name: 'output',
        type: 'GPUClassificationFitOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'counts', type: U32, description: 'Rows per class.'},
      {name: 'medians', type: F32, description: 'Median per class.'},
      {name: 'absoluteDeviations', type: F32, description: 'Absolute deviation per class.'},
      {name: 'squaredDeviations', type: F32, description: 'Squared deviation per class.'},
      {name: 'summary', type: F32, description: 'Overall fit statistics including GADF.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUClassificationFit} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'fit'});
// values, classes, outputs: graph views
graph.add(
  new GPUClassificationFit({
    values,
    classes,
    classCount: 5,
    output: {counts, medians, absoluteDeviations, squaredDeviations, summary}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'classification',
      seed: 42,
      caption: 'Within-class spread versus overall spread.'
    },
    related: ['class-assignment'],
    lensScenario: 'gpu-spatial-analysis.classification-fit',
    reference: 'api/gpu-spatial-analysis/operators-dynamics',
    maturity: 'stable'
  },
  {
    id: 'transition-matrix',
    symbol: 'GPUTransitionMatrix',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Transition matrix',
    summary:
      'Counts class-to-class moves between periods and turns them into transition probabilities.',
    explanation:
      'Use it to see how places move between classes over time, such as income quintiles. Rows are classes at time t, columns at t + lag. A condition class splits the matrix by context.',
    dataTypes: {input: ['table', 'time-series'], output: ['table']},
    useCases: ['detect-change', 'model'],
    domains: ['demographics', 'spatial-statistics'],
    params: [
      {
        name: 'classes',
        type: U32,
        required: true,
        description: 'Class per place and period, period-major.'
      },
      {name: 'periodCount', type: 'number', required: true, description: 'Number of periods.'},
      {name: 'classCount', type: 'number', required: true, description: 'Number of classes.'},
      {
        name: 'periodLag',
        type: 'number',
        description: 'Periods between observations.',
        default: '1'
      },
      {
        name: 'conditionClasses',
        type: U32,
        description: 'Conditioning class per place and period.'
      },
      {name: 'conditionCount', type: 'number', description: 'Number of condition classes.'},
      {
        name: 'minimumTransitions',
        type: 'number',
        description: 'Row total below which a row is sparse.'
      },
      {
        name: 'output',
        type: 'GPUTransitionMatrixOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'counts', type: U32, description: 'Transition counts.'},
      {name: 'probabilities', type: F32, description: 'Row-normalized probabilities.'},
      {name: 'rowTotals', type: U32, description: 'Transitions out of each class.'},
      {name: 'steadyStates', type: F32, description: 'Optional long-run distribution.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUTransitionMatrix} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'transitions'});
// classes (places x periods, uint32), outputs: graph views
graph.add(
  new GPUTransitionMatrix({
    classes,
    periodCount: 6,
    classCount: 5,
    minimumTransitions: 5,
    output: {counts, probabilities, rowTotals, steadyStates}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'time-series',
      seed: 43,
      caption: 'Probability of moving from one class to another.'
    },
    related: ['spatial-markov', 'lisa-markov', 'rank-markov', 'transition-diagnostics'],
    lensScenario: 'gpu-spatial-analysis.transition-matrix',
    reference: 'api/gpu-spatial-analysis/operators-dynamics',
    maturity: 'stable'
  },
  {
    id: 'spatial-markov',
    symbol: 'GPUSpatialMarkov',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Spatial Markov',
    summary: 'Transition matrices conditioned on the class of each place’s spatial lag.',
    explanation:
      'Use it to ask whether a rich neighbor changes a place’s chances of moving up. It classifies the weights-based lag, then builds one transition matrix per lag class. Compare matrices with the diagnostics.',
    dataTypes: {input: ['table', 'spatial-weights'], output: ['table']},
    useCases: ['detect-change', 'model'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      {name: 'values', type: F32, required: true, description: 'Values per place and period.'},
      {name: 'classes', type: U32, required: true, description: 'Class per place and period.'},
      {name: 'classCount', type: 'number', required: true, description: 'Number of classes.'},
      {name: 'weights', type: WEIGHTS, required: true, description: 'Neighbor weights.'},
      {name: 'periodCount', type: 'number', required: true, description: 'Number of periods.'},
      {
        name: 'maximumLagClassCount',
        type: 'number',
        required: true,
        description: 'Compile-time lag class cap.'
      },
      {
        name: 'lagParameters',
        type: F32,
        required: true,
        description: 'Lag classification parameters.'
      },
      {
        name: 'lagMethods',
        type: 'readonly GPUClassBreaksMethod[]',
        description: 'How lag classes are cut.'
      }
    ],
    outputs: [
      {name: 'counts', type: U32, description: 'Counts per lag class.'},
      {name: 'probabilities', type: F32, description: 'Probabilities per lag class.'},
      {name: 'lagBreaks', type: F32, description: 'Breaks of the lag classification.'},
      {name: 'homogeneity', type: F32, description: 'Optional test of equal matrices.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUSpatialMarkov} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'spatial-markov'});
// values, classes, weights, lagParameters, outputs: graph views
graph.add(
  new GPUSpatialMarkov({
    values,
    classes,
    classCount: 5,
    weights,
    periodCount: 6,
    maximumLagClassCount: 5,
    lagParameters,
    output: {counts, probabilities, lagBreaks, lagClassCount, homogeneity}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'time-series', seed: 44, caption: 'One transition matrix per neighbor class.'},
    related: ['transition-matrix', 'lisa-markov', 'transition-diagnostics'],
    lensScenario: 'gpu-spatial-analysis.spatial-markov',
    reference: 'api/gpu-spatial-analysis/operators-dynamics',
    maturity: 'stable'
  },
  {
    id: 'lisa-markov',
    symbol: 'GPULISAMarkov',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'LISA Markov',
    summary: 'Transition matrix between local Moran quadrants (HH, LH, LL, HL) across periods.',
    explanation:
      'Use it to see how hot and cold spots persist or flip over time. Each period gets a local Moran quadrant, optionally gated by significance. Transitions between those states form the matrix.',
    dataTypes: {input: ['table', 'spatial-weights'], output: ['table']},
    useCases: ['detect-change', 'find-patterns'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      {name: 'values', type: F32, required: true, description: 'Values per place and period.'},
      {name: 'weights', type: WEIGHTS, required: true, description: 'Neighbor weights.'},
      {name: 'periodCount', type: 'number', required: true, description: 'Number of periods.'},
      {
        name: 'parameters',
        type: F32,
        required: true,
        description: 'Local Moran parameters, including significance level.'
      },
      {
        name: 'gating',
        type: "'significance' | 'none'",
        description: 'Whether non-significant places form their own state.'
      },
      {
        name: 'periodLag',
        type: 'number',
        description: 'Periods between observations.',
        default: '1'
      },
      {
        name: 'minimumTransitions',
        type: 'number',
        description: 'Row total below which a row is sparse.'
      },
      {name: 'mask', type: U32, description: 'Zero excludes a row.'}
    ],
    outputs: [
      {name: 'counts', type: U32, description: 'Transition counts between states.'},
      {name: 'probabilities', type: F32, description: 'Row-normalized probabilities.'},
      {name: 'quadrants', type: U32, description: 'Optional state per place and period.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPULISAMarkov} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'lisa-markov'});
// values, weights, parameters (getGPUSpatialAutocorrelationParameterValues), outputs: graph views
graph.add(
  new GPULISAMarkov({
    values,
    weights,
    periodCount: 6,
    parameters,
    gating: 'significance',
    output: {counts, probabilities, quadrants}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'hotspots', seed: 45, caption: 'Hot and cold spots moving between periods.'},
    related: ['spatial-markov', 'transition-matrix', 'transition-diagnostics'],
    lensScenario: 'gpu-spatial-analysis.lisa-markov',
    reference: 'api/gpu-spatial-analysis/operators-dynamics',
    maturity: 'stable'
  },
  {
    id: 'rank-markov',
    symbol: 'GPURankMarkov',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Rank Markov',
    summary: 'Transitions of each place’s rank across periods, with mobility and rank diagnostics.',
    explanation:
      'Use it to study rank mobility without picking class breaks. Places are ranked each period (full or geographic rank) and rank moves become the transitions. Ties break by the giddy rule or by row id.',
    dataTypes: {input: ['table', 'time-series'], output: ['table']},
    useCases: ['detect-change', 'model'],
    domains: ['demographics', 'spatial-statistics'],
    params: [
      {name: 'values', type: F32, required: true, description: 'Values per place and period.'},
      {name: 'periodCount', type: 'number', required: true, description: 'Number of periods.'},
      {
        name: 'mode',
        type: "'full-rank' | 'geographic-rank'",
        description: 'Rank across all places or within geography.'
      },
      {name: 'tieOrder', type: "'giddy' | 'row-id'", description: 'Tie-breaking rule.'},
      {name: 'mask', type: U32, description: 'Zero excludes a row.'},
      {
        name: 'output',
        type: 'GPUTransitionDiagnosticsOutput & {...}',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'ranks', type: U32, description: 'Rank per place and period.'},
      {name: 'mobility', type: F32, description: 'Mobility per period pair.'},
      {name: 'counts', type: U32, description: 'Transition counts.'},
      {name: 'steadyStates', type: F32, description: 'Optional long-run distribution.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPURankMarkov} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'rank-markov'});
// values (places x periods), outputs: graph views
graph.add(
  new GPURankMarkov({
    values,
    periodCount: 6,
    mode: 'full-rank',
    tieOrder: 'giddy',
    output: {ranks, mobility, counts, probabilities}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'time-series', seed: 46, caption: 'Rank trajectories across periods.'},
    related: ['transition-matrix', 'spatial-markov'],
    lensScenario: 'gpu-spatial-analysis.rank-markov',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  },
  {
    id: 'transition-diagnostics',
    symbol: 'GPUTransitionDiagnostics',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Transition diagnostics',
    summary: 'Tests homogeneity and finds steady states of transition counts.',
    explanation:
      'Use it to judge whether conditional transition matrices differ and where the process settles. Homogeneity is a chi-square across conditions; steady states come from iterating the normalized matrix. Sparse rows are flagged.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['model', 'detect-change'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      {
        name: 'counts',
        type: U32,
        required: true,
        description: 'Transition counts, one matrix per condition.'
      },
      {name: 'classCount', type: 'number', required: true, description: 'Number of classes.'},
      {name: 'conditionCount', type: 'number', description: 'Number of conditions.'},
      {
        name: 'minimumTransitions',
        type: 'number',
        description: 'Row total below which a row is sparse.'
      },
      {
        name: 'output',
        type: 'GPUTransitionDiagnosticsOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'homogeneity', type: F32, description: 'Chi-square test of equal matrices.'},
      {name: 'steadyStates', type: F32, description: 'Long-run distribution per condition.'},
      {name: 'stateSparsity', type: U32, description: 'Sparse rows per condition.'},
      {name: 'status', type: '{converged?}', description: 'Steady-state convergence flag.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUTransitionDiagnostics} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'diagnostics'});
// counts from GPUTransitionMatrix or GPUSpatialMarkov, outputs: graph views
graph.add(
  new GPUTransitionDiagnostics({
    counts,
    classCount: 5,
    conditionCount: 5,
    minimumTransitions: 5,
    output: {homogeneity, steadyStates, stateSparsity, status: {converged}}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'histogram',
      seed: 47,
      caption: 'Long-run class shares from a transition matrix.'
    },
    related: ['transition-matrix', 'spatial-markov', 'lisa-markov'],
    lensScenario: 'gpu-spatial-analysis.transition-diagnostics',
    reference: 'api/gpu-spatial-analysis/operators-dynamics',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- region statistics
  {
    id: 'region-statistics',
    symbol: 'GPURegionStatistics',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Region statistics',
    summary:
      'Count, sum, mean, extremes and histogram of values inside a rectangle, polygon, radius or pick.',
    explanation:
      'Use it for live brush-and-summarize interaction on huge point sets. The region is a per-frame view, so dragging never recompiles. An optional grid index avoids testing every row.',
    dataTypes: {input: ['points', 'polygons'], output: ['table']},
    useCases: ['summarize', 'find-patterns'],
    domains: ['data-engineering', 'cartography'],
    params: [
      {
        name: 'region',
        type: 'GPURegionSelection',
        required: true,
        description: 'Rectangle, polygon, radius, pick or mask.'
      },
      {name: 'positions', type: 'GPUFloat32Positions', description: 'Point locations.'},
      {name: 'values', type: F32, description: 'Values to summarize.'},
      {name: 'sourceIds', type: 'GPUUint32Rows', description: 'Ids reported for selected rows.'},
      {name: 'histogram', type: 'GPURegionHistogramProps', description: 'Bin count and range.'},
      {
        name: 'spatialIndex',
        type: 'GPURegionStatisticsGridIndex',
        description: 'Grid index to skip distant rows.'
      },
      {
        name: 'spatialContext',
        type: 'GPUSpatialContext',
        description: 'Planar or geographic frame.'
      },
      {
        name: 'output',
        type: 'GPURegionStatisticsOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {
        name: 'summary',
        type: U32,
        description: 'Packed count, sum, mean, min, max, flags, histogram.'
      },
      {name: 'mask', type: U32, description: 'Optional selected-row mask.'},
      {name: 'selectedRows', type: 'GPUCompactOutput', description: 'Optional compacted ids.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPURegionStatistics, GPUParameterBuffer} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'brush'});
const bounds = new GPUParameterBuffer(device, {id: 'bounds', format: 'float32', length: 4});
// positions, values, sourceIds, summary, selectedRows: graph views
graph.add(new GPURegionStatistics({
  region: {kind: 'rectangle', bounds: bounds.importToGraph(graph)},
  positions, values, sourceIds,
  histogram: {binCount: 16},
  output: {summary, selectedRows}
}));
const compiled = graph.compile();
bounds.write(new Float32Array([minX, minY, maxX, maxY])); // every pointer move
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'points', seed: 51, caption: 'Drag a region; statistics update on the GPU.'},
    related: ['region-statistics-readback', 'region-mask', 'pick-region-mask', 'zonal-statistics'],
    lensScenario: 'gpu-spatial-analysis.region-statistics',
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  {
    id: 'region-statistics-readback',
    symbol: 'GPURegionStatisticsReadback',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Region statistics readback',
    summary:
      'Reads a region summary back to the CPU through a small ring, without stalling the frame.',
    explanation:
      'Use it to show numbers in the UI while the GPU keeps brushing. `encodeRead` copies the summary into a free ring slot and returns a ticket; `read` resolves it later as plain numbers.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['summarize'],
    domains: ['data-engineering'],
    params: [
      {
        name: 'id',
        type: 'string',
        description: 'Resource id.',
        default: "'region-statistics-readback'"
      },
      {
        name: 'binCount',
        type: 'number',
        description: 'Histogram bins in the summary.',
        default: '0'
      },
      {name: 'slotCount', type: 'number', description: 'Ring slots in flight.', default: '3'}
    ],
    outputs: [
      {
        name: 'selectedCount, valueCount',
        type: 'number',
        description: 'Rows selected and rows with a value.'
      },
      {name: 'sum, mean, minimum, maximum', type: 'number', description: 'Value statistics.'},
      {name: 'histogram', type: 'Uint32Array', description: 'Bin counts.'},
      {name: 'selectionTruncated, ...', type: 'boolean', description: 'Capacity flags.'}
    ],
    example: `import {GPURegionStatisticsReadback} from '${ANALYSIS}';

const readback = new GPURegionStatisticsReadback(device, {binCount: 16});

// each frame, after encoding GPURegionStatistics into \`encoder\`
const ticket = readback.encodeRead(encoder, summaryBuffer);
device.submit(encoder.finish());
if (ticket) {
  const {selectedCount, mean, histogram} = await readback.read(ticket);
  updateLegend(selectedCount, mean, histogram);
}

readback.destroy();`,
    preview: {kind: 'pipeline', seed: 52, caption: 'GPU summary slots drained a frame later.'},
    related: ['region-statistics', 'region-mask'],
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  {
    id: 'region-mask',
    symbol: 'GPURegionMask',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Region mask',
    summary: 'Marks which points fall inside a rectangle or polygon, as a reusable uint32 mask.',
    explanation:
      "Use it to filter once and reuse the mask in density, statistics or rendering. The shape is a per-frame view, so dragging is free. Mask feeds `GPURegionStatistics` as `kind: 'mask'`.",
    dataTypes: {input: ['points', 'polygons'], output: ['table']},
    useCases: ['relate', 'summarize'],
    domains: ['data-engineering', 'cartography'],
    params: [
      {
        name: 'positions',
        type: 'GPUFloat32Positions',
        required: true,
        description: 'Point locations.'
      },
      {
        name: 'region',
        type: 'GPURegionShape',
        required: true,
        description: 'Rectangle or polygon.'
      },
      {name: 'spatialContext', type: 'GPUSpatialContext', description: 'Frame of the points.'},
      {
        name: 'regionSpatialContext',
        type: 'GPUSpatialContext',
        description: 'Frame of the region.'
      },
      {
        name: 'output',
        type: 'GPURegionMaskOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'mask', type: U32, description: '1 inside, 0 outside, per point.'},
      {name: 'status', type: 'GPUOverflowStatusPort', description: 'Capacity and lon/lat flags.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPURegionMask, GPUParameterBuffer} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'mask'});
const bounds = new GPUParameterBuffer(device, {id: 'bounds', format: 'float32', length: 4});
// positions, mask, status: graph views
graph.add(
  new GPURegionMask({
    positions,
    region: {kind: 'rectangle', bounds: bounds.importToGraph(graph)},
    output: {mask, status}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'points', seed: 53, caption: 'Points inside the shape light up.'},
    related: ['pick-region-mask', 'region-statistics'],
    lensScenario: 'gpu-spatial-analysis.region-mask',
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  {
    id: 'pick-region-mask',
    symbol: 'GPUPickRegionMask',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Pick region mask',
    summary: 'Turns a screen-space region pick into a per-object mask.',
    explanation:
      'Use it to select what is visible inside a drawn rectangle: render with picking, then convert object ids to a mask. The mask then drives statistics or highlights, with no CPU readback.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['relate', 'summarize'],
    domains: ['cartography', 'data-engineering'],
    params: [
      {
        name: 'picks',
        type: U32,
        required: true,
        description: 'Region-pick result [count, overflow, ids...].'
      },
      {name: 'batchIndex', type: 'number', description: 'Only objects from this batch.'},
      {
        name: 'output',
        type: 'GPURegionMaskOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'mask', type: U32, description: '1 for picked objects.'},
      {name: 'status', type: 'GPUOverflowStatusPort', description: 'Pick overflow flag.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUPickRegionMask} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'pick-mask'});
// picks: output of GPUIndexPickingTarget.addRegionPass; mask, status: graph views
graph.add(
  new GPUPickRegionMask({
    picks,
    batchIndex: 0,
    output: {mask, status}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'polygons', seed: 54, caption: 'Rubber-band selection becomes a mask.'},
    related: ['region-mask', 'region-statistics'],
    lensScenario: 'gpu-spatial-analysis.pick-region-mask',
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  {
    id: 'zonal-statistics',
    symbol: 'GPUZonalStatistics',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Zonal statistics',
    summary:
      'Count, sum, mean, min, max and deviation of points or raster values within each polygon.',
    explanation:
      "Use it to roll point events or raster cells up to zones such as tracts or parcels. Polygons are indexed and tested on the GPU, then reduced per zone. `sumOrder: 'sorted'` makes sums bit-reproducible.",
    dataTypes: {input: ['points', 'polygons', 'raster'], output: ['table']},
    useCases: ['summarize', 'relate'],
    domains: ['data-engineering', 'demographics'],
    params: [
      {
        name: 'features',
        type: 'GPUZonalStatisticsFeatures',
        required: true,
        description: 'Polygons, or precomputed point-to-feature rows.'
      },
      {name: 'positions', type: 'GPUFloat32Positions', description: 'Point locations.'},
      {name: 'values', type: F32, description: 'Values to aggregate.'},
      {name: 'weights', type: F32, description: 'Per-point weight.'},
      {name: 'areas', type: F32, description: 'Feature areas for density outputs.'},
      {
        name: 'sumOrder',
        type: "'atomic' | 'sorted'",
        description: 'Reproducibility of float sums.'
      },
      {name: 'spatialContext', type: 'GPUSpatialContext', description: 'Frame of the points.'},
      {
        name: 'output',
        type: 'GPUZonalStatisticsOutput',
        required: true,
        description: 'Any subset of statistics.'
      }
    ],
    outputs: [
      {name: 'counts', type: U32, description: 'Points per zone.'},
      {name: 'sums / means', type: F32, description: 'Value sum and mean per zone.'},
      {name: 'minima / maxima', type: F32, description: 'Extremes per zone.'},
      {name: 'standardDeviations', type: F32, description: 'Spread per zone.'},
      {name: 'densities', type: F32, description: 'Count per unit area.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUZonalStatistics} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'zonal'});
// tract geometry views, point positions and values, outputs: graph views
graph.add(
  new GPUZonalStatistics({
    features: {kind: 'polygons', positions: tractPositions,
      featureOffsets, polygonOffsets, ringOffsets, candidateCapacity: 1 << 20},
    positions: pointPositions,
    values,
    sumOrder: 'sorted',
    output: {counts, means, standardDeviations}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'polygon-overlay',
      seed: 55,
      caption: 'Points rolled up to the polygons that contain them.'
    },
    related: ['region-statistics', 'focal-statistics', 'line-length-per-polygon'],
    lensScenario: 'gpu-spatial-analysis.zonal-statistics',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- regionalization
  {
    id: 'spatial-weights-minimum-spanning-tree',
    symbol: 'GPUSpatialWeightsMinimumSpanningTree',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Minimum spanning tree',
    summary:
      'Builds the minimum spanning tree of a contiguity graph weighted by attribute dissimilarity.',
    explanation:
      'Use it as the first step of SKATER regionalization. Edges join neighbors and cost their attribute distance; the tree keeps the cheapest connected set. Disconnected islands get separate component labels.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['relate', 'find-patterns'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      {name: 'weights', type: WEIGHTS, required: true, description: 'Contiguity graph.'},
      {name: 'values', type: F32, required: true, description: 'Attribute columns, row-major.'},
      {name: 'columnCount', type: 'number', description: 'Attribute columns per place.'},
      {name: 'standardize', type: 'boolean', description: 'Z-score columns first.'},
      {
        name: 'output',
        type: 'GPUSpatialWeightsMinimumSpanningTreeOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'treeEdgeFlags', type: U32, description: '1 for edges kept in the tree.'},
      {name: 'componentLabels', type: U32, description: 'Connected component per place.'},
      {name: 'edgeDissimilarities', type: F32, description: 'Optional cost of each tree edge.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUSpatialWeightsMinimumSpanningTree} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'mst'});
// weights: contiguity from GPUContiguityWeights; values, outputs: graph views
graph.add(
  new GPUSpatialWeightsMinimumSpanningTree({
    weights,
    values,
    columnCount: 3,
    standardize: true,
    output: {treeEdgeFlags, componentLabels}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'spatial-weights',
      seed: 61,
      caption: 'The cheapest tree through neighboring polygons.'
    },
    related: ['skater-regions', 'ward-regions'],
    lensScenario: 'gpu-spatial-analysis.spatial-weights-minimum-spanning-tree',
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  {
    id: 'skater-regions',
    symbol: 'GPUSkaterRegions',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'SKATER regions',
    summary: 'Partitions places into contiguous, homogeneous regions by cutting a spanning tree.',
    explanation:
      'Use it to build regions that stay connected and similar inside. It repeatedly cuts the tree edge that most reduces within-region variance until `regionCount` is reached. A minimum region size guards against slivers.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['find-patterns', 'classify'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      {name: 'weights', type: WEIGHTS, required: true, description: 'Contiguity graph.'},
      {
        name: 'treeEdgeFlags',
        type: U32,
        required: true,
        description: 'Tree edges from the spanning tree step.'
      },
      {
        name: 'componentLabels',
        type: U32,
        required: true,
        description: 'Components from the spanning tree step.'
      },
      {name: 'values', type: F32, required: true, description: 'Attribute columns, row-major.'},
      {
        name: 'maximumRegionCount',
        type: 'number',
        required: true,
        description: 'Compile-time region cap.'
      },
      {
        name: 'parameters',
        type: U32,
        required: true,
        description: 'regionCount and minimumRegionSize.'
      },
      {name: 'columnCount', type: 'number', description: 'Attribute columns per place.'},
      {
        name: 'output',
        type: 'GPUSkaterRegionsOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {name: 'labels', type: U32, description: 'Region id per place.'},
      {name: 'regionCount', type: U32, description: 'Regions produced.'},
      {name: 'cutEdges', type: U32, description: 'Tree edges cut, in order.'},
      {name: 'cutGains', type: F32, description: 'Variance reduction of each cut.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUSkaterRegions, GPUParameterBuffer,
  getGPUSkaterRegionsParameterValues} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'skater'});
const settings = {regionCount: 8, minimumRegionSize: 5};
const parameters = new GPUParameterBuffer(device, {
  id: 'skater-parameters', format: 'uint32', length: 2,
  values: getGPUSkaterRegionsParameterValues(settings)
});
// weights, tree flags, components, values, outputs: graph views
graph.add(new GPUSkaterRegions({
  weights, treeEdgeFlags, componentLabels, values, columnCount: 3,
  maximumRegionCount: 8, parameters: parameters.importToGraph(graph),
  output: {labels, regionCount, cutEdges, cutGains}
}));
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'polygons',
      seed: 62,
      caption: 'Tracts grouped into connected, similar regions.'
    },
    related: [
      'spatial-weights-minimum-spanning-tree',
      'region-partition-evaluation',
      'ward-regions'
    ],
    lensScenario: 'gpu-spatial-analysis.skater-regions',
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  {
    id: 'ward-regions',
    symbol: 'GPUWardRegions',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Ward regions',
    summary: 'Agglomerative Ward clustering constrained to contiguous neighbors.',
    explanation:
      'Use it for hierarchical regions that merge the pair raising variance least. Only adjacent regions merge, so every region stays connected. `mutual-nearest` merges many pairs per round for speed.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['find-patterns', 'classify'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      {name: 'weights', type: WEIGHTS, required: true, description: 'Contiguity graph.'},
      {name: 'values', type: F32, required: true, description: 'Attribute columns, row-major.'},
      {
        name: 'targetRegionCount',
        type: 'number',
        required: true,
        description: 'Regions to stop at.'
      },
      {name: 'maximumIterations', type: 'number', required: true, description: 'Merge round cap.'},
      {
        name: 'parameters',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Seed, from getGPUWardRegionsParameterValues.'
      },
      {
        name: 'mergeSchedule',
        type: "'sequential' | 'mutual-nearest'",
        description: 'How merges are batched.'
      },
      {name: 'columnCount', type: 'number', description: 'Attribute columns per place.'},
      {name: 'standardize', type: 'boolean', description: 'Z-score columns first.'}
    ],
    outputs: [
      {name: 'labels', type: U32, description: 'Region id per place.'},
      {name: 'objective', type: F32, description: 'Within-region sum of squares.'},
      {name: 'status', type: U32, description: 'Iterations and convergence.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUWardRegions, GPUParameterBuffer, getGPUWardRegionsParameterValues,
  GPU_WARD_REGIONS_PARAMETER_LENGTH} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'ward'});
const parameters = new GPUParameterBuffer(device, {
  id: 'ward-parameters', format: 'uint32', length: GPU_WARD_REGIONS_PARAMETER_LENGTH,
  values: getGPUWardRegionsParameterValues({seed: 1})
});
// weights, values, outputs: graph views
graph.add(new GPUWardRegions({
  weights, values, columnCount: 3, standardize: true,
  targetRegionCount: 8, maximumIterations: 512, mergeSchedule: 'mutual-nearest',
  parameters: parameters.importToGraph(graph),
  output: {labels, objective, status}
}));
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'polygons', seed: 63, caption: 'Neighbors merge until eight regions remain.'},
    related: ['skater-regions', 'azp-regions', 'max-p-regions'],
    lensScenario: 'gpu-spatial-analysis.ward-regions',
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  {
    id: 'azp-regions',
    symbol: 'GPUAZPRegions',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'AZP regions',
    summary:
      'Improves a region partition by moving border places to reduce within-region variance.',
    explanation:
      'Use it to polish a starting partition (from SKATER or Ward) under contiguity. Moves are batched and deterministic, bounded by `maximumIterations`. Optional extensive thresholds keep regions above a minimum.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['find-patterns', 'classify'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      {name: 'weights', type: WEIGHTS, required: true, description: 'Contiguity graph.'},
      {name: 'values', type: F32, required: true, description: 'Attribute columns, row-major.'},
      {
        name: 'initialLabels',
        type: U32,
        required: true,
        description: 'Starting region id per place.'
      },
      {
        name: 'regionCapacity',
        type: 'number',
        required: true,
        description: 'Compile-time region cap.'
      },
      {name: 'maximumIterations', type: 'number', required: true, description: 'Move round cap.'},
      {
        name: 'parameters',
        type: U32,
        required: true,
        description: 'Seed, from getGPUAZPRegionsParameterValues.'
      },
      {name: 'extensiveValues', type: F32, description: 'Summed attribute for a region floor.'},
      {name: 'extensiveThreshold', type: F32, description: 'Minimum summed attribute per region.'}
    ],
    outputs: [
      {name: 'labels', type: U32, description: 'Improved region id per place.'},
      {name: 'objective', type: F32, description: 'Within-region sum of squares.'},
      {name: 'status', type: U32, description: 'Iterations and convergence.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUAZPRegions, GPUParameterBuffer, getGPUAZPRegionsParameterValues,
  GPU_AZP_REGIONS_PARAMETER_LENGTH} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'azp'});
const parameters = new GPUParameterBuffer(device, {
  id: 'azp-parameters', format: 'uint32', length: GPU_AZP_REGIONS_PARAMETER_LENGTH,
  values: getGPUAZPRegionsParameterValues({seed: 9})
});
// weights, values, initialLabels (from SKATER), outputs: graph views
graph.add(new GPUAZPRegions({
  weights, values, columnCount: 3, initialLabels,
  regionCapacity: 8, maximumIterations: 64,
  parameters: parameters.importToGraph(graph),
  output: {labels, objective, status}
}));
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'polygons', seed: 64, caption: 'Border tracts swap regions to tighten them.'},
    related: ['skater-regions', 'ward-regions', 'region-partition-evaluation'],
    lensScenario: 'gpu-spatial-analysis.azp-regions',
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  {
    id: 'max-p-regions',
    symbol: 'GPUMaxPRegions',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Max-p regions',
    summary: 'Finds as many contiguous regions as possible while each meets a minimum threshold.',
    explanation:
      'Use it when regions must reach a floor, such as 10,000 people, and you want the most regions. It grows regions from several random starts, then improves them. The best start wins.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['find-patterns', 'classify'],
    domains: ['demographics', 'spatial-statistics'],
    params: [
      {name: 'weights', type: WEIGHTS, required: true, description: 'Contiguity graph.'},
      {name: 'values', type: F32, required: true, description: 'Similarity attribute columns.'},
      {
        name: 'extensiveValues',
        type: F32,
        required: true,
        description: 'Attribute that must reach the threshold.'
      },
      {
        name: 'threshold',
        type: F32,
        required: true,
        description: 'Minimum extensive total per region.'
      },
      {
        name: 'parameters',
        type: U32,
        required: true,
        description: 'Seed, from getGPUMaxPRegionsParameterValues.'
      },
      {name: 'startCount', type: 'number', description: 'Random starts to try.'},
      {name: 'maximumGrowthSteps', type: 'number', description: 'Region growth cap.'},
      {name: 'maximumIterations', type: 'number', description: 'Improvement round cap.'}
    ],
    outputs: [
      {name: 'labels', type: U32, description: 'Region id per place.'},
      {name: 'objective', type: F32, description: 'Within-region sum of squares.'},
      {name: 'status', type: U32, description: 'Region count and feasibility.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPUMaxPRegions, GPUParameterBuffer, getGPUMaxPRegionsParameterValues,
  GPU_MAX_P_REGIONS_PARAMETER_LENGTH} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'max-p'});
const parameters = new GPUParameterBuffer(device, {
  id: 'max-p-parameters', format: 'uint32', length: GPU_MAX_P_REGIONS_PARAMETER_LENGTH,
  values: getGPUMaxPRegionsParameterValues({seed: 4})
});
// weights, values, population, threshold (one row), outputs: graph views
graph.add(new GPUMaxPRegions({
  weights, values, columnCount: 2, extensiveValues: population, threshold,
  startCount: 16, parameters: parameters.importToGraph(graph),
  output: {labels, objective, status}
}));
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'polygons',
      seed: 65,
      caption: 'As many regions as the population floor allows.'
    },
    related: ['skater-regions', 'azp-regions', 'region-partition-evaluation'],
    lensScenario: 'gpu-spatial-analysis.max-p-regions',
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  {
    id: 'region-partition-evaluation',
    symbol: 'GPURegionPartitionEvaluation',
    kind: 'class',
    importPath: ANALYSIS,
    title: 'Region partition evaluation',
    summary: 'Scores a region partition: sizes, within-region sum of squares and contiguity.',
    explanation:
      'Use it to compare candidate partitions from different methods or parameters. Lower within-region sum of squares means more homogeneous regions. Pass weights to also check that every region is connected.',
    dataTypes: {input: ['table', 'spatial-weights'], output: ['table']},
    useCases: ['model', 'classify'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      {name: 'values', type: F32, required: true, description: 'Attribute columns, row-major.'},
      {name: 'labels', type: U32, required: true, description: 'Region id per place.'},
      {name: 'columnCount', type: 'number', description: 'Attribute columns per place.'},
      {name: 'labelCapacity', type: 'number', description: 'Compile-time label cap.'},
      {
        name: 'weights',
        type: 'GPUSpatialWeights | GPUChunkedSpatialWeights',
        description: 'Contiguity graph for connectivity checks.'
      },
      {
        name: 'output',
        type: 'GPURegionPartitionEvaluationOutput',
        required: true,
        description: 'Destination views.'
      }
    ],
    outputs: [
      {
        name: 'summary',
        type: F32,
        description: 'Total sum of squares, within, between, region count.'
      },
      {name: 'regionSizes', type: U32, description: 'Places per region.'},
      {name: 'regionWithinSumOfSquares', type: F32, description: 'Spread inside each region.'}
    ],
    example: `import {GPUCommandGraph} from '${GRAPH}';
import {GPURegionPartitionEvaluation} from '${ANALYSIS}';

const graph = new GPUCommandGraph(device, {id: 'evaluate'});
// values, labels (from any regionalization), weights, outputs: graph views
graph.add(
  new GPURegionPartitionEvaluation({
    values,
    columnCount: 3,
    labels,
    weights,
    output: {summary, regionSizes, regionWithinSumOfSquares}
  })
);
graph.compile().encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'histogram',
      seed: 66,
      caption: 'Region sizes and internal spread at a glance.'
    },
    related: ['skater-regions', 'ward-regions', 'azp-regions', 'max-p-regions'],
    lensScenario: 'gpu-spatial-analysis.region-partition-evaluation',
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  }
];
