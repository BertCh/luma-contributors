// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group: terrain, relief and light. Terrain derivatives, landforms, peaks, viewsheds,
// horizons, solar light, shadows and relief shading.

import type {CatalogEntry, OutputDoc, ParamDoc} from '../schema';

const IMPORT_PATH = '@luma-contributors/analysis/gpu-terrain';
const REFERENCE = 'api/gpu-terrain/operators';
const SETUP_NOTE =
  '// graph, grid, elevation, parameters and the output views are set up as in the Terrain quick start';

const gridParameter: ParamDoc = {
  name: 'grid',
  type: 'GPURasterMetadata',
  required: true,
  description: 'Width, height, cell size and origin from makeGPURasterMetadata.'
};
const elevationParameter: ParamDoc = {
  name: 'elevation',
  type: 'GPURasterBand',
  required: true,
  description: 'Float32 elevation band; invalid or no-data cells are skipped.'
};
const parametersParameter: ParamDoc = {
  name: 'parameters',
  type: 'GraphDataView<"float32">',
  required: true,
  description: 'Per-frame settings, packed by the matching getGPU...ParameterValues.'
};
const cellSizeModeParameter: ParamDoc = {
  name: 'cellSizeMode',
  type: 'GPURasterCellSizeMode | undefined',
  default: "'uniform'",
  description: "Ground cell size: 'uniform' meters, 'geographic' degrees or 'web-mercator'."
};
const validityOutput: OutputDoc = {
  name: 'outputValidity',
  type: 'GraphDataView<"uint32">',
  description: 'Nonzero where the cell has a valid result.'
};

export const entries: CatalogEntry[] = [
  // ---------------------------------------------------------------- terrain derivatives
  {
    id: 'slope-aspect-hillshade',
    symbol: 'GPUTerrainDerivatives',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Slope, aspect and hillshade',
    summary: "Horn's 3x3 slope, aspect and hillshade for an elevation tile.",
    explanation:
      'The first step of suitability, erosion and mapping work. Two Sobel gradients feed one fused shade kernel; the light direction and z factor change per frame without recompiling.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['measure', 'classify'],
    domains: ['hydrology', 'cartography'],
    params: [
      gridParameter,
      elevationParameter,
      parametersParameter,
      {
        name: 'output',
        type: 'GPUTerrainDerivativesOutput',
        required: true,
        description: 'Views for slope, aspect, hillshade and validity.'
      },
      cellSizeModeParameter,
      {
        name: 'slopeUnits',
        type: 'GPUTerrainSlopeUnits | undefined',
        default: "'degrees'",
        description: "Slope unit: 'degrees' or 'percent'."
      },
      {
        name: 'rowDirection',
        type: 'GPURasterRowDirection | undefined',
        default: "'south'",
        description: 'Direction in which the row index increases.'
      },
      {
        name: 'borderMode',
        type: 'GPURasterBorderMode | undefined',
        default: "'clamp'",
        description: 'Sobel treatment at the tile border.'
      }
    ],
    outputs: [
      {
        name: 'slope',
        type: 'GraphDataView<"float32">',
        description: 'Steepness per cell, in slopeUnits.'
      },
      {
        name: 'aspect',
        type: 'GraphDataView<"float32">',
        description: 'Downslope compass direction in degrees.'
      },
      {
        name: 'hillshade',
        type: 'GraphDataView<"float32">',
        description: 'Lit value in 0..1 for the chosen sun.'
      },
      validityOutput
    ],
    example: `import {
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUTerrainDerivatives({
  grid, elevation,
  parameters: parameters.importToGraph(graph),
  output: {slope, aspect, hillshade}
}));

// every frame: move the sun without recompiling
parameters.write(getGPUTerrainDerivativesParameterValues(
  {azimuthDegrees: 315, altitudeDegrees: 45, zFactor: 2}, grid
));`,
    preview: {
      kind: 'terrain',
      caption: 'Hillshade of the Grand Canyon, sun from the northwest.',
      seed: 401
    },
    related: ['relief-shading', 'terrain-curvature', 'terrain-ruggedness'],
    lensScenario: 'gpu-terrain.terrain-derivatives',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'terrain-curvature',
    symbol: 'GPUTerrainCurvature',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Terrain curvature',
    summary: 'Profile, plan, mean, Gaussian and other second-order curvatures of a surface.',
    explanation:
      'Separates ridges, valleys and flow-convergent ground. Choose the derivative estimator with `method`; positive is convex, negative is concave.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['measure', 'find-patterns'],
    domains: ['hydrology', 'cartography'],
    params: [
      gridParameter,
      elevationParameter,
      parametersParameter,
      {
        name: 'method',
        type: 'GPUTerrainCurvatureMethod | undefined',
        description: "Estimator: 'evans-young', 'zevenbergen-thorne' or 'florinsky'."
      },
      {
        name: 'output',
        type: 'GPUTerrainCurvatureOutput',
        required: true,
        description: 'Map of curvature kind to output view, plus ringCurvature.'
      },
      {
        name: 'ringRadii',
        type: 'readonly number[] | undefined',
        description: 'Radii of the multi-radius ring curvature.'
      },
      cellSizeModeParameter,
      {
        name: 'borderMode',
        type: "'clamp' | 'nodata' | undefined",
        description: 'Border treatment of the stencil.'
      }
    ],
    outputs: [
      {
        name: 'curvatures',
        type: 'Partial<Record<GPUTerrainCurvatureKind, GraphDataView<"float32">>>',
        description: 'One float32 view per requested curvature kind.'
      },
      {
        name: 'ringCurvature',
        type: 'GraphDataView<"float32">',
        description: 'Multi-radius ring curvature.'
      },
      validityOutput
    ],
    example: `import {
  GPUTerrainCurvature,
  getGPUTerrainCurvatureParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
// profile, plan: float32 views on the graph
graph.add(new GPUTerrainCurvature({
  grid, elevation,
  parameters: parameters.importToGraph(graph),
  method: 'florinsky',
  output: {curvatures: {profile, plan}}
}));
parameters.write(getGPUTerrainCurvatureParameterValues({zFactor: 1}, grid));`,
    preview: {
      kind: 'terrain',
      caption: 'Convex crests positive, concave hollows negative.',
      seed: 402
    },
    related: ['slope-aspect-hillshade', 'geomorphons', 'terrain-critical-points'],
    lensScenario: 'gpu-terrain.terrain-curvature',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'terrain-ruggedness',
    symbol: 'GPUTerrainRuggedness',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'TPI, TRI and roughness',
    summary: 'gdaldem-compatible TPI, TRI and roughness over a 3x3 window in one pass.',
    explanation:
      'Best for quick habitat and hazard roughness layers. All three indices share one fused kernel; `edgeMode` picks the gdaldem border behavior.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['measure'],
    domains: ['ecology', 'hydrology'],
    params: [
      gridParameter,
      elevationParameter,
      {
        name: 'output',
        type: 'GPUTerrainRuggednessOutput',
        required: true,
        description: 'Views for TPI, TRI, roughness and validity.'
      },
      {
        name: 'terrainRuggednessAlgorithm',
        type: 'GPUTerrainRuggednessAlgorithm | undefined',
        default: "'riley'",
        description: "TRI form: 'riley' (root of squared sums) or 'wilson'."
      },
      {
        name: 'edgeMode',
        type: 'GPUTerrainRuggednessEdgeMode | undefined',
        default: "'nodata'",
        description: "Border cells: 'nodata' or 'extrapolate'."
      }
    ],
    outputs: [
      {
        name: 'topographicPositionIndex',
        type: 'GraphDataView<"float32">',
        description: 'Cell minus the mean of its 8 neighbors.'
      },
      {
        name: 'terrainRuggednessIndex',
        type: 'GraphDataView<"float32">',
        description: 'Neighbor-difference ruggedness (Riley or Wilson).'
      },
      {
        name: 'roughness',
        type: 'GraphDataView<"float32">',
        description: 'Largest minus smallest value in the window.'
      },
      validityOutput
    ],
    example: `import {GPUTerrainRuggedness} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUTerrainRuggedness({
  grid, elevation,
  terrainRuggednessAlgorithm: 'riley',
  output: {terrainRuggednessIndex, roughness}
}));`,
    preview: {kind: 'terrain', caption: 'Rough ground bright, plains dark.', seed: 403},
    related: ['vector-ruggedness', 'topographic-position', 'slope-aspect-hillshade'],
    lensScenario: 'gpu-terrain.terrain-ruggedness',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'vector-ruggedness',
    symbol: 'GPUTerrainVectorRuggedness',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Vector ruggedness (VRM)',
    summary: 'Vector ruggedness measure: how much surface normals disagree in a window.',
    explanation:
      'Best for roughness that ignores plain steepness (Sappington et al. 2007). The window radius is read per frame, so changing it never recompiles.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['measure'],
    domains: ['ecology', 'hydrology'],
    params: [
      gridParameter,
      elevationParameter,
      parametersParameter,
      {
        name: 'maximumRadius',
        type: 'number | undefined',
        description: 'Largest window half-width in cells; fixed at compile time.'
      },
      {
        name: 'output',
        type: 'GPUTerrainVectorRuggednessOutput',
        required: true,
        description: 'Views for vectorRuggedness and validity.'
      },
      cellSizeModeParameter,
      {
        name: 'borderMode',
        type: 'GPURasterBorderMode | undefined',
        description: 'Window treatment at the tile border.'
      }
    ],
    outputs: [
      {
        name: 'vectorRuggedness',
        type: 'GraphDataView<"float32">',
        description: 'VRM in 0..1; 0 is smooth, near 1 is very rugged.'
      },
      validityOutput
    ],
    example: `import {
  GPUTerrainVectorRuggedness,
  getGPUTerrainVectorRuggednessParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUTerrainVectorRuggedness({
  grid, elevation, maximumRadius: 7,
  parameters: parameters.importToGraph(graph),
  output: {vectorRuggedness}
}));
parameters.write(getGPUTerrainVectorRuggednessParameterValues({radius: 3}, grid));`,
    preview: {
      kind: 'terrain',
      caption: 'Chaotic slopes light up; smooth slopes stay dark.',
      seed: 404
    },
    related: ['terrain-ruggedness', 'topographic-position'],
    lensScenario: 'gpu-terrain.terrain-vector-ruggedness',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'terrain-contours',
    symbol: 'GPUTerrainContours',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Terrain contours',
    summary: 'Marching-squares contour segments for several elevation levels into bounded buffers.',
    explanation:
      'Best for live contour lines drawn straight from GPU buffers. Each level owns a vertex buffer and a count; one flag reports overflow.',
    dataTypes: {input: ['terrain'], output: ['lines']},
    useCases: ['transform', 'classify'],
    domains: ['cartography', 'hydrology'],
    params: [
      gridParameter,
      elevationParameter,
      {
        name: 'levels',
        type: 'readonly GPUTerrainContourLevel[]',
        required: true,
        description: 'Per level: elevation (number or view), vertices, status, optional draw.'
      },
      {
        name: 'output',
        type: 'GPUTerrainContoursOutput',
        required: true,
        description: 'Holds status.overflow, one aggregate flag.'
      }
    ],
    outputs: [
      {
        name: 'status',
        type: 'GPUTerrainContoursStatus',
        description: 'Overflow flag; nonzero if any level exceeded capacity.'
      }
    ],
    example: `import {GPUTerrainContours} from '${IMPORT_PATH}';

${SETUP_NOTE}
// vertices*, status*: views sized for the segments you expect
graph.add(new GPUTerrainContours({
  grid, elevation,
  levels: [
    {level: 500, vertices: vertices500, status: status500},
    {level: 600, vertices: vertices600, status: status600}
  ],
  output: {status: {overflow}}
}));`,
    preview: {kind: 'contours', caption: 'Contour lines across the Grand Canyon.', seed: 405},
    related: ['slope-aspect-hillshade', 'relief-shading'],
    lensScenario: 'gpu-terrain.terrain-contours',
    reference: REFERENCE,
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- landforms and position
  {
    id: 'geomorphons',
    symbol: 'GPUGeomorphons',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Geomorphons',
    summary: 'Classifies each cell into one of ten landforms from its lines of sight.',
    explanation:
      'Best for scale-aware landform maps: peak, ridge, shoulder, spur, slope, hollow, footslope, valley, pit and flat. Eight rays compare higher and lower horizons, following GRASS `r.geomorphon` (Jasiewicz and Stepinski 2013).',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['classify', 'find-patterns'],
    domains: ['hydrology', 'ecology'],
    params: [
      gridParameter,
      elevationParameter,
      parametersParameter,
      {
        name: 'searchRadius',
        type: 'number',
        required: true,
        description: 'Longest line of sight, in cells.'
      },
      {
        name: 'skipRadius',
        type: 'number | undefined',
        default: '0',
        description: 'Cells next to the center ignored along every ray.'
      },
      {
        name: 'comparison',
        type: 'GPUGeomorphonComparison | undefined',
        default: "'anglev1'",
        description: "Zenith/nadir reduction: 'anglev1', 'anglev2' or 'anglev2-distance'."
      },
      {
        name: 'output',
        type: 'GPUGeomorphonsOutput',
        required: true,
        description: 'Views for forms, ternary, pattern and validity.'
      },
      cellSizeModeParameter
    ],
    outputs: [
      {
        name: 'forms',
        type: 'GraphDataView<"uint32">',
        description: 'Landform code 1..10 (GPU_GEOMORPHON_FORMS).'
      },
      {
        name: 'ternary',
        type: 'GraphDataView<"uint32">',
        description: 'Ternary pattern id of the eight rays.'
      },
      {
        name: 'pattern',
        type: 'GraphDataView<"uint32">',
        description: 'Packed pattern of the eight rays.'
      },
      validityOutput
    ],
    example: `import {
  GPUGeomorphons,
  getGPUGeomorphonsParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUGeomorphons({
  grid, elevation,
  parameters: parameters.importToGraph(graph),
  searchRadius: 20, skipRadius: 1,
  output: {forms}
}));
parameters.write(getGPUGeomorphonsParameterValues({flatThresholdDegrees: 1}, grid));`,
    preview: {
      kind: 'classification',
      caption: 'Ten landform classes across a ridged terrain.',
      seed: 406
    },
    related: ['weiss-landforms', 'terrain-critical-points', 'topographic-position'],
    lensScenario: 'gpu-terrain.geomorphons',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'weiss-landforms',
    symbol: 'GPUTerrainWeissLandforms',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Weiss landforms',
    summary: 'Ten landform classes from topographic position at two scales plus slope.',
    explanation:
      'Best for canyons, valleys, ridges and plains from a DEM alone (Weiss 2001). Small- and large-scale TPI are standardized, thresholded into low, mid and high, then combined with slope.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['classify'],
    domains: ['hydrology', 'ecology'],
    params: [
      gridParameter,
      elevationParameter,
      parametersParameter,
      {
        name: 'smallScale',
        type: 'GPUTerrainTopographicPositionScale | undefined',
        default: '{radius: 3}',
        description: 'Small neighborhood in cells.'
      },
      {
        name: 'largeScale',
        type: 'GPUTerrainTopographicPositionScale | undefined',
        default: '{radius: 15}',
        description: 'Large neighborhood in cells.'
      },
      {
        name: 'standardization',
        type: 'GPUTerrainWeissStandardization | undefined',
        default: "'global'",
        description: "TPI standardization: 'global' or 'local'."
      },
      {
        name: 'output',
        type: 'GPUTerrainWeissLandformsOutput',
        required: true,
        description: 'Views for landforms, standardizedPosition and validity.'
      },
      cellSizeModeParameter
    ],
    outputs: [
      {
        name: 'landforms',
        type: 'GraphDataView<"uint32">',
        description: 'Class 1..10 (GPU_TERRAIN_WEISS_LANDFORMS).'
      },
      {
        name: 'standardizedPosition',
        type: 'GraphDataView<"float32">',
        description: 'Standardized small-scale position.'
      },
      validityOutput
    ],
    example: `import {
  GPUTerrainWeissLandforms,
  getGPUTerrainWeissLandformsParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUTerrainWeissLandforms({
  grid, elevation,
  parameters: parameters.importToGraph(graph),
  smallScale: {radius: 3}, largeScale: {radius: 15},
  output: {landforms}
}));
parameters.write(getGPUTerrainWeissLandformsParameterValues({slopeThresholdDegrees: 5}, grid));`,
    preview: {
      kind: 'classification',
      caption: 'Valleys, slopes and ridges classified from one DEM.',
      seed: 407
    },
    related: ['geomorphons', 'topographic-position'],
    lensScenario: 'gpu-terrain.terrain-weiss-landforms',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'topographic-position',
    symbol: 'GPUTerrainTopographicPosition',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Multiscale topographic position',
    summary:
      'TPI over square annuli, deviation from mean elevation and the scale of maximum deviation.',
    explanation:
      'Shows which scale a ridge or hollow belongs to (Lindsay et al. 2015). Exact summed-area tables make every radius cost the same.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['measure', 'find-patterns'],
    domains: ['ecology', 'hydrology'],
    params: [
      gridParameter,
      elevationParameter,
      {
        name: 'scales',
        type: 'readonly GPUTerrainTopographicPositionScale[]',
        required: true,
        description: 'Window radii in cells, optionally with innerRadius.'
      },
      {
        name: 'quantum',
        type: 'number | undefined',
        description: 'Elevation quantum of the exact summed-area table, a power of two.'
      },
      {
        name: 'output',
        type: 'GPUTerrainTopographicPositionOutput',
        required: true,
        description: 'Views for TPI, DEV, DEVmax, scale and validity.'
      }
    ],
    outputs: [
      {
        name: 'topographicPositionIndex',
        type: 'GraphDataView<"float32">',
        description: 'Cell minus the annulus mean.'
      },
      {
        name: 'deviationFromMean',
        type: 'GraphDataView<"float32">',
        description: 'Standardized deviation from the window mean (DEV).'
      },
      {
        name: 'maximumDeviation',
        type: 'GraphDataView<"float32">',
        description: 'Largest deviation over the scales (DEVmax).'
      },
      {
        name: 'maximumDeviationRadius',
        type: 'GraphDataView<"uint32">',
        description: 'Radius at which DEVmax occurs.'
      }
    ],
    example: `import {GPUTerrainTopographicPosition} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUTerrainTopographicPosition({
  grid, elevation,
  scales: [{radius: 2}, {radius: 8}, {radius: 32}],
  output: {maximumDeviation, maximumDeviationRadius}
}));`,
    preview: {
      kind: 'terrain',
      caption: 'Ridges and hollows, strongest at their own scale.',
      seed: 408
    },
    related: ['weiss-landforms', 'simple-local-relief', 'multi-scale-relief'],
    lensScenario: 'gpu-terrain.terrain-topographic-position',
    reference: REFERENCE,
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- peaks and features
  {
    id: 'terrain-critical-points',
    symbol: 'GPUTerrainCriticalPoints',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Critical points',
    summary: 'Labels every pixel as regular, peak, pit, saddle, boundary or no-data.',
    explanation:
      'Finds the topological skeleton of a surface. Counting sign changes of the height difference around each pixel ring separates peaks, pits and saddles.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['find-patterns', 'classify'],
    domains: ['hydrology', 'ecology'],
    params: [
      gridParameter,
      elevationParameter,
      {
        name: 'connectivity',
        type: '8 | 6 | undefined',
        default: '8',
        description: 'Ring: 8 (Peucker-Douglas) or 6 (Freudenthal triangulation).'
      },
      {
        name: 'output',
        type: 'GPUTerrainCriticalPointsOutput',
        required: true,
        description: 'Views for classes, signChanges and counts.'
      }
    ],
    outputs: [
      {
        name: 'classes',
        type: 'GraphDataView<"uint32">',
        description: 'Class code 0..5 (GPU_TERRAIN_CRITICAL_POINT).'
      },
      {
        name: 'signChanges',
        type: 'GraphDataView<"uint32">',
        description: 'Ring sign changes per pixel.'
      },
      {name: 'counts', type: 'GraphDataView<"uint32">', description: 'Pixels per class.'}
    ],
    example: `import {GPUTerrainCriticalPoints} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUTerrainCriticalPoints({
  grid, elevation, connectivity: 8,
  output: {classes, counts}
}));`,
    preview: {
      kind: 'points',
      caption: 'Peaks, pits and saddles on a bumpy surface.',
      seed: 409
    },
    related: ['terrain-summits', 'peak-prominence', 'geomorphons'],
    lensScenario: 'gpu-terrain.terrain-critical-points',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'terrain-summits',
    symbol: 'GPUTerrainSummits',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Summits',
    summary: 'DEM summits: the highest point of a metric disc that stands out from its ring.',
    explanation:
      'Lists real peaks and ignore noise bumps. A pixel is a summit if it tops its disc and drops by at least `minimumDrop` to the boundary; the radius is per frame.',
    dataTypes: {input: ['terrain'], output: ['points', 'raster']},
    useCases: ['find-patterns'],
    domains: ['hydrology', 'ecology'],
    params: [
      gridParameter,
      elevationParameter,
      parametersParameter,
      cellSizeModeParameter,
      {
        name: 'maximumRadiusPixels',
        type: 'number | undefined',
        default: '16',
        description: 'Largest disc radius in cells (1 to 64).'
      },
      {
        name: 'incompleteNeighborhood',
        type: "'reject' | 'ignore' | undefined",
        default: "'reject'",
        description: 'Discs that leave the grid or hit no-data: reject or ignore.'
      },
      {
        name: 'pyramid',
        type: 'GPURasterExtremaPyramidOutput | undefined',
        description: 'Shared extrema pyramid to reuse.'
      },
      {
        name: 'output',
        type: 'GPUTerrainSummitsOutput',
        required: true,
        description: 'Mask, drop and compact summit list.'
      }
    ],
    outputs: [
      {
        name: 'summitMask',
        type: 'GraphDataView<"uint32">',
        description: 'Nonzero at summit pixels.'
      },
      {
        name: 'drop',
        type: 'GraphDataView<"float32">',
        description: 'Drop to the boundary ring per pixel.'
      },
      {name: 'summits', type: 'GPUCompactOutput', description: 'Compact list of summit pixels.'},
      {
        name: 'summitDrop',
        type: 'GraphDataView<"float32">',
        description: 'Drop for each listed summit.'
      }
    ],
    example: `import {
  GPUTerrainSummits,
  getGPUTerrainSummitsParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUTerrainSummits({
  grid, elevation, maximumRadiusPixels: 16,
  parameters: parameters.importToGraph(graph),
  output: {summitMask, drop}
}));
parameters.write(getGPUTerrainSummitsParameterValues({radius: 8, minimumDrop: 30}, grid));`,
    preview: {kind: 'terrain', caption: 'Summits marked on a noisy mountain range.', seed: 410},
    related: ['peak-prominence', 'peak-snap', 'terrain-critical-points'],
    lensScenario: 'gpu-terrain.terrain-summits',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'peak-prominence',
    symbol: 'GPUTerrainPeakProminence',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Peak prominence',
    summary:
      'Topographic prominence of every peak, key saddle and parent peak, entirely on the GPU.',
    explanation:
      'Ranks peaks by how far they rise above the saddle that connects them to higher ground. It builds the upper excursion-set merge tree (elder rule, zero-dimensional persistence).',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['measure', 'find-patterns'],
    domains: ['hydrology', 'ecology'],
    params: [
      gridParameter,
      elevationParameter,
      {
        name: 'output',
        type: 'GPUTerrainPeakProminenceOutput',
        required: true,
        description: 'Views for prominence, keySaddle and parentPeak.'
      }
    ],
    outputs: [
      {
        name: 'prominence',
        type: 'GraphDataView<"float32">',
        description: 'Prominence at each peak pixel, in elevation units.'
      },
      {
        name: 'keySaddle',
        type: 'GraphDataView<"float32">',
        description: 'Elevation of the key saddle.'
      },
      {
        name: 'parentPeak',
        type: 'GraphDataView<"uint32">',
        description: 'Cell index of the parent peak.'
      }
    ],
    example: `import {GPUTerrainPeakProminence} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUTerrainPeakProminence({
  grid, elevation,
  output: {prominence, keySaddle, parentPeak}
}));`,
    preview: {kind: 'terrain', caption: 'Peaks sized by prominence, not height.', seed: 411},
    related: ['terrain-summits', 'peak-snap', 'terrain-critical-points'],
    lensScenario: 'gpu-terrain.terrain-peak-prominence',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'peak-snap',
    symbol: 'GPUTerrainPeakSnap',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Peak snap',
    summary: 'Snaps catalogue peak points to the DEM summit near them.',
    explanation:
      'Reconciles peak databases with the terrain model. Each candidate searches a disc and moves only if the move and height change are within limits; a status code says what happened.',
    dataTypes: {input: ['terrain', 'points'], output: ['points']},
    useCases: ['relate', 'transform'],
    domains: ['hydrology', 'cartography'],
    params: [
      gridParameter,
      elevationParameter,
      {
        name: 'candidates',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Candidate positions to snap.'
      },
      {
        name: 'candidateHeights',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Catalogue heights, to reject moves with large height change.'
      },
      {
        name: 'candidateRadii',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Per-candidate search radius in ground meters.'
      },
      {
        name: 'maximumRadiusPixels',
        type: 'number | undefined',
        description: 'Largest search disc in cells.'
      },
      parametersParameter,
      {
        name: 'output',
        type: 'GPUTerrainPeakSnapOutput',
        required: true,
        description: 'Views for positions, heights, snapStatus and snapDistance.'
      }
    ],
    outputs: [
      {name: 'positions', type: 'GraphDataView<"float32x2">', description: 'Snapped positions.'},
      {
        name: 'heights',
        type: 'GraphDataView<"float32">',
        description: 'DEM height at each snapped position.'
      },
      {
        name: 'snapStatus',
        type: 'GraphDataView<"uint32">',
        description: 'Code from GPU_TERRAIN_PEAK_SNAP_STATUS.'
      },
      {name: 'snapDistance', type: 'GraphDataView<"float32">', description: 'Distance moved.'}
    ],
    example: `import {
  GPUTerrainPeakSnap,
  getGPUTerrainPeakSnapParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUTerrainPeakSnap({
  grid, elevation, candidates, candidateHeights,
  parameters: parameters.importToGraph(graph),
  output: {positions, heights, snapStatus}
}));
parameters.write(getGPUTerrainPeakSnapParameterValues(
  {radius: 150, maximumMove: 300, maximumHeightChange: 50}, grid
));`,
    preview: {
      kind: 'points',
      caption: 'Catalogue points pulled onto the nearest summit.',
      seed: 412
    },
    related: ['terrain-summits', 'peak-prominence'],
    lensScenario: 'gpu-terrain.terrain-peak-snap',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'profile-peaks',
    symbol: 'GPUProfilePeaks',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Profile peaks',
    summary: 'Local maxima of many 1-D profiles with prominence and sub-sample refinement.',
    explanation:
      'Best for skylines, elevation profiles and any ragged batch of series. Each profile is a CSR span; peaks need a windowed prominence, are refined by a parabola and thinned by non-maximum suppression.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['find-patterns'],
    domains: ['hydrology', 'data-engineering'],
    params: [
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'All samples of all profiles, concatenated.'
      },
      {
        name: 'offsets',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'CSR offsets, profileCount + 1 rows.'
      },
      {
        name: 'validity',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Nonzero samples take part.'
      },
      parametersParameter,
      {
        name: 'window',
        type: 'number',
        required: true,
        description: 'Half-width in samples for the prominence test.'
      },
      {
        name: 'suppressionRadius',
        type: 'number | undefined',
        description: 'Keep only the largest peak within this many samples.'
      },
      {
        name: 'wrap',
        type: 'boolean | undefined',
        description: 'Treat each profile as circular, as for a 360-degree skyline.'
      },
      {
        name: 'output',
        type: 'GPUProfilePeaksOutput',
        required: true,
        description: 'Views for prominence, refinement, mask and compact peaks.'
      }
    ],
    outputs: [
      {
        name: 'prominence',
        type: 'GraphDataView<"float32">',
        description: 'Windowed prominence per sample.'
      },
      {
        name: 'refinedIndex',
        type: 'GraphDataView<"float32">',
        description: 'Sub-sample peak position.'
      },
      {
        name: 'peakMask',
        type: 'GraphDataView<"uint32">',
        description: 'Nonzero at surviving peaks.'
      },
      {name: 'peaks', type: 'GPUCompactOutput', description: 'Compact list of peaks.'}
    ],
    example: `import {
  GPUProfilePeaks,
  getGPUProfilePeaksParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
// values: all profiles end to end; offsets: profileCount + 1 start indices
graph.add(new GPUProfilePeaks({
  values, offsets, window: 8, suppressionRadius: 4, wrap: true,
  parameters: parameters.importToGraph(graph),
  output: {peakMask, refinedIndex}
}));
parameters.write(getGPUProfilePeaksParameterValues({minimumProminence: 0.5}));`,
    preview: {kind: 'histogram', caption: 'Peaks found along a skyline profile.', seed: 413},
    related: ['point-horizon-profile', 'peak-prominence'],
    lensScenario: 'gpu-terrain.profile-peaks',
    reference: REFERENCE,
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- visibility
  {
    id: 'viewshed',
    symbol: 'GPUTerrainViewshed',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Viewshed',
    summary: 'Which cells one observer can see over an elevation tile.',
    explanation:
      'Best for tower siting, scenic views and exposure. Sight lines reach every cell with earth-curvature drop; the observer moves per frame.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['relate', 'measure'],
    domains: ['hydrology', 'three-d'],
    params: [
      gridParameter,
      elevationParameter,
      parametersParameter,
      cellSizeModeParameter,
      {
        name: 'heightReference',
        type: 'GPUTerrainHeightReference | undefined',
        default: "'ground'",
        description: "Observer height is 'ground'-relative or 'absolute'."
      },
      {
        name: 'traversal',
        type: 'GPUTerrainSightLineTraversal | undefined',
        description: 'Sight-line traversal: marching or the extrema pyramid.'
      },
      {
        name: 'toleranceParameters',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Packed tolerance band that yields marginal results.'
      },
      {
        name: 'output',
        type: 'GPUTerrainViewshedOutput',
        required: true,
        description: 'visibility buffer or visibilityTexture.'
      }
    ],
    outputs: [
      {
        name: 'visibility',
        type: 'GraphDataView<"uint32">',
        description: 'Code per cell: hidden, visible, outOfRange, noData, marginal.'
      },
      {
        name: 'visibilityTexture',
        type: 'GraphTextureView<"r32uint" | "rgba32uint">',
        description: 'The same codes as a texture, for rendering.'
      }
    ],
    example: `import {
  GPUTerrainViewshed,
  getGPUTerrainViewshedParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUTerrainViewshed({
  grid, elevation,
  parameters: parameters.importToGraph(graph),
  output: {visibility}
}));

// every frame: drag the observer ([column, row] in pixel space)
parameters.write(getGPUTerrainViewshedParameterValues(
  {observer: [120, 80], observerHeight: 2, maximumDistance: 5000}, grid
));`,
    preview: {
      kind: 'terrain',
      caption: 'Visible ground from one observer, hidden ground in shadow.',
      seed: 414
    },
    related: ['cumulative-viewshed', 'line-of-sight', 'point-horizon-visibility'],
    lensScenario: 'gpu-terrain.terrain-viewshed',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'cumulative-viewshed',
    symbol: 'GPUTerrainCumulativeViewshed',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Cumulative viewshed',
    summary: 'Counts, for every cell, how many observers can see it.',
    explanation:
      'Use it to rank scenic spots, plan sensor networks or find places nobody sees. Observers are processed in batches.',
    dataTypes: {input: ['terrain', 'points'], output: ['raster']},
    useCases: ['summarize', 'relate'],
    domains: ['hydrology', 'urban'],
    params: [
      gridParameter,
      elevationParameter,
      {
        name: 'observers',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Observer [column, row] positions.'
      },
      {
        name: 'observerHeights',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Per-observer eye height.'
      },
      parametersParameter,
      {
        name: 'observersPerDispatch',
        type: 'number | undefined',
        description: 'Observers processed per node; bounds work per submit.'
      },
      cellSizeModeParameter,
      {
        name: 'output',
        type: 'GPUTerrainCumulativeViewshedOutput',
        required: true,
        description: 'visibleCount, and optionally marginalCount.'
      }
    ],
    outputs: [
      {
        name: 'visibleCount',
        type: 'GraphDataView<"uint32">',
        description: 'Observers that see each cell.'
      },
      {
        name: 'marginalCount',
        type: 'GraphDataView<"uint32">',
        description: 'Observers for which the cell is marginal.'
      }
    ],
    example: `import {
  GPUTerrainCumulativeViewshed,
  getGPUTerrainLineOfSightParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
// observers: float32x2 [column, row] per tower
graph.add(new GPUTerrainCumulativeViewshed({
  grid, elevation, observers,
  parameters: parameters.importToGraph(graph),
  output: {visibleCount}
}));
// shares the line-of-sight packer
parameters.write(getGPUTerrainLineOfSightParameterValues(
  {observerHeight: 30, maximumDistance: 20000}, grid
));`,
    preview: {kind: 'hotspots', caption: 'Brighter cells are seen by more towers.', seed: 415},
    related: ['viewshed', 'line-of-sight'],
    lensScenario: 'gpu-terrain.terrain-cumulative-viewshed',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'line-of-sight',
    symbol: 'GPUTerrainLineOfSight',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Line of sight',
    summary: 'Tests many explicit observer-target pairs against an elevation tile.',
    explanation:
      'Best for point-to-point checks such as tower links or peak-to-peak visibility. Every pair is a row, and optional clearance says how far a target could sink and stay visible.',
    dataTypes: {input: ['terrain', 'table'], output: ['table']},
    useCases: ['relate'],
    domains: ['hydrology', 'three-d'],
    params: [
      gridParameter,
      elevationParameter,
      {
        name: 'pairs',
        type: 'GraphDataView<"float32x4">',
        required: true,
        description: 'Rows of [observerColumn, observerRow, targetColumn, targetRow].'
      },
      {
        name: 'pairHeights',
        type: 'GraphDataView<"float32x2"> | undefined',
        description: 'Rows of [observerHeight, targetHeight] overriding settings.'
      },
      parametersParameter,
      {
        name: 'heightReference',
        type: 'GPUTerrainHeightReference | undefined',
        default: "'ground'",
        description: "Observer height is 'ground'-relative or 'absolute'."
      },
      cellSizeModeParameter,
      {
        name: 'output',
        type: 'GPUTerrainLineOfSightOutput',
        required: true,
        description: 'visibility per pair, optional clearance.'
      }
    ],
    outputs: [
      {
        name: 'visibility',
        type: 'GraphDataView<"uint32">',
        description: 'GPU_TERRAIN_VISIBILITY code per pair.'
      },
      {
        name: 'clearance',
        type: 'GraphDataView<"float32">',
        description: 'Meters the target could sink and stay visible.'
      }
    ],
    example: `import {
  GPUTerrainLineOfSight,
  getGPUTerrainLineOfSightParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
// pairs: [observerColumn, observerRow, targetColumn, targetRow] per row
graph.add(new GPUTerrainLineOfSight({
  grid, elevation, pairs,
  parameters: parameters.importToGraph(graph),
  output: {visibility, clearance}
}));
parameters.write(getGPUTerrainLineOfSightParameterValues(
  {observerHeight: 10, targetHeight: 2, curvatureCoefficient: 0.0000000785}, grid
));`,
    preview: {kind: 'lines', caption: 'Sight lines clear or blocked by terrain.', seed: 416},
    related: ['viewshed', 'cumulative-viewshed', 'point-horizon-visibility'],
    lensScenario: 'gpu-terrain.terrain-line-of-sight',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'point-horizon-profile',
    symbol: 'GPUPointHorizonProfile',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Point horizon profile',
    summary: 'The skyline seen from many observers: elevation angle for every azimuth.',
    explanation:
      'Best for panoramas, peak identification and sunrise checks. Each observer casts `azimuthCount` rays over the terrain window and keeps the highest angle.',
    dataTypes: {input: ['terrain', 'points'], output: ['table']},
    useCases: ['measure'],
    domains: ['three-d', 'hydrology'],
    params: [
      gridParameter,
      elevationParameter,
      {
        name: 'observers',
        type: 'GraphDataView<"float32x4">',
        required: true,
        description: 'Rows of [column, row, height, 0].'
      },
      {
        name: 'azimuthCount',
        type: 'number | undefined',
        description: 'Rays per observer around the horizon.'
      },
      {
        name: 'maximumDistance',
        type: 'number',
        required: true,
        description: 'Longest ray in ground meters (compile-time).'
      },
      {
        name: 'cellSize',
        type: 'number',
        required: true,
        description: 'Nominal ground meters per pixel; sizes the step lattice.'
      },
      parametersParameter,
      {
        name: 'output',
        type: 'GPUPointHorizonProfileOutput',
        required: true,
        description: 'Views for tangent, skylineAngle, distance and samples.'
      }
    ],
    outputs: [
      {
        name: 'skylineAngle',
        type: 'GraphDataView<"float32">',
        description: 'Horizon elevation angle per observer and ray.'
      },
      {
        name: 'distance',
        type: 'GraphDataView<"float32">',
        description: 'Distance to the skyline point.'
      },
      {
        name: 'tangent',
        type: 'GraphDataView<"float32">',
        description: 'Best tangent of the angle per ray.'
      }
    ],
    example: `import {
  GPUPointHorizonProfile,
  getGPUPointHorizonProfileParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
// observers: float32x4 rows of [column, row, height, 0]
graph.add(new GPUPointHorizonProfile({
  grid, elevation, observers, azimuthCount: 360,
  maximumDistance: 50000, cellSize: 30,
  parameters: parameters.importToGraph(graph),
  output: {skylineAngle, distance}
}));
parameters.write(getGPUPointHorizonProfileParameterValues(
  {maximumDistance: 50000}, grid
));`,
    preview: {
      kind: 'time-series',
      caption: 'Skyline angle around a full circle of azimuths.',
      seed: 417
    },
    related: ['point-horizon-visibility', 'profile-peaks', 'horizon-angles'],
    lensScenario: 'gpu-terrain.point-horizon-profile',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'point-horizon-visibility',
    symbol: 'GPUPointHorizonVisibility',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Point horizon visibility',
    summary:
      'Classifies targets as hidden, marginal or visible from observers, and flags skyline ones.',
    explanation:
      'Finds which peaks or towers you can see from a place. Each target is tested against the observer horizon with an angular tolerance, then checked against the skyline.',
    dataTypes: {input: ['terrain', 'points'], output: ['table']},
    useCases: ['relate'],
    domains: ['three-d', 'hydrology'],
    params: [
      gridParameter,
      elevationParameter,
      {
        name: 'observers',
        type: 'GraphDataView<"float32x4">',
        required: true,
        description: 'Rows of [column, row, height, 0].'
      },
      {
        name: 'targets',
        type: 'GraphDataView<"float32x4">',
        required: true,
        description: 'Rows of [column, row, height, observerIndex].'
      },
      {
        name: 'maximumDistance',
        type: 'number',
        required: true,
        description: 'Longest ray in ground meters (compile-time).'
      },
      {
        name: 'cellSize',
        type: 'number',
        required: true,
        description: 'Nominal ground meters per pixel; sizes the step lattice.'
      },
      parametersParameter,
      {
        name: 'output',
        type: 'GPUPointHorizonVisibilityOutput',
        required: true,
        description: 'visibility codes and optional details.'
      }
    ],
    outputs: [
      {
        name: 'visibility',
        type: 'GraphDataView<"uint32">',
        description: 'hidden, marginal or visible, with skyline flag.'
      },
      {
        name: 'details',
        type: 'GraphDataView<"float32x4">',
        description: 'Angles and margins behind the decision.'
      }
    ],
    example: `import {
  GPUPointHorizonVisibility,
  getGPUPointHorizonVisibilityParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
// targets: [column, row, height, observerIndex] per peak
graph.add(new GPUPointHorizonVisibility({
  grid, elevation, observers, targets,
  maximumDistance: 80000, cellSize: 30,
  parameters: parameters.importToGraph(graph),
  output: {visibility, details}
}));
parameters.write(getGPUPointHorizonVisibilityParameterValues(
  {maximumDistance: 80000, toleranceDegrees: 0.05}, grid
));`,
    preview: {
      kind: 'lines',
      caption: 'Peaks seen above, or hidden behind, the skyline.',
      seed: 418
    },
    related: ['point-horizon-profile', 'line-of-sight', 'viewshed'],
    lensScenario: 'gpu-terrain.point-horizon-visibility',
    reference: REFERENCE,
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- horizon and shadow
  {
    id: 'horizon-angles',
    symbol: 'GPUTerrainHorizon',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Horizon angles and sky view',
    summary: 'Horizon-angle maps per azimuth, plus sky-view factor and openness.',
    explanation:
      'The shared input for shadows, solar irradiance and sky-view shading. Rays march in `directionCount` sectors to `maximumRadius`; the horizon can be stored as float32 or packed unorm16.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['measure', 'simulate'],
    domains: ['hydrology', 'remote-sensing'],
    params: [
      gridParameter,
      elevationParameter,
      parametersParameter,
      {
        name: 'maximumRadius',
        type: 'number',
        required: true,
        description: 'Longest ray, in cells.'
      },
      {
        name: 'directionCount',
        type: 'number | undefined',
        default: '16',
        description: 'Azimuth sectors, an integer in [4, 64].'
      },
      {
        name: 'algorithm',
        type: 'GPUTerrainHorizonAlgorithm | undefined',
        default: "'march'",
        description: "'march' rays or the 'sweep' scan."
      },
      {
        name: 'horizonFormat',
        type: 'GPUTerrainHorizonFormat | undefined',
        default: "'float32'",
        description: "Horizon storage: 'float32' degrees or 'unorm16'."
      },
      {
        name: 'output',
        type: 'GPUTerrainHorizonOutput',
        required: true,
        description: 'Views for horizon, skyViewFactor, openness and validity.'
      }
    ],
    outputs: [
      {
        name: 'horizon',
        type: 'GraphDataView<"float32"> | GraphDataView<"uint32">',
        description: 'Horizon angle per cell and sector.'
      },
      {
        name: 'skyViewFactor',
        type: 'GraphDataView<"float32">',
        description: 'Visible sky fraction in 0..1.'
      },
      {
        name: 'positiveOpenness',
        type: 'GraphDataView<"float32">',
        description: 'Mean zenith angle to the horizon.'
      },
      {
        name: 'negativeOpenness',
        type: 'GraphDataView<"float32">',
        description: 'Mean nadir angle to the horizon.'
      }
    ],
    example: `import {
  GPUTerrainHorizon,
  getGPUTerrainHorizonParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUTerrainHorizon({
  grid, elevation, directionCount: 16, maximumRadius: 100,
  parameters: parameters.importToGraph(graph),
  output: {horizon, skyViewFactor}
}));
parameters.write(getGPUTerrainHorizonParameterValues({zFactor: 1}, grid));`,
    preview: {
      kind: 'terrain',
      caption: 'Sky-view factor: valleys dark, summits bright.',
      seed: 419
    },
    related: ['solar-shadow-mask', 'solar-irradiance', 'terrain-cast-shadow', 'relief-shading'],
    lensScenario: 'gpu-terrain.terrain-horizon',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'terrain-cast-shadow',
    symbol: 'GPUTerrainCastShadow',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Terrain cast shadow',
    summary: 'Single-sun terrain shadows without a horizon map.',
    explanation:
      'Best for one live sun position. Each cell marches toward the sun and reports soft sun visibility; it is cheaper than a horizon map when the sun moves every frame.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['simulate'],
    domains: ['cartography', 'hydrology'],
    params: [
      gridParameter,
      elevationParameter,
      parametersParameter,
      {
        name: 'maximumRadius',
        type: 'number | undefined',
        description: 'Longest march toward the sun, in cells.'
      },
      cellSizeModeParameter,
      {
        name: 'rowDirection',
        type: 'GPURasterRowDirection | undefined',
        default: "'south'",
        description: 'Direction in which the row index increases.'
      },
      {
        name: 'output',
        type: 'GPUTerrainCastShadowOutput',
        required: true,
        description: 'Views for sunVisibility, horizonAngle and validity.'
      }
    ],
    outputs: [
      {
        name: 'sunVisibility',
        type: 'GraphDataView<"float32">',
        description: '1 lit, 0 shadowed, soft at the edge.'
      },
      {
        name: 'horizonAngle',
        type: 'GraphDataView<"float32">',
        description: 'Horizon angle toward the sun.'
      },
      validityOutput
    ],
    example: `import {
  GPUTerrainCastShadow,
  getGPUTerrainCastShadowParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUTerrainCastShadow({
  grid, elevation, maximumRadius: 200,
  parameters: parameters.importToGraph(graph),
  output: {sunVisibility}
}));
parameters.write(getGPUTerrainCastShadowParameterValues(
  {azimuthDegrees: 250, altitudeDegrees: 12}, grid
));`,
    preview: {kind: 'terrain', caption: 'Long evening shadows cast by ridges.', seed: 420},
    related: ['solar-shadow-mask', 'horizon-angles', 'solar-position'],
    lensScenario: 'gpu-terrain.terrain-cast-shadow',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'solar-shadow-mask',
    symbol: 'GPUSolarShadowMask',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Solar shadow mask',
    summary: 'Soft or hard terrain shadows from a horizon map and a per-frame sun.',
    explanation:
      'Animates the sun over a fixed terrain. Horizon angles are computed once; each frame only compares them with the sun and mixes sun and ambient light.',
    dataTypes: {input: ['terrain', 'raster'], output: ['raster']},
    useCases: ['simulate'],
    domains: ['cartography', 'remote-sensing'],
    params: [
      gridParameter,
      {
        name: 'horizon',
        type: 'GraphDataView<"float32"> | GraphDataView<"uint32">',
        required: true,
        description: 'Horizon map from GPUTerrainHorizon.'
      },
      {
        name: 'directionCount',
        type: 'number',
        required: true,
        description: 'Azimuth sectors in the horizon map.'
      },
      {
        name: 'horizonFormat',
        type: 'GPUTerrainHorizonFormat | undefined',
        default: "'float32'",
        description: "Horizon storage: 'float32' or 'unorm16'."
      },
      parametersParameter,
      {
        name: 'slope',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Adds a surface-facing term (with aspect).'
      },
      {
        name: 'aspect',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Adds a surface-facing term (with slope).'
      },
      {
        name: 'output',
        type: 'GPUSolarShadowMaskOutput',
        required: true,
        description: 'Views for sunVisibility and illumination.'
      }
    ],
    outputs: [
      {
        name: 'sunVisibility',
        type: 'GraphDataView<"float32">',
        description: 'Fraction of the sun disk visible.'
      },
      {
        name: 'illumination',
        type: 'GraphDataView<"float32">',
        description: 'Sun plus ambient light, scaled by intensities.'
      },
      {
        name: 'sunVisibilityTexture',
        type: 'GraphTextureView<"r32float" | "rgba32float">',
        description: 'Optional storage texture receiving sunVisibility.'
      },
      {
        name: 'illuminationTexture',
        type: 'GraphTextureView<"r32float" | "rgba32float">',
        description: 'Optional storage texture receiving illumination.'
      }
    ],
    example: `import {
  GPUSolarShadowMask,
  getGPUSolarShadowMaskParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
// horizon: output of GPUTerrainHorizon with directionCount 16
graph.add(new GPUSolarShadowMask({
  grid, horizon, directionCount: 16,
  parameters: parameters.importToGraph(graph),
  output: {sunVisibility, illumination}
}));
parameters.write(getGPUSolarShadowMaskParameterValues(
  {azimuthDegrees: 135, altitudeDegrees: 30, ambientIntensity: 0.2}
));`,
    preview: {
      kind: 'terrain',
      caption: 'Morning sun: east faces lit, west faces in shade.',
      seed: 421
    },
    related: ['horizon-angles', 'terrain-cast-shadow', 'solar-irradiance', 'solar-position'],
    lensScenario: 'gpu-terrain.solar-shadow-mask',
    reference: REFERENCE,
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- solar
  {
    id: 'solar-position',
    symbol: 'GPUSolarPosition',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Solar position',
    summary: 'Sun azimuth and altitude for every longitude/latitude row at one instant.',
    explanation:
      'Drives shadows, lighting and daylight masks for many places at once. It follows the NOAA solar calculator; the timestamp is a per-frame setting.',
    dataTypes: {input: ['points'], output: ['table']},
    useCases: ['measure', 'simulate'],
    domains: ['remote-sensing', 'urban'],
    params: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Longitude, latitude per row, in degrees.'
      },
      parametersParameter,
      {
        name: 'refraction',
        type: 'boolean | undefined',
        description: 'Apply atmospheric refraction to the altitude.'
      },
      {
        name: 'output',
        type: 'GPUSolarPositionOutput',
        required: true,
        description: 'Views for azimuth, altitude and daylight.'
      }
    ],
    outputs: [
      {
        name: 'azimuth',
        type: 'GraphDataView<"float32">',
        description: 'Degrees clockwise from north.'
      },
      {
        name: 'altitude',
        type: 'GraphDataView<"float32">',
        description: 'Degrees above the horizon.'
      },
      {
        name: 'daylight',
        type: 'GraphDataView<"uint32">',
        description: 'Nonzero when the sun is up.'
      }
    ],
    example: `import {
  GPUSolarPosition,
  getGPUSolarPositionParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
// positions: float32x2 [longitude, latitude] per place
graph.add(new GPUSolarPosition({
  positions, refraction: true,
  parameters: parameters.importToGraph(graph),
  output: {azimuth, altitude, daylight}
}));
parameters.write(getGPUSolarPositionParameterValues(
  {timestamp: new Date('2026-06-21T12:00:00Z')}
));`,
    preview: {kind: 'points', caption: 'Day and night across a field of places.', seed: 422},
    related: ['solar-shadow-mask', 'solar-irradiance', 'terrain-cast-shadow'],
    lensScenario: 'gpu-terrain.solar-position',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'solar-irradiance',
    symbol: 'GPUSolarIrradiance',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Solar irradiance',
    summary: 'Sun-hours and insolation over a day or date range from a horizon map.',
    explanation:
      'Best for solar siting and snow or vegetation modeling. A sun table lists sun positions over time; each cell integrates direct and diffuse light that clears its horizon.',
    dataTypes: {input: ['terrain', 'raster'], output: ['raster']},
    useCases: ['simulate', 'summarize'],
    domains: ['remote-sensing', 'urban'],
    params: [
      gridParameter,
      {
        name: 'horizon',
        type: 'GraphDataView<"float32"> | GraphDataView<"uint32">',
        required: true,
        description: 'Horizon map from GPUTerrainHorizon.'
      },
      {
        name: 'directionCount',
        type: 'number',
        required: true,
        description: 'Azimuth sectors in the horizon map.'
      },
      {
        name: 'sunTable',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Sun samples from getGPUSolarIrradianceSunTable.'
      },
      {
        name: 'sampleCapacity',
        type: 'number',
        required: true,
        description: 'Most sun samples the table can hold.'
      },
      {
        name: 'slope',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Surface tilt; pass with aspect for the incidence angle.'
      },
      parametersParameter,
      {
        name: 'output',
        type: 'GPUSolarIrradianceOutput',
        required: true,
        description: 'Views for sunHours, insolation and validity.'
      }
    ],
    outputs: [
      {
        name: 'sunHours',
        type: 'GraphDataView<"float32">',
        description: 'Hours of direct sun per cell.'
      },
      {
        name: 'insolation',
        type: 'GraphDataView<"float32">',
        description: 'Accumulated energy per cell.'
      },
      validityOutput
    ],
    example: `import {
  GPUSolarIrradiance,
  getGPUSolarIrradianceSunTable,
  getGPUSolarIrradianceParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
const sunTable = getGPUSolarIrradianceSunTable({
  longitude: 7.9, latitude: 46.5,
  start: new Date('2026-06-21T00:00:00Z'), end: new Date('2026-06-22T00:00:00Z'),
  stepMinutes: 15
}); // upload it as sunTableView
graph.add(new GPUSolarIrradiance({
  grid, horizon, directionCount: 16, sunTable: sunTableView, sampleCapacity: 96,
  parameters: parameters.importToGraph(graph),
  output: {sunHours, insolation}
}));
parameters.write(getGPUSolarIrradianceParameterValues({sampleCount: 96}));`,
    preview: {
      kind: 'hotspots',
      caption: 'Sun-hours: south slopes bright, north slopes dim.',
      seed: 423
    },
    related: ['horizon-angles', 'solar-shadow-mask', 'solar-position'],
    lensScenario: 'gpu-terrain.solar-irradiance',
    reference: REFERENCE,
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- relief shading
  {
    id: 'relief-shading',
    symbol: 'GPUReliefShading',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Relief shading',
    summary: 'Multidirectional hillshade and a Swiss/Imhof-style cartographic relief blend.',
    explanation:
      'Best for map-ready terrain. Several lights are weighted by aspect, then mixed with optional sky-view, texture shade, curvature and elevation tints.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['classify', 'transform'],
    domains: ['cartography'],
    params: [
      gridParameter,
      elevationParameter,
      parametersParameter,
      {
        name: 'skyViewFactor',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Darkens enclosed ground; from GPUTerrainHorizon.'
      },
      {
        name: 'textureShade',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Adds fine detail; from GPUTextureShading.'
      },
      {
        name: 'curvature',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Adds ridge and valley emphasis.'
      },
      {
        name: 'imhofSwing',
        type: 'boolean | undefined',
        description: 'Swing the light azimuth with aspect, Imhof style.'
      },
      {
        name: 'output',
        type: 'GPUReliefShadingOutput',
        required: true,
        description: 'Views for hillshade, relief, packed color and validity.'
      }
    ],
    outputs: [
      {
        name: 'hillshade',
        type: 'GraphDataView<"float32">',
        description: 'Multidirectional hillshade in 0..1.'
      },
      {
        name: 'relief',
        type: 'GraphDataView<"float32">',
        description: 'Blended cartographic relief in 0..1.'
      },
      {
        name: 'color',
        type: 'GraphDataView<"uint32">',
        description: 'Packed RGBA with elevation tint.'
      }
    ],
    example: `import {
  GPUReliefShading,
  getGPUReliefShadingParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUReliefShading({
  grid, elevation, skyViewFactor,
  parameters: parameters.importToGraph(graph),
  output: {relief, color}
}));
parameters.write(getGPUReliefShadingParameterValues(
  {lights: 'mdow', skyViewStrength: 0.5, tintStrength: 0.3}, grid
));`,
    preview: {
      kind: 'terrain',
      caption: 'Multidirectional relief with warm and cool tints.',
      seed: 424
    },
    related: ['slope-aspect-hillshade', 'texture-shading', 'horizon-angles', 'relief-blend'],
    lensScenario: 'gpu-terrain.relief-shading',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'texture-shading',
    symbol: 'GPUTextureShading',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Texture shading',
    summary: "Leland Brown's texture shading: crisp ridge and valley detail at every scale.",
    explanation:
      'Reveals fine structure that hillshade flattens. A fractional Laplacian is approximated by a multi-scale Gaussian cascade; `detail` trades small and large features.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['classify', 'transform'],
    domains: ['cartography'],
    params: [
      gridParameter,
      elevationParameter,
      parametersParameter,
      {
        name: 'levelCount',
        type: 'number | undefined',
        default: '6',
        description: 'Band-pass levels, 1 to 8.'
      },
      {
        name: 'baseSigma',
        type: 'number | undefined',
        default: '1',
        description: 'Finest Gaussian sigma, in pixels.'
      },
      {
        name: 'hasNodata',
        type: 'boolean | undefined',
        default: 'true',
        description: 'Set false when the DEM has no no-data to skip masking.'
      },
      {
        name: 'downsampleLevels',
        type: 'boolean | undefined',
        default: 'true',
        description: 'Downsample coarse levels; false is exact but slower.'
      },
      {
        name: 'output',
        type: 'GPUTextureShadingOutput',
        required: true,
        description: 'Views for textureShade and validity.'
      }
    ],
    outputs: [
      {
        name: 'textureShade',
        type: 'GraphDataView<"float32">',
        description: 'Signed texture value, ridges up and valleys down.'
      },
      validityOutput,
      {
        name: 'textureShadeTexture',
        type: 'GraphTextureView<"r32float" | "rgba32float">',
        description: 'Optional storage texture receiving textureShade.'
      }
    ],
    example: `import {
  GPUTextureShading,
  getGPUTextureShadingParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUTextureShading({
  grid, elevation, levelCount: 6,
  parameters: parameters.importToGraph(graph),
  output: {textureShade}
}));
parameters.write(getGPUTextureShadingParameterValues({detail: 0.5, gain: 1}));`,
    preview: {kind: 'terrain', caption: 'Ridge and valley texture at every scale.', seed: 425},
    related: ['relief-shading', 'slope-aspect-hillshade'],
    lensScenario: 'gpu-terrain.texture-shading',
    reference: REFERENCE,
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- relief visualization
  {
    id: 'simple-local-relief',
    symbol: 'GPUSimpleLocalRelief',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Simple local relief (SLRM)',
    summary: 'Elevation minus its local mean: small features pop out of the regional trend.',
    explanation:
      'Finds archaeological banks, ditches and mounds. The mean comes from an exact summed-area table, so any radius costs the same; the radius is read per frame.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['detect-change', 'find-patterns'],
    domains: ['remote-sensing', 'cartography'],
    params: [
      gridParameter,
      elevationParameter,
      parametersParameter,
      {
        name: 'maximumRadius',
        type: 'number',
        required: true,
        description: 'Largest window half-width in cells; fixed at compile time.'
      },
      {
        name: 'quantum',
        type: 'number | undefined',
        description: 'Elevation quantum of the exact summed-area table, a power of two.'
      },
      {
        name: 'output',
        type: 'GPUSimpleLocalReliefOutput',
        required: true,
        description: 'Views for relief and validity.'
      }
    ],
    outputs: [
      {
        name: 'relief',
        type: 'GraphDataView<"float32">',
        description: 'Signed local relief in elevation units.'
      },
      validityOutput
    ],
    example: `import {
  GPUSimpleLocalRelief,
  getGPUSimpleLocalReliefParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUSimpleLocalRelief({
  grid, elevation, maximumRadius: 20,
  parameters: parameters.importToGraph(graph),
  output: {relief}
}));
parameters.write(getGPUSimpleLocalReliefParameterValues({radius: 10}));`,
    preview: {
      kind: 'raster',
      caption: 'Subtle mounds revealed after the trend is removed.',
      seed: 426
    },
    related: ['multi-scale-relief', 'local-dominance', 'relief-blend', 'topographic-position'],
    lensScenario: 'gpu-terrain.simple-local-relief',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'multi-scale-relief',
    symbol: 'GPUMultiScaleRelief',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Multi-scale relief (MSRM)',
    summary: 'Mean difference of successive low-pass surfaces: features between two sizes.',
    explanation:
      'Isolates features within a size band, such as house platforms or field banks. Choose `featureMinimum`, `featureMaximum` and a `scalingFactor`; the filters use exact summed-area tables.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['detect-change', 'find-patterns'],
    domains: ['remote-sensing', 'cartography'],
    params: [
      gridParameter,
      elevationParameter,
      parametersParameter,
      {
        name: 'resolution',
        type: 'number',
        required: true,
        description: 'Cell size used to convert feature sizes to radii.'
      },
      {
        name: 'featureMinimum',
        type: 'number',
        required: true,
        description: 'Smallest feature size, in ground units.'
      },
      {
        name: 'featureMaximum',
        type: 'number',
        required: true,
        description: 'Largest feature size, in ground units.'
      },
      {
        name: 'scalingFactor',
        type: 'number',
        required: true,
        description: 'Step between successive filter radii.'
      },
      {
        name: 'output',
        type: 'GPUMultiScaleReliefOutput',
        required: true,
        description: 'Views for relief and validity.'
      }
    ],
    outputs: [
      {name: 'relief', type: 'GraphDataView<"float32">', description: 'Signed multi-scale relief.'},
      validityOutput
    ],
    example: `import {
  GPUMultiScaleRelief,
  getGPUMultiScaleReliefParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPUMultiScaleRelief({
  grid, elevation,
  resolution: 1, featureMinimum: 2, featureMaximum: 20, scalingFactor: 2,
  parameters: parameters.importToGraph(graph),
  output: {relief}
}));
parameters.write(getGPUMultiScaleReliefParameterValues({verticalExaggeration: 1}));`,
    preview: {
      kind: 'raster',
      caption: 'Only features inside the chosen size band remain.',
      seed: 427
    },
    related: ['simple-local-relief', 'local-dominance', 'topographic-position'],
    lensScenario: 'gpu-terrain.multi-scale-relief',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'local-dominance',
    symbol: 'GPULocalDominance',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Local dominance',
    summary: 'How far an observer on the surface rises above terrain in a ring of distances.',
    explanation:
      'Best for subtle mounds and banks without shadows or a light direction (Hesse 2010). Sample directions and distances set the ring; the observer height is per frame.',
    dataTypes: {input: ['terrain'], output: ['raster']},
    useCases: ['detect-change', 'find-patterns'],
    domains: ['remote-sensing', 'cartography'],
    params: [
      gridParameter,
      elevationParameter,
      parametersParameter,
      {
        name: 'minimumRadius',
        type: 'number | undefined',
        description: 'Inner radius of the ring, in cells.'
      },
      {
        name: 'maximumRadius',
        type: 'number | undefined',
        description: 'Outer radius of the ring, in cells.'
      },
      {
        name: 'radiusIncrement',
        type: 'number | undefined',
        description: 'Spacing of sampled distances.'
      },
      {
        name: 'angularResolution',
        type: 'number | undefined',
        description: 'Angle between sampled directions, in degrees.'
      },
      {
        name: 'output',
        type: 'GPULocalDominanceOutput',
        required: true,
        description: 'Views for dominance and validity.'
      }
    ],
    outputs: [
      {
        name: 'dominance',
        type: 'GraphDataView<"float32">',
        description: 'Mean rise above surrounding terrain; high on mounds.'
      },
      validityOutput
    ],
    example: `import {
  GPULocalDominance,
  getGPULocalDominanceParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
graph.add(new GPULocalDominance({
  grid, elevation, minimumRadius: 5, maximumRadius: 20,
  radiusIncrement: 1, angularResolution: 15,
  parameters: parameters.importToGraph(graph),
  output: {dominance}
}));
parameters.write(getGPULocalDominanceParameterValues({observerHeight: 1.7}));`,
    preview: {kind: 'raster', caption: 'Mounds stand out; flat ground fades.', seed: 428},
    related: ['simple-local-relief', 'multi-scale-relief', 'relief-blend'],
    lensScenario: 'gpu-terrain.local-dominance',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'relief-blend',
    symbol: 'GPUReliefBlend',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Relief blend (VAT)',
    summary: 'Blends up to five single-band visualizations into one greyscale image.',
    explanation:
      'Combines hillshade, sky-view, openness and local relief into the Visualization for Archaeological Topography. Each layer is stretched, optionally inverted, and blended with a mode and opacity.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['classify', 'transform'],
    domains: ['remote-sensing', 'cartography'],
    params: [
      gridParameter,
      {
        name: 'layers',
        type: 'readonly GraphDataView<"float32">[]',
        required: true,
        description: 'Up to five layers, bottom to top.'
      },
      parametersParameter,
      {
        name: 'output',
        type: 'GPUReliefBlendOutput',
        required: true,
        description: 'Views for blend, packed color and validity.'
      }
    ],
    outputs: [
      {name: 'blend', type: 'GraphDataView<"float32">', description: 'Blended greyscale in 0..1.'},
      {name: 'color', type: 'GraphDataView<"uint32">', description: 'Packed RGBA of the blend.'},
      validityOutput
    ],
    example: `import {
  GPUReliefBlend,
  GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL,
  getGPUReliefBlendParameterValues
} from '${IMPORT_PATH}';

${SETUP_NOTE}
// layers: hillshade, sky-view, openness, local relief views
graph.add(new GPUReliefBlend({
  grid, layers,
  parameters: parameters.importToGraph(graph),
  output: {blend}
}));
parameters.write(getGPUReliefBlendParameterValues(
  GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL
));`,
    preview: {kind: 'raster', caption: 'Five layers blended into one readable image.', seed: 429},
    related: ['relief-shading', 'simple-local-relief', 'local-dominance', 'horizon-angles'],
    lensScenario: 'gpu-terrain.relief-blend',
    reference: REFERENCE,
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- decode and repair
  {
    id: 'terrain-rgb-decode',
    symbol: 'GPUTerrainRGBDecode',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Terrain-RGB decode',
    summary: 'Decodes Terrarium or Mapbox RGB elevation tiles into float32 heights and validity.',
    explanation:
      'Turns web terrain tiles into a DEM on the GPU. Pick the `encoding`, then mark no-data by alpha, color or a valid range.',
    dataTypes: {input: ['raster'], output: ['terrain']},
    useCases: ['transform'],
    domains: ['data-engineering', 'hydrology'],
    params: [
      gridParameter,
      {
        name: 'encoding',
        type: 'GPUTerrainRGBEncoding',
        required: true,
        description: "Tile encoding: 'terrarium' or 'mapbox'."
      },
      {
        name: 'input',
        type: 'GPUTerrainRGBDecodeInput',
        required: true,
        description: 'RGB(A) tile as a texture or packed buffer.'
      },
      {
        name: 'alphaNoData',
        type: 'boolean | undefined',
        description: 'Treat zero alpha as no-data.'
      },
      {
        name: 'noDataRGB',
        type: 'readonly [number, number, number] | undefined',
        description: 'RGB triple that means no-data.'
      },
      {
        name: 'validRange',
        type: 'readonly [number, number] | undefined',
        default: '[-11000, 9000]',
        description: 'Heights outside this range are invalid.'
      },
      {
        name: 'clampBathymetry',
        type: 'boolean | undefined',
        description: 'Clamp ocean depths to a floor value.'
      },
      {
        name: 'output',
        type: 'GPUTerrainRGBDecodeOutput',
        required: true,
        description: 'Views for values and outputValidity.'
      }
    ],
    outputs: [
      {name: 'values', type: 'GraphDataView<"float32">', description: 'Elevation in meters.'},
      {
        name: 'outputValidity',
        type: 'GraphDataView<"uint32">',
        description: 'Nonzero for valid pixels.'
      }
    ],
    example: `import {GPUTerrainRGBDecode} from '${IMPORT_PATH}';

${SETUP_NOTE}
// tile: the RGBA tile, uploaded as a texture or buffer view
graph.add(new GPUTerrainRGBDecode({
  grid, encoding: 'terrarium', input: tile,
  alphaNoData: true,
  output: {values: elevationValues, outputValidity}
}));`,
    preview: {kind: 'raster', caption: 'RGB-encoded tile decoded into heights.', seed: 430},
    related: ['spike-repair', 'slope-aspect-hillshade'],
    lensScenario: 'gpu-terrain.terrain-rgb-decode',
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'spike-repair',
    symbol: 'GPUTerrainSpikeRepair',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spike repair',
    summary: 'Repairs +/-1 high-byte errors in Terrarium-style elevation tiles.',
    explanation:
      'Best when decoded web tiles show 256 m steps or cliffs. It groups pixels into components and shifts those whose borders agree on a multiple of the step; this is opt-in and bounded by an iteration budget.',
    dataTypes: {input: ['terrain'], output: ['terrain']},
    useCases: ['transform'],
    domains: ['data-engineering', 'hydrology'],
    params: [
      gridParameter,
      elevationParameter,
      {
        name: 'step',
        type: 'number | undefined',
        default: '256',
        description: 'Height of one encoding step of the faulty byte.'
      },
      {
        name: 'jump',
        type: 'number | undefined',
        default: '200',
        description: 'Neighbor jumps up to this keep pixels in one component.'
      },
      {
        name: 'tolerance',
        type: 'number | undefined',
        default: '40',
        description: 'Allowed deviation of a seam jump from a multiple of step.'
      },
      {
        name: 'agreement',
        type: 'number | undefined',
        default: '0.8',
        description: 'Share of a component border that must agree to shift it.'
      },
      {
        name: 'maximumComponentIterations',
        type: 'number | undefined',
        default: '32',
        description: 'Iteration budget for labeling components.'
      },
      {
        name: 'output',
        type: 'GPUTerrainSpikeRepairOutput',
        required: true,
        description: 'Views for values, outputValidity, labels and statistics.'
      }
    ],
    outputs: [
      {name: 'values', type: 'GraphDataView<"float32">', description: 'Repaired elevation.'},
      {
        name: 'outputValidity',
        type: 'GraphDataView<"uint32">',
        description: 'Nonzero for valid pixels.'
      },
      {
        name: 'statistics',
        type: 'GraphDataView<"uint32">',
        description: 'Jump, repaired, shifted and remaining counts, converged flag.'
      }
    ],
    example: `import {GPUTerrainSpikeRepair} from '${IMPORT_PATH}';

${SETUP_NOTE}
// elevation: the output of GPUTerrainRGBDecode
graph.add(new GPUTerrainSpikeRepair({
  grid, elevation,
  maximumComponentIterations: 32,
  output: {values: repaired, outputValidity, statistics}
}));`,
    preview: {
      kind: 'raster',
      caption: 'Stair-step defects removed from a decoded tile.',
      seed: 431
    },
    related: ['terrain-rgb-decode', 'slope-aspect-hillshade'],
    lensScenario: 'gpu-terrain.terrain-spike-repair',
    reference: REFERENCE,
    maturity: 'experimental'
  }
];
