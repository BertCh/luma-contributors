// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group "cells-index-join": discrete global grid cells (H3, Quadbin, S2), spherical
// geometry, Hilbert keys and spatial joins.

import type {CatalogEntry} from '../schema';

const IMPORT_PATH = '@luma-contributors/analysis/gpu-spatial-analysis';

export const entries: CatalogEntry[] = [
  // ---------------------------------------------------------------------------------------------
  // Keys and geometry of cells
  // ---------------------------------------------------------------------------------------------
  {
    id: 'point-to-cell',
    symbol: 'GPUPointToCell',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Point to cell',
    summary: 'Converts longitude/latitude points to 64-bit grid cell keys, one key per row.',
    explanation:
      'The first step of any grid workflow: H3, Quadbin, quadkey, geohash and S2 keys come out of one kernel. Keys are `uint32x2` (low, high) words, the Arrow `Uint64` layout. Rewrite positions or the mask and the same graph re-encodes without recompiling.',
    dataTypes: {input: ['points'], output: ['cells']},
    useCases: ['index'],
    domains: ['data-engineering', 'urban'],
    params: [
      {
        name: 'family',
        type: 'GPUCellIndexFamily',
        required: true,
        description: "'quadbin', 'h3', 'quadkey', 'geohash' or 's2'."
      },
      {
        name: 'resolution',
        type: 'number',
        required: true,
        description: 'Resolution or level; fixed at compile time.'
      },
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Longitude/latitude in degrees.'
      },
      {
        name: 'mask',
        type: 'GraphDataView<"uint32">',
        description: 'Per-row 0/1 flag; masked rows get the zero key.'
      },
      {
        name: 'precision',
        type: '"float32" | "extended"',
        default: "'float32'",
        description: 'H3 only: extended arithmetic for finer resolutions.'
      },
      {
        name: 'output',
        type: 'GPUPointToCellOutput',
        required: true,
        description: 'Destination views for cells and optional validity.'
      }
    ],
    outputs: [
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        description: 'One 64-bit key per row, zero when invalid.'
      },
      {name: 'validity', type: 'GraphDataView<"uint32">', description: 'Optional 0/1 flag per row.'}
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUPointToCell} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'keys'});
const positions = graph.importGPUData('positions', lonLatData); // GPUData<'float32x2'>
const cells = graph.importGPUData('cells', cellData); // GPUData<'uint32x2'>, caller-owned

graph.add(new GPUPointToCell({family: 'h3', resolution: 9, positions, output: {cells}}));

const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,
    preview: {
      kind: 'cells',
      seed: 9001,
      caption: 'Each point falls into exactly one hexagon at the chosen resolution.'
    },
    related: ['cell-geometry', 'cell-aggregation', 'cell-topology'],
    lensScenario: 'gpu-spatial-analysis.point-to-cell',
    reference: 'api/gpu-spatial-analysis/operators-cells',
    maturity: 'stable'
  },
  {
    id: 'cell-geometry',
    symbol: 'GPUCellGeometry',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Cell geometry',
    summary: 'Decodes cell keys to centers and boundary polygons in longitude/latitude degrees.',
    explanation:
      'Draws any grid column, or to turn cells into polygons for joins. H3 boundaries match h3-js `cellToBoundary` exactly, including pentagons and face-crossing vertices. Boundaries use a fixed vertex stride with a per-row vertex count.',
    dataTypes: {input: ['cells'], output: ['points', 'polygons']},
    useCases: ['transform', 'index'],
    domains: ['cartography', 'data-engineering'],
    params: [
      {
        name: 'family',
        type: 'GPUCellGeometryFamily',
        required: true,
        description: 'Any index family, or A5.'
      },
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'Cell keys to decode.'
      },
      {
        name: 'wordOrder',
        type: 'GPUCellWordOrder',
        default: "'little-endian'",
        description: "Key word layout; 'high-low' also accepted."
      },
      {
        name: 'maximumVertexCount',
        type: 'number',
        description: 'Boundary stride; H3 needs up to 10.'
      },
      {
        name: 'output',
        type: 'GPUCellGeometryOutput',
        required: true,
        description: 'Any of centers, boundaries and vertexCounts.'
      }
    ],
    outputs: [
      {
        name: 'centers',
        type: 'GraphDataView<"float32x2">',
        description: 'Cell centers, one per row.'
      },
      {
        name: 'boundaries',
        type: 'GraphDataView<"float32x2">',
        description: 'Boundary vertices at a fixed stride per row.'
      },
      {
        name: 'vertexCounts',
        type: 'GraphDataView<"uint32">',
        description: 'Valid vertices per row.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCellGeometry, GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'cell-geometry'});
const cells = graph.importGPUData('cells', cellData); // GPUData<'uint32x2'>
const centers = graph.importGPUData('centers', centerData); // GPUData<'float32x2'>
const boundaries = graph.importGPUData('boundaries', boundaryData); // rows * 10 vertices
const vertexCounts = graph.importGPUData('vertex-counts', countData); // GPUData<'uint32'>

graph.add(new GPUCellGeometry({
  family: 'h3', cells,
  maximumVertexCount: GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT,
  output: {centers, boundaries, vertexCounts}
}));
const compiled = await graph.compileAsync();`,
    preview: {kind: 'cells', seed: 9002, caption: 'Keys become hexagon outlines and centers.'},
    related: ['point-to-cell', 'cell-set-outline', 'cell-measures'],
    lensScenario: 'gpu-spatial-analysis.cell-geometry',
    reference: 'api/gpu-spatial-analysis/operators-cells',
    maturity: 'stable'
  },
  {
    id: 'cell-measures',
    symbol: 'GPUCellMeasures',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'H3 cell measures',
    summary:
      'Computes exact spherical area, perimeter, edge length and pentagon flag for H3 cells.',
    explanation:
      'Normalizes counts by true cell area, since H3 cells are not all the same size. Areas come from the spherical boundary, not a table of averages. Pentagons are flagged and have five edges.',
    dataTypes: {input: ['cells'], output: ['table']},
    useCases: ['measure'],
    domains: ['cartography', 'demographics'],
    params: [
      {name: 'family', type: '"h3"', required: true, description: 'Only H3 is supported.'},
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'H3 cell keys.'
      },
      {
        name: 'units',
        type: '"meters" | "kilometers"',
        default: "'meters'",
        description: 'Unit of lengths and areas.'
      },
      {name: 'maximumEdgeCount', type: 'number', default: '6', description: 'Edge slots per cell.'},
      {
        name: 'output',
        type: 'GPUCellMeasuresOutput',
        required: true,
        description: 'Any of the measure columns below.'
      }
    ],
    outputs: [
      {name: 'areas', type: 'GraphDataView<"float32">', description: 'Cell area, square units.'},
      {name: 'perimeters', type: 'GraphDataView<"float32">', description: 'Boundary length.'},
      {
        name: 'edgeLengths',
        type: 'GraphDataView<"float32">',
        description: 'Edge lengths per cell.'
      },
      {name: 'edgeCounts', type: 'GraphDataView<"uint32">', description: 'Edges per cell: 5 or 6.'},
      {name: 'pentagons', type: 'GraphDataView<"uint32">', description: '1 for pentagon cells.'}
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCellMeasures} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'measures'});
const cells = graph.importGPUData('cells', cellData); // GPUData<'uint32x2'>
const areas = graph.importGPUData('areas', areaData); // GPUData<'float32'>
const pentagons = graph.importGPUData('pentagons', flagData); // GPUData<'uint32'>

graph.add(new GPUCellMeasures({
  family: 'h3', cells, units: 'kilometers', output: {areas, pentagons}
}));
const compiled = await graph.compileAsync();`,
    preview: {kind: 'cells', seed: 9003, caption: 'Hexagons differ in area; pentagons stand out.'},
    related: ['s2-cell-measures', 'cell-geometry'],
    lensScenario: 'gpu-spatial-analysis.cell-measures',
    reference: 'api/gpu-spatial-analysis/operators-cells',
    maturity: 'stable'
  },
  {
    id: 'h3-cell-identities',
    symbol: 'GPUH3CellIdentities',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'H3 edge and vertex ids',
    summary:
      'Derives exact H3 directed-edge or canonical vertex identities for cells, including pentagons.',
    explanation:
      'Builds flow graphs between cells or to share vertices across neighboring cells. The identities are exact integers, so they join and deduplicate without tolerance. Choose `directed-edges` or `vertices` at compile time.',
    dataTypes: {input: ['cells'], output: ['cells', 'network']},
    useCases: ['index'],
    domains: ['data-engineering', 'transport'],
    params: [
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'H3 cell keys.'
      },
      {
        name: 'operation',
        type: '"directed-edges" | "vertices"',
        required: true,
        description: 'Which identity set to derive per cell.'
      },
      {
        name: 'output',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'Identity keys, a fixed fan-out per input cell.'
      }
    ],
    outputs: [
      {
        name: 'output',
        type: 'GraphDataView<"uint32x2">',
        description: 'Edge or vertex keys, laid out per input cell.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUH3CellIdentities} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'h3-edges'});
const cells = graph.importGPUData('cells', cellData); // GPUData<'uint32x2'>
const edges = graph.importGPUData('edges', edgeData); // GPUData<'uint32x2'>, 6 per cell

graph.add(new GPUH3CellIdentities({cells, operation: 'directed-edges', output: edges}));
const compiled = await graph.compileAsync();`,
    preview: {kind: 'cells', seed: 9004, caption: 'Six directed edges leave every hexagon.'},
    related: ['cell-topology', 'cell-grid-path'],
    lensScenario: 'gpu-spatial-analysis.h3-cell-identities',
    reference: 'api/gpu-spatial-analysis/expansion-lenses',
    maturity: 'stable'
  },

  // ---------------------------------------------------------------------------------------------
  // Aggregating and comparing cells
  // ---------------------------------------------------------------------------------------------
  {
    id: 'cell-aggregation',
    symbol: 'GPUCellAggregation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Cell aggregation',
    summary: 'Bins rows into grid cells and reduces them to a compact, sorted per-cell table.',
    explanation:
      'Turns millions of points into a heat grid. Rows are keyed, sorted and reduced with integer atomics, so counts and sums are exact and repeatable. Output capacity is fixed; on overflow the smallest keys are kept and a flag is set.',
    dataTypes: {input: ['points', 'cells'], output: ['cells', 'table']},
    useCases: ['summarize', 'index'],
    domains: ['urban', 'data-engineering'],
    params: [
      {name: 'family', type: 'GPUCellFamily', required: true, description: "'quadbin' or 'h3'."},
      {
        name: 'resolution',
        type: 'number',
        required: true,
        description: 'Output resolution; finer keys are truncated.'
      },
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        description: 'Longitude/latitude rows (Quadbin).'
      },
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        description: 'Pre-keyed rows instead of positions.'
      },
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        description: 'Optional values to sum, min and max per cell.'
      },
      {name: 'mask', type: 'GraphDataView<"uint32">', description: 'Per-frame 0/1 row filter.'},
      {
        name: 'sumScale',
        type: 'number',
        default: '65536',
        description: 'Fixed-point scale of the exact sums.'
      },
      {
        name: 'output',
        type: 'GPUCellTable',
        required: true,
        description: 'Table views; capacity is cells.length.'
      }
    ],
    outputs: [
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        description: 'Occupied cell keys, ascending.'
      },
      {name: 'counts', type: 'GraphDataView<"uint32">', description: 'Rows per cell.'},
      {name: 'sumValues', type: 'GraphDataView<"float32">', description: 'Sum of values per cell.'},
      {name: 'minimums', type: 'GraphDataView<"float32">', description: 'Smallest value per cell.'},
      {name: 'maximums', type: 'GraphDataView<"float32">', description: 'Largest value per cell.'},
      {
        name: 'status',
        type: 'GPUCardinalityPort & GPUOverflowStatusPort',
        description: 'Occupied count and overflow flag.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCellAggregation} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'heat-grid'});
const positions = graph.importGPUData('positions', lonLatData); // GPUData<'float32x2'>
const values = graph.importGPUData('values', valueData); // GPUData<'float32'>
const table = importCellTable(graph, 4096); // your helper: cells, counts, sumValues, status views

graph.add(new GPUCellAggregation({
  family: 'quadbin', resolution: 14, positions, values, output: table
}));

const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,
    preview: {kind: 'cells', seed: 9005, caption: 'Points collapse into counted cells.'},
    related: ['cell-rollup', 'cell-pyramid', 'point-to-cell', 'cell-table-compare'],
    lensScenario: 'gpu-spatial-analysis.cell-aggregation',
    reference: 'api/gpu-spatial-analysis/operators-cells',
    maturity: 'stable'
  },
  {
    id: 'cell-rollup',
    symbol: 'GPUCellRollup',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Cell rollup',
    summary:
      'Rolls a sorted cell table up to a coarser resolution without re-reading the source rows.',
    explanation:
      'Zooms out a grid you already aggregated. Parent keys truncate the cell path, which keeps key order, so no sort is needed. Counts and sums are integer sums and match direct aggregation bit for bit.',
    dataTypes: {input: ['cells', 'table'], output: ['cells', 'table']},
    useCases: ['summarize'],
    domains: ['urban', 'data-engineering'],
    params: [
      {name: 'family', type: 'GPUCellFamily', required: true, description: "'quadbin' or 'h3'."},
      {
        name: 'sourceResolution',
        type: 'number',
        required: true,
        description: 'Resolution of the source table.'
      },
      {
        name: 'resolution',
        type: 'number',
        required: true,
        description: 'Coarser target resolution.'
      },
      {
        name: 'source',
        type: 'GPUCellTable',
        required: true,
        description: 'Sorted table from aggregation or another rollup.'
      },
      {name: 'sumScale', type: 'number', description: 'Fixed-point scale; must match the source.'},
      {name: 'output', type: 'GPUCellTable', required: true, description: 'Parent table views.'}
    ],
    outputs: [
      {name: 'cells', type: 'GraphDataView<"uint32x2">', description: 'Parent keys, ascending.'},
      {name: 'counts', type: 'GraphDataView<"uint32">', description: 'Summed child counts.'},
      {
        name: 'status',
        type: 'GPUCardinalityPort & GPUOverflowStatusPort',
        description: 'Parent count; source overflow carries through.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCellAggregation, GPUCellRollup} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'rollup'});
const positions = graph.importGPUData('positions', lonLatData);
const fine = importCellTable(graph, 4096); // your helper: table views on this graph
const coarse = importCellTable(graph, 1024);

graph.add(new GPUCellAggregation({family: 'quadbin', resolution: 16, positions, output: fine}));
graph.add(new GPUCellRollup({
  family: 'quadbin', sourceResolution: 16, resolution: 12, source: fine, output: coarse
}));
const compiled = await graph.compileAsync();`,
    preview: {kind: 'cells', seed: 9006, caption: 'Four children sum into one parent.'},
    related: ['cell-aggregation', 'cell-pyramid', 'cell-level-selection'],
    lensScenario: 'gpu-spatial-analysis.cell-rollup',
    reference: 'api/gpu-spatial-analysis/operators-cells',
    maturity: 'stable'
  },
  {
    id: 'cell-pyramid',
    symbol: 'GPUCellPyramid',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Cell pyramid',
    summary:
      'Aggregates rows once at the finest level, then rolls each coarser level out of the one below.',
    explanation:
      'Best for zoomable heat maps: every zoom level is ready before the user pans. List `resolutions` finest first. Pair it with level selection to switch levels without recomputing.',
    dataTypes: {input: ['points', 'cells'], output: ['cells', 'table']},
    useCases: ['summarize', 'index'],
    domains: ['urban', 'cartography'],
    params: [
      {name: 'family', type: 'GPUCellFamily', required: true, description: "'quadbin' or 'h3'."},
      {
        name: 'resolutions',
        type: 'readonly number[]',
        required: true,
        description: 'Levels to build, finest first.'
      },
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        description: 'Longitude/latitude rows (Quadbin).'
      },
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        description: 'Pre-keyed rows instead of positions.'
      },
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        description: 'Optional values summed per cell.'
      },
      {name: 'mask', type: 'GraphDataView<"uint32">', description: 'Per-frame 0/1 row filter.'},
      {
        name: 'output',
        type: 'GPUCellPyramidOutput',
        required: true,
        description: 'One table per level, plus optional level counts.'
      }
    ],
    outputs: [
      {
        name: 'levels',
        type: 'readonly GPUCellTable[]',
        description: 'One cell table per requested resolution.'
      },
      {
        name: 'levelCounts',
        type: 'GraphDataView<"uint32">',
        description: 'Occupied cells per level.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCellPyramid} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'pyramid'});
const positions = graph.importGPUData('positions', lonLatData);
const levels = [18, 14, 10, 6].map(() => importCellTable(graph, 8192)); // your helper
const levelCounts = graph.importGPUData('level-counts', countData); // GPUData<'uint32'>

graph.add(new GPUCellPyramid({
  family: 'quadbin', positions, resolutions: [18, 14, 10, 6],
  output: {levels, levelCounts}
}));
const compiled = await graph.compileAsync();`,
    preview: {kind: 'cells', seed: 9007, caption: 'The same points at four zoom levels.'},
    related: ['cell-rollup', 'cell-level-selection', 'cell-aggregation'],
    lensScenario: 'gpu-spatial-analysis.cell-pyramid',
    reference: 'api/gpu-spatial-analysis/operators-cells',
    maturity: 'stable'
  },
  {
    id: 'cell-level-selection',
    symbol: 'GPUCellLevelSelection',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Cell level selection',
    summary:
      'Publishes the active pyramid level count, first row and draw arguments from one per-frame index.',
    explanation:
      'Switches zoom levels in a layer with no readback and no recompute. It reads one `activeLevel` word and writes the level count and indirect draw arguments. Keep it in its own small graph beside the pyramid.',
    dataTypes: {input: ['cells', 'table'], output: ['table']},
    useCases: ['summarize', 'classify'],
    domains: ['cartography', 'data-engineering'],
    params: [
      {
        name: 'levelCounts',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Occupied cells per level, from the pyramid.'
      },
      {
        name: 'activeLevel',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Per-frame level index; clamped to the last level.'
      },
      {
        name: 'levelFirstRows',
        type: 'readonly number[]',
        description: 'First row of each level in a packed table.'
      },
      {
        name: 'output',
        type: 'GPUCellLevelSelectionOutput',
        required: true,
        description: 'Status and optional draw arguments.'
      }
    ],
    outputs: [
      {name: 'status', type: 'GPUCardinalityPort', description: 'Cell count of the active level.'},
      {
        name: 'firstRow',
        type: 'GraphDataView<"uint32">',
        description: 'First row of the active level.'
      },
      {
        name: 'drawArguments',
        type: 'GraphDataView<"uint32">',
        description: 'Indirect draw words: instance count and first instance.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCellLevelSelection, GPUParameterBuffer} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'level-select'});
const activeLevel = new GPUParameterBuffer(device, {id: 'level', format: 'uint32', length: 1});
const levelCounts = graph.importGPUData('level-counts', countData); // from the pyramid
const count = graph.importGPUData('count', countWord); // GPUData<'uint32'>
const drawArguments = graph.importGPUData('draw', drawData); // GPUData<'uint32'>

graph.add(new GPUCellLevelSelection({
  levelCounts, activeLevel: activeLevel.importToGraph(graph),
  output: {status: {count}, drawArguments}
}));
activeLevel.write(Uint32Array.of(2)); // switch zoom: no recompile`,
    preview: {kind: 'cells', seed: 9008, caption: 'One index picks which level is drawn.'},
    related: ['cell-pyramid', 'cell-rollup'],
    lensScenario: 'gpu-spatial-analysis.cell-level-selection',
    reference: 'api/gpu-spatial-analysis/operators-cells',
    maturity: 'stable'
  },
  {
    id: 'cell-table-compare',
    symbol: 'GPUCellTableCompare',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Cell table compare',
    summary: 'Outer-joins two sorted cell tables into a before/after diff map.',
    explanation:
      'Best for period-over-period or A-versus-B maps on the same grid. Cells present in only one table are kept and marked in `presence`. Choose `count` or `sum` as the measure, and request delta, ratio, percent change or a z-score.',
    dataTypes: {input: ['cells', 'table'], output: ['cells', 'table']},
    useCases: ['detect-change', 'summarize'],
    domains: ['urban', 'spatial-statistics'],
    params: [
      {
        name: 'before',
        type: 'GPUCellTable',
        required: true,
        description: 'Earlier or baseline table.'
      },
      {
        name: 'after',
        type: 'GPUCellTable',
        required: true,
        description: 'Later or comparison table.'
      },
      {
        name: 'measure',
        type: 'GPUCellTableCompareMeasure',
        default: "'count'",
        description: "'count' or 'sum'."
      },
      {name: 'sumScale', type: 'number', description: 'Fixed-point scale of both tables.'},
      {
        name: 'zScore',
        type: 'GPUCellTableCompareZScore',
        description: 'Options for a per-cell z-score column.'
      },
      {
        name: 'output',
        type: 'GPUCellTableCompareOutput',
        required: true,
        description: 'Any of the diff columns below.'
      }
    ],
    outputs: [
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        description: 'Union of occupied cells, ascending.'
      },
      {
        name: 'presence',
        type: 'GraphDataView<"uint32">',
        description: 'Which side holds the cell.'
      },
      {name: 'delta', type: 'GraphDataView<"float32">', description: 'after minus before.'},
      {name: 'ratio', type: 'GraphDataView<"float32">', description: 'after divided by before.'},
      {
        name: 'percentChange',
        type: 'GraphDataView<"float32">',
        description: 'Relative change in percent.'
      },
      {name: 'zScore', type: 'GraphDataView<"float32">', description: 'Standardized change.'},
      {
        name: 'status',
        type: '{count, overflow, requiredCount?}',
        description: 'Union cell count and overflow flag.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCellTableCompare} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'change-map'});
const before = importCellTable(graph, 4096); // your helper: sorted tables from aggregation
const after = importCellTable(graph, 4096);
const cells = graph.importGPUData('cells', unionKeyData); // GPUData<'uint32x2'>
const delta = graph.importGPUData('delta', deltaData); // GPUData<'float32'>
const status = importCardinalityStatus(graph); // your helper: count and overflow words

graph.add(new GPUCellTableCompare({
  before, after, measure: 'count', output: {cells, delta, status}
}));
const compiled = await graph.compileAsync();`,
    preview: {kind: 'cells', seed: 9009, caption: 'Red where counts grew, blue where they fell.'},
    related: ['cell-aggregation', 'cell-rollup'],
    lensScenario: 'gpu-spatial-analysis.cell-table-compare',
    reference: 'api/gpu-spatial-analysis/operators-cells',
    maturity: 'stable'
  },

  // ---------------------------------------------------------------------------------------------
  // Topology of cells
  // ---------------------------------------------------------------------------------------------
  {
    id: 'cell-topology',
    symbol: 'GPUCellTopology',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Cell topology',
    summary:
      'Finds disks, rings, parents and children of Quadbin and H3 cells with a fixed fan-out per cell.',
    explanation:
      'Best for neighborhoods on a grid: `gridDisk`, `gridRing`, `cellToParent` and `cellToChildren`. Each input cell owns a fixed block of output slots, so results stay aligned with the input rows. An optional mask skips rows.',
    dataTypes: {input: ['cells'], output: ['cells', 'spatial-weights']},
    useCases: ['relate', 'index'],
    domains: ['urban', 'spatial-statistics'],
    params: [
      {name: 'family', type: 'GPUCellFamily', required: true, description: "'quadbin' or 'h3'."},
      {
        name: 'operation',
        type: 'GPUCellTopologyOperation',
        required: true,
        description: "{type: 'disk' | 'ring', k} or parent and children."
      },
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'Input cell keys.'
      },
      {
        name: 'wordOrder',
        type: 'GPUCellWordOrder',
        default: "'little-endian'",
        description: "Key word layout; 'high-low' also accepted."
      },
      {name: 'mask', type: 'GraphDataView<"uint32">', description: 'Per-row 0/1 filter.'},
      {
        name: 'output',
        type: 'GPUCellTopologyOutput',
        required: true,
        description: 'Cells, plus optional distances and counts.'
      }
    ],
    outputs: [
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        description: 'Result keys, a fixed block per input row.'
      },
      {
        name: 'distances',
        type: 'GraphDataView<"uint32">',
        description: 'Grid distance of each result slot.'
      },
      {name: 'counts', type: 'GraphDataView<"uint32">', description: 'Valid slots per input row.'}
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCellTopology} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'neighbors'});
const cells = graph.importGPUData('cells', cellData); // GPUData<'uint32x2'>
const disk = graph.importGPUData('disk', diskData); // 19 slots per cell for k = 2
const counts = graph.importGPUData('counts', countData); // GPUData<'uint32'>

graph.add(new GPUCellTopology({
  family: 'h3', operation: {type: 'disk', k: 2}, cells, output: {cells: disk, counts}
}));
const compiled = await graph.compileAsync();`,
    preview: {kind: 'cells', seed: 9010, caption: 'A disk of radius two around the chosen cell.'},
    related: ['cell-compaction', 'cell-grid-path', 'cell-cover', 's2-cell-hierarchy'],
    lensScenario: 'gpu-spatial-analysis.cell-topology',
    reference: 'api/gpu-spatial-analysis/operators-cells',
    maturity: 'stable'
  },
  {
    id: 'cell-compaction',
    symbol: 'GPUCellCompaction',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Cell compaction',
    summary:
      'Compacts a cell set to its minimal mixed-resolution cover, or expands it to one resolution.',
    explanation:
      'Use `compact` to store a region with the fewest cells, and `uncompact` to bring it back to one resolution for joins. Complete sibling groups merge into their parent. Works on Quadbin and H3.',
    dataTypes: {input: ['cells'], output: ['cells']},
    useCases: ['transform', 'index'],
    domains: ['data-engineering', 'cartography'],
    params: [
      {name: 'family', type: 'GPUCellFamily', required: true, description: "'quadbin' or 'h3'."},
      {
        name: 'operation',
        type: 'GPUCellCompactOperation | GPUCellUncompactOperation',
        required: true,
        description: "{type: 'compact' | 'uncompact', resolution}."
      },
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'Input cell keys.'
      },
      {name: 'mask', type: 'GraphDataView<"uint32">', description: 'Per-row 0/1 filter.'},
      {
        name: 'count',
        type: 'GraphDataView<"uint32">',
        description: 'GPU-side row count when the input is partly filled.'
      },
      {
        name: 'output',
        type: 'GPUCellCompactionOutput',
        required: true,
        description: 'Cells plus count and overflow status.'
      }
    ],
    outputs: [
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        description: 'Compacted or expanded keys.'
      },
      {
        name: 'status',
        type: '{count, requiredCount?, overflow, droppedCount?}',
        description: 'Written count and overflow flag.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCellCompaction} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'compact'});
const cells = graph.importGPUData('cells', cellData); // GPUData<'uint32x2'>, all res 9
const compact = graph.importGPUData('compact', outputData); // GPUData<'uint32x2'>
const count = graph.importGPUData('count', countWord); // GPUData<'uint32'>
const overflow = graph.importGPUData('overflow', overflowWord); // GPUData<'uint32'>

graph.add(new GPUCellCompaction({
  family: 'h3', operation: {type: 'compact', resolution: 9}, cells,
  output: {cells: compact, status: {count, overflow}}
}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'tessellation',
      seed: 9011,
      caption: 'Complete sibling groups merge into larger cells.'
    },
    related: ['cell-topology', 'cell-cover', 's2-cell-hierarchy'],
    lensScenario: 'gpu-spatial-analysis.cell-compaction',
    reference: 'api/gpu-spatial-analysis/operators-cells',
    maturity: 'stable'
  },
  {
    id: 'cell-grid-path',
    symbol: 'GPUCellGridPath',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'H3 grid path',
    summary:
      'Computes H3 grid distance and the cell path between arbitrary origin and destination pairs.',
    explanation:
      'Measures steps between cells or to trace a route of cells between two places. Neighbor tests are just `distance == 1`. Paths are written at a fixed stride, with a per-pair length.',
    dataTypes: {input: ['cells'], output: ['cells', 'table']},
    useCases: ['measure', 'route'],
    domains: ['transport', 'urban'],
    params: [
      {name: 'family', type: '"h3"', required: true, description: 'Only H3 is supported.'},
      {
        name: 'origins',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'Start cell of each pair.'
      },
      {
        name: 'destinations',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'End cell of each pair.'
      },
      {name: 'maximumPathLength', type: 'number', description: 'Path slots per pair.'},
      {name: 'mask', type: 'GraphDataView<"uint32">', description: 'Per-pair 0/1 filter.'},
      {
        name: 'output',
        type: 'GPUCellGridPathOutput',
        required: true,
        description: 'Any of distances, cells and counts.'
      }
    ],
    outputs: [
      {name: 'distances', type: 'GraphDataView<"uint32">', description: 'Grid steps per pair.'},
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        description: 'Path cells at a fixed stride.'
      },
      {name: 'counts', type: 'GraphDataView<"uint32">', description: 'Valid path cells per pair.'}
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCellGridPath} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'grid-path'});
const origins = graph.importGPUData('origins', originData); // GPUData<'uint32x2'>
const destinations = graph.importGPUData('destinations', destinationData);
const distances = graph.importGPUData('distances', distanceData); // GPUData<'uint32'>

graph.add(new GPUCellGridPath({family: 'h3', origins, destinations, output: {distances}}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'shortest-path',
      seed: 9012,
      caption: 'A chain of cells joins origin and destination.'
    },
    related: ['cell-topology', 'h3-cell-identities'],
    lensScenario: 'gpu-spatial-analysis.cell-grid-path',
    reference: 'api/gpu-spatial-analysis/operators-cells',
    maturity: 'stable'
  },
  {
    id: 'cell-cover',
    symbol: 'GPUCellCover',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Cell cover',
    summary: 'Polyfills polygon features with Quadbin or H3 cells.',
    explanation:
      'Turns zones into grid cells for joins and rollups. `center` keeps cells whose center is inside; Quadbin also supports `full` and `intersects`. Results are bounded by `candidateCapacity`, with overflow flagged on the GPU.',
    dataTypes: {input: ['polygons'], output: ['cells']},
    useCases: ['transform', 'relate'],
    domains: ['cartography', 'data-engineering'],
    params: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Polygon vertices in longitude/latitude.'
      },
      {
        name: 'featureOffsets',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Polygon range per feature.'
      },
      {
        name: 'ringOffsets',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Vertex range per ring.'
      },
      {
        name: 'family',
        type: 'GPUCellFamily',
        required: true,
        description: "'quadbin' or 'h3' (h3 supports center only)."
      },
      {name: 'resolution', type: 'number', required: true, description: 'Cell resolution.'},
      {
        name: 'containment',
        type: 'GPUCellCoverContainment',
        default: "'center'",
        description: "'center', 'full' or 'intersects'."
      },
      {
        name: 'candidateCapacity',
        type: 'number',
        required: true,
        description: 'Upper bound on candidate cells.'
      },
      {
        name: 'output',
        type: 'GPUCellCoverOutput',
        required: true,
        description: 'featureIds, cells and status.'
      }
    ],
    outputs: [
      {
        name: 'featureIds',
        type: 'GraphDataView<"uint32">',
        description: 'Source feature of each cell.'
      },
      {name: 'cells', type: 'GraphDataView<"uint32x2">', description: 'Covering cell keys.'},
      {
        name: 'core',
        type: 'GraphDataView<"uint32">',
        description: 'Optional flag: cell lies fully inside.'
      },
      {
        name: 'status',
        type: '{count, overflow}',
        description: 'count, overflow, candidateOverflow and diagnostics.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCellCover} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'polyfill'});
// Polygon columns in GeoArrow layout, imported with graph.importGPUData().
const {positions, featureOffsets, polygonOffsets, ringOffsets} = polygonViews;
const {featureIds, cells, status} = coverViews; // output views, capacity 1 << 16

graph.add(new GPUCellCover({
  positions, featureOffsets, polygonOffsets, ringOffsets,
  family: 'h3', resolution: 8, candidateCapacity: 1 << 16,
  output: {featureIds, cells, status}
}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'cell-cover',
      seed: 9013,
      caption: 'A polygon filled with the cells whose centers it contains.'
    },
    related: ['cell-set-outline', 'cell-compaction', 'cell-aggregation', 's2-polygon-covering'],
    lensScenario: 'gpu-spatial-analysis.cell-cover',
    reference: 'api/gpu-spatial-analysis/operators-cells',
    maturity: 'stable'
  },
  {
    id: 'cell-set-outline',
    symbol: 'GPUCellSetOutline',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Cell set outline',
    summary: 'Extracts the boundary edges of a set of H3 or Quadbin cells.',
    explanation:
      'Draws the outline of a region made of cells, as `cellsToMultiPolygon` does, without ring assembly. Edges shared by two cells in the set drop out. Add `rings` to stitch the edges into closed rings.',
    dataTypes: {input: ['cells'], output: ['lines']},
    useCases: ['transform'],
    domains: ['cartography'],
    params: [
      {
        name: 'family',
        type: 'GPUCellSetOutlineFamily',
        required: true,
        description: "'h3' or 'quadbin'."
      },
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'Cell keys of the region.'
      },
      {
        name: 'count',
        type: 'GraphDataView<"uint32">',
        description: 'GPU-side number of valid rows.'
      },
      {
        name: 'groups',
        type: 'GraphDataView<"uint32">',
        description: 'Group id per cell; outlines are per group.'
      },
      {
        name: 'rings',
        type: 'GPUCellSetOutlineRings',
        description: 'Also assemble edges into rings.'
      },
      {
        name: 'output',
        type: 'GPUCellSetOutlineOutput',
        required: true,
        description: 'Edge endpoints and status.'
      }
    ],
    outputs: [
      {
        name: 'rows',
        type: "GraphDataView<'uint32'>",
        description: 'Input row of the cell owning each edge.'
      },
      {
        name: 'endpoints',
        type: 'GraphDataView<"float32x4">',
        description: 'Boundary edge as start and end lon/lat.'
      },
      {name: 'cells', type: 'GraphDataView<"uint32x2">', description: 'Owning cell of each edge.'},
      {
        name: 'edgeIndices',
        type: 'GraphDataView<"uint32">',
        description: 'Edge index within its cell.'
      },
      {
        name: 'status',
        type: '{count, overflow, requiredCount?}',
        description: 'Edge count and overflow flag.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCellSetOutline} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'outline'});
const cells = graph.importGPUData('cells', cellData); // GPUData<'uint32x2'>
const {rows, cells: edgeCells, edgeIndices, endpoints, status} = edgeViews; // output views

graph.add(new GPUCellSetOutline({
  family: 'h3', cells,
  output: {rows, cells: edgeCells, edgeIndices, endpoints, status}
}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'polygons',
      seed: 9014,
      caption: 'Only the outer edges of the cell set remain.'
    },
    related: ['cell-cover', 'cell-geometry'],
    lensScenario: 'gpu-spatial-analysis.cell-set-outline',
    reference: 'api/gpu-spatial-analysis/operators-cells',
    maturity: 'stable'
  },

  // ---------------------------------------------------------------------------------------------
  // S2 cells and spherical geometry
  // ---------------------------------------------------------------------------------------------
  {
    id: 's2-cell-hierarchy',
    symbol: 'GPUS2CellHierarchy',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'S2 cell hierarchy',
    summary:
      'Computes exact S2 parents, children, leaf ranges and edge neighbors at all 31 levels.',
    explanation:
      'Moves up and down the S2 tree or step to adjacent cells across cube faces. All arithmetic is integer, so it is exact, including mixed-level batches. Invalid keys and impossible requests report a status code.',
    dataTypes: {input: ['cells'], output: ['cells']},
    useCases: ['index', 'relate'],
    domains: ['data-engineering', 'cartography'],
    params: [
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'S2 cell ids.'
      },
      {
        name: 'operation',
        type: '"parent" | "children" | "range" | "neighbors"',
        required: true,
        description: 'What to derive.'
      },
      {name: 'level', type: 'number', description: 'Target level for parent, children and range.'},
      {
        name: 'output',
        type: '{cells, status}',
        required: true,
        description: 'Result keys and status codes.'
      }
    ],
    outputs: [
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        description: 'Result ids; zero when invalid.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: '0 ok, 1 invalid key, 2 impossible request.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUS2CellHierarchy} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 's2-parents'});
const cells = graph.importGPUData('cells', cellData); // GPUData<'uint32x2'>
const parents = graph.importGPUData('parents', parentData); // GPUData<'uint32x2'>
const status = graph.importGPUData('status', statusData); // GPUData<'uint32'>

graph.add(new GPUS2CellHierarchy({
  cells, operation: 'parent', level: 10, output: {cells: parents, status}
}));
const compiled = await graph.compileAsync();`,
    preview: {kind: 'tessellation', seed: 9015, caption: 'Each cell splits into four children.'},
    related: ['s2-cell-measures', 's2-polygon-covering', 's2-cap-covering', 'cell-topology'],
    lensScenario: 'gpu-spatial-analysis.s2-cell-hierarchy',
    reference: 'api/gpu-spatial-analysis/global-coordinates',
    maturity: 'stable'
  },
  {
    id: 's2-cell-measures',
    symbol: 'GPUS2CellMeasures',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'S2 cell measures',
    summary: 'Measures S2 cell area in steradians and perimeter in radians at levels 0 to 30.',
    explanation:
      'Normalizes by true cell size on the sphere. Local increments avoid the cancellation that plagues fine cells. Multiply by the squared or plain radius of your sphere for physical units.',
    dataTypes: {input: ['cells'], output: ['table']},
    useCases: ['measure'],
    domains: ['cartography', 'demographics'],
    params: [
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'S2 cell ids.'
      },
      {
        name: 'output',
        type: '{measures, status}',
        required: true,
        description: 'Measures and status views.'
      }
    ],
    outputs: [
      {
        name: 'measures',
        type: 'GraphDataView<"float32x2">',
        description: '(area in sr, perimeter in rad) per cell.'
      },
      {name: 'status', type: 'GraphDataView<"uint32">', description: '1 when the key is invalid.'}
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUS2CellMeasures} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 's2-measures'});
const cells = graph.importGPUData('cells', cellData); // GPUData<'uint32x2'>
const measures = graph.importGPUData('measures', measureData); // GPUData<'float32x2'>
const status = graph.importGPUData('status', statusData); // GPUData<'uint32'>

graph.add(new GPUS2CellMeasures({cells, output: {measures, status}}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'cells',
      seed: 9016,
      caption: 'Area and perimeter of each S2 cell on the sphere.'
    },
    related: ['s2-cell-hierarchy', 'cell-measures'],
    lensScenario: 'gpu-spatial-analysis.s2-cell-measures',
    reference: 'api/gpu-spatial-analysis/global-coordinates',
    maturity: 'stable'
  },
  {
    id: 's2-cap-covering',
    symbol: 'GPUS2CapCovering',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'S2 cap covering',
    summary: 'Covers spherical caps (a center and an angular radius) with S2 cells at one level.',
    explanation:
      'Best for "everything within this radius" on a sphere, as a set of cells. `exterior` returns every cell touching the cap; `interior` only cells fully inside. Levels 0 to 8, results in ascending key order.',
    dataTypes: {input: ['points'], output: ['cells']},
    useCases: ['relate', 'index'],
    domains: ['data-engineering', 'cartography'],
    params: [
      {
        name: 'caps',
        type: 'GraphDataView<"float32x4">',
        required: true,
        description: 'Unit xyz center and angular radius per cap.'
      },
      {name: 'level', type: 'number', required: true, description: 'S2 level, 0 to 8.'},
      {
        name: 'mode',
        type: '"exterior" | "interior"',
        default: "'exterior'",
        description: 'Touching cells or fully contained cells.'
      },
      {
        name: 'capacity',
        type: 'number',
        required: true,
        description: 'Maximum cells written per cap.'
      },
      {name: 'tolerance', type: 'number', description: 'Safety margin in radians (default 1e-5).'},
      {
        name: 'output',
        type: '{cells, counts, status}',
        required: true,
        description: 'Cells, per-cap counts and status.'
      }
    ],
    outputs: [
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        description: 'Covering cells, capacity per cap.'
      },
      {
        name: 'counts',
        type: 'GraphDataView<"uint32x2">',
        description: '(written, required) per cap.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: '1 invalid cap, 2 capacity exhausted.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUS2CapCovering} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 's2-caps'});
const caps = graph.importGPUData('caps', capData); // GPUData<'float32x4'>: x, y, z, radius
const cells = graph.importGPUData('cells', cellData); // GPUData<'uint32x2'>, 256 per cap
const counts = graph.importGPUData('counts', countData); // GPUData<'uint32x2'>
const status = graph.importGPUData('status', statusData); // GPUData<'uint32'>

graph.add(new GPUS2CapCovering({
  caps, level: 5, mode: 'exterior', capacity: 256, output: {cells, counts, status}
}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'buffers',
      seed: 9017,
      caption: 'A circular cap on the sphere, covered by cells.'
    },
    related: ['s2-polygon-covering', 's2-cell-hierarchy', 'spherical-convex-contains'],
    lensScenario: 'gpu-spatial-analysis.s2-cap-covering',
    reference: 'api/gpu-spatial-analysis/global-coordinates',
    maturity: 'stable'
  },
  {
    id: 's2-polygon-covering',
    symbol: 'GPUS2PolygonCovering',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'S2 polygon covering',
    summary: 'Builds an adaptive mixed-level S2 interior or exterior cover of spherical polygons.',
    explanation:
      'Indexes polygons on the sphere with the fewest cells, large inside and fine at the edge. The traversal is bounded and runs entirely on the GPU, up to level 15. Set `complement` to cover the outside instead.',
    dataTypes: {input: ['polygons'], output: ['cells']},
    useCases: ['transform', 'index'],
    domains: ['cartography', 'data-engineering'],
    params: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x3">',
        required: true,
        description: 'Unit xyz ring vertices.'
      },
      {
        name: 'ringOffsets',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Vertex range per ring.'
      },
      {
        name: 'maximumLevel',
        type: 'number',
        required: true,
        description: 'Finest S2 level to refine to.'
      },
      {name: 'minimumLevel', type: 'number', default: '0', description: 'Coarsest level allowed.'},
      {
        name: 'mode',
        type: '"exterior" | "interior"',
        default: "'exterior'",
        description: 'Cells touching the polygon, or inside it.'
      },
      {name: 'capacity', type: 'number', required: true, description: 'Maximum cells written.'},
      {name: 'complement', type: 'boolean', description: 'Cover the region outside the polygon.'},
      {
        name: 'output',
        type: '{cells, counts, status, visitedCount?}',
        required: true,
        description: 'Cells, counts and status.'
      }
    ],
    outputs: [
      {
        name: 'cells',
        type: 'GraphDataView<"uint32x2">',
        description: 'Mixed-level covering cells.'
      },
      {
        name: 'counts',
        type: 'GraphDataView<"uint32x2">',
        description: '(written, required) cell counts.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: 'Nonzero when invalid or capacity-limited.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUS2PolygonCovering} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 's2-cover'});
const positions = graph.importGPUData('positions', xyzData); // GPUData<'float32x3'>
const ringOffsets = graph.importGPUData('rings', ringData); // GPUData<'uint32'>
const cells = graph.importGPUData('cells', cellData); // GPUData<'uint32x2'>, 512 rows
const counts = graph.importGPUData('counts', countData); // GPUData<'uint32x2'>
const status = graph.importGPUData('status', statusData); // GPUData<'uint32'>

graph.add(new GPUS2PolygonCovering({
  positions, ringOffsets, maximumLevel: 12, capacity: 512, output: {cells, counts, status}
}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'tessellation',
      seed: 9018,
      caption: 'Big cells inside, small cells along the edge.'
    },
    related: ['s2-cap-covering', 's2-cell-hierarchy', 'cell-cover', 'spherical-polygon-contains'],
    lensScenario: 'gpu-spatial-analysis.s2-polygon-covering',
    reference: 'api/gpu-spatial-analysis/global-coordinates',
    maturity: 'stable'
  },
  {
    id: 'spherical-convex-contains',
    symbol: 'GPUSphericalConvexContains',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spherical convex contains',
    summary:
      'Tests points against convex spherical polygons with explicit boundary and complement rules.',
    explanation:
      'Best for fast containment on the globe when polygons are convex and inside a hemisphere, such as cell boundaries. Choose `closed` or `open` boundaries and flip to the complement region. Status separates invalid, boundary and clean results.',
    dataTypes: {input: ['points', 'polygons'], output: ['table']},
    useCases: ['relate'],
    domains: ['cartography', 'data-engineering'],
    params: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x3">',
        required: true,
        description: 'Unit xyz polygon vertices.'
      },
      {
        name: 'offsets',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Vertex range per polygon.'
      },
      {
        name: 'queries',
        type: 'GraphDataView<"float32x3">',
        required: true,
        description: 'Unit xyz query points.'
      },
      {
        name: 'polygonIndices',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Polygon tested by each query.'
      },
      {
        name: 'boundary',
        type: '"closed" | "open"',
        default: "'closed'",
        description: 'Whether the tolerance band counts as inside.'
      },
      {name: 'complement', type: 'boolean', description: 'Reverse inside and outside.'},
      {
        name: 'tolerance',
        type: 'number',
        default: '1e-6',
        description: 'Angular predicate band in radians.'
      },
      {
        name: 'output',
        type: '{contains, status}',
        required: true,
        description: 'Result flags and status.'
      }
    ],
    outputs: [
      {
        name: 'contains',
        type: 'GraphDataView<"uint32">',
        description: '1 when the query is inside.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: '1 invalid input, 2 boundary hit.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSphericalConvexContains} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'convex-contains'});
const positions = graph.importGPUData('positions', xyzData); // GPUData<'float32x3'>
const offsets = graph.importGPUData('offsets', offsetData); // GPUData<'uint32'>
const queries = graph.importGPUData('queries', queryData); // GPUData<'float32x3'>
const polygonIndices = graph.importGPUData('polygon-ids', idData); // GPUData<'uint32'>
const contains = graph.importGPUData('contains', flagData); // GPUData<'uint32'>
const status = graph.importGPUData('status', statusData); // GPUData<'uint32'>

graph.add(new GPUSphericalConvexContains({
  positions, offsets, queries, polygonIndices, output: {contains, status}
}));
const compiled = await graph.compileAsync();`,
    preview: {kind: 'polygons', seed: 9019, caption: 'Points inside a convex region on the globe.'},
    related: ['spherical-polygon-contains', 'spherical-convex-intersection', 's2-cap-covering'],
    lensScenario: 'gpu-spatial-analysis.spherical-convex-contains',
    reference: 'api/gpu-spatial-analysis/global-coordinates',
    maturity: 'stable'
  },
  {
    id: 'spherical-polygon-contains',
    symbol: 'GPUSphericalPolygonContains',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spherical polygon contains',
    summary: 'Tests points against nonconvex spherical polygons with holes, using winding numbers.',
    explanation:
      'Best for containment on the globe when polygons have holes, concavities or enclose a pole. Rings follow GeoArrow offsets. Boundary and complement semantics are explicit.',
    dataTypes: {input: ['points', 'polygons'], output: ['table']},
    useCases: ['relate'],
    domains: ['cartography', 'data-engineering'],
    params: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x3">',
        required: true,
        description: 'Unit xyz ring vertices.'
      },
      {
        name: 'ringOffsets',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Vertex range per ring.'
      },
      {
        name: 'polygonOffsets',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Ring range per polygon.'
      },
      {
        name: 'queries',
        type: 'GraphDataView<"float32x3">',
        required: true,
        description: 'Unit xyz query points.'
      },
      {
        name: 'polygonIndices',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Polygon tested by each query.'
      },
      {
        name: 'boundary',
        type: '"closed" | "open"',
        default: "'closed'",
        description: 'Whether the tolerance band counts as inside.'
      },
      {name: 'complement', type: 'boolean', description: 'Reverse inside and outside.'},
      {
        name: 'output',
        type: '{contains, status}',
        required: true,
        description: 'Result flags and status.'
      }
    ],
    outputs: [
      {
        name: 'contains',
        type: 'GraphDataView<"uint32">',
        description: '1 when the query is inside.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: 'Nonzero when invalid or on the boundary.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSphericalPolygonContains} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'polygon-contains'});
const positions = graph.importGPUData('positions', xyzData); // GPUData<'float32x3'>
const ringOffsets = graph.importGPUData('rings', ringData); // GPUData<'uint32'>
const polygonOffsets = graph.importGPUData('polygons', polygonData); // GPUData<'uint32'>
const queries = graph.importGPUData('queries', queryData); // GPUData<'float32x3'>
const polygonIndices = graph.importGPUData('polygon-ids', idData); // GPUData<'uint32'>
const {contains, status} = resultViews; // output views

graph.add(new GPUSphericalPolygonContains({
  positions, ringOffsets, polygonOffsets, queries, polygonIndices,
  output: {contains, status}
}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'polygons',
      seed: 9020,
      caption: 'A polygon with a hole: inside, outside, on the edge.'
    },
    related: ['spherical-convex-contains', 'spherical-polygon-overlay', 's2-polygon-covering'],
    lensScenario: 'gpu-spatial-analysis.spherical-polygon-contains',
    reference: 'api/gpu-spatial-analysis/global-coordinates',
    maturity: 'stable'
  },
  {
    id: 'spherical-convex-intersection',
    symbol: 'GPUSphericalConvexIntersection',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spherical convex intersection',
    summary:
      'Intersects paired convex spherical polygons and optionally reports overlap area fractions.',
    explanation:
      'Best for overlap weights between prepared cell boundaries, for example when regridding between grids. Each pair must sit strictly inside an open hemisphere. Optional measures give intersection and union area in steradians.',
    dataTypes: {input: ['polygons'], output: ['polygons']},
    useCases: ['relate', 'transform'],
    domains: ['cartography', 'data-engineering'],
    params: [
      {
        name: 'firstPositions',
        type: 'GraphDataView<"float32x3">',
        required: true,
        description: 'Unit xyz vertices of the first polygons.'
      },
      {
        name: 'secondPositions',
        type: 'GraphDataView<"float32x3">',
        required: true,
        description: 'Unit xyz vertices of the second polygons.'
      },
      {
        name: 'firstOffsets',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Vertex range per first polygon.'
      },
      {
        name: 'secondOffsets',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Vertex range per second polygon.'
      },
      {
        name: 'capacity',
        type: 'number',
        required: true,
        description: 'Output vertex stride per pair, 1 to 64.'
      },
      {
        name: 'tolerance',
        type: 'number',
        default: '1e-6',
        description: 'Angular predicate band in radians.'
      },
      {
        name: 'output',
        type: '{positions, counts, status, measures?}',
        required: true,
        description: 'Vertices, counts, status and measures.'
      }
    ],
    outputs: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x3">',
        description: 'Intersection vertices at a fixed stride.'
      },
      {
        name: 'counts',
        type: 'GraphDataView<"uint32x2">',
        description: '(written, required) vertices per pair.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: 'Bits: 1 invalid, 2 capacity, 4 uncertain.'
      },
      {
        name: 'measures',
        type: 'GraphDataView<"float32x4">',
        description: 'Intersection, source and target fractions, union.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSphericalConvexIntersection} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'convex-intersection'});
const firstPositions = graph.importGPUData('first', firstData); // GPUData<'float32x3'>
const secondPositions = graph.importGPUData('second', secondData); // GPUData<'float32x3'>
const firstOffsets = graph.importGPUData('first-offsets', firstOffsetData);
const secondOffsets = graph.importGPUData('second-offsets', secondOffsetData);
const {positions, counts, status, measures} = resultViews; // output views

graph.add(new GPUSphericalConvexIntersection({
  firstPositions, secondPositions, firstOffsets, secondOffsets, capacity: 16,
  output: {positions, counts, status, measures}
}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'polygon-overlay',
      seed: 9021,
      caption: 'Two convex cells overlap in a smaller convex polygon.'
    },
    related: ['spherical-convex-contains', 'spherical-polygon-overlay'],
    lensScenario: 'gpu-spatial-analysis.spherical-convex-intersection',
    reference: 'api/gpu-spatial-analysis/global-coordinates',
    maturity: 'stable'
  },
  {
    id: 'spherical-polygon-overlay',
    symbol: 'GPUSphericalPolygonOverlay',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spherical polygon overlay',
    summary:
      'Computes Boolean overlay (intersection, union, difference) of nonconvex polygons on the sphere.',
    explanation:
      'Combines regions with holes on the globe. Operands are projected into a hemisphere chart around `chartCenter`, so keep both inside it. Capacity is bounded and overflow is reported, not hidden.',
    dataTypes: {input: ['polygons'], output: ['polygons']},
    useCases: ['transform', 'relate'],
    domains: ['cartography'],
    params: [
      {
        name: 'left',
        type: 'GPUSphericalPolygonOperand',
        required: true,
        description: 'First operand: xyz positions plus offsets.'
      },
      {
        name: 'right',
        type: 'GPUSphericalPolygonOperand',
        required: true,
        description: 'Second operand.'
      },
      {
        name: 'operation',
        type: '"intersection" | "union" | "difference" | "symmetric-difference"',
        required: true,
        description: 'Boolean operation.'
      },
      {
        name: 'chartCenter',
        type: 'readonly [number, number, number]',
        required: true,
        description: 'Unit xyz center of the hemisphere chart.'
      },
      {
        name: 'vertexTolerance',
        type: 'number',
        required: true,
        description: 'Vertex merge distance.'
      },
      {
        name: 'capacity',
        type: 'GPUPolygonOverlayCapacity',
        required: true,
        description: 'Bounds for intersections and output sizes.'
      },
      {
        name: 'leftComplement',
        type: 'boolean',
        description: 'Use the outside of the left operand.'
      },
      {
        name: 'output',
        type: 'GPUSphericalPolygonOverlayOutput',
        required: true,
        description: 'Geometry views and complement flags.'
      }
    ],
    outputs: [
      {
        name: 'geometry',
        type: 'Record<string, GraphDataView>',
        description: 'Result positions (xyz) and ring, polygon offsets.'
      },
      {
        name: 'complement',
        type: 'GraphDataView<"uint32">',
        description: '1 when a result is the complement region.'
      },
      {
        name: 'boundary',
        type: 'Record<string, GraphDataView>',
        description: 'Boundary edges, operand and source ids, and status.'
      },
      {name: 'status', type: '{count, overflow}', description: 'Overflow and uncertainty flags.'}
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSphericalPolygonOverlay} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'spherical-overlay'});
// Operands: {positions (xyz), ringOffsets, polygonOffsets, ...} views on this graph.
const {left, right, capacity, output} = overlayViews;

graph.add(new GPUSphericalPolygonOverlay({
  left, right, operation: 'intersection',
  chartCenter: [0, 1, 0], vertexTolerance: 1e-5,
  capacity, output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();`,
    preview: {
      kind: 'polygon-overlay',
      seed: 9022,
      caption: 'Intersection of two polygons on the sphere.'
    },
    related: [
      'spherical-arc-overlay',
      'spherical-polygon-contains',
      'spherical-convex-intersection'
    ],
    lensScenario: 'gpu-spatial-analysis.spherical-polygon-overlay',
    reference: 'api/gpu-spatial-analysis/global-coordinates',
    maturity: 'partial'
  },
  {
    id: 'spherical-arc-overlay',
    symbol: 'GPUSphericalArcOverlay',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spherical arc overlay',
    summary:
      'Intersects great-circle arc rings directly on the sphere, including regions larger than a hemisphere.',
    explanation:
      'Best when a region exceeds a hemisphere and chart-based overlay would fail. It arranges arcs and assembles rings intrinsically. Work is bounded to 128 input and 1024 output vertices.',
    dataTypes: {input: ['polygons'], output: ['polygons']},
    useCases: ['transform', 'relate'],
    domains: ['cartography'],
    params: [
      {
        name: 'left',
        type: 'GPUSphericalArcOperand',
        required: true,
        description: 'Ring positions, offsets and referenceInside.'
      },
      {
        name: 'right',
        type: 'GPUSphericalArcOperand',
        required: true,
        description: 'Second operand.'
      },
      {
        name: 'referencePoint',
        type: 'readonly [number, number, number]',
        required: true,
        description: 'Unit xyz point with known inside status.'
      },
      {
        name: 'operation',
        type: '"intersection" | "union" | "difference" | "symmetric-difference"',
        required: true,
        description: 'Boolean operation.'
      },
      {
        name: 'tolerance',
        type: 'number',
        default: '1e-6',
        description: 'Angular predicate band in radians.'
      },
      {
        name: 'output',
        type: '{positions, ringOffsets, counts, status}',
        required: true,
        description: 'Result rings and status.'
      }
    ],
    outputs: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x3">',
        description: 'Result ring vertices (xyz).'
      },
      {
        name: 'ringOffsets',
        type: 'GraphDataView<"uint32">',
        description: 'Vertex range per result ring.'
      },
      {
        name: 'counts',
        type: 'GraphDataView<"uint32">',
        description: 'Written ring and vertex counts.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: 'Nonzero when unresolved or over capacity.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSphericalArcOverlay} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'arc-overlay'});
// Operands: {positions (xyz), ringOffsets, referenceInside} with views on this graph.
const {left, right, positions, ringOffsets, counts, status} = arcViews;

graph.add(new GPUSphericalArcOverlay({
  left, right, referencePoint: [0, 0, 1], operation: 'union',
  output: {positions, ringOffsets, counts, status}
}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'polygon-overlay',
      seed: 9023,
      caption: 'Union of two caps larger than a hemisphere.'
    },
    related: ['spherical-polygon-overlay', 'spherical-polygon-contains'],
    lensScenario: 'gpu-spatial-analysis.spherical-arc-overlay',
    reference: 'api/gpu-spatial-analysis/global-coordinates',
    maturity: 'partial'
  },

  // ---------------------------------------------------------------------------------------------
  // Hilbert keys
  // ---------------------------------------------------------------------------------------------
  {
    id: 'hilbert-keys',
    symbol: 'GPUHilbertKeys',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Hilbert keys',
    summary:
      'Computes 2D Hilbert curve indices for points or box centers, optionally sorting rows along the curve.',
    explanation:
      'Orders data so neighbors in space are neighbors in memory, which speeds up joins, indexes and tile building. Hilbert order clusters better than Morton. Orders 1 to 16; invalid rows get a sentinel key and sort last.',
    dataTypes: {input: ['points', 'polygons'], output: ['table']},
    useCases: ['index'],
    domains: ['data-engineering'],
    params: [
      {name: 'positions', type: 'GraphDataView<"float32x2">', description: 'Points to key.'},
      {
        name: 'minima',
        type: 'GraphDataView<"float32x2">',
        description: 'Box minima, with maxima, to key box centers.'
      },
      {name: 'maxima', type: 'GraphDataView<"float32x2">', description: 'Box maxima.'},
      {
        name: 'bounds',
        type: 'GraphDataView<"float32">',
        description: 'Per-frame [minX, minY, maxX, maxY]; reduced from data if omitted.'
      },
      {
        name: 'order',
        type: 'number',
        default: 'GPU_HILBERT_DEFAULT_ORDER',
        description: 'Curve order, 1 to 16 (default 15).'
      },
      {
        name: 'output',
        type: 'GPUHilbertKeysOutput',
        required: true,
        description: 'keys, plus optional sortedRows and sortedKeys.'
      }
    ],
    outputs: [
      {name: 'keys', type: 'GraphDataView<"uint32">', description: 'Hilbert index per item.'},
      {
        name: 'sortedRows',
        type: 'GraphDataView<"uint32">',
        description: 'Stable curve-order permutation of rows.'
      },
      {name: 'sortedKeys', type: 'GraphDataView<"uint32">', description: 'Keys in curve order.'}
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUHilbertKeys} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'hilbert'});
const positions = graph.importGPUData('positions', pointData); // GPUData<'float32x2'>
const keys = graph.importGPUData('keys', keyData); // GPUData<'uint32'>
const sortedRows = graph.importGPUData('sorted-rows', rowData); // GPUData<'uint32'>

graph.add(new GPUHilbertKeys({positions, order: 10, output: {keys, sortedRows}}));
const compiled = await graph.compileAsync();`,
    preview: {kind: 'lines', seed: 9024, caption: 'A space-filling curve visits every cell once.'},
    related: ['spatial-join-prepared', 'spatial-predicate-join', 'point-to-cell'],
    lensScenario: 'gpu-spatial-analysis.hilbert-keys',
    reference: 'api/gpu-spatial-analysis/operators-geometry',
    maturity: 'stable'
  },

  // ---------------------------------------------------------------------------------------------
  // Spatial joins
  // ---------------------------------------------------------------------------------------------
  {
    id: 'spatial-predicate-join',
    symbol: 'GPUSpatialPredicateJoin',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial predicate join',
    summary:
      'Joins two feature sets by a spatial predicate: the GPU equivalent of GeoPandas sjoin.',
    explanation:
      'Best for intersects, contains, within, dwithin, touches, crosses, overlaps or any DE-9IM pattern between points, lines and polygons. A BVH narrows candidates, then exact predicates decide each pair. Pairs come back sorted, with overflow reported on the GPU.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['table', 'spatial-weights']},
    useCases: ['relate'],
    domains: ['data-engineering', 'urban'],
    params: [
      {
        name: 'left',
        type: 'GPUSpatialJoinGeometry',
        required: true,
        description: 'Points, lines or polygons port.'
      },
      {name: 'right', type: 'GPUSpatialJoinGeometry', required: true, description: 'Indexed side.'},
      {
        name: 'predicate',
        type: 'SpatialPredicateName',
        required: true,
        description: "'intersects', 'contains', 'dwithin', 'relate' ..."
      },
      {
        name: 'distance',
        type: 'number | GraphDataView<"float32">',
        description: "Radius for 'dwithin'; a view changes per frame."
      },
      {
        name: 'how',
        type: 'GPUSpatialJoinHow',
        default: "'inner'",
        description: "'inner' pairs, or 'anti' for unmatched left rows."
      },
      {
        name: 'candidateCapacity',
        type: 'number',
        description: 'Maximum bounding-box candidate pairs.'
      },
      {
        name: 'prepared',
        type: 'GPUSpatialJoinPrepared',
        description: 'Reuse a prebuilt right-hand index.'
      },
      {
        name: 'output',
        type: 'GPUSpatialPredicateJoinOutput',
        required: true,
        description: 'pairs, weights, relate, unmatched or status.'
      }
    ],
    outputs: [
      {
        name: 'pairs',
        type: 'GPUCompactPairPort',
        description: 'leftIds, rightIds and status, sorted by pair.'
      },
      {
        name: 'weights',
        type: 'GPUSpatialWeights',
        description: 'Optional matches as a weights graph.'
      },
      {
        name: 'relate',
        type: 'GraphDataView<"uint32">',
        description: 'Optional DE-9IM matrix per pair.'
      },
      {
        name: 'unmatched',
        type: 'GPUCompactOutput',
        description: 'Left rows with no match (anti join).'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSpatialPredicateJoin} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'sjoin'});
const points = {kind: 'points', positions: pointPositions} as const; // views on this graph
const polygons = {kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets} as const;
const {leftIds, rightIds, status} = pairViews; // output views

graph.add(new GPUSpatialPredicateJoin({
  left: points, right: polygons, predicate: 'within',
  candidateCapacity: 1 << 20,
  output: {pairs: {leftIds, rightIds, status}}
}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'polygon-overlay',
      seed: 9025,
      caption: 'Parks matched to the tracts they intersect.'
    },
    related: [
      'point-in-polygon-join',
      'spatial-join-candidates',
      'spatial-join-prepared',
      'nearest-feature-join'
    ],
    lensScenario: 'gpu-spatial-analysis.spatial-predicate-join',
    reference: 'api/gpu-spatial-analysis/operators-joins',
    maturity: 'stable'
  },
  {
    id: 'point-in-polygon-join',
    symbol: 'GPUPointInPolygonJoin',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Point in polygon join',
    summary: 'Joins planar points to the polygon features that contain them.',
    explanation:
      'Tags points with the zone they fall in, or to count points per zone. Candidates come from a BVH and an exact test decides each pair. Choose a `half-open`, `closed` or `open` boundary rule for points on an edge.',
    dataTypes: {input: ['points', 'polygons'], output: ['table']},
    useCases: ['relate', 'summarize'],
    domains: ['data-engineering', 'urban'],
    params: [
      {
        name: 'points',
        type: 'GPUPointGeometryPort',
        required: true,
        description: 'Points to locate.'
      },
      {
        name: 'polygons',
        type: 'GPUPolygonGeometryPort-like',
        required: true,
        description: 'Polygon offsets and positions of the zones.'
      },
      {
        name: 'candidateCapacity',
        type: 'number',
        required: true,
        description: 'Maximum bounding-box candidates.'
      },
      {
        name: 'boundaryRule',
        type: 'GPUPointInPolygonBoundaryRule',
        default: "'half-open'",
        description: "'half-open', 'closed' or 'open'."
      },
      {
        name: 'prepared',
        type: 'GPUSpatialJoinPrepared',
        description: 'Reuse a prebuilt polygon index.'
      },
      {
        name: 'edgeSlabs',
        type: 'boolean',
        description: 'Speed up very large polygons with edge slabs.'
      },
      {
        name: 'output',
        type: 'GPUPointInPolygonJoinOutput',
        required: true,
        description: 'pointFeatureIds, counts, matches, status.'
      }
    ],
    outputs: [
      {
        name: 'pointFeatureIds',
        type: 'GPUUint32Rows',
        description: 'Containing feature id per point.'
      },
      {
        name: 'featureCounts',
        type: 'GraphDataView<"uint32">',
        description: 'Points inside each feature.'
      },
      {
        name: 'matches',
        type: 'GPUCompactOutput',
        description: 'Optional compact list of matched points.'
      },
      {
        name: 'status',
        type: '{count, overflow}',
        description: 'overflow, candidateCount and uncertainCount.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUPointInPolygonJoin} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'pip'});
const points = {kind: 'points', positions: pointPositions} as const; // views on this graph
const polygons = {kind: 'polygons' as const, featureOffsets, polygonOffsets, ringOffsets, positions: zonePositions};
const {pointFeatureIds, featureCounts, status} = joinViews; // output views

graph.add(new GPUPointInPolygonJoin({
  points, polygons, candidateCapacity: 1 << 18,
  output: {pointFeatureIds, featureCounts, status}
}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'polygons',
      seed: 9026,
      caption: 'Points take the color of the zone that contains them.'
    },
    related: ['spatial-predicate-join', 'spatial-join-prepared', 'buffer-selection'],
    lensScenario: 'gpu-spatial-analysis.point-in-polygon-join',
    reference: 'api/gpu-spatial-analysis/operators-joins',
    maturity: 'stable'
  },
  {
    id: 'spatial-join-candidates',
    symbol: 'GPUSpatialJoinCandidates',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial join candidates',
    summary:
      'Lists every left/right feature pair whose bounding boxes overlap, optionally expanded by a distance.',
    explanation:
      'The broad phase for your own exact test, or a way to inspect how much work a join will do. The right side is indexed in a BVH. Pairs land in a compact, capacity-bounded list.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['table']},
    useCases: ['relate', 'index'],
    domains: ['data-engineering'],
    params: [
      {name: 'left', type: 'GPUSpatialJoinGeometry', required: true, description: 'Query side.'},
      {name: 'right', type: 'GPUSpatialJoinGeometry', required: true, description: 'Indexed side.'},
      {
        name: 'distance',
        type: 'number | GraphDataView<"float32">',
        description: 'Expand boxes by this distance.'
      },
      {name: 'leafCapacity', type: 'number', description: 'BVH leaf size.'},
      {
        name: 'prepared',
        type: 'GPUSpatialJoinPrepared',
        description: 'Reuse a prebuilt right-hand index.'
      },
      {
        name: 'output',
        type: 'GPUCompactPairPort',
        required: true,
        description: 'leftIds, rightIds and status.'
      }
    ],
    outputs: [
      {name: 'leftIds', type: 'GraphDataView<"uint32">', description: 'Left feature of each pair.'},
      {
        name: 'rightIds',
        type: 'GraphDataView<"uint32">',
        description: 'Right feature of each pair.'
      },
      {name: 'status', type: '{count, overflow}', description: 'count, overflow and requiredCount.'}
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSpatialJoinCandidates} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'candidates'});
const left = {kind: 'points', positions: pointPositions} as const; // views on this graph
const right = {kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets} as const;
const {leftIds, rightIds, status} = pairViews; // output views

graph.add(new GPUSpatialJoinCandidates({left, right, output: {leftIds, rightIds, status}}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'polygons',
      seed: 9027,
      caption: 'Bounding boxes that overlap become candidate pairs.'
    },
    related: ['spatial-predicate-join', 'spatial-join-prepared', 'hilbert-keys'],
    lensScenario: 'gpu-spatial-analysis.spatial-join-candidates',
    reference: 'api/gpu-spatial-analysis/operators-joins',
    maturity: 'stable'
  },
  {
    id: 'spatial-join-prepared',
    symbol: 'GPUSpatialJoinPrepared',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Prepared join index',
    summary: 'Builds the bounds and BVH of a static feature set once, then reuses it across joins.',
    explanation:
      'Best when the right-hand features rarely change but the left side moves every frame. Pass it as `prepared` to predicate, point-in-polygon, candidate and nearest joins. It rebuilds only when `rebuildWhen` says so.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['table']},
    useCases: ['index', 'relate'],
    domains: ['data-engineering'],
    params: [
      {
        name: 'geometry',
        type: 'GPUSpatialJoinGeometry',
        required: true,
        description: 'Features to index.'
      },
      {name: 'leafCapacity', type: 'number', description: 'BVH leaf size.'},
      {
        name: 'spatialSort',
        type: 'boolean',
        description: 'Sort features along a curve before building.'
      },
      {
        name: 'edgeSlabs',
        type: 'boolean',
        description: 'Precompute edge slabs for large polygons.'
      },
      {
        name: 'partitioning',
        type: 'GPUPartitionDescriptor',
        description: 'Source batch cores and halos.'
      },
      {
        name: 'rebuildWhen',
        type: '((parameters: any) => boolean)',
        description: 'Predicate that invalidates the index.'
      }
    ],
    outputs: [
      {
        name: 'prepared',
        type: 'GPUSpatialJoinPrepared',
        description: 'Pass as `prepared` to any join.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSpatialJoinPrepared, GPUSpatialPredicateJoin} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'prepared-join'});
const zones = {kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets} as const;
const prepared = new GPUSpatialJoinPrepared({id: 'zones', geometry: zones});
const left = {kind: 'points', positions: pointPositions} as const; // moves every frame
const {leftIds, rightIds, status} = pairViews; // output views

graph.add(new GPUSpatialPredicateJoin({
  left, right: zones, predicate: 'within', prepared, candidateCapacity: 1 << 18,
  output: {pairs: {leftIds, rightIds, status}}
}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'tessellation',
      seed: 9028,
      caption: 'Build the index once; query it every frame.'
    },
    related: [
      'spatial-predicate-join',
      'point-in-polygon-join',
      'nearest-feature-join',
      'hilbert-keys'
    ],
    lensScenario: 'gpu-spatial-analysis.spatial-join-prepared',
    reference: 'api/gpu-spatial-analysis/operators-joins',
    maturity: 'stable'
  },
  {
    id: 'nearest-feature-join',
    symbol: 'GPUNearestFeatureJoin',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Nearest feature join',
    summary:
      'Joins each query to its nearest feature, or its k nearest, by planar or great-circle distance.',
    explanation:
      'Use it to snap points to roads, find the closest station or build neighbor lists. Set `radius` to cap the search and `k` with `neighborCapacity` for several neighbors. Ties break by lowest id or keep all.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['table', 'spatial-weights']},
    useCases: ['relate', 'route'],
    domains: ['transport', 'urban'],
    params: [
      {
        name: 'queries',
        type: 'GPUNearestQueryGeometry',
        required: true,
        description: 'Points, lines or polygons to match.'
      },
      {
        name: 'features',
        type: 'GPUNearestFeatureGeometry',
        required: true,
        description: 'Candidate features.'
      },
      {
        name: 'radius',
        type: 'GraphDataView<"float32">',
        description: 'Per-frame maximum distance.'
      },
      {name: 'k', type: 'number', default: '1', description: 'Neighbors per query.'},
      {name: 'neighborCapacity', type: 'number', description: 'Neighbor slots per query.'},
      {
        name: 'ties',
        type: 'GPUNearestTieMode',
        default: "'lowest-id'",
        description: "'lowest-id' or 'all'."
      },
      {
        name: 'exclusive',
        type: 'boolean',
        description: 'Skip the feature with the same id as the query.'
      },
      {
        name: 'output',
        type: 'GPUNearestFeatureJoinOutput',
        required: true,
        description: 'Nearest ids, distances, neighbors and status.'
      }
    ],
    outputs: [
      {name: 'nearestFeatureIds', type: 'GPUUint32Rows', description: 'Closest feature per query.'},
      {name: 'nearestDistances', type: 'GraphDataView<"float32">', description: 'Distance to it.'},
      {
        name: 'neighborIds',
        type: 'GraphDataView<"uint32">',
        description: 'k nearest ids, slots per query.'
      },
      {
        name: 'neighborDistances',
        type: 'GraphDataView<"float32">',
        description: 'Distances for each neighbor slot.'
      },
      {name: 'status', type: '{count, overflow}', description: 'overflow and candidateCount.'}
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUNearestFeatureJoin} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'nearest'});
const queries = {kind: 'points', positions: queryPositions} as const; // views on this graph
const features = {kind: 'points', positions: stationPositions} as const;
const {nearestFeatureIds, nearestDistances, status} = nearestViews; // output views

graph.add(new GPUNearestFeatureJoin({
  queries, features, candidateCapacity: 1 << 16,
  output: {nearestFeatureIds, nearestDistances, status}
}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'spatial-weights',
      seed: 9029,
      caption: 'Each query reaches for its closest feature.'
    },
    related: ['nearest-feature-weights', 'buffer-selection', 'spatial-predicate-join'],
    lensScenario: 'gpu-spatial-analysis.nearest-feature-join',
    reference: 'api/gpu-spatial-analysis/operators-joins',
    maturity: 'stable'
  },
  {
    id: 'nearest-feature-weights',
    symbol: 'GPUNearestFeatureWeights',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Nearest-feature weights',
    summary:
      'Turns k-nearest join output into a spatial weights graph, binary or inverse-distance.',
    explanation:
      'Feeds k-nearest-neighbor weights into spatial statistics such as Moran or Getis-Ord, as libpysal `KNN` does. It reads the neighbor columns of a nearest join. Exclude self-matches for point sets joined to themselves.',
    dataTypes: {input: ['table'], output: ['spatial-weights']},
    useCases: ['relate', 'find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      {
        name: 'neighborIds',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Neighbor ids from the nearest join.'
      },
      {
        name: 'neighborCounts',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Valid neighbors per query.'
      },
      {
        name: 'neighborDistances',
        type: 'GraphDataView<"float32">',
        description: 'Distances; needed for inverse-distance.'
      },
      {
        name: 'slotCapacity',
        type: 'number',
        required: true,
        description: 'Neighbor slots per query.'
      },
      {
        name: 'weightType',
        type: '"binary" | "inverse-distance"',
        default: "'binary'",
        description: 'Weight kind.'
      },
      {name: 'excludeSelf', type: 'boolean', description: 'Drop each row from its own neighbors.'},
      {name: 'power', type: 'number', default: '1', description: 'Inverse-distance exponent.'},
      {
        name: 'output',
        type: 'GPUSpatialWeightsOutput',
        required: true,
        description: 'CSR weights views plus status.'
      }
    ],
    outputs: [
      {name: 'offsets', type: 'GraphDataView<"uint32">', description: 'CSR row offsets.'},
      {name: 'neighbors', type: 'GraphDataView<"uint32">', description: 'CSR neighbor ids.'},
      {name: 'weights', type: 'GraphDataView<"float32">', description: 'CSR weight per neighbor.'},
      {name: 'status', type: '{count, overflow}', description: 'overflow and requiredCount.'}
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUNearestFeatureWeights} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'knn-weights'});
// neighborIds, neighborCounts, neighborDistances come from GPUNearestFeatureJoin (k = 8).
const {neighborIds, neighborCounts, neighborDistances} = nearestViews;
const weights = weightViews; // GPUSpatialWeights output views plus status

graph.add(new GPUNearestFeatureWeights({
  neighborIds, neighborCounts, neighborDistances, slotCapacity: 8,
  excludeSelf: true, weightType: 'inverse-distance', output: weights
}));
const compiled = await graph.compileAsync();`,
    preview: {
      kind: 'spatial-weights',
      seed: 9030,
      caption: 'Each point links to its k nearest neighbors.'
    },
    related: ['nearest-feature-join', 'buffer-selection'],
    lensScenario: 'gpu-spatial-analysis.nearest-feature-weights',
    reference: 'api/gpu-spatial-analysis/operators-joins',
    maturity: 'stable'
  },
  {
    id: 'buffer-selection',
    symbol: 'GPUBufferSelection',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Buffer selection',
    summary: 'Selects points within a per-frame planar distance of point or polyline features.',
    explanation:
      'Best for interactive "within X of this road" filters: drag the distance and the selection updates with no recompile. It writes a mask, a compact id list and an instance count for indirect draws. Overflow is flagged on the GPU.',
    dataTypes: {input: ['points', 'lines'], output: ['points']},
    useCases: ['relate', 'transform'],
    domains: ['transport', 'urban'],
    params: [
      {
        name: 'points',
        type: 'GPUPointGeometryPort',
        required: true,
        description: 'Points to test.'
      },
      {
        name: 'features',
        type: 'GPUNearestFeatureSource',
        required: true,
        description: 'Point or polyline features.'
      },
      {
        name: 'distance',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-frame selection distance.'
      },
      {
        name: 'candidateCapacity',
        type: 'number',
        required: true,
        description: 'Maximum candidate pairs.'
      },
      {name: 'spatialSort', type: 'boolean', description: 'Sort points along a curve first.'},
      {
        name: 'output',
        type: 'GPUBufferSelectionOutput',
        required: true,
        description: 'outputMask, matches, drawInstanceCount ...'
      }
    ],
    outputs: [
      {name: 'outputMask', type: 'GPUUint32Rows', description: '0/1 flag per point.'},
      {
        name: 'matches',
        type: 'GPUCompactOutput',
        description: 'Selected ids in ascending row order.'
      },
      {
        name: 'drawInstanceCount',
        type: 'GraphDataView<"uint32">',
        description: 'Selected count for an indirect draw.'
      },
      {
        name: 'distances',
        type: 'GraphDataView<"float32">',
        description: 'Distance of each point to its nearest feature.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUBufferSelection, GPUParameterBuffer} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'buffer-select'});
const distance = new GPUParameterBuffer(device, {id: 'distance', format: 'float32', length: 1});
const points = {kind: 'points', positions: pointPositions} as const; // views on this graph
const features = {kind: 'points', positions: roadPositions} as const;
const {outputMask} = selectionViews; // output views

graph.add(new GPUBufferSelection({
  points, features, distance: distance.importToGraph(graph),
  candidateCapacity: 1 << 18, output: {outputMask}
}));
distance.write(Float32Array.of(250)); // drag the slider: no recompile`,
    preview: {
      kind: 'buffers',
      seed: 9031,
      caption: 'Points inside the distance band stay selected.'
    },
    related: ['nearest-feature-join', 'point-in-polygon-join', 'spatial-predicate-join'],
    lensScenario: 'gpu-spatial-analysis.buffer-selection',
    reference: 'api/gpu-spatial-analysis/operators-joins',
    maturity: 'stable'
  }
];
