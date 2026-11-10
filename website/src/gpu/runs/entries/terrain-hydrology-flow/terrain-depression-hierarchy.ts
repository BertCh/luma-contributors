// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainDepressionHierarchy on a cratered DEM: every cell is assigned to the depression it
// belongs to (colors), and the area and volume curves of all depressions at the chosen elevation
// stages are written as bars. With too small a nodeCapacity the tree is rejected.

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import {GPUTerrainDepressionHierarchy} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun, type RunInputContext} from '../../define-run';
import {parameters} from './terrain-depression-hierarchy.parameters';
import {
  createNestedCraterTerrainInputs,
  getTerrainCell,
  getTerrainContextLayer
} from './shared/terrain';

const NO_NODE = 0xffffffff;
const CHART_RECTANGLE = [0.04, 0.04, 0.96, 0.3] as const;

function createDepressionInputs(context: RunInputContext) {
  const terrain = createNestedCraterTerrainInputs(context, 40);
  // Water leaves the map across the whole border.
  const borderMask = new Uint32Array(terrain.cellCount);
  for (let cell = 0; cell < terrain.cellCount; cell++) {
    const column = cell % terrain.columns;
    const row = Math.floor(cell / terrain.columns);
    const onBorder =
      column === 0 || row === 0 || row === terrain.rows - 1 || column === terrain.columns - 1;
    borderMask[cell] = onBorder ? 1 : 0;
  }
  return {...terrain, borderMask, focus: {node: -2, source: null as Uint32Array | null}};
}

function getRestPoint(
  inputs: {
    subPitCenters: [number, number][];
    columns: number;
    rows: number;
    cellSize: number;
  },
  toUnit: (point: readonly [number, number]) => readonly [number, number]
) {
  const [column, row] = inputs.subPitCenters[0];
  return toUnit([
    (column / inputs.columns) * (inputs.columns * inputs.cellSize),
    (1 - row / inputs.rows) * (inputs.rows * inputs.cellSize)
  ]);
}

export default defineOperationRun({
  parameters,
  caption:
    'Water depth when each depression is filled to its spill level; the chart is the hovered depression over the stages.',
  createInputs: context => createDepressionInputs(context),
  build(builder, terrain, values) {
    const stageCount = Math.round(values.stages);
    const nodeCapacity = Math.round(values.nodeCapacity);
    const [low, high] = terrain.elevationRange;
    const stageLevels = Float32Array.from(
      {length: stageCount},
      (_, index) => low + ((index + 1) / stageCount) * (high - low)
    );
    const elevation = builder.upload('dem', 'float32', terrain.elevation);
    const outlets = builder.upload('outlets', 'uint32', terrain.borderMask);
    const nodes = builder.allocate('nodes', 'uint32', nodeCapacity * 4);
    const levels = builder.allocate('levels', 'float32', nodeCapacity * 2);
    const cellNodes = builder.allocate('cell-nodes', 'uint32', terrain.cellCount);
    const curves = builder.allocate('curves', 'float32', nodeCapacity * stageCount * 2);
    builder.graph.add(
      new GPUTerrainDepressionHierarchy({
        grid: terrain.grid,
        elevation: elevation.view,
        outlets: outlets.view,
        stages: builder.upload('stages', 'float32', stageLevels).view,
        nodeCapacity,
        output: {
          nodes: nodes.view,
          levels: levels.view,
          cellNodes: cellNodes.view,
          curves: curves.view,
          status: builder.allocate('status', 'uint32', 3).view
        }
      })
    );
    // How deep water stands over every cell when its depression is full: the spill level of the
    // depression that owns the cell, minus the ground. Depressions that drain off the map stay empty.
    const fillDepth = builder.allocate('fill-depth', 'float32', terrain.cellCount);
    builder.graph.add({
      getCommandNodes: graph => [
        createWGSLKernelNode(graph, {
          id: 'fill-depth',
          operation: 'fill-depth',
          bindings: [
            {name: 'owners', view: cellNodes.view, type: 'u32', access: 'read'},
            {name: 'levels', view: levels.view, type: 'f32', access: 'read'},
            {name: 'ground', view: elevation.view, type: 'f32', access: 'read'},
            {name: 'depth', view: fillDepth.view, type: 'f32', access: 'read_write'}
          ],
          invocationCount: terrain.cellCount,
          body: `let node = owners[ownersOffset + index];
  let spill = levels[levelsOffset + min(node, ${nodeCapacity - 1}u) * 2u + 1u];
  let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
  depth[depthOffset + index] = select(missing, spill - ground[groundOffset + index], node < ${nodeCapacity}u && spill < 1.0e20);`
        })
      ]
    });
    // The hovered depression, its parent and the chart of its curves: written by the pointer.
    const highlight = builder.allocate('highlight', 'uint32', terrain.cellCount);
    const curve = builder.allocate('curve', 'float32', stageCount * 2);
    return {
      outputs: {elevation, nodes, levels, cellNodes, curves, fillDepth, highlight, curve}
    };
  },
  rebuildOn: ['nodeCapacity', 'stages'],
  // The pointer picks the depression under it: its cells and everything draining into it are lit,
  // the parent it merges into is tinted, and the chart shows its area and volume at each stage.
  interaction: {
    kind: 'probe',
    hint: 'hover a depression',
    rest: ({inputs, toUnit}) => getRestPoint(inputs, toUnit),
    resolve(pointer, {inputs, outputs, values, world, peek, toUnit}) {
      const cellNodes = peek(outputs.cellNodes) as Uint32Array | null;
      const nodes = peek(outputs.nodes) as Uint32Array | null;
      const levels = peek(outputs.levels);
      const curves = peek(outputs.curves);
      if (!cellNodes || !nodes || !levels || !curves) return {};
      const nodeCapacity = Math.round(values.nodeCapacity);
      const stageCount = Math.round(values.stages);
      const cell = getTerrainCell(inputs, world, pointer.x, pointer.y);
      const [restX, restY] = getRestPoint(inputs, toUnit);
      const isRest = Math.hypot(pointer.unitX - restX, pointer.unitY - restY) < 0.003;
      const focused = !isRest && cellNodes[cell.index] < nodeCapacity ? cellNodes[cell.index] : -1;
      // The chart keeps the depression under the pointer even at rest, so it always shows curves.
      const charted = cellNodes[cell.index] < nodeCapacity ? cellNodes[cell.index] : -1;
      const focusKey = focused * 100000 + charted * 100 + stageCount;
      if (inputs.focus.node !== focusKey || inputs.focus.source !== cellNodes) {
        inputs.focus = {node: focusKey, source: cellNodes};
        const parent = focused < 0 ? NO_NODE : nodes[focused * 4];
        const isInside = (start: number, ancestor: number) => {
          let node = start;
          for (let step = 0; step < nodeCapacity && node < nodeCapacity; step++) {
            if (node === ancestor) return true;
            node = nodes[node * 4];
          }
          return false;
        };
        outputs.highlight.write(
          Uint32Array.from(cellNodes, owner => {
            if (focused < 0 || owner >= nodeCapacity) return 0;
            if (isInside(owner, focused)) return 2;
            return parent < nodeCapacity && isInside(owner, parent) ? 1 : 0;
          })
        );
        let maximumArea = 1e-9;
        let maximumVolume = 1e-9;
        for (let index = 0; index < nodeCapacity * stageCount; index++) {
          maximumArea = Math.max(maximumArea, curves[index * 2] || 0);
          maximumVolume = Math.max(maximumVolume, curves[index * 2 + 1] || 0);
        }
        const chart = new Float32Array(stageCount * 2);
        if (charted >= 0) {
          for (let stage = 0; stage < stageCount; stage++) {
            chart[stage] = (curves[(charted * stageCount + stage) * 2] || 0) / maximumArea;
            chart[stageCount + stage] =
              (curves[(charted * stageCount + stage) * 2 + 1] || 0) / maximumVolume;
          }
        }
        outputs.curve.write(chart);
      }
      if (focused < 0) return {focus: -1, spotlight: null};
      const spill = levels[focused * 2 + 1];
      const parent = nodes[focused * 4];
      const volume = curves[(focused * stageCount + stageCount - 1) * 2 + 1] || 0;
      return {
        focus: focused,
        spotlight: null,
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: `depression ${focused} · spill ${Number.isFinite(spill) ? spill.toFixed(3) : 'open'} · ${parent < nodeCapacity ? `parent ${parent}` : 'root'} · volume ${volume.toFixed(2)}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    getTerrainContextLayer(inputs, outputs.elevation),
    {
      layer: 'raster',
      values: outputs.fillDepth,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: [0, 0.06],
      alpha: 0.85
    },
    {
      layer: 'raster',
      values: outputs.highlight,
      columns: inputs.columns,
      rows: inputs.rows,
      categories: [
        [0, 0, 0, 0],
        [1, 1, 1, 0.22],
        [1, 0.92, 0.35, 0.5]
      ]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.curve,
      ramp: 'sequential',
      range: [0, 1],
      rectangle: CHART_RECTANGLE
    }
  ]
});
