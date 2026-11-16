// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUAnalysisResultClasses} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getKeplerResultToneColor, getKeplerResultRampColors} from '@luma-contributors/kepler';
import {packGPUColor} from '@luma-contributors/analysis/dataframe-analysis';
import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';

/** Runtime ordinal codes retain the contributor's quadrant order; ontology owns the colours. */
export function getAnalysisMoranColors() {
  const classes = getGPUAnalysisResultClasses('local-moran-clusters', 'catalog-text');
  return ['not-significant', 'HH', 'LH', 'LL', 'HL', 'not-tested'].map(value =>
    getKeplerResultToneColor(classes.find(entry => entry.value === value)!.tone)
  );
}
export function getAnalysisHotspotColors() {
  return [...getGPUAnalysisResultClasses('hot-spot-confidence-bins')]
    .sort((first, second) => Number(first.value) - Number(second.value))
    .map(entry => getKeplerResultToneColor(entry.tone));
}

/** One bounded lookup row per possible produced class count (the runtime supports at most eight). */
export function getAnalysisRampPalette() {
  return new Uint32Array(
    Array.from({length: 8}, (_, countIndex) => {
      const colors = getKeplerResultRampColors('sequential', countIndex + 1);
      return Array.from({length: 8}, (_, index) =>
        packGPUColor(...colors[Math.min(index, countIndex)])
      );
    }).flat()
  );
}
export function addAnalysisRampPalette(
  graph: GPUCommandGraph,
  palettes: GraphDataView<'uint32'>,
  classCount: GraphDataView<'uint32'>
) {
  const palette = createTransientView(graph, 'analysis-active-palette', 'uint32', 8);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'analysis-active-palette',
      operation: 'Analysis result colours',
      variant: 'produced-classes',
      bindings: [
        {name: 'palettes', view: palettes, type: 'u32', access: 'read'},
        {name: 'classCount', view: classCount, type: 'u32', access: 'read'},
        {name: 'palette', view: palette, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 8,
      body: `let count = clamp(classCount[classCountOffset], 1u, 8u);
      palette[paletteOffset + index] = palettes[palettesOffset + (count - 1u) * 8u + index];`
    })
  );
  return palette;
}

/** Display codes distinguish untested rows without changing published analysis quadrants/bins. */
export function addAnalysisInferenceColorIndices(
  graph: GPUCommandGraph,
  codes: GraphDataView<'uint32'> | GraphDataView<'sint32'>,
  pValues: GraphDataView<'float32'>,
  hotspot: boolean
) {
  const indices = createTransientView(
    graph,
    'analysis-inference-color-indices',
    'uint32',
    codes.length
  );
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'analysis-inference-color-indices',
      operation: 'Analysis result colours',
      variant: hotspot ? 'hotspot' : 'moran',
      bindings: [
        {name: 'codes', view: codes, type: hotspot ? 'i32' : 'u32', access: 'read'},
        {name: 'pValues', view: pValues, type: 'f32', access: 'read'},
        {name: 'indices', view: indices, type: 'u32', access: 'read_write'}
      ],
      invocationCount: codes.length,
      body: `let probability = pValues[pValuesOffset + index];
      let tested = abs(probability) <= 3.402823466e38;
      indices[indicesOffset + index] = select(${hotspot ? '0xffffffffu' : '5u'},
        ${hotspot ? 'u32(codes[codesOffset + index] + 3)' : 'codes[codesOffset + index]'}, tested);`
    })
  );
  return indices;
}
