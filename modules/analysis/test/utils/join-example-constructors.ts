// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import nearTransitScene from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/joins/near-transit.scene';
import {entries} from '../../../../website/src/catalog/entries/regression-weights';
import type {GPUPairGatherProps} from '../../src/gpu-spatial-analysis/pair-gather/gpu-pair-gather';
import type {GPUBufferSelectionProps} from '../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';

function evaluateConstructor<Props>(
  source: string,
  name: string,
  bindings: Record<string, unknown>
): Props {
  const prefix = `new ${name}(`;
  const start = source.indexOf(prefix);
  const end = source.indexOf('\n  })', start);
  if (start < 0 || end < 0) throw new Error(`Missing authored ${name} constructor`);
  const argument = source.slice(start + prefix.length, end) + '\n  }';
  return new Function(...Object.keys(bindings), `return (${argument});`)(
    ...Object.values(bindings)
  );
}

/** Evaluates the authored card, retaining the actual pair and attribute view identities. */
export function getPairGatherCatalogProps(bindings: Record<string, unknown>): GPUPairGatherProps {
  const example = entries.find(entry => entry.id === 'pair-gather')?.example;
  if (!example) throw new Error('Missing pair-gather catalog example');
  return evaluateConstructor(example, 'GPUPairGather', bindings);
}

/** Evaluates the rendered constructor rather than reconstructing its operands in a test. */
export function getNearTransitSnippetProps(
  features: 'stations' | 'bus' | 'lines',
  distance: number,
  bindings: Record<string, unknown>
): GPUBufferSelectionProps {
  const render = nearTransitScene.snippet;
  if (typeof render !== 'function') throw new Error('Missing near-transit snippet renderer');
  const snippet = render({
    points: 'observations',
    features,
    line: 'all',
    distance,
    spatialSort: false,
    colorBy: 'distance',
    ramp: 'cividis',
    showBuffer: true,
    showOutside: true,
    pointSize: 3,
    opacity: 1
  });
  return evaluateConstructor(snippet, 'GPUBufferSelection', bindings);
}
