// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPURasterContrastEqualization,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

type Options = {
  mode: 'clahe' | 'match-histograms';
  width: number;
  height: number;
  tileWidth: number;
  tileHeight: number;
  binCount: number;
  texture?: boolean;
};

function addContrast(context: KernelAuditContext, options: Options): void {
  const {graph, view} = context;
  const {width, height, mode, tileWidth, tileHeight, binCount} = options;
  const count = width * height;
  const grid = makeGPURasterMetadata({width, height});
  const metadata = {
    id: 'source',
    format: 'float32' as const,
    validity: view('uint32', count),
    scale: 0.5,
    offset: 2,
    noDataValue: -9999
  };
  const input: GPURasterBand = options.texture
    ? {
        ...metadata,
        storage: {
          kind: 'texture',
          view: graph.createTextureView(context.texture('r32float', width, height))
        }
      }
    : {...metadata, storage: {kind: 'buffer', values: view('float32', count)}};
  const contributor = new GPURasterContrastEqualization({
    grid,
    input,
    mode,
    tileWidth,
    tileHeight,
    binCount,
    reference:
      mode === 'match-histograms'
        ? {
            grid,
            input: {
              id: 'reference',
              format: 'uint32',
              storage: {kind: 'buffer', values: view('uint32', count)},
              validity: view('uint32', count),
              noDataValue: 0xffffffff,
              scale: 0.25
            }
          }
        : undefined,
    parameters: view('float32', 8),
    output: {
      values: view('float32', count),
      validity: view('uint32', count),
      status: {approximation: view('uint32', 1)}
    }
  });
  const nodes = contributor.getCommandNodes(graph);
  if (nodes.length > contributor.estimatedCommandNodeCount) {
    throw new Error(
      `raster-contrast-equalization estimated ${contributor.estimatedCommandNodeCount} nodes but declared ${nodes.length}`
    );
  }
  for (const node of nodes) graph.add(node);
}

export const RASTER_CONTRAST_EQUALIZATION_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPURasterContrastEqualization',
    name: 'CLAHE odd tiles and minimum bins',
    build: context =>
      addContrast(context, {
        mode: 'clahe',
        width: 7,
        height: 5,
        tileWidth: 3,
        tileHeight: 4,
        binCount: 2
      })
  },
  {
    exportName: 'GPURasterContrastEqualization',
    name: 'CLAHE maximum cells and tile area',
    build: context =>
      addContrast(context, {
        mode: 'clahe',
        width: 4096,
        height: 4096,
        tileWidth: 256,
        tileHeight: 256,
        binCount: 256
      })
  },
  {
    exportName: 'GPURasterContrastEqualization',
    name: 'CLAHE maximum histogram length',
    build: context =>
      addContrast(context, {
        mode: 'clahe',
        width: 4096,
        height: 4096,
        tileWidth: 16,
        tileHeight: 16,
        binCount: 256
      })
  },
  {
    exportName: 'GPURasterContrastEqualization',
    name: 'CLAHE calibrated texture',
    build: context =>
      addContrast(context, {
        mode: 'clahe',
        width: 32,
        height: 32,
        tileWidth: 8,
        tileHeight: 8,
        binCount: 256,
        texture: true
      })
  },
  {
    exportName: 'GPURasterContrastEqualization',
    name: 'matching maximum cells and bins',
    build: context =>
      addContrast(context, {
        mode: 'match-histograms',
        width: 4096,
        height: 4096,
        tileWidth: 32,
        tileHeight: 32,
        binCount: 256
      })
  },
  {
    exportName: 'GPURasterContrastEqualization',
    name: 'matching minimum bins and texture',
    build: context =>
      addContrast(context, {
        mode: 'match-histograms',
        width: 9,
        height: 7,
        tileWidth: 1,
        tileHeight: 1,
        binCount: 2,
        texture: true
      })
  }
];
