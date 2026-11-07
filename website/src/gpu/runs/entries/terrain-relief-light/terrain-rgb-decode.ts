// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainRGBDecode on an elevation tile packed the way web terrain services ship it: height in the
// red, green and blue bytes (Terrarium or Mapbox), with transparent pixels where data is missing.
// The kernel unpacks the bytes to meters. The context is the red channel as stored; the
// output is the decoded terrain, with nodata cells (transparent pixels, heights outside validRange)
// left blank. All three props are construction props.

import {GPUTerrainRGBDecode} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawGrid, makeTerrainGrid} from './support/terrain';
import {parameters} from './terrain-rgb-decode.parameters';

/** Packs a height as the RGBA word of an encoded tile pixel (red in the low byte). */
function packHeight(height: number, encoding: 'terrarium' | 'mapbox', alpha: number): number {
  let red: number;
  let green: number;
  let blue: number;
  if (encoding === 'terrarium') {
    const shifted = height + 32768;
    red = Math.floor(shifted / 256);
    green = Math.floor(shifted) % 256;
    blue = Math.floor((shifted - Math.floor(shifted)) * 256);
  } else {
    const shifted = Math.round((height + 10000) * 10);
    red = Math.floor(shifted / 65536);
    green = Math.floor(shifted / 256) % 256;
    blue = shifted % 256;
  }
  return (red | (green << 8) | (blue << 16) | (alpha << 24)) >>> 0;
}

export default defineOperationRun({
  parameters,
  rebuildOn: ['encoding', 'alphaNoData', 'validMaximum'],
  caption: 'An RGBA elevation tile in; heights in meters, with missing data marked, out.',
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(random, world, 48);
    // Transparent holes: pixels the service had no data for, left holding a wrong height of 0.
    const holes = Array.from({length: 4}, () => ({
      column: Math.floor(random() * (terrain.columns - 10)),
      row: Math.floor(random() * (terrain.rows - 8)),
      width: 4 + Math.floor(random() * 6),
      height: 3 + Math.floor(random() * 5)
    }));
    const isHole = (index: number) => {
      const column = index % terrain.columns;
      const row = Math.floor(index / terrain.columns);
      return holes.some(
        hole =>
          column >= hole.column &&
          column < hole.column + hole.width &&
          row >= hole.row &&
          row < hole.row + hole.height
      );
    };
    return {...terrain, isHole};
  },
  build(builder, terrain, values) {
    const encoding = values.encoding === 1 ? 'mapbox' : 'terrarium';
    const words = Uint32Array.from(terrain.elevation, (height, index) =>
      terrain.isHole(index) ? packHeight(0, encoding, 0) : packHeight(height, encoding, 255)
    );
    const tile = builder.upload('tile', 'uint32', words);
    const storedRed = builder.upload(
      'stored-red',
      'float32',
      Float32Array.from(words, word => word & 255)
    );
    const decoded = builder.allocate('decoded', 'float32', words.length);
    const outputValidity = builder.allocate('output-validity', 'uint32', words.length);
    builder.graph.add(
      new GPUTerrainRGBDecode({
        grid: makeTerrainGrid(terrain),
        encoding,
        input: {buffer: tile.view},
        alphaNoData: values.alphaNoData === 1,
        validRange: [-11000, values.validMaximum],
        output: {values: decoded.view, outputValidity: outputValidity.view}
      })
    );
    return {outputs: {storedRed, decoded}};
  },
  draw: ({inputs, outputs}) => [
    drawGrid(inputs, outputs.storedRed, {context: true}),
    drawGrid(inputs, outputs.decoded, {hillshade: 0.5, elevation: 60})
  ]
});
