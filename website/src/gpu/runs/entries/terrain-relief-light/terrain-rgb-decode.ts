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
  rebuildOn: ['alphaNoData', 'validMaximum'],
  caption: 'An RGBA elevation tile in; heights in meters, with missing data marked, out.',
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(world, 48);
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
    // Both encodings decode to the same heights, so the tile is packed as Terrarium.
    const encoding = 'terrarium';
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
  // The pointer is a curtain: the red byte as stored on the left, the decoded meters on the right.
  interaction: {
    kind: 'swipe',
    hint: 'swipe tile and meters',
    rest: [0.5, 0.5],
    resolve: (pointer, {values}) => ({
      curtain: pointer.x,
      glyphs: [{shape: 'curtain', x: pointer.x}],
      readout: `RGBA tile | metres up to ${values.validMaximum} m`
    })
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = interaction.curtain ?? world.width / 2;
    return [
      // Terrarium's red byte steps once per 256 m, so the stored picture is terraced.
      drawGrid(inputs, outputs.storedRed, {range: [128, 138], clip: {side: 'left', x: curtain}}),
      drawGrid(inputs, outputs.decoded, {
        range: [650, 2300],
        hillshade: 0.5,
        elevation: 60,
        clip: {side: 'right', x: curtain}
      })
    ];
  }
});
