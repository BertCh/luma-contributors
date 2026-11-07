// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {build} from 'esbuild';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const showcaseDirectory = resolve(scriptDirectory, '..');
const sourceDirectory = resolve(showcaseDirectory, 'src/scenes/raster');
const manifest = JSON.parse(
  await readFile(resolve(showcaseDirectory, 'public/data/dixie-spectral/manifest.json'), 'utf8')
);
const sourceRaster = manifest.rasters.nir_before;
const [minimumEasting, , , maximumNorthing] = sourceRaster.boundsProjected;
const sourceCellSize = sourceRaster.cellSizeM;
const gridSize = 64;
const sourceRow = 0;
const sourceColumn = 32;
const cropWest = minimumEasting + sourceColumn * sourceCellSize;
const cropNorth = maximumNorthing - sourceRow * sourceCellSize;
const temporaryDirectory = await mkdtemp(resolve(tmpdir(), 'grid-alignment-reference-'));
const helperBundlePath = resolve(temporaryDirectory, 'grid-alignment-projection.mjs');
const outputPath = process.argv[2] ?? '/tmp/grid-alignment-adaptive-projection.json';

try {
  await build({
    entryPoints: [resolve(sourceDirectory, 'grid-alignment-projection.ts')],
    outfile: helperBundlePath,
    bundle: true,
    platform: 'node',
    format: 'esm'
  });
  const {createGridAlignmentProjection, getGridAlignmentDestinationCenter} = await import(
    pathToFileURL(helperBundlePath).href
  );
  const destinationCenter = getGridAlignmentDestinationCenter(
    cropWest,
    cropNorth,
    gridSize,
    sourceCellSize
  );
  const sourceNorthwest = [minimumEasting, maximumNorthing];
  const halfWidth = (gridSize * sourceCellSize) / 2;
  const projection = createGridAlignmentProjection(destinationCenter, sourceNorthwest, halfWidth);
  const centers = [];
  for (let row = 0; row < gridSize; row++) {
    for (let column = 0; column < gridSize; column++) {
      const localMercatorCenter = [
        (column + 0.5) * sourceCellSize - halfWidth,
        halfWidth - (row + 0.5) * sourceCellSize
      ];
      centers.push({
        row,
        column,
        localMercatorCenter,
        providerOffset: projection.evaluateProvider(localMercatorCenter),
        planOffset: projection.evaluatePlan(localMercatorCenter)
      });
    }
  }
  assert.equal(centers.length, gridSize * gridSize);
  const output = {
    sourceDataset: manifest.id,
    sourceCrs: manifest.crs,
    sourceDimensions: [sourceRaster.width, sourceRaster.height],
    crop: {row: sourceRow, column: sourceColumn, width: gridSize, height: gridSize},
    destinationCenter,
    globalNorthwest: sourceNorthwest,
    sourceGridAffine: [sourceCellSize, 0, 0, 0, -sourceCellSize, 0],
    plan: {
      precision: 'double-single',
      degree: projection.plan.degree,
      toleranceMeters: projection.sampledTolerance,
      patchCount: projection.patchCount,
      maximumSampledErrorMeters: projection.maximumError
    },
    centers
  };
  await writeFile(outputPath, `${JSON.stringify(output)}\n`);
  console.log(
    JSON.stringify({
      outputPath,
      centerCount: centers.length,
      destinationCenter,
      globalNorthwest: sourceNorthwest,
      plan: output.plan
    })
  );
} finally {
  await rm(temporaryDirectory, {recursive: true, force: true});
}
