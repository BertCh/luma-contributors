// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {setAnalysisSheetParameter} from './analysis-sheet-controls.mjs';

export const NETWORK_CATEGORICAL_OPERATIONS = [
  ['kernel-flow-smoothing', 'GPUKernelFlowSmoothing'],
  ['map-matched-edge-volumes', 'GPUMapMatchedEdgeVolumes'],
  ['radiation-flow-model', 'GPURadiationFlowModel'],
  ['network-potential-path-area', 'GPUNetworkPotentialPathArea'],
  ['network-travel-time-percentiles', 'GPUNetworkTravelTimePercentiles'],
  ['colocation-quotient', 'GPUColocationQuotient'],
  ['local-category-dependence', 'GPULocalCategoryDependence'],
  ['spatial-entropy', 'GPUSpatialEntropy'],
  ['category-pair-counts', 'GPUCategoryPairCounts'],
  ['geographical-detector', 'GPUGeographicalDetector'],
  ['redcap-single-linkage', 'GPURedcapSingleLinkage'],
  ['healpix-point-to-cell', 'GPUHealpixPointToCell'],
  ['healpix-cell-aggregation', 'GPUHealpixCellAggregation'],
  ['datum-grid-transform', 'GPUDatumGridTransform'],
  ['sinkhorn-divergence', 'GPUSinkhornDivergence']
];

/** Run the actual panel and require a GPU result, failing promptly on a visible catalog error. */
export async function runNetworkCategoricalCatalog(page, panel, contributor) {
  // OD text supplied by flow-h3 has known EPSG:4326 coordinates and identity columns.
  // Selecting them explicitly confirms those roles under AX preflight, as a user must.
  if (await panel.getByLabel('Operation', {exact: true}).inputValue() === 'kernel-flow-smoothing') {
    await setAnalysisSheetParameter(panel, 'coordinateSystem', 'EPSG:4326');
    await setAnalysisSheetParameter(panel, 'originProperty', 'origin');
    await setAnalysisSheetParameter(panel, 'destinationProperty', 'destination');
  }
  await panel.getByTestId('catalog-run').click();
  await page.waitForFunction(expected => {
    const error = document.querySelector('[data-testid="catalog-error"]');
    if (error?.textContent?.trim()) throw new Error(error.textContent);
    const engine = document.querySelector('[data-testid="catalog-engine"]')?.textContent ?? '';
    return engine.includes('Engine: GPU') && engine.includes(expected);
  }, contributor, {timeout: 60000});
  assert.match(await panel.getByTestId('catalog-result').innerText(), /[1-9]\d* rows/);
}
