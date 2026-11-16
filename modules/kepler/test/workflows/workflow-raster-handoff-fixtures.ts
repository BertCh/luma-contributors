// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {CatalogOperationBinding} from '../../src/analysis/catalog-binding-types';
import {CURATED_SURFACE_PREPARATION_BINDING} from '../../src/curated/bindings/curated-surface-preparation-binding';
import {CURATED_DRAINAGE_NETWORK_BINDING} from '../../src/curated/bindings/curated-drainage-network-binding';
import {createDrainageNetworkSample} from '../../src/curated/samples/drainage-network';
import {profileKeplerWorkbenchData} from '../../src/descriptors/workbench-adapters';
import {createWorkflowDocument} from '../../src/workflows/workflow-document';
import {addWorkflowStep} from '../../src/workflows/workflow-edit';
import {makeWorkflowRasterData} from '../../src/workflows/workflow-data';
import {
  createWorkflowOperationRegistry,
  makeWorkflowOperationFromCatalogBinding
} from '../../src/workflows/workflow-operations';
import {getWorkflowSourceOperations} from '../../src/workflows/workflow-source-operations';
import {makeFixtureHost} from './workflow-fixtures';

/** Actual adapters and automatic port selection, shared by dispatch and GPU checks. */
export function createRasterHandoffFixture(
  device: Device,
  surfaceBinding: CatalogOperationBinding = CURATED_SURFACE_PREPARATION_BINDING,
  drainageBinding: CatalogOperationBinding = CURATED_DRAINAGE_NETWORK_BINDING
) {
  const {elevation} = createDrainageNetworkSample();
  elevation.band.values[4] = -999;
  elevation.band.validity[4] = 0;
  const data = makeWorkflowRasterData(elevation, 9);
  const host = makeFixtureHost(
    [
      {
        id: elevation.id,
        label: 'Planar elevation',
        rowCount: 0,
        resident: false,
        fields: [],
        geometryFields: []
      }
    ],
    {[elevation.id]: data}
  );
  host.getDevice = () => device;
  host.getAnalysisProfile = () => profileKeplerWorkbenchData(data);
  const registry = createWorkflowOperationRegistry([
    ...getWorkflowSourceOperations(),
    makeWorkflowOperationFromCatalogBinding(surfaceBinding),
    makeWorkflowOperationFromCatalogBinding(drainageBinding)
  ]);
  const source = addWorkflowStep(
    createWorkflowDocument({title: 'Prepare elevation then trace drainage'}),
    registry,
    {
      id: 'source',
      operationId: 'source-dataset',
      parameters: {dataset: elevation.id, geometryMode: 'none'}
    }
  );
  const preparation = addWorkflowStep(source.document, registry, {
    id: 'prepare',
    operationId: surfaceBinding.id,
    parameters: {maximumDistance: 1, method: 'nearest'},
    connectFrom: {stepId: source.stepId, output: 'data'}
  });
  const drainage = addWorkflowStep(preparation.document, registry, {
    id: 'drainage',
    operationId: drainageBinding.id,
    parameters: {streamThreshold: 1},
    connectFrom: {stepId: preparation.stepId, output: 'prepared-surface'}
  });
  return {host, registry, document: drainage.document, elevation};
}
