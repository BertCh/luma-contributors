// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeWorkflowDataFromFeatures} from '../../src/workflows/workflow-data';

export function createCatalogIcpWorkflowInputs() {
  const makeCloud = (id: string, label: string) => {
    const cloud = makeWorkflowDataFromFeatures(
      {
        type: 'FeatureCollection',
        features: [
          [0, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]
        ].map(coordinates => ({geometry: {type: 'Point', coordinates}, properties: {}}))
      },
      {id, label}
    );
    cloud.table.coordinateSystem = 'planar';
    return cloud;
  };
  return {
    source: makeCloud('source-cloud', 'Source cloud'),
    target: makeCloud('target-cloud', 'Target cloud')
  };
}
