// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createElement} from 'react';
import {createRoot} from 'react-dom/client';
import * as arrow from 'apache-arrow';
import {SourcePreviewPanel} from '../../src/source-preview-panel';

const source = arrow.tableFromArrays({amount: new Float64Array([0.1, 0.2]), region: ['north', 'south'], geometry: [{x: 1, y: 2}, {x: 3, y: 4}]});
window.sourceUploadCalls = [];
createRoot(document.querySelector('#root')).render(createElement(SourcePreviewPanel, {
  source, label: 'Imported observations', onPublish: async (table, normalization) => {
    window.sourceUploadCalls.push({rowCount: table.numRows, normalization});
  }
}));
