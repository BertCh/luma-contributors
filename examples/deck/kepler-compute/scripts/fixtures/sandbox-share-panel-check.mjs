// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createElement, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {SandboxSharePanel} from '../../src/sandbox-share-panel';
const state = {schemaVersion: 1, sources: [{dataset: {datasetId: 'resident-observations', revision: 1}, reference: {kind: 'fixture', id: 'luma-kepler-observations', version: 1, parameters: {seed: 42, rowCount: 120}}, schema: [], normalization: {}}], nativeConfig: {config: {visState: {filters: []}}}, selections: [{source: {datasetId: 'resident-observations', revision: 1}, revision: 1, value: {kind: 'native-kepler-config', filters: []}}], analyses: []};
window.sharedStateLoads = [];
function App() {
  const [busy, setBusy] = useState(false);
  window.setShareBusy = setBusy;
  return createElement(SandboxSharePanel, {disabled: busy, saveState: () => JSON.stringify(state), loadState: serialized => window.sharedStateLoads.push(JSON.parse(serialized))});
}
createRoot(document.querySelector('#root')).render(createElement(App));
