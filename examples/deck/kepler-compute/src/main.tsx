// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import './label-format-host';
import './native-utils-host';
import 'maplibre-gl/dist/maplibre-gl.css';
import './style.css';
import './path-layer-picking-patch';
import './scenegraph-area-lighting-patch';
import './app-services';
import './split-view';
import './publication';
import './layers';
import './native-hosts';
import './selection';
import './resident-operations';
import './app-store';
import './store-subscription';
import './native-config';
import './recovery';
import './device-init';
import './map-container';
import './app';
import './main-raster-live';
import './dev-test-hooks';
import {createRoot} from 'react-dom/client';
import {Provider} from 'react-redux';
import {appState} from './app-state';
import {installSqlFixture} from './sql-fixture-install';
import {AUTO_PREPARE_FLOAT64_POLICY} from './auto-prepare-plan';
import {store} from './app-store';
import {App} from './app';
import {effect} from './app-services';
import {publicationsInProgress, publishSource} from './publication';
import {getActiveRuntime} from './native-config';

const rootElement = document.getElementById('root');

if (!rootElement) throw new Error('Missing application root');

createRoot(rootElement).render(
  <Provider store={store}>
    <App />
  </Provider>
);

window.addEventListener('beforeunload', () => {
  effect.runtime = null;
  void appState.deviceRecovery.dispose().catch(() => {});
});

installSqlFixture(
  {
    isReady: () =>
      Boolean(appState.runtime) &&
      !appState.runtime?.session.deviceLost &&
      !appState.runtime?.isLayerRegistrationPending &&
      appState.deviceRecovery.status.phase === 'ready' &&
      publicationsInProgress.size === 0,
    publishTable: async ({datasetId, label, table}) => {
      const owner = getActiveRuntime();
      appState.requestedPublicationId = datasetId;
      await publishSource(
        {table, label, isCurrent: () => appState.runtime === owner},
        {float64: AUTO_PREPARE_FLOAT64_POLICY}
      );
    }
  },
  location.search
);
