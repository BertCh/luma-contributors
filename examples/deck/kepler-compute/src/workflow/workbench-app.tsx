// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useSyncExternalStore} from 'react';
import {analysisDatasetContext} from '../analysis-dataset-context';
import {rasterLiveHost} from '../main-raster-live';
import {WorkbenchDialogs} from './workbench-dialogs';
import {WorkbenchDock} from './workbench-dock';
import {bindWorkbenchExtension} from './workbench-extension';
import {createWorkbenchHost} from './workbench-host';
import {WorkbenchSidePanel} from './workbench-side-panel';
import {createWorkbenchStore, type WorkbenchStore} from './workbench-store';
import {connectWorkbenchAnalysisSampleSources} from './workbench-analysis-samples';

let store: WorkbenchStore | null = null;

/** The one workbench of the app, created on first use. */
export function getWorkbenchStore(): WorkbenchStore {
  if (!store) {
    store = createWorkbenchStore({
      host: createWorkbenchHost({
        getRasterSources: () => {
          const model = rasterLiveHost.getPanelModel();
          return [...model.rasterLayers, ...model.bitmapLayers].flatMap(layer => {
            const source =
              rasterLiveHost.raster.getSource(layer.id) ??
              rasterLiveHost.bitmap.getSource(layer.id);
            return source ? [{source, label: layer.label}] : [];
          });
        }
      })
    });
    connectWorkbenchAnalysisSampleSources(store);
    bindWorkbenchExtension(store);
    // Acceptance scripts drive the workbench through the store.
    if (import.meta.env.DEV) Object.assign(window, {__keplerWorkflowTest: {store}});
  }
  return store;
}

/** Horizontal space Kepler's side panel takes when open and when collapsed, in CSS pixels. */
const SIDE_PANEL_INSET = {open: 340, collapsed: 36} as const;

/**
 * The dock and its dialogs. The open dock spans the window below Kepler; the host reads its height
 * with `useWorkbenchDockInset` and shortens Kepler by it, so Kepler's bottom widget, map controls
 * and side panel are never covered. The collapsed pill reads the map's left edge from the
 * `--workbench-dock-left` custom property set here.
 * `externalRevision` changes whenever the host's datasets, filters or selection change, so source
 * steps are re-keyed.
 */
export function WorkbenchMount({
  externalRevision,
  sidePanelOpen
}: {
  externalRevision: number;
  sidePanelOpen: boolean;
}): React.JSX.Element {
  const workbench = getWorkbenchStore();
  const profileVersion = useSyncExternalStore(
    analysisDatasetContext.subscribe,
    analysisDatasetContext.getVersion,
    analysisDatasetContext.getVersion
  );
  useEffect(() => workbench.notifyExternalChange(), [workbench, externalRevision, profileVersion]);
  useEffect(() => {
    document.documentElement.style.setProperty(
      '--workbench-dock-left',
      `${sidePanelOpen ? SIDE_PANEL_INSET.open : SIDE_PANEL_INSET.collapsed}px`
    );
    return () => {
      document.documentElement.style.removeProperty('--workbench-dock-left');
    };
  }, [sidePanelOpen]);
  return (
    <>
      <WorkbenchDock store={workbench} />
      <WorkbenchDialogs store={workbench} />
    </>
  );
}

function WorkflowsIcon({height = '20px'}: {height?: string}): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 20 20"
      width={height}
      height={height}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      aria-hidden="true"
    >
      <rect x="1.5" y="3" width="6" height="4.5" rx="1" />
      <rect x="12.5" y="3" width="6" height="4.5" rx="1" />
      <rect x="7" y="12.5" width="6" height="4.5" rx="1" />
      <path d="M4.5 7.5v2.5a2 2 0 0 0 2 2H10M15.5 7.5v2.5a2 2 0 0 1-2 2H10v.5" />
    </svg>
  );
}

function WorkbenchSidePanelContainer(): React.JSX.Element {
  return <WorkbenchSidePanel store={getWorkbenchStore()} />;
}

/** Replaces Kepler's `CustomPanelsFactory`: one Workflows tab in the side panel. */
export function WorkbenchPanelsFactory() {
  // The side panel renders `panels[].component` for the active tab; this element stays empty.
  const WorkbenchPanels = (_props: {activeSidePanel: string | null}) => null;
  WorkbenchPanels.panels = [
    {
      id: 'workflows',
      label: 'Workflows',
      iconComponent: WorkflowsIcon,
      component: WorkbenchSidePanelContainer
    }
  ];
  WorkbenchPanels.getProps = () => ({});
  return WorkbenchPanels;
}

/** Height the open dock takes from the bottom of the window; zero while it is collapsed. */
export function useWorkbenchDockInset(): number {
  const workbench = getWorkbenchStore();
  const dock = useSyncExternalStore(
    workbench.subscribe,
    () => workbench.getState().dock,
    () => workbench.getState().dock
  );
  return dock.open ? dock.height : 0;
}
