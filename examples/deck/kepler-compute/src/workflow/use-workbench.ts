// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createContext, useContext, useSyncExternalStore} from 'react';
import type {WorkbenchState, WorkbenchStore} from './workbench-store';

/** Subscribes a component to the workbench snapshot. */
export function useWorkbenchState(store: WorkbenchStore): WorkbenchState {
  return useSyncExternalStore(store.subscribe, store.getState, store.getState);
}

/** Lets canvas nodes (rendered by React Flow) reach the store. */
export const WorkbenchStoreContext = createContext<WorkbenchStore | null>(null);

export function useWorkbenchStore(): WorkbenchStore {
  const store = useContext(WorkbenchStoreContext);
  if (!store) throw new Error('WorkbenchStoreContext is not provided');
  return store;
}
