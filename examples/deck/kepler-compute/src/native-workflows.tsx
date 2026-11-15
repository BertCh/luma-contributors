// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createContext, useContext, type ComponentProps, type ReactNode} from 'react';
import {LayerManagerFactory, LoadDataModalFactory, PanelHeaderFactory} from '@kepler.gl/components';
import styled from 'styled-components';

export const NATIVE_WORKFLOW_MESSAGES = {
  en: {
    'nativeWorkflows.prepareData': 'Prepare data',
    'nativeWorkflows.sampleMaps': 'Sample maps',
    'nativeWorkflows.queryData': 'SQL'
  }
};

/** Workflows are supplied by the host and rendered by their owning Kepler factories. */
export const NativeWorkflows = createContext<{
  analysis?: ReactNode;
  sources?: ReactNode;
  samples?: ReactNode;
  sql?: ReactNode;
  status?: ReactNode;
  /** Compact memory summary for the side panel and the full manager for its modal. */
  memory?: ReactNode;
  memoryManager?: ReactNode;
  help?: ReactNode;
  exportData?: ReactNode;
  saveWorkspace?: ReactNode;
  unavailable?: boolean;
}>({});

/** Inherit the native theme in both the dark sidebar and light modal surfaces. */
export const NativeWorkflow = styled.div.attrs({className: 'native-workflow'})`
  color: ${({theme}) => ('textColor' in theme ? String(theme.textColor) : '#dfe3e8')};
  .ReactModal__Content & { color: ${({theme}) => ('textColorLT' in theme ? String(theme.textColorLT) : '#3a414c')}; }
  font-size: 12px;
  line-height: 1.5;
  min-width: 0;
  padding: 12px;
  overflow-wrap: anywhere;
  & h2 { font-size: 15px; margin: 8px 0; }
  & label { display: block; margin: 12px 0 6px; }
  & select, & textarea, & input:not([type=checkbox]):not([type=radio]):not([type=range]) {
    box-sizing: border-box; max-width: 100%; width: 100%; padding: 6px;
    color: inherit; background: transparent; border: 1px solid currentColor; border-radius: 2px;
    font: inherit;
  }
  & option { color: #29323c; background: white; }
  & button { font: inherit; cursor: pointer; padding: 6px 10px; color: inherit;
    background: transparent; border: 1px solid currentColor; border-radius: 2px; }
  & button:disabled { opacity: 0.5; cursor: default; }
  & :focus-visible { outline: 2px solid #0097e6; outline-offset: 2px; }
  & fieldset { min-width: 0; border: 0; padding: 0; margin: 0; }
  & details { margin: 10px 0; }
  & summary { cursor: pointer; font-weight: 600; }
  & .hint { color: inherit; }
  & .actions { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0; }
  /* 4.5:1 or better on the dark sidebar (#ff9b9b) and the light modal (#b42318). */
  & .error { color: #ff9b9b; white-space: pre-wrap; }
  .ReactModal__Content & .error { color: #b42318; }
  @media (forced-colors: active) { & :focus-visible { outline-color: Highlight; } & button { border-color: ButtonText; } }
`;

export function NativeLayerManagerFactory(...dependencies: Parameters<typeof LayerManagerFactory>) {
  const LayerManager = LayerManagerFactory(...dependencies);
  return function NativeLayerManager(props: ComponentProps<typeof LayerManager>) {
    const workflows = useContext(NativeWorkflows);
    return (
      <>
        <LayerManager {...props} />
        <NativeWorkflow>
          <fieldset disabled={workflows.unavailable}>
            <details open data-testid="native-analysis">
              <summary>Analysis</summary>
              {workflows.analysis}
            </details>
          </fieldset>
          <details data-testid="native-memory">
            <summary>Memory</summary>
            {workflows.memory}
          </details>
          {workflows.help}
        </NativeWorkflow>
      </>
    );
  };
}
NativeLayerManagerFactory.deps = LayerManagerFactory.deps;

function PrepareData() {
  const workflows = useContext(NativeWorkflows);
  return (
    <NativeWorkflow>
      <fieldset disabled={workflows.unavailable}>{workflows.sources}</fieldset>
    </NativeWorkflow>
  );
}

function QueryData() {
  const workflows = useContext(NativeWorkflows);
  return (
    <NativeWorkflow>
      <fieldset disabled={workflows.unavailable}>
        {workflows.sql}
        {workflows.sources}
      </fieldset>
    </NativeWorkflow>
  );
}

function SampleMaps() {
  const workflows = useContext(NativeWorkflows);
  return (
    <NativeWorkflow>
      <fieldset disabled={workflows.unavailable}>{workflows.samples}</fieldset>
    </NativeWorkflow>
  );
}

export function NativeLoadDataModalFactory(
  ...dependencies: Parameters<typeof LoadDataModalFactory>
) {
  const LoadDataModal = LoadDataModalFactory(...dependencies);
  function NativeLoadDataModal(props: ComponentProps<typeof LoadDataModal>) {
    // Replacement is a separate native transaction; never publish a new source in its place.
    const loadingMethods = props.loadingMethods ?? LoadDataModal.defaultLoadingMethods ?? [];
    return (
      <LoadDataModal
        {...props}
        loadingMethods={
          props.replaceDatasetId
            ? loadingMethods
            : [
                ...loadingMethods,
                {id: 'sample-maps', label: 'nativeWorkflows.sampleMaps', elementType: SampleMaps},
                {
                  id: 'prepare-data',
                  label: 'nativeWorkflows.prepareData',
                  elementType: PrepareData
                },
                {id: 'query-data', label: 'nativeWorkflows.queryData', elementType: QueryData}
              ]
        }
      />
    );
  }
  NativeLoadDataModal.defaultLoadingMethods = LoadDataModal.defaultLoadingMethods;
  return NativeLoadDataModal;
}
NativeLoadDataModalFactory.deps = LoadDataModalFactory.deps;

// Status grows after dataset operations (policy controls and result notices). Keep it scrollable
// so the native dataset-operation panel retains a usable body instead of collapsing to zero height.
const NativeWorkflowStatus = styled(NativeWorkflow)`
  flex: 0 1 auto;
  min-height: 0;
  max-height: 40vh;
  overflow-y: auto;
`;

export function NativePanelHeaderFactory(...dependencies: Parameters<typeof PanelHeaderFactory>) {
  const PanelHeader = PanelHeaderFactory(...dependencies);
  return function NativePanelHeader(props: ComponentProps<typeof PanelHeader>) {
    const workflows = useContext(NativeWorkflows);
    return (
      <>
        <PanelHeader {...props} />
        <NativeWorkflowStatus>{workflows.status}</NativeWorkflowStatus>
      </>
    );
  };
}
NativePanelHeaderFactory.deps = PanelHeaderFactory.deps;
