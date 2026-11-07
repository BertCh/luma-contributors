// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useContext, useEffect, type ComponentProps} from 'react';
import {ModalContainerFactory} from '@kepler.gl/components';
import {NativeWorkflow, NativeWorkflows} from './native-workflows';

export type ResidentWorkflow = 'data-export' | 'workspace-save';
export type ResidentWorkflowOptions = {
  isResidentDataset(datasetId: string): boolean;
};

/** Guard the owning container, including its footer actions, before native CPU export can run. */
export function createResidentModalContainerFactory(options: ResidentWorkflowOptions) {
  function ResidentModalContainerFactory(
    ...dependencies: Parameters<typeof ModalContainerFactory>
  ) {
    const NativeContainer = ModalContainerFactory(...dependencies);
    const ModalDialog = dependencies[9];
    return function ResidentModalContainer(props: ComponentProps<typeof NativeContainer>) {
      const workflows = useContext(NativeWorkflows);
      const resident = Object.keys(props.visState.datasets).some(options.isResidentDataset);
      const workflow = getResidentModalRoute(props.uiState.currentModal, resident);
      const isMemoryModal = props.uiState.currentModal === 'residentMemory';
      useEffect(() => {
        if (!workflow && !isMemoryModal) return;
        // Match ModalContainer's document-level Escape handling, including after a download
        // moves focus outside the dialog's own React event tree.
        const handleKeyUp = (event: KeyboardEvent) => {
          if (event.key === 'Escape') props.uiStateActions.toggleModal(null);
        };
        document.addEventListener('keyup', handleKeyUp);
        return () => document.removeEventListener('keyup', handleKeyUp);
      }, [workflow, isMemoryModal, props.uiStateActions]);
      const close = () => props.uiStateActions.toggleModal(null);
      if (props.uiState.currentModal === 'residentMemory')
        return (
          <ModalDialog
            isOpen
            close
            footer={false}
            theme={undefined}
            onRequestClose={close}
            onCancel={close}
            onConfirm={close}
            contentLabel="Memory manager"
            cssStyle="width: min(900px, 90vw); padding: 24px; max-height: 90vh; overflow: auto;"
          >
            <NativeWorkflow>{workflows.memoryManager}</NativeWorkflow>
          </ModalDialog>
        );
      if (!workflow) return <NativeContainer {...props} />;
      return (
        <ModalDialog
          isOpen
          close
          footer={false}
          theme={undefined}
          onRequestClose={close}
          onCancel={close}
          onConfirm={close}
          contentLabel={workflow === 'data-export' ? 'Export data' : 'Save map'}
          cssStyle="width: min(760px, 90vw); padding: 24px; max-height: 90vh; overflow: auto;"
        >
          <NativeWorkflow>
            <fieldset disabled={workflows.unavailable}>
              {workflow === 'data-export' ? workflows.exportData : workflows.saveWorkspace}
            </fieldset>
          </NativeWorkflow>
        </ModalDialog>
      );
    };
  }
  ResidentModalContainerFactory.deps = ModalContainerFactory.deps;
  return ResidentModalContainerFactory;
}

export function getResidentModalRoute(
  modal: unknown,
  resident: boolean
): ResidentWorkflow | undefined {
  if (!resident) return undefined;
  if (modal === 'exportData') return 'data-export';
  if (['exportMap', 'saveMap', 'shareMap', 'overwriteMap'].includes(String(modal)))
    return 'workspace-save';
  return undefined;
}
