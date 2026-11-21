// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createContext, memo, useContext, useMemo, type KeyboardEvent} from 'react';
import {Handle, Position, type NodeProps} from '@xyflow/react';
import styled from 'styled-components';
import {
  updateWorkflowStep,
  getWorkflowOutputKind,
  type WorkflowInputPortSpec,
  type WorkflowOutputPortSpec
} from '@luma-contributors/kepler';
import {useWorkbenchState, useWorkbenchStore} from './use-workbench';
import {
  WORKBENCH_CATEGORY_LABELS,
  WORKBENCH_COLORS,
  WORKBENCH_DOCK,
  WORKBENCH_PORT_COLORS,
  WORKBENCH_PORT_LABELS,
  WORKBENCH_STATUS,
  WORKBENCH_TEST_IDS
} from './workbench-contract';
import {
  WorkbenchIconButton,
  WorkbenchMuted,
  WorkbenchStatusBadge,
  WorkbenchSurface
} from './workbench-styles';
import {
  getWorkbenchStepStatus,
  getWorkbenchStepSummary,
  makeWorkbenchHandleId
} from './workbench-canvas-model';

/** Data the canvas hands each React Flow node; the node reads everything else from the store. */
export type WorkbenchNodeData = {stepId: string};

export type WorkbenchPortEndpoint = {stepId: string; direction: 'in' | 'out'; port: string};
export type WorkbenchKeyboardConnection = {
  pending: WorkbenchPortEndpoint | null;
  activate(endpoint: WorkbenchPortEndpoint): void;
};
/** Transient keyboard gesture shared by the nodes of one canvas. */
export const WorkbenchConnectionContext = createContext<WorkbenchKeyboardConnection | null>(null);

const NodeSurface = styled(WorkbenchSurface)<{$selected: boolean; $disabled: boolean}>`
  width: ${WORKBENCH_DOCK.nodeWidth}px;
  min-width: 0;
  border: 1px ${({$disabled}) => ($disabled ? 'dashed' : 'solid')}
    ${({$disabled}) => ($disabled ? WORKBENCH_COLORS.border : WORKBENCH_COLORS.borderStrong)};
  border-radius: 4px;
  outline: ${({$selected}) => ($selected ? `2px solid ${WORKBENCH_COLORS.accent}` : 'none')};
`;
const Header = styled.div<{$disabled: boolean}>`
  display: flex;
  gap: 5px;
  align-items: center;
  padding: 7px 8px;
  border-bottom: 1px solid ${WORKBENCH_COLORS.border};
  & > div { flex: 1; min-width: 0; }
  & > button { flex: 0 0 26px; }
  & strong {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: ${({$disabled}) => ($disabled ? WORKBENCH_COLORS.textMuted : WORKBENCH_COLORS.text)};
  }
  & small { font-size: 10px; }
`;
const ToggleButton = styled(WorkbenchIconButton)<{$purpose: 'display' | 'disable'}>`
  && { position: relative; color: ${WORKBENCH_COLORS.textMuted}; }
  &&[aria-pressed='true'] {
    color: ${({$purpose}) => ($purpose === 'display' ? WORKBENCH_COLORS.accent : WORKBENCH_COLORS.warning)};
    border-color: ${({$purpose}) => ($purpose === 'display' ? WORKBENCH_COLORS.accent : WORKBENCH_COLORS.warning)};
  }
`;
const PaletteDots = styled.span`
  position: absolute;
  right: -2px;
  bottom: -2px;
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 1px;
  width: max-content;
  max-width: 26px;
`;
const PaletteDot = styled.span`
  box-sizing: border-box;
  flex: 0 0 8px;
  width: 8px;
  height: 8px;
  border: 1px solid ${WORKBENCH_COLORS.surface};
  border-radius: 50%;
  background: ${WORKBENCH_COLORS.accent};
`;
const Ports = styled.div`
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 5px;
  padding: 5px 0;
  min-width: 0;
  & > div { min-width: 0; }
`;
const Port = styled.div<{$output?: boolean}>`
  position: relative;
  min-width: 0;
  min-height: 26px;
  padding: 3px 10px;
  text-align: ${({$output}) => ($output ? 'right' : 'left')};
  & small {
    display: block;
    font-size: 10px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
`;
const PortLabel = styled.span`
  display: block;
  min-width: 0;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;
const PortHandle = styled(Handle)<{$color: string}>`
  && {
    width: 12px;
    height: 12px;
    background: ${({$color}) => $color};
    border-color: ${WORKBENCH_COLORS.surface};
  }
  &::after { content: ''; position: absolute; inset: -6px; }
  &:focus-visible { outline: 2px solid ${WORKBENCH_COLORS.focus}; outline-offset: 2px; }
`;
const Footer = styled.div`
  min-width: 0;
  max-width: 100%;
  padding: 5px 8px;
  border-top: 1px solid ${WORKBENCH_COLORS.border};
  color: ${WORKBENCH_COLORS.textMuted};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 11px;
`;

/** One step on the canvas: title, typed ports, status badge, display and disable toggles, summary. */
export const WorkbenchNode = memo(function WorkbenchNode(
  props: NodeProps
): React.JSX.Element | null {
  const store = useWorkbenchStore();
  const state = useWorkbenchState(store);
  const keyboardConnection = useContext(WorkbenchConnectionContext);
  const stepId = typeof props.data['stepId'] === 'string' ? props.data['stepId'] : props.id;
  const step = useMemo(
    () => state.document?.steps.find(candidate => candidate.id === stepId),
    [state.document, stepId]
  );
  const operation = step ? store.registry.get(step.operationId) : undefined;
  const ports = useMemo(() => {
    const inputs: WorkflowInputPortSpec[] = [...(operation?.inputs ?? [])];
    const outputs: WorkflowOutputPortSpec[] = [...(operation?.outputs ?? [])];
    if (!step) return {inputs, outputs};
    // Saved connections survive catalog changes, including ports no longer declared.
    for (const name of Object.keys(step.inputs)) {
      if (!inputs.some(port => port.name === name))
        inputs.push({
          name,
          label: `${name} (unknown input)`,
          accepts: ['table'],
          maximumRows: 0,
          optional: true
        });
    }
    for (const target of state.document?.steps ?? []) {
      for (const connection of Object.values(target.inputs)) {
        if (connection.stepId === step.id && !outputs.some(port => port.name === connection.output))
          outputs.push({
            name: connection.output,
            label: `${connection.output} (unknown output)`,
            kind: 'table',
            maximumRows: 0,
            identityColumns: []
          });
      }
    }
    return {inputs, outputs};
  }, [operation, step, state.document]);
  if (!step) return null;
  const stepState = state.runner?.steps[step.id];
  const status = getWorkbenchStepStatus(step, operation, stepState);
  const title = operation ? (step.label ?? operation.title) : step.operationId;
  const output = operation?.outputs[0];
  const displayed = Boolean(output && step.display?.includes(output.name));
  const coloredOutputs = (operation?.outputs ?? []).flatMap(port => {
    const colorColumn = step.displayStyle?.[port.name]?.colorColumn;
    return step.display?.includes(port.name) && colorColumn !== undefined
      ? [{output: port.name, colorColumn}]
      : [];
  });
  const summary = getWorkbenchStepSummary(step, operation, stepState);
  function activatePort(
    event: KeyboardEvent<HTMLDivElement>,
    direction: 'in' | 'out',
    port: string
  ) {
    if (
      (event.key !== 'Enter' && event.key !== ' ') ||
      event.ctrlKey ||
      event.metaKey ||
      event.altKey ||
      event.shiftKey
    )
      return;
    event.preventDefault();
    event.stopPropagation();
    keyboardConnection?.activate({stepId, direction, port});
  }
  function isPendingPort(direction: 'in' | 'out', port: string): boolean {
    const pending = keyboardConnection?.pending;
    return pending?.stepId === stepId && pending.direction === direction && pending.port === port;
  }
  return (
    <NodeSurface
      data-testid={WORKBENCH_TEST_IDS.node(step.id)}
      data-status={status}
      $selected={Boolean(props.selected)}
      $disabled={Boolean(step.disabled)}
    >
      <Header $disabled={Boolean(step.disabled)}>
        <div>
          <WorkbenchMuted as="small">
            {operation ? WORKBENCH_CATEGORY_LABELS[operation.category] : 'Unknown operation'}
          </WorkbenchMuted>
          <strong title={title}>{title}</strong>
          <WorkbenchStatusBadge
            $status={status}
            data-status={status}
            data-testid={WORKBENCH_TEST_IDS.nodeStatus(step.id)}
          >
            {WORKBENCH_STATUS[status].label}
          </WorkbenchStatusBadge>
        </div>
        {output && (
          <ToggleButton
            $purpose="display"
            className="nodrag nopan"
            data-testid={WORKBENCH_TEST_IDS.nodeDisplay(step.id)}
            aria-label={`Show ${title} on the map`}
            title="Show on the map (D)"
            aria-pressed={displayed}
            onClick={() => {
              if (!output) return;
              store.edit(`display:${step.id}`, document =>
                updateWorkflowStep(document, step.id, {
                  display: displayed
                    ? step.display?.filter(name => name !== output.name)
                    : [...(step.display ?? []), output.name]
                })
              );
            }}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 20 20"
              aria-hidden="true"
              focusable="false"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            >
              <path
                d="M2 10c2-4 5-6 8-6s6 2 8 6c-2 4-5 6-8 6s-6-2-8-6Z"
                fill={displayed ? 'currentColor' : 'none'}
              />
              <circle
                cx="10"
                cy="10"
                r="2.5"
                fill={displayed ? WORKBENCH_COLORS.surface : 'none'}
                stroke={displayed ? WORKBENCH_COLORS.surface : 'currentColor'}
              />
            </svg>
            {coloredOutputs.length > 0 && (
              <PaletteDots>
                {coloredOutputs.map(({output, colorColumn}) => (
                  <PaletteDot
                    key={output}
                    title={`Coloured by ${colorColumn}`}
                    role="img"
                    aria-label={`Coloured by ${colorColumn}`}
                  />
                ))}
              </PaletteDots>
            )}
          </ToggleButton>
        )}
        <ToggleButton
          $purpose="disable"
          className="nodrag nopan"
          data-testid={WORKBENCH_TEST_IDS.nodeDisable(step.id)}
          aria-label={`Disable ${title}`}
          title="Disable step (B)"
          aria-pressed={Boolean(step.disabled)}
          onClick={() =>
            store.edit(`disabled:${step.id}`, document =>
              updateWorkflowStep(document, step.id, {disabled: !step.disabled})
            )
          }
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 20 20"
            aria-hidden="true"
            focusable="false"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          >
            <path d="M10 2v7M5.5 4.5a7 7 0 1 0 9 0" />
          </svg>
        </ToggleButton>
      </Header>
      <Ports>
        <div>
          {ports.inputs.map(port => (
            <Port key={port.name}>
              <PortHandle
                id={makeWorkbenchHandleId('in', port.name)}
                type="target"
                position={Position.Left}
                $color={WORKBENCH_PORT_COLORS[port.accepts[0] ?? 'table']}
                data-testid={WORKBENCH_TEST_IDS.inputPort(step.id, port.name)}
                aria-label={`${title}: ${port.label} input`}
                tabIndex={0}
                role="button"
                title={`Connect ${port.label}`}
                aria-pressed={isPendingPort('in', port.name)}
                onKeyDown={event => activatePort(event, 'in', port.name)}
              />
              <PortLabel title={port.label}>{port.label}</PortLabel>
              {!port.optional && !Object.hasOwn(step.inputs, port.name) && (
                <WorkbenchMuted as="small">required</WorkbenchMuted>
              )}
            </Port>
          ))}
        </div>
        <div>
          {ports.outputs.map(port => {
            const kind =
              stepState?.result?.outputs[port.name]?.kind ??
              (state.document
                ? getWorkflowOutputKind(state.document, store.registry, step.id, port.name, {
                    host: store.host
                  })
                : port.kind);
            return (
              <Port key={port.name} $output>
                <PortHandle
                  id={makeWorkbenchHandleId('out', port.name)}
                  type="source"
                  position={Position.Right}
                  $color={WORKBENCH_PORT_COLORS[kind]}
                  data-testid={WORKBENCH_TEST_IDS.outputPort(step.id, port.name)}
                  aria-label={`${title}: ${port.label} output, ${WORKBENCH_PORT_LABELS[kind]}`}
                  tabIndex={0}
                  role="button"
                  title={`Connect ${port.label}`}
                  aria-pressed={isPendingPort('out', port.name)}
                  onKeyDown={event => activatePort(event, 'out', port.name)}
                />
                <PortLabel title={port.label}>{port.label}</PortLabel>
                <WorkbenchMuted as="small" title={WORKBENCH_PORT_LABELS[kind]}>
                  {WORKBENCH_PORT_LABELS[kind]}
                </WorkbenchMuted>
              </Port>
            );
          })}
        </div>
      </Ports>
      <Footer title={summary} aria-live="polite">
        {summary}
      </Footer>
    </NodeSurface>
  );
});
