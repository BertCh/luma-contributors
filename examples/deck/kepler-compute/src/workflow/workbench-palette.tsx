// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useId, useMemo, useRef, useState, type KeyboardEvent} from 'react';
import styled from 'styled-components';
import {
  addWorkflowStep,
  searchWorkflowOperations,
  type WorkflowEngineSupport,
  type WorkflowOperation,
  type WorkflowOperationCategory,
  type WorkflowOperationMatch,
  type WorkflowOperationRegistry,
  type WorkflowPortKind
} from '@luma-contributors/kepler';
import {
  WORKBENCH_CATEGORY_LABELS,
  WORKBENCH_COLORS,
  WORKBENCH_PORT_LABELS,
  WORKBENCH_TEST_IDS
} from './workbench-contract';
import type {WorkbenchState, WorkbenchStore} from './workbench-store';
import {
  WorkbenchButton,
  WorkbenchField,
  WorkbenchMuted,
  WorkbenchPortSwatch,
  WorkbenchSection,
  WorkbenchSurface
} from './workbench-styles';
import {useWorkbenchState} from './use-workbench';

const MAXIMUM_VISIBLE_OPERATIONS = 60;
const MAXIMUM_RECENT_OPERATIONS = 6;
export type PalettePortKind = WorkflowPortKind | 'any' | 'any-geometry';
const PALETTE_PORT_LABELS: Record<PalettePortKind, string> = {
  ...WORKBENCH_PORT_LABELS,
  any: 'Any',
  'any-geometry': 'Any geometry'
};
const ENGINE_LABELS: Record<WorkflowEngineSupport, string> = {
  gpu: 'GPU',
  'gpu-or-cpu': 'GPU or CPU',
  'cpu-reference': 'CPU',
  host: 'Host'
};
// Session-only usage cache; never persisted into workflow documents.
let recentOperationIds: readonly string[] = [];

export function updatePaletteRecentOperations(
  operationIds: readonly string[],
  operationId: string
): string[] {
  return [...new Set([operationId, ...operationIds])].slice(0, MAXIMUM_RECENT_OPERATIONS);
}

export function groupPaletteOperations(
  matches: readonly WorkflowOperationMatch[],
  recentIds: readonly string[],
  query: string,
  category?: WorkflowOperationCategory
): {recent: WorkflowOperationMatch[]; operations: WorkflowOperationMatch[]} {
  if (query.trim() || category) return {recent: [], operations: [...matches]};
  const byId = new Map(matches.map(match => [match.operation.id, match]));
  const recent = [...new Set(recentIds)]
    .slice(0, MAXIMUM_RECENT_OPERATIONS)
    .flatMap(operationId => {
      const match = byId.get(operationId);
      return match ? [match] : [];
    });
  const recentIdsInResults = new Set(recent.map(match => match.operation.id));
  return {
    recent,
    operations: matches.filter(match => !recentIdsInResults.has(match.operation.id))
  };
}

export function getPaletteCategoryCounts(
  registry: WorkflowOperationRegistry,
  matches: readonly WorkflowOperationMatch[]
): Array<{category: 'all' | WorkflowOperationCategory; label: string; count: number}> {
  const categories = new Set(registry.list().map(operation => operation.category));
  const counts = new Map<WorkflowOperationCategory, number>();
  for (const {operation} of matches)
    counts.set(operation.category, (counts.get(operation.category) ?? 0) + 1);
  return [
    {category: 'all', label: 'All', count: matches.length},
    ...(Object.keys(WORKBENCH_CATEGORY_LABELS) as WorkflowOperationCategory[])
      .filter(category => categories.has(category) && (counts.get(category) ?? 0) > 0)
      .map(category => ({
        category,
        label: WORKBENCH_CATEGORY_LABELS[category],
        count: counts.get(category) ?? 0
      }))
  ];
}

export function getPalettePortLine(operation: WorkflowOperation): {
  inputs: PalettePortKind[];
  outputs: WorkflowPortKind[];
} {
  const inputs = operation.inputs.flatMap((input): PalettePortKind[] => {
    const kinds = new Set(input.accepts);
    if (kinds.has('point') && kinds.has('line') && kinds.has('polygon')) {
      if (kinds.has('features') && kinds.has('table')) return ['any'];
      if (!kinds.has('table')) return ['any-geometry'];
    }
    return [...kinds];
  });
  return {
    inputs: [...new Set(inputs)],
    outputs: [...new Set(operation.outputs.map(output => output.kind))]
  };
}

export function getPaletteConnection(
  state: WorkbenchState,
  registry: WorkflowOperationRegistry
): {producerLabel: string; outputLabel: string; kind: WorkflowPortKind | undefined} | null {
  const connection = state.palette.connectFrom;
  if (!connection) return null;
  const producer = state.document?.steps.find(step => step.id === connection.stepId);
  const operation = producer && registry.get(producer.operationId);
  const output = operation?.outputs.find(port => port.name === connection.output);
  return {
    producerLabel:
      producer?.label ?? operation?.title ?? producer?.operationId ?? connection.stepId,
    outputLabel: output?.label ?? connection.output,
    kind:
      state.runner?.steps[connection.stepId]?.result?.outputs[connection.output]?.kind ??
      output?.kind
  };
}

/** Returns undefined for keys that should retain their native text-field behaviour. */
export function movePaletteHighlight(
  key: string,
  index: number,
  count: number
): number | undefined {
  if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(key)) return undefined;
  if (count === 0) return -1;
  if (key === 'Home') return 0;
  if (key === 'End') return count - 1;
  if (key === 'ArrowDown') return (index + 1 + count) % count;
  return (index - 1 + count) % count;
}

/** One edit transaction captures the id returned by the edit, including collision suffixes. */
export function addPaletteOperation(store: WorkbenchStore, operationId: string): boolean {
  const {palette} = store.getState();
  let stepId: string | undefined;
  const added = store.edit('add-step', document => {
    const result = addWorkflowStep(document, store.registry, {
      operationId,
      position: palette.position,
      connectFrom: palette.connectFrom
    });
    stepId = result.stepId;
    return result.document;
  });
  if (!added || !stepId) return false;
  recentOperationIds = updatePaletteRecentOperations(recentOperationIds, operationId);
  store.select([stepId]);
  store.closePalette();
  store.setDock({inspectorTab: 'parameters'});
  return true;
}

const Palette = styled(WorkbenchSurface)`
  position: fixed;
  top: 12vh;
  left: calc(var(--workbench-dock-left, 0px) + (100vw - var(--workbench-dock-left, 0px)) / 2);
  transform: translateX(-50%);
  z-index: 60;
  display: flex;
  flex-direction: column;
  width: min(680px, calc(100vw - var(--workbench-dock-left, 0px) - 48px));
  max-height: 72vh;
  container-type: inline-size;
  border: 1px solid ${WORKBENCH_COLORS.borderStrong};
  border-radius: 4px;
  box-shadow: 0 8px 24px ${WORKBENCH_COLORS.surfaceSunken};
  overflow: hidden;
`;
const SearchSection = styled.div`
  flex: none;
  padding: 8px;
  border-bottom: 1px solid ${WORKBENCH_COLORS.border};
`;
const Categories = styled.div`
  display: flex;
  gap: 4px;
  overflow-x: auto;
`;
const KeyboardHint = styled(WorkbenchMuted)`
  display: block;
  margin: -3px 0 7px;
  font-size: 11px;
  white-space: nowrap;
`;
const Reading = styled.label`
  display: flex;
  align-items: center;
  gap: 6px;
  min-height: 24px;
  margin-bottom: 6px;
`;
const Results = styled.div`
  display: grid;
  grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
  grid-template-rows: minmax(0, 1fr);
  flex: 1 1 auto;
  min-height: 0;
  overflow: hidden;
  @container (max-width: 580px) {
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
  }
`;
const OperationList = styled.div`
  min-width: 0;
  min-height: 0;
  overflow-x: hidden;
  overflow-y: auto;
  & h3 {
    margin: 5px 8px;
    font-size: 12px;
    color: ${WORKBENCH_COLORS.textMuted};
  }
`;
const OperationRow = styled(WorkbenchButton)`
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 2px;
  width: 100%;
  height: auto;
  min-height: 52px;
  padding: 5px 8px;
  line-height: 1.2;
  border: 0;
  border-bottom: 1px solid ${WORKBENCH_COLORS.border};
  border-radius: 0;
  white-space: normal;
  text-align: left;
  &[aria-selected='true'] {
    background: ${WORKBENCH_COLORS.surfaceRaised};
    box-shadow: inset 2px 0 ${WORKBENCH_COLORS.accent};
  }
`;
const RowTitle = styled.span`
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  min-width: 0;
  gap: 6px;
`;
const OperationTitle = styled.span`
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;
const EngineTag = styled(WorkbenchMuted)`
  flex: none;
  white-space: nowrap;
`;
const RowMetadata = styled.span`
  display: flex;
  justify-content: space-between;
  gap: 6px;
  min-width: 0;
  font-size: 11px;
  white-space: nowrap;
`;
const PortLine = styled.span`
  display: block;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 11px;
  & > * + * { margin-left: 5px; }
`;
const Kind = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 3px;
`;
const Detail = styled(WorkbenchSection)`
  min-width: 0;
  min-height: 0;
  overflow-x: hidden;
  overflow-y: auto;
  overflow-wrap: anywhere;
  border-left: 1px solid ${WORKBENCH_COLORS.border};
  border-bottom: 0;
  & h4 { margin: 8px 0 3px; font-size: 12px; }
  & p { margin: 0 0 6px; }
  & ul { margin: 0; padding-left: 16px; }
`;
const EmptyMessage = styled.p`
  padding: 8px;
  margin: 0;
`;

function PortKinds({kinds}: {kinds: readonly PalettePortKind[]}): React.JSX.Element {
  return (
    <>
      {kinds.length === 0 ? <WorkbenchMuted>None</WorkbenchMuted> : null}
      {kinds.map(kind => (
        <Kind key={kind}>
          <WorkbenchPortSwatch
            $kind={kind === 'any' ? 'table' : kind === 'any-geometry' ? 'features' : kind}
            aria-hidden="true"
          />
          {PALETTE_PORT_LABELS[kind]}
        </Kind>
      ))}
    </>
  );
}

function OperationDetail({operation}: {operation: WorkflowOperation}): React.JSX.Element {
  return (
    <Detail aria-label={`${operation.title} details`}>
      <h3>{operation.title}</h3>
      <p>{operation.summary}</p>
      <h4>Inputs</h4>
      {operation.inputs.length === 0 && <p>None</p>}
      {operation.inputs.map(input => (
        <p key={input.name}>
          {input.label}:{' '}
          {input.accepts.map(kind => WORKBENCH_PORT_LABELS[kind].toLowerCase()).join(', ')}, up to{' '}
          {input.maximumRows.toLocaleString()} rows
          {input.maximumVertices !== undefined &&
            `, ${input.maximumVertices.toLocaleString()} vertices`}
          {input.optional && ' (optional)'}
        </p>
      ))}
      <h4>Outputs</h4>
      {operation.outputs.length === 0 && <p>None</p>}
      {operation.outputs.map(output => (
        <p key={output.name}>
          {output.label}: {WORKBENCH_PORT_LABELS[output.kind].toLowerCase()}, up to{' '}
          {output.maximumRows.toLocaleString()} rows
        </p>
      ))}
      {operation.declaredBehavior.length > 0 && (
        <>
          <h4>Behaviour</h4>
          <ul>
            {operation.declaredBehavior.slice(0, 2).map((behaviour, index) => (
              <li key={index}>{behaviour}</li>
            ))}
          </ul>
        </>
      )}
      <h4>Limitation</h4>
      <p>{operation.limitation}</p>
    </Detail>
  );
}

/** Mounted only while open, so transient search and filters reset for every invocation. */
export function WorkbenchPaletteDialog({
  store,
  initialQuery = ''
}: {
  store: WorkbenchStore;
  initialQuery?: string;
}): React.JSX.Element {
  const state = useWorkbenchState(store);
  const [query, setQuery] = useState(initialQuery);
  const [category, setCategory] = useState<WorkflowOperationCategory>();
  const [compatibleOnly, setCompatibleOnly] = useState(true);
  const [highlight, setHighlight] = useState(0);
  const search = useRef<HTMLInputElement>(null);
  const palette = useRef<HTMLDivElement>(null);
  const listId = useId();
  const connection = useMemo(
    () => getPaletteConnection(state, store.registry),
    [state.palette, state.document, state.runner, store.registry]
  );
  const compatibleWith = compatibleOnly ? connection?.kind : undefined;
  const allMatches = useMemo(
    () => searchWorkflowOperations(store.registry, query, {compatibleWith, limit: Infinity}),
    [store.registry, query, compatibleWith]
  );
  const categories = useMemo(
    () => getPaletteCategoryCounts(store.registry, allMatches),
    [store.registry, allMatches]
  );
  const matches = useMemo(
    () =>
      searchWorkflowOperations(store.registry, query, {category, compatibleWith, limit: Infinity}),
    [store.registry, query, category, compatibleWith]
  );
  const groups = useMemo(
    () => groupPaletteOperations(matches, recentOperationIds, query, category),
    [matches, query, category]
  );
  const rows = useMemo(
    () => [...groups.recent, ...groups.operations].slice(0, MAXIMUM_VISIBLE_OPERATIONS),
    [groups]
  );
  const selectedIndex = rows.length ? Math.max(0, Math.min(highlight, rows.length - 1)) : -1;
  const highlighted = rows[selectedIndex];
  const highlightedId = highlighted ? `${listId}-${highlighted.operation.id}` : undefined;

  useEffect(() => {
    const element = palette.current;
    if (!element) return;
    const ownerDocument = element.ownerDocument;
    const opener = ownerDocument.activeElement;
    search.current?.focus();
    function closeOutside(event: PointerEvent) {
      if (event.target instanceof Node && !element!.contains(event.target)) store.closePalette();
    }
    ownerDocument.addEventListener('pointerdown', closeOutside);
    return () => {
      ownerDocument.removeEventListener('pointerdown', closeOutside);
      const canvas = ownerDocument.querySelector<HTMLElement>(
        `[data-testid="${WORKBENCH_TEST_IDS.canvas}"]`
      );
      if (canvas) canvas.focus();
      else if (opener instanceof HTMLElement && opener.isConnected) opener.focus();
    };
  }, [store]);
  useEffect(() => {
    if (highlightedId)
      palette.current?.ownerDocument
        .getElementById(highlightedId)
        ?.scrollIntoView?.({block: 'nearest'});
  }, [highlightedId]);

  function handleSearchKey(event: KeyboardEvent<HTMLInputElement>) {
    if (event.nativeEvent.isComposing) return;
    const next = movePaletteHighlight(event.key, selectedIndex, rows.length);
    if (next !== undefined) {
      event.preventDefault();
      setHighlight(next);
    } else if (event.key === 'Enter') {
      event.preventDefault();
      if (highlighted) addPaletteOperation(store, highlighted.operation.id);
    }
  }

  return (
    <Palette
      ref={palette}
      role="dialog"
      aria-label="Add a step"
      data-testid={WORKBENCH_TEST_IDS.palette}
      onKeyDown={event => {
        event.stopPropagation();
        if (event.nativeEvent.isComposing) return;
        if (event.key === 'Escape') {
          event.preventDefault();
          store.closePalette();
        }
      }}
    >
      <SearchSection>
        <WorkbenchField>
          <label htmlFor={`${listId}-search`}>Search operations</label>
          <input
            ref={search}
            id={`${listId}-search`}
            data-testid={WORKBENCH_TEST_IDS.paletteSearch}
            role="combobox"
            aria-autocomplete="list"
            aria-expanded="true"
            aria-controls={listId}
            aria-activedescendant={highlightedId}
            aria-describedby={`${listId}-hint`}
            value={query}
            onChange={event => {
              setQuery(event.target.value);
              setHighlight(0);
            }}
            onKeyDown={handleSearchKey}
          />
        </WorkbenchField>
        <KeyboardHint id={`${listId}-hint`}>↑ ↓ to move · Enter to add · Esc to close</KeyboardHint>
        {connection && (
          <>
            <div>
              Reading {connection.producerLabel} · {connection.outputLabel} (
              {connection.kind ? WORKBENCH_PORT_LABELS[connection.kind] : 'Unknown kind'})
            </div>
            <Reading>
              <input
                type="checkbox"
                checked={compatibleOnly}
                onChange={event => {
                  setCompatibleOnly(event.target.checked);
                  setHighlight(0);
                }}
              />
              Compatible only
            </Reading>
          </>
        )}
        <Categories aria-label="Operation categories">
          {categories.map(chip => (
            <WorkbenchButton
              key={chip.category}
              data-testid={WORKBENCH_TEST_IDS.paletteCategory(chip.category)}
              aria-pressed={chip.category === (category ?? 'all')}
              onClick={() => {
                setCategory(chip.category === 'all' ? undefined : chip.category);
                setHighlight(0);
              }}
            >
              {chip.label} ({chip.count.toLocaleString()})
            </WorkbenchButton>
          ))}
        </Categories>
      </SearchSection>
      <Results>
        <OperationList id={listId} role="listbox" aria-label="Operations">
          {rows.map((match, index) => {
            const {operation} = match;
            const ports = getPalettePortLine(operation);
            return (
              <div role="presentation" key={operation.id}>
                {index === 0 && groups.recent.length > 0 && <h3 role="presentation">Recent</h3>}
                {index === groups.recent.length && groups.recent.length > 0 && (
                  <h3 role="presentation">Operations</h3>
                )}
                <OperationRow
                  type="button"
                  id={`${listId}-${operation.id}`}
                  role="option"
                  aria-selected={index === selectedIndex}
                  tabIndex={-1}
                  data-testid={WORKBENCH_TEST_IDS.paletteItem(operation.id)}
                  onMouseEnter={() => setHighlight(index)}
                  onClick={() => addPaletteOperation(store, operation.id)}
                >
                  <RowTitle>
                    <OperationTitle title={operation.title}>{operation.title}</OperationTitle>
                    <EngineTag>{ENGINE_LABELS[operation.engine]}</EngineTag>
                  </RowTitle>
                  <RowMetadata>
                    <WorkbenchMuted>{WORKBENCH_CATEGORY_LABELS[operation.category]}</WorkbenchMuted>
                    {match.compatibility === 'maybe' && (
                      <WorkbenchMuted>checked when run</WorkbenchMuted>
                    )}
                  </RowMetadata>
                  <PortLine
                    title={`${ports.inputs.map(kind => PALETTE_PORT_LABELS[kind]).join(', ') || 'None'} → ${ports.outputs.map(kind => WORKBENCH_PORT_LABELS[kind]).join(', ') || 'None'}`}
                  >
                    <PortKinds kinds={ports.inputs} />
                    <span aria-hidden="true">→</span>
                    <PortKinds kinds={ports.outputs} />
                  </PortLine>
                </OperationRow>
              </div>
            );
          })}
          {rows.length === 0 && <EmptyMessage>No operation matches "{query}".</EmptyMessage>}
          {matches.length > rows.length && (
            <EmptyMessage>
              {(matches.length - rows.length).toLocaleString()} more: refine the search
            </EmptyMessage>
          )}
        </OperationList>
        {highlighted && <OperationDetail operation={highlighted.operation} />}
      </Results>
    </Palette>
  );
}

/** Searchable add-step menu. */
export function WorkbenchPalette({store}: {store: WorkbenchStore}): React.JSX.Element | null {
  const state = useWorkbenchState(store);
  return state.palette.open ? <WorkbenchPaletteDialog store={store} /> : null;
}
