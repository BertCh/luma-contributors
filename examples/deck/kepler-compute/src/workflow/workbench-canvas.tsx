// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  applyNodeChanges,
  useReactFlow,
  useNodesInitialized,
  type Connection,
  type NodeTypes
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import styled from 'styled-components';
import {
  checkWorkflowConnection,
  connectWorkflowSteps,
  disconnectWorkflowInput,
  moveWorkflowSteps,
  removeWorkflowSteps
} from '@luma-contributors/kepler';
import type {WorkbenchStore} from './workbench-store';
import {WorkbenchStoreContext, useWorkbenchState} from './use-workbench';
import {
  WORKBENCH_COLORS,
  WORKBENCH_PORT_COLORS,
  WORKBENCH_STATUS,
  WORKBENCH_TEST_IDS
} from './workbench-contract';
import {WorkbenchButton, WorkbenchMuted, WorkbenchSurface} from './workbench-styles';
import {
  WorkbenchNode,
  WorkbenchConnectionContext,
  type WorkbenchPortEndpoint
} from './workbench-node';
import {
  makeWorkbenchEdges,
  makeWorkbenchNodes,
  getWorkbenchStepStatus,
  parseWorkbenchHandleId,
  reconcileWorkbenchNodes,
  applyWorkbenchSelectionChanges,
  shouldShowWorkbenchMinimap,
  getWorkbenchFitViewOptions,
  getWorkbenchNodeBounds,
  getWorkbenchRevealViewport,
  getWorkbenchGroupRevealViewport,
  WORKBENCH_CANVAS_LAYOUT,
  type WorkbenchFlowNode,
  type WorkbenchFlowEdge
} from './workbench-canvas-model';

/** What the dock needs from the canvas for its keyboard shortcuts and the palette. */
export type WorkbenchCanvasApi = {
  /** Fits the given steps, or the whole graph, in view. */
  frame(stepIds?: readonly string[]): void;
  /** Centre of the visible canvas in canvas coordinates (where a new step goes by default). */
  getViewportCenter(): {x: number; y: number};
};

type CanvasProps = {store: WorkbenchStore; onApi?(api: WorkbenchCanvasApi | null): void};
const NODE_TYPES: NodeTypes = {step: WorkbenchNode};
const DEFAULT_VIEWPORT = {x: 0, y: 0, zoom: 1} as const;
const CanvasSurface = styled(WorkbenchSurface)`
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 100px;
  --xy-background-color: ${WORKBENCH_COLORS.surface};
  --xy-background-color-default: ${WORKBENCH_COLORS.surface};
  & .react-flow__controls { box-shadow: none; border: 1px solid ${WORKBENCH_COLORS.border}; }
  & .react-flow__controls-button {
    min-width: 26px;
    min-height: 26px;
    background: ${WORKBENCH_COLORS.surfaceRaised};
    color: ${WORKBENCH_COLORS.text};
    border-color: ${WORKBENCH_COLORS.border};
  }
  & .react-flow__controls-button svg { fill: currentColor; }
  & .react-flow__controls-button:focus-visible { outline: 2px solid ${WORKBENCH_COLORS.focus}; }
  &:focus-visible { outline: 2px solid ${WORKBENCH_COLORS.focus}; outline-offset: -2px; }
  && .react-flow__node:focus-visible {
    outline: 2px solid ${WORKBENCH_COLORS.focus};
    outline-offset: 3px;
  }
  @media (prefers-reduced-motion: reduce) {
    & .react-flow__edge-path { animation: none !important; }
  }
`;
const EmptyState = styled.div`
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  pointer-events: none;
  & > div { display: flex; gap: 6px; }
  & button { pointer-events: auto; }
`;

function readConnection(connection: Connection | WorkbenchFlowEdge) {
  const source = parseWorkbenchHandleId(connection.sourceHandle ?? '');
  const target = parseWorkbenchHandleId(connection.targetHandle ?? '');
  if (source?.direction !== 'out' || target?.direction !== 'in') return null;
  return {
    from: {stepId: connection.source, output: source.port},
    to: {stepId: connection.target, input: target.port}
  };
}

export function WorkbenchCanvasContent({store, onApi}: CanvasProps): React.JSX.Element {
  const state = useWorkbenchState(store);
  const document = state.document;
  const flow = useReactFlow<WorkbenchFlowNode, WorkbenchFlowEdge>();
  const nodesInitialized = useNodesInitialized();
  const container = useRef<HTMLDivElement>(null);
  const flowContainer = useRef<HTMLDivElement>(null);
  const reducedMotion = useRef(false);
  const previousStepIds = useRef(new Set(document?.steps.map(step => step.id)));
  const revealSteps = useRef<{ids: readonly string[]; usedFallback: boolean} | null>(null);
  const initialNodes = useMemo(
    () => (document ? makeWorkbenchNodes(document, state.selectedStepIds) : []),
    [document, state.selectedStepIds]
  );
  const [localNodes, setLocalNodes] = useState(initialNodes);
  const [selectedEdgeIds, setSelectedEdgeIds] = useState<readonly string[]>([]);
  const [pendingPort, setPendingPort] = useState<WorkbenchPortEndpoint | null>(null);
  const [canvasSize, setCanvasSize] = useState({width: 0, height: 0});
  const showMinimap = shouldShowWorkbenchMinimap(canvasSize.height, document?.steps.length ?? 0);
  const graphBounds = useMemo(() => getWorkbenchNodeBounds(localNodes), [localNodes]);
  const fitViewOptions = useMemo(
    () => getWorkbenchFitViewOptions(showMinimap, canvasSize, graphBounds),
    [showMinimap, canvasSize, graphBounds?.width, graphBounds?.height]
  );
  useEffect(() => {
    setLocalNodes(current => reconcileWorkbenchNodes(current, initialNodes));
  }, [initialNodes]);
  useEffect(() => {
    const element = flowContainer.current;
    if (!element) return;
    const measure = () => {
      const {width, height} = element.getBoundingClientRect();
      setCanvasSize(previous =>
        previous.width === width && previous.height === height ? previous : {width, height}
      );
    };
    measure();
    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(measure);
      observer.observe(element);
      return () => observer.disconnect();
    }
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, [document?.id]);
  const commitPositions = useCallback(
    (positions: Record<string, {x: number; y: number}>) => {
      store.edit('move', current => moveWorkflowSteps(current, positions));
      // Normalisation can succeed without changing document identity.
      const current = store.getState();
      const desired = current.document
        ? makeWorkbenchNodes(current.document, current.selectedStepIds)
        : [];
      setLocalNodes(nodes => reconcileWorkbenchNodes(nodes, desired));
    },
    [store]
  );
  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const updatePreference = () => {
      reducedMotion.current = preference.matches;
    };
    updatePreference();
    preference.addEventListener('change', updatePreference);
    return () => preference.removeEventListener('change', updatePreference);
  }, []);
  useEffect(() => {
    const identifiers = new Set(document?.steps.map(step => step.id));
    const previous = previousStepIds.current;
    if (identifiers.size !== previous.size || [...identifiers].some(id => !previous.has(id))) {
      const additions = [...identifiers].filter(id => !previous.has(id));
      revealSteps.current =
        additions.length > 0 && identifiers.size === previous.size + additions.length
          ? {ids: additions, usedFallback: false}
          : null;
      previousStepIds.current = identifiers;
    }
    const pending = revealSteps.current;
    function readSelectedAdditions(selected: readonly string[]): readonly string[] | null {
      if (!pending || !selected.length) return null;
      if (selected.length === 1 && pending.ids.length !== 1) return null;
      return selected.every(id => pending.ids.includes(id)) ? selected : null;
    }
    const selectedAdditions = readSelectedAdditions(state.selectedStepIds);
    if (
      !pending ||
      !selectedAdditions ||
      !flow.viewportInitialized ||
      canvasSize.width <= 0 ||
      canvasSize.height <= 0
    )
      return;
    function readNodes(ids: readonly string[]): WorkbenchFlowNode[] {
      return ids.flatMap(id => {
        const node = flow.getNode?.(id) ?? localNodes.find(candidate => candidate.id === id);
        return node ? [node] : [];
      });
    }
    function hasMeasurements(nodes: readonly WorkbenchFlowNode[]): boolean {
      return nodes.every(
        node => node.measured?.width !== undefined && node.measured.height !== undefined
      );
    }
    const nodes = readNodes(selectedAdditions);
    if (nodes.length !== selectedAdditions.length) return;
    const measured = hasMeasurements(nodes);
    if (pending.usedFallback && !measured) return;
    let frame: number | undefined;
    function reveal() {
      if (revealSteps.current !== pending) return;
      const selected = readSelectedAdditions(store.getState().selectedStepIds);
      if (!selected) return;
      const currentNodes = readNodes(selected);
      const rectangle = getWorkbenchNodeBounds(currentNodes);
      const bounds = flowContainer.current?.getBoundingClientRect();
      if (
        currentNodes.length !== selected.length ||
        !rectangle ||
        !bounds ||
        bounds.width <= 0 ||
        bounds.height <= 0
      )
        return;
      revealSteps.current = hasMeasurements(currentNodes)
        ? null
        : {...pending!, usedFallback: true};
      const reveal =
        selected.length > 1 ? getWorkbenchGroupRevealViewport : getWorkbenchRevealViewport;
      const viewport = reveal(rectangle, flow.getViewport(), bounds);
      if (viewport) void flow.setViewport(viewport, {duration: reducedMotion.current ? 0 : 200});
    }
    // Measurements can arrive after this render. The frame also lets initial fit/layout settle.
    if (measured && nodesInitialized) frame = window.requestAnimationFrame(reveal);
    const fallback =
      !(measured && nodesInitialized) && !pending.usedFallback
        ? setTimeout(() => {
            frame = window.requestAnimationFrame(reveal);
          }, 500)
        : undefined;
    return () => {
      if (fallback !== undefined) clearTimeout(fallback);
      if (frame !== undefined) window.cancelAnimationFrame(frame);
    };
  }, [document, state.selectedStepIds, localNodes, flow, canvasSize, nodesInitialized, store]);
  useEffect(() => {
    onApi?.({
      frame(stepIds) {
        void flow.fitView({
          ...fitViewOptions,
          nodes: stepIds?.length ? stepIds.map(id => ({id})) : undefined,
          duration: reducedMotion.current ? 0 : 200
        });
      },
      getViewportCenter() {
        const bounds =
          flowContainer.current?.getBoundingClientRect() ??
          container.current?.getBoundingClientRect();
        const viewport = flow.getViewport();
        return {
          x: ((bounds?.width ?? 0) / 2 - viewport.x) / viewport.zoom,
          y: ((bounds?.height ?? 0) / 2 - viewport.y) / viewport.zoom
        };
      }
    });
    return () => onApi?.(null);
  }, [flow, onApi, fitViewOptions]);
  const edges = useMemo(
    () =>
      document
        ? makeWorkbenchEdges(document, store.registry, state.runner).map(edge => ({
            ...edge,
            selected: selectedEdgeIds.includes(edge.id),
            type: 'smoothstep',
            animated: edge.data?.state === 'running',
            style: {
              stroke:
                edge.data?.state === 'invalid'
                  ? WORKBENCH_COLORS.danger
                  : WORKBENCH_PORT_COLORS[edge.data?.kind ?? 'table'],
              strokeDasharray: edge.data?.state === 'stale' ? '5 4' : undefined
            }
          }))
        : [],
    [document, store.registry, state.runner, selectedEdgeIds]
  );
  useEffect(() => {
    setSelectedEdgeIds(previous => {
      const remaining = previous.filter(id => edges.some(edge => edge.id === id));
      return remaining.length === previous.length ? previous : remaining;
    });
  }, [edges]);
  const activatePort = useCallback(
    (endpoint: WorkbenchPortEndpoint) => {
      if (!pendingPort || pendingPort.direction === endpoint.direction) {
        const repeated =
          pendingPort?.stepId === endpoint.stepId && pendingPort.port === endpoint.port;
        setPendingPort(repeated ? null : endpoint);
        store.setNotice({
          kind: 'info',
          message: repeated
            ? 'Connection cancelled.'
            : `Choose an ${endpoint.direction === 'out' ? 'input' : 'output'} port, then press Enter to connect. Press Escape to cancel.`
        });
        return;
      }
      const source = endpoint.direction === 'out' ? endpoint : pendingPort;
      const target = endpoint.direction === 'in' ? endpoint : pendingPort;
      if (
        store.edit('connect', current =>
          connectWorkflowSteps(current, store.registry, {
            from: {stepId: source.stepId, output: source.port},
            to: {stepId: target.stepId, input: target.port}
          })
        )
      ) {
        setPendingPort(null);
        store.setNotice({kind: 'info', message: 'Connected the ports.'});
      }
    },
    [pendingPort, store]
  );
  const keyboardConnection = useMemo(
    () => ({pending: pendingPort, activate: activatePort}),
    [pendingPort, activatePort]
  );
  const nodeColors = useMemo(
    () =>
      new Map(
        document?.steps.map(step => [
          step.id,
          WORKBENCH_STATUS[
            getWorkbenchStepStatus(
              step,
              store.registry.get(step.operationId),
              state.runner?.steps[step.id]
            )
          ].color
        ])
      ),
    [document, store.registry, state.runner]
  );
  const validateConnection = useCallback(
    (connection: Connection | WorkbenchFlowEdge) => {
      const edit = readConnection(connection);
      return Boolean(
        document && edit && checkWorkflowConnection(document, store.registry, edit).valid
      );
    },
    [document, store.registry]
  );
  return (
    <CanvasSurface
      ref={container}
      data-testid={WORKBENCH_TEST_IDS.canvas}
      aria-label="Workflow graph"
      tabIndex={0}
      onKeyDown={event => {
        if (
          event.defaultPrevented ||
          event.ctrlKey ||
          event.metaKey ||
          event.altKey ||
          event.shiftKey ||
          state.dialog ||
          state.palette.open
        )
          return;
        if (
          (event.target as HTMLElement).closest(
            'input, textarea, select, [contenteditable="true"], [role="textbox"]'
          )
        )
          return;
        if (event.key === 'Escape' && pendingPort) {
          event.preventDefault();
          event.stopPropagation();
          setPendingPort(null);
          store.setNotice({kind: 'info', message: 'Connection cancelled.'});
        } else if (
          (event.key === 'Delete' || event.key === 'Backspace') &&
          selectedEdgeIds.length
        ) {
          // Consume the connection deletion before the dock handles step-only deletion.
          event.preventDefault();
          event.stopPropagation();
          if (
            store.edit('delete-selection', current => {
              const disconnected = edges.reduce((updated, edge) => {
                const input = parseWorkbenchHandleId(edge.targetHandle ?? '');
                return selectedEdgeIds.includes(edge.id) && input?.direction === 'in'
                  ? disconnectWorkflowInput(updated, edge.target, input.port)
                  : updated;
              }, current);
              return removeWorkflowSteps(disconnected, store.getState().selectedStepIds);
            })
          )
            setSelectedEdgeIds([]);
        }
      }}
      onPointerDownCapture={event => {
        // Focus before React Flow can stop pointer events bubbling from its pane.
        if (
          event.target instanceof Element &&
          !event.target.closest('button, input, select, textarea, a, .react-flow__node')
        )
          container.current?.focus({preventScroll: true});
      }}
    >
      <WorkbenchConnectionContext.Provider value={keyboardConnection}>
        {document && (
          <ReactFlow<WorkbenchFlowNode, WorkbenchFlowEdge>
            ref={flowContainer}
            nodes={localNodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            colorMode="dark"
            minZoom={fitViewOptions.minZoom}
            maxZoom={2}
            defaultViewport={DEFAULT_VIEWPORT}
            fitView
            fitViewOptions={fitViewOptions}
            deleteKeyCode={null}
            zoomOnDoubleClick={false}
            onNodesChange={changes => {
              const previousSelection = store.getState().selectedStepIds;
              const nextSelection = applyWorkbenchSelectionChanges(
                previousSelection,
                changes.filter(change => change.type === 'select')
              );
              if (nextSelection !== previousSelection) store.select(nextSelection);
              const flowChanges = changes.filter(
                change => change.type !== 'remove' && change.type !== 'select'
              );
              if (flowChanges.length) setLocalNodes(nodes => applyNodeChanges(flowChanges, nodes));
              const positions = Object.fromEntries(
                changes.flatMap(change =>
                  change.type === 'position' && change.dragging !== true && change.position
                    ? [[change.id, change.position]]
                    : []
                )
              );
              if (Object.keys(positions).length) commitPositions(positions);
            }}
            onNodeDragStop={(_event, node, draggedNodes) =>
              commitPositions(
                Object.fromEntries(
                  (draggedNodes.length ? draggedNodes : [node]).map(dragged => [
                    dragged.id,
                    dragged.position
                  ])
                )
              )
            }
            isValidConnection={validateConnection}
            onConnect={connection => {
              const edit = readConnection(connection);
              if (edit)
                store.edit('connect', current =>
                  connectWorkflowSteps(current, store.registry, edit)
                );
            }}
            onConnectEnd={(event, connection) => {
              if (
                connection.isValid ||
                connection.toNode ||
                connection.fromHandle?.type !== 'source'
              )
                return;
              const handle = parseWorkbenchHandleId(connection.fromHandle.id ?? '');
              if (handle?.direction !== 'out' || !connection.fromNode) return;
              const target = event.target;
              if (!(target instanceof Element) || !target.closest('.react-flow__pane')) return;
              const point = 'changedTouches' in event ? event.changedTouches[0] : event;
              if (point)
                store.openPalette({
                  position: flow.screenToFlowPosition({x: point.clientX, y: point.clientY}),
                  connectFrom: {stepId: connection.fromNode.id, output: handle.port}
                });
            }}
            onEdgesChange={changes => {
              setSelectedEdgeIds(previous =>
                applyWorkbenchSelectionChanges(
                  previous,
                  changes.filter(change => change.type === 'select')
                )
              );
              const removed = new Set(
                changes.filter(change => change.type === 'remove').map(change => change.id)
              );
              if (removed.size)
                store.edit('disconnect', current =>
                  edges.reduce((updated, edge) => {
                    const input = parseWorkbenchHandleId(edge.targetHandle ?? '');
                    return removed.has(edge.id) && input?.direction === 'in'
                      ? disconnectWorkflowInput(updated, edge.target, input.port)
                      : updated;
                  }, current)
                );
            }}
            onDoubleClick={event => {
              if (
                event.target instanceof Element &&
                event.target.classList.contains('react-flow__pane')
              )
                store.openPalette({
                  position: flow.screenToFlowPosition({x: event.clientX, y: event.clientY})
                });
            }}
            onError={(_code, message) => store.setNotice({kind: 'error', message})}
          >
            <Background variant={BackgroundVariant.Dots} color={WORKBENCH_COLORS.border} />
            {document.steps.length > 0 && (
              <Controls showInteractive={false} fitViewOptions={fitViewOptions} />
            )}
            {showMinimap && (
              <MiniMap
                position="bottom-right"
                style={{
                  width: WORKBENCH_CANVAS_LAYOUT.minimapWidth,
                  height: WORKBENCH_CANVAS_LAYOUT.minimapHeight,
                  margin: WORKBENCH_CANVAS_LAYOUT.minimapMargin,
                  opacity: 0.75
                }}
                nodeColor={node => nodeColors.get(node.id) ?? WORKBENCH_COLORS.textMuted}
                bgColor={WORKBENCH_COLORS.surfaceRaised}
                nodeStrokeColor={WORKBENCH_COLORS.border}
                maskColor={WORKBENCH_COLORS.surfaceSunken}
              />
            )}
          </ReactFlow>
        )}
        {(!document || document.steps.length === 0) && (
          <EmptyState data-testid={WORKBENCH_TEST_IDS.canvasEmpty}>
            <strong>{document ? 'Add a step to begin' : 'No workflow yet'}</strong>
            {document ? (
              <>
                <WorkbenchButton onClick={() => store.openPalette()}>Add step</WorkbenchButton>
                <WorkbenchMuted>Press N, or click here and press Tab</WorkbenchMuted>
              </>
            ) : (
              <div>
                <WorkbenchButton onClick={() => store.createWorkflow()}>
                  New workflow
                </WorkbenchButton>
                <WorkbenchButton onClick={() => store.setDialog('templates')}>
                  Start from a template
                </WorkbenchButton>
              </div>
            )}
          </EmptyState>
        )}
      </WorkbenchConnectionContext.Provider>
    </CanvasSurface>
  );
}

/** React Flow canvas of the active workflow. */
export function WorkbenchCanvas({store, onApi}: CanvasProps): React.JSX.Element | null {
  const state = useWorkbenchState(store);
  return (
    <WorkbenchStoreContext.Provider value={store}>
      <ReactFlowProvider key={state.document?.id ?? 'empty'}>
        <WorkbenchCanvasContent store={store} onApi={onApi} />
      </ReactFlowProvider>
    </WorkbenchStoreContext.Provider>
  );
}
