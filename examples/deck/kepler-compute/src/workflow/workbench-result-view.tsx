// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useMemo, useRef, useState} from 'react';
import styled from 'styled-components';
import {
  getWorkflowSuggestions,
  estimateWorkflowPreflight,
  profileKeplerWorkbenchData,
  formatKeplerResultReport,
  setWorkflowDisplayStyle,
  makeWorkflowCSVText,
  makeWorkflowGeoJSONText,
  summarizeWorkflowColumn,
  updateWorkflowStep,
  type WorkflowColumnSummary,
  type WorkflowData,
  type WorkflowStep,
  type WorkflowStepState
} from '@luma-contributors/kepler';
import {AnalysisResultCard} from '../analysis-result-card';
import {
  createAnalysisResultCardModel,
  addAnalysisResultNextStep,
  type AnalysisResultNextAction
} from '../analysis-result-card-model';
import {
  WORKBENCH_COLORS,
  WORKBENCH_PORT_LABELS,
  WORKBENCH_STATUS,
  WORKBENCH_TEST_IDS
} from './workbench-contract';
import {
  formatWorkbenchCell,
  formatWorkbenchDuration,
  formatWorkbenchSummaryKey,
  getWorkbenchPreviewPage,
  getWorkbenchColumnDisplayStyle,
  makeWorkbenchDownloadName,
  makeWorkbenchHistogram,
  type WorkbenchHistogram
} from './workbench-result-model';
import type {WorkbenchStore} from './workbench-store';
import {
  WorkbenchButton,
  WorkbenchIconButton,
  WorkbenchMuted,
  WorkbenchNoticeBar,
  WorkbenchPortSwatch,
  WorkbenchSection,
  WorkbenchStatusBadge,
  WorkbenchSurface,
  WorkbenchTable
} from './workbench-styles';
import {useWorkbenchState} from './use-workbench';
import {WorkbenchRunTrace} from './workbench-run-log';

const ResultSurface = styled(WorkbenchSurface)`
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  height: 100%;
  overflow: auto;
  & > [aria-live], & > ${WorkbenchNoticeBar} { flex-shrink: 0; }

  /* The host inspector uses content sizing for the other tabs. */
  [data-testid='${WORKBENCH_TEST_IDS.inspector}']:has(&) {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
  }
  [data-testid='${WORKBENCH_TEST_IDS.inspector}']:has(&) > [role='tablist'] {
    flex: none;
  }
  [role='tabpanel']:has(> &) {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 0;
  }
`;
const RunHeader = styled.div`
  flex: none;
  padding: 4px 8px;
  border-bottom: 1px solid ${WORKBENCH_COLORS.border};
`;
const RunLine = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  min-height: 24px;
  white-space: nowrap;
`;
const EngineLabel = styled.span`
  flex: 0 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
`;
const ResultRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin: 6px 0;
`;
const OutputSection = styled(WorkbenchSection)`
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  /* Include the header, minimum table height and pager in the section's minimum size. */
  min-height: auto;
  padding: 4px 8px;
`;
const OutputHeader = styled.div`
  display: flex;
  align-items: center;
  flex: none;
  gap: 3px;
  min-height: 26px;
`;
const OutputLabel = styled.span`
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;
const TableScroll = styled.div`
  flex: 1 1 auto;
  min-height: 96px;
  overflow: auto;
`;
const Pager = styled(ResultRow)`
  flex: none;
  flex-wrap: nowrap;
  justify-content: space-between;
  gap: 3px;
  margin: 0;
  min-height: 26px;
`;
const Definitions = styled.dl`
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 3px 8px;
  margin: 8px 0;
  & dt { color: ${WORKBENCH_COLORS.textMuted}; }
  & dd { margin: 0; overflow-wrap: anywhere; }
`;
const Disclosure = styled.details`
  flex: none;
  margin: 0;
  & > summary {
    min-height: 24px;
    cursor: pointer;
    padding: 3px 0;
  }
  & > summary:focus-visible {
    outline: 2px solid ${WORKBENCH_COLORS.focus};
    outline-offset: 1px;
  }
`;

const Metadata = styled.div`
  flex: none;
  padding: 0 8px;
`;
const OutputDisclosure = styled(Disclosure)`
  & > summary {
    display: flex;
    align-items: center;
    gap: 3px;
    padding: 0;
    min-height: 26px;
    list-style: none;
  }
  & > summary::-webkit-details-marker { display: none; }
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  min-height: auto;
  &[open]::details-content {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
    min-height: auto;
  }
`;
const ColumnPanel = styled.div`
  flex: none;
`;
const PublicationLine = styled(WorkbenchMuted)`
  flex: none;
`;
const PublicationNotice = styled(WorkbenchNoticeBar)`
  flex: none;
`;

function ResultActionIcon({
  action
}: {
  action: 'display' | 'focus' | 'csv' | 'geojson';
}): React.JSX.Element {
  return (
    <svg
      aria-hidden="true"
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.2"
    >
      {action === 'display' ? (
        <>
          <path d="m1.5 3 4-1.5 5 2 4-1.5v11l-4 1.5-5-2-4 1.5Z" />
          <path d="M5.5 1.5v11m5-9v11" />
        </>
      ) : action === 'focus' ? (
        <>
          <path d="M1.5 6V1.5H6m4 0h4.5V6m0 4v4.5H10m-4 0H1.5V10" />
          <circle cx="8" cy="8" r="3" />
        </>
      ) : (
        <>
          <path d="M8 1v7m-3-3 3 3 3-3M2 10v4h12v-4" />
          <text x="8" y="12.5" textAnchor="middle" fontSize="4" fill="currentColor" stroke="none">
            {action === 'csv' ? 'CSV' : 'JSON'}
          </text>
        </>
      )}
    </svg>
  );
}

type CachedColumnSummary = {summary: WorkflowColumnSummary; histogram: WorkbenchHistogram};

// WorkflowData is immutable. A weak cache survives preview unmounts without retaining old results.
const COLUMN_SUMMARY_CACHE = new WeakMap<WorkflowData, Map<string, CachedColumnSummary>>();

function getColumnSummary(data: WorkflowData, column: string): CachedColumnSummary {
  let summaries = COLUMN_SUMMARY_CACHE.get(data);
  if (!summaries) {
    summaries = new Map();
    COLUMN_SUMMARY_CACHE.set(data, summaries);
  }
  let cached = summaries.get(column);
  if (!cached) {
    const summary = summarizeWorkflowColumn(data, column);
    cached = {
      summary,
      histogram:
        summary.type === 'int32' || summary.type === 'float64'
          ? makeWorkbenchHistogram(data.table.columns[column], 12)
          : {edges: [], counts: []}
    };
    summaries.set(column, cached);
  }
  return cached;
}

function ColumnSummary({
  column,
  summary,
  histogram,
  close,
  colouring
}: {
  column: string;
  summary: WorkflowColumnSummary;
  histogram: WorkbenchHistogram;
  close(): void;
  colouring?: {active: boolean; output: string; apply(): void};
}): React.JSX.Element {
  const closeButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    closeButton.current?.focus();
  }, []);
  const maximumCount = Math.max(0, ...histogram.counts);
  const busiestIndex = histogram.counts.indexOf(maximumCount);
  const distribution = maximumCount
    ? `Distribution: most values between ${formatWorkbenchCell(histogram.edges[busiestIndex], 'float64')} and ${formatWorkbenchCell(histogram.edges[busiestIndex + 1], 'float64')}`
    : 'Distribution: no finite values';
  return (
    <ColumnPanel
      role="dialog"
      aria-label={`${column} summary`}
      onKeyDown={event => {
        if (event.key === 'Escape') {
          event.stopPropagation();
          close();
        }
      }}
    >
      <ResultRow>
        <strong>{column}</strong>
        <WorkbenchButton
          type="button"
          ref={closeButton}
          onClick={close}
          aria-label={`Close ${column} summary`}
        >
          Close
        </WorkbenchButton>
      </ResultRow>
      {colouring && (
        <ResultRow>
          <WorkbenchButton
            type="button"
            data-testid={WORKBENCH_TEST_IDS.displayColorColumn(colouring.output)}
            aria-pressed={colouring.active}
            onClick={colouring.apply}
          >
            {colouring.active ? 'Remove colouring' : 'Colour the map by this column'}
          </WorkbenchButton>
        </ResultRow>
      )}
      <Definitions>
        <dt>Type</dt>
        <dd>{summary.type}</dd>
        <dt>Nulls</dt>
        <dd>{summary.nullCount.toLocaleString()}</dd>
        {summary.minimum !== undefined && (
          <>
            <dt>Minimum</dt>
            <dd>{formatWorkbenchCell(summary.minimum, summary.type)}</dd>
          </>
        )}
        {summary.maximum !== undefined && (
          <>
            <dt>Maximum</dt>
            <dd>{formatWorkbenchCell(summary.maximum, summary.type)}</dd>
          </>
        )}
        {summary.mean !== undefined && (
          <>
            <dt>Mean</dt>
            <dd>{formatWorkbenchCell(summary.mean, summary.type)}</dd>
          </>
        )}
        {summary.distinctCount !== undefined && (
          <>
            <dt>Distinct values</dt>
            <dd>
              {summary.distinctCount.toLocaleString()}
              {summary.distinctLimitReached ? '+' : ''}
            </dd>
          </>
        )}
      </Definitions>
      {(summary.type === 'int32' || summary.type === 'float64') && (
        <>
          <svg viewBox="0 0 240 60" width="240" height="60" role="img" aria-label={distribution}>
            <title>{distribution}</title>
            {histogram.counts.map((count, index) => {
              const width = 240 / histogram.counts.length;
              const height = maximumCount ? (count / maximumCount) * 60 : 0;
              return (
                <rect
                  key={histogram.edges[index]}
                  x={index * width}
                  y={60 - height}
                  width={Math.max(1, width - 1)}
                  height={height}
                  fill={WORKBENCH_COLORS.accent}
                />
              );
            })}
          </svg>
          <p>{distribution}</p>
        </>
      )}
      {summary.topValues && (
        <ul aria-label="Top values">
          {summary.topValues.map(entry => (
            <li key={entry.value} title={entry.value}>
              {formatWorkbenchCell(entry.value, 'string')}: {entry.count.toLocaleString()}
            </li>
          ))}
        </ul>
      )}
    </ColumnPanel>
  );
}

function OutputPreview({
  data,
  output,
  label,
  store,
  step
}: {
  data: WorkflowData;
  output: string;
  label: string;
  store: WorkbenchStore;
  step: WorkflowStep;
}): React.JSX.Element {
  const [pageIndex, setPageIndex] = useState(0);
  const [openColumn, setOpenColumn] = useState<string | null>(null);
  const opener = useRef<HTMLButtonElement | null>(null);
  const page = useMemo(() => getWorkbenchPreviewPage(data, pageIndex, 50), [data, pageIndex]);
  const selectedSummary = useMemo(() => {
    if (openColumn === null || !Object.hasOwn(data.table.columns, openColumn)) return null;
    return getColumnSummary(data, openColumn);
  }, [data, openColumn]);
  const canColour = Boolean(step.display?.includes(output) || store.host.publishOutput);
  function colourColumn(): void {
    if (openColumn === null || selectedSummary === null || !canColour) return;
    const column = openColumn;
    store.edit(`display-style:${step.id}:${output}`, document => {
      const current = document.steps.find(candidate => candidate.id === step.id);
      const display = current?.display ?? [];
      if (display.includes(output) && current?.displayStyle?.[output]?.colorColumn === column) {
        return setWorkflowDisplayStyle(document, step.id, output, undefined);
      }
      const displayed = display.includes(output)
        ? document
        : updateWorkflowStep(document, step.id, {
            display: [...display, output]
          });
      return setWorkflowDisplayStyle(
        displayed,
        step.id,
        output,
        getWorkbenchColumnDisplayStyle(selectedSummary.summary)
      );
    });
  }
  function closeSummary(): void {
    setOpenColumn(null);
    opener.current?.focus();
  }
  return (
    <>
      {selectedSummary && openColumn !== null && (
        <ColumnSummary
          key={openColumn}
          column={openColumn}
          {...selectedSummary}
          close={closeSummary}
          colouring={
            canColour && openColumn !== data.table.geometryColumn
              ? {
                  active:
                    Boolean(step.display?.includes(output)) &&
                    step.displayStyle?.[output]?.colorColumn === openColumn,
                  output,
                  apply: colourColumn
                }
              : undefined
          }
        />
      )}
      <TableScroll>
        <WorkbenchTable
          data-testid={WORKBENCH_TEST_IDS.resultTable(output)}
          aria-label={`${label} preview`}
        >
          <thead>
            <tr>
              {page.columns.map(column => (
                <th key={column} scope="col">
                  <WorkbenchButton
                    type="button"
                    $variant="ghost"
                    aria-label={`Summarize ${column}`}
                    aria-haspopup="dialog"
                    aria-expanded={openColumn === column}
                    onClick={event => {
                      opener.current = event.currentTarget;
                      setOpenColumn(column);
                    }}
                  >
                    {column}
                  </WorkbenchButton>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {page.rows.map((row, rowIndex) => (
              <tr key={page.startRow + rowIndex}>
                {row.map((value, columnIndex) => {
                  const column = page.columns[columnIndex];
                  const type =
                    column === data.table.geometryColumn
                      ? 'geometry'
                      : data.table.columnTypes[column];
                  return (
                    <td
                      key={column}
                      className={typeof value === 'number' ? 'number' : undefined}
                      title={typeof value === 'string' ? value : undefined}
                    >
                      {value === null ? (
                        <WorkbenchMuted>null</WorkbenchMuted>
                      ) : (
                        formatWorkbenchCell(value, type)
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </WorkbenchTable>
      </TableScroll>
      <Pager>
        {page.pageCount > 1 ? (
          <>
            <WorkbenchButton
              type="button"
              aria-label={`Previous ${label} page`}
              disabled={page.pageIndex === 0}
              onClick={() => setPageIndex(page.pageIndex - 1)}
            >
              Previous
            </WorkbenchButton>
            <span aria-live="polite">
              Rows {page.startRow.toLocaleString()} to {page.endRow.toLocaleString()} of{' '}
              {page.rowCount.toLocaleString()}
            </span>
            <WorkbenchButton
              type="button"
              aria-label={`Next ${label} page`}
              disabled={page.pageIndex + 1 >= page.pageCount}
              onClick={() => setPageIndex(page.pageIndex + 1)}
            >
              Next
            </WorkbenchButton>
          </>
        ) : (
          <span aria-live="polite">{page.rowCount.toLocaleString()} rows</span>
        )}
      </Pager>
    </>
  );
}

function ResultOutput({
  store,
  workflowId,
  step,
  stepState,
  output,
  data,
  label,
  initiallyExpanded,
  multiple
}: {
  store: WorkbenchStore;
  workflowId: string;
  step: WorkflowStep;
  stepState: WorkflowStepState;
  output: string;
  data: WorkflowData;
  label: string;
  initiallyExpanded: boolean;
  multiple: boolean;
}): React.JSX.Element {
  const [expanded, setExpanded] = useState(initiallyExpanded);
  const published = stepState.published?.[output];
  const publicationIssue = stepState.issues.find(
    issue => issue.code === 'publish-failed' && (!issue.port || issue.port === output)
  );
  function toggleDisplay(): void {
    store.edit(`display:${step.id}:${output}`, document => {
      const current = document.steps.find(candidate => candidate.id === step.id);
      const display = current?.display ?? [];
      return updateWorkflowStep(document, step.id, {
        display: display.includes(output)
          ? display.filter(name => name !== output)
          : [...display, output]
      });
    });
  }
  function download(extension: 'csv' | 'geojson'): void {
    try {
      const text = extension === 'csv' ? makeWorkflowCSVText(data) : makeWorkflowGeoJSONText(data);
      const blob = new Blob([text], {
        type: extension === 'csv' ? 'text/csv;charset=utf-8' : 'application/geo+json'
      });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      try {
        anchor.href = url;
        anchor.download = makeWorkbenchDownloadName(workflowId, step.id, output, extension);
        document.body.append(anchor);
        anchor.click();
      } finally {
        anchor.remove();
        setTimeout(() => URL.revokeObjectURL(url), 0);
      }
    } catch (error) {
      store.setNotice({
        kind: 'error',
        message: `${step.label ?? store.registry.get(step.operationId)?.title ?? step.operationId}: ${label} download failed: ${error instanceof Error ? error.message : String(error)}`
      });
    }
  }
  const colorColumn = step.displayStyle?.[output]?.colorColumn;
  const colouringLabel = colorColumn === undefined ? '' : ` · coloured by ${colorColumn}`;
  const headingText = `${label} · ${WORKBENCH_PORT_LABELS[data.kind]} · ${data.table.rowCount.toLocaleString()} rows · ${Object.keys(data.table.columns).length.toLocaleString()} columns${colouringLabel}`;
  const heading = (
    <OutputLabel title={headingText}>
      <strong>{label}</strong> <WorkbenchPortSwatch $kind={data.kind} aria-hidden="true" />{' '}
      {WORKBENCH_PORT_LABELS[data.kind]} · {data.table.rowCount.toLocaleString()} rows ·{' '}
      {Object.keys(data.table.columns).length.toLocaleString()} columns
      {colouringLabel}
    </OutputLabel>
  );
  const actions = (
    <>
      <WorkbenchIconButton
        type="button"
        data-testid={WORKBENCH_TEST_IDS.resultDisplay(output)}
        aria-label={`Show ${label} on the map`}
        title={`Show ${label} on the map`}
        aria-pressed={Boolean(step.display?.includes(output))}
        onClick={toggleDisplay}
      >
        <ResultActionIcon action="display" />
      </WorkbenchIconButton>
      {published && store.host.focusOutput && (
        <WorkbenchIconButton
          type="button"
          data-testid={WORKBENCH_TEST_IDS.resultFocus(output)}
          aria-label="Zoom the map to this output"
          title="Zoom the map to this output"
          onClick={() => store.host.focusOutput?.(published.datasetId)}
        >
          <ResultActionIcon action="focus" />
        </WorkbenchIconButton>
      )}
      <WorkbenchIconButton
        type="button"
        data-testid={WORKBENCH_TEST_IDS.resultDownload(output, 'csv')}
        aria-label={`Download ${label} CSV`}
        title={`Download ${label} CSV`}
        onClick={() => download('csv')}
      >
        <ResultActionIcon action="csv" />
      </WorkbenchIconButton>
      {data.table.geometryColumn !== undefined &&
        Object.hasOwn(data.table.columns, data.table.geometryColumn) && (
          <WorkbenchIconButton
            type="button"
            data-testid={WORKBENCH_TEST_IDS.resultDownload(output, 'geojson')}
            aria-label={`Download ${label} GeoJSON`}
            title={`Download ${label} GeoJSON`}
            onClick={() => download('geojson')}
          >
            <ResultActionIcon action="geojson" />
          </WorkbenchIconButton>
        )}
    </>
  );
  const content = (
    <>
      {published && <PublicationLine>On the map as &quot;{published.label}&quot;</PublicationLine>}
      {publicationIssue && (
        <PublicationNotice $kind="error" aria-live="polite">
          {publicationIssue.message}
        </PublicationNotice>
      )}
      {data.analysisReport && !data.analysisRun && (
        <details>
          <summary>Analysis report</summary>
          <pre style={{whiteSpace: 'pre-wrap'}}>
            {formatKeplerResultReport(data.analysisReport)}
          </pre>
        </details>
      )}
      {data.analysisPresentation && !step.displayStyle?.[output]?.colorColumn && (
        <ul aria-label={`${label} legend`}>
          {data.analysisPresentation.legend.map((entry, index) => (
            <li key={index}>
              <span
                aria-hidden="true"
                style={{
                  display: 'inline-block',
                  width: 12,
                  height: 12,
                  backgroundColor: `rgb(${entry.color.join(',')})`
                }}
              />{' '}
              {entry.label}
              {entry.count === undefined ? '' : `: ${entry.count}`}
            </li>
          ))}
        </ul>
      )}
      <OutputPreview
        key={output}
        data={data}
        output={output}
        label={label}
        store={store}
        step={step}
      />
    </>
  );
  return (
    <OutputSection data-testid={WORKBENCH_TEST_IDS.resultOutput(output)}>
      {multiple ? (
        <OutputDisclosure open={expanded} onToggle={event => setExpanded(event.currentTarget.open)}>
          <summary>
            <span aria-hidden="true">{expanded ? '▾' : '▸'}</span>
            {heading}
            {actions}
          </summary>
          {expanded && content}
        </OutputDisclosure>
      ) : (
        <>
          <OutputHeader>
            {heading}
            {actions}
          </OutputHeader>
          {content}
        </>
      )}
    </OutputSection>
  );
}

/** Outputs, preview table, column summaries and diagnostics of one step. */
export function WorkbenchResultView({store}: {store: WorkbenchStore}): React.JSX.Element | null {
  const state = useWorkbenchState(store);
  const preflight = useMemo(
    () =>
      state.document
        ? estimateWorkflowPreflight(state.document, store.registry, {
            state: state.runner,
            profileSources: true,
            host: store.host
          })
        : undefined,
    [state.document, state.runner, store.registry, store.host]
  );
  const selected = useMemo(
    () =>
      state.selectedStepIds.length === 1
        ? state.document?.steps.find(step => step.id === state.selectedStepIds[0])
        : undefined,
    [state.document, state.selectedStepIds]
  );
  const stepState = selected ? state.runner?.steps[selected.id] : undefined;
  const result = stepState?.result;
  const analysisOutput = Object.values(result?.outputs ?? {}).find(data => data.analysisRun);
  const card = useMemo(() => {
    const run = analysisOutput?.analysisRun;
    if (!run) return undefined;
    const sensitivity = selected?.sensitivity;
    return createAnalysisResultCardModel(
      sensitivity ? {...run.result, sensitivity} : run.result,
      run.context,
      {
        report: sensitivity ? undefined : analysisOutput.analysisReport,
        executionEngine:
          (selected?.engine ?? state.document?.settings.engine) === 'cpu' || !store.host.getDevice()
            ? 'cpu'
            : 'gpu',
        outputProfiles: Object.values(result?.outputs ?? {}).map(
          data => data.analysisProfile ?? profileKeplerWorkbenchData(data)
        )
      }
    );
  }, [
    analysisOutput,
    result,
    store.host,
    state.revision,
    selected?.engine,
    selected?.sensitivity,
    state.document?.settings.engine
  ]);
  const operation = selected ? store.registry.get(selected.operationId) : undefined;
  const outputs = useMemo(() => {
    if (!result) return [];
    const names = [...(operation?.outputs.map(output => output.name) ?? [])];
    for (const name of Object.keys(result.outputs)) if (!names.includes(name)) names.push(name);
    return names
      .filter(name => Object.hasOwn(result.outputs, name))
      .map(name => ({
        name,
        data: result.outputs[name],
        label:
          operation?.outputs.find(output => output.name === name)?.label ??
          result.outputs[name].table.label ??
          name
      }));
  }, [result, operation]);
  const diagnostics = useMemo(() => {
    const priority = {error: 0, warning: 1, info: 2};
    return [...(result?.diagnostics ?? [])].sort(
      (first, second) => priority[first.severity] - priority[second.severity]
    );
  }, [result]);
  const error = stepState?.error;
  const suggestions = useMemo(() => {
    if (!state.document || !selected || !error) return [];
    try {
      return getWorkflowSuggestions(state.document, store.registry, state.runner, {
        ...error,
        stepId: selected.id
      });
    } catch {
      // Optional repairs must not prevent inspection of the original run error and result.
      return [];
    }
  }, [state.document, state.runner, selected, error, store.registry]);
  if (!state.document || !selected)
    return (
      <WorkbenchSurface data-testid={WORKBENCH_TEST_IDS.result}>
        <WorkbenchSection>Select one step to see its output.</WorkbenchSection>
      </WorkbenchSurface>
    );
  const status =
    stepState?.status ?? (selected.disabled ? 'disabled' : operation ? 'ready' : 'not-configured');
  const issue = stepState?.issues.find(candidate => candidate.severity === 'error');
  const stale = result && result.key !== stepState?.key;
  const engineLabel =
    result?.engine === 'gpu'
      ? `GPU${result.dispatchedContributors.length ? ` · ${result.dispatchedContributors.join(', ')}` : ''}`
      : result?.engine === 'cpu-reference'
        ? 'CPU reference'
        : 'Host';
  return (
    <ResultSurface data-testid={WORKBENCH_TEST_IDS.result} data-status={status}>
      <RunHeader>
        <RunLine data-testid={WORKBENCH_TEST_IDS.resultEngine}>
          <WorkbenchStatusBadge $status={status} data-status={status} aria-live="polite">
            {status === 'running' ? 'Running…' : WORKBENCH_STATUS[status].label}
          </WorkbenchStatusBadge>
          {result && (
            <>
              <EngineLabel title={engineLabel}>{engineLabel}</EngineLabel>
              <span>
                {result.cacheHit
                  ? 'from cache'
                  : formatWorkbenchDuration(result.durationMilliseconds)}
              </span>
              <time dateTime={new Date(result.completedAt).toISOString()}>
                {new Date(result.completedAt).toLocaleTimeString(undefined, {
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                  hour12: false
                })}
              </time>
            </>
          )}
        </RunLine>
        {result?.fallbackReason && <div>GPU not used: {result.fallbackReason}</div>}
      </RunHeader>
      <div aria-live="polite">
        {error && (
          <>
            <WorkbenchNoticeBar $kind="error">{error.message}</WorkbenchNoticeBar>
            {suggestions.length > 0 && (
              <ResultRow>
                {suggestions.map(suggestion => (
                  <WorkbenchButton
                    type="button"
                    key={suggestion.id}
                    data-testid={WORKBENCH_TEST_IDS.suggestion(suggestion.id)}
                    title={suggestion.description}
                    onClick={() =>
                      store.edit(`suggestion:${suggestion.id}`, document =>
                        suggestion.apply(document, store.registry)
                      )
                    }
                  >
                    {suggestion.label}
                  </WorkbenchButton>
                ))}
              </ResultRow>
            )}
          </>
        )}
        {!result && !error && (
          <WorkbenchSection>
            {issue?.message ??
              (!operation
                ? `${selected.label ?? selected.operationId}: This operation is unavailable.`
                : status === 'running'
                  ? 'The output will appear when the step finishes.'
                  : 'Run this step to see its output')}
          </WorkbenchSection>
        )}
      </div>
      {stale && (
        <WorkbenchNoticeBar $kind="info">
          <span>This result is from an earlier configuration. Run the step to update it.</span>
          <WorkbenchButton
            type="button"
            disabled={Boolean(state.runner?.running)}
            onClick={() => {
              void store.run({target: selected.id});
            }}
          >
            Run to here
          </WorkbenchButton>
        </WorkbenchNoticeBar>
      )}
      {card && (
        <AnalysisResultCard
          key={`${state.document.id}:${selected.id}:${result!.key}`}
          model={card}
          inStack
          showLegends={false}
          actionsDisabledReason={
            stale || state.runner?.running
              ? 'Run the current configuration before saving or adding a next step.'
              : undefined
          }
          onSaveWorkflow={() => {
            if (!store.saveTemplate())
              throw new Error(store.getState().notice?.message ?? 'Could not save this workflow.');
          }}
          onNext={(action: AnalysisResultNextAction) => {
            const output = outputs.find(candidate => candidate.data.table.id === action.tableId);
            const input = Object.entries(action.applicability.inputAssignments).find(
              ([, datasetId]) => datasetId === action.tableId
            )?.[0];
            if (!output || !input || action.applicability.status === 'blocked')
              throw new Error('Choose an applicable output before adding a next step.');
            let addedStepId = '';
            const changed = store.edit('result:next', document => {
              const added = addAnalysisResultNextStep(document, store.registry, action, {
                stepId: selected.id,
                output: output.name
              });
              addedStepId = added.stepId;
              return added.document;
            });
            if (!changed)
              throw new Error(
                store.getState().notice?.message ?? 'Could not add the next analysis.'
              );
            store.select([addedStepId]);
            store.setDock({open: true, inspectorTab: 'parameters'});
          }}
        />
      )}
      {stepState &&
        outputs.map((output, index) => (
          <ResultOutput
            key={`${state.document!.id}:${selected.id}:${output.name}:${result!.key}`}
            store={store}
            workflowId={state.document!.id}
            step={selected}
            stepState={stepState}
            output={output.name}
            data={output.data}
            label={output.label}
            multiple={outputs.length > 1}
            initiallyExpanded={index === 0}
          />
        ))}
      {result && (
        <Metadata>
          <Disclosure>
            <summary>Summary ({Object.keys(result.summary).length.toLocaleString()})</summary>
            <Definitions aria-label="Result summary">
              {Object.entries(result.summary).map(([key, value]) => (
                <div key={key} style={{display: 'contents'}}>
                  <dt>{formatWorkbenchSummaryKey(key)}</dt>
                  <dd>
                    {typeof value === 'string' ? value : formatWorkbenchCell(value, 'float64')}
                  </dd>
                </div>
              ))}
            </Definitions>
          </Disclosure>
          <Disclosure
            data-testid={WORKBENCH_TEST_IDS.resultDiagnostics}
            open={diagnostics.some(
              diagnostic => diagnostic.severity === 'warning' || diagnostic.severity === 'error'
            )}
          >
            <summary>Diagnostics ({diagnostics.length.toLocaleString()})</summary>
            {diagnostics.length > 0 && (
              <ul>
                {diagnostics.map((diagnostic, index) => (
                  <li key={`${diagnostic.code}:${index}`}>
                    <strong>{formatWorkbenchSummaryKey(diagnostic.severity)}</strong> ·{' '}
                    {diagnostic.code} · {diagnostic.count.toLocaleString()} · {diagnostic.message}
                  </li>
                ))}
              </ul>
            )}
          </Disclosure>
          <Disclosure>
            <summary>
              Parameters used ({Object.keys(result.parameters).length.toLocaleString()})
            </summary>
            <Definitions>
              {Object.entries(result.parameters).map(([key, value]) => (
                <div key={key} style={{display: 'contents'}}>
                  <dt>
                    {operation?.parameters.find(parameter => parameter.name === key)?.label ??
                      formatWorkbenchSummaryKey(key)}
                  </dt>
                  <dd>
                    {Array.isArray(value)
                      ? value.map(number => formatWorkbenchCell(number, 'float64')).join(', ')
                      : typeof value === 'string'
                        ? value
                        : formatWorkbenchCell(value, '')}
                  </dd>
                </div>
              ))}
            </Definitions>
          </Disclosure>
        </Metadata>
      )}
      <Metadata>
        {preflight && (
          <Disclosure>
            <summary>Preflight and materialisation</summary>
            <WorkbenchMuted>
              Cold-run logical GPU scratch; input/output buffers, staging and host heap excluded.
            </WorkbenchMuted>
            {preflight.steps
              .filter(estimate => estimate.stepId === selected.id)
              .map(estimate => (
                <div key={estimate.stepId}>
                  Scratch:{' '}
                  {estimate.scratchBytes === null
                    ? 'unknown'
                    : `${estimate.scratchBytes.toLocaleString()} B`}
                  {' · '}Command nodes: {estimate.commandNodeCount ?? 'unknown'}
                  <div>{estimate.reason}</div>
                </div>
              ))}
            {preflight.boundaries
              .filter(
                boundary => boundary.consumer === selected.id || boundary.producer === selected.id
              )
              .map(boundary => (
                <div
                  key={`${boundary.producer}:${boundary.output}:${boundary.consumer}:${boundary.input}`}
                >
                  {boundary.producer}.{boundary.output} → {boundary.consumer}.{boundary.input}:{' '}
                  {boundary.materialization === 'cpu-raster' ? 'CPU raster band' : 'CPU table'}
                </div>
              ))}
          </Disclosure>
        )}
        {result &&
          state.runLog
            .filter(run =>
              run.steps.some(
                entry =>
                  entry.stepId === selected.id &&
                  entry.key === result.key &&
                  entry.status === 'done'
              )
            )
            .slice(0, 1)
            .map(run => <WorkbenchRunTrace key={run.runId} run={run} registry={store.registry} />)}
      </Metadata>
    </ResultSurface>
  );
}
