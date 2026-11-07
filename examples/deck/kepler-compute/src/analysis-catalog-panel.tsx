// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useMemo, useRef, useState} from 'react';
import type * as arrow from 'apache-arrow';
import type KeplerTable from '@kepler.gl/table';
import type {Device} from '@luma.gl/core';
import {
  KEPLER_CATALOG_BINDINGS,
  getCatalogBinding,
  getCatalogDefaultParameters,
  listCatalogEntries,
  matchCatalogInvocationInputs,
  parseCatalogInvocation,
  parseCatalogFeatureCollection,
  listCatalogLensScenarios,
  runCatalogLensScenario,
  runCatalogOperationAsync,
  serializeCatalogInvocation,
  type CatalogLensScenario,
  type CatalogFeatureCollection,
  type CatalogOperationBinding,
  type CatalogOutputTable,
  type CatalogParameterSpec,
  type CatalogRunResult
} from '@luma-contributors/kepler';
import {makeArrowTableFromCatalogTable} from './analysis-catalog-table';
import {
  AnalysisCatalogLensMapping,
  EMPTY_LENS_MAPPING,
  readLensUserData,
  type LensMappingDraft
} from './analysis-catalog-lens-mapping';
import {
  listCatalogInvocations,
  recordCatalogInvocation,
  subscribeCatalogInvocations,
  type CatalogInvocationRecord
} from './catalog-invocation-store';
import {
  findNativeGeometryFields,
  getCatalogSampleInputs,
  readCatalogFeaturesFromNativeDataset
} from './analysis-catalog-input';

type InputDraft = {
  mode: 'sample' | 'text' | 'native';
  text: string;
  datasetId: string;
  field: string;
};

/** Catalog discovery plus the typed run form for operations that have executable bindings. */
export function AnalysisCatalogPanel({
  datasets,
  device,
  onPublish
}: {
  datasets: readonly KeplerTable[];
  /** GPU device of the compute runtime. Without it only operations with a CPU engine run. */
  device?: Device | null;
  /** The host stages the table in the Sources workflow; nothing is uploaded here. */
  onPublish(table: arrow.Table, label: string, geometryColumn?: string): void;
}) {
  const [lensScenarios, setLensScenarios] = useState<CatalogLensScenario[]>([]);
  const [lensScenarioId, setLensScenarioId] = useState('');
  const [lensMapping, setLensMapping] = useState<LensMappingDraft>(EMPTY_LENS_MAPPING);
  const [engineChoice, setEngineChoice] = useState<'auto' | 'cpu'>('auto');
  const [running, setRunning] = useState(false);
  const entries = useMemo(() => listCatalogEntries({lensScenarios}), [lensScenarios]);
  const [operationId, setOperationId] = useState(KEPLER_CATALOG_BINDINGS[0].id);
  const binding = getCatalogBinding(operationId) as CatalogOperationBinding;
  const [parameters, setParameters] = useState<Record<string, unknown>>(
    getCatalogDefaultParameters(binding)
  );
  const [drafts, setDrafts] = useState<Record<string, InputDraft>>({});
  const [result, setResult] = useState<CatalogRunResult | null>(null);
  const [error, setError] = useState('');
  const [invocationText, setInvocationText] = useState('');
  const [showBlocked, setShowBlocked] = useState(false);
  const lastInputs = useRef<Record<string, CatalogFeatureCollection>>({});
  const runnableCount = entries.filter(entry => entry.runnable !== 'no').length;
  const blocked = entries.filter(entry => entry.runnable === 'no');
  const geometryDatasets = datasets.map(dataset => ({
    dataset,
    fields: findNativeGeometryFields(dataset as never)
  }));

  const [savedInvocations, setSavedInvocations] = useState(listCatalogInvocations());
  useEffect(
    () => subscribeCatalogInvocations(() => setSavedInvocations(listCatalogInvocations())),
    []
  );
  const loadSavedInvocation = (record: CatalogInvocationRecord) =>
    invoke(() => {
      const restored = parseCatalogInvocation(record.invocation);
      setOperationId(restored.binding.id);
      setParameters({...restored.invocation.parameters});
      setEngineChoice(record.engine);
      setDrafts(current => ({
        ...current,
        ...Object.fromEntries(
          Object.entries(record.inputs).map(([name, input]) => [
            `${record.operationId}:${name}`,
            {
              mode: input.mode,
              text: input.mode === 'text' ? input.text : '',
              datasetId: input.mode === 'native' ? input.datasetId : '',
              field: input.mode === 'native' ? input.field : ''
            }
          ])
        )
      }));
      setResult(null);
    });

  const getDraft = (name: string): InputDraft =>
    drafts[`${operationId}:${name}`] ?? {mode: 'sample', text: '', datasetId: '', field: ''};
  const setDraft = (name: string, patch: Partial<InputDraft>) =>
    setDrafts(current => ({
      ...current,
      [`${operationId}:${name}`]: {...getDraft(name), ...patch}
    }));
  const selectOperation = (id: string) => {
    const next = getCatalogBinding(id) as CatalogOperationBinding;
    setOperationId(id);
    setParameters(getCatalogDefaultParameters(next));
    setResult(null);
    setError('');
  };
  const readInputs = (): Record<string, CatalogFeatureCollection> => {
    const samples = getCatalogSampleInputs(binding.id);
    const inputs: Record<string, CatalogFeatureCollection> = {};
    for (const spec of binding.inputs) {
      const draft = getDraft(spec.name);
      if (draft.mode === 'sample')
        inputs[spec.name] = parseCatalogFeatureCollection(samples[spec.name]);
      else if (draft.mode === 'text') inputs[spec.name] = parseCatalogFeatureCollection(draft.text);
      else {
        const source = datasets.find(dataset => dataset.id === draft.datasetId);
        // A native dataset and geometry field must be chosen explicitly.
        if (!source || !draft.field) throw new Error('Choose a dataset and geometry field');
        inputs[spec.name] = readCatalogFeaturesFromNativeDataset(source as never, draft.field);
      }
    }
    return inputs;
  };
  const invoke = (action: () => void) => {
    try {
      action();
      setError('');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  };
  const run = async () => {
    setRunning(true);
    try {
      const inputs = readInputs();
      lastInputs.current = inputs;
      // Saved with the workspace: normalized parameters, input hashes and where each input came from.
      recordCatalogInvocation({
        operationId: binding.id,
        invocation: serializeCatalogInvocation(binding, parameters, inputs),
        inputs: Object.fromEntries(
          binding.inputs.map(spec => {
            const draft = getDraft(spec.name);
            return [
              spec.name,
              draft.mode === 'native'
                ? {mode: 'native' as const, datasetId: draft.datasetId, field: draft.field}
                : draft.mode === 'text'
                  ? {mode: 'text' as const, text: draft.text}
                  : {mode: 'sample' as const}
            ];
          })
        ),
        engine: engineChoice
      });
      setResult(
        await runCatalogOperationAsync(binding, parameters, inputs, {
          device: device ?? undefined,
          engine: engineChoice
        })
      );
      setError('');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setRunning(false);
    }
  };
  const loadLensScenarios = async () => {
    try {
      // Scenario modules register themselves on import; this is only paid when asked for.
      await import('@luma-contributors/lenses/computation-lenses/scenarios/index');
      const scenarios = listCatalogLensScenarios();
      setLensScenarios(scenarios);
      setLensScenarioId(current => current || scenarios[0]?.id || '');
      setError('');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  };
  const runLensScenario = async () => {
    setRunning(true);
    try {
      // Lens scenarios dispatch GPU contributors on embedded datasets and need a device.
      if (!device) throw new Error('GPU device required');
      const userDataset = lensMapping.datasetId
        ? datasets.find(dataset => dataset.id === lensMapping.datasetId)
        : undefined;
      // With a table chosen, the mapped numeric fields stand in for the embedded dataset.
      const output = await runCatalogLensScenario(device, lensScenarioId, {
        ...(userDataset
          ? {userData: readLensUserData(userDataset as never, lensMapping.mapping)}
          : {})
      });
      setResult({
        operationId: lensScenarioId,
        engine: 'gpu',
        dispatchedContributors: output.dispatchedContributors ?? [],
        parameters: {},
        tables: output.tables,
        diagnostics: output.diagnostics,
        summary: output.summary
      });
      setError('');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setRunning(false);
    }
  };
  const serialize = () =>
    invoke(() => {
      const inputs = readInputs();
      lastInputs.current = inputs;
      setInvocationText(serializeCatalogInvocation(binding, parameters, inputs));
    });
  const restore = () =>
    invoke(() => {
      const restored = parseCatalogInvocation(invocationText);
      setOperationId(restored.binding.id);
      setParameters({...restored.invocation.parameters});
      setResult(null);
      if (!matchCatalogInvocationInputs(restored.invocation, lastInputs.current))
        setError('Parameters restored. Current inputs differ from the serialized invocation.');
    });
  const publish = (table: CatalogOutputTable) =>
    invoke(() =>
      onPublish(makeArrowTableFromCatalogTable(table), table.label, table.geometryColumn)
    );

  return (
    <section
      aria-label="Analysis catalog"
      data-testid="analysis-catalog"
      style={{borderTop: '1px solid #405168', marginTop: 18, paddingTop: 10}}
    >
      <h2 style={{fontSize: 16, marginBottom: 8}}>Contributor catalog</h2>
      <p className="hint" data-testid="catalog-summary">
        {runnableCount} of {entries.length} contributors are runnable here;{' '}
        {KEPLER_CATALOG_BINDINGS.length} bindings are registered. With a GPU device bindings
        dispatch the real graph contributors; the CPU reference engine is the oracle and the
        fallback when no device is available.{' '}
        <span data-testid="catalog-device">
          {device ? `GPU device: ${device.type}` : 'No GPU device'}
        </span>
      </p>
      <label htmlFor="catalog-engine">Engine</label>
      <select
        id="catalog-engine"
        value={engineChoice}
        onChange={event => setEngineChoice(event.target.value as 'auto' | 'cpu')}
        style={{width: '100%'}}
      >
        <option value="auto">GPU when available</option>
        <option value="cpu">CPU reference</option>
      </select>
      <button
        type="button"
        aria-expanded={showBlocked}
        aria-controls="catalog-blocked-list"
        onClick={() => setShowBlocked(value => !value)}
      >
        {showBlocked ? 'Hide' : 'Show'} {blocked.length} not runnable
      </button>
      {showBlocked && (
        <ul
          id="catalog-blocked-list"
          data-testid="catalog-blocked"
          style={{maxHeight: 160, overflow: 'auto', fontSize: 12}}
        >
          {blocked.slice(0, 300).map(entry => (
            <li key={entry.contributorName} title={entry.reason}>
              {entry.contributorName} · {entry.capabilityId}
            </li>
          ))}
          <li>{blocked[0]?.reason}</li>
        </ul>
      )}
      <details data-testid="catalog-lens">
        <summary>Lens scenarios ({lensScenarios.length} loaded)</summary>
        <p className="hint">
          Registered lens scenarios run a contributor on an embedded dataset and show its recorded
          graph. Scenarios backed by an embedded dataframe dataset can instead run on one of your
          tables through a typed field mapping. Loading registers the scenario modules.
        </p>
        <button data-testid="catalog-lens-load" onClick={loadLensScenarios}>
          Load lens scenarios
        </button>
        {lensScenarios.length > 0 && (
          <>
            <select
              aria-label="Lens scenario"
              value={lensScenarioId}
              onChange={event => {
                setLensScenarioId(event.target.value);
                setLensMapping(EMPTY_LENS_MAPPING);
              }}
              style={{width: '100%'}}
            >
              {lensScenarios.map(scenario => (
                <option key={scenario.id} value={scenario.id}>
                  {scenario.element} · {scenario.title}
                </option>
              ))}
            </select>
            <AnalysisCatalogLensMapping
              scenarioId={lensScenarioId}
              datasets={datasets}
              draft={lensMapping}
              onChange={setLensMapping}
            />
            <button
              data-testid="catalog-lens-run"
              onClick={runLensScenario}
              disabled={running || !device}
            >
              Run scenario
            </button>
          </>
        )}
      </details>
      <label htmlFor="catalog-operation">Operation</label>
      <select
        id="catalog-operation"
        value={operationId}
        style={{width: '100%'}}
        onChange={event => selectOperation(event.target.value)}
      >
        {KEPLER_CATALOG_BINDINGS.map(candidate => (
          <option key={candidate.id} value={candidate.id}>
            {candidate.domain} · {candidate.title} [
            {candidate.engine === 'gpu'
              ? 'GPU'
              : candidate.engine === 'gpu-or-cpu'
                ? 'GPU/CPU'
                : 'CPU'}
            ]
          </option>
        ))}
      </select>
      <p className="hint">{binding.summary}</p>
      <details>
        <summary>Contract</summary>
        <ul style={{fontSize: 12}}>
          {binding.declaredBehavior.map(line => (
            <li key={line}>{line}</li>
          ))}
          <li>Limitation: {binding.limitation}</li>
          <li>Cleanup: {binding.cleanup}</li>
          <li>
            Outputs:{' '}
            {binding.outputs
              .map(
                output =>
                  `${output.label} (≤ ${output.maximumRows} rows; ${output.identityColumns.join(', ')})`
              )
              .join('; ')}
          </li>
          <li>Contributors mirrored: {binding.contributorNames.join(', ') || 'none'}</li>
          <li>GPU contributors dispatched: {binding.gpuContributorNames.join(', ') || 'none'}</li>
        </ul>
      </details>
      {binding.parameters.map(spec => (
        <ParameterField
          key={`${binding.id}:${spec.name}`}
          spec={spec}
          value={parameters[spec.name]}
          onChange={value => setParameters(current => ({...current, [spec.name]: value}))}
        />
      ))}
      {binding.inputs.map(spec => {
        const draft = getDraft(spec.name);
        const selectedDataset = geometryDatasets.find(item => item.dataset.id === draft.datasetId);
        return (
          <fieldset key={spec.name} style={{marginTop: 8}}>
            <legend>
              {spec.label} ({spec.accepts.join('/')}, ≤ {spec.maximumFeatures} features)
            </legend>
            <select
              aria-label={`${spec.label} source`}
              value={draft.mode}
              onChange={event =>
                setDraft(spec.name, {mode: event.target.value as InputDraft['mode']})
              }
              style={{width: '100%'}}
            >
              <option value="sample">Built-in sample</option>
              <option value="text">GeoJSON text</option>
              <option value="native">Native dataset</option>
            </select>
            {draft.mode === 'text' && (
              <textarea
                aria-label={`${spec.label} GeoJSON`}
                rows={4}
                value={draft.text}
                onChange={event => setDraft(spec.name, {text: event.target.value})}
                style={{width: '100%', boxSizing: 'border-box'}}
              />
            )}
            {draft.mode === 'native' && (
              <>
                <select
                  aria-label={`${spec.label} dataset`}
                  value={draft.datasetId}
                  onChange={event =>
                    setDraft(spec.name, {datasetId: event.target.value, field: ''})
                  }
                  style={{width: '100%'}}
                >
                  <option value="">Choose dataset</option>
                  {geometryDatasets
                    .filter(item => item.fields.length)
                    .map(item => (
                      <option key={item.dataset.id} value={item.dataset.id}>
                        {item.dataset.label}
                      </option>
                    ))}
                </select>
                <select
                  aria-label={`${spec.label} geometry field`}
                  value={draft.field}
                  onChange={event => setDraft(spec.name, {field: event.target.value})}
                  style={{width: '100%'}}
                >
                  <option value="">Choose geometry field</option>
                  {(selectedDataset?.fields ?? []).map(field => (
                    <option key={field} value={field}>
                      {field}
                    </option>
                  ))}
                </select>
              </>
            )}
          </fieldset>
        );
      })}
      {savedInvocations.length > 0 && (
        <details data-testid="catalog-saved-invocations" open>
          <summary>
            Saved runs ({savedInvocations.length}, travel with the workspace bundle)
          </summary>
          <ul style={{fontSize: 12}}>
            {savedInvocations.map(record => (
              <li key={record.id}>
                {record.id}{' '}
                <button onClick={() => loadSavedInvocation(record)}>Load into form</button>
              </li>
            ))}
          </ul>
        </details>
      )}
      <p style={{marginTop: 8}}>
        <button data-testid="catalog-run" onClick={run} disabled={running}>
          {running ? 'Running' : 'Run'}
        </button>{' '}
        <button onClick={serialize}>Serialize invocation</button>{' '}
        <button onClick={() => setResult(null)} disabled={!result}>
          Clear result
        </button>
      </p>
      {invocationText && (
        <>
          <textarea
            aria-label="Serialized invocation"
            rows={3}
            value={invocationText}
            onChange={event => setInvocationText(event.target.value)}
            style={{width: '100%', boxSizing: 'border-box'}}
          />
          <button onClick={restore}>Restore parameters</button>
        </>
      )}
      {error && (
        <p role="alert" className="error" data-testid="catalog-error">
          {error}
        </p>
      )}
      {result && (
        <div data-testid="catalog-result">
          <p className="hint" data-testid="catalog-engine">
            Engine: {result.engine === 'gpu' ? 'GPU' : 'CPU reference'}
            {result.dispatchedContributors.length
              ? ` (${result.dispatchedContributors.join(', ')})`
              : ''}
            {result.fallbackReason ? ` · GPU not used: ${result.fallbackReason}` : ''}
          </p>
          <p className="hint" data-testid="catalog-result-summary">
            {Object.entries(result.summary)
              .map(([key, value]) => `${key}: ${value}`)
              .join(' · ')}
          </p>
          <ul style={{fontSize: 12}}>
            {result.diagnostics.map(item => (
              <li key={item.code}>
                [{item.severity}] {item.code} ×{item.count}: {item.message}
              </li>
            ))}
          </ul>
          {result.tables.map(table => (
            <div key={table.id}>
              <strong>
                {table.label}: {table.rowCount} rows
              </strong>{' '}
              <button
                data-testid="catalog-publish"
                disabled={table.rowCount === 0}
                onClick={() => publish(table)}
              >
                Stage in Sources
              </button>
              <TablePreview table={table} />
            </div>
          ))}
          <p className="hint">
            Staged results appear in the Sources tab; publishing uploads them as a native dataset.
          </p>
        </div>
      )}
    </section>
  );
}

function TablePreview({table}: {table: CatalogOutputTable}) {
  const names = Object.keys(table.columnTypes).filter(name => name !== table.geometryColumn);
  return (
    <div style={{overflow: 'auto', maxHeight: 160}}>
      <table aria-label="Result preview" style={{fontSize: 11}}>
        <thead>
          <tr>
            {names.map(name => (
              <th key={name} scope="col">
                {name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({length: Math.min(8, table.rowCount)}, (_, row) => (
            <tr key={row}>
              {names.map(name => {
                const value = table.columns[name][row];
                return (
                  <td key={name}>
                    {typeof value === 'number' && !Number.isInteger(value)
                      ? value.toFixed(3)
                      : String(value ?? '')}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ParameterField({
  spec,
  value,
  onChange
}: {
  spec: CatalogParameterSpec;
  value: unknown;
  onChange(value: unknown): void;
}) {
  const id = `catalog-parameter-${spec.name}`;
  const label = spec.unit
    ? `${spec.label} (${spec.unit === 'crs' ? 'CRS units' : spec.unit})`
    : spec.label;
  if (spec.kind === 'boolean')
    return (
      <label htmlFor={id} style={{display: 'block'}}>
        <input
          id={id}
          type="checkbox"
          checked={Boolean(value)}
          onChange={event => onChange(event.target.checked)}
        />{' '}
        {label}
      </label>
    );
  if (spec.kind === 'enum')
    return (
      <>
        <label htmlFor={id}>{label}</label>
        <select
          id={id}
          value={String(value)}
          onChange={event => onChange(event.target.value)}
          style={{width: '100%'}}
        >
          {spec.options?.map(option => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </>
    );
  return (
    <>
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        value={Array.isArray(value) ? value.join(', ') : String(value ?? '')}
        onChange={event => onChange(event.target.value)}
        style={{width: '100%', boxSizing: 'border-box'}}
      />
    </>
  );
}
