// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useId, useMemo, useState} from 'react';
import {
  GPU_ANALYSIS_QUESTION_RECORDS,
  GPU_SPATIAL_ANALYSIS_DATA_KINDS,
  matchGPUAnalysisFieldRole,
  type GPUAnalysisQuestion
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  createWorkflowAnalysisTemplateFromDocument,
  exportWorkflowAnalysisTemplateJSON,
  getWorkflowSaveTemplateSlots,
  mapWorkflowAnalysisTemplate,
  saveWorkflowAnalysisTemplate,
  planWorkflow,
  type CatalogParameterValue,
  type WorkflowAnalysisRoleSource,
  type WorkflowAnalysisTemplate,
  type WorkflowDocument,
  type WorkflowTemplateMapping
} from '@luma-contributors/kepler';
import {WorkbenchButton, WorkbenchField, WorkbenchSection} from './workbench-styles';
import {getWorkbenchAnalysisProfiles} from './workbench-analysis-launch';
import {useWorkbenchState} from './use-workbench';
import type {WorkbenchStore} from './workbench-store';

function readError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Configure a reusable stack using the same field roles and promoted settings as other forms. */
export function WorkbenchSaveTemplate({
  document,
  store,
  onClose
}: {
  document: WorkflowDocument;
  store: WorkbenchStore;
  onClose(): void;
}) {
  const identifier = useId();
  const candidates = useMemo(
    () => getWorkflowSaveTemplateSlots(document, store.registry),
    [document, store.registry]
  );
  const [title, setTitle] = useState(document.title);
  const [method, setMethod] = useState(document.description ?? document.title);
  const [question, setQuestion] = useState<GPUAnalysisQuestion>(
    () =>
      createWorkflowAnalysisTemplateFromDocument({
        id: document.id,
        title: document.title,
        summary: document.description ?? document.title,
        document
      }).question
  );
  const [slots, setSlots] = useState(candidates);
  const [exposed, setExposed] = useState<WorkflowAnalysisRoleSource[]>([]);
  const [error, setError] = useState('');
  const settings = document.steps.flatMap(step =>
    step.operationId.startsWith('source-')
      ? []
      : (store.registry.get(step.operationId)?.parameters ?? [])
          .filter(
            parameter =>
              ['number', 'integer', 'boolean', 'enum', 'number-list', 'string'].includes(
                parameter.kind
              ) &&
              !candidates.some(slot =>
                slot.fields.some(field =>
                  field.sources.some(
                    source => source.stepId === step.id && source.parameter === parameter.name
                  )
                )
              )
          )
          .map(parameter => ({step, parameter}))
  );
  return (
    <WorkbenchSection aria-label="Save reusable template" data-testid="ax-save-template">
      <h3>Save a reusable template</h3>
      <WorkbenchField>
        <label htmlFor={`${identifier}-title`}>Title</label>
        <input
          id={`${identifier}-title`}
          value={title}
          onChange={event => setTitle(event.target.value)}
        />
      </WorkbenchField>
      <WorkbenchField>
        <label htmlFor={`${identifier}-question`}>Question</label>
        <select
          id={`${identifier}-question`}
          value={question}
          onChange={event => setQuestion(event.target.value as GPUAnalysisQuestion)}
        >
          {GPU_ANALYSIS_QUESTION_RECORDS.map(record => (
            <option key={record.id} value={record.id}>
              {record.label}
            </option>
          ))}
        </select>
      </WorkbenchField>
      <WorkbenchField>
        <label htmlFor={`${identifier}-method`}>Method and result reading</label>
        <textarea
          id={`${identifier}-method`}
          value={method}
          onChange={event => setMethod(event.target.value)}
        />
      </WorkbenchField>
      <p>
        Choose reusable datasets and field roles. Unchecked datasets and fields keep their current
        settings.
      </p>
      {candidates.map(candidate => {
        const selected = slots.find(slot => slot.name === candidate.name);
        return (
          <fieldset key={candidate.name}>
            <legend>
              <label>
                <input
                  type="checkbox"
                  checked={Boolean(selected)}
                  onChange={event =>
                    setSlots(
                      event.target.checked
                        ? [...slots, candidate]
                        : slots.filter(slot => slot.name !== candidate.name)
                    )
                  }
                />
                {candidate.label} as a dataset slot
              </label>
            </legend>
            {selected && (
              <>
                <label>
                  Slot label{' '}
                  <input
                    value={selected.label}
                    onChange={event =>
                      setSlots(
                        slots.map(slot =>
                          slot.name === selected.name ? {...slot, label: event.target.value} : slot
                        )
                      )
                    }
                  />
                </label>
                <label>
                  Data kind{' '}
                  <select
                    value={selected.kind}
                    onChange={event =>
                      setSlots(
                        slots.map(slot =>
                          slot.name === selected.name
                            ? {...slot, kind: event.target.value as typeof slot.kind}
                            : slot
                        )
                      )
                    }
                  >
                    {GPU_SPATIAL_ANALYSIS_DATA_KINDS.map(kind => (
                      <option key={kind}>{kind}</option>
                    ))}
                  </select>
                </label>
                {candidate.fields.map(field => (
                  <div key={field.role}>
                    <label>
                      <input
                        type="checkbox"
                        checked={selected.fields.some(current => current.role === field.role)}
                        onChange={event =>
                          setSlots(
                            slots.map(slot =>
                              slot.name === selected.name
                                ? {
                                    ...slot,
                                    fields: event.target.checked
                                      ? [...slot.fields, field]
                                      : slot.fields.filter(current => current.role !== field.role)
                                  }
                                : slot
                            )
                          )
                        }
                      />
                      {field.label} (
                      {field.sources
                        .map(source => `${source.stepId}.${source.parameter}`)
                        .join(', ')}
                      )
                    </label>
                  </div>
                ))}
              </>
            )}
          </fieldset>
        );
      })}
      <fieldset>
        <legend>Exposed settings</legend>
        {settings.map(({step, parameter}) => (
          <div key={`${step.id}.${parameter.name}`}>
            <label>
              <input
                type="checkbox"
                checked={exposed.some(
                  target => target.stepId === step.id && target.parameter === parameter.name
                )}
                onChange={event =>
                  setExposed(
                    event.target.checked
                      ? [...exposed, {stepId: step.id, parameter: parameter.name}]
                      : exposed.filter(
                          target => target.stepId !== step.id || target.parameter !== parameter.name
                        )
                  )
                }
              />
              {step.label ?? step.id}: {parameter.label}
            </label>
          </div>
        ))}
      </fieldset>
      {error && <p role="alert">{error}</p>}
      <WorkbenchButton
        data-testid="ax-save-template-apply"
        disabled={!title.trim() || !method.trim()}
        onClick={() => {
          try {
            const template = saveWorkflowAnalysisTemplate(
              document,
              {id: document.id, title, method, question, slots, exposed},
              store.registry
            );
            if (!store.saveAnalysisTemplate?.(template))
              throw new Error(
                store.getState().notice?.message ?? 'The template could not be saved.'
              );
            onClose();
          } catch (failure) {
            setError(readError(failure));
          }
        }}
      >
        Save template
      </WorkbenchButton>{' '}
      <WorkbenchButton onClick={onClose}>Cancel</WorkbenchButton>
    </WorkbenchSection>
  );
}

/** Explicit source/column mapping reuses AX compatibility and the standard execution lifecycle. */
export function WorkbenchSavedTemplate({
  template,
  store
}: {
  template: WorkflowAnalysisTemplate;
  store: WorkbenchStore;
}) {
  useWorkbenchState(store);
  const [mappingOpen, setMappingOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [datasetIds, setDatasetIds] = useState<Record<string, string>>({});
  const [columns, setColumns] = useState<WorkflowTemplateMapping['columns']>({});
  const [parameters, setParameters] = useState<Record<string, CatalogParameterValue>>({});
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const profiles = getWorkbenchAnalysisProfiles(store);
  const datasets = Object.fromEntries(
    template.slots.flatMap(slot => {
      const profile = profiles.find(profile => profile.datasetId === datasetIds[slot.name]);
      return profile ? [[slot.name, profile]] : [];
    })
  );
  let document: WorkflowDocument | undefined;
  let mappingError = '';
  if (mappingOpen) {
    try {
      document = mapWorkflowAnalysisTemplate(
        template,
        {datasets, columns, parameters},
        store.registry
      );
      const plan = planWorkflow(document, {registry: store.registry, host: store.host});
      const failures = [
        ...plan.issues,
        ...Object.values(plan.steps).flatMap(step => step.issues)
      ].filter(issue => issue.severity === 'error');
      if (failures.length) {
        document = undefined;
        mappingError = failures.map(issue => issue.message).join('\n');
      }
    } catch (failure) {
      mappingError = readError(failure);
    }
  }
  return (
    <article data-testid={`ax-saved-template-${template.id}`}>
      <h4>{template.title}</h4>
      <p>{template.method}</p>
      <p>
        {template.slots.length} reusable datasets · {template.exposed.length} exposed settings
      </p>
      <WorkbenchButton onClick={() => setMappingOpen(!mappingOpen)}>Map my data</WorkbenchButton>{' '}
      <WorkbenchButton onClick={() => setExportOpen(!exportOpen)}>
        Export template JSON
      </WorkbenchButton>{' '}
      <WorkbenchButton disabled={busy} onClick={() => store.deleteAnalysisTemplate?.(template.id)}>
        Delete template
      </WorkbenchButton>
      {exportOpen && (
        <WorkbenchField>
          <label>
            Template JSON
            <textarea
              aria-label={`Template JSON: ${template.title}`}
              readOnly
              value={exportWorkflowAnalysisTemplateJSON(template)}
            />
          </label>
        </WorkbenchField>
      )}
      {mappingOpen && (
        <fieldset disabled={busy}>
          <legend>Map datasets and fields</legend>
          {template.slots.map(slot => {
            const profile = datasets[slot.name];
            return (
              <div key={slot.name}>
                <label>
                  {slot.label}
                  <select
                    aria-label={`Dataset: ${slot.label}`}
                    value={datasetIds[slot.name] ?? ''}
                    onChange={event => {
                      setDatasetIds({...datasetIds, [slot.name]: event.target.value});
                      setColumns(
                        Object.fromEntries(
                          Object.entries(columns).filter(
                            ([key]) =>
                              !slot.fields.some(field => key === `${slot.name}.${field.role}`)
                          )
                        )
                      );
                    }}
                  >
                    <option value="">Choose dataset</option>
                    {profiles.map(profile => (
                      <option
                        key={profile.datasetId}
                        value={profile.datasetId}
                        disabled={
                          profile.kind !== slot.kind &&
                          !(slot.kind === 'network' && profile.kind === 'lines')
                        }
                      >
                        {profile.label} ({profile.kind})
                      </option>
                    ))}
                  </select>
                </label>
                {slot.fields.map(field => {
                  const key = `${slot.name}.${field.role}`;
                  const compatible = matchGPUAnalysisFieldRole(field, profile?.columns ?? []);
                  return (
                    <label key={field.role}>
                      {field.label}
                      <select
                        aria-label={`${slot.label}: ${field.label}`}
                        multiple={Boolean(field.multiple)}
                        disabled={!profile}
                        value={
                          field.multiple ? [...(columns[key] ?? [])] : (columns[key]?.[0] ?? '')
                        }
                        onChange={event =>
                          setColumns({
                            ...columns,
                            [key]: Array.from(event.target.selectedOptions)
                              .map(option => option.value)
                              .filter(Boolean)
                          })
                        }
                      >
                        {!field.multiple && <option value="">Choose column</option>}
                        {(profile?.columns ?? []).map(column => {
                          const candidate = compatible.find(
                            candidate => candidate.name === column.name
                          );
                          return (
                            <option key={column.name} disabled={!candidate} value={column.name}>
                              {column.name}
                              {candidate?.warning ? ` — ${candidate.warning}` : ''}
                            </option>
                          );
                        })}
                      </select>
                    </label>
                  );
                })}
              </div>
            );
          })}
          {template.document.inputs
            .filter(input => template.exposed.some(exposed => exposed.inputId === input.id))
            .map(input => (
              <label key={input.id}>
                {input.label}
                {input.kind === 'enum' ? (
                  <select
                    aria-label={input.label}
                    value={String(parameters[input.id] ?? input.value)}
                    onChange={event =>
                      setParameters({...parameters, [input.id]: event.target.value})
                    }
                  >
                    {input.options?.map(option => (
                      <option key={option}>{option}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    aria-label={input.label}
                    type={
                      input.kind === 'boolean'
                        ? 'checkbox'
                        : input.kind === 'number' || input.kind === 'integer'
                          ? 'number'
                          : 'text'
                    }
                    checked={
                      input.kind === 'boolean'
                        ? Boolean(parameters[input.id] ?? input.value)
                        : undefined
                    }
                    value={
                      input.kind === 'boolean'
                        ? undefined
                        : String(parameters[input.id] ?? input.value)
                    }
                    onChange={event =>
                      setParameters({
                        ...parameters,
                        [input.id]:
                          input.kind === 'boolean'
                            ? event.target.checked
                            : input.kind === 'number' || input.kind === 'integer'
                              ? Number(event.target.value)
                              : input.kind === 'number-list'
                                ? event.target.value.split(',').map(Number)
                                : event.target.value
                      })
                    }
                  />
                )}
              </label>
            ))}
          {mappingError && <p role="status">{mappingError}</p>}
          <WorkbenchButton
            disabled={!document || busy}
            onClick={async () => {
              if (!document) return;
              setBusy(true);
              setError('');
              try {
                const workflowId = store.addWorkflow(document);
                if (!workflowId)
                  throw new Error(
                    store.getState().notice?.message ?? 'The workflow could not be added.'
                  );
                const record = await store.run();
                if (record?.status !== 'done')
                  throw new Error(
                    record?.steps
                      .flatMap(step => (step.error ? [step.error.message] : []))
                      .join('\n') || 'Run did not finish.'
                  );
                store.setDialog(null);
              } catch (failure) {
                setError(readError(failure));
              } finally {
                setBusy(false);
              }
            }}
          >
            Run mapped template
          </WorkbenchButton>
        </fieldset>
      )}
      {error && <p role="alert">{error}</p>}
    </article>
  );
}
