// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useId, useRef, useState} from 'react';
import styled from 'styled-components';
import {
  GPU_ANALYSIS_ROW_SEMANTICS,
  type GPUAnalysisRowSemantics
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {
  getCatalogBinding,
  type CatalogParameterValue,
  type KeplerCuratedAnalysis,
  type KeplerCuratedAnalysisMapping,
  type KeplerCuratedAnalysisMappingChange,
  type KeplerDatasetProfile
} from '@luma-contributors/kepler';
import {
  getWorkbenchMappingParameterRows,
  getWorkbenchMappingParameters
} from './workbench-analysis-launch';
import {WorkbenchParameterValue} from './workbench-parameter-form';
import {WorkbenchButton, WorkbenchField, WorkbenchSection} from './workbench-styles';
import {WORKBENCH_COLORS} from './workbench-contract';

const MappingSection = styled(WorkbenchSection)`
  & option:disabled { color: ${WORKBENCH_COLORS.textMuted}; }
  & fieldset { margin: 12px 0; border: 1px solid ${WORKBENCH_COLORS.border}; }
`;

/** Render the shared typed mapping; no local role inference or dataset matching. */
export function WorkbenchAnalysisMapping({
  analysis,
  mapping,
  profiles,
  onChange,
  onParametersChange,
  onRowSemanticsChange,
  onRun,
  onCancel,
  busy = false,
  executionIssues = []
}: {
  analysis: KeplerCuratedAnalysis;
  mapping: KeplerCuratedAnalysisMapping;
  profiles: readonly KeplerDatasetProfile[];
  onChange(change: KeplerCuratedAnalysisMappingChange): void;
  onParametersChange(parameters: Record<string, CatalogParameterValue>): void;
  onRowSemanticsChange?(datasetId: string, semantics: GPUAnalysisRowSemantics): void;
  onRun(): void;
  onCancel(): void;
  busy?: boolean;
  executionIssues?: readonly string[];
}) {
  const identifier = useId();
  const [drafts, setDrafts] = useState<Record<string, string[]>>({});
  useEffect(() => setDrafts({}), [mapping]);
  const [parameterProblems, setParameterProblems] = useState<Record<string, string>>({});
  const currentParameterProblems = useRef(parameterProblems);
  currentParameterProblems.current = parameterProblems;
  const explicitParameters = getWorkbenchMappingParameters(mapping);
  const rows = getWorkbenchMappingParameterRows(analysis, mapping, profiles);
  const binding =
    analysis.execution.status === 'available'
      ? getCatalogBinding(analysis.execution.bindingId)
      : undefined;
  return (
    <MappingSection
      data-testid="ax-analysis-mapping"
      aria-label={`Map fields for ${analysis.title}`}
    >
      <h3>{analysis.title}</h3>
      <p role="status" data-testid="ax-mapping-progress">
        {mapping.progress.mapped} of {mapping.progress.required} mapped
      </p>
      {mapping.slots.map(slot => {
        const profile = profiles.find(profile => profile.datasetId === slot.datasetId);
        const slotSpecification = analysis.slots.find(candidate => candidate.name === slot.slot)!;
        return (
          <fieldset key={slot.slot} disabled={busy}>
            <legend>{slot.label}</legend>
            <WorkbenchField>
              <label htmlFor={`${identifier}-${slot.slot}`}>Dataset</label>
              <select
                id={`${identifier}-${slot.slot}`}
                data-testid={`ax-mapping-dataset-${slot.slot}`}
                value={slot.datasetId ?? ''}
                onChange={event => onChange({slot: slot.slot, datasetId: event.target.value})}
              >
                <option value="" disabled>
                  Choose a dataset
                </option>
                {slot.candidates.map(candidate => (
                  <option
                    key={candidate.datasetId}
                    value={candidate.datasetId}
                    disabled={candidate.fit !== 'fits'}
                    title={candidate.warning ?? candidate.reason}
                  >
                    {candidate.label}
                  </option>
                ))}
              </select>
              {slot.reason && <p>{slot.reason}</p>}
              {slot.state === 'choose' && slot.datasetId && (
                <WorkbenchButton
                  onClick={() => onChange({slot: slot.slot, datasetId: slot.datasetId!})}
                >
                  Use this dataset
                </WorkbenchButton>
              )}
            </WorkbenchField>
            {slot.roles.map(role => {
              const roleKey = `${slot.slot}.${role.role}`;
              const selected = drafts[roleKey] ?? [...role.columns];
              return (
                <WorkbenchField key={role.role}>
                  <label htmlFor={`${identifier}-${roleKey}`}>
                    {role.label}
                    {role.required ? ' (required)' : ' (optional)'}
                  </label>
                  <select
                    id={`${identifier}-${roleKey}`}
                    data-testid={`ax-mapping-column-${roleKey}`}
                    multiple={Boolean(role.multiple)}
                    value={role.multiple ? selected : (selected[0] ?? '')}
                    onChange={event => {
                      const columns = Array.from(
                        event.target.selectedOptions,
                        option => option.value
                      ).filter(Boolean);
                      if (
                        role.multiple &&
                        (columns.length < role.multiple.minimum ||
                          columns.length > role.multiple.maximum)
                      ) {
                        setDrafts(previous => ({...previous, [roleKey]: columns}));
                      } else {
                        setDrafts(previous => {
                          const next = {...previous};
                          delete next[roleKey];
                          return next;
                        });
                        onChange({slot: slot.slot, role: role.role, columns});
                      }
                    }}
                  >
                    {!role.multiple && (
                      <option value="" disabled={role.required}>
                        {role.required ? 'Choose a column' : 'None'}
                      </option>
                    )}
                    {(profile?.columns ?? []).map(column => {
                      const candidate = role.candidates.find(
                        candidate => candidate.name === column.name
                      );
                      return (
                        <option
                          key={column.name}
                          value={column.name}
                          disabled={!candidate || Boolean(candidate.unavailable)}
                          title={
                            candidate?.unavailable ??
                            candidate?.warning ??
                            (!candidate
                              ? `This field does not have a type accepted by ${role.label}.`
                              : undefined)
                          }
                        >
                          {column.name}
                        </option>
                      );
                    })}
                  </select>
                  {role.reason && <p>{role.reason}</p>}
                  {drafts[roleKey] && (
                    <p role="alert">
                      Choose {role.multiple?.minimum} to {role.multiple?.maximum} columns.
                    </p>
                  )}
                  {role.state === 'choose' && role.columns.length > 0 && (
                    <WorkbenchButton
                      onClick={() =>
                        onChange({slot: slot.slot, role: role.role, columns: role.columns})
                      }
                    >
                      Use these fields
                    </WorkbenchButton>
                  )}
                </WorkbenchField>
              );
            })}
            {profile && slotSpecification.rowSemantics?.length && onRowSemanticsChange && (
              <WorkbenchField>
                <label htmlFor={`${identifier}-${slot.slot}-rows`}>
                  What does each row represent?
                </label>
                <select
                  id={`${identifier}-${slot.slot}-rows`}
                  data-testid={`ax-mapping-rows-${slot.slot}`}
                  value={profile.rowSemantics?.value ?? ''}
                  onChange={event =>
                    onRowSemanticsChange(
                      profile.datasetId,
                      event.target.value as GPUAnalysisRowSemantics
                    )
                  }
                >
                  <option value="" disabled>
                    Choose the meaning of a row
                  </option>
                  {GPU_ANALYSIS_ROW_SEMANTICS.map(semantics => (
                    <option key={semantics} value={semantics}>
                      {semantics}
                    </option>
                  ))}
                </select>
              </WorkbenchField>
            )}
          </fieldset>
        );
      })}
      {binding?.parameters.some(parameter => parameter.name === 'coordinateSystem') && (
        <WorkbenchField>
          <label htmlFor={`${identifier}-coordinates`}>Coordinate system</label>
          <select
            id={`${identifier}-coordinates`}
            data-testid="ax-mapping-coordinate-system"
            disabled={busy}
            value={String(explicitParameters['coordinateSystem'] ?? '')}
            onChange={event => {
              const parameters = {...explicitParameters};
              if (event.target.value) parameters['coordinateSystem'] = event.target.value;
              else delete parameters['coordinateSystem'];
              onParametersChange(parameters);
            }}
          >
            <option value="">Use the dataset declaration</option>
            <option value="EPSG:4326">Longitude and latitude (EPSG:4326)</option>
            <option value="planar">Planar coordinates</option>
          </select>
        </WorkbenchField>
      )}
      <fieldset disabled={busy}>
        <legend>Analysis settings</legend>
        {rows
          .filter(row => row.spec.name !== 'coordinateSystem')
          .map(row => (
            <WorkbenchField key={row.spec.name}>
              <WorkbenchParameterValue
                spec={row.spec}
                value={row.value}
                bindingIdentity={`${mapping.analysisId}:${mapping.slots.map(slot => slot.datasetId).join(',')}`}
                testId={`ax-mapping-parameter-${row.spec.name}`}
                onValidityChange={problem => {
                  const next = {...currentParameterProblems.current};
                  if (problem) next[row.spec.name] = problem;
                  else delete next[row.spec.name];
                  currentParameterProblems.current = next;
                  setParameterProblems(next);
                }}
                onCommit={value =>
                  onParametersChange({...explicitParameters, [row.spec.name]: value})
                }
              />
            </WorkbenchField>
          ))}
      </fieldset>
      {mapping.problems
        .filter(problem => !problem.acknowledged)
        .map((problem, index) => (
          <p key={index}>{problem.message}</p>
        ))}
      {executionIssues.map((issue, index) => (
        <p role="alert" key={index}>
          {issue}
        </p>
      ))}
      <WorkbenchButton
        data-testid="ax-mapping-run"
        $variant="primary"
        disabled={
          busy ||
          !mapping.ready ||
          Object.keys(drafts).length > 0 ||
          Object.keys(parameterProblems).length > 0 ||
          executionIssues.length > 0
        }
        onClick={() => {
          if (
            !busy &&
            mapping.ready &&
            Object.keys(drafts).length === 0 &&
            Object.keys(currentParameterProblems.current).length === 0 &&
            executionIssues.length === 0
          )
            onRun();
        }}
      >
        {busy ? 'Please wait…' : 'Run analysis'}
      </WorkbenchButton>{' '}
      <WorkbenchButton disabled={busy} onClick={onCancel}>
        Back to questions
      </WorkbenchButton>
    </MappingSection>
  );
}
