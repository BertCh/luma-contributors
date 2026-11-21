// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUAnalysisCautionText,
  getGPUAnalysisQuestion,
  type GPUAnalysisFieldRoleSpec,
  type GPUAnalysisInterpretationSpec,
  type GPUAnalysisQuestion,
  type GPUAnalysisRowSemantics,
  type GPUSpatialAnalysisDataKind
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {KEPLER_CURATED_ANALYSES} from '../curated/curated-analyses';
import {getKeplerCuratedAnalysisSampleMetadata} from '../curated/curated-analysis-samples';
import type {KeplerCuratedAnalysis} from '../curated/curated-analysis-types';
import {getKeplerOperationDescriptor} from '../descriptors/operation-descriptors';
import type {KeplerOperationParameterDescriptor} from '../descriptors/operation-descriptor-types';
import {createWorkflowDocument} from './workflow-document';
import {promoteWorkflowParameter} from './workflow-edit';
import {getWorkflowOperationRegistry} from './workflow-operations';
import {parseWorkflowAnalysisTemplate} from './workflow-analysis-template-parser';
import type {WorkflowConnection, WorkflowDocument, WorkflowStep} from './workflow-types';

export const WORKFLOW_ANALYSIS_TEMPLATE_VERSION = 1;

/** A role writes column names to existing workflow parameters; it creates no new execution API. */
export type WorkflowAnalysisRoleSource = {stepId: string; parameter: string};

export type WorkflowAnalysisTemplateSlot = {
  name: string;
  label: string;
  kind: GPUSpatialAnalysisDataKind;
  rowSemantics?: readonly GPUAnalysisRowSemantics[];
  /** The source output replaced when the slot is mapped to another dataset. */
  source: WorkflowConnection;
  fields: readonly (GPUAnalysisFieldRoleSpec & {
    sources: readonly WorkflowAnalysisRoleSource[];
  })[];
};

export type WorkflowAnalysisTemplate = {
  schemaVersion: typeof WORKFLOW_ANALYSIS_TEMPLATE_VERSION;
  id: string;
  title: string;
  question: GPUAnalysisQuestion;
  method: string;
  curatedAnalysisId?: string;
  document: WorkflowDocument;
  slots: readonly WorkflowAnalysisTemplateSlot[];
  /** Sources deliberately retained as fixed inputs when saving selected reusable slots. */
  fixedSources?: readonly string[];
  exposed: readonly {
    /** Existing promoted workflow input, also usable by more than one step. */
    inputId: string;
    target: WorkflowAnalysisRoleSource;
    /** Descriptor policy kept verbatim: `input` names the target operation's input port. */
    defaultPolicy: KeplerOperationParameterDescriptor['defaultPolicy'];
  }[];
  /** A reference only: does not bundle datasets or imply that an unavailable recipe can run. */
  sample?: {id: string; license: string; attribution: string; reference: string};
  reads: {
    summary: string;
    cautions: readonly string[];
    /** Shared sentence/caution vocabulary for named outputs, evaluated from the actual run. */
    outputs?: readonly {
      source: WorkflowConnection;
      interpretation: GPUAnalysisInterpretationSpec;
    }[];
  };
};

function makeStep(id: string, operationId: string): WorkflowStep {
  return {id, operationId, inputs: {}, parameters: {}, position: {x: 0, y: 0}};
}

/** Result text comes from the same descriptors as the generated forms and result cards. */
function readDescriptorCautions(operationIds: readonly string[]): string[] {
  return [
    ...new Set(
      operationIds.flatMap(operationId => {
        const descriptor = getKeplerOperationDescriptor(operationId);
        return [
          ...(descriptor?.notes ?? []),
          ...(descriptor?.outputs.flatMap(
            output =>
              output.interpretation?.cautions.map(
                caution => getGPUAnalysisCautionText(caution).text
              ) ?? []
          ) ?? [])
        ];
      })
    )
  ];
}

function readDescriptorOutputs(
  steps: readonly WorkflowStep[]
): NonNullable<WorkflowAnalysisTemplate['reads']['outputs']> {
  return steps.flatMap(
    step =>
      getKeplerOperationDescriptor(step.operationId)?.outputs.flatMap(output =>
        output.interpretation
          ? [
              {
                source: {stepId: step.id, output: output.tableId},
                interpretation: output.interpretation
              }
            ]
          : []
      ) ?? []
  );
}

type WorkflowAnalysisTemplateField = WorkflowAnalysisTemplateSlot['fields'][number];

/** One mapped column must satisfy every consumer; incompatible constraints keep separate roles. */
function mergeTemplateFields(
  previous: WorkflowAnalysisTemplateField,
  incoming: WorkflowAnalysisTemplateField
): WorkflowAnalysisTemplateField | undefined {
  const required = previous.required === true || incoming.required === true;
  // Conditions name a parameter on their own consumer. Combining them across steps would
  // require a scoped OR expression, which the shared field-role contract does not represent.
  if (!required && (previous.required !== false || incoming.required !== false)) return undefined;
  const accepts = previous.accepts.filter(value => incoming.accepts.includes(value));
  const measurement =
    previous.measurement && incoming.measurement
      ? previous.measurement.filter(value => incoming.measurement!.includes(value))
      : (previous.measurement ?? incoming.measurement);
  const minimum = Math.max(previous.multiple?.minimum ?? 1, incoming.multiple?.minimum ?? 1);
  const maximum = Math.min(previous.multiple?.maximum ?? 1, incoming.multiple?.maximum ?? 1);
  if (!accepts.length || measurement?.length === 0 || minimum > maximum) return undefined;
  const sources = [...previous.sources];
  for (const source of incoming.sources) {
    if (
      !sources.some(
        previous => previous.stepId === source.stepId && previous.parameter === source.parameter
      )
    )
      sources.push(source);
  }
  return {
    ...previous,
    accepts,
    ...(measurement ? {measurement} : {}),
    required,
    ...(previous.multiple || incoming.multiple ? {multiple: {minimum, maximum}} : {}),
    sources
  };
}

/** Adapt a gallery document without changing its embedded samples or execution behavior. */
export function createWorkflowAnalysisTemplateFromDocument(metadata: {
  id: string;
  title: string;
  summary: string;
  document: WorkflowDocument;
}): WorkflowAnalysisTemplate {
  const {document} = metadata;
  const sources = document.steps.filter(step =>
    ['source-dataset', 'source-geojson', 'source-drawing'].includes(step.operationId)
  );
  const slots: WorkflowAnalysisTemplateSlot[] = sources.map(source => {
    // These operations retain geometry and column names. Follow their output ports to the
    // analysis descriptor that constrains the reusable source, guarding shared paths/cycles.
    const pending: WorkflowConnection[] = [{stepId: source.id, output: 'data'}];
    const visited = new Set<string>();
    const consumers: {
      step: WorkflowStep;
      descriptor: NonNullable<ReturnType<typeof getKeplerOperationDescriptor>>['inputs'][number];
    }[] = [];
    while (pending.length) {
      const output = pending.pop()!;
      const key = `${output.stepId}.${output.output}`;
      if (visited.has(key)) continue;
      visited.add(key);
      for (const step of document.steps) {
        for (const [input, connection] of Object.entries(step.inputs)) {
          if (connection.stepId !== output.stepId || connection.output !== output.output) continue;
          const descriptor = getKeplerOperationDescriptor(step.operationId)?.inputs.find(
            candidate => candidate.name === input
          );
          if (descriptor) consumers.push({step, descriptor});
          if (['table-filter-rows', 'table-sort-rows'].includes(step.operationId)) {
            for (const port of getWorkflowOperationRegistry().get(step.operationId)?.outputs ?? [])
              pending.push({stepId: step.id, output: port.name});
          }
        }
      }
    }
    const fields = new Map<string, WorkflowAnalysisTemplateSlot['fields'][number]>();
    for (const {step, descriptor} of consumers) {
      for (const field of descriptor?.fields ?? []) {
        const {
          parameter,
          measurementByGeometry: ignoredGeometry,
          autoAssign: ignoredAutoAssign,
          ...role
        } = field;
        const previous = fields.get(role.role);
        const reference = {stepId: step.id, parameter};
        const incoming = {...role, sources: [reference]};
        const merged = previous && mergeTemplateFields(previous, incoming);
        if (!previous || merged) fields.set(role.role, merged ?? incoming);
        else {
          const baseRole = `${role.role}-${step.id}`;
          let separateRole = baseRole;
          for (let suffix = 2; fields.has(separateRole); suffix++)
            separateRole = `${baseRole}-${suffix}`;
          fields.set(separateRole, {...incoming, role: separateRole});
        }
      }
    }
    return {
      name: source.id,
      label: source.label ?? source.id,
      kind: consumers.find(consumer => consumer.descriptor)?.descriptor?.kind ?? 'attributes',
      source: {stepId: source.id, output: 'data'},
      fields: [...fields.values()]
    };
  });
  const question =
    document.steps
      .map(step => getKeplerOperationDescriptor(step.operationId)?.questions[0])
      .find(candidate => candidate !== undefined) ?? 'prepare-data';
  return parseWorkflowAnalysisTemplate(
    JSON.stringify({
      schemaVersion: WORKFLOW_ANALYSIS_TEMPLATE_VERSION,
      id: metadata.id,
      title: metadata.title.endsWith('?') ? metadata.title : getGPUAnalysisQuestion(question).label,
      question,
      method: metadata.summary,
      document,
      slots,
      exposed: document.steps.flatMap(step =>
        Object.entries(step.parameters).flatMap(([parameter, binding]) =>
          binding.from === 'input'
            ? [
                {
                  inputId: binding.inputId,
                  target: {stepId: step.id, parameter},
                  // These ranges were authored for a fixed gallery sample, rather than a profiled dataset.
                  defaultPolicy: {from: 'constant'}
                }
              ]
            : []
        )
      ),
      sample: {
        id: metadata.id,
        license: 'MIT',
        attribution: 'vis.gl contributors',
        reference: 'modules/kepler/src/workflows/workflow-templates.ts'
      },
      reads: {
        summary: metadata.summary,
        cautions: readDescriptorCautions(document.steps.map(step => step.operationId)),
        outputs: readDescriptorOutputs(document.steps)
      }
    })
  );
}

/** Build the single-step document from the curated registry, including unavailable placeholders. */
export function createCuratedWorkflowAnalysisTemplate(
  analysis: KeplerCuratedAnalysis
): WorkflowAnalysisTemplate {
  const execution = analysis.execution;
  const operationId =
    execution.status === 'available' ? execution.bindingId : `curated-${analysis.id}`;
  let document = createWorkflowDocument({id: analysis.id, title: analysis.title});
  const operation = makeStep('analysis', operationId);
  const slots = analysis.slots.map(slot => {
    const source = makeStep(`source-${slot.name}`, 'source-dataset');
    // Symbolic dataset/geometry references are replaced by the host's slot mapping.
    source.parameters = {
      dataset: {from: 'static', value: slot.name},
      geometryMode: {
        from: 'static',
        value: slot.kind === 'surface' || slot.kind === 'attributes' ? 'none' : 'geojson-field'
      },
      ...(slot.kind === 'surface' || slot.kind === 'attributes'
        ? {}
        : {geometryField: {from: 'static' as const, value: 'geometry'}})
    };
    document.steps.push(source);
    const input = execution.status === 'available' ? execution.inputs[slot.name] : slot.name;
    if (input) operation.inputs[input] = {stepId: source.id, output: 'data'};
    return {
      ...slot,
      source: {stepId: source.id, output: 'data'},
      fields: slot.fields.map(field => {
        const parameter =
          execution.status === 'available'
            ? execution.fieldParameters[`${slot.name}.${field.role}`]
            : field.role;
        return {...field, sources: parameter ? [{stepId: operation.id, parameter}] : []};
      })
    };
  });
  if (execution.status === 'available') {
    operation.parameters = Object.fromEntries(
      Object.entries(execution.parameters).map(([name, value]) => [name, {from: 'static', value}])
    );
  }
  document.steps.push(operation);
  const descriptor = getKeplerOperationDescriptor(operationId);
  const exposed: WorkflowAnalysisTemplate['exposed'][number][] = [];
  for (const parameter of analysis.exposedParameters) {
    const promoted = promoteWorkflowParameter(
      document,
      getWorkflowOperationRegistry(),
      operation.id,
      parameter
    );
    document = promoted.document;
    const defaultPolicy = Object.hasOwn(
      execution.status === 'available' ? execution.parameters : {},
      parameter
    )
      ? {from: 'constant' as const}
      : (descriptor?.parameters.find(candidate => candidate.name === parameter)?.defaultPolicy ?? {
          from: 'constant' as const
        });
    // An unconditional descriptor preset is already the declared starting value. Data rules
    // remain unresolved until mapping supplies profiles; the promoted value is their fallback.
    if (defaultPolicy.from === 'preset' && defaultPolicy.whenGeometry === undefined) {
      document.inputs.find(input => input.id === promoted.inputId)!.value = defaultPolicy.value;
    }
    exposed.push({
      inputId: promoted.inputId,
      target: {stepId: operation.id, parameter},
      defaultPolicy
    });
  }
  return parseWorkflowAnalysisTemplate(
    JSON.stringify({
      schemaVersion: WORKFLOW_ANALYSIS_TEMPLATE_VERSION,
      id: analysis.id,
      curatedAnalysisId: analysis.id,
      title: analysis.title,
      question: analysis.question,
      method: analysis.method,
      document,
      slots,
      exposed,
      ...(analysis.sampleId
        ? {
            sample: getKeplerCuratedAnalysisSampleMetadata(analysis.sampleId)
          }
        : {}),
      reads: {
        summary: analysis.method,
        cautions:
          execution.status === 'needs-binding'
            ? [execution.reason]
            : readDescriptorCautions([operationId]),
        outputs: readDescriptorOutputs(document.steps)
      }
    })
  );
}

/** All shared curated records retain their identity, including records awaiting execution bindings. */
export function listCuratedWorkflowAnalysisTemplates(): readonly WorkflowAnalysisTemplate[] {
  return KEPLER_CURATED_ANALYSES.map(createCuratedWorkflowAnalysisTemplate);
}
