// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {KEPLER_CURATED_ANALYSES} from '../../src/curated/curated-analyses';
import {
  KEPLER_CURATED_ANALYSIS_SAMPLE_IDS,
  getKeplerCuratedAnalysisSampleMetadata
} from '../../src/curated/curated-analysis-samples';
import {getKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {
  mapWorkflowAnalysisTemplate,
  saveWorkflowAnalysisTemplate
} from '../../src/workflows/workflow-save-template';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {makeFixtureHost} from './workflow-fixtures';
import type {WorkflowDocument} from '../../src/workflows/workflow-types';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import {makeWorkflowDataFromFeatures} from '../../src/workflows/workflow-data';
import {addWorkflowStep, connectWorkflowSteps} from '../../src/workflows/workflow-edit';
import {getWorkflowOperationRegistry} from '../../src/workflows/workflow-operations';
import {
  createCuratedWorkflowAnalysisTemplate,
  createWorkflowAnalysisTemplateFromDocument,
  listCuratedWorkflowAnalysisTemplates,
  type WorkflowAnalysisTemplate
} from '../../src/workflows/workflow-analysis-template';
import {
  parseWorkflowAnalysisTemplate,
  serializeWorkflowAnalysisTemplate,
  validateWorkflowAnalysisTemplate,
  WORKFLOW_ANALYSIS_TEMPLATE_LIMITS
} from '../../src/workflows/workflow-analysis-template-parser';
import {
  listWorkflowAnalysisTemplates,
  listWorkflowTemplates
} from '../../src/workflows/workflow-templates';

function makeTemplate(identifier = 'density-surface'): WorkflowAnalysisTemplate {
  return createCuratedWorkflowAnalysisTemplate(
    KEPLER_CURATED_ANALYSES.find(analysis => analysis.id === identifier)!
  );
}

function readCodes(template: unknown): string[] {
  return validateWorkflowAnalysisTemplate(template).issues.map(issue => issue.code);
}

function replacePolicy(policy: unknown): unknown {
  const template = makeTemplate();
  return {...template, exposed: [{...template.exposed[0], defaultPolicy: policy}]};
}

/** Two independent analyses share the measured points, with the supplied consumer order. */
function adaptSharedPointsTemplate(operationIds: readonly string[]): WorkflowAnalysisTemplate {
  const registry = getWorkflowOperationRegistry();
  let document = makeTemplate().document;
  document.steps = document.steps.filter(step => step.operationId === 'source-dataset');
  document.inputs = [];
  document = addWorkflowStep(document, registry, {
    id: 'areas',
    operationId: 'source-dataset',
    parameters: {dataset: 'areas', geometryField: 'geometry'}
  }).document;
  for (const operationId of operationIds) {
    document = addWorkflowStep(document, registry, {id: operationId, operationId}).document;
    document = connectWorkflowSteps(document, registry, {
      from: {stepId: 'source-points', output: 'data'},
      to: {stepId: operationId, input: 'points'}
    });
    if (registry.get(operationId)!.inputs.some(input => input.name === 'polygons')) {
      document = connectWorkflowSteps(document, registry, {
        from: {stepId: 'areas', output: 'data'},
        to: {stepId: operationId, input: 'polygons'}
      });
    }
  }
  return createWorkflowAnalysisTemplateFromDocument({
    id: 'shared-points',
    title: 'Shared points',
    summary: 'Interpolate the measured values and summarise them by area.',
    document
  });
}

describe('AX workflow analysis template envelope', () => {
  test('builds all 28 curated records from the shared registry with unchanged roles and questions', () => {
    const templates = listCuratedWorkflowAnalysisTemplates();
    expect(templates).toHaveLength(28);
    expect(templates.map(template => template.id)).toEqual(
      KEPLER_CURATED_ANALYSES.map(analysis => analysis.id)
    );
    for (const [index, template] of templates.entries()) {
      const analysis = KEPLER_CURATED_ANALYSES[index];
      expect(template.question).toBe(analysis.question);
      expect(template.title).toBe(analysis.title);
      expect(template.method).toBe(analysis.method);
      expect(template.curatedAnalysisId).toBe(analysis.id);
      expect(
        template.slots.map(({source, fields, ...slot}) => ({
          ...slot,
          fields: fields.map(({sources, ...field}) => field)
        }))
      ).toEqual(JSON.parse(JSON.stringify(analysis.slots)));
      expect(template.reads.summary.length).toBeGreaterThan(20);
      expect(
        parseWorkflowAnalysisTemplate(JSON.stringify(serializeWorkflowAnalysisTemplate(template)))
      ).toEqual(template);
      const validation = validateWorkflowAnalysisTemplate(template);
      if (analysis.execution.status === 'available') {
        expect(validation.issues, `${analysis.id}: ${JSON.stringify(validation.issues)}`).toEqual(
          []
        );
        expect(validation.executable).toBe(true);
        expect(template.exposed.map(exposed => exposed.target.parameter)).toEqual(
          analysis.exposedParameters
        );
      } else {
        expect(validation.executable).toBe(false);
        expect(validation.issues.some(issue => issue.code === 'unknown-operation')).toBe(true);
        expect(template.reads.cautions).toContain(analysis.execution.reason);
      }
    }
  });

  test('adapts all existing gallery chains while preserving the six-document API', () => {
    const gallery = listWorkflowTemplates();
    expect(gallery).toHaveLength(6);
    for (const template of gallery) {
      expect(template.analysisTemplate.document).toEqual(template.document);
      expect(template.analysisTemplate.title.endsWith('?')).toBe(true);
      expect(
        validateWorkflowAnalysisTemplate(template.analysisTemplate).issues,
        JSON.stringify(validateWorkflowAnalysisTemplate(template.analysisTemplate).issues)
      ).toEqual([]);
      expect(template.analysisTemplate.slots).toHaveLength(
        template.document.steps.filter(step => step.operationId === 'source-geojson').length
      );
    }
    const templates = listWorkflowAnalysisTemplates();
    expect(templates).toHaveLength(34);
    expect(new Set(templates.map(template => template.id)).size).toBe(34);
  });

  test('retains descriptor default policies and binding presets rather than inventing new rules', () => {
    const template = makeTemplate();
    const descriptor = getKeplerOperationDescriptor('kernel-density-grid')!;
    expect(template.exposed[0].defaultPolicy).toEqual(
      descriptor.parameters.find(parameter => parameter.name === 'bandwidth')!.defaultPolicy
    );
    expect(
      template.exposed.find(parameter => parameter.target.parameter === 'kernel')!.defaultPolicy
    ).toEqual({from: 'constant'});
    expect(template.exposed.some(parameter => parameter.defaultPolicy.from === 'data')).toBe(true);
    expect(makeTemplate('hot-spot-map').reads.cautions.length).toBeGreaterThan(0);
  });

  test('result readings reuse shared interpretation records and validate their output references', () => {
    const template = makeTemplate('hot-spot-map');
    const outputs = template.reads.outputs!;
    expect(outputs.length).toBeGreaterThan(0);
    const descriptor = getKeplerOperationDescriptor('recipe-hot-spot-lattice')!;
    for (const output of outputs) {
      expect(output.interpretation).toEqual(
        descriptor.outputs.find(candidate => candidate.tableId === output.source.output)!
          .interpretation
      );
    }
    const output = outputs[0];
    const replaceOutput = (changed: unknown) => ({
      ...template,
      reads: {...template.reads, outputs: [changed]}
    });
    expect(
      readCodes(replaceOutput({...output, source: {...output.source, stepId: 'missing'}}))
    ).toContain('unknown-step');
    expect(
      readCodes(replaceOutput({...output, source: {...output.source, output: 'missing'}}))
    ).toContain('unknown-output-port');
    expect(() =>
      parseWorkflowAnalysisTemplate(
        replaceOutput({...output, interpretation: {summary: 'invented', cautions: []}})
      )
    ).toThrow();
    expect(() =>
      parseWorkflowAnalysisTemplate(
        replaceOutput({...output, interpretation: {summary: 'value-range', cautions: ['invented']}})
      )
    ).toThrow();
  });

  test('references only licensed repository synthetic samples when a sample is present', () => {
    for (const template of listCuratedWorkflowAnalysisTemplates()) {
      if (!template.sample) continue;
      expect(KEPLER_CURATED_ANALYSIS_SAMPLE_IDS).toContain(template.sample.id);
      expect(template.sample.license).toBe('MIT');
      expect(template.sample.attribution).toContain('vis.gl contributors');
      expect(template.sample.reference).toBe(
        getKeplerCuratedAnalysisSampleMetadata(template.sample.id).reference
      );
    }
    expect(makeTemplate('drainage-network').sample?.id).toBe('drainage-network-slope');
  });

  test('loadable unknown operations are rejected only by executable validation', () => {
    const template = makeTemplate();
    template.document.steps.find(step => step.id === 'analysis')!.operationId = 'future-analysis';
    const parsed = parseWorkflowAnalysisTemplate(template);
    expect(parsed.document.steps.at(-1)!.operationId).toBe('future-analysis');
    expect(readCodes(parsed)).toContain('unknown-operation');
    expect(
      parseWorkflowAnalysisTemplate(JSON.stringify(serializeWorkflowAnalysisTemplate(parsed)))
    ).toEqual(parsed);
  });

  test('missing dataset slots fail executable validation with the source identity', () => {
    const template = {...makeTemplate(), slots: []};
    expect(parseWorkflowAnalysisTemplate(template).slots).toEqual([]);
    expect(validateWorkflowAnalysisTemplate(template)).toMatchObject({
      executable: false,
      issues: expect.arrayContaining([
        expect.objectContaining({
          code: 'missing-slot',
          message: expect.stringContaining('source-points')
        })
      ])
    });
  });

  test('cycles and missing graph references use the existing planner', () => {
    const template = makeTemplate();
    const step = template.document.steps.find(step => step.id === 'analysis')!;
    const second = {
      ...step,
      id: 'second',
      inputs: {points: {stepId: 'analysis', output: 'density'}}
    };
    template.document.steps.push(second);
    step.inputs['points'] = {stepId: 'second', output: 'density'};
    expect(() => parseWorkflowAnalysisTemplate(template)).not.toThrow();
    expect(readCodes(template)).toContain('cycle');
    template.document.steps.pop();
    step.inputs['points'] = {stepId: 'missing', output: 'data'};
    expect(readCodes(template)).toContain('unknown-step');
    step.inputs['points'] = {stepId: 'source-points', output: 'missing'};
    expect(readCodes(template)).toContain('unknown-output-port');
  });

  test('invalid slot output, duplicate source and non-source step are diagnosed', () => {
    const template = makeTemplate();
    const slot = template.slots[0];
    expect(
      readCodes({...template, slots: [{...slot, source: {...slot.source, output: 'missing'}}]})
    ).toContain('unknown-output-port');
    expect(readCodes({...template, slots: [slot, {...slot, name: 'duplicate'}]})).toContain(
      'duplicate-slot-source'
    );
    expect(
      readCodes({...template, slots: [{...slot, source: {stepId: 'analysis', output: 'density'}}]})
    ).toContain('invalid-slot-source');
  });

  test('role parameters must exist and consume the referenced slot', () => {
    const template = makeTemplate();
    const slot = template.slots[0];
    const field = slot.fields[0];
    const change = (sources: typeof field.sources) => ({
      ...template,
      slots: [{...slot, fields: [{...field, required: true, sources}]}]
    });
    expect(readCodes(change([]))).toContain('missing-role-source');
    expect(readCodes(change([{stepId: 'missing', parameter: 'weightProperty'}]))).toContain(
      'unknown-step'
    );
    expect(readCodes(change([{stepId: 'analysis', parameter: 'missing'}]))).toContain(
      'unknown-parameter'
    );
    expect(readCodes(change([{stepId: 'analysis', parameter: 'bandwidth'}]))).toContain(
      'invalid-role-parameter'
    );
    expect(readCodes(change([{stepId: 'source-points', parameter: 'dataset'}]))).toContain(
      'invalid-role-parameter'
    );
    // Use a real two-source gallery chain so an unrelated target remains a valid graph node.
    const gallery = listWorkflowTemplates().find(
      template => template.id === 'nearest-facilities'
    )!.analysisTemplate;
    const firstSlot = gallery.slots[0];
    const unrelated = {
      ...gallery,
      slots: [
        {...firstSlot, fields: [{...field, sources: [{stepId: 'facilities', parameter: 'name'}]}]},
        ...gallery.slots.slice(1)
      ]
    };
    expect(readCodes(unrelated)).toContain('unrelated-role-source');
  });

  test('exposed parameters must address their promoted input exactly', () => {
    const template = makeTemplate();
    expect(
      readCodes({...template, exposed: [{...template.exposed[0], inputId: 'absent'}]})
    ).toContain('invalid-exposed-input');
    expect(
      readCodes({
        ...template,
        exposed: [{...template.exposed[0], target: {stepId: 'analysis', parameter: 'missing'}}]
      })
    ).toContain('unknown-parameter');
    expect(
      readCodes(replacePolicy({from: 'data', input: 'absent', rule: 'standard-distance'}))
    ).toContain('unknown-input-port');
    expect(
      readCodes(
        replacePolicy({from: 'data', input: 'points', rule: 'column-variance', role: 'absent'})
      )
    ).toContain('unknown-role');
    expect(
      readCodes(replacePolicy({from: 'preset', value: -1, reason: 'invalid bandwidth'}))
    ).toContain('invalid-default-value');
  });

  test('default factor references and cycles are validated', () => {
    expect(
      readCodes(
        replacePolicy({
          from: 'data',
          input: 'points',
          rule: 'standard-distance',
          factorByParameter: {parameter: 'absent', factors: {gaussian: 1}}
        })
      )
    ).toContain('unknown-parameter');
    expect(
      readCodes(
        replacePolicy({
          from: 'data',
          input: 'points',
          rule: 'standard-distance',
          factorByParameter: {parameter: 'bandwidth', factors: {gaussian: 1}}
        })
      )
    ).toContain('default-cycle');
    const template = makeTemplate();
    const exposed = template.exposed.slice(0, 2).map((parameter, index, parameters) => ({
      ...parameter,
      defaultPolicy: {
        from: 'data' as const,
        input: 'points',
        rule: 'standard-distance' as const,
        factorByParameter: {
          parameter: parameters[1 - index].target.parameter,
          factors: {gaussian: 1}
        }
      }
    }));
    expect(readCodes({...template, exposed})).toContain('default-cycle');
  });

  test('field-measurement values and preset geometry inputs are checked', () => {
    expect(
      readCodes(
        replacePolicy({
          from: 'field-measurement',
          input: 'points',
          role: 'weight',
          values: {count: 100}
        })
      )
    ).toEqual([]);
    expect(
      readCodes(
        replacePolicy({
          from: 'field-measurement',
          input: 'points',
          role: 'weight',
          values: {count: -1}
        })
      )
    ).toContain('invalid-default-value');
    expect(
      readCodes(
        replacePolicy({
          from: 'preset',
          input: 'missing',
          whenGeometry: 'point',
          value: 100,
          reason: 'constant example'
        })
      )
    ).toContain('unknown-input-port');
  });

  test('declared slot geometry participates in the existing port validator', () => {
    const template = makeTemplate();
    const slots = [{...template.slots[0], kind: 'polygons' as const}];
    expect(readCodes({...template, slots})).toContain('incompatible-port');
  });

  test('roles follow the correct input and shared targets require the same role', () => {
    const template = makeTemplate('count-by-area');
    const slot = template.slots[0];
    const field = slot.fields[0];
    const wrongSlot = {...template.slots[1], fields: [field]};
    expect(readCodes({...template, slots: [slot, wrongSlot]})).toContain('unrelated-role-source');
    expect(readCodes(makeTemplate('period-comparison'))).toEqual([]);
    const secondField = {...field, role: 'another-role'};
    expect(
      readCodes({
        ...template,
        slots: [{...slot, fields: [field, secondField]}, ...template.slots.slice(1)]
      })
    ).toContain('conflicting-role-source');
  });

  test('role conditions refer to valid controlling parameters and accepted values', () => {
    const template = makeTemplate('count-by-area');
    const slot = template.slots[0];
    const field = slot.fields[0];
    const replaceCondition = (whenParameter: string, equals: readonly string[]) => ({
      ...template,
      slots: [
        {...slot, fields: [{...field, required: {whenParameter, equals}}]},
        ...template.slots.slice(1)
      ]
    });
    expect(readCodes(replaceCondition('missing', ['count']))).toContain('unknown-parameter');
    expect(readCodes(replaceCondition('statistic', ['invalid-statistic']))).toContain(
      'invalid-role-condition'
    );
    expect(readCodes(replaceCondition('statistic', ['count', 'sum']))).toEqual([]);
  });

  test('round-trips shared role conditions and validates every nested comparison', () => {
    const template = makeTemplate('count-by-area');
    const slot = template.slots[0];
    const field = slot.fields[0];
    const replaceCondition = (required: typeof field.required): WorkflowAnalysisTemplate => ({
      ...template,
      slots: [{...slot, fields: [{...field, required}]}, ...template.slots.slice(1)]
    });
    for (const required of [
      {parameter: 'statistic', equals: ['sum']},
      {
        anyOf: [
          {parameter: 'statistic', equals: ['sum']},
          {anyOf: [{whenParameter: 'statistic', equals: ['mean']}]}
        ]
      }
    ]) {
      const changed = replaceCondition(required);
      expect(
        parseWorkflowAnalysisTemplate(JSON.stringify(serializeWorkflowAnalysisTemplate(changed)))
      ).toEqual(changed);
      expect(readCodes(changed)).toEqual([]);
    }
    const issues = validateWorkflowAnalysisTemplate(
      replaceCondition({
        anyOf: [
          {parameter: 'statistic', equals: ['sum']},
          {
            anyOf: [
              {whenParameter: 'missing', equals: ['count']},
              {parameter: 'statistic', equals: ['invalid-statistic', false, 3]}
            ]
          }
        ]
      })
    ).issues;
    expect(issues.filter(issue => issue.code === 'unknown-parameter')).toHaveLength(1);
    expect(issues.filter(issue => issue.code === 'invalid-role-condition')).toHaveLength(3);
    expect(issues.find(issue => issue.code === 'unknown-parameter')?.message).toContain('missing');
  });

  test.each([
    {accepts: ['unknown']},
    {accepts: []},
    {measurement: ['unknown']},
    {required: {whenParameter: 'statistic', equals: []}},
    {required: {parameter: 'statistic', equals: []}},
    {required: {anyOf: []}},
    {required: {anyOf: [{parameter: 'statistic', equals: ['sum']}, {anyOf: []}]}},
    {required: {anyOf: [{parameter: 'statistic', equals: [null]}]}},
    {required: {parameter: 'statistic', whenParameter: 'statistic', equals: ['sum']}},
    {required: {anyOf: [{parameter: 'statistic', equals: ['sum']}], equals: ['sum']}},
    {multiple: {minimum: 3, maximum: 2}},
    {sources: [{stepId: 'analysis', parameter: 'weightProperty', unknown: true}]}
  ])('rejects malformed field metadata %j', changed => {
    const template = makeTemplate();
    const slot = template.slots[0];
    expect(() =>
      parseWorkflowAnalysisTemplate({
        ...template,
        slots: [{...slot, fields: [{...slot.fields[0], ...changed}]}]
      })
    ).toThrow();
  });

  test('descriptor presets initialize promoted inputs and survive serialization', () => {
    for (const template of listCuratedWorkflowAnalysisTemplates()) {
      for (const exposed of template.exposed) {
        if (
          exposed.defaultPolicy.from === 'preset' &&
          exposed.defaultPolicy.whenGeometry === undefined
        ) {
          expect(
            template.document.inputs.find(input => input.id === exposed.inputId)!.value
          ).toEqual(exposed.defaultPolicy.value);
        }
      }
    }
  });

  test('policy values must fit the promoted input range as well as the operation range', () => {
    const template = makeTemplate();
    const input = template.document.inputs.find(input => input.id === template.exposed[0].inputId)!;
    input.maximum = 150;
    input.value = 100;
    const exposed = [
      {
        ...template.exposed[0],
        defaultPolicy: {from: 'preset' as const, value: 200, reason: 'Outside promoted range'}
      }
    ];
    expect(readCodes({...template, exposed})).toContain('invalid-default-value');
  });

  test('serialization omits undefined TypeScript optionals while parsing stays strict', () => {
    const template = {...makeTemplate(), sample: undefined};
    expect(() => parseWorkflowAnalysisTemplate(template)).toThrow();
    const serialized = serializeWorkflowAnalysisTemplate(template);
    expect(Object.hasOwn(serialized, 'sample')).toBe(false);
    expect(parseWorkflowAnalysisTemplate(JSON.stringify(serialized))).toEqual(serialized);
  });

  test('parsing and serialization defensively copy documents, roles and policies', () => {
    const original = makeTemplate();
    const copy = serializeWorkflowAnalysisTemplate(original);
    expect(copy).toEqual(original);
    expect(copy).not.toBe(original);
    expect(copy.document).not.toBe(original.document);
    expect(copy.slots[0].fields[0]).not.toBe(original.slots[0].fields[0]);
    expect(copy.exposed[0].defaultPolicy).not.toBe(original.exposed[0].defaultPolicy);
    expect(JSON.stringify(serializeWorkflowAnalysisTemplate(copy))).toBe(JSON.stringify(copy));
  });

  test.each([
    ['template member', (template: WorkflowAnalysisTemplate) => ({...template, unexpected: true})],
    [
      'question',
      (template: WorkflowAnalysisTemplate) => ({...template, question: 'unknown-question'})
    ],
    [
      'slot kind',
      (template: WorkflowAnalysisTemplate) => ({
        ...template,
        slots: [{...template.slots[0], kind: 'unknown-kind'}]
      })
    ],
    ['missing reads', (template: WorkflowAnalysisTemplate) => ({...template, reads: undefined})],
    [
      'empty summary',
      (template: WorkflowAnalysisTemplate) => ({...template, reads: {summary: '', cautions: []}})
    ],
    [
      'wrong caution',
      (template: WorkflowAnalysisTemplate) => ({
        ...template,
        reads: {summary: 'Result', cautions: [4]}
      })
    ],
    [
      'reads member',
      (template: WorkflowAnalysisTemplate) => ({
        ...template,
        reads: {...template.reads, extra: true}
      })
    ],
    [
      'sample license',
      (template: WorkflowAnalysisTemplate) => ({
        ...template,
        sample: {...template.sample, license: ''}
      })
    ],
    [
      'duplicate slots',
      (template: WorkflowAnalysisTemplate) => ({
        ...template,
        slots: [...template.slots, template.slots[0]]
      })
    ],
    [
      'duplicate exposed',
      (template: WorkflowAnalysisTemplate) => ({
        ...template,
        exposed: [...template.exposed, template.exposed[0]]
      })
    ]
  ])('rejects malformed %s', (_name, transform) => {
    expect(() => parseWorkflowAnalysisTemplate(transform(makeTemplate()))).toThrow();
    expect(validateWorkflowAnalysisTemplate(transform(makeTemplate())).executable).toBe(false);
  });

  test.each([
    {from: 'data', input: 'points', rule: 'made-up'},
    {from: 'data', input: 'points', rule: 'column-variance'},
    {from: 'constant', invented: 1},
    {from: 'data', input: 'points', rule: 'standard-distance', fallback: {rule: 'made-up'}},
    {from: 'data', input: 'points', rule: 'standard-distance', fallback: {rule: 'column-variance'}},
    {from: 'preset', value: 10},
    {from: 'preset', value: 10, reason: 'test', whenGeometry: 'point'},
    {from: 'field-measurement', input: 'points', role: 'weight', values: {madeUp: 1}}
  ])('rejects malformed policy $from', policy => {
    expect(() => parseWorkflowAnalysisTemplate(replacePolicy(policy))).toThrow();
  });

  test('rejects accessor, inherited, special, sparse, non-finite and cyclic object data safely', () => {
    let invoked = false;
    const accessor = {...makeTemplate()};
    Object.defineProperty(accessor, 'method', {
      enumerable: true,
      get() {
        invoked = true;
        return 'unsafe';
      }
    });
    expect(() => parseWorkflowAnalysisTemplate(accessor)).toThrow(/own data field/);
    expect(invoked).toBe(false);
    expect(() =>
      parseWorkflowAnalysisTemplate(Object.assign(Object.create({extra: true}), makeTemplate()))
    ).toThrow(/inherited/);
    expect(() => parseWorkflowAnalysisTemplate('{"__proto__":{}}')).toThrow(/special/);
    expect(() => parseWorkflowAnalysisTemplate({...makeTemplate(), slots: new Array(2)})).toThrow(
      /sparse/
    );
    expect(() =>
      parseWorkflowAnalysisTemplate(replacePolicy({from: 'preset', value: NaN, reason: 'invalid'}))
    ).toThrow();
    const cyclic: Record<string, unknown> = {...makeTemplate()};
    cyclic['cycle'] = cyclic;
    expect(() => parseWorkflowAnalysisTemplate(cyclic)).toThrow(/cyclic/);
  });

  test('rejects a weight role that writes a source label instead of a column', () => {
    const template = makeTemplate();
    const slot = template.slots[0];
    const slots = [
      {
        ...slot,
        fields: [
          {
            ...slot.fields[0],
            required: true,
            sources: [{stepId: 'source-points', parameter: 'datasetLabel'}]
          }
        ]
      }
    ];
    expect(validateWorkflowAnalysisTemplate({...template, slots})).toMatchObject({
      executable: false,
      issues: expect.arrayContaining([expect.objectContaining({code: 'invalid-role-parameter'})])
    });
  });

  test('accepts a real column parameter on a geometry-free table and checks its input provenance', () => {
    const template = makeTemplate();
    const slot = template.slots[0];
    const consumer = template.document.steps.find(step => step.id === 'analysis')!;
    consumer.operationId = 'table-sort-rows';
    consumer.inputs = {rows: {...slot.source}};
    consumer.parameters = {
      column: {from: 'static', value: 'severity'},
      order: {from: 'static', value: 'ascending'}
    };
    template.document.inputs = [];
    const tableTemplate: WorkflowAnalysisTemplate = {
      ...template,
      exposed: [],
      reads: {summary: 'Sort the severity values.', cautions: []},
      slots: [
        {
          ...slot,
          kind: 'attributes',
          fields: [{...slot.fields[0], sources: [{stepId: consumer.id, parameter: 'column'}]}]
        }
      ]
    };
    expect(
      getWorkflowOperationRegistry()
        .get('table-sort-rows')!
        .parameters.find(parameter => parameter.name === 'column')!.kind
    ).toBe('column');
    expect(validateWorkflowAnalysisTemplate(tableTemplate).issues).toEqual([]);
    tableTemplate.document = addWorkflowStep(
      tableTemplate.document,
      getWorkflowOperationRegistry(),
      {
        id: 'other',
        operationId: 'source-dataset',
        parameters: {dataset: 'other', geometryMode: 'none'}
      }
    ).document;
    tableTemplate.document.steps.find(step => step.id === consumer.id)!.inputs['rows'] = {
      stepId: 'other',
      output: 'data'
    };
    const slots = [
      ...tableTemplate.slots,
      {
        ...tableTemplate.slots[0],
        name: 'other',
        source: {stepId: 'other', output: 'data'},
        fields: []
      }
    ];
    expect(readCodes({...tableTemplate, slots})).toContain('unrelated-role-source');
  });

  test.each([
    'attributes',
    'summary',
    'pairs',
    'regions'
  ] as const)('rejects geometry-free %s slots connected to a point input', kind => {
    const template = makeTemplate();
    const slots = [{...template.slots[0], kind}];
    expect(validateWorkflowAnalysisTemplate({...template, slots})).toMatchObject({
      executable: false,
      issues: expect.arrayContaining([expect.objectContaining({code: 'incompatible-port'})])
    });
  });

  test.each([
    ['idw-grid', 'zonal-point-statistics'],
    ['zonal-point-statistics', 'idw-grid']
  ])('merges required measured-value constraints regardless of consumer order %j', (...operationIds) => {
    const template = adaptSharedPointsTemplate(operationIds);
    const value = template.slots
      .find(slot => slot.name === 'source-points')!
      .fields.find(field => field.role === 'value')!;
    expect(value.required).toBe(true);
    expect(value.measurement).toEqual(['amount', 'rate']);
    expect(value.sources).toEqual(
      operationIds.map(stepId => ({stepId, parameter: 'valueProperty'}))
    );
    expect(validateWorkflowAnalysisTemplate(template).issues).toEqual([]);
  });

  test('retains separate roles when consumer conditions cannot be applied to both operations', () => {
    const template = adaptSharedPointsTemplate([
      'recipe-points-in-polygons-choropleth',
      'zonal-point-statistics'
    ]);
    const fields = template.slots.find(slot => slot.name === 'source-points')!.fields;
    expect(fields).toHaveLength(2);
    expect(fields[0].required).toMatchObject({whenParameter: 'statistic'});
    expect(fields[1].required).toBe(false);
    expect(fields[0].role).not.toBe(fields[1].role);
    expect(fields.map(field => field.sources)).toEqual([
      [{stepId: 'recipe-points-in-polygons-choropleth', parameter: 'valueProperty'}],
      [{stepId: 'zonal-point-statistics', parameter: 'valueProperty'}]
    ]);
    expect(validateWorkflowAnalysisTemplate(template).issues).toEqual([]);
  });

  test('uses serialized size for object parsing and serialization, including escaped cautions', () => {
    const template = makeTemplate();
    const cautions = Array.from({length: 22}, (_, index) => '\0'.repeat(16380) + index);
    const oversized = {...template, reads: {...template.reads, cautions}};
    expect(JSON.stringify(oversized).length).toBeGreaterThan(
      WORKFLOW_ANALYSIS_TEMPLATE_LIMITS.maximumCharacters
    );
    expect(() => parseWorkflowAnalysisTemplate(oversized)).toThrow(/maximumCharacters/);
    expect(() => serializeWorkflowAnalysisTemplate(oversized)).toThrow(/maximumCharacters/);
    const withinLimit = {...template, reads: {...template.reads, cautions: cautions.slice(0, 21)}};
    const serialized = serializeWorkflowAnalysisTemplate(withinLimit);
    expect(JSON.stringify(serialized).length).toBeLessThanOrEqual(
      WORKFLOW_ANALYSIS_TEMPLATE_LIMITS.maximumCharacters
    );
    expect(parseWorkflowAnalysisTemplate(JSON.stringify(serialized))).toEqual(serialized);
  });

  test('enforces named envelope limits and rejects invalid JSON', () => {
    expect(() => parseWorkflowAnalysisTemplate('x')).toThrow(/valid JSON/);
    expect(() =>
      parseWorkflowAnalysisTemplate(
        ' '.repeat(WORKFLOW_ANALYSIS_TEMPLATE_LIMITS.maximumCharacters + 1)
      )
    ).toThrow(/maximumCharacters/);
    expect(() =>
      parseWorkflowAnalysisTemplate({
        ...makeTemplate(),
        method: 'x'.repeat(WORKFLOW_ANALYSIS_TEMPLATE_LIMITS.maximumTextCharacters + 1)
      })
    ).toThrow(/maximumTextCharacters/);
  });
});

test('infers point source slots through filters and sorts, then maps and runs a second dataset', async () => {
  const registry = getWorkflowOperationRegistry();
  const original = makeTemplate().document;
  let document: WorkflowDocument = {
    ...original,
    inputs: [],
    steps: original.steps.filter(step => step.operationId === 'source-dataset')
  };
  const sourceId = document.steps[0].id;
  const filter = addWorkflowStep(document, registry, {
    operationId: 'table-filter-rows',
    parameters: {expression: 'true'},
    connectFrom: {stepId: sourceId, output: 'data'},
    connectTo: 'rows'
  });
  const sort = addWorkflowStep(filter.document, registry, {
    operationId: 'table-sort-rows',
    parameters: {column: 'renamed_value'},
    connectFrom: {stepId: filter.stepId, output: 'filtered'},
    connectTo: 'rows'
  });
  document = addWorkflowStep(sort.document, registry, {
    operationId: 'kernel-density-grid',
    parameters: {cellSize: 1, bandwidth: 2},
    connectFrom: {stepId: sort.stepId, output: 'sorted'},
    connectTo: 'points'
  }).document;
  const inferred = createWorkflowAnalysisTemplateFromDocument({
    id: 'filtered-density',
    title: 'Where are the points concentrated?',
    summary: 'Count point events after a preserving table chain.',
    document
  });
  expect(inferred.slots[0].kind).toBe('points');
  const saved = saveWorkflowAnalysisTemplate(
    document,
    {
      id: inferred.id,
      title: inferred.title,
      question: inferred.question,
      method: inferred.method,
      slots: inferred.slots,
      exposed: []
    },
    registry
  );
  const collection = {
    type: 'FeatureCollection' as const,
    features: Array.from({length: 12}, (_, index) => ({
      type: 'Feature' as const,
      geometry: {type: 'Point', coordinates: [index % 4, Math.floor(index / 4)]},
      properties: {renamed_value: index + 1}
    }))
  };
  const profile = profileKeplerFeatureCollection(collection, {
    datasetId: 'second-points',
    label: 'Second points',
    coordinateSystem: 'planar',
    declarations: {rowSemantics: 'events'}
  });
  const mapped = mapWorkflowAnalysisTemplate(
    saved,
    {datasets: {[saved.slots[0].name]: profile}, columns: {}},
    registry
  );
  const data = makeWorkflowDataFromFeatures(collection, {
    id: profile.datasetId,
    label: profile.label
  });
  data.analysisProfile = profile;
  const host = makeFixtureHost(
    [
      {
        id: profile.datasetId,
        label: profile.label,
        rowCount: 12,
        resident: false,
        fields: [],
        geometryFields: ['geometry']
      }
    ],
    {[profile.datasetId]: data}
  );
  host.getAnalysisProfile = () => profile;
  const runner = new WorkflowRunner({host, registry});
  try {
    runner.setDocument(mapped);
    const record = await runner.run();
    expect(record.status, JSON.stringify(record)).toBe('done');
  } finally {
    await runner.destroy();
  }
});
