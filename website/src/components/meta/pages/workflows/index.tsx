// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import {formatResourceBytes, formatResourceCount, formatCount} from '../../../../lens/format';
import {Section} from '../../../ui/section';
import {findExampleLines} from '../../examples';
import {
  CodeWalk,
  type CodeWalkStep,
  ExampleCode,
  Figure,
  InShort,
  Prose,
  StatStrip,
  Term,
  TopicShell,
  type TryItLink
} from '../../page-kit';
import type {TopicExample, TopicPageProps} from '../../topic-types';
import {ChainFigure} from './chain-figure';
import {OneGraphFigure} from './one-graph-figure';
import {
  LOCAL_MORAN_RECORD,
  type Workflow,
  WORKFLOWS,
  type WorkflowRecord,
  WORKFLOW_TEMPLATES
} from './workflows';
import './workflows.css';

const SCENARIO_ROOT = 'modules/lenses/src/computation-lenses/scenarios';

const TRY_IT: readonly TryItLink[] = [
  ...WORKFLOWS.map(workflow => ({kind: 'function' as const, id: workflow.recipeEntry})),
  {kind: 'reference', slug: 'api/gpu-spatial-analysis/recipes', label: 'Recipes'},
  {kind: 'topic', slug: 'anatomy', note: 'one operation, line by line'}
];

function describeRecord(record: WorkflowRecord): string {
  return `${formatCount(record.nodeCount)} graph nodes, ${formatCount(record.dependencies)} dependencies, ${formatResourceCount({count: record.buffers, inventoryKnown: record.inventoryKnown})} of ${formatResourceBytes(record.bytes, record.bytesKind)}`;
}

function WorkflowBlock({workflow, example}: {workflow: Workflow; example?: TopicExample}) {
  const required = workflow.steps.filter(step => !step.optional).length;
  const optional = workflow.steps.length - required;
  const adapters = workflow.steps.filter(step => !step.entry).length;
  return (
    <article className="workflows-block" id={`workflow-${workflow.id}`}>
      <h3 className="workflows-block__title">{workflow.title}</h3>
      <p className="workflows-block__question">{workflow.question}</p>
      <p>{workflow.template.method}</p>
      <p>
        {workflow.template.execution.status === 'available'
          ? `Curated binding: ${workflow.template.execution.bindingId}.`
          : workflow.template.execution.reason}
      </p>
      <h4>Related recipe recording</h4>
      {workflow.recordNote ? <p>{workflow.recordNote}</p> : null}
      <p className="workflows-block__start">
        <span className="workflows-block__tag">You start with</span>
        {workflow.start}
      </p>
      <Figure
        notice={`${required} stages ran in the recorded graph${optional === 1 ? '; the faded one runs only with its option' : optional ? `; the ${optional} faded ones run only with their option` : ''}.`}
        caption={
          adapters
            ? 'Cards without a picture are kernels the recipe adds itself to fit one output to the next input; small bookkeeping kernels are left out.'
            : 'Every stage is a published contributor with its own function page.'
        }
      >
        <ChainFigure
          steps={workflow.steps}
          label={`${workflow.title}: the steps of ${workflow.recipe} in the order it adds them`}
        />
      </Figure>
      {example ? <ExampleCode example={example} label={workflow.exampleSource} /> : null}
      <dl className="workflows-block__facts">
        <div>
          <dt>Recorded</dt>
          <dd>
            {describeRecord(workflow.record)}
            {workflow.recordNote ? `; ${workflow.recordNote}` : ''}.
          </dd>
        </div>
        <div>
          <dt>Reports</dt>
          <dd>
            <code>status.stages</code>:{' '}
            {workflow.stages.map((stage, index) => (
              <span key={stage}>
                {index ? ', ' : ''}
                <code>{stage}</code>
              </span>
            ))}
          </dd>
        </div>
        <div>
          <dt>You choose</dt>
          <dd>{workflow.choice}</dd>
        </div>
      </dl>
    </article>
  );
}

function getRecipeResultSteps(code: string): CodeWalkStep[] {
  return [
    {
      id: 'contributors',
      title: 'contributors',
      text: 'The contributors the recipe added, in execution order; each adapter stage added with addGlue counts as one.',
      lines: findExampleLines(code, 'contributors:')
    },
    {
      id: 'outputs',
      title: 'outputs',
      text: 'The named results. They are caller-owned when you passed views for them in the recipe’s outputs; only those can be read back.',
      lines: findExampleLines(code, 'outputs: Outputs')
    },
    {
      id: 'intermediates',
      title: 'intermediates',
      text: 'Ports between stages, for drawing or the next chain. Graph-owned, so not readable on the CPU.',
      lines: findExampleLines(code, 'intermediates: Intermediates')
    },
    {
      id: 'status',
      title: 'status.stages',
      text: 'One entry per stage that can overflow, fall short or fail to converge: GPU-written uint32 words named by the stage.',
      lines: findExampleLines(code, 'export type GPURecipeStageStatusPort', '};')
    }
  ];
}

export function Page({topic, code}: TopicPageProps) {
  const localMoran = code['local-moran'];
  const recipeResult = code['recipe-result'];
  const [activeField, setActiveField] = useState('outputs');
  // One workflow is open at a time; the first is in the prerendered page.
  const [workflowId, setWorkflowId] = useState(WORKFLOWS[0].id);
  const openWorkflow = WORKFLOWS.find(workflow => workflow.id === workflowId) ?? WORKFLOWS[0];
  // Contributors in the chain: one graph.add per contributor in the example.
  const contributorCount = localMoran ? localMoran.code.split('graph.add(').length - 1 : 0;
  return (
    <TopicShell topic={topic} tryIt={TRY_IT}>
      <InShort
        points={[
          <>
            A <Term id="contributor" /> takes <Term id="view">views</Term> and writes views. Pass
            one contributor’s outputs as the next one’s inputs and you have a chain.
          </>,
          <>
            The whole chain is one <Term id="command-graph" />, compiled once. Intermediates stay on
            the GPU; only what you <Term id="readback">read back</Term> leaves it.
          </>,
          <>
            A <Term id="recipe" /> is a function that adds a standard chain for you and returns its
            outputs and a per-stage <Term id="status" />.
          </>
        ]}
      />

      <Section title="One graph">
        <Prose>
          Local Moran’s I needs neighbours, and the neighbours need row-standardized weights. You
          add three contributors to one graph and hand each the views the previous one wrote.
        </Prose>
        <Figure
          notice="The CPU line is crossed twice: your buffers in, the result out."
          caption="Between steps the data is a view of a GPU buffer, so nothing is copied or read back."
        >
          <OneGraphFigure />
        </Figure>
        {localMoran ? (
          <ExampleCode
            example={localMoran}
            label={`${SCENARIO_ROOT}/gpu-spatial-analysis/local-moran.example.ts, the code behind the recorded run`}
            lines={findExampleLines(localMoran.code, 'weights: queen.outputWeights')}
          />
        ) : null}
        <StatStrip
          stats={[
            {value: formatCount(contributorCount), label: 'contributors you add'},
            {
              value: formatCount(LOCAL_MORAN_RECORD.nodeCount),
              label: 'graph nodes they compiled to'
            },
            {value: formatCount(LOCAL_MORAN_RECORD.dependencies), label: 'dependencies'},
            {
              value: formatResourceBytes(LOCAL_MORAN_RECORD.bytes, LOCAL_MORAN_RECORD.bytesKind),
              label: 'in buffers, none read between steps'
            }
          ]}
        />
      </Section>

      <Section title="Curated workflows">
        <Prose>
          These questions and methods come from the same templates as the analysis library.
          Availability describes the current Kepler binding.
        </Prose>
        <div className="workflows-index">
          {WORKFLOW_TEMPLATES.map(template => (
            <details key={template.id} data-template-id={template.id}>
              <summary>{template.title}</summary>
              <p>{template.name}</p>
              <p>{template.method}</p>
              <p>{template.slots.map(slot => slot.label).join('; ')}</p>
              <p>
                {template.execution.status === 'available'
                  ? `Available: ${template.execution.bindingId}`
                  : template.execution.reason}
              </p>
            </details>
          ))}
        </div>
      </Section>

      <Section title="Five recipe recordings">
        <Prose>
          Each is a recipe with a recorded run. Pick a question. A card shows the operation’s own
          recorded picture, not this chain’s intermediate; the arrow line says what it passes on.
        </Prose>
        <div className="workflows-index" role="tablist" aria-label="Workflows">
          {WORKFLOWS.map(workflow => (
            <button
              key={workflow.id}
              type="button"
              role="tab"
              id={`workflow-tab-${workflow.id}`}
              aria-selected={workflow.id === openWorkflow.id}
              aria-controls="workflow-panel"
              onClick={() => setWorkflowId(workflow.id)}
            >
              <strong>{workflow.title}</strong>
              <span>{workflow.question}</span>
            </button>
          ))}
        </div>
        <div
          id="workflow-panel"
          role="tabpanel"
          aria-labelledby={`workflow-tab-${openWorkflow.id}`}
        >
          <WorkflowBlock
            key={openWorkflow.id}
            workflow={openWorkflow}
            example={code[openWorkflow.example]}
          />
        </div>
      </Section>

      <Section title="Recipe or by hand">
        <Prose>
          A recipe is no more than <code>graph.add</code> calls in chain order, plus small adapter
          kernels and sensible sizes and defaults. Call it when your question is the standard one;
          chain contributors yourself when you need an option it does not expose or a step it lacks.
        </Prose>
        <Prose>
          Either way you get one graph. Most recipes take your own views in <code>outputs</code>:
          pass them for anything you want to read, because graph-owned views carry data between
          steps but cannot be read back.
        </Prose>
        {recipeResult ? (
          <CodeWalk
            example={recipeResult}
            steps={getRecipeResultSteps(recipeResult.code)}
            active={activeField}
            onActive={setActiveField}
            label="What every recipe returns: GPURecipeResult and its status types, from source"
          />
        ) : null}
      </Section>
    </TopicShell>
  );
}
