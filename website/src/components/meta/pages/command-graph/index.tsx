// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Link} from '@tanstack/react-router';
import {useState} from 'react';
import {formatCount} from '../../../../lens/format';
import summaryJson from '../../../../lens/summary.json';
import type {LensSummary, LensViewModel} from '../../../../lens/types';
import {Section} from '../../../ui/section';
import {findExampleLines} from '../../examples';
import {LensGraphPanel} from '../../lens-graph-panel';
import {
  ExampleCode,
  Figure,
  InShort,
  MissingRun,
  Prose,
  RunLocator,
  Term,
  TopicShell
} from '../../page-kit';
import type {TopicPageProps} from '../../topic-types';
import {COMMAND_GRAPH_SCENARIOS} from './data';
import {DeclareFigure} from './declare-figure';
import {HazardCards, HazardShareBar} from './hazard-cards';
import './command-graph.css';

const summary = summaryJson as LensSummary;

type OperationId = 'contiguity' | 'transform' | 'moran';

const OPERATIONS: readonly {id: OperationId; title: string; operator: string}[] = [
  {id: 'contiguity', title: 'contiguity weights', operator: 'GPUContiguityWeights'},
  {id: 'transform', title: 'weights transform', operator: 'GPUSpatialWeightsTransform'},
  {id: 'moran', title: 'local Moran', operator: 'GPULocalMoran'}
];

/**
 * Which of the three chained operations a folded group belongs to, by label prefix. The snapshot
 * tap is the recording's own readback, so it belongs to none.
 */
function getOperationOfGroup(label: string): OperationId | undefined {
  if (label.startsWith('contiguity-weights-')) return 'contiguity';
  if (label === 'spatial-weights-transform') return 'transform';
  if (label === 'snapshot-tap') return undefined;
  return 'moran';
}

function getOperations(model: LensViewModel) {
  return OPERATIONS.map(operation => {
    const groups = model.nodes.filter(node => getOperationOfGroup(node.label) === operation.id);
    return {
      ...operation,
      nodeIds: groups.map(node => node.id),
      groupCount: groups.length,
      kernelCount: groups.reduce((total, node) => total + node.count, 0)
    };
  });
}

export function Page({topic, models, code}: TopicPageProps) {
  const timeWindow = models[COMMAND_GRAPH_SCENARIOS.timeWindowFilter];
  const moran = models[COMMAND_GRAPH_SCENARIOS.localMoran];
  const declareKernels = code['declareKernels'];
  const localMoranChain = code['localMoranChain'];
  const [operation, setOperation] = useState<OperationId | 'all'>('all');
  const operations = moran ? getOperations(moran) : [];
  const highlighted = operations.find(candidate => candidate.id === operation)?.nodeIds;
  const firstEarlyGroup = moran ? findEarlyGroup(moran) : undefined;
  const crossingOrderHazards =
    moran?.edges.filter(edge => edge.hazards?.some(kind => kind !== 'raw')).length ?? 0;
  // One classify invocation per row.
  const rowCount = timeWindow?.nodes.find(node => node.variant === 'classify')?.invocations ?? 0;

  return (
    <TopicShell
      topic={topic}
      tryIt={[
        {
          kind: 'function',
          id: 'time-window-filter',
          note: `${timeWindow?.recordedNodeCount ?? 0} graph nodes in the recorded run on this page.`
        },
        {kind: 'function', id: 'local-morans-i', note: 'Three chained operations in one graph.'},
        {
          kind: 'reference',
          slug: 'api/gpu-spatial-analysis/concepts',
          label: 'Spatial analysis concepts',
          note: 'Graphs, views and outputs in the reference.'
        },
        {kind: 'topic', slug: 'scheduling', note: 'What happens to the order after compile.'}
      ]}
    >
      <RunLocator stages={['build', 'compile']} />
      <InShort
        points={[
          'Each kernel lists the buffers it reads and the buffers it writes. Nothing else.',
          'Where two kernels touch one buffer and one of them writes, the graph adds a hazard: an edge that fixes their order.',
          'Compile turns those edges into one order. You never write an edge.'
        ]}
      />

      <Section title="Declare, never order">
        <Prose>
          A <Term id="command-graph" /> is a list of graph nodes and the{' '}
          <Term id="buffer">buffers</Term> each one touches. The time window filter below keeps the
          rows of {formatCount(rowCount)} timestamps that fall inside a window and packs their ids
          in {timeWindow?.recordedNodeCount ?? 0} graph nodes, recorded on a real GPU.
        </Prose>
        {timeWindow ? (
          <Figure
            notice="Nobody wrote these edges."
            caption="Each line on the left is one graph node reading a buffer that an earlier node wrote. Pick a node to light the buffers it reads and the nodes they came from."
          >
            <DeclareFigure model={timeWindow} />
          </Figure>
        ) : (
          <MissingRun scenarioId={COMMAND_GRAPH_SCENARIOS.timeWindowFilter} />
        )}
        {declareKernels ? (
          <>
            <Prose>
              This is how the filter adds its compute kernels. Each binding says <code>read</code>{' '}
              or <code>read_write</code>; that becomes a <code>storage-read</code> or{' '}
              <code>storage-read-write</code> use, and luma.gl infers the edges from those uses.
            </Prose>
            <Figure>
              <ExampleCode
                example={declareKernels}
                lines={findExampleLines(declareKernels.code, 'bindings: [', '],')}
                label="GPUTimeWindowFilter, abridged"
              />
            </Figure>
          </>
        ) : null}
      </Section>

      <Section title="Three hazards">
        <Prose>
          A <Term id="hazard" /> appears wherever two kernels touch one buffer and at least one of
          them writes it. There are three kinds; each card shows the first one in a real record.
        </Prose>
        <Figure
          notice="Every edge is one of three kinds."
          caption={`Read after write is most of them. The bar counts all ${formatCount(summary.dependencies.raw + summary.dependencies.war + summary.dependencies.waw)} dependencies over the ${summary.recordCount} records.`}
        >
          <HazardCards
            models={{
              raw: timeWindow,
              war: models[COMMAND_GRAPH_SCENARIOS.classBreaks],
              waw: models[COMMAND_GRAPH_SCENARIOS.hilbertKeys]
            }}
          />
          <HazardShareBar summary={summary} />
        </Figure>
      </Section>

      <Section title="A real analysis">
        {localMoranChain ? (
          <Prose>
            Local Moran on census tracts chains three operations. The code adds them one after
            another; the only link between them is that one operation's output view is the next
            one's input.
          </Prose>
        ) : null}
        {localMoranChain ? (
          <Figure>
            <ExampleCode
              example={localMoranChain}
              label="local-moran.example.ts, the recorded run"
            />
          </Figure>
        ) : null}
        {moran ? (
          <Figure
            notice={`${formatCount(moran.recordedNodeCount)} graph nodes, ${formatCount(moran.recordedEdgeCount)} edges, none written by hand.`}
            caption={`It folds into ${moran.nodes.length} groups (xN marks repeated kernels), so most write-after-read and write-after-write hazards fall inside a group: the key counts all of them, the lines show the ${crossingOrderHazards} that cross between groups. Pick an operation to see where its kernels sit.`}
          >
            <div
              className="meta-toggle command-graph-operations"
              role="group"
              aria-label="Operation"
            >
              <button
                type="button"
                aria-pressed={operation === 'all'}
                onClick={() => setOperation('all')}
              >
                all three
              </button>
              {operations.map(candidate => (
                <button
                  key={candidate.id}
                  type="button"
                  aria-pressed={operation === candidate.id}
                  onClick={() => setOperation(candidate.id)}
                >
                  {candidate.title}{' '}
                  <span className="command-graph-operations__count">
                    {candidate.groupCount} {candidate.groupCount === 1 ? 'group' : 'groups'},{' '}
                    {formatCount(candidate.kernelCount)}{' '}
                    {candidate.kernelCount === 1 ? 'kernel' : 'kernels'}
                  </span>
                </button>
              ))}
            </div>
            <LensGraphPanel scenario={moran} showHazards highlightNodeIds={highlighted} />
          </Figure>
        ) : (
          <MissingRun scenarioId={COMMAND_GRAPH_SCENARIOS.localMoran} />
        )}
      </Section>

      <Section title="What compile does">
        <Prose>
          <Term id="compile">Compile</Term> walks the same read and write sets, collects the hazards
          and sorts the kernels into an order that respects every one. Kernels that do not depend on
          each other keep the order they were added in; the graph then freezes and compiles only
          once.
        </Prose>
        {firstEarlyGroup ? (
          <Prose>
            The order is not the order of the code. <code>{firstEarlyGroup.label}</code> belongs to
            local Moran, added last, yet it sits among the contiguity-weights kernels: nothing it
            touches is written before it, so it does not have to wait.
          </Prose>
        ) : null}
        <Prose>
          What that order costs per frame, and what changes without compiling again, is the{' '}
          <Link to="/how-it-works/$topic" params={{topic: 'scheduling'}}>
            next page
          </Link>
          .
        </Prose>
      </Section>
    </TopicShell>
  );
}

/** The first local Moran group that the recorded order places before a contiguity-weights group. */
function findEarlyGroup(model: LensViewModel) {
  let lastContiguity = -1;
  model.nodes.forEach((node, index) => {
    if (getOperationOfGroup(node.label) === 'contiguity') lastContiguity = index;
  });
  return model.nodes
    .slice(0, Math.max(0, lastContiguity))
    .find(node => getOperationOfGroup(node.label) === 'moran');
}
