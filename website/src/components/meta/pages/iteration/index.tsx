// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import summary from '../../../../lens/summary.json';
import {formatCount} from '../../../../lens/format';
import type {LensSummary} from '../../../../lens/types';
import {Section} from '../../../ui/section';
import {findExampleLines} from '../../examples';
import {ExampleCode, Figure, InShort, Prose, StatStrip, Term, TopicShell} from '../../page-kit';
import type {TopicPageProps} from '../../topic-types';
import {
  BudgetFigure,
  GatedList,
  GateDiagram,
  LoopFigure,
  RoundStrip,
  SearchFigure
} from './figures';
import {
  DISPATCHED_WORKGROUPS,
  FIRST_EMPTY_ROUND,
  FRONTIER_WORKGROUP_SIZE,
  GATED,
  PEAK_ROUND,
  PEAK_SIZE,
  SEARCH,
  WORKING_ROUNDS
} from './search-data';
import './iteration.css';

const SUMMARY = summary as LensSummary;
const [EVEN_BUFFER, ODD_BUFFER] = SEARCH.gateBuffers;
const IDLE_ROUNDS = SEARCH.relaxRoundCount - WORKING_ROUNDS;

export function Page({topic, code}: TopicPageProps) {
  const example = code.reachability;
  return (
    <TopicShell
      topic={topic}
      tryIt={[
        {kind: 'function', id: 'network-reachability', note: 'the search on this page'},
        {kind: 'function', id: 'network-service-areas', note: 'the same rounds, per facility'},
        {kind: 'function', id: 'cost-distance', note: 'gated rounds on a raster'},
        {
          kind: 'reference',
          slug: 'api/gpu-network/operators',
          label: 'Network operators',
          note: 'maximumIterations, localIterations and status'
        },
        {kind: 'topic', slug: 'contract', note: 'rule 4: iteration with a budget'}
      ]}
    >
      <InShort
        points={[
          <>
            An iterative search is written into the graph when the graph is built:{' '}
            <code>maximumIterations: {SEARCH.maximumIterations}</code> gives{' '}
            {SEARCH.relaxRoundCount} relax kernels, one per round.
          </>,
          <>
            Each round writes the next round's workgroup count on the GPU. An empty queue dispatches
            zero workgroups, so the CPU never asks whether the search is done.
          </>,
          <>
            When the budget runs out, the result is still published, with <code>converged = 0</code>
            .
          </>
        ]}
      />

      <Section title="A loop without readback">
        <Prose>
          A shortest-path search repeats one <Term id="round" />: relax the roads out of every node
          whose cost just improved, until nothing improves. On the CPU you would write{' '}
          <code>while (!done)</code>, but <code>done</code> lives in a GPU buffer, and a{' '}
          <Term id="readback" /> is asynchronous: the CPU waits for the GPU to finish before it can
          submit the next round.
        </Prose>
        <Prose>
          A contributor never reads <Term id="status" /> back (contract rule 9), so the rounds are
          laid out ahead of time. This code produced the run on this page; the highlighted line
          builds {SEARCH.relaxRoundCount} rounds.
        </Prose>
        {example ? (
          <Figure>
            <ExampleCode
              example={example}
              lines={findExampleLines(example.code, 'maximumIterations')}
              label="network-reachability.example.ts"
            />
          </Figure>
        ) : null}
        <Figure
          notice="The library never runs the top loop."
          caption="Every round is a kernel in the graph; one submission runs them back to back, and rounds after the search ends do nothing."
        >
          <LoopFigure />
        </Figure>
      </Section>

      <Section title="A recorded search">
        <Prose>
          This run searched {formatCount(SEARCH.nodeCount)} intersections of the Chicago Loop from
          one start, with travel time as the cost and <code>localIterations</code> ={' '}
          {SEARCH.localIterations}, so each round moves one road further. Each round relaxes the
          queue of nodes whose cost improved in the round before.
        </Prose>
        <Prose>
          A node lights at the last round whose queue it joined: its cost is final from then on (the
          record's <code>lastImprovedRound</code> stamp). A node can be queued more than once, so a
          bar can count more nodes than the map lights at that round.
        </Prose>
        <Figure
          notice={`The queue grows to ${PEAK_SIZE} nodes at round ${PEAK_ROUND} and drains to 0 at round ${FIRST_EMPTY_ROUND}.`}
          caption="Drag the round or press play; the large dots are this round's queue. Roads © OpenStreetMap contributors (ODbL)."
        >
          <SearchFigure />
        </Figure>
      </Section>

      <Section title="Zero workgroups">
        <Prose>
          Every push that opens a new {FRONTIER_WORKGROUP_SIZE}-entry chunk of the next queue adds
          one <Term id="workgroup" /> to the next round's indirect <Term id="dispatch" />. A round
          that received no pushes dispatches zero workgroups, and so does every round after it: that
          is the <Term id="gate" />.
        </Prose>
        <Figure
          notice={`${WORKING_ROUNDS} of ${SEARCH.relaxRoundCount} rounds did work.`}
          caption={`The other ${IDLE_ROUNDS} are kernels in the graph that dispatched nothing; no readback decided that.`}
        >
          <RoundStrip />
        </Figure>
        <StatStrip
          stats={[
            {value: SEARCH.relaxRoundCount, label: 'rounds laid out'},
            {value: WORKING_ROUNDS, label: 'rounds that did work'},
            {
              value: `${Math.round((100 * IDLE_ROUNDS) / SEARCH.relaxRoundCount)}%`,
              label: 'rounds that dispatched nothing'
            },
            {
              value: `${DISPATCHED_WORKGROUPS} of ${formatCount(SEARCH.relaxRoundCount * SEARCH.relaxMaximumWorkgroups)}`,
              label: 'workgroups dispatched of the most the graph allows'
            }
          ]}
        />
        <Prose>
          Round k reads its count from <code>{EVEN_BUFFER}</code> or <code>{ODD_BUFFER}</code> and
          writes round k + 1's into the other, because one buffer cannot be an indirect argument and
          a storage write in the same dispatch. The round index sits in the same 256-byte slot,
          never in the shader text, so one compiled pipeline serves all {SEARCH.relaxRoundCount}{' '}
          rounds.
        </Prose>
        <Figure
          notice={`Round ${FIRST_EMPTY_ROUND} reads x = 0 from the slot round ${FIRST_EMPTY_ROUND - 1} left it.`}
          caption="Teal boxes are kernels; each reads one slot and writes the next. luma.gl runs a GPU-gated node as an indirect dispatch whose x may be zero."
        >
          <GateDiagram />
        </Figure>
        <Prose>
          Contributors without a queue gate on a state word instead (
          <code>createWGSLDispatchGate</code>): once a one-thread <Term id="kernel" /> disarms it,
          every later round launches nothing.
        </Prose>
      </Section>

      <Section title="When the budget runs out">
        <Prose>
          The number of rounds is fixed when the graph is built. If the search needs more, the
          result is still published: nodes settled by then keep their cost, the rest are not final
          yet, and status says so.
        </Prose>
        <Figure
          notice="Below the rounds the search needed, converged is 0 and the outer roads are not final."
          caption={`This cuts the recorded stamps at the budget; it is not a new GPU run. Contributors with an iterationLimitReached word also set it to 1; reachability reports converged and iterationCount.`}
        >
          <BudgetFigure />
        </Figure>
        <Prose>
          The budget is checked once, at construction. Too many rounds throws{' '}
          <code>{SEARCH.iterationsMessage}</code>; a contributor whose kernel count grows with a
          prop exposes <code>estimatedCommandNodeCount</code> and throws past{' '}
          {formatCount(SEARCH.commandNodeBudget)}, as in <code>{SEARCH.budgetMessage}</code>.
        </Prose>
        <Prose>
          Graph size grows with the budget: this search needed {WORKING_ROUNDS} rounds, and its
          graph holds {SEARCH.graphNodeCount} nodes because it was built for{' '}
          {SEARCH.maximumIterations}.
        </Prose>
      </Section>

      <Section title="Where else">
        <Prose>
          {GATED.recordsWithGates} of {GATED.recordCount} recorded runs have gated kernels:{' '}
          {formatCount(SUMMARY.gatedKernels)} of {formatCount(SUMMARY.kernels)} kernels over all
          records. Regionalization, triangulation and terrain flow gate the most.
        </Prose>
        <Figure
          notice="In the largest graphs, most kernels are gated."
          caption="Filled: gated kernels; outline: all kernels of the record. The network search above is shown for scale."
        >
          <GatedList />
        </Figure>
      </Section>
    </TopicShell>
  );
}
