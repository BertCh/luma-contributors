// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The three hazard kinds, each shown on a real pair of kernels and the buffer between them, and the
// share of each kind over every recorded dependency.

import {formatCount} from '../../../../lens/format';
import type {LensSummary, LensViewModel} from '../../../../lens/types';
import {shortenName} from './declare-figure';
import {findFirstHazard, type HazardKind} from './hazards';

const HAZARD_TEXT: Record<
  HazardKind,
  {title: string; rule: string; first: string; second: string}
> = {
  raw: {
    title: 'read after write',
    rule: 'A kernel that reads a buffer runs after the kernel that last wrote it.',
    first: 'writes',
    second: 'reads'
  },
  war: {
    title: 'write after read',
    rule: 'A kernel that overwrites a buffer waits until every earlier reader has finished with it.',
    first: 'reads',
    second: 'then writes'
  },
  waw: {
    title: 'write after write',
    rule: 'Two kernels that write the same buffer keep their order, so the later value is the one that stays.',
    first: 'writes',
    second: 'writes again'
  }
};

export const HAZARD_KINDS: readonly HazardKind[] = ['raw', 'war', 'waw'];

export function HazardCards({
  models
}: {
  /** The record to take each kind's first instance from. */
  models: Record<HazardKind, LensViewModel | undefined>;
}) {
  return (
    <ul className="command-graph-hazards" aria-label="The three hazard kinds, each on a real run">
      {HAZARD_KINDS.map(kind => {
        const model = models[kind];
        const instance = model ? findFirstHazard(model, kind) : undefined;
        const text = HAZARD_TEXT[kind];
        if (!model || !instance) {
          return (
            <li key={kind} className="command-graph-hazard" data-hazard={kind}>
              <strong>{text.title}</strong>
              <span>{text.rule}</span>
            </li>
          );
        }
        const first = shortenName(model.nodes[instance.from].id, model.title);
        const second = shortenName(model.nodes[instance.to].id, model.title);
        const buffer = shortenName(instance.buffer, model.title);
        return (
          <li key={kind} className="command-graph-hazard" data-hazard={kind}>
            <strong className="command-graph-hazard__title">{text.title}</strong>
            <span className="command-graph-hazard__rule">{text.rule}</span>
            <ol
              className="command-graph-hazard__pair"
              aria-label={`In ${model.title}: ${first} ${text.first} ${buffer}; ${second} ${text.second} it, so ${second} waits for ${first}.`}
            >
              <li className="command-graph-hazard__kernel">
                <span className="command-graph-hazard__step">step {instance.from + 1}</span>
                {first}
              </li>
              <li className="command-graph-hazard__verb">{text.first}</li>
              <li className="command-graph-hazard__buffer">{buffer}</li>
              <li className="command-graph-hazard__verb">{text.second}</li>
              <li className="command-graph-hazard__kernel">
                <span className="command-graph-hazard__step">step {instance.to + 1}</span>
                {second}
              </li>
            </ol>
            <span className="command-graph-hazard__source">
              from <code>{model.id}</code>
            </span>
          </li>
        );
      })}
    </ul>
  );
}

/** One bar split by the share of each hazard kind over every recorded dependency. */
export function HazardShareBar({summary}: {summary: LensSummary}) {
  const {dependencies, recordCount} = summary;
  const total = dependencies.raw + dependencies.war + dependencies.waw;
  const share = (kind: HazardKind) => (100 * dependencies[kind]) / total;
  return (
    <div className="command-graph-share">
      <div
        className="command-graph-share__bar"
        role="img"
        aria-label={`Of ${formatCount(total)} dependencies over ${recordCount} records: ${HAZARD_KINDS.map(kind => `${share(kind).toFixed(1)} percent ${HAZARD_TEXT[kind].title}`).join(', ')}.`}
      >
        {HAZARD_KINDS.map(kind => (
          <span key={kind} data-hazard={kind} style={{width: `${share(kind)}%`}} />
        ))}
      </div>
      <ul className="command-graph-share__key" aria-hidden="true">
        {HAZARD_KINDS.map(kind => (
          <li key={kind} data-hazard={kind}>
            <i />
            {HAZARD_TEXT[kind].title} <b>{share(kind).toFixed(1)}%</b>{' '}
            <span>{formatCount(dependencies[kind])}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
