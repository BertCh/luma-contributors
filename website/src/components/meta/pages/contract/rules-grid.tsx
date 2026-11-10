// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The rules of dev-docs/contributor-contract.md, numbered as in that file, in five families.

import {Link} from '@tanstack/react-router';
import type {TopicSlug} from '../../topic-types';
import './contract.css';

type Rule = {number: number; title: string; text: string; topic?: TopicSlug};
type Family = {title: string; rules: readonly Rule[]};

const FAMILIES: readonly Family[] = [
  {
    title: 'Shape',
    rules: [
      {
        number: 1,
        title: 'One class per operation',
        text: 'A contributor is a class with Props and Output types; a recipe composes contributors.'
      },
      {
        number: 12,
        title: 'Registered',
        text: 'Every exported contributor has an entry in the operator catalog.'
      },
      {
        number: 13,
        title: 'One public home',
        text: 'The contract is imported from analysis-core, and no luma.gl symbol is re-exported.',
        topic: 'upstream-first'
      }
    ]
  },
  {
    title: 'Parallel by construction',
    rules: [
      {
        number: 2,
        title: 'No loop over rows',
        text: 'Totals, counts and compaction use scan, reduce, sort and compaction, never one invocation looping over the data.',
        topic: 'building-blocks'
      },
      {
        number: 3,
        title: 'Candidates by index',
        text: 'Neighbours come from a grid index or a sort, not from all pairs.',
        topic: 'building-blocks'
      },
      {
        number: 4,
        title: 'Rounds with a budget',
        text: 'Rounds are gated on the GPU, and a search that hits its limit still publishes its result.',
        topic: 'iteration'
      }
    ]
  },
  {
    title: 'Parameters and context',
    rules: [
      {
        number: 5,
        title: 'Per-frame parameters',
        text: 'A number that changes no buffer shape lives in a parameter buffer, so changing it does not recompile.',
        topic: 'scheduling'
      },
      {
        number: 6,
        title: 'Declared context',
        text: 'Coordinates, raster grids and time come from a declared context, not from literals in a shader.'
      }
    ]
  },
  {
    title: 'Data conventions',
    rules: [
      {
        number: 7,
        title: 'One no-data rule',
        text: 'A mask filters input rows; validity flags output rows; both are uint32.',
        topic: 'data-on-the-gpu'
      },
      {
        number: 8,
        title: 'One output layout',
        text: 'Variable-length output is a compact output: ids, a clamped count and an overflow flag.',
        topic: 'data-on-the-gpu'
      },
      {
        number: 11,
        title: 'Shared numerics',
        text: 'Random numbers, order-independent sums and exact tests come from shared code, not copies.'
      }
    ]
  },
  {
    title: 'Status and limits',
    rules: [
      {
        number: 9,
        title: 'Named status',
        text: 'Status uses the port names of the contract; a contributor never reads it back.'
      },
      {
        number: 10,
        title: 'Named limits',
        text: 'Construction throws an Error that starts with the contributor ID and names the prop and its range.'
      }
    ]
  }
];

export function RulesGrid() {
  return (
    <div className="contract-families">
      {FAMILIES.map(family => (
        <section key={family.title} aria-label={family.title}>
          <h3 className="contract-family__title">{family.title}</h3>
          <ul className="contract-family__cards">
            {family.rules.map(rule => (
              <li key={rule.number} className="contract-card">
                <span className="contract-card__number">rule {rule.number}</span>
                <span className="contract-card__title">{rule.title}</span>
                <p className="contract-card__text">{rule.text}</p>
                {rule.topic ? (
                  <Link
                    to="/how-it-works/$topic"
                    params={{topic: rule.topic}}
                    className="contract-card__link"
                  >
                    How it works: {rule.topic.replaceAll('-', ' ')}
                  </Link>
                ) : null}
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
