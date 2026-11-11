// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// One workflow as a row of step cards that wraps: each card shows the operation's own recorded
// picture (both themes in the HTML, CSS shows the one that matches), its title linked to its
// function page, and what it passes to the next step with the format.

import {Link} from '@tanstack/react-router';
import {getEntry} from '../../../../catalog';
import {getPublicUrl} from '../../../../public-url';
import type {WorkflowStep} from './workflows';

function StepPicture({id, title}: {id: string; title: string}) {
  return (
    <div className="workflows-step__picture">
      {(['dark', 'light'] as const).map(theme => (
        <img
          key={theme}
          className={`workflows-poster-${theme}`}
          src={getPublicUrl(`runs/${id}-${theme}.webp`)}
          width={640}
          height={360}
          alt={`${title}: the picture of its own recorded run`}
          loading="lazy"
          decoding="async"
        />
      ))}
    </div>
  );
}

function StepCard({step, number}: {step: WorkflowStep; number: number}) {
  const entry = step.entry ? getEntry(step.entry) : undefined;
  const kind = entry ? 'contributor' : 'adapter';
  return (
    <li
      className="workflows-step"
      data-kind={kind}
      data-optional={step.optional ? 'true' : undefined}
    >
      {entry ? (
        <StepPicture id={entry.id} title={entry.title} />
      ) : (
        <div className="workflows-step__picture workflows-step__picture--empty">
          <span>{step.label?.startsWith('GPU') ? 'contributor' : 'adapter kernel'}</span>
        </div>
      )}
      <div className="workflows-step__body">
        <span className="workflows-step__number">{number}</span>
        {entry ? (
          <Link to="/fn/$id" params={{id: entry.id}} className="workflows-step__title">
            {entry.title}
          </Link>
        ) : (
          <span className="workflows-step__title">{step.label}</span>
        )}
        {entry ? <code className="workflows-step__symbol">{entry.symbol}</code> : null}
        <p className="workflows-step__passes">
          <span aria-hidden="true">→ </span>
          {step.passes} <code>{step.format}</code>
        </p>
        {step.optional ? (
          <span className="workflows-step__optional">
            only with <code>{step.optional}</code>
          </span>
        ) : null}
      </div>
    </li>
  );
}

export function ChainFigure({steps, label}: {steps: readonly WorkflowStep[]; label: string}) {
  return (
    <ol className="workflows-chain" aria-label={label}>
      {steps.map((step, index) => (
        <StepCard key={step.source} step={step} number={index + 1} />
      ))}
    </ol>
  );
}
