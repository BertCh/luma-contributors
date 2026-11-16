// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Prefer recorded chain intermediates; operation previews keep uncaptured stages visible.

import {Link} from '@tanstack/react-router';
import {getEntry} from '../../../../catalog/index';
import {EntryPreview} from '../../../catalog/entry-preview';
import {getPublicUrl} from '../../../../public-url';
import manifest from '../../../../gpu/runs/workflow-intermediates.manifest.json';
import {
  findWorkflowStepPictures,
  type WorkflowPictureManifest,
  type WorkflowPicture
} from '../../../../gpu/runs/workflow-intermediates';
import type {WorkflowStep} from './workflows';
import {WORKFLOWS} from './workflows';

function StepPicture({pictures, title}: {pictures: WorkflowPicture[]; title: string}) {
  return (
    <div className="workflows-step__picture">
      {pictures.map(picture => (
        <img
          key={picture.theme}
          className={`workflows-poster-${picture.theme}`}
          src={getPublicUrl(picture.path)}
          data-run-id={picture.runId}
          data-stage-id={picture.stageId}
          data-resource-ids={picture.sources.map(source => source.resourceId).join(' ')}
          width={640}
          height={360}
          alt={`${title}: intermediate from one chained drainage run on Grand Canyon elevation`}
          loading="lazy"
          decoding="async"
        />
      ))}
    </div>
  );
}

function StepCard({
  step,
  number,
  templateId
}: {
  step: WorkflowStep;
  number: number;
  templateId: string;
}) {
  const entry = step.entry ? getEntry(step.entry) : undefined;
  const kind = entry ? 'contributor' : 'adapter';
  const pictures = findWorkflowStepPictures(
    manifest as WorkflowPictureManifest,
    templateId,
    number - 1,
    step.entry
  );
  return (
    <li
      className="workflows-step"
      data-kind={kind}
      data-optional={step.optional ? 'true' : undefined}
    >
      {entry && pictures.length === 2 ? (
        <StepPicture pictures={pictures} title={entry.title} />
      ) : entry ? (
        <div className="workflows-step__picture" data-picture-source="operation-poster">
          <EntryPreview entry={entry} live={false} aspectRatio={16 / 9} />
        </div>
      ) : (
        <div className="workflows-step__picture workflows-step__picture--empty">
          <span>{step.label}</span>
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
  const templateId = WORKFLOWS.find(workflow => workflow.steps === steps)?.templateId ?? '';
  const hasPictures = (manifest as WorkflowPictureManifest).runs.some(
    run => run.templateId === templateId
  );
  return (
    <>
      <p>
        {hasPictures
          ? 'Step pictures show one separate drainage run on a 96 × 60 Grand Canyon elevation grid. The recipe metrics below describe the earlier recording.'
          : 'Step pictures show individual operation examples. Chain intermediate captures are pending (WEB-HIW-03).'}
      </p>
      <ol className="workflows-chain" aria-label={label}>
        {steps.map((step, index) => (
          <StepCard key={step.source} step={step} number={index + 1} templateId={templateId} />
        ))}
      </ol>
    </>
  );
}
