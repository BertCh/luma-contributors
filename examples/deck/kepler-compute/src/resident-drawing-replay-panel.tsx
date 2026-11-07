// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useMemo, useState} from 'react';
import type {Feature, Geometry} from 'geojson';
import {
  planDrawingReplay,
  stepDrawingReplay,
  type DrawingReplayAnnotation,
  type DrawingReplayState,
  type DrawingReplayStep,
  type DrawingReplayViewport
} from './resident-drawing-replay';

import {clamp} from '@math.gl/core';

export type ResidentDrawingReplayProps = {
  features: readonly Feature<Geometry>[];
  annotations?: readonly DrawingReplayAnnotation[];
  viewport: DrawingReplayViewport;
  /** Fly the map to the step and highlight it (drawn shapes become the selected feature). */
  onFocusStep(step: DrawingReplayStep): void;
  stepDurationMilliseconds?: number;
};

/** Step through drawn shapes and annotations with fly-to and highlight; no native replay exists. */
export function ResidentDrawingReplay(props: ResidentDrawingReplayProps) {
  const {features, annotations = [], viewport, onFocusStep} = props;
  const durationMilliseconds = props.stepDurationMilliseconds ?? 2500;
  const steps = useMemo(
    () => planDrawingReplay(features, annotations, viewport),
    [features, annotations, viewport.width, viewport.height]
  );
  const [state, setState] = useState<DrawingReplayState>({index: 0, playing: false});
  const [started, setStarted] = useState(false);
  const index = clamp(state.index, 0, steps.length - 1);
  const step = steps[index];
  useEffect(() => {
    if (started && step) onFocusStep(step);
  }, [started, index, step?.id]);
  useEffect(() => {
    if (!state.playing) return undefined;
    const timer = setTimeout(
      () => setState(current => stepDrawingReplay(current, steps.length, 'tick')),
      durationMilliseconds
    );
    return () => clearTimeout(timer);
  }, [state.playing, index, steps.length, durationMilliseconds]);
  if (!steps.length) return null;
  const act = (action: Parameters<typeof stepDrawingReplay>[2]) => {
    setStarted(true);
    setState(current => stepDrawingReplay({...current, index}, steps.length, action));
  };
  return (
    <section aria-label="Drawing replay" data-testid="drawing-replay">
      <h3>Replay</h3>
      <button type="button" onClick={() => act('previous')}>
        Previous step
      </button>
      <button
        type="button"
        data-testid="drawing-replay-play"
        onClick={() =>
          act(state.playing ? 'pause' : index + 1 >= steps.length ? 'restart' : 'play')
        }
      >
        {state.playing ? 'Pause replay' : 'Play replay'}
      </button>
      <button type="button" onClick={() => act('next')}>
        Next step
      </button>
      <p role="status" data-testid="drawing-replay-status">
        {started
          ? `Step ${index + 1} of ${steps.length}: ${step.kind} ${step.label}`
          : `${steps.length} steps (${steps.filter(s => s.kind === 'feature').length} shapes, ${steps.filter(s => s.kind === 'annotation').length} annotations)`}
      </p>
    </section>
  );
}
