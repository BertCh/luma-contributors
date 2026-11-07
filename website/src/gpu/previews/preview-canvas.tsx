// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ReactElement} from 'react';
import type {PreviewKind} from '../../catalog/schema.ts';
import {GpuCanvas} from '../gpu-canvas.tsx';
import {PreviewFallback} from './fallback.tsx';
import {getPreviewParameters, getPreviewScene, isPreviewInteractive} from './index.ts';

export type PreviewCanvasProps = {
  kind: PreviewKind;
  seed?: number;
  /** Accessible description of the picture. */
  label?: string;
  aspectRatio?: number;
  className?: string;
  /** Render one slider per parameter under the canvas. */
  controls?: boolean;
  /** Forward the pointer; defaults to on for kinds that react to it. */
  interactive?: boolean;
};

/** A preview kind drawn live with luma.gl, over its static SVG fallback. */
export function PreviewCanvas({
  kind,
  seed = 1,
  label,
  aspectRatio,
  className,
  controls = false,
  interactive
}: PreviewCanvasProps): ReactElement {
  return (
    <GpuCanvas
      scene={getPreviewScene(kind)}
      fallback={<PreviewFallback kind={kind} seed={seed} />}
      seed={seed}
      label={label ?? `${kind.replace(/-/g, ' ')} preview`}
      aspectRatio={aspectRatio}
      className={className}
      controls={controls}
      parameters={getPreviewParameters(kind)}
      interactive={interactive ?? isPreviewInteractive(kind)}
    />
  );
}
