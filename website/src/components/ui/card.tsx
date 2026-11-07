// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ElementType, ReactNode} from 'react';

export type CardProps = {
  /** Visual at the top of the card: a thumbnail, <PreviewFallback/> or a canvas. */
  media?: ReactNode;
  title?: ReactNode;
  children?: ReactNode;
  /** Small monospace line at the bottom. */
  meta?: ReactNode;
  /** Lifts and outlines on hover. Put one link in the title; it stretches over the card. */
  interactive?: boolean;
  as?: ElementType;
  className?: string;
};

export function Card({
  media,
  title,
  children,
  meta,
  interactive = false,
  as: Component = 'article',
  className = ''
}: CardProps) {
  return (
    <Component className={`card ${interactive ? 'card--interactive' : ''} ${className}`}>
      {media ? <div className="card__media">{media}</div> : null}
      <div className="card__body">
        {title ? <h3 className="card__title">{title}</h3> : null}
        {children ? <div className="card__text">{children}</div> : null}
        {meta ? <div className="card__meta">{meta}</div> : null}
      </div>
    </Component>
  );
}
