// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ReactNode} from 'react';

export type SectionProps = {
  title?: ReactNode;
  lede?: ReactNode;
  /** Right-aligned content in the heading row, for example a "See all" link. */
  aside?: ReactNode;
  id?: string;
  children: ReactNode;
};

export function Section({title, lede, aside, id, children}: SectionProps) {
  return (
    <section className="section" id={id}>
      {title ? (
        <div className="section__head">
          <div>
            <h2 className="section__title">{title}</h2>
            {lede ? <p className="section__lede">{lede}</p> : null}
          </div>
          {aside ? <div className="section__aside">{aside}</div> : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}
