// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ReactNode} from 'react';

export type PageHeaderProps = {
  title: ReactNode;
  lede?: ReactNode;
  /** Small monospace line above the title, for example an icon and "Data type". */
  eyebrow?: ReactNode;
  actions?: ReactNode;
};

export function PageHeader({title, lede, eyebrow, actions}: PageHeaderProps) {
  return (
    <header className="page-header">
      {eyebrow ? <div className="page-header__eyebrow">{eyebrow}</div> : null}
      <h1 className="page-header__title">{title}</h1>
      {lede ? <p className="page-header__lede">{lede}</p> : null}
      {actions ? <div className="page-header__actions">{actions}</div> : null}
    </header>
  );
}
