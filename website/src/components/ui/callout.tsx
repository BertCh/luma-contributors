// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ReactNode} from 'react';

export function Callout({
  title,
  tone = 'note',
  children
}: {
  title?: ReactNode;
  tone?: 'note' | 'warning';
  children: ReactNode;
}) {
  return (
    <aside className={`callout ${tone === 'warning' ? 'callout--warning' : ''}`}>
      {title ? <strong className="callout__title">{title}</strong> : null}
      {children}
    </aside>
  );
}
