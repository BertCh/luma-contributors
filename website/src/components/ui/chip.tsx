// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ReactNode} from 'react';

export type ChipProps = {
  children: ReactNode;
  tone?: 'neutral' | 'accent' | 'signal';
  active?: boolean;
  icon?: ReactNode;
  className?: string;
};

export function getChipClassName({
  tone = 'neutral',
  active = false,
  className = ''
}: Partial<ChipProps>): string {
  return [
    'chip',
    tone !== 'neutral' ? `chip--${tone}` : '',
    active ? 'chip--active' : '',
    className
  ]
    .filter(Boolean)
    .join(' ');
}

/** A static label. For a link, put getChipClassName() on a <Link> instead. */
export function Chip({children, icon, ...options}: ChipProps) {
  return (
    <span className={getChipClassName(options)}>
      {icon}
      {children}
    </span>
  );
}
