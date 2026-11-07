// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ReactNode} from 'react';
import {
  getConsumerStateMessage,
  makeConsumerState,
  type ResidentConsumerName,
  type ResidentConsumerState
} from './resident-consumer-state';

export type ResidentConsumerNoticeProps = {
  consumer: ResidentConsumerName;
  status?: Exclude<ResidentConsumerState['status'], 'ready'>;
  /** Raw reason or error text; empty falls back to the shared state message. */
  reason?: string;
  className?: string;
  children?: ReactNode;
};

/** One presentation of the shared consumer states: errors are alerts, everything else is a status. */
export function ResidentConsumerNotice({
  consumer,
  status = 'error',
  reason = '',
  className,
  children
}: ResidentConsumerNoticeProps) {
  const message = getConsumerStateMessage(makeConsumerState(status, consumer, reason));
  return (
    <p
      role={status === 'error' ? 'alert' : 'status'}
      className={className}
      data-resident-consumer-state={status}
      data-resident-consumer={consumer}
    >
      {message}
      {children}
    </p>
  );
}
