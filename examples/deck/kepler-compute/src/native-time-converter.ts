// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {NativeTimeParser} from './native-column-arrow';

/** Native Kepler time parser (`timeToUnixMilli` from @kepler.gl/utils), installed by the host. */
export type TimeToUnixMilliConverter = NativeTimeParser;

let converter: TimeToUnixMilliConverter | null = null;

/** Installs the native time converter; called once by `native-utils-host`. */
export function setTimeToUnixMilliConverter(nextConverter: TimeToUnixMilliConverter): void {
  converter = nextConverter;
}

/** Converts a native time value; requires the host to have installed the converter. */
export const timeToUnixMilli: TimeToUnixMilliConverter = (value, format) => {
  // Host has not installed timeToUnixMilli: import './native-utils-host' from the browser entry.
  if (!converter) throw new Error('timeToUnixMilli converter not installed');
  return converter(value, format);
};
