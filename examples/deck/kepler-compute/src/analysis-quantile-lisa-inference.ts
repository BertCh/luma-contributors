// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveKeplerQuantileLISAAlternative,
  type KeplerQuantileLISASettings
} from '@luma-contributors/kepler';
import {
  normalizeQuantileLisaSettings as normalizeMembershipSettings,
  type QuantileLisaSettings as MembershipSettings
} from './analysis-quantile-lisa-settings';

export type QuantileLisaSettings = MembershipSettings &
  Partial<Pick<KeplerQuantileLISASettings, 'alternative'>>;

/** Existing resident settings retain their legacy tail; new choices survive membership normalization. */
export function normalizeQuantileLisaSettings(
  settings: QuantileLisaSettings,
  rowCount: number
): MembershipSettings & Pick<KeplerQuantileLISASettings, 'alternative'> {
  const alternative = resolveKeplerQuantileLISAAlternative(settings, undefined, 'geoda-legacy');
  return {...normalizeMembershipSettings(settings, rowCount), alternative};
}

/** Label numeric p-values with the tail that produced them, including saved legacy settings. */
export function getQuantileLisaPValueLabel(
  alternative?: KeplerQuantileLISASettings['alternative']
): string {
  const resolved = resolveKeplerQuantileLISAAlternative({
    alternative: alternative === undefined ? 'geoda-legacy' : alternative
  });
  return resolved === 'geoda-legacy'
    ? 'Legacy pseudo p-value (geoda-legacy)'
    : `Quantile LISA p-value (${resolved})`;
}
