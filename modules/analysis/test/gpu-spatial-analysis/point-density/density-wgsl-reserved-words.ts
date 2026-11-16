// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  stripWGSLComments,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

/** Scan every identifier against the repository's complete WGSL reserved-word list. */
export function getDensityReservedWords(source: string): string[] {
  return [...new Set(stripWGSLComments(source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [])].filter(
    token => WGSL_RESERVED_WORDS.has(token)
  );
}
