// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  WGSL_RESERVED_WORDS,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

// The shared explicit list matches W3C WGSL §16.2 (6 November 2026), independently of the parser:
// https://www.w3.org/TR/WGSL/#reserved-words
const FLOW_FORBIDDEN_WGSL_WORDS = WGSL_RESERVED_WORDS;

/** Checks every identifier token, including references and members, against the full list. */
export function getForbiddenFlowWGSLIdentifiers(source: string): string[] {
  return [...new Set(stripWGSLComments(source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [])].filter(
    identifier => FLOW_FORBIDDEN_WGSL_WORDS.has(identifier)
  );
}
