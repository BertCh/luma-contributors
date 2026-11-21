// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Stable categories for host-side analysis input refusals. */
export type AnalysisValidationCode =
  | 'MISSING_PROP'
  | 'VIEW_FORMAT'
  | 'VIEW_LENGTH'
  | 'INTEGER_RANGE'
  | 'NUMBER_RANGE'
  | 'INVALID_OPTION'
  | 'REQUIREMENT';

/** Explicit producer identity; legacy string IDs remain accepted by the named validators. */
export type AnalysisValidationIdentity =
  | string
  | {readonly id: string; readonly contributor: string; readonly messagePrefix?: string};

/** Details that a generic requirement cannot infer from a boolean. */
export type AnalysisValidationDetails = {
  code: AnalysisValidationCode;
  path: string;
  received: unknown;
  expected: unknown;
};

/** A classifiable construction-time refusal, with the original caller value preserved. */
export class AnalysisValidationError extends Error {
  readonly code: AnalysisValidationCode;
  readonly contributor: string;
  readonly id: string;
  readonly path: string;
  readonly received: unknown;
  readonly expected: unknown;

  constructor(
    identity: AnalysisValidationIdentity,
    message: string,
    details: AnalysisValidationDetails
  ) {
    const id = typeof identity === 'string' ? identity : identity.id;
    super(`${typeof identity === 'string' ? id : (identity.messagePrefix ?? id)} ${message}`);
    this.name = 'AnalysisValidationError';
    this.id = id;
    // A legacy ID carries no class information. Callers can supply explicit identity when needed.
    this.contributor = typeof identity === 'string' ? identity : identity.contributor;
    this.code = details.code;
    this.path = details.path;
    this.received = details.received;
    this.expected = details.expected;
  }
}

/** Rejects string options outside their documented values, without implicit fallback. */
export function validateOption<Value extends string>(
  id: AnalysisValidationIdentity,
  name: string,
  value: unknown,
  acceptedValues: readonly Value[]
): asserts value is Value {
  if (typeof value !== 'string' || !acceptedValues.includes(value as Value)) {
    throw new AnalysisValidationError(
      id,
      `${name} must be ${acceptedValues.join(' or ')}; received ${String(value)}`,
      {code: 'INVALID_OPTION', path: name, received: value, expected: {values: acceptedValues}}
    );
  }
}

/**
 * Throws `${id} ${requirement}` unless the construction-time requirement holds. Use it where no
 * more specific capacity, count, extent or power-of-two validator fits; the requirement names the
 * prop and the accepted range, such as `'sampleSpacing must be a positive finite number'`.
 *
 * @param id Contributor ID, for the message.
 * @param isSatisfied Whether the requirement holds.
 * @param requirement The requirement, stated as what must be true.
 */
export function validateRequirement(
  id: AnalysisValidationIdentity,
  isSatisfied: boolean,
  requirement: string,
  details?: AnalysisValidationDetails
): asserts isSatisfied {
  if (!isSatisfied) {
    throw new AnalysisValidationError(
      id,
      requirement,
      details ?? {
        code: 'REQUIREMENT',
        path: requirement.split(' ')[0],
        received: isSatisfied,
        expected: requirement
      }
    );
  }
}
