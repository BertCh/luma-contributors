// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Keep the final-list, combined event-candidate and scratch-only evidence distinct in lenses. */
export function getZoneEventLossCaptures<View>(
  events: {overflow: View; candidateOverflow: View},
  scratchCandidateOverflow: View
) {
  return {
    eventOverflow: {
      view: events.overflow,
      role: 'intermediate' as const,
      meaning:
        'Final event-list capacity loss; crossings are incomplete when this or eventCandidateOverflow is nonzero. Dwell is independent of final-list capacity.'
    },
    eventCandidateOverflow: {
      view: events.candidateOverflow,
      role: 'intermediate' as const,
      meaning:
        'Candidate scratch or per-track event-list loss; crossings are incomplete when nonzero. Check candidateScratchOverflow separately for dwell.'
    },
    scratchCandidateOverflow: {
      view: scratchCandidateOverflow,
      role: 'intermediate' as const,
      meaning:
        'Candidate scratch loss; crossings, visits and dwell totals may all be incomplete when nonzero.'
    }
  };
}
