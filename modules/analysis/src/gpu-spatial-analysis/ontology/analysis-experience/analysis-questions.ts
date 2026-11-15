// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_ANALYSIS_QUESTIONS} from './analysis-experience-types';
import type {GPUAnalysisQuestion, GPUAnalysisQuestionRecord} from './analysis-experience-types';

/**
 * The words shown to a user for each question, in the order of `GPU_ANALYSIS_QUESTIONS`. Labels are
 * phrased the way a person asks, so that a host can list them as a menu of what to find out.
 */
export const GPU_ANALYSIS_QUESTION_RECORDS: readonly GPUAnalysisQuestionRecord[] = [
  {
    id: 'where-concentrated',
    label: 'Where are things concentrated?',
    summary:
      'Shows where events or objects crowd together or thin out, as a density surface or a pattern summary.',
    examples: [
      'Where do emergency calls cluster across a city?',
      'Where are the busiest pick-up points for a bike share?',
      'Where do sightings of a bird species concentrate in a reserve?'
    ]
  },
  {
    id: 'where-significant',
    label: 'Where do values cluster or stand out?',
    summary:
      'Finds places whose values group together or stand out from their surroundings; where a method gives a p-value, it tests whether the pattern is stronger than chance.',
    examples: [
      'Which neighbourhoods are hot spots of a disease rate, beyond random variation?',
      'Which stores sell far more than the stores around them?',
      'Where is forest loss significantly higher than nearby areas?'
    ]
  },
  {
    id: 'how-much-per-area',
    label: 'How much is there in each area?',
    summary:
      'Gives totals, averages, rates or shares for each polygon, cell or zone so that areas can be compared.',
    examples: [
      'How many patients live in each postcode?',
      'What is the total sales of the shops in each district?',
      'What is the mean elevation of each watershed?'
    ]
  },
  {
    id: 'what-is-near-or-reachable',
    label: 'What is nearby, or what can be reached?',
    summary:
      'Finds the features within a distance, the closest feature, the places reachable by travel, and how well a network connects.',
    examples: [
      'Which homes are within a ten minute walk of a clinic?',
      'Which store is nearest to each customer?',
      'How far can a delivery van go from the depot in thirty minutes?'
    ]
  },
  {
    id: 'how-related',
    label: 'How are these things related?',
    summary:
      'Measures and models how one quantity depends on others, including how that relation changes across space.',
    examples: [
      'How does air pollution relate to asthma admissions?',
      'Does the effect of floor area on house price differ between districts?',
      'Which habitat variables explain where a species is found?'
    ]
  },
  {
    id: 'value-where-unmeasured',
    label: 'What is the value where nothing was measured?',
    summary:
      'Estimates a quantity at places without a measurement from the measurements around them, and checks how far such estimates can be trusted.',
    examples: [
      'What is the rainfall at a farm between two weather stations?',
      'What is the likely soil moisture across a field sampled at forty points?',
      'What is the noise level on streets with no sensor?'
    ]
  },
  {
    id: 'what-changed',
    label: 'What changed over time?',
    summary:
      'Compares periods, fields or a series of dates to find where, when and how much a quantity moved.',
    examples: [
      'Where did vegetation decline between two summers?',
      'Which areas are newly becoming crime hot spots?',
      'When did a river catchment shift in its rainfall pattern?'
    ]
  },
  {
    id: 'how-it-moves',
    label: 'How does it move or flow?',
    summary:
      'Follows things along paths and networks, or water and wind over the land, and summarises the movement or where it spends time.',
    examples: [
      'Where do delivery vehicles stop and for how long?',
      'Which roads carry the most traffic from tracked trips?',
      'Where does rain run off a hillside and which stream does it join?'
    ]
  },
  {
    id: 'shape-of-the-surface',
    label: 'What is the shape of the surface?',
    summary:
      'Describes the form of a terrain or scanned surface: slope, curvature, relief, visibility, profiles and contours.',
    examples: [
      'How steep is each slope of a mountain road?',
      'Which houses can see a proposed wind turbine?',
      'Where are the ridges and valleys of a survey area?'
    ]
  },
  {
    id: 'which-groups-or-regions',
    label: 'Which things belong together?',
    summary:
      'Divides places or objects into groups, contiguous regions or connected components that are alike or linked.',
    examples: [
      'Which areas form natural sales territories?',
      'Which hospitals in a network form tight communities?',
      'Which patches of a landscape group into one habitat?'
    ]
  },
  {
    id: 'what-is-at-each-place',
    label: 'What is at each place, and how does it score?',
    summary:
      'Says what a place is like, such as water, vegetation or land cover, or how it scores on several weighted criteria.',
    examples: [
      'Which pixels of a satellite image are open water?',
      'How suitable is each parcel for a new clinic, given distance, income and flood risk?',
      'What share of each pixel is forest, bare soil or built surface?'
    ]
  },
  {
    id: 'prepare-data',
    label: 'How do I get my data ready?',
    summary:
      'Cleans, converts, indexes or relates data so that an analysis can run on it, without answering a question itself.',
    examples: [
      'Are the polygons of a boundary file valid, and can they be repaired?',
      'How do I project survey points to a metric coordinate system?',
      'How do I define who counts as a neighbour before testing clustering?'
    ]
  }
];

const QUESTION_RECORD_BY_ID = new Map(
  GPU_ANALYSIS_QUESTION_RECORDS.map(record => [record.id, record])
);

/** Returns the record of one question. Throws for an id outside `GPU_ANALYSIS_QUESTIONS`. */
export function getGPUAnalysisQuestion(id: GPUAnalysisQuestion): GPUAnalysisQuestionRecord {
  const record = QUESTION_RECORD_BY_ID.get(id);
  if (!record) {
    throw new Error(
      `getGPUAnalysisQuestion: unknown question id "${String(id)}"; accepted ids are ${GPU_ANALYSIS_QUESTIONS.join(', ')}`
    );
  }
  return record;
}
