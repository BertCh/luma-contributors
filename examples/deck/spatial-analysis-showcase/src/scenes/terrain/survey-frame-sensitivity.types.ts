// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type SurveyFrameSensitivityOptions = {
  view: 'raw' | 'corrected' | 'uncertainty';
  controlRegion: 'west' | 'east' | 'north' | 'all-paired';
  assumedDifferenceSd: number;
  correlationLength: number;
};
