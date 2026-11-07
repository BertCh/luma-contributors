// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** A stable, vote-weighted state panel for the rank operator (which supports at most 64 rows). */
export type ElectionRankPanel = {
  stateFips: Uint32Array;
  values: Float32Array;
  stateCount: number;
};

/**
 * Aggregates complete-panel counties to states with one fixed county set over time. Each state
 * value is the Democratic two-party share weighted by each county's two-party votes, derived from
 * total, Democratic and Republican vote shares. States without supported counties are omitted; DC
 * is included when represented by the county dataset.
 */
export function makeElectionRankPanel(
  countyFips: Uint32Array,
  countyValid: Uint8Array,
  countyShares: readonly Float32Array[],
  countyDemShares: readonly Float32Array[],
  countyRepShares: readonly Float32Array[],
  countyTotalVotes: readonly Float32Array[]
): ElectionRankPanel {
  const stateRows: number[][] = Array.from({length: 57}, () => []);
  for (let county = 0; county < countyFips.length; county++) {
    if (!countyValid[county]) continue;
    const stateFips = Math.floor(countyFips[county] / 1000);
    if (stateFips >= 1 && stateFips <= 56) stateRows[stateFips].push(county);
  }

  const includedStates: number[] = [];
  const stateSeries: number[][] = [];
  for (let stateFips = 0; stateFips < stateRows.length; stateFips++) {
    const counties = stateRows[stateFips];
    if (counties.length === 0) continue;
    const series: number[] = [];
    let complete = true;
    for (let period = 0; period < countyShares.length; period++) {
      let democraticVotes = 0;
      let twoPartyVotes = 0;
      for (const county of counties) {
        const votes =
          countyTotalVotes[period][county] *
          (countyDemShares[period][county] + countyRepShares[period][county]);
        const share = countyShares[period][county];
        democraticVotes += share * votes;
        twoPartyVotes += votes;
      }
      const stateShare = democraticVotes / twoPartyVotes;
      if (!(twoPartyVotes > 0) || !Number.isFinite(stateShare)) {
        complete = false;
        break;
      }
      series.push(stateShare);
    }
    if (complete) {
      includedStates.push(stateFips);
      stateSeries.push(series);
    }
  }
  const values = new Float32Array(includedStates.length * countyShares.length);
  for (let period = 0; period < countyShares.length; period++)
    for (let stateIndex = 0; stateIndex < includedStates.length; stateIndex++)
      values[period * includedStates.length + stateIndex] = stateSeries[stateIndex][period];
  return {
    stateFips: Uint32Array.from(includedStates),
    values,
    stateCount: includedStates.length
  };
}
