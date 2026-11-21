// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const panelSpectrumRepetitions = new WeakMap<object, number>();

/** Private composition metadata; never a public contributor prop. */
export function withPanelErrorSpectrum<Props extends object>(
  props: Props,
  repetitions: number
): Props {
  panelSpectrumRepetitions.set(props, repetitions);
  return props;
}

/** Read composition metadata before the constructor snapshots its public props. */
export function getPanelErrorSpectrumRepetitions(props: object): number {
  return panelSpectrumRepetitions.get(props) ?? 1;
}
