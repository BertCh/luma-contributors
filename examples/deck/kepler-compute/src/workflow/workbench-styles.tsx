// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import styled, {css} from 'styled-components';
import type {WorkflowPortKind, WorkflowStepStatus} from '@luma-contributors/kepler';
import {WORKBENCH_COLORS, WORKBENCH_PORT_COLORS, WORKBENCH_STATUS} from './workbench-contract';

/** Shared primitives of the workbench. Components compose these instead of restyling controls. */

const focusRing = css`
  &:focus-visible {
    outline: 2px solid ${WORKBENCH_COLORS.focus};
    outline-offset: 1px;
  }
`;

/** Root of every workbench surface; sets type, colour and form control defaults. */
export const WorkbenchSurface = styled.div`
  box-sizing: border-box;
  color: ${WORKBENCH_COLORS.text};
  background: ${WORKBENCH_COLORS.surface};
  font-family: ff-clan-web-pro, 'Helvetica Neue', Helvetica, sans-serif;
  font-size: 12px;
  line-height: 1.45;
  & *,
  & *::before,
  & *::after {
    box-sizing: border-box;
  }
  & input,
  & select,
  & textarea {
    width: 100%;
    min-width: 0;
    padding: 5px 7px;
    color: ${WORKBENCH_COLORS.text};
    background: ${WORKBENCH_COLORS.surfaceSunken};
    border: 1px solid ${WORKBENCH_COLORS.border};
    border-radius: 3px;
    font: inherit;
    ${focusRing}
  }
  & input[type='checkbox'],
  & input[type='radio'] {
    width: auto;
    accent-color: ${WORKBENCH_COLORS.accent};
  }
  & input[aria-invalid='true'],
  & textarea[aria-invalid='true'] {
    border-color: ${WORKBENCH_COLORS.danger};
  }
  & option {
    color: ${WORKBENCH_COLORS.text};
    background: ${WORKBENCH_COLORS.surfaceRaised};
  }
  @media (forced-colors: active) {
    & :focus-visible {
      outline-color: Highlight;
    }
  }
`;

export const WorkbenchButton = styled.button<{$variant?: 'primary' | 'danger' | 'ghost'}>`
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 26px;
  padding: 0 9px;
  white-space: nowrap;
  cursor: pointer;
  font: inherit;
  font-weight: 500;
  border-radius: 3px;
  border: 1px solid
    ${({$variant}) =>
      $variant === 'primary'
        ? WORKBENCH_COLORS.accent
        : $variant === 'ghost'
          ? 'transparent'
          : WORKBENCH_COLORS.border};
  color: ${({$variant}) =>
    $variant === 'primary'
      ? WORKBENCH_COLORS.accentText
      : $variant === 'danger'
        ? WORKBENCH_COLORS.danger
        : WORKBENCH_COLORS.text};
  background: ${({$variant}) => ($variant === 'primary' ? WORKBENCH_COLORS.accent : 'transparent')};
  &:hover:not(:disabled) {
    border-color: ${WORKBENCH_COLORS.borderStrong};
    background: ${({$variant}) =>
      $variant === 'primary' ? WORKBENCH_COLORS.focus : WORKBENCH_COLORS.surfaceRaised};
  }
  &:disabled {
    opacity: 0.45;
    cursor: default;
  }
  &[aria-pressed='true'] {
    border-color: ${WORKBENCH_COLORS.accent};
    color: ${WORKBENCH_COLORS.accent};
  }
  ${focusRing}
`;

/** Square button for a single glyph; always give it an `aria-label` and a `title`. */
export const WorkbenchIconButton = styled(WorkbenchButton)`
  width: 26px;
  padding: 0;
  justify-content: center;
`;

export const WorkbenchToolbarRow = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  min-height: 36px;
  padding: 4px 8px;
  background: ${WORKBENCH_COLORS.surfaceRaised};
  border-bottom: 1px solid ${WORKBENCH_COLORS.border};
  overflow-x: auto;
`;

export const WorkbenchSeparator = styled.span`
  flex: none;
  width: 1px;
  height: 18px;
  background: ${WORKBENCH_COLORS.border};
`;

export const WorkbenchSpacer = styled.span`
  flex: 1 1 auto;
`;

export const WorkbenchSection = styled.section`
  padding: 10px 12px;
  border-bottom: 1px solid ${WORKBENCH_COLORS.border};
  & > h3 {
    margin: 0 0 8px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: ${WORKBENCH_COLORS.textMuted};
  }
`;

export const WorkbenchField = styled.div`
  display: grid;
  gap: 3px;
  margin-bottom: 9px;
  & > label,
  & > .label {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 6px;
    color: ${WORKBENCH_COLORS.textMuted};
  }
  & .unit {
    color: ${WORKBENCH_COLORS.textMuted};
    font-variant-numeric: tabular-nums;
  }
  & .error {
    color: ${WORKBENCH_COLORS.danger};
  }
  & .hint {
    color: ${WORKBENCH_COLORS.textMuted};
  }
`;

export const WorkbenchMuted = styled.span`
  color: ${WORKBENCH_COLORS.textMuted};
`;

export const WorkbenchTabs = styled.div.attrs({role: 'tablist'})`
  display: flex;
  border-bottom: 1px solid ${WORKBENCH_COLORS.border};
  & > button {
    flex: 1 1 0;
    height: 30px;
    cursor: pointer;
    font: inherit;
    color: ${WORKBENCH_COLORS.textMuted};
    background: transparent;
    border: 0;
    border-bottom: 2px solid transparent;
    ${focusRing}
  }
  & > button[aria-selected='true'] {
    color: ${WORKBENCH_COLORS.text};
    border-bottom-color: ${WORKBENCH_COLORS.accent};
  }
`;

/** Status pill; pair it with `data-status` so scripts can read the status. */
export const WorkbenchStatusBadge = styled.span<{$status: WorkflowStepStatus}>`
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 1px 7px;
  border-radius: 9px;
  font-size: 11px;
  white-space: nowrap;
  color: ${({$status}) => WORKBENCH_STATUS[$status].color};
  border: 1px solid ${({$status}) => WORKBENCH_STATUS[$status].color};
  &::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    ${({$status}) =>
      $status === 'running' &&
      css`
        animation: workbench-pulse 1s ease-in-out infinite;
      `}
  }
  @keyframes workbench-pulse {
    50% {
      opacity: 0.25;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    &::before {
      animation: none;
    }
  }
`;

/** Small swatch naming a port kind by colour; always render the kind label next to it as text. */
export const WorkbenchPortSwatch = styled.span<{$kind: WorkflowPortKind}>`
  display: inline-block;
  flex: none;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: ${({$kind}) => WORKBENCH_PORT_COLORS[$kind]};
`;

export const WorkbenchNoticeBar = styled.div<{$kind: 'info' | 'error'}>`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  color: ${({$kind}) => ($kind === 'error' ? WORKBENCH_COLORS.danger : WORKBENCH_COLORS.text)};
  background: ${WORKBENCH_COLORS.surfaceSunken};
  border-bottom: 1px solid ${WORKBENCH_COLORS.border};
`;

export const WorkbenchTable = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  & th,
  & td {
    padding: 3px 6px;
    text-align: left;
    white-space: nowrap;
    max-width: 220px;
    overflow: hidden;
    text-overflow: ellipsis;
    border-bottom: 1px solid ${WORKBENCH_COLORS.border};
  }
  & th {
    position: sticky;
    top: 0;
    color: ${WORKBENCH_COLORS.textMuted};
    background: ${WORKBENCH_COLORS.surfaceRaised};
    font-weight: 600;
  }
  & td.number {
    text-align: right;
  }
`;
