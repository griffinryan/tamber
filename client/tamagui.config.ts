import { createInterFont } from '@tamagui/font-inter';
import { createFont, createTamagui } from '@tamagui/core';
import { shorthands } from '@tamagui/shorthands';
import { tokens } from './tokens';

const alagard = createFont({
  family: 'Alagard',
  size: {
    1: 12,
    2: 14,
    3: 16,
    4: 18,
    5: 22,
    6: 28,
    7: 36,
  },
  lineHeight: {
    1: 16,
    2: 18,
    3: 20,
    4: 22,
    5: 26,
    6: 32,
    7: 40,
  },
  weight: {
    4: '400',
    6: '600',
    7: '700',
  },
  letterSpacing: {
    4: 0,
    6: 0.2,
    7: 0.4,
  },
});

const inter = createInterFont();

const themes = {
  dark: {
    background: tokens.color.background,
    panel: tokens.color.panel,
    accent: tokens.color.accent,
    accentAlt: tokens.color.accentAlt,
    text: tokens.color.text,
    textMuted: tokens.color.textMuted,
    border: tokens.color.border,
    glow: tokens.color.glow,
  },
};

export const config = createTamagui({
  defaultTheme: 'dark',
  shorthands,
  themes,
  tokens,
  fonts: {
    heading: alagard,
    body: inter,
  },
});

export default config;
