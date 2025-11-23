import React from 'react';
import { Text, XStack } from 'tamagui';

type Variant = 'info' | 'success' | 'warn' | 'danger';

const variantStyles: Record<Variant, { bg: string; fg: string }> = {
  info: { bg: 'rgba(124, 140, 255, 0.18)', fg: '#cdd4ff' },
  success: { bg: 'rgba(125, 242, 164, 0.16)', fg: '#b7f6cf' },
  warn: { bg: 'rgba(255, 219, 127, 0.18)', fg: '#ffe6b6' },
  danger: { bg: 'rgba(255, 143, 177, 0.18)', fg: '#ffc2d7' },
};

type Props = {
  children: React.ReactNode;
  variant?: Variant;
};

export const Pill: React.FC<Props> = ({ children, variant = 'info' }) => {
  const palette = variantStyles[variant];
  return (
    <XStack
      alignItems="center"
      paddingVertical="$1.5"
      paddingHorizontal="$3"
      borderRadius="$2"
      backgroundColor={palette.bg}
    >
      <Text fontSize={12} color={palette.fg} letterSpacing={0.2}>
        {children}
      </Text>
    </XStack>
  );
};
