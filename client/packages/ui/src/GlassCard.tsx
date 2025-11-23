import React from 'react';
import { LinearGradient } from '@tamagui/linear-gradient';
import { YStack } from '@tamagui/stacks';

type Props = {
  children: React.ReactNode;
};

export const GlassCard: React.FC<Props> = ({ children }) => {
  return (
    <YStack
      overflow="hidden"
      borderRadius="$3"
      borderWidth={1}
      borderColor="$border"
      backgroundColor="$panel"
      shadowColor="rgba(75, 214, 255, 0.22)"
      shadowOffset={{ width: 0, height: 18 }}
      shadowRadius={36}
    >
      <LinearGradient
        start={[0, 0]}
        end={[1, 1]}
        colors={['rgba(124, 140, 255, 0.14)', 'rgba(126, 241, 255, 0.08)']}
        width="100%"
        height={6}
      />
      <YStack padding="$4" gap="$3">
        {children}
      </YStack>
    </YStack>
  );
};
