import React from 'react';
import { Circle, Text, XStack, YStack } from 'tamagui';

type Props = {
  status?: string;
  error?: string | null;
  baseUrl: string;
};

export const StatusCard: React.FC<Props> = ({ status, error, baseUrl }) => {
  const subtitle = error ? 'Unable to reach worker' : 'Worker status';
  const stateLabel = error ? 'offline' : status ?? 'checking';
  const glow = error ? '#ff5c8d' : '#7df2a4';

  return (
    <YStack
      borderRadius="$3"
      borderWidth={1}
      borderColor="$border"
      backgroundColor="$panel"
      padding="$4"
      gap="$2"
      shadowColor="rgba(75, 214, 255, 0.25)"
      shadowOffset={{ width: 0, height: 12 }}
      shadowRadius={32}
    >
      <XStack ai="center" gap="$2">
        <Circle size={12} backgroundColor={glow} shadowColor={glow} shadowRadius={18} shadowOffset={{ width: 0, height: 0 }} />
        <Text color="$textMuted" fontSize={12} textTransform="uppercase" letterSpacing={0.4}>
          {subtitle}
        </Text>
      </XStack>
      <Text color="$text" fontSize={24} fontFamily="$heading" letterSpacing={0.6}>
        {stateLabel}
      </Text>
      <Text color="$textMuted" fontSize={13}>
        {baseUrl}
      </Text>
      {error ? (
        <Text color="#ffb3c8" fontSize={13}>
          {error}
        </Text>
      ) : !status ? (
        <Text color="$textMuted" fontSize={13}>
          Waiting for response…
        </Text>
      ) : null}
    </YStack>
  );
};
