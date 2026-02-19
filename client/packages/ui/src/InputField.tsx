import React from 'react';
import { Input } from '@tamagui/input';
import { Text } from '@tamagui/text';
import { XStack } from '@tamagui/stacks';

type Props = {
  value: string;
  placeholder?: string;
  onChangeText: (value: string) => void;
  prefix?: string;
  multiline?: boolean;
  rows?: number;
};

export const InputField: React.FC<Props> = ({ value, placeholder, onChangeText, prefix, multiline = false, rows = 3 }) => {
  return (
    <XStack alignItems="flex-start" gap="$2">
      {prefix ? (
        <Text color="$textMuted" fontSize={14} paddingTop="$2">
          {prefix}
        </Text>
      ) : null}
      <Input
        flex={1}
        multiline={multiline}
        numberOfLines={rows}
        value={value}
        placeholder={placeholder}
        onChangeText={onChangeText}
        backgroundColor="rgba(255,255,255,0.02)"
        borderColor="$border"
        borderWidth={1}
        padding="$3"
        borderRadius="$3"
        color="$text"
        placeholderTextColor="$textMuted"
      />
    </XStack>
  );
};
