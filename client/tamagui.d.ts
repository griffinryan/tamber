import { TamaguiCustomConfig } from '@tamagui/core';
import { config } from './tamagui.config';

declare module '@tamagui/core' {
  // eslint-disable-next-line @typescript-eslint/no-empty-interface
  export interface TamaguiCustomConfig extends TamaguiCustomConfig<typeof config> {}
}
