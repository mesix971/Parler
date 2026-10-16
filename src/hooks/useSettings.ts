import { useEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import { useSettingsStore } from "../stores/settingsStore";
import type { AppSettings as Settings, AudioDevice } from "@/bindings";

interface UseSettingsReturn {
  // State
  settings: Settings | null;
  isLoading: boolean;
  isUpdating: (key: string) => boolean;
  audioDevices: AudioDevice[];
  outputDevices: AudioDevice[];
  audioFeedbackEnabled: boolean;
  postProcessModelOptions: Record<string, string[]>;

  // Actions
  updateSetting: <K extends keyof Settings>(
    key: K,
    value: Settings[K],
  ) => Promise<void>;
  resetSetting: (key: keyof Settings) => Promise<void>;
  refreshSettings: () => Promise<void>;
  refreshAudioDevices: () => Promise<void>;
  refreshOutputDevices: () => Promise<void>;

  // Binding-specific actions
  updateBinding: (id: string, binding: string) => Promise<void>;
  resetBinding: (id: string) => Promise<void>;

  // Convenience getters
  getSetting: <K extends keyof Settings>(key: K) => Settings[K] | undefined;

  // Post-processing helpers
  setPostProcessProvider: (providerId: string) => Promise<void>;
  updatePostProcessBaseUrl: (
    providerId: string,
    baseUrl: string,
  ) => Promise<void>;
  updatePostProcessApiKey: (
    providerId: string,
    apiKey: string,
  ) => Promise<void>;
  updatePostProcessModel: (providerId: string, model: string) => Promise<void>;
  fetchPostProcessModels: (providerId: string) => Promise<string[]>;
}

export const useSettings = (): UseSettingsReturn => {
  // Abonnement limité aux champs utilisés (comparaison superficielle) : les
  // composants ne se re-rendent plus pour des changements qui ne les
  // concernent pas (sons personnalisés, réglages par défaut…).
  const store = useSettingsStore(
    useShallow((s) => ({
      settings: s.settings,
      isLoading: s.isLoading,
      // Lu via isUpdatingKey, mais sélectionné pour re-rendre à chaque changement
      isUpdatingMap: s.isUpdating,
      audioDevices: s.audioDevices,
      outputDevices: s.outputDevices,
      postProcessModelOptions: s.postProcessModelOptions,
      initialize: s.initialize,
      isUpdatingKey: s.isUpdatingKey,
      updateSetting: s.updateSetting,
      resetSetting: s.resetSetting,
      refreshSettings: s.refreshSettings,
      refreshAudioDevices: s.refreshAudioDevices,
      refreshOutputDevices: s.refreshOutputDevices,
      updateBinding: s.updateBinding,
      resetBinding: s.resetBinding,
      getSetting: s.getSetting,
      setPostProcessProvider: s.setPostProcessProvider,
      updatePostProcessBaseUrl: s.updatePostProcessBaseUrl,
      updatePostProcessApiKey: s.updatePostProcessApiKey,
      updatePostProcessModel: s.updatePostProcessModel,
      fetchPostProcessModels: s.fetchPostProcessModels,
    })),
  );

  // Initialize on first mount
  useEffect(() => {
    if (store.isLoading) {
      store.initialize();
    }
  }, [store.initialize, store.isLoading]);

  return {
    settings: store.settings,
    isLoading: store.isLoading,
    isUpdating: store.isUpdatingKey,
    audioDevices: store.audioDevices,
    outputDevices: store.outputDevices,
    audioFeedbackEnabled: store.settings?.audio_feedback || false,
    postProcessModelOptions: store.postProcessModelOptions,
    updateSetting: store.updateSetting,
    resetSetting: store.resetSetting,
    refreshSettings: store.refreshSettings,
    refreshAudioDevices: store.refreshAudioDevices,
    refreshOutputDevices: store.refreshOutputDevices,
    updateBinding: store.updateBinding,
    resetBinding: store.resetBinding,
    getSetting: store.getSetting,
    setPostProcessProvider: store.setPostProcessProvider,
    updatePostProcessBaseUrl: store.updatePostProcessBaseUrl,
    updatePostProcessApiKey: store.updatePostProcessApiKey,
    updatePostProcessModel: store.updatePostProcessModel,
    fetchPostProcessModels: store.fetchPostProcessModels,
  };
};
