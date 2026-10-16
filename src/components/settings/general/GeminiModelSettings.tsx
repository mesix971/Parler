import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { SettingContainer } from "../../ui/SettingContainer";
import { Dropdown } from "../../ui/Dropdown";
import { useSettings } from "../../../hooks/useSettings";
import { commands } from "@/bindings";

/** Modèle conseillé pour la dictée : dédié à la transcription. */
const RECOMMENDED_MODEL = "gemini-3.5-transcribe";

/** Liste de secours si Google est injoignable (hors ligne…). */
const FALLBACK_MODELS = [
  RECOMMENDED_MODEL,
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
];

/**
 * Choix du modèle Gemini utilisé pour transcrire. Affiché seulement quand
 * Gemini sert à la transcription (modèle principal ou audio long) et qu'une
 * clé API est enregistrée. La liste vient de Google : uniquement les modèles
 * réellement accessibles avec la clé.
 */
export const GeminiModelSettings: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting } = useSettings();
  const [models, setModels] = useState<string[]>(FALLBACK_MODELS);

  const apiKey = getSetting("gemini_api_key");
  const hasKey = !!apiKey && apiKey.trim().length > 0;
  const usesGemini =
    getSetting("selected_model") === "gemini-api" ||
    getSetting("long_audio_model") === "gemini-api";
  const currentModel = getSetting("gemini_model") ?? "gemini-2.5-flash";

  useEffect(() => {
    if (!hasKey || !usesGemini) return;
    let cancelled = false;
    commands
      .listGeminiTranscriptionModels()
      .then((result) => {
        if (!cancelled && result.status === "ok" && result.data.length > 0) {
          setModels(result.data);
        } else if (result.status === "error") {
          console.warn("Gemini model list unavailable:", result.error);
        }
      })
      .catch((error) => console.error("Failed to load Gemini models:", error));
    return () => {
      cancelled = true;
    };
  }, [hasKey, usesGemini]);

  if (!hasKey || !usesGemini) {
    return null;
  }

  // Le modèle actuel reste toujours proposé, même s'il n'est plus listé.
  const ids = models.includes(currentModel)
    ? models
    : [currentModel, ...models];
  const options = ids.map((id) => ({
    value: id,
    label:
      id === RECOMMENDED_MODEL
        ? t("settings.gemini.recommended", { model: id })
        : id,
  }));

  return (
    <SettingsGroup
      title={t("settings.gemini.title")}
      description={t("settings.gemini.description")}
    >
      <SettingContainer
        title={t("settings.gemini.model")}
        description={t("settings.gemini.modelDescription")}
        descriptionMode="tooltip"
        grouped={true}
      >
        <Dropdown
          options={options}
          selectedValue={currentModel}
          onSelect={(value) => void updateSetting("gemini_model", value)}
        />
      </SettingContainer>
    </SettingsGroup>
  );
};
