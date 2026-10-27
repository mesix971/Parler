import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { SettingContainer } from "../../ui/SettingContainer";
import { Dropdown } from "../../ui/Dropdown";
import { Button } from "../../ui/Button";
import { GeminiKeyDialog } from "../GeminiKeyDialog";
import { useSettings } from "../../../hooks/useSettings";
import { commands } from "@/bindings";

/**
 * Modèle conseillé pour la dictée : 500 dictées par jour sur le niveau gratuit
 * (25 pour gemini-3.5-transcribe), 1 à 4 s et plus juste que Transcribe sur
 * de vraies dictées (2026-10-27). gemini-2.5-flash, conseillé avant, n'est
 * plus ouvert aux nouveaux projets Google (HTTP 404). Parler bascule tout
 * seul sur Transcribe si ce modèle est saturé ou à sa limite.
 */
const RECOMMENDED_MODEL = "gemini-3.5-flash-lite";

/** Liste de secours si Google est injoignable (hors ligne…). */
const FALLBACK_MODELS = [RECOMMENDED_MODEL, "gemini-3.5-transcribe"];

/**
 * Clé API et modèle Gemini utilisés pour transcrire. Affiché seulement quand
 * Gemini sert à la transcription (modèle principal ou audio long). La clé se
 * change ici (avant la 0.13.0, une fois enregistrée, elle ne se changeait
 * plus nulle part). La liste des modèles vient de Google : uniquement ceux
 * réellement accessibles avec la clé.
 */
export const GeminiModelSettings: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting } = useSettings();
  const [models, setModels] = useState<string[]>(FALLBACK_MODELS);
  // Fenêtre ouverte pour la clé principale ou pour la clé de secours
  const [keyDialog, setKeyDialog] = useState<
    "gemini_api_key" | "gemini_api_key_backup" | null
  >(null);

  const apiKey = getSetting("gemini_api_key");
  const hasKey = !!apiKey && apiKey.trim().length > 0;
  const backupKey = getSetting("gemini_api_key_backup");
  const hasBackup = !!backupKey && backupKey.trim().length > 0;
  const usesGemini =
    getSetting("selected_model") === "gemini-api" ||
    getSetting("long_audio_model") === "gemini-api";
  const currentModel = getSetting("gemini_model") ?? RECOMMENDED_MODEL;

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
  }, [apiKey, hasKey, usesGemini]);

  if (!usesGemini) {
    return null;
  }

  // Le modèle conseillé en tête ; le modèle actuel reste toujours proposé,
  // même s'il n'est plus listé.
  const ordered = models.includes(RECOMMENDED_MODEL)
    ? [RECOMMENDED_MODEL, ...models.filter((id) => id !== RECOMMENDED_MODEL)]
    : models;
  const ids = ordered.includes(currentModel)
    ? ordered
    : [currentModel, ...ordered];
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
        title={t("settings.gemini.apiKey")}
        description={t("settings.gemini.apiKeyDescription")}
        descriptionMode="tooltip"
        grouped={true}
      >
        <Button
          variant="secondary"
          size="sm"
          onClick={() => setKeyDialog("gemini_api_key")}
        >
          {hasKey
            ? t("settings.gemini.changeKey")
            : t("settings.gemini.addKey")}
        </Button>
      </SettingContainer>
      {hasKey && (
        <SettingContainer
          title={t("settings.gemini.backupKey")}
          description={t("settings.gemini.backupKeyDescription")}
          descriptionMode="tooltip"
          grouped={true}
        >
          <div className="flex gap-2">
            {hasBackup && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => void updateSetting("gemini_api_key_backup", "")}
              >
                {t("settings.gemini.removeKey")}
              </Button>
            )}
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setKeyDialog("gemini_api_key_backup")}
            >
              {hasBackup
                ? t("settings.gemini.changeKey")
                : t("settings.gemini.addKey")}
            </Button>
          </div>
        </SettingContainer>
      )}
      {hasKey && (
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
      )}
      {keyDialog && (
        <GeminiKeyDialog
          setting={keyDialog}
          onClose={() => setKeyDialog(null)}
        />
      )}
    </SettingsGroup>
  );
};
