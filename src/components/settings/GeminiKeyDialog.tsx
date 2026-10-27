import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { commands } from "@/bindings";
import { useSettings } from "@/hooks/useSettings";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";

/** Erreurs où Google n'a pas pu juger la clé */
const UNREACHABLE = ["network", "timeout", "gemini_unavailable"];

interface GeminiKeyDialogProps {
  /** Clé principale (par défaut) ou clé de secours payante */
  setting?: "gemini_api_key" | "gemini_api_key_backup";
  onClose: () => void;
  /** Appelé une fois la clé enregistrée */
  onSaved?: () => void | Promise<void>;
}

/**
 * Fenêtre de saisie de la clé API Gemini, pour une première clé (onglet
 * Modèles) comme pour la changer (Général → API Gemini).
 */
export const GeminiKeyDialog: React.FC<GeminiKeyDialogProps> = ({
  setting = "gemini_api_key",
  onClose,
  onSaved,
}) => {
  const { t } = useTranslation();
  const { updateSetting } = useSettings();
  const [keyInput, setKeyInput] = useState("");
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Pas de fermeture pendant la vérification : la clé serait enregistrée
  // quand même, fenêtre fermée.
  const close = () => {
    if (!checking) onClose();
  };

  const handleSave = async () => {
    const key = keyInput.trim();
    if (!key || checking) return;

    // La clé est testée auprès de Google avant d'être enregistrée : une clé
    // erronée était auparavant acceptée sans rien dire, et chaque dictée
    // échouait ensuite en silence.
    setChecking(true);
    setError(null);
    try {
      const check = await commands.checkGeminiApiKey(key);
      if (check.status === "error") {
        if (check.error === "gemini_invalid_key") {
          setError(t("settings.gemini.keyRejected"));
          return;
        }
        // Google injoignable (hors ligne…) : la clé ne peut pas être testée,
        // on l'enregistre quand même en le signalant. Crédits épuisés ou
        // quota : Google a répondu, la clé est bonne mais ne transcrira pas.
        toast.warning(
          UNREACHABLE.includes(check.error)
            ? t("settings.gemini.keySavedUnchecked")
            : t(`overlay.errors.${check.error}`, {
                defaultValue: t("settings.gemini.keySavedUnchecked"),
              }),
        );
      }
    } catch (e) {
      console.error("Failed to check Gemini API key:", e);
      toast.warning(t("settings.gemini.keySavedUnchecked"));
    } finally {
      setChecking(false);
    }

    await updateSetting(setting, key);
    onClose();
    await onSaved?.();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={close}
      onKeyDown={(e) => {
        if (e.key === "Escape") close();
      }}
    >
      <div
        className="bg-background border border-mid-gray/40 rounded-xl p-5 w-96 shadow-2xl space-y-4"
        role="dialog"
        aria-modal="true"
        aria-labelledby="gemini-key-dialog-title"
        aria-describedby="gemini-key-dialog-description"
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <h3 id="gemini-key-dialog-title" className="text-base font-semibold">
            {setting === "gemini_api_key"
              ? t("settings.gemini.apiKeyRequired")
              : t("settings.gemini.backupKey")}
          </h3>
          <p
            id="gemini-key-dialog-description"
            className="text-sm text-text/60 mt-1"
          >
            {setting === "gemini_api_key"
              ? t("settings.gemini.apiKeyRequiredDescription")
              : t("settings.gemini.backupKeyDescription")}
          </p>
        </div>
        <Input
          autoFocus
          type="password"
          value={keyInput}
          onChange={(e) => {
            setKeyInput(e.target.value);
            setError(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") void handleSave();
          }}
          placeholder={t("settings.gemini.apiKeyPlaceholder")}
          aria-label={t("settings.gemini.apiKey")}
          aria-invalid={error !== null}
          className="w-full"
        />
        {error && (
          <p className="text-sm text-red-500" role="alert">
            {error}
          </p>
        )}
        <div className="flex justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={close} disabled={checking}>
            {t("settings.gemini.cancel")}
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={() => void handleSave()}
            disabled={!keyInput.trim() || checking}
          >
            {checking
              ? t("settings.gemini.checking")
              : t("settings.gemini.save")}
          </Button>
        </div>
      </div>
    </div>
  );
};
