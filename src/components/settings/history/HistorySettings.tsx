import React, { useState, useEffect, useCallback, useRef } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { AudioPlayer } from "../../ui/AudioPlayer";
import { Button } from "../../ui/Button";
import {
  Copy,
  Star,
  Check,
  Trash2,
  FolderOpen,
  RefreshCw,
  Loader2,
} from "lucide-react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { readFile } from "@tauri-apps/plugin-fs";
import { ask } from "@tauri-apps/plugin-dialog";
import { commands, type HistoryEntry } from "@/bindings";
import { formatDateTime } from "@/utils/dateFormat";
import { useOsType } from "@/hooks/useOsType";
import { useSettings } from "@/hooks/useSettings";
import { useModelStore } from "@/stores/modelStore";
import { unwrap } from "@/lib/utils/result";
import { formatKeyCombination } from "@/lib/utils/keyboard";
import { getTranslatedModelName } from "@/lib/utils/modelTranslation";

interface OpenRecordingsButtonProps {
  onClick: () => void;
  label: string;
}

const OpenRecordingsButton: React.FC<OpenRecordingsButtonProps> = ({
  onClick,
  label,
}) => (
  <Button
    onClick={onClick}
    variant="secondary"
    size="sm"
    className="flex items-center gap-2"
    title={label}
  >
    <FolderOpen className="w-4 h-4" aria-hidden />
    <span>{label}</span>
  </Button>
);

/** En-tête et cadre communs aux trois états de la page (chargement, vide, liste). */
const HistoryLayout: React.FC<{
  onOpenFolder: () => void;
  children: React.ReactNode;
}> = ({ onOpenFolder, children }) => {
  const { t } = useTranslation();
  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <div className="space-y-2">
        <div className="px-4 flex items-center justify-between">
          <div>
            <h2 className="text-xs font-medium text-mid-gray uppercase tracking-wide">
              {t("settings.history.title")}
            </h2>
          </div>
          <OpenRecordingsButton
            onClick={onOpenFolder}
            label={t("settings.history.openFolder")}
          />
        </div>
        <div className="bg-background border border-mid-gray/20 rounded-lg overflow-visible">
          {children}
        </div>
      </div>
    </div>
  );
};

export const HistorySettings: React.FC = () => {
  const { t } = useTranslation();
  const osType = useOsType();
  const { getSetting } = useSettings();
  const [historyEntries, setHistoryEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);

  const loadHistoryEntries = useCallback(async () => {
    try {
      setHistoryEntries(await unwrap(commands.getHistoryEntries()));
    } catch (error) {
      console.error("Failed to load history entries:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHistoryEntries();

    // Listen for history update events
    const unlistenPromise = listen("history-updated", () => {
      loadHistoryEntries();
    });

    return () => {
      unlistenPromise.then((unlisten) => unlisten());
    };
  }, [loadHistoryEntries]);

  const toggleSaved = async (id: number) => {
    try {
      await unwrap(commands.toggleHistoryEntrySaved(id));
      // No need to reload here - the event listener will handle it
    } catch (error) {
      console.error("Failed to toggle saved status:", error);
      toast.error(t("settings.history.actionFailed"));
    }
  };

  const copyToClipboard = async (text: string): Promise<boolean> => {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (error) {
      console.error("Failed to copy to clipboard:", error);
      toast.error(t("settings.history.copyError"));
      return false;
    }
  };

  const getAudioUrl = useCallback(
    async (fileName: string) => {
      try {
        const path = await unwrap(commands.getAudioFilePath(fileName));
        if (osType === "linux") {
          const fileData = await readFile(path);
          const blob = new Blob([fileData], { type: "audio/wav" });

          return URL.createObjectURL(blob);
        }

        return convertFileSrc(path, "asset");
      } catch (error) {
        console.error("Failed to get audio file path:", error);
        return null;
      }
    },
    [osType],
  );

  const deleteAudioEntry = async (id: number) => {
    await unwrap(commands.deleteHistoryEntry(id));
  };

  const openRecordingsFolder = async () => {
    try {
      await unwrap(commands.openRecordingsFolder());
    } catch (error) {
      console.error("Failed to open recordings folder:", error);
      toast.error(t("settings.history.actionFailed"));
    }
  };

  if (loading) {
    return (
      <HistoryLayout onOpenFolder={openRecordingsFolder}>
        <div className="px-4 py-3 text-center text-text/60">
          {t("settings.history.loading")}
        </div>
      </HistoryLayout>
    );
  }

  if (historyEntries.length === 0) {
    const shortcut = getSetting("bindings")?.transcribe?.current_binding;
    return (
      <HistoryLayout onOpenFolder={openRecordingsFolder}>
        <div className="px-4 py-3 text-center text-text/60 space-y-1">
          <p>{t("settings.history.empty")}</p>
          {shortcut && (
            <p className="text-xs text-text/50">
              {t("settings.history.emptyHint", {
                shortcut: formatKeyCombination(shortcut, osType),
              })}
            </p>
          )}
        </div>
      </HistoryLayout>
    );
  }

  return (
    <HistoryLayout onOpenFolder={openRecordingsFolder}>
      <div className="divide-y divide-mid-gray/20">
        {historyEntries.map((entry) => (
          <HistoryEntryComponent
            key={entry.id}
            entry={entry}
            onToggleSaved={() => toggleSaved(entry.id)}
            onCopyText={() =>
              copyToClipboard(
                entry.post_processed_text ?? entry.transcription_text,
              )
            }
            getAudioUrl={getAudioUrl}
            deleteAudio={deleteAudioEntry}
          />
        ))}
      </div>
    </HistoryLayout>
  );
};

interface HistoryEntryProps {
  entry: HistoryEntry;
  onToggleSaved: () => void;
  onCopyText: () => Promise<boolean>;
  getAudioUrl: (fileName: string) => Promise<string | null>;
  deleteAudio: (id: number) => Promise<void>;
}

const HistoryEntryComponent: React.FC<HistoryEntryProps> = ({
  entry,
  onToggleSaved,
  onCopyText,
  getAudioUrl,
  deleteAudio,
}) => {
  const { t, i18n } = useTranslation();
  const [showCopied, setShowCopied] = useState(false);
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);
  const pickerRef = useRef<HTMLDivElement>(null);
  const models = useModelStore((s) => s.models);

  const downloadedModels = models.filter((m) => m.is_downloaded);
  const hasText = entry.transcription_text.trim().length > 0;

  const handleLoadAudio = useCallback(
    () => getAudioUrl(entry.file_name),
    [getAudioUrl, entry.file_name],
  );

  const handleCopyText = async () => {
    if (await onCopyText()) {
      setShowCopied(true);
      setTimeout(() => setShowCopied(false), 2000);
    }
  };

  const handleDeleteEntry = async () => {
    // Suppression définitive (texte + audio) : on demande confirmation.
    const confirmed = await ask(t("settings.history.deleteConfirm"), {
      title: t("settings.history.delete"),
      kind: "warning",
    });
    if (!confirmed) return;
    try {
      await deleteAudio(entry.id);
    } catch (error) {
      console.error("Failed to delete entry:", error);
      toast.error(t("settings.history.deleteError"));
    }
  };

  const handleReprocess = async (modelId: string) => {
    setShowModelPicker(false);
    setReprocessing(true);
    try {
      await unwrap(commands.reprocessHistoryEntry(entry.id, modelId));
    } catch (error) {
      console.error("Failed to reprocess entry:", error);
      toast.error(t("settings.history.reprocessError"));
    } finally {
      setReprocessing(false);
    }
  };

  useEffect(() => {
    if (!showModelPicker) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) {
        setShowModelPicker(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showModelPicker]);

  const formattedDate = formatDateTime(String(entry.timestamp), i18n.language);
  const reprocessLabel = reprocessing
    ? t("settings.history.reprocessing")
    : t("settings.history.reprocess");
  const saveLabel = entry.saved
    ? t("settings.history.unsave")
    : t("settings.history.save");

  return (
    <div className="px-4 py-2 pb-5 flex flex-col gap-3">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <p className="text-sm font-medium">{formattedDate}</p>
          {entry.model_name && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-mid-gray/10 text-text/50 font-medium">
              {entry.model_name}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <div className="relative" ref={pickerRef}>
            <button
              type="button"
              onClick={() =>
                !reprocessing && setShowModelPicker(!showModelPicker)
              }
              disabled={reprocessing}
              className="p-2 rounded-md text-text/50 hover:text-logo-primary transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              title={reprocessLabel}
              aria-label={reprocessLabel}
              aria-haspopup="menu"
              aria-expanded={showModelPicker}
            >
              {reprocessing ? (
                <Loader2 width={16} height={16} className="animate-spin" />
              ) : (
                <RefreshCw width={16} height={16} />
              )}
            </button>
            {showModelPicker && downloadedModels.length > 0 && (
              <div
                className="absolute right-0 top-full mt-1 z-50 bg-background border border-mid-gray/20 rounded-lg shadow-lg py-1 min-w-[200px]"
                role="menu"
              >
                <p className="px-3 py-1 text-xs text-text/50 font-medium">
                  {t("settings.history.selectModel")}
                </p>
                {downloadedModels.map((model) => (
                  <button
                    type="button"
                    role="menuitem"
                    key={model.id}
                    onClick={() => handleReprocess(model.id)}
                    className="w-full text-left px-3 py-1.5 text-sm hover:bg-mid-gray/10 transition-colors cursor-pointer"
                  >
                    {getTranslatedModelName(model, t)}
                  </button>
                ))}
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={handleCopyText}
            disabled={!hasText && !entry.post_processed_text}
            className="text-text/50 hover:text-logo-primary  hover:border-logo-primary transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
            title={t("settings.history.copyToClipboard")}
            aria-label={t("settings.history.copyToClipboard")}
          >
            {showCopied ? (
              <Check width={16} height={16} />
            ) : (
              <Copy width={16} height={16} />
            )}
          </button>
          <button
            type="button"
            onClick={onToggleSaved}
            className={`p-2 rounded-md transition-colors cursor-pointer ${
              entry.saved
                ? "text-logo-primary hover:text-logo-primary/80"
                : "text-text/50 hover:text-logo-primary"
            }`}
            title={saveLabel}
            aria-label={saveLabel}
            aria-pressed={entry.saved}
          >
            <Star
              width={16}
              height={16}
              fill={entry.saved ? "currentColor" : "none"}
            />
          </button>
          <button
            type="button"
            onClick={handleDeleteEntry}
            className="text-text/50 hover:text-logo-primary transition-colors cursor-pointer"
            title={t("settings.history.delete")}
            aria-label={t("settings.history.delete")}
          >
            <Trash2 width={16} height={16} />
          </button>
        </div>
      </div>
      {entry.post_processed_text ? (
        <div className="space-y-2 pb-2">
          <div>
            <div className="flex items-center gap-2 mb-0.5">
              <span className="text-xs font-medium text-text/50">
                {t("settings.history.postProcessed")}
              </span>
              {entry.post_process_action_key != null && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-logo-primary/10 text-logo-primary font-medium">
                  {t("settings.history.action", {
                    key: entry.post_process_action_key,
                  })}
                </span>
              )}
            </div>
            <p className="italic text-text/90 text-sm select-text cursor-text">
              {entry.post_processed_text}
            </p>
          </div>
          <div>
            <span className="text-xs font-medium text-text/50 mb-0.5 block">
              {t("settings.history.originalTranscript")}
            </span>
            <p className="italic text-text/50 text-sm select-text cursor-text">
              {entry.transcription_text}
            </p>
          </div>
        </div>
      ) : hasText ? (
        <p className="italic text-text/90 text-sm pb-2 select-text cursor-text">
          {entry.transcription_text}
        </p>
      ) : (
        // Dictée dont la transcription a échoué : l'audio est gardé pour
        // pouvoir la relancer avec le bouton « Retraiter ».
        <p className="text-sm pb-2 text-text/50">
          {t("settings.history.noText")}
        </p>
      )}
      <AudioPlayer onLoadRequest={handleLoadAudio} className="w-full" />
    </div>
  );
};
