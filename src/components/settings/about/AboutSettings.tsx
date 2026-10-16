import React, { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { getVersion } from "@tauri-apps/api/app";
import { openUrl } from "@tauri-apps/plugin-opener";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { SettingContainer } from "../../ui/SettingContainer";
import { Button } from "../../ui/Button";
import { AppDataDirectory } from "../AppDataDirectory";
import { AppLanguageSelector } from "../AppLanguageSelector";
import { ExportImportSettings } from "../ExportImportSettings";
import { LogDirectory } from "../debug";

/** Dépôt du code source de cette version (fork Windows de Parler). */
const SOURCE_CODE_URL = "https://github.com/mesix971/Parler";

export const AboutSettings: React.FC = () => {
  const { t } = useTranslation();
  const [version, setVersion] = useState<string | null>(null);

  useEffect(() => {
    const fetchVersion = async () => {
      try {
        const appVersion = await getVersion();
        setVersion(appVersion);
      } catch (error) {
        console.error("Failed to get app version:", error);
        setVersion("");
      }
    };

    fetchVersion();
  }, []);

  const handleDonateClick = async () => {
    try {
      await openUrl("https://handy.computer/donate");
    } catch (error) {
      console.error("Failed to open donate link:", error);
    }
  };

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <SettingsGroup title={t("settings.about.title")}>
        <AppLanguageSelector descriptionMode="tooltip" grouped={true} />
        <SettingContainer
          title={t("settings.about.version.title")}
          description={t("settings.about.version.description")}
          grouped={true}
        >
          {version ? (
            // eslint-disable-next-line i18next/no-literal-string
            <span className="text-sm font-mono">v{version}</span>
          ) : (
            version !== null && (
              <span className="text-sm text-text/60">
                {t("settings.about.versionUnknown")}
              </span>
            )
          )}
        </SettingContainer>

        <SettingContainer
          title={t("settings.about.supportDevelopment.title")}
          description={t("settings.about.supportDevelopment.description")}
          grouped={true}
        >
          <Button variant="primary" size="md" onClick={handleDonateClick}>
            {t("settings.about.supportDevelopment.button")}
          </Button>
        </SettingContainer>

        <SettingContainer
          title={t("settings.about.sourceCode.title")}
          description={t("settings.about.sourceCode.description")}
          grouped={true}
        >
          <Button
            variant="secondary"
            size="md"
            onClick={() =>
              openUrl(SOURCE_CODE_URL).catch((error) =>
                console.error("Failed to open source code link:", error),
              )
            }
          >
            {t("settings.about.sourceCode.button")}
          </Button>
        </SettingContainer>

        <AppDataDirectory descriptionMode="tooltip" grouped={true} />
        <ExportImportSettings grouped={true} />
        <LogDirectory grouped={true} />
      </SettingsGroup>

      <SettingsGroup title={t("settings.about.acknowledgments.title")}>
        <SettingContainer
          title={t("settings.about.acknowledgments.whisper.title")}
          description={t("settings.about.acknowledgments.whisper.description")}
          grouped={true}
          layout="stacked"
        >
          <div className="text-sm text-mid-gray">
            {t("settings.about.acknowledgments.whisper.details")}
          </div>
        </SettingContainer>
      </SettingsGroup>
    </div>
  );
};
