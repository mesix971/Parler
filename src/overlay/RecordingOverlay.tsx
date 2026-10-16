import { listen } from "@tauri-apps/api/event";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import "./RecordingOverlay.css";
import { commands } from "@/bindings";
import i18n, { syncLanguageFromSettings } from "@/i18n";
import { getLanguageDirection } from "@/lib/utils/rtl";

type OverlayState = "recording" | "transcribing" | "processing" | "error";

/** Codes d'erreur envoyés par le backend (événement « overlay-error »). */
const KNOWN_ERROR_CODES = [
  "mic_unavailable",
  "no_speech",
  "model_not_loaded",
  "gemini_invalid_key",
  "gemini_quota",
  "gemini_unavailable",
  "gemini_blocked",
  "gemini_model_unavailable",
  "timeout",
  "network",
  "post_process_failed",
  "post_process_not_configured",
  "paste_failed",
  "transcription_failed",
] as const;

interface ActionInfo {
  key: number;
  name: string;
}

/** Durée d'affichage du nom d'une action choisie pendant l'enregistrement. */
const ACTION_NAME_DURATION_MS = 2000;

const MicIcon: React.FC = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="rgba(255,255,255,0.8)"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
    <line x1="12" x2="12" y1="19" y2="22" />
  </svg>
);

const DotsIcon: React.FC = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="rgba(255,255,255,0.7)"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>
);

const AlertIcon: React.FC = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="#ff8a8a"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <circle cx="12" cy="12" r="10" />
    <line x1="12" x2="12" y1="8" y2="12" />
    <line x1="12" x2="12.01" y1="16" y2="16" />
  </svg>
);

const XIcon: React.FC = () => (
  <svg
    width="12"
    height="12"
    viewBox="0 0 24 24"
    fill="none"
    stroke="rgba(255,255,255,0.7)"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <line x1="18" x2="6" y1="6" y2="18" />
    <line x1="6" x2="18" y1="6" y2="18" />
  </svg>
);

const PauseIcon: React.FC = () => (
  <svg
    width="12"
    height="12"
    viewBox="0 0 24 24"
    fill="rgba(255,255,255,0.7)"
    stroke="none"
    aria-hidden="true"
  >
    <rect x="6" y="4" width="4" height="16" rx="1" />
    <rect x="14" y="4" width="4" height="16" rx="1" />
  </svg>
);

const PlayIcon: React.FC = () => (
  <svg
    width="12"
    height="12"
    viewBox="0 0 24 24"
    fill="rgba(255,255,255,0.7)"
    stroke="none"
    aria-hidden="true"
  >
    <polygon points="6,4 20,12 6,20" />
  </svg>
);

const formatTime = (s: number) => {
  const min = Math.floor(s / 60);
  const sec = s % 60;
  return `${min}:${sec.toString().padStart(2, "0")}`;
};

const TimerDisplay: React.FC<{ startTime: number; isPaused: boolean }> = ({
  startTime,
  isPaused,
}) => {
  const [display, setDisplay] = useState("0:00");

  useEffect(() => {
    if (isPaused) {
      return;
    }
    // Le texte change une fois par seconde : un rafraîchissement toutes les
    // 250 ms suffit (au lieu de 60 fois par seconde).
    const tick = () => {
      const elapsed = Math.max(0, Math.floor((Date.now() - startTime) / 1000));
      setDisplay(formatTime(elapsed));
    };
    tick();
    const interval = setInterval(tick, 250);
    return () => clearInterval(interval);
  }, [startTime, isPaused]);

  return <div className="timer-text">{display}</div>;
};

const AudioBars: React.FC = () => {
  const barsRef = useRef<HTMLDivElement>(null);
  const smoothedRef = useRef<number[]>(Array(16).fill(0));

  useEffect(() => {
    let isMounted = true;
    let unlisten: (() => void) | null = null;

    listen<number[]>("mic-level", (event) => {
      const newLevels = event.payload;
      const smoothed = smoothedRef.current.map((prev, i) => {
        const target = newLevels[i] || 0;
        return prev * 0.65 + target * 0.35;
      });
      smoothedRef.current = smoothed;

      if (barsRef.current) {
        const bars = barsRef.current.children;
        for (let i = 0; i < bars.length; i++) {
          const v = smoothed[i] || 0;
          const el = bars[i] as HTMLElement;
          el.style.height = `${Math.min(20, 3 + Math.pow(v, 0.6) * 17)}px`;
          el.style.opacity = `${Math.max(0.25, v * 1.4)}`;
        }
      }
    })
      .then((fn) => {
        // Démonté avant la fin de l'inscription : on se désinscrit aussitôt.
        if (isMounted) {
          unlisten = fn;
        } else {
          fn();
        }
      })
      .catch((e) => console.error("Failed to listen to mic levels:", e));

    return () => {
      isMounted = false;
      unlisten?.();
    };
  }, []);

  return (
    <div className="bars-container" ref={barsRef} aria-hidden="true">
      {Array.from({ length: 9 }, (_, i) => (
        <div key={i} className="bar" />
      ))}
    </div>
  );
};

const RecordingOverlay: React.FC = () => {
  const { t } = useTranslation();
  const [isVisible, setIsVisible] = useState(false);
  const [state, setState] = useState<OverlayState>("recording");
  const [errorCode, setErrorCode] = useState<string>("transcription_failed");
  const [timerStart, setTimerStart] = useState(() => Date.now());
  const [selectedAction, setSelectedAction] = useState<ActionInfo | null>(null);
  const [showActionName, setShowActionName] = useState(false);
  const [cancelPending, setCancelPending] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const cancelTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const actionNameTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pauseStartRef = useRef<number>(0);
  const direction = getLanguageDirection(i18n.language);

  const handleCancel = useCallback(() => {
    commands
      .cancelOperation()
      .catch((e) => console.error("Failed to cancel:", e));
  }, []);

  const handleTogglePause = useCallback(() => {
    commands
      .togglePause()
      .catch((e) => console.error("Failed to toggle pause:", e));
  }, []);

  useEffect(() => {
    let isMounted = true;
    let cleanupListeners: (() => void) | undefined;

    const clearActionNameTimer = () => {
      if (actionNameTimerRef.current) {
        clearTimeout(actionNameTimerRef.current);
        actionNameTimerRef.current = null;
      }
    };

    const setupEventListeners = async () => {
      const unlistenShow = await listen<OverlayState>(
        "show-overlay",
        (event) => {
          // L'état est appliqué tout de suite : attendre la synchro de la
          // langue (aller-retour avec le backend) retardait l'affichage et
          // pouvait écraser un « hide-overlay » arrivé entre-temps.
          const overlayState = event.payload;
          setState(overlayState);
          setIsVisible(true);
          setIsPaused(false);
          if (overlayState === "recording") {
            setTimerStart(Date.now());
            setSelectedAction(null);
            setShowActionName(false);
            clearActionNameTimer();
          }
          void syncLanguageFromSettings();
        },
      );

      const unlistenError = await listen<string>("overlay-error", (event) => {
        setErrorCode(event.payload);
        setState("error");
        setIsVisible(true);
        setCancelPending(false);
      });

      const unlistenHide = await listen("hide-overlay", () => {
        setIsVisible(false);
        setSelectedAction(null);
        setShowActionName(false);
        setCancelPending(false);
        setIsPaused(false);
        clearActionNameTimer();
        if (cancelTimerRef.current) {
          clearTimeout(cancelTimerRef.current);
          cancelTimerRef.current = null;
        }
      });

      const unlistenCancelPending = await listen("cancel-pending", () => {
        setCancelPending(true);
        if (cancelTimerRef.current) {
          clearTimeout(cancelTimerRef.current);
        }
        cancelTimerRef.current = setTimeout(() => {
          setCancelPending(false);
          cancelTimerRef.current = null;
        }, 1700);
      });

      const unlistenAction = await listen<ActionInfo>(
        "action-selected",
        (event) => {
          setSelectedAction(event.payload);
          setShowActionName(true);
          clearActionNameTimer();
          actionNameTimerRef.current = setTimeout(() => {
            setShowActionName(false);
            actionNameTimerRef.current = null;
          }, ACTION_NAME_DURATION_MS);
        },
      );

      const unlistenDeselect = await listen("action-deselected", () => {
        setSelectedAction(null);
        setShowActionName(false);
        clearActionNameTimer();
      });

      const unlistenPause = await listen<boolean>(
        "recording-paused",
        (event) => {
          const paused = event.payload;
          setIsPaused(paused);
          if (paused) {
            pauseStartRef.current = Date.now();
          } else {
            const pauseDuration = Date.now() - pauseStartRef.current;
            setTimerStart((prev) => prev + pauseDuration);
          }
        },
      );

      const unlistenAll = () => {
        unlistenShow();
        unlistenError();
        unlistenHide();
        unlistenCancelPending();
        unlistenAction();
        unlistenDeselect();
        unlistenPause();
      };

      if (!isMounted) {
        unlistenAll();
        return;
      }

      cleanupListeners = unlistenAll;
    };

    setupEventListeners().catch((e) =>
      console.error("Failed to set up overlay listeners:", e),
    );
    return () => {
      isMounted = false;
      cleanupListeners?.();
      if (cancelTimerRef.current) {
        clearTimeout(cancelTimerRef.current);
      }
      clearActionNameTimer();
    };
  }, []);

  const isRecording = state === "recording";
  // Composants animés montés uniquement quand l'overlay est visible : rien ne
  // tourne en arrière-plan entre deux dictées.
  const showLiveRecording = isVisible && isRecording && !cancelPending;
  const errorKey = (KNOWN_ERROR_CODES as readonly string[]).includes(errorCode)
    ? errorCode
    : "transcription_failed";

  const leftIcon = isRecording ? (
    <MicIcon />
  ) : state === "error" ? (
    <AlertIcon />
  ) : (
    <DotsIcon />
  );

  return (
    <div
      dir={direction}
      className={`overlay-wrapper ${isVisible ? "is-visible" : "is-hidden"}`}
    >
      <div className={`recording-overlay state-${state}`}>
        <div className="overlay-left">{leftIcon}</div>

        {selectedAction && isRecording && (
          <div className="action-badge" aria-hidden="true">
            {selectedAction.key}
          </div>
        )}

        <div className="overlay-middle" aria-live="polite">
          {showLiveRecording && showActionName && selectedAction && (
            <div className="action-name" title={selectedAction.name}>
              {selectedAction.name}
            </div>
          )}
          {showLiveRecording && !(showActionName && selectedAction) && (
            <>
              <TimerDisplay startTime={timerStart} isPaused={isPaused} />
              <AudioBars />
            </>
          )}
          {isRecording && cancelPending && (
            <div className="cancel-confirm-text">
              {t("overlay.cancelConfirm")}
            </div>
          )}
          {state === "transcribing" && (
            <div className="transcribing-text">{t("overlay.transcribing")}</div>
          )}
          {state === "processing" && (
            <div className="transcribing-text">{t("overlay.processing")}</div>
          )}
          {state === "error" && (
            <div
              className="error-text"
              role="alert"
              title={t(`overlay.errors.${errorKey}`)}
            >
              {t(`overlay.errors.${errorKey}`)}
            </div>
          )}
        </div>

        <div className="overlay-right">
          {isRecording && (
            <>
              <button
                type="button"
                className="overlay-button"
                onClick={handleTogglePause}
                aria-label={isPaused ? t("overlay.resume") : t("overlay.pause")}
                title={isPaused ? t("overlay.resume") : t("overlay.pause")}
              >
                {isPaused ? <PlayIcon /> : <PauseIcon />}
              </button>
              <button
                type="button"
                className="overlay-button"
                onClick={handleCancel}
                aria-label={t("overlay.cancel")}
                title={t("overlay.cancel")}
              >
                <XIcon />
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default RecordingOverlay;
