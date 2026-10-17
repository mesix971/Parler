# Changelog

## [0.11.0] - 2026-10-17

### Ajouté

- Retours à la ligne à la voix : « à la ligne », « retour à la ligne », « nouvelle ligne », « saut de ligne » (et « new line ») insèrent un retour à la ligne, « nouveau paragraphe » une ligne vide, « point à la ligne » un point puis un retour ; le mot suivant prend une majuscule (sauf « iPhone », adresses). Une commande ne compte que suivie d'une ponctuation ou en fin de dictée, pour laisser intacts « va à la ligne 12 » ou « une nouvelle ligne de produits ». Les variantes renvoyées par Gemini sont reconnues (« {à la ligne} » entre accolades, « A la ligne » sans accent).
- Ponctuation à la voix : « point d'exclamation » et « point d'interrogation » deviennent « ! » et « ? ».
- Hésitations françaises retirées du texte (« euh », « euuuh », « heu »…), avec les virgules qui les encadrent et la majuscule reportée sur le mot suivant.

### Modifié

- Modèle Gemini recommandé : `gemini-2.5-flash` (le plus rapide, et il garde les commandes dictées que `gemini-3.5-transcribe` supprime).
- L'historique affiche les retours à la ligne des dictées.
- Transcription à température 0 sur les modèles Gemini 2.x : une même dictée donne toujours le même texte (à 1.0, Gemini variait d'un essai à l'autre). Les Gemini 3 gardent la valeur par défaut, comme le recommande Google.

### Corrigé

- « Ah », « Eh », « ha » et « mm » ne sont plus retirés comme des hésitations anglaises (« Ah bon », « Eh bien », « 5 mm » restaient amputés).

## [0.10.0] - 2026-10-17

### Ajouté

- Choix du modèle Gemini de transcription dans les réglages (Général), avec la liste des modèles réellement disponibles pour la clé ; `gemini-3.5-transcribe`, dédié à la transcription, est signalé comme recommandé.

### Corrigé

- Liste des modèles Gemini complète (elle s'arrêtait aux 50 premiers modèles renvoyés par Google).

## [0.9.0] - 2026-10-16

Version Windows (fork mesix971/Parler), après audit complet.

### Ajouté

- Messages d'erreur dans la pastille de dictée : micro indisponible, rien entendu, clé Gemini invalide, quota dépassé, réseau coupé, délai dépassé, retouche IA en échec, collage impossible.
- La clé API Gemini est vérifiée auprès de Google avant d'être enregistrée.
- Compatible avec `gemini-3.5-transcribe`, le modèle Gemini dédié à la transcription (texte renvoyé dans `audioTranscription`).
- Une dictée dont la transcription échoue reste dans l'historique (audio conservé) pour être retraitée.
- Traduction française complète (60 textes manquants) et textes de l'interface traduits.

### Corrigé

- La clé API n'est plus écrite en clair dans le journal ; le texte dicté non plus.
- Profil de build Windows : LTO « thin » et `panic = "unwind"` (le LTO complet faisait planter whisper.cpp).
- Appels Gemini et LLM bornés par des délais : plus de « Transcription… » sans fin si le réseau cale.
- Le micro ne peut plus geler l'application (ouverture vérifiée, arrêt borné, flux cassé rouvert).
- Pastille fantôme quand le micro ne démarrait pas ; pastille masquée par erreur quand on relançait une dictée aussitôt.
- Animations de la pastille et nom d'action désormais visible (affiché dans la pastille).
- Les refus du backend dans les réglages sont détectés (retour à l'ancienne valeur + message).
- Historique : noms de fichiers uniques à la milliseconde, suppressions transactionnelles, fichiers audio orphelins nettoyés, base illisible mise de côté au lieu de bloquer le démarrage.
- Import de réglages sécurisé (clés, script externe et adresses des fournisseurs conservés) ; export sans clés API.
- Réglages partiellement illisibles réparés champ par champ au lieu d'être tous réinitialisés.

### Amélioré

- Environ 370 ms gagnées à chaque fin de dictée (micro fermé en arrière-plan) et connexion Gemini réutilisée ; réflexion désactivée sur Gemini 2.5 Flash.
- Sécurité de la fenêtre : CSP, accès aux fichiers limité aux enregistrements, droits séparés par fenêtre.
- Accessibilité : navigation au clavier, boutons étiquetés, contrastes de la pastille.
- Dépendance `geist` retirée (alertes de sécurité de Next.js) ; outils de build mis à jour.
- Livrable nommé `Parler-Setup-0.9.0.exe` via `bun run build:windows`.

## [0.3.0] - 2025-07-11

### Added

- **Translate to English** setting: Added automatic translation of speech to English
- Settings refactored into React hooks for better state management
- Audio device switching capability
- Hysteresis to VAD (Voice Activity Detection) for more stable recording

### Changed

- Major audio backend refactor for improved performance and reliability
- Moved audio toolkit into src-tauri directory for better permissions handling
- Model files no longer need to be downloaded separately for releases
- Updated settings components and transcription logic

### Fixed

- Audio toolkit permissions issues
- Various stability improvements

## [0.2.3] - 2025-07-03

### Fixed

- Keycode bug that was causing input issues
- Whisper model optimization: switched to unquantized Whisper Turbo, updated Whisper Medium quantization to 4_1

## [0.2.2] - 2025-07-02

### Fixed

- Removed 50ms delay feature flag for Windows (now applies to all platforms for consistency)

## [0.2.1] - 2025-07-01

### Added

- Ctrl+Space key binding for Windows platform

### Fixed

- Windows crash issue
- Model loading on startup when available
- Windows paste functionality bug

## [0.2.0] - 2025-06-30

### Added

- **Microphone activation on demand**: More efficient resource usage
- Less permissive VAD settings for better accuracy

### Changed

- Improved microphone management and activation system

## [0.1.6] - 2025-06-30

### Added

- **Multiple models support**: Users can now select from different transcription models
- Model selection onboarding flow
- Cleanup and refactoring of model management

### Changed

- Enhanced user experience with model selection interface
- Better language and UI tweaks

## [0.1.5] - 2025-06-27

### Added

- **Different start and stop recording sounds**: Enhanced audio feedback
- Recording sound samples for better user experience

## [0.1.4] - 2025-06-27

### Fixed

- Build issues
- Auto-update functionality improvements

## [0.1.3] - 2025-06-26

### Fixed

- Paste functionality using enigo library for better cross-platform compatibility

## [0.1.2] - 2025-06-26

### Added

- **Auto-update functionality**: Application can now automatically update itself
- Footer displaying current version
- Improved menu system

### Changed

- Better user interface for version management
- Enhanced update workflow

## [0.1.1] - 2025-06-25

### Added

- **Comprehensive build system**: Support for Windows, macOS, and Linux
- Windows code signing for trusted installation
- Ubuntu/Linux build support with Vulkan
- Model file download and packaging for releases
- GitHub Actions CI/CD workflow

### Changed

- Improved build process and release workflow
- Better cross-platform compatibility

### Fixed

- Various build-related issues across platforms

## [0.1.0] - 2025-05-16

### Added

- **Initial release** of Handy
- Basic speech-to-text transcription functionality
- Voice Activity Detection (VAD) for automatic recording
- Cross-platform support (macOS, Windows, Linux)
- **Tauri-based desktop application** with React frontend
- **Global keyboard shortcuts** for activation
- **Clipboard integration** for automatic text insertion
- **LLM integration** for enhanced transcription processing
- **Configurable settings** including:
  - Custom key bindings
  - Audio device selection
  - Microphone settings
  - Push-to-talk functionality
- **System tray integration** with recording indicators
- **Accessibility permissions** handling for macOS
- **Settings persistence** with unified settings store
- **Background operation** capability
- **Multiple audio format support** with on-the-fly resampling
- **Whisper model integration** for high-quality transcription
- **MIT License** for open-source distribution

### Technical Implementation

- Built with Tauri (Rust backend) and React (TypeScript frontend)
- Audio processing with cpal and whisper-rs
- Real-time transcription with performance optimizations
- Cross-platform keyboard event handling
- Modular architecture with managers for audio, models, and transcription
