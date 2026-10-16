use tauri::AppHandle;

#[tauri::command]
#[specta::specta]
pub fn change_gemini_api_key_setting(app: AppHandle, api_key: String) -> Result<(), String> {
    let api_key = api_key.trim().to_string();
    let mut settings = crate::settings::get_settings(&app);
    settings.gemini_api_key = if api_key.is_empty() {
        None
    } else {
        Some(api_key)
    };
    crate::settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_gemini_model_setting(app: AppHandle, model: String) -> Result<(), String> {
    let model = model.trim().to_string();
    // Le nom du modèle finit dans l'URL de l'API : on refuse tout caractère exotique.
    if model.is_empty()
        || !model
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return Err(format!("Invalid Gemini model name: {:?}", model));
    }
    let mut settings = crate::settings::get_settings(&app);
    settings.gemini_model = model;
    crate::settings::write_settings(&app, settings);
    Ok(())
}

/// Modèles Gemini utilisables pour la transcription avec la clé enregistrée
/// (dédiés à la transcription d'abord). En cas d'échec, renvoie un code
/// d'erreur traduit par l'interface.
#[tauri::command]
#[specta::specta]
pub async fn list_gemini_transcription_models(app: AppHandle) -> Result<Vec<String>, String> {
    let api_key = crate::settings::get_settings(&app)
        .gemini_api_key
        .unwrap_or_default();
    crate::gemini_client::list_transcription_models(&api_key)
        .await
        .map_err(|e| {
            log::warn!("Gemini model list failed: {}", e);
            e.kind.code().to_string()
        })
}

/// Vérifie une clé API Gemini auprès de Google avant de l'enregistrer.
/// En cas d'échec, renvoie un code d'erreur traduit par l'interface
/// (« gemini_invalid_key », « network »…).
#[tauri::command]
#[specta::specta]
pub async fn check_gemini_api_key(api_key: String) -> Result<(), String> {
    crate::gemini_client::check_api_key(&api_key)
        .await
        .map_err(|e| {
            log::warn!("Gemini API key check failed: {}", e);
            e.kind.code().to_string()
        })
}
