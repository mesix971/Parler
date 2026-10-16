use anyhow::Result;
use chrono::{DateTime, Local, Utc};
use log::{debug, error, info, warn};
use rusqlite::{params, Connection, OptionalExtension, Row};
use rusqlite_migration::{Migrations, M};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};

use crate::audio_toolkit::save_wav_file;

/// Database migrations for transcription history.
/// Each migration is applied in order. The library tracks which migrations
/// have been applied using SQLite's user_version pragma.
///
/// Note: For users upgrading from tauri-plugin-sql, migrate_from_tauri_plugin_sql()
/// converts the old _sqlx_migrations table tracking to the user_version pragma,
/// ensuring migrations don't re-run on existing databases.
static MIGRATIONS: &[M] = &[
    M::up(
        "CREATE TABLE IF NOT EXISTS transcription_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            saved BOOLEAN NOT NULL DEFAULT 0,
            title TEXT NOT NULL,
            transcription_text TEXT NOT NULL
        );",
    ),
    M::up("ALTER TABLE transcription_history ADD COLUMN post_processed_text TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN post_process_prompt TEXT;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN post_process_action_key INTEGER;"),
    M::up("ALTER TABLE transcription_history ADD COLUMN model_name TEXT;"),
];

/// Colonnes lues pour construire une [`HistoryEntry`] (voir `entry_from_row`).
const ENTRY_COLUMNS: &str = "id, file_name, timestamp, saved, title, transcription_text, post_processed_text, post_process_prompt, post_process_action_key, model_name";

/// Délai d'attente si la base est momentanément verrouillée par une autre
/// connexion (au lieu d'échouer aussitôt avec « database is locked »).
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Âge minimal d'un fichier audio sans entrée en base avant sa suppression
/// (évite de supprimer un enregistrement en cours de sauvegarde).
const ORPHAN_MIN_AGE: Duration = Duration::from_secs(3600);

#[derive(Clone, Debug, Serialize, Deserialize, Type)]
pub struct HistoryEntry {
    pub id: i64,
    pub file_name: String,
    pub timestamp: i64,
    pub saved: bool,
    pub title: String,
    pub transcription_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
    pub post_process_action_key: Option<u8>,
    pub model_name: Option<String>,
}

fn entry_from_row(row: &Row<'_>) -> rusqlite::Result<HistoryEntry> {
    Ok(HistoryEntry {
        id: row.get("id")?,
        file_name: row.get("file_name")?,
        timestamp: row.get("timestamp")?,
        saved: row.get("saved")?,
        title: row.get("title")?,
        transcription_text: row.get("transcription_text")?,
        post_processed_text: row.get("post_processed_text")?,
        post_process_prompt: row.get("post_process_prompt")?,
        post_process_action_key: row
            .get::<_, Option<i64>>("post_process_action_key")?
            .and_then(|v| u8::try_from(v).ok()),
        model_name: row.get("model_name")?,
    })
}

/// Entrées non favorites au-delà des `limit` plus récentes (à supprimer).
fn entries_beyond_limit(conn: &Connection, limit: usize) -> Result<Vec<(i64, String)>> {
    let mut stmt = conn.prepare(
        "SELECT id, file_name FROM transcription_history WHERE saved = 0 ORDER BY timestamp DESC, id DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, i64>("id")?, row.get::<_, String>("file_name")?))
    })?;
    let entries = rows.collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(entries.into_iter().skip(limit).collect())
}

/// Entrées non favorites plus anciennes que `cutoff_timestamp` (en secondes).
fn entries_older_than(conn: &Connection, cutoff_timestamp: i64) -> Result<Vec<(i64, String)>> {
    let mut stmt = conn.prepare(
        "SELECT id, file_name FROM transcription_history WHERE saved = 0 AND timestamp < ?1",
    )?;
    let rows = stmt.query_map(params![cutoff_timestamp], |row| {
        Ok((row.get::<_, i64>("id")?, row.get::<_, String>("file_name")?))
    })?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// Supprime les lignes en une seule transaction (tout ou rien).
fn delete_rows(conn: &mut Connection, ids: &[i64]) -> Result<()> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare("DELETE FROM transcription_history WHERE id = ?1")?;
        for id in ids {
            stmt.execute(params![id])?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Inverse le statut « favori » ; erreur si l'entrée n'existe pas.
fn toggle_saved(conn: &Connection, id: i64) -> Result<bool> {
    let changed = conn.execute(
        "UPDATE transcription_history SET saved = NOT saved WHERE id = ?1",
        params![id],
    )?;
    if changed == 0 {
        anyhow::bail!("History entry not found: {}", id);
    }
    Ok(conn.query_row(
        "SELECT saved FROM transcription_history WHERE id = ?1",
        params![id],
        |row| row.get("saved"),
    )?)
}

/// L'erreur signale-t-elle une base SQLite corrompue ou qui n'en est pas une ?
/// (et non une erreur passagère comme un fichier momentanément verrouillé)
fn is_corruption_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<rusqlite::Error>(),
            Some(rusqlite::Error::SqliteFailure(failure, _))
                if matches!(
                    failure.code,
                    rusqlite::ErrorCode::DatabaseCorrupt | rusqlite::ErrorCode::NotADatabase
                )
        )
    })
}

/// Un nom de fichier venu de l'interface ne doit désigner qu'un fichier du
/// dossier des enregistrements (pas de chemin, pas de lecteur, pas de « .. »).
fn is_plain_file_name(file_name: &str) -> bool {
    !file_name.is_empty()
        && file_name != "."
        && file_name != ".."
        && !file_name.contains(['/', '\\', ':'])
}

pub struct HistoryManager {
    app_handle: AppHandle,
    recordings_dir: PathBuf,
    db_path: PathBuf,
}

impl HistoryManager {
    pub fn new(app_handle: &AppHandle) -> Result<Self> {
        // Create recordings directory in app data dir
        let app_data_dir = app_handle.path().app_data_dir()?;
        let recordings_dir = app_data_dir.join("recordings");
        let db_path = app_data_dir.join("history.db");

        // Ensure recordings directory exists
        if !recordings_dir.exists() {
            fs::create_dir_all(&recordings_dir)?;
            debug!("Created recordings directory: {:?}", recordings_dir);
        }

        let manager = Self {
            app_handle: app_handle.clone(),
            recordings_dir,
            db_path,
        };

        // Initialize database and run migrations synchronously.
        // Erreur passagère (fichier verrouillé par un antivirus…) : on réessaie.
        // Base réellement corrompue : elle est mise de côté (renommée, jamais
        // supprimée) et une base neuve est créée, pour que l'application démarre.
        let mut attempt = 0;
        loop {
            attempt += 1;
            match manager.init_database() {
                Ok(()) => {
                    manager.remove_orphan_recordings();
                    break;
                }
                Err(e) if is_corruption_error(&e) => {
                    error!("History database is corrupted ({:#}), setting it aside", e);
                    manager.set_aside_database()?;
                    manager.init_database()?;
                    break;
                }
                Err(e) if attempt < 3 => {
                    warn!("History database not ready ({:#}), retrying", e);
                    std::thread::sleep(Duration::from_secs(1));
                }
                Err(e) => return Err(e),
            }
        }

        Ok(manager)
    }

    /// Renomme une base corrompue en `history.db.illisible-<date>`, avec ses
    /// fichiers de journal SQLite (`-wal`, `-shm`) pour ne rien perdre.
    fn set_aside_database(&self) -> Result<()> {
        let stamp = Local::now().format("%Y%m%d-%H%M%S");
        for suffix in ["", "-wal", "-shm"] {
            let source = self.db_path.with_file_name(format!("history.db{}", suffix));
            if !source.exists() {
                continue;
            }
            let aside = self
                .db_path
                .with_file_name(format!("history.db.illisible-{}{}", stamp, suffix));
            fs::rename(&source, &aside)?;
            warn!("Corrupted history database file kept as {:?}", aside);
        }
        Ok(())
    }

    fn init_database(&self) -> Result<()> {
        info!("Initializing database at {:?}", self.db_path);

        let mut conn = self.get_connection()?;

        // Mode WAL : lectures et écritures concurrentes sans « database is locked ».
        if let Err(e) = conn.pragma_update(None, "journal_mode", "WAL") {
            warn!("Could not enable WAL journal mode: {}", e);
        }

        // Handle migration from tauri-plugin-sql to rusqlite_migration
        // tauri-plugin-sql used _sqlx_migrations table, rusqlite_migration uses user_version pragma
        self.migrate_from_tauri_plugin_sql(&conn)?;

        // Create migrations object and run to latest version
        let migrations = Migrations::new(MIGRATIONS.to_vec());

        // Validate migrations in debug builds
        #[cfg(debug_assertions)]
        migrations.validate().expect("Invalid migrations");

        // Get current version before migration
        let version_before: i32 =
            conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
        debug!("Database version before migration: {}", version_before);

        // Copie de sécurité avant de modifier le schéma d'une base existante.
        if version_before > 0 && (version_before as usize) < MIGRATIONS.len() {
            let backup = self
                .db_path
                .with_file_name(format!("history.db.avant-migration-v{}", version_before));
            // VACUUM INTO produit une copie cohérente même en mode WAL.
            let result = match backup.to_str() {
                Some(path) if !backup.exists() => conn
                    .execute("VACUUM INTO ?1", params![path])
                    .map(|_| ())
                    .map_err(|e| e.to_string()),
                Some(_) => Ok(()), // copie déjà faite lors d'un essai précédent
                None => Err("non-UTF-8 path".to_string()),
            };
            match result {
                Ok(()) => info!("History database backed up to {:?}", backup),
                Err(e) => warn!("Could not back up history database: {}", e),
            }
        }

        // Apply any pending migrations
        migrations.to_latest(&mut conn)?;

        // Index sur l'ordre d'affichage ; créé hors migrations pour ne pas
        // changer la version de la base (une ancienne version de l'appli
        // refuserait sinon de démarrer après un retour arrière).
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_history_saved_ts ON transcription_history(saved, timestamp DESC);",
        )?;

        // Get version after migration
        let version_after: i32 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;

        if version_after > version_before {
            info!(
                "Database migrated from version {} to {}",
                version_before, version_after
            );
        } else {
            debug!("Database already at latest version {}", version_after);
        }

        Ok(())
    }

    /// Migrate from tauri-plugin-sql's migration tracking to rusqlite_migration's.
    /// tauri-plugin-sql used a _sqlx_migrations table, while rusqlite_migration uses
    /// SQLite's user_version pragma. This function checks if the old system was in use
    /// and sets the user_version accordingly so migrations don't re-run.
    fn migrate_from_tauri_plugin_sql(&self, conn: &Connection) -> Result<()> {
        // Check if the old _sqlx_migrations table exists
        let has_sqlx_migrations: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='_sqlx_migrations'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(false);

        if !has_sqlx_migrations {
            return Ok(());
        }

        // Check current user_version
        let current_version: i32 =
            conn.pragma_query_value(None, "user_version", |row| row.get(0))?;

        if current_version > 0 {
            // Already migrated to rusqlite_migration system
            return Ok(());
        }

        // Get the highest version from the old migrations table
        let old_version: i32 = conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM _sqlx_migrations WHERE success = 1",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if old_version > 0 {
            info!(
                "Migrating from tauri-plugin-sql (version {}) to rusqlite_migration",
                old_version
            );

            // Set user_version to match the old migration state
            conn.pragma_update(None, "user_version", old_version)?;

            info!(
                "Migration tracking converted: user_version set to {}",
                old_version
            );
        }

        Ok(())
    }

    fn get_connection(&self) -> Result<Connection> {
        let conn = Connection::open(&self.db_path)?;
        conn.busy_timeout(BUSY_TIMEOUT)?;
        Ok(conn)
    }

    /// Supprime les fichiers audio qui ne sont plus référencés par aucune
    /// entrée (restes d'une suppression interrompue), s'ils ont plus d'une heure.
    fn remove_orphan_recordings(&self) {
        let referenced: std::collections::HashSet<String> =
            match self.get_connection().and_then(|conn| {
                let mut stmt = conn.prepare("SELECT file_name FROM transcription_history")?;
                let names = stmt
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<_>>()?;
                Ok(names)
            }) {
                Ok(names) => names,
                Err(e) => {
                    warn!("Orphan recordings check skipped: {}", e);
                    return;
                }
            };

        let Ok(dir) = fs::read_dir(&self.recordings_dir) else {
            return;
        };
        let mut removed = 0;
        for entry in dir.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let is_app_recording = (name.starts_with("handy-") || name.starts_with("parler-"))
                && name.ends_with(".wav");
            if !is_app_recording || referenced.contains(&name) {
                continue;
            }
            let old_enough = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|age| age > ORPHAN_MIN_AGE);
            if old_enough && fs::remove_file(entry.path()).is_ok() {
                removed += 1;
            }
        }
        if removed > 0 {
            info!("Removed {} orphan recording file(s)", removed);
        }
    }

    /// Nom de fichier unique pour un nouvel enregistrement (à la milliseconde,
    /// avec suffixe en cas de collision).
    fn new_recording_file_name(&self) -> String {
        let millis = Utc::now().timestamp_millis();
        let mut file_name = format!("parler-{}.wav", millis);
        let mut suffix = 1;
        while self.recordings_dir.join(&file_name).exists() {
            file_name = format!("parler-{}-{}.wav", millis, suffix);
            suffix += 1;
        }
        file_name
    }

    /// Save a transcription to history (both database and WAV file)
    pub async fn save_transcription(
        &self,
        audio_samples: Vec<f32>,
        transcription_text: String,
        post_processed_text: Option<String>,
        post_process_prompt: Option<String>,
        post_process_action_key: Option<u8>,
        model_name: Option<String>,
    ) -> Result<()> {
        let timestamp = Utc::now().timestamp();
        let file_name = self.new_recording_file_name();
        let title = self.format_timestamp_title(timestamp);

        // Save WAV file
        let file_path = self.recordings_dir.join(&file_name);
        save_wav_file(&file_path, &audio_samples).await?;

        // Save to database (sans ligne en base, le fichier audio serait orphelin)
        if let Err(e) = self.save_to_database(
            file_name,
            timestamp,
            title,
            transcription_text,
            post_processed_text,
            post_process_prompt,
            post_process_action_key,
            model_name,
        ) {
            if let Err(remove_err) = fs::remove_file(&file_path) {
                warn!("Could not remove unsaved recording: {}", remove_err);
            }
            return Err(e);
        }

        // Clean up old entries : un échec ici n'annule pas la sauvegarde.
        if let Err(e) = self.cleanup_old_entries() {
            warn!("Failed to clean up old history entries: {}", e);
        }

        // Emit history updated event
        if let Err(e) = self.app_handle.emit("history-updated", ()) {
            error!("Failed to emit history-updated event: {}", e);
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn save_to_database(
        &self,
        file_name: String,
        timestamp: i64,
        title: String,
        transcription_text: String,
        post_processed_text: Option<String>,
        post_process_prompt: Option<String>,
        post_process_action_key: Option<u8>,
        model_name: Option<String>,
    ) -> Result<()> {
        let conn = self.get_connection()?;
        conn.execute(
            "INSERT INTO transcription_history (file_name, timestamp, saved, title, transcription_text, post_processed_text, post_process_prompt, post_process_action_key, model_name) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![file_name, timestamp, false, title, transcription_text, post_processed_text, post_process_prompt, post_process_action_key.map(|k| k as i64), model_name],
        )?;

        debug!("Saved transcription to database");
        Ok(())
    }

    pub fn cleanup_old_entries(&self) -> Result<()> {
        let retention_period = crate::settings::get_recording_retention_period(&self.app_handle);

        match retention_period {
            crate::settings::RecordingRetentionPeriod::Never => {
                // Don't delete anything
                Ok(())
            }
            crate::settings::RecordingRetentionPeriod::PreserveLimit => {
                // Use the old count-based logic with history_limit
                let limit = crate::settings::get_history_limit(&self.app_handle);
                self.cleanup_by_count(limit)
            }
            _ => {
                // Use time-based logic
                self.cleanup_by_time(retention_period)
            }
        }
    }

    /// Supprime des entrées puis leurs fichiers audio. Les lignes sont
    /// supprimées d'un bloc (transaction) ; un fichier qui ne peut pas être
    /// effacé (verrouillé par le lecteur audio…) le sera au prochain démarrage.
    fn delete_entries_and_files(&self, entries: &[(i64, String)]) -> Result<usize> {
        if entries.is_empty() {
            return Ok(0);
        }

        let mut conn = self.get_connection()?;
        let ids: Vec<i64> = entries.iter().map(|(id, _)| *id).collect();
        delete_rows(&mut conn, &ids)?;

        for (_, file_name) in entries {
            if !is_plain_file_name(file_name) {
                continue;
            }
            let file_path = self.recordings_dir.join(file_name);
            if file_path.exists() {
                if let Err(e) = fs::remove_file(&file_path) {
                    error!("Failed to delete WAV file {}: {}", file_name, e);
                } else {
                    debug!("Deleted old WAV file: {}", file_name);
                }
            }
        }

        Ok(entries.len())
    }

    fn cleanup_by_count(&self, limit: usize) -> Result<()> {
        let conn = self.get_connection()?;
        let entries_to_delete = entries_beyond_limit(&conn, limit)?;
        drop(conn);

        let deleted_count = self.delete_entries_and_files(&entries_to_delete)?;
        if deleted_count > 0 {
            debug!("Cleaned up {} old history entries by count", deleted_count);
        }

        Ok(())
    }

    fn cleanup_by_time(
        &self,
        retention_period: crate::settings::RecordingRetentionPeriod,
    ) -> Result<()> {
        // Calculate cutoff timestamp (current time minus retention period)
        let now = Utc::now().timestamp();
        let cutoff_timestamp = match retention_period {
            crate::settings::RecordingRetentionPeriod::Days3 => now - (3 * 24 * 60 * 60), // 3 days in seconds
            crate::settings::RecordingRetentionPeriod::Weeks2 => now - (2 * 7 * 24 * 60 * 60), // 2 weeks in seconds
            crate::settings::RecordingRetentionPeriod::Months3 => now - (3 * 30 * 24 * 60 * 60), // 3 months in seconds (approximate)
            _ => return Ok(()),
        };

        let conn = self.get_connection()?;
        let entries_to_delete = entries_older_than(&conn, cutoff_timestamp)?;
        drop(conn);

        let deleted_count = self.delete_entries_and_files(&entries_to_delete)?;
        if deleted_count > 0 {
            debug!(
                "Cleaned up {} old history entries based on retention period",
                deleted_count
            );
        }

        Ok(())
    }

    pub async fn get_history_entries(&self) -> Result<Vec<HistoryEntry>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(&format!(
            "SELECT {} FROM transcription_history ORDER BY timestamp DESC, id DESC",
            ENTRY_COLUMNS
        ))?;

        let entries = stmt
            .query_map([], entry_from_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        Ok(entries)
    }

    pub fn get_latest_entry(&self) -> Result<Option<HistoryEntry>> {
        let conn = self.get_connection()?;
        Self::get_latest_entry_with_conn(&conn)
    }

    fn get_latest_entry_with_conn(conn: &Connection) -> Result<Option<HistoryEntry>> {
        let entry = conn
            .query_row(
                &format!(
                    "SELECT {} FROM transcription_history ORDER BY timestamp DESC, id DESC LIMIT 1",
                    ENTRY_COLUMNS
                ),
                [],
                entry_from_row,
            )
            .optional()?;

        Ok(entry)
    }

    pub async fn toggle_saved_status(&self, id: i64) -> Result<()> {
        let conn = self.get_connection()?;
        let new_saved = toggle_saved(&conn, id)?;

        debug!("Toggled saved status for entry {}: {}", id, new_saved);

        // Emit history updated event
        if let Err(e) = self.app_handle.emit("history-updated", ()) {
            error!("Failed to emit history-updated event: {}", e);
        }

        Ok(())
    }

    /// Chemin du fichier audio d'une entrée. Refuse tout nom qui sortirait du
    /// dossier des enregistrements.
    pub fn get_audio_file_path(&self, file_name: &str) -> Result<PathBuf> {
        if !is_plain_file_name(file_name) {
            anyhow::bail!("Invalid recording file name: {:?}", file_name);
        }
        Ok(self.recordings_dir.join(file_name))
    }

    pub fn update_transcription_text(
        &self,
        id: i64,
        new_text: &str,
        model_name: Option<&str>,
    ) -> Result<()> {
        let conn = self.get_connection()?;
        conn.execute(
            "UPDATE transcription_history SET transcription_text = ?1, model_name = ?2 WHERE id = ?3",
            params![new_text, model_name, id],
        )?;

        debug!("Updated transcription text for entry {}", id);

        if let Err(e) = self.app_handle.emit("history-updated", ()) {
            error!("Failed to emit history-updated event: {}", e);
        }

        Ok(())
    }

    pub async fn get_entry_by_id(&self, id: i64) -> Result<Option<HistoryEntry>> {
        let conn = self.get_connection()?;
        Self::get_entry_by_id_with_conn(&conn, id)
    }

    fn get_entry_by_id_with_conn(conn: &Connection, id: i64) -> Result<Option<HistoryEntry>> {
        let entry = conn
            .query_row(
                &format!(
                    "SELECT {} FROM transcription_history WHERE id = ?1",
                    ENTRY_COLUMNS
                ),
                [id],
                entry_from_row,
            )
            .optional()?;

        Ok(entry)
    }

    pub async fn delete_entry(&self, id: i64) -> Result<()> {
        let mut conn = self.get_connection()?;

        let entry = Self::get_entry_by_id_with_conn(&conn, id)?;

        // Ligne d'abord, fichier ensuite (même ordre que le nettoyage
        // automatique) : un fichier qui résiste est repris au prochain démarrage.
        delete_rows(&mut conn, &[id])?;

        if let Some(entry) = entry {
            if let Ok(file_path) = self.get_audio_file_path(&entry.file_name) {
                if file_path.exists() {
                    if let Err(e) = fs::remove_file(&file_path) {
                        error!("Failed to delete audio file {}: {}", entry.file_name, e);
                    }
                }
            }
        }

        debug!("Deleted history entry with id: {}", id);

        // Emit history updated event
        if let Err(e) = self.app_handle.emit("history-updated", ()) {
            error!("Failed to emit history-updated event: {}", e);
        }

        Ok(())
    }

    fn format_timestamp_title(&self, timestamp: i64) -> String {
        if let Some(utc_datetime) = DateTime::from_timestamp(timestamp, 0) {
            // Convert UTC to local timezone
            let local_datetime = utc_datetime.with_timezone(&Local);
            local_datetime.format("%B %e, %Y - %l:%M%p").to_string()
        } else {
            format!("Recording {}", timestamp)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::{params, Connection};

    /// Base en mémoire construite avec les VRAIES migrations de l'application.
    fn setup_conn() -> Connection {
        let mut conn = Connection::open_in_memory().expect("open in-memory db");
        Migrations::new(MIGRATIONS.to_vec())
            .to_latest(&mut conn)
            .expect("apply migrations");
        conn
    }

    fn insert_entry(conn: &Connection, timestamp: i64, text: &str, post_processed: Option<&str>) {
        conn.execute(
            "INSERT INTO transcription_history (file_name, timestamp, saved, title, transcription_text, post_processed_text, post_process_prompt, post_process_action_key)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                format!("handy-{}.wav", timestamp),
                timestamp,
                false,
                format!("Recording {}", timestamp),
                text,
                post_processed,
                Option::<String>::None,
                Option::<i64>::None
            ],
        )
        .expect("insert history entry");
    }

    fn count(conn: &Connection) -> i64 {
        conn.query_row("SELECT COUNT(*) FROM transcription_history", [], |r| {
            r.get(0)
        })
        .unwrap()
    }

    #[test]
    fn migrations_are_valid() {
        Migrations::new(MIGRATIONS.to_vec())
            .validate()
            .expect("migrations are valid");
    }

    #[test]
    fn get_latest_entry_returns_none_when_empty() {
        let conn = setup_conn();
        let entry = HistoryManager::get_latest_entry_with_conn(&conn).expect("fetch latest entry");
        assert!(entry.is_none());
    }

    #[test]
    fn get_latest_entry_returns_newest_entry() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "first", None);
        insert_entry(&conn, 200, "second", Some("processed"));

        let entry = HistoryManager::get_latest_entry_with_conn(&conn)
            .expect("fetch latest entry")
            .expect("entry exists");

        assert_eq!(entry.timestamp, 200);
        assert_eq!(entry.transcription_text, "second");
        assert_eq!(entry.post_processed_text.as_deref(), Some("processed"));
    }

    #[test]
    fn same_second_entries_are_ordered_by_id() {
        let conn = setup_conn();
        insert_entry(&conn, 100, "a", None);
        conn.execute(
            "INSERT INTO transcription_history (file_name, timestamp, saved, title, transcription_text) VALUES ('b.wav', 100, 0, 't', 'b')",
            [],
        )
        .unwrap();
        let latest = HistoryManager::get_latest_entry_with_conn(&conn)
            .unwrap()
            .unwrap();
        assert_eq!(latest.transcription_text, "b");
    }

    #[test]
    fn cleanup_by_count_keeps_newest_and_saved_entries() {
        let mut conn = setup_conn();
        for ts in 1..=7 {
            insert_entry(&conn, ts, "x", None);
        }
        // L'entrée la plus ancienne est en favori : jamais supprimée.
        conn.execute(
            "UPDATE transcription_history SET saved = 1 WHERE timestamp = 1",
            [],
        )
        .unwrap();

        let to_delete = entries_beyond_limit(&conn, 3).unwrap();
        let ids: Vec<i64> = to_delete.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids.len(), 3); // 6 non favorites, on en garde 3
        delete_rows(&mut conn, &ids).unwrap();

        assert_eq!(count(&conn), 4);
        let saved: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM transcription_history WHERE saved = 1",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(saved, 1);
    }

    #[test]
    fn cleanup_by_time_selects_only_old_unsaved_entries() {
        let conn = setup_conn();
        insert_entry(&conn, 10, "vieux", None);
        insert_entry(&conn, 1000, "récent", None);
        let old = entries_older_than(&conn, 500).unwrap();
        assert_eq!(old.len(), 1);
        assert_eq!(old[0].1, "handy-10.wav");
    }

    #[test]
    fn toggle_saved_flips_and_rejects_unknown_ids() {
        let conn = setup_conn();
        insert_entry(&conn, 1, "x", None);
        let id: i64 = conn
            .query_row("SELECT id FROM transcription_history", [], |r| r.get(0))
            .unwrap();
        assert!(toggle_saved(&conn, id).unwrap());
        assert!(!toggle_saved(&conn, id).unwrap());
        assert!(toggle_saved(&conn, id + 1000).is_err());
    }

    #[test]
    fn only_real_corruption_sets_the_database_aside() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("history.db");
        fs::write(&path, "ceci n'est pas une base SQLite. ".repeat(10)).expect("write");
        let conn = Connection::open(&path).expect("open");
        let err = conn
            .pragma_query_value(None, "user_version", |r| r.get::<_, i32>(0))
            .expect_err("not a database");
        assert!(is_corruption_error(&anyhow::Error::new(err)));
        assert!(!is_corruption_error(&anyhow::anyhow!("database is locked")));
    }

    #[test]
    fn file_names_cannot_escape_recordings_dir() {
        assert!(is_plain_file_name("parler-123.wav"));
        assert!(!is_plain_file_name("../settings_store.json"));
        assert!(!is_plain_file_name("..\\x.wav"));
        assert!(!is_plain_file_name("C:\\Windows\\x.wav"));
        assert!(!is_plain_file_name("sub/x.wav"));
        assert!(!is_plain_file_name(".."));
        assert!(!is_plain_file_name(""));
    }
}
