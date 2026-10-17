use natural::phonetics::soundex;
use once_cell::sync::Lazy;
use regex::Regex;
use strsim::levenshtein;

/// Builds an n-gram string by cleaning and concatenating words
///
/// Strips punctuation from each word, lowercases, and joins without spaces.
/// This allows matching "Charge B" against "ChargeBee".
fn build_ngram(words: &[&str]) -> String {
    words
        .iter()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .collect::<Vec<_>>()
        .concat()
}

/// Finds the best matching custom word for a candidate string
///
/// Uses Levenshtein distance and Soundex phonetic matching to find
/// the best match above the given threshold.
///
/// # Arguments
/// * `candidate` - The cleaned/lowercased candidate string to match
/// * `custom_words` - Original custom words (for returning the replacement)
/// * `custom_words_nospace` - Custom words with spaces removed, lowercased (for comparison)
/// * `threshold` - Maximum similarity score to accept
///
/// # Returns
/// The best matching custom word and its score, if any match was found
fn find_best_match<'a>(
    candidate: &str,
    custom_words: &'a [String],
    custom_words_nospace: &[String],
    threshold: f64,
) -> Option<(&'a String, f64)> {
    if candidate.is_empty() || candidate.len() > 50 {
        return None;
    }

    let mut best_match: Option<&String> = None;
    let mut best_score = f64::MAX;

    for (i, custom_word_nospace) in custom_words_nospace.iter().enumerate() {
        // Skip if lengths are too different (optimization + prevents over-matching)
        // Use percentage-based check: max 25% length difference (prevents n-grams from
        // matching significantly shorter custom words, e.g., "openaigpt" vs "openai")
        let len_diff = (candidate.len() as i32 - custom_word_nospace.len() as i32).abs() as f64;
        let max_len = candidate.len().max(custom_word_nospace.len()) as f64;
        let max_allowed_diff = (max_len * 0.25).max(2.0); // At least 2 chars difference allowed
        if len_diff > max_allowed_diff {
            continue;
        }

        // Calculate Levenshtein distance (normalized by length)
        let levenshtein_dist = levenshtein(candidate, custom_word_nospace);
        let max_len = candidate.len().max(custom_word_nospace.len()) as f64;
        let levenshtein_score = if max_len > 0.0 {
            levenshtein_dist as f64 / max_len
        } else {
            1.0
        };

        // Calculate phonetic similarity using Soundex
        let phonetic_match = soundex(candidate, custom_word_nospace);

        // Combine scores: favor phonetic matches, but also consider string similarity
        let combined_score = if phonetic_match {
            levenshtein_score * 0.3 // Give significant boost to phonetic matches
        } else {
            levenshtein_score
        };

        // Accept if the score is good enough (configurable threshold)
        if combined_score < threshold && combined_score < best_score {
            best_match = Some(&custom_words[i]);
            best_score = combined_score;
        }
    }

    best_match.map(|m| (m, best_score))
}

/// Applies custom word corrections to transcribed text using fuzzy matching
///
/// This function corrects words in the input text by finding the best matches
/// from a list of custom words using a combination of:
/// - Levenshtein distance for string similarity
/// - Soundex phonetic matching for pronunciation similarity
/// - N-gram matching for multi-word speech artifacts (e.g., "Charge B" -> "ChargeBee")
///
/// # Arguments
/// * `text` - The input text to correct
/// * `custom_words` - List of custom words to match against
/// * `threshold` - Maximum similarity score to accept (0.0 = exact match, 1.0 = any match)
///
/// # Returns
/// The corrected text with custom words applied
pub fn apply_custom_words(text: &str, custom_words: &[String], threshold: f64) -> String {
    if custom_words.is_empty() {
        return text.to_string();
    }

    // Pre-compute lowercase versions to avoid repeated allocations
    let custom_words_lower: Vec<String> = custom_words.iter().map(|w| w.to_lowercase()).collect();

    // Pre-compute versions with spaces removed for n-gram comparison
    let custom_words_nospace: Vec<String> = custom_words_lower
        .iter()
        .map(|w| w.replace(' ', ""))
        .collect();

    let words: Vec<&str> = text.split_whitespace().collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < words.len() {
        let mut matched = false;

        // Try n-grams from longest (3) to shortest (1) - greedy matching
        for n in (1..=3).rev() {
            if i + n > words.len() {
                continue;
            }

            let ngram_words = &words[i..i + n];
            let ngram = build_ngram(ngram_words);

            if let Some((replacement, _score)) =
                find_best_match(&ngram, custom_words, &custom_words_nospace, threshold)
            {
                // Extract punctuation from first and last words of the n-gram
                let (prefix, _) = extract_punctuation(ngram_words[0]);
                let (_, suffix) = extract_punctuation(ngram_words[n - 1]);

                // Preserve case from first word
                let corrected = preserve_case_pattern(ngram_words[0], replacement);

                result.push(format!("{}{}{}", prefix, corrected, suffix));
                i += n;
                matched = true;
                break;
            }
        }

        if !matched {
            result.push(words[i].to_string());
            i += 1;
        }
    }

    result.join(" ")
}

/// Preserves the case pattern of the original word when applying a replacement
fn preserve_case_pattern(original: &str, replacement: &str) -> String {
    if original.chars().all(|c| c.is_uppercase()) {
        replacement.to_uppercase()
    } else if original.chars().next().is_some_and(|c| c.is_uppercase()) {
        let mut chars: Vec<char> = replacement.chars().collect();
        if let Some(first_char) = chars.get_mut(0) {
            *first_char = first_char.to_uppercase().next().unwrap_or(*first_char);
        }
        chars.into_iter().collect()
    } else {
        replacement.to_string()
    }
}

/// Extracts punctuation prefix and suffix from a word
fn extract_punctuation(word: &str) -> (&str, &str) {
    let prefix_end = word.chars().take_while(|c| !c.is_alphanumeric()).count();
    let suffix_start = word
        .char_indices()
        .rev()
        .take_while(|(_, c)| !c.is_alphanumeric())
        .count();

    let prefix = if prefix_end > 0 {
        &word[..prefix_end]
    } else {
        ""
    };

    let suffix = if suffix_start > 0 {
        &word[word.len() - suffix_start..]
    } else {
        ""
    };

    (prefix, suffix)
}

/// Filler words to remove from transcriptions
/// (« ah », « eh », « ha » et « mm » retirés de la liste d'origine : en
/// français ce sont des mots — « Ah bon », « Eh bien », « 5 mm ».)
const FILLER_WORDS: &[&str] = &[
    "uh", "um", "uhm", "umm", "uhh", "uhhh", "hmm", "hm", "mmm", "mh", "ehh",
];

static MULTI_SPACE_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s{2,}").unwrap());

/// Collapses repeated 1-2 letter words (3+ repetitions) to a single instance.
/// E.g., "wh wh wh wh" -> "wh", "I I I I" -> "I"
fn collapse_stutters(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }

    let mut result: Vec<&str> = Vec::new();
    let mut i = 0;

    while i < words.len() {
        let word = words[i];
        let word_lower = word.to_lowercase();

        // Only process 1-2 letter words
        if word_lower.len() <= 2 && word_lower.chars().all(|c| c.is_alphabetic()) {
            // Count consecutive repetitions (case-insensitive)
            let mut count = 1;
            while i + count < words.len() && words[i + count].to_lowercase() == word_lower {
                count += 1;
            }

            // If 3+ repetitions, collapse to single instance
            if count >= 3 {
                result.push(word);
                i += count;
            } else {
                result.push(word);
                i += 1;
            }
        } else {
            result.push(word);
            i += 1;
        }
    }

    result.join(" ")
}

/// Pre-compiled filler word patterns (built lazily)
static FILLER_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    FILLER_WORDS
        .iter()
        .map(|word| {
            // Match filler word with word boundaries, optionally followed by comma or period
            Regex::new(&format!(r"(?i)\b{}\b[,.]?", regex::escape(word))).unwrap()
        })
        .collect()
});

/// Hésitations françaises, variantes allongées comprises (« euh », « euuuh »,
/// « heu », « eeeh »). « eu » seul (participe d'avoir) n'est jamais touché.
/// Groupes : `pre` = espaces et virgule avant, `filler` = hésitation, `post` =
/// virgule, points de suspension ou ponctuation finale après (`stop` pour
/// « . », « ! » ou « ? »), `ws` = espaces suivants, `next` = mot suivant.
static FRENCH_FILLER_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)(?P<pre>[ \t]*,?[ \t]*)\b(?P<filler>e+u+h+|h+e+u+h*|e{3,}h*|eu{2,}h*)\b(?P<post>[ \t]*(?:,|…|\.\.\.|(?P<stop>[.!?])))?(?P<ws>[ \t]*)(?P<next>\p{L}\S*)?",
    )
    .unwrap()
});

/// Commandes de mise en page dictées (groupe 1). Elles ne comptent que suivies
/// d'une ponctuation, d'une accolade fermante ou de la fin du texte : c'est
/// ainsi que Gemini les écrit (pause à l'oral), alors que « va à la ligne 12 »
/// ou « une nouvelle ligne de produits » restent du texte. Gemini 3.5
/// Transcribe les encadre parfois (« {à la ligne} ») et Gemini 2.5 Flash
/// écrit « A la ligne » sans accent : ce « A » n'est accepté qu'en majuscule,
/// pour ne pas couper « il a la ligne fixe ».
static LINE_BREAK_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)[ \t]*(?:[,;:][ \t]*)?[{(\[]?[ \t]*\b(point [aà] la ligne|nouveau paragraphe|new paragraph|retour [aà] la ligne|(?:à|(?-i:A)) la ligne|nouvelle ligne|saut de ligne|new line)\b[ \t]*(?:[})\]][ \t]*[.,;:]?|[.,;:]|$)[ \t]*",
    )
    .unwrap()
});

/// Premier mot d'une ligne créée par une commande dictée.
static LINE_START_WORD_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n(\p{L}\S*)").unwrap());

/// Ponctuation dictée que Gemini 2.5 Flash recopie en toutes lettres
/// (« je vais bien point d'exclamation. »). Le point ou la virgule qui
/// l'encadre est absorbé. Groupes : 1 = une ou plusieurs commandes à la
/// suite, 2 = espaces qui suivent, 3 = mot suivant (majuscule après la fin
/// de phrase).
static SPOKEN_PUNCTUATION_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)[ \t]*[.,]?[ \t]*(\bpoint[ \t]+d['’][ \t]*(?:exclamation|interrogation)\b(?:[ \t]*[.,])?(?:[ \t]*\bpoint[ \t]+d['’][ \t]*(?:exclamation|interrogation)\b(?:[ \t]*[.,])?)*)([ \t]*)(\p{L}\S*)?",
    )
    .unwrap()
});

/// Points d'exclamation ou d'interrogation séparés par des espaces
/// (« son? !! ») : on les colle (« son?!! »).
static STACKED_MARKS_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[!?](?:[ \t]+[!?])+").unwrap());

fn starts_uppercase(s: &str) -> bool {
    s.chars().next().is_some_and(|c| c.is_uppercase())
}

/// Met une majuscule au début d'un mot, sauf s'il en porte déjà une à
/// l'intérieur (« iPhone ») ou si c'est une adresse (« https://… »).
fn capitalize_word(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) if !word.contains("://") && !chars.clone().any(char::is_uppercase) => {
            first.to_uppercase().chain(chars).collect()
        }
        _ => word.to_string(),
    }
}

/// Le texte qui précède se termine par une fin de phrase (ou est vide).
fn ends_sentence(before: &str) -> bool {
    before
        .trim_end_matches([' ', '\t'])
        .chars()
        .last()
        .is_none_or(|c| matches!(c, '.' | '!' | '?' | '…' | '\n'))
}

fn remove_french_fillers_once(text: &str) -> String {
    FRENCH_FILLER_PATTERN
        .replace_all(text, |caps: &regex::Captures| {
            let before = &text[..caps.get(0).map_or(0, |m| m.start())];
            let sentence_start = ends_sentence(before);
            let pre = &caps["pre"];
            let ws = &caps["ws"];
            let stop = caps.name("stop").map(|m| m.as_str());
            let lead = if pre.contains([' ', '\t']) { " " } else { "" };
            let comma_before = pre.contains(',');
            let comma_after = caps.name("post").is_some() && stop.is_none();
            let capitalize = starts_uppercase(&caps["filler"])
                || stop.is_some()
                || (sentence_start && !before.trim().is_empty());
            let next = caps.name("next").map(|m| {
                if capitalize {
                    capitalize_word(m.as_str())
                } else {
                    m.as_str().to_string()
                }
            });
            // Ponctuation gardée : « . Merci » avec ses espaces
            let after_stop = format!("{}{}", ws, next.as_deref().unwrap_or_default());
            // Sans mot à la suite (chiffre, guillemet, fin du texte), on garde
            // les espaces qui le séparaient.
            let rest = next.unwrap_or_else(|| ws.to_string());
            match stop {
                // « Euh. Je pense » : la phrase commence au mot suivant
                Some(_) if sentence_start => format!("{}{}", lead, rest),
                // « c'est bon, euh. Merci » : la ponctuation finale reste
                Some(".") => format!(".{}", after_stop),
                Some(mark) => format!(" {}{}", mark, after_stop),
                None => match (comma_before, comma_after) {
                    // « que, euh, la » → « que la »
                    (true, true) => format!(" {}", rest),
                    // « Marc, euh je » → « Marc, je »
                    (true, false) => format!(", {}", rest),
                    // « Marc. Euh, je » → « Marc. Je » ; « que euh la » → « que la »
                    _ => format!("{}{}", lead, rest),
                },
            }
        })
        .into_owned()
}

/// Retire les hésitations françaises. Une hésitation en début de phrase
/// (« Euh, je pense ») transmet sa majuscule au mot suivant (« Je pense »).
/// Les virgules qui l'encadrent (« que, euh, la ») et le point qui la suit en
/// début de phrase (« Euh. Je pense ») disparaissent avec elle.
pub fn remove_french_fillers(text: &str) -> String {
    // Plusieurs passes : dans « euh, euh, je », le mot suivant capturé par la
    // première hésitation est la seconde.
    let mut cleaned = text.to_string();
    for _ in 0..4 {
        let pass = remove_french_fillers_once(&cleaned);
        if pass == cleaned {
            break;
        }
        cleaned = pass;
    }
    if cleaned == text {
        return cleaned;
    }
    // « Bon, euh » en fin de dictée : pas de virgule pendante.
    MULTI_SPACE_PATTERN
        .replace_all(&cleaned, " ")
        .trim()
        .trim_end_matches(',')
        .trim_end()
        .to_string()
}

/// Remplace « point d'exclamation » et « point d'interrogation » dictés par
/// « ! » et « ? », précédés d'une espace à la française (sauf en début de
/// ligne). Plusieurs à la suite se cumulent (« !!! ») et le mot suivant prend
/// une majuscule.
pub fn apply_spoken_punctuation(text: &str) -> String {
    let replaced = SPOKEN_PUNCTUATION_PATTERN.replace_all(text, |caps: &regex::Captures| {
        let before = &text[..caps.get(0).map_or(0, |m| m.start())];
        let space = if before.is_empty() || before.ends_with('\n') {
            ""
        } else {
            " "
        };
        let marks: String = caps[1]
            .to_lowercase()
            .split(|c: char| !c.is_alphabetic())
            .filter_map(|word| match word {
                "exclamation" => Some('!'),
                "interrogation" => Some('?'),
                _ => None,
            })
            .collect();
        let after = match caps.get(3) {
            Some(word) => format!(" {}", capitalize_word(word.as_str())),
            None => caps[2].to_string(),
        };
        format!("{}{}{}", space, marks, after)
    });
    STACKED_MARKS_PATTERN
        .replace_all(&replaced, |caps: &regex::Captures| {
            caps[0]
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
        })
        .trim_matches(|c| c == ' ' || c == '\t')
        .to_string()
}

/// Remplace les commandes de mise en page dictées par de vrais retours à la
/// ligne : « à la ligne », « retour à la ligne », « nouvelle ligne »,
/// « saut de ligne », « new line » → retour à la ligne ; « nouveau
/// paragraphe », « new paragraph » → ligne vide ; « point à la ligne » →
/// point (s'il n'y en a pas déjà un) puis retour à la ligne. Le premier mot de
/// chaque nouvelle ligne prend une majuscule.
pub fn apply_voice_commands(text: &str) -> String {
    let replaced = LINE_BREAK_PATTERN.replace_all(text, |caps: &regex::Captures| {
        let before = &text[..caps.get(0).map_or(0, |m| m.start())];
        let command = caps[1].to_lowercase();
        if command.contains("paragraph") {
            "\n\n"
        } else if command.starts_with("point") && !ends_sentence(before) {
            ".\n"
        } else {
            "\n"
        }
    });
    LINE_START_WORD_PATTERN
        .replace_all(&replaced, |caps: &regex::Captures| {
            format!("\n{}", capitalize_word(&caps[1]))
        })
        .trim_matches(|c| c == ' ' || c == '\t')
        .to_string()
}

/// Filters transcription output by removing filler words and stutter artifacts.
///
/// This function cleans up raw transcription text by:
/// 1. Removing filler words (uh, um, hmm, etc.)
/// 2. Collapsing repeated 1-2 letter stutters (e.g., "wh wh wh" -> "wh")
/// 3. Cleaning up excess whitespace
///
/// # Arguments
/// * `text` - The raw transcription text to filter
///
/// # Returns
/// The filtered text with filler words and stutters removed
pub fn filter_transcription_output(text: &str) -> String {
    let mut filtered = text.to_string();

    // Remove filler words
    for pattern in FILLER_PATTERNS.iter() {
        filtered = pattern.replace_all(&filtered, "").to_string();
    }

    // Collapse repeated 1-2 letter words (stutter artifacts like "wh wh wh wh")
    filtered = collapse_stutters(&filtered);

    // Clean up multiple spaces to single space
    filtered = MULTI_SPACE_PATTERN.replace_all(&filtered, " ").to_string();

    // Hésitations françaises (« euh », « heu »…)
    filtered = remove_french_fillers(filtered.trim());

    // Commandes dictées (« à la ligne », « nouveau paragraphe »…) après le
    // nettoyage des espaces : les retours à la ligne qu'elles insèrent ne
    // doivent plus être fusionnés en espaces, ni retirés en fin de dictée.
    filtered = apply_voice_commands(&filtered);

    // Ponctuation dictée en dernier : passée avant, elle avalerait le point
    // de « À la ligne. Point d'exclamation » et le retour à la ligne
    // mangerait ensuite le « ! » à la place.
    apply_spoken_punctuation(&filtered)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sorties réelles de Gemini sur un message de test lu par la voix
    /// française de Windows (relevées le 2026-10-17).
    #[test]
    fn french_dictation_from_gemini_2_5_flash() {
        let raw = "Salut Marc. Euh, je voulais te dire que, euh, la réunion est décalée à demain. À la ligne. Euh, tu peux prévenir les autres ? Nouveau paragraphe. Merci, à plus.";
        assert_eq!(
            filter_transcription_output(raw),
            "Salut Marc. Je voulais te dire que la réunion est décalée à demain.\nTu peux prévenir les autres ?\n\nMerci, à plus."
        );
    }

    #[test]
    fn french_dictation_from_gemini_3_8_flash() {
        let raw = "Salut Marc, euh je voulais te dire que euh la réunion est décalée à demain. À la ligne, euh tu peux prévenir les autres. Nouveau paragraphe. Merci, à plus.";
        assert_eq!(
            filter_transcription_output(raw),
            "Salut Marc, je voulais te dire que la réunion est décalée à demain.\nTu peux prévenir les autres.\n\nMerci, à plus."
        );
    }

    #[test]
    fn french_fillers_variants_and_false_friends() {
        assert_eq!(remove_french_fillers("Euuuh... bonjour"), "Bonjour");
        assert_eq!(remove_french_fillers("Heu, c'est bon, euh."), "C'est bon.");
        assert_eq!(remove_french_fillers("eeeh je sais pas"), "je sais pas");
        // « eu » (avoir), « heure », « Europe », « beurre » ne sont pas des hésitations
        assert_eq!(
            remove_french_fillers("J'ai eu une heure en Europe avec du beurre."),
            "J'ai eu une heure en Europe avec du beurre."
        );
    }

    #[test]
    fn voice_commands_for_line_breaks() {
        assert_eq!(
            apply_voice_commands("Bonjour, à la ligne, comment ça va"),
            "Bonjour\nComment ça va"
        );
        assert_eq!(
            apply_voice_commands("Bonjour point à la ligne. Merci"),
            "Bonjour.\nMerci"
        );
        assert_eq!(
            apply_voice_commands("Bonjour. Point à la ligne. Merci"),
            "Bonjour.\nMerci"
        );
        assert_eq!(
            apply_voice_commands("Première idée. Retour à la ligne. Seconde idée."),
            "Première idée.\nSeconde idée."
        );
        assert_eq!(
            apply_voice_commands("Hello, new paragraph, see you"),
            "Hello\n\nSee you"
        );
        // Une commande en fin de dictée garde son retour à la ligne
        assert_eq!(
            apply_voice_commands("Fin du message. À la ligne."),
            "Fin du message.\n"
        );
        // Texte sans commande : inchangé
        assert_eq!(
            apply_voice_commands("Rien à signaler, à demain."),
            "Rien à signaler, à demain."
        );
        // « a » (avoir) en minuscule n'est pas une commande
        assert_eq!(
            apply_voice_commands("Il a la ligne fixe."),
            "Il a la ligne fixe."
        );
        // Deux commandes de suite : ligne vide
        assert_eq!(
            apply_voice_commands("Bonjour. À la ligne. À la ligne. Merci"),
            "Bonjour.\n\nMerci"
        );
        assert_eq!(
            apply_voice_commands("Nouvelle ligne, nouvelle ligne"),
            "\n\n"
        );
        // Points de suspension gardés tels quels
        assert_eq!(
            apply_voice_commands("Je sais pas... À la ligne. Bref"),
            "Je sais pas...\nBref"
        );
        // Pas de majuscule forcée sur « iPhone » ni sur une adresse
        assert_eq!(
            apply_voice_commands("Regarde. À la ligne. iPhone et à la ligne, https://x.fr"),
            "Regarde.\niPhone et\nhttps://x.fr"
        );
    }

    /// Les commandes ne comptent que suivies d'une ponctuation, d'une accolade
    /// ou de la fin du texte : ailleurs, ce sont des mots.
    #[test]
    fn voice_commands_false_friends() {
        for phrase in [
            "Va à la ligne 12 du fichier.",
            "Tape ça à la ligne de commande.",
            "Ajoute une nouvelle ligne au tableau.",
            "Une nouvelle ligne de produits arrive.",
            "Il y a un point à la ligne 3.",
            "Tu peux revenir à la ligne ?",
        ] {
            assert_eq!(apply_voice_commands(phrase), phrase);
        }
    }

    #[test]
    fn french_fillers_edge_cases() {
        // Hésitations doublées
        assert_eq!(remove_french_fillers("Je sais euh, euh pas"), "Je sais pas");
        assert_eq!(remove_french_fillers("Euh, euh, je pense"), "Je pense");
        // Point qui suit une hésitation en début de phrase
        assert_eq!(
            remove_french_fillers("Euh. Je pense que ça marche."),
            "Je pense que ça marche."
        );
        assert_eq!(
            remove_french_fillers("À demain. Euh. Je file."),
            "À demain. Je file."
        );
        assert_eq!(remove_french_fillers("Euh."), "");
        // Point final gardé ailleurs
        assert_eq!(
            remove_french_fillers("C'est bon, euh. Merci"),
            "C'est bon. Merci"
        );
        assert_eq!(remove_french_fillers("Tu viens euh ?"), "Tu viens ?");
        // Majuscule après une fin de phrase, même si l'hésitation n'en a pas
        assert_eq!(
            remove_french_fillers("Salut. euh je file"),
            "Salut. Je file"
        );
        // Pas de virgule pendante en fin de dictée
        assert_eq!(remove_french_fillers("Bon, euh"), "Bon");
        // Un chiffre qui suit garde son espace
        assert_eq!(
            remove_french_fillers("J'ai euh 3 enfants"),
            "J'ai 3 enfants"
        );
        // Ponctuation sans hésitation : intacte (« .env », « ...args »)
        assert_eq!(
            remove_french_fillers("Ouvre le fichier .env et foo(a, ...args)"),
            "Ouvre le fichier .env et foo(a, ...args)"
        );
    }

    /// Mots français que la liste d'hésitations anglaise d'origine abîmait.
    #[test]
    fn french_words_kept_by_english_filler_list() {
        for phrase in [
            "Eh bien, on y va.",
            "Ah bon ? Ah, Marc est là.",
            "Il est tombé 5 mm de pluie.",
            "Un terrain de 10 ha.",
        ] {
            assert_eq!(filter_transcription_output(phrase), phrase);
        }
    }

    /// Vraies dictées du 2026-10-17, telles que rendues par Gemini 3.5
    /// Transcribe (commandes parfois entre accolades) puis Gemini 2.5 Flash
    /// (« A » sans accent, ponctuation recopiée en toutes lettres).
    #[test]
    fn real_dictations_from_gemini_3_5_transcribe() {
        assert_eq!(
            filter_transcription_output(
                "OK, je vois ce que tu veux dire. {à la ligne} Je vais très bien. {à la ligne} Ça va le son ?"
            ),
            "OK, je vois ce que tu veux dire.\nJe vais très bien.\nÇa va le son ?"
        );
        assert_eq!(
            filter_transcription_output("Ça va à la ligne. Tu vas bien à la ligne."),
            "Ça va\nTu vas bien\n"
        );
    }

    #[test]
    fn real_dictations_from_gemini_2_5_flash() {
        assert_eq!(
            filter_transcription_output("Ça va à la ligne. Tu vas bien ? À la ligne."),
            "Ça va\nTu vas bien ?\n"
        );
        assert_eq!(
            filter_transcription_output(
                "Hello, ça va? A la ligne. Est-ce que tu vas très très bien? A la ligne. Ouais, je vais bien point d'exclamation. OK, est-ce que tu as passé une bonne journée point d'interrogation."
            ),
            "Hello, ça va?\nEst-ce que tu vas très très bien?\nOuais, je vais bien ! OK, est-ce que tu as passé une bonne journée ?"
        );
        assert_eq!(
            filter_transcription_output(
                "Alors qu'est-ce que tu veux dire à la ligne. Je vais très bien. À la ligne. point d'exclamation point d'exclamation point d'exclamation. Ça va le son? Point d'exclamation point d'exclamation."
            ),
            "Alors qu'est-ce que tu veux dire\nJe vais très bien.\n!!! Ça va le son?!!"
        );
    }

    #[test]
    fn spoken_punctuation() {
        assert_eq!(
            apply_spoken_punctuation("Tu viens point d'interrogation"),
            "Tu viens ?"
        );
        assert_eq!(
            apply_spoken_punctuation("Génial, point d'exclamation merci"),
            "Génial ! Merci"
        );
        assert_eq!(
            apply_spoken_punctuation("Vraiment point d’interrogation point d’exclamation"),
            "Vraiment ?!"
        );
        // Un chiffre qui suit garde son espace
        assert_eq!(
            apply_spoken_punctuation("Bravo point d'exclamation 3 buts"),
            "Bravo ! 3 buts"
        );
        // Texte sans commande : inchangé (« points d'exclamation » est un nom)
        assert_eq!(
            apply_spoken_punctuation("Évite les points d'exclamation."),
            "Évite les points d'exclamation."
        );
        // Sortie réelle de Gemini 2.5 Flash à température 0 (virgules entre
        // les commandes, point de fin de phrase avant)
        assert_eq!(
            apply_spoken_punctuation(
                "Je vais très bien. Point d'exclamation, point d'exclamation, point d'exclamation. Ça va le son ? Point d'exclamation, point d'exclamation."
            ),
            "Je vais très bien !!! Ça va le son ?!!"
        );
        // Caractères accentués juste avant la commande
        assert_eq!(apply_spoken_punctuation("Été point d'exclamation"), "Été !");
    }

    #[test]
    fn test_apply_custom_words_exact_match() {
        let text = "hello world";
        let custom_words = vec!["Hello".to_string(), "World".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_apply_custom_words_fuzzy_match() {
        let text = "helo wrold";
        let custom_words = vec!["hello".to_string(), "world".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_preserve_case_pattern() {
        assert_eq!(preserve_case_pattern("HELLO", "world"), "WORLD");
        assert_eq!(preserve_case_pattern("Hello", "world"), "World");
        assert_eq!(preserve_case_pattern("hello", "WORLD"), "WORLD");
    }

    #[test]
    fn test_extract_punctuation() {
        assert_eq!(extract_punctuation("hello"), ("", ""));
        assert_eq!(extract_punctuation("!hello?"), ("!", "?"));
        assert_eq!(extract_punctuation("...hello..."), ("...", "..."));
    }

    #[test]
    fn test_empty_custom_words() {
        let text = "hello world";
        let custom_words = vec![];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_filter_filler_words() {
        let text = "So um I was thinking uh about this";
        let result = filter_transcription_output(text);
        assert_eq!(result, "So I was thinking about this");
    }

    #[test]
    fn test_filter_filler_words_case_insensitive() {
        let text = "UM this is UH a test";
        let result = filter_transcription_output(text);
        assert_eq!(result, "this is a test");
    }

    #[test]
    fn test_filter_filler_words_with_punctuation() {
        let text = "Well, um, I think, uh. that's right";
        let result = filter_transcription_output(text);
        assert_eq!(result, "Well, I think, that's right");
    }

    #[test]
    fn test_filter_cleans_whitespace() {
        let text = "Hello    world   test";
        let result = filter_transcription_output(text);
        assert_eq!(result, "Hello world test");
    }

    #[test]
    fn test_filter_trims() {
        let text = "  Hello world  ";
        let result = filter_transcription_output(text);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_filter_combined() {
        let text = "  Um, so I was, uh, thinking about this  ";
        let result = filter_transcription_output(text);
        assert_eq!(result, "so I was, thinking about this");
    }

    #[test]
    fn test_filter_preserves_valid_text() {
        let text = "This is a completely normal sentence.";
        let result = filter_transcription_output(text);
        assert_eq!(result, "This is a completely normal sentence.");
    }

    #[test]
    fn test_filter_stutter_collapse() {
        let text = "w wh wh wh wh wh wh wh wh wh why";
        let result = filter_transcription_output(text);
        assert_eq!(result, "w wh why");
    }

    #[test]
    fn test_filter_stutter_short_words() {
        let text = "I I I I think so so so so";
        let result = filter_transcription_output(text);
        assert_eq!(result, "I think so");
    }

    #[test]
    fn test_filter_stutter_mixed_case() {
        let text = "No NO no NO no";
        let result = filter_transcription_output(text);
        assert_eq!(result, "No");
    }

    #[test]
    fn test_filter_stutter_preserves_two_repetitions() {
        let text = "no no is fine";
        let result = filter_transcription_output(text);
        assert_eq!(result, "no no is fine");
    }

    #[test]
    fn test_apply_custom_words_ngram_two_words() {
        let text = "il cui nome è Charge B, che permette";
        let custom_words = vec!["ChargeBee".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert!(result.contains("ChargeBee,"));
        assert!(!result.contains("Charge B"));
    }

    #[test]
    fn test_apply_custom_words_ngram_three_words() {
        let text = "use Chat G P T for this";
        let custom_words = vec!["ChatGPT".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert!(result.contains("ChatGPT"));
    }

    #[test]
    fn test_apply_custom_words_prefers_longer_ngram() {
        let text = "Open AI GPT model";
        let custom_words = vec!["OpenAI".to_string(), "GPT".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "OpenAI GPT model");
    }

    #[test]
    fn test_apply_custom_words_ngram_preserves_case() {
        let text = "CHARGE B is great";
        let custom_words = vec!["ChargeBee".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert!(result.contains("CHARGEBEE"));
    }

    #[test]
    fn test_apply_custom_words_ngram_with_spaces_in_custom() {
        // Custom word with space should also match against split words
        let text = "using Mac Book Pro";
        let custom_words = vec!["MacBook Pro".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert!(result.contains("MacBook"));
    }

    #[test]
    fn test_apply_custom_words_trailing_number_not_doubled() {
        // Verify that trailing non-alpha chars (like numbers) aren't double-counted
        // between build_ngram stripping them and extract_punctuation capturing them
        let text = "use GPT4 for this";
        let custom_words = vec!["GPT-4".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        // Should NOT produce "GPT-44" (double-counting the trailing 4)
        assert!(
            !result.contains("GPT-44"),
            "got double-counted result: {}",
            result
        );
    }
}
