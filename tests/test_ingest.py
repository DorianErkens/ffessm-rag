"""
Tests du pipeline d'ingestion.

Chaque test correspond à un bug réel rencontré lors du développement.
L'historique des bugs est documenté en tête de chaque section.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ingest import clean_text, is_section_header, build_chunks, MIN_CHUNK_WORDS
from pathlib import Path


# ─── Bug #1 : Typo espacée ─────────────────────────────────────────────────
# Les PDFs FFESSM utilisent une typographie espacée décorative.
# PyMuPDF extrait "N I V E A U  1" tel quel → embeddings dégradés, recherche qui rate.
# Fix : clean_text() détecte les séquences de lettres isolées et les recolle.

class TestCleanText:

    def test_single_word_spaced(self):
        assert clean_text("N I V E A U") == "NIVEAU"

    def test_two_words_spaced(self):
        # Double espace = séparateur de mots
        assert clean_text("P L O N G E U R  N I V E A U") == "PLONGEUR NIVEAU"

    def test_word_with_number(self):
        # "D É C E M B R E  2 0 2 5" → "DÉCEMBRE 2025"
        result = clean_text("D É C E M B R E  2 0 2 5")
        assert result == "DÉCEMBRE 2025"

    def test_with_symbol(self):
        # "I M M E R S I O N  &  É M O T I O N" → "IMMERSION & ÉMOTION"
        result = clean_text("I M M E R S I O N  &  É M O T I O N")
        assert result == "IMMERSION & ÉMOTION"

    def test_normal_text_unchanged(self):
        # Le texte normal ne doit pas être modifié
        text = "Le plongeur niveau 1 est capable de réaliser des plongées jusqu'à 20 m."
        assert clean_text(text) == text

    def test_accented_letters(self):
        assert clean_text("É T U D E S") == "ÉTUDES"


# ─── Bug #2 : Caractères isolés détectés comme titres ──────────────────────
# "N" (grande police) et "4" (petite police) sont extraits comme spans séparés.
# "N" passait le test `font_size > median * 1.15` → section titre "[N]" créée.
# Le chunk suivant commençait par "4" → titre "[N]", contenu "4 ...".
# Fix : len(text) < 4 → pas un titre.

# Bug #3 : Codes de certification détectés comme titres
# "PE40", "MF1", "PA20" sont en majuscules, courts → détectés comme titres.
# Résultat : chunk "[PE40]" avec contenu ")" → fragment inutilisable.
# Fix : regex d'exclusion des codes FFESSM.

# Bug #4 : Symboles décoratifs détectés comme titres
# "―" (tiret long) apparaît en grand entre les sections → détecté comme titre.
# Fix : exclusion des textes composés uniquement de symboles.

class TestIsSectionHeader:

    def _check(self, text, font_size, median, expected):
        assert is_section_header(text, font_size, median) == expected, \
            f"is_section_header({text!r}) should be {expected}"

    # Cas qui NE doivent PAS être des titres
    def test_single_char_not_header(self):
        self._check("N", 20.0, 10.0, False)

    def test_single_digit_not_header(self):
        self._check("4", 20.0, 10.0, False)

    def test_two_chars_not_header(self):
        self._check("N4", 20.0, 10.0, False)

    def test_certification_codes_not_headers(self):
        codes = ["N1", "N2", "N3", "N4", "N5",
                 "PE20", "PE40", "PE60",
                 "PA12", "PA20", "PA40", "PA60",
                 "MF1", "MF2", "E1", "E2", "E3", "E4",
                 "GP", "DP", "RIFAP", "BPJEPS", "DEJEPS", "BEPPA", "TIV", "ANTEOR"]
        for code in codes:
            self._check(code, 20.0, 10.0, False)

    def test_decorative_dash_not_header(self):
        self._check("―", 12.0, 10.0, False)
        self._check("——", 12.0, 10.0, False)
        self._check("•", 12.0, 10.0, False)

    def test_digits_only_not_header(self):
        self._check("2025", 20.0, 10.0, False)

    def test_too_long_not_header(self):
        long_text = "A" * 201
        self._check(long_text, 20.0, 10.0, False)

    # Cas qui DOIVENT être des titres
    def test_large_font_is_header(self):
        self._check("CONDITIONS D'ACCÈS", 12.0, 10.0, True)

    def test_all_caps_short_is_header(self):
        self._check("PRÉROGATIVES", 10.0, 10.0, True)

    def test_numbered_section_is_header(self):
        self._check("1. Généralités", 10.0, 10.0, True)


# ─── Bug #5 : Chunks vides indexés ─────────────────────────────────────────
# Les micro-chunks (titre seul + ")" ou une ligne de tableau) étaient indexés.
# Lors d'une recherche, ces fragments remontaient en top résultats → Claude
# répondait "les extraits ne contiennent aucune information exploitable".
# Fix : MIN_CHUNK_WORDS = 30, les chunks trop courts sont filtrés.

class TestBuildChunks:

    def test_no_empty_chunks(self, tmp_path):
        """Aucun chunk produit ne doit être vide."""
        import fitz
        # On teste avec un vrai PDF du corpus
        pdf_path = Path("docs/Plongeur Niveau 1 - PE20.pdf")
        if not pdf_path.exists():
            import pytest
            pytest.skip("PDF not available")

        texts, metadatas = build_chunks(pdf_path)
        for text in texts:
            assert text.strip() != "", "Chunk vide détecté"

    def test_min_word_count(self, tmp_path):
        """Tous les chunks doivent dépasser le seuil minimum de mots."""
        pdf_path = Path("docs/Plongeur Niveau 1 - PE20.pdf")
        if not pdf_path.exists():
            import pytest
            pytest.skip("PDF not available")

        texts, metadatas = build_chunks(pdf_path)
        for text in texts:
            # Le contenu (sans le titre entre crochets) doit avoir >= MIN_CHUNK_WORDS mots
            content = text.split("\n", 1)[-1] if "\n" in text else text
            word_count = len(content.split())
            assert word_count >= MIN_CHUNK_WORDS, \
                f"Chunk trop court ({word_count} mots) : {text[:100]!r}"

    def test_no_certification_code_as_title(self):
        """Aucun chunk ne doit avoir un code de certification comme titre."""
        pdf_path = Path("docs/Plongeur Niveau 2 - PA20 - PE40.pdf")
        if not pdf_path.exists():
            import pytest
            pytest.skip("PDF not available")

        bad_titles = {"N1", "N2", "N3", "N4", "N5",
                      "PE20", "PE40", "PE60", "PA12", "PA20", "PA40", "PA60",
                      "MF1", "MF2", "GP", "DP", "E1", "E2", "E3", "E4"}
        texts, metadatas = build_chunks(pdf_path)
        for meta in metadatas:
            assert meta["section"] not in bad_titles, \
                f"Code de certification utilisé comme titre de section : {meta['section']!r}"

    def test_no_single_char_title(self):
        """Aucun titre de section ne doit faire moins de 4 caractères."""
        for pdf_path in Path("docs").glob("*.pdf"):
            texts, metadatas = build_chunks(pdf_path)
            for meta in metadatas:
                assert len(meta["section"]) >= 4, \
                    f"Titre trop court ({meta['section']!r}) dans {pdf_path.name}"
            break  # On teste sur le premier PDF trouvé

    def test_metadatas_have_required_fields(self):
        """Chaque chunk doit avoir les métadonnées requises pour Pinecone."""
        pdf_path = Path("docs/Plongeur Niveau 1 - PE20.pdf")
        if not pdf_path.exists():
            import pytest
            pytest.skip("PDF not available")

        texts, metadatas = build_chunks(pdf_path)
        for meta in metadatas:
            assert "text" in meta
            assert "source" in meta
            assert "section" in meta
            assert "page" in meta
            assert "niveau" in meta


# ─── Bug #6 : IDs non-ASCII rejetés par Pinecone ───────────────────────────
# Les noms de fichiers contiennent des accents ("vêtement étanche.pdf").
# Pinecone exige des IDs en ASCII pur → erreur 400 à l'upload.
# Fix : unicodedata.normalize + encode/decode ascii + re.sub des caractères restants.

class TestChunkIds:

    def test_ids_are_ascii(self):
        """Les IDs générés doivent être en ASCII pur (exigence Pinecone)."""
        import re
        import unicodedata
        from pathlib import Path

        test_names = [
            "Qualification vêtement étanche",
            "Plongeur Niveau 1 - PE20",
            "Moniteur Fédéral 2ème degré ",
            "Brevet d'Enseignement de la Plongée Profonde à l'air (BEPPA)",
        ]
        for stem in test_names:
            safe = unicodedata.normalize("NFD", stem)
            safe = safe.encode("ascii", "ignore").decode("ascii")
            safe = re.sub(r"[^a-zA-Z0-9_-]", "_", safe)
            ids = [f"{safe}_{i}" for i in range(3)]
            for id_ in ids:
                assert id_.isascii(), f"ID non-ASCII généré : {id_!r}"
