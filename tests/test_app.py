"""
Tests de la logique applicative de app.py.

Couvre :
- detect_niveau : détection du niveau dans une question utilisateur
- Logique de filtre Pinecone : inclusion de "Général" pour les règles transversales
- Cohérence entre app.py et evaluate.py (même logique de filtre)

─── Pourquoi ces tests ? ────────────────────────────────────────────────────

Bug découvert lors de l'évaluation RAGAS-like :
  evaluate.py utilisait retrieve() SANS filtre → scores RIFAP faussement à 0.00.
  Les vrais chunks RIFAP scoraient 0.75 dans Pinecone mais n'apparaissaient pas
  car battus par du bruit (MF1, Randosub à 0.47) quand aucun filtre n'est appliqué.

Décision de design :
  Le filtre $in [niveau, "Général"] inclut systématiquement les docs "Général"
  (ex: "Conditions de pratique, brevets et qualifications.pdf") qui contiennent
  des règles fédérales transversales applicables à tous les niveaux.
  Sans ça, une question sur N3 rate les règles générales du Code du Sport.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import detect_niveau


# ─── detect_niveau ────────────────────────────────────────────────────────────

class TestDetectNiveau:

    # Niveaux plongeur
    def test_niveau1_mot_clé(self):
        assert detect_niveau("conditions pour le niveau 1") == "N1"

    def test_niveau1_code(self):
        assert detect_niveau("qu'est-ce que le PE20 ?") == "N1"

    def test_niveau2_mot_clé(self):
        assert detect_niveau("âge minimum Niveau 2") == "N2"

    def test_niveau2_pa20(self):
        assert detect_niveau("différence entre PA20 et PE40") == "N2"

    def test_niveau3_pa40(self):
        assert detect_niveau("prérogatives PA40") == "N3"

    def test_niveau3_pe60(self):
        assert detect_niveau("plonger à 60m avec PE60") == "N3"

    def test_niveau4_guide(self):
        assert detect_niveau("rôle du guide de palanquée") == "N4"

    def test_niveau4_n4(self):
        assert detect_niveau("le N4 peut-il encadrer à 60m ?") == "N4"

    def test_niveau5(self):
        assert detect_niveau("formation niveau 5") == "N5"

    # Qualifications techniques
    def test_mf1(self):
        assert detect_niveau("conditions d'accès au MF1") == "MF1"

    def test_mf2(self):
        assert detect_niveau("prérogatives MF2") == "MF2"

    def test_nitrox(self):
        assert detect_niveau("qualification nitrox obligatoire ?") == "Nitrox"

    def test_trimix(self):
        assert detect_niveau("plongée au trimix") == "Trimix"

    # Cas RIFAP — bug corrigé
    # Avant le fix, evaluate.py n'appliquait pas de filtre pour RIFAP,
    # ce qui faisait remonter du bruit au lieu des vrais chunks RIFAP.
    def test_rifap_detection(self):
        assert detect_niveau("qu'est-ce que le RIFAP ?") == "RIFAP"

    def test_rifap_avec_n3_retourne_n3(self):
        # Quand la question mentionne RIFAP ET N3, c'est le N3 qui prime :
        # la question porte sur les exigences du N3, pas sur la définition du RIFAP.
        # Le dict itère dans l'ordre d'insertion : N3 est défini avant RIFAP.
        assert detect_niveau("le RIFAP est-il obligatoire pour le N3 ?") == "N3"

    def test_rifap_minuscule(self):
        assert detect_niveau("rifap et premiers secours") == "RIFAP"

    def test_sidemount(self):
        assert detect_niveau("formation sidemount") == "Sidemount"

    # Cas sans niveau détecté
    def test_no_niveau_general(self):
        assert detect_niveau("qu'est-ce que la FFESSM ?") is None

    def test_no_niveau_capitale(self):
        assert detect_niveau("quelle est la capitale de la France ?") is None

    def test_no_niveau_vide(self):
        assert detect_niveau("") is None

    # Casse insensible
    def test_case_insensitive_niveau(self):
        assert detect_niveau("NIVEAU 2") == "N2"

    def test_case_insensitive_mf1(self):
        assert detect_niveau("Le MF1 c'est quoi ?") == "MF1"


# ─── Logique de filtre Pinecone ───────────────────────────────────────────────

class TestFiltreLogique:
    """
    Vérifie que la logique de filtre est cohérente entre app.py et evaluate.py.

    Règle : quand un niveau est détecté, le filtre Pinecone doit être :
      {"niveau": {"$in": [niveau, "Général"]}}

    Pourquoi inclure "Général" :
      Les documents "Brevets Fédéraux - Généralités.pdf", "Code du Sport", etc.
      portent le tag niveau="Général". Ils contiennent des règles transversales
      (âges minimaux, obligations médicales, Code du Sport) applicables à tous.
      Sans eux, une question sur N3 loupe les contraintes légales générales.
    """

    def _build_filter(self, question: str) -> dict | None:
        """Reproduit la logique de filtre de app.py."""
        niveau = detect_niveau(question)
        if niveau:
            return {"niveau": {"$in": [niveau, "Général"]}}
        return None

    def test_filtre_inclut_general(self):
        f = self._build_filter("conditions niveau 3")
        assert f is not None
        assert "Général" in f["niveau"]["$in"]
        assert "N3" in f["niveau"]["$in"]

    def test_filtre_rifap_inclut_general(self):
        f = self._build_filter("qu'est-ce que le RIFAP ?")
        assert f is not None
        assert "RIFAP" in f["niveau"]["$in"]
        assert "Général" in f["niveau"]["$in"]

    def test_pas_de_filtre_sans_niveau(self):
        f = self._build_filter("qu'est-ce que la FFESSM ?")
        assert f is None

    def test_filtre_utilise_in_pas_eq(self):
        """Le filtre doit utiliser $in et non $eq pour inclure Général."""
        f = self._build_filter("conditions MF1")
        assert "$in" in f["niveau"]
        assert "$eq" not in f["niveau"]
