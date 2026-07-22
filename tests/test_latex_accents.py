"""LaTeX accent-macro decoding.

Regression suite for the off-domain author-matching bug: ``_LATEX_CMD_RE`` strips
a LaTeX command and substitutes a SPACE. Accent macros whose command is a
*letter* (``\\H``, ``\\c``, ``\\k``, ``\\v``, ``\\u``, ...) are separated from
their base character by whitespace (``{\\H u}``), so the space survived and tore
the word apart::

    Heged{\\H u}s  ->  "Heged us"  ->  surname key "us"
    Erd{\\H o}s    ->  "Erd os"    ->  surname key "os"

Every author with a Hungarian (o"/u"), Polish (ogonek/stroke), Turkish/Romanian
(cedilla/breve) or Scandinavian (ring) name was therefore unmatchable against the
same name in Crossref/DBLP. ``{\\'a}``/``{\\"o}`` escaped the bug only by
accident -- ``'`` and ``"`` are not in ``[a-zA-Z]``.
"""

from bibtex_updater.utils import (
    authors_last_names,
    last_name_from_person,
    latex_to_plain,
    normalize_title_for_match,
)


class TestAccentMacroDecoding:
    """Letter-command accent macros decode to the accented character."""

    def test_hungarian_double_acute_u(self):
        assert latex_to_plain("Heged{\\H u}s") == "Hegedűs"

    def test_hungarian_double_acute_o(self):
        # Erdős: the most-cited surname in combinatorics normalized to "os".
        assert latex_to_plain("Erd{\\H o}s") == "Erdős"

    def test_cedilla(self):
        assert latex_to_plain("Ak{\\c c}ay") == "Akçay"

    def test_ogonek(self):
        assert latex_to_plain("Wa{\\l}{\\k e}sa") == "Wałęsa"

    def test_caron(self):
        assert latex_to_plain("{\\v C}eleda") == "Čeleda"

    def test_breve(self):
        assert latex_to_plain("Gh{\\u a}iurcă") == "Ghăiurcă"

    def test_ring_above(self):
        assert latex_to_plain('{\\r A}str{\\"o}m') == "Åström"

    def test_dot_above(self):
        assert latex_to_plain("{\\.Z}elazny") == "Żelazny"

    def test_macron(self):
        assert latex_to_plain("{\\=a}") == "ā"

    def test_tilde(self):
        assert latex_to_plain("Mu{\\~n}oz") == "Muñoz"

    def test_grave(self):
        assert latex_to_plain("Cr{\\`e}me") == "Crème"

    def test_circumflex(self):
        assert latex_to_plain("For{\\^e}t") == "Forêt"


class TestAccentMacroSpellingForms:
    """The same macro accepts several spellings; all must decode identically."""

    def test_braced_group_with_space(self):
        assert latex_to_plain("Heged{\\H u}s") == "Hegedűs"

    def test_command_then_braced_argument(self):
        assert latex_to_plain("Heged\\H{u}s") == "Hegedűs"

    def test_bare_command_space_letter(self):
        assert latex_to_plain("Heged\\H us") == "Hegedűs"

    def test_single_quote_form_still_works(self):
        # Regression guard: these never broke and must keep working.
        assert latex_to_plain("Frank{\\'o}") == "Frankó"

    def test_double_quote_form_still_works(self):
        assert latex_to_plain('Schr\\"odinger') == "Schrödinger"


class TestStandaloneGlyphMacros:
    """Macros that name a whole glyph rather than an accent."""

    def test_polish_l_stroke(self):
        assert latex_to_plain("{\\l}odz") == "łodz"

    def test_slashed_o(self):
        assert latex_to_plain("S{\\o}ndergaard") == "Søndergaard"

    def test_sharp_s(self):
        assert latex_to_plain("Rei{\\ss}") == "Reiß"

    def test_ring_a(self):
        assert latex_to_plain('{\\AA}str{\\"o}m') == "Åström"

    def test_ae_ligature(self):
        assert latex_to_plain("K{\\ae}rgaard") == "Kærgaard"

    def test_dotless_i(self):
        assert latex_to_plain("{\\i}") == "ı"

    def test_accent_over_dotless_i_macro(self):
        # "{\\'\\i}" is the standard BibTeX spelling of í in Czech/Spanish names.
        # The accent's argument is itself a MACRO, not a bare letter.
        assert latex_to_plain("Jarn{\\'\\i}k") == "Jarník"

    def test_accent_over_dotless_i_in_full_name(self):
        assert latex_to_plain("Va{\\v s}{\\'\\i}{\\v c}ek") == "Vašíček"

    def test_accent_over_dotless_j_macro(self):
        assert latex_to_plain("{\\v \\j}") == "ǰ"


class TestNonAccentCommandsUnaffected:
    """Formatting/other commands keep their existing behaviour."""

    def test_formatting_command_stripped(self):
        # Unchanged behaviour: a formatting command is removed with its argument.
        # Preserving the argument text is a separate concern from accent decoding
        # and is deliberately out of scope here.
        assert latex_to_plain("\\textbf{Deep} Learning").strip() == "Learning"

    def test_escaped_ampersand(self):
        result = latex_to_plain("Test \\& more")
        assert "Test" in result and "more" in result

    def test_math_mode_stripped(self):
        assert "x" not in latex_to_plain("Bound $x^2$ here").replace("here", "")

    def test_plain_text_untouched(self):
        assert latex_to_plain("Deep Learning") == "Deep Learning"

    def test_empty(self):
        assert latex_to_plain("") == ""


class TestSurnameKeysAfterDecoding:
    """The bug's actual blast radius: surname keys used for author matching."""

    def test_hegedus_surname_key(self):
        assert last_name_from_person("Heged{\\H u}s, Csaba") == "hegedus"

    def test_erdos_surname_key(self):
        assert last_name_from_person("Erd{\\H o}s, Paul") == "erdos"

    def test_akcay_surname_key(self):
        assert last_name_from_person("Ak{\\c c}ay, A.") == "akcay"

    def test_walesa_surname_key(self):
        assert last_name_from_person("Wa{\\l}{\\k e}sa, Lech") == "walesa"

    def test_latex_and_unicode_spellings_agree(self):
        """The entry side (LaTeX) and the API side (Unicode) must reduce alike."""
        assert last_name_from_person("Heged{\\H u}s, Csaba") == last_name_from_person("Csaba Hegedűs")

    def test_varga_bibliography_regression(self):
        """The exact author pair that scored 0.0 similarity in the Varga run."""
        entry = "Heged{\\H u}s, Csaba and Frank{\\'o}, Attila and Varga, P{\\'a}l"
        api = "Csaba Hegedűs and Attila Frankó and Pál Varga"
        assert authors_last_names(entry, limit=10) == authors_last_names(api, limit=10)
        assert authors_last_names(entry, limit=10) == ["hegedus", "franko", "varga"]


class TestTitleNormalizationAfterDecoding:
    """Accent macros in titles must not shatter title tokens either."""

    def test_hungarian_title_token_preserved(self):
        assert normalize_title_for_match("A h{\\'a}l{\\'o}zat min{\\H o}s{\\'e}ge") == "a halozat minosege"

    def test_latex_and_unicode_titles_agree(self):
        assert normalize_title_for_match("Min{\\H o}s{\\'e}g") == normalize_title_for_match("Minőség")
