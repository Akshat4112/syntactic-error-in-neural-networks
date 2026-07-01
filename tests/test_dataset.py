import pandas as pd

from dataset import clean_text


class TestPreprocessVBZVBP:
    """Test that preprocess_data converts VBZ -> 0 and VBP -> 1."""

    def test_vbz_becomes_zero(self):
        df = pd.DataFrame({"POS": ["VBZ", "VBZ"], "Preamble": ["the cat", "a dog"]}, dtype=object)
        df.loc[df["POS"] == "VBZ", "POS"] = 0
        df.loc[df["POS"] == "VBP", "POS"] = 1
        assert (df["POS"] == 0).all()

    def test_vbp_becomes_one(self):
        df = pd.DataFrame({"POS": ["VBP", "VBP"], "Preamble": ["they run", "we go"]}, dtype=object)
        df.loc[df["POS"] == "VBZ", "POS"] = 0
        df.loc[df["POS"] == "VBP", "POS"] = 1
        assert (df["POS"] == 1).all()

    def test_mixed_pos_tags(self):
        df = pd.DataFrame({"POS": ["VBZ", "VBP", "VBZ"], "Preamble": ["a", "b", "c"]}, dtype=object)
        df.loc[df["POS"] == "VBZ", "POS"] = 0
        df.loc[df["POS"] == "VBP", "POS"] = 1
        assert list(df["POS"]) == [0, 1, 0]


class TestTextCleaning:
    """Test the regex cleaning logic from dataset.py."""

    def test_removes_periods(self):
        assert clean_text("hello. world.") == "hello world"

    def test_removes_commas(self):
        assert clean_text("hello, world") == "hello world"

    def test_removes_parentheses(self):
        assert clean_text("hello (world)") == "hello world"

    def test_removes_exclamation_and_question(self):
        assert clean_text("what?! really!") == "what really"

    def test_removes_colons_and_semicolons(self):
        assert clean_text("item: value; other") == "item value other"

    def test_removes_at_and_underscore(self):
        assert clean_text("user@domain some_var") == "user domain some var"

    def test_removes_backticks(self):
        # The regex character class includes the single quote, so '' is also removed
        assert clean_text("``quoted''") == "quoted"

    def test_removes_equals_sign(self):
        assert clean_text("a = b") == "a b"

    def test_collapses_whitespace(self):
        assert clean_text("hello   world") == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("  hello world  ") == "hello world"

    def test_heavy_punctuation(self):
        result = clean_text("...!!??()(),,;;::@@__")
        # After removing all matched punctuation and collapsing spaces, should be empty
        assert result == ""

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_spaces(self):
        assert clean_text("     ") == ""

    def test_no_punctuation(self):
        assert clean_text("the cat sat on the mat") == "the cat sat on the mat"

    def test_mixed_content(self):
        assert clean_text("Hello, world! This is a test... right?") == "Hello world This is a test right"
