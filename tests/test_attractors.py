import pytest

from syntaxbench.attractors import load_attractor_bins, membership_report


def test_official_membership_and_denominators(tmp_path):
    raw = tmp_path / "data" / "rnn_agr_simple"
    raw.mkdir(parents=True)
    for i in range(6):
        (raw / f"numpred.test.{i}").write_text(
            f"VBZ\tbin {i}\nVBP\tshared\n",
            encoding="utf-8",
        )
    with (raw / "numpred.test.0").open("a", encoding="utf-8") as stream:
        stream.write("VBP\tshared\n")
    bins = load_attractor_bins(tmp_path)
    assert tuple(bins) == tuple(range(6))
    assert bins[5][0] == (0, "bin 5")
    report = membership_report(bins)
    assert report["bins"]["0"]["count"] == 3
    assert report["bins"]["0"]["duplicate_rows"] == 1
    assert report["bins"]["5"]["count"] == 2
    assert report["cross_bin_shared_unique_records"]["0:5"] == 1
    assert report["unique_records_across_bins"] == 7
    (raw / "numpred.test.5").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty attractor bin"):
        load_attractor_bins(tmp_path)
