# Explicit dataset retrieval

Source: [Linzen et al. upstream README at pinned revision](https://github.com/TalLinzen/rnn_agreement/blob/0001b91aab94c0f834f354a04ed22ba15cefd8da/README.md).
The archive URL is https://tallinzen.net/media/rnn_agreement/rnn_agr_simple.tar.gz.
HTTPS retrieval on 2026-11-21 in [Actions run 44922540789](https://github.com/Akshat4112/syntactic-error-in-neural-networks/actions/runs/44922540789) observed 41,326,466 bytes and SHA-256 `3e0e0c194696dc53378791ac9a75a35d134f4de0dc9253de7cc0a5c8e48b3142`. This is an observed digest, not an upstream signed checksum. Extraction initially rejected AppleDouble metadata; regular metadata files in the expected tree are now ignored.

Use `syntaxbench --project-root WORKSPACE fetch` with the pinned SHA-256 (override explicitly with `--sha256 TRUSTED_DIGEST`). For offline use, place the archive in `WORKSPACE/data/rnn_agr_simple.tar.gz` and add `--offline`. Existing extracted files are compared against the verified archive; conflicting data are not overwritten. No automatic download occurs during ordinary preprocessing or imports. The legacy explicit `preprocess --download` option also uses the pinned digest.

This change does not establish dataset redistribution permission or validate official split membership. Those are tracked separately. No dataset is added to Git by this PR. Repeat extraction verification remains pending.
