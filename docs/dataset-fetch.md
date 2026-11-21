# Explicit dataset retrieval

Source: [Linzen et al. upstream README at pinned revision](https://github.com/TalLinzen/rnn_agreement/blob/0001b91aab94c0f834f354a04ed22ba15cefd8da/README.md).
The archive URL is https://tallinzen.net/media/rnn_agreement/rnn_agr_simple.tar.gz.
The archive digest and HTTPS endpoint availability have not yet been independently verified. Do not substitute a guessed checksum.

Use `syntaxbench --project-root WORKSPACE fetch --sha256 TRUSTED_DIGEST` with a verified SHA-256 value. For offline use, place the archive in `WORKSPACE/data/rnn_agr_simple.tar.gz` and add `--offline`. Existing extracted files are compared against the verified archive; conflicting data are not overwritten. No automatic download occurs during ordinary preprocessing or imports. The legacy explicit `preprocess --download` option also requires the digest.

This change does not establish dataset redistribution permission or validate official split membership. Those are tracked separately. No dataset is added to Git by this PR. The verified default digest remains outstanding.
