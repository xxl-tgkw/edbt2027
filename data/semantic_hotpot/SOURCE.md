# HotpotQA semantic fixture

This fixture is derived from the public HotpotQA distractor validation split. Questions are queries and title-prefixed context sentences are documents. The local all-MiniLM-L6-v2 encoder produces 384-dimensional normalized vectors.

The fixture is used only for dynamic vector-index protocol validation. It is not an agent trajectory, not a production update trace, and does not claim end-to-end QA accuracy. See the manifest for source and model hashes.
