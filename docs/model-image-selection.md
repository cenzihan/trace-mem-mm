# Model-authorized image return

This candidate starts from the fixed v4.3.10 baseline. An exact query-image hash
still puts its original source into the same candidate pool. It no longer
forces that source into the final response when the evidence selector omitted
it or returned an empty list. Selected source order, original content, same-Add
neighbors, one GPT call, and the existing 32-candidate / 8-image / 8-selection /
1500-byte view limits are unchanged.

The hypothesis is narrower than general abstention improvement: image identity
alone does not establish evidence for the requested fact. Synthetic tests show
the baseline's unconditional union can override an explicit empty choice. They
do not establish the cause of any official benchmark failure or a score gain.

New independent positive-image, image-conditioned missing-fact and unrelated
subject controls must be frozen before paid validation. Retain the same
selector output for a baseline-versus-candidate packing comparison. Do not
deploy if these controls fail. No official protected examples are used.
