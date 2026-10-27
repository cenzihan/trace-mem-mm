# Choice context candidate

`tracemem-mm-mem0-options-v4.3.0` preserves the raw Mem0 store and joint evidence
selector, and uses the optional top-level Search `options` field. The previous
joint version validated this field but did not use it for retrieval or selection.
The [official API guide](https://agentmemoryleaderboard.ai/api-guide) describes
options as question context without a gold answer. This candidate does not
inspect official evaluation content during development.

The complete original question, images, and every supplied choice retain their
order. Choices are untrusted alternatives, never source facts or an answer key.
One structured text view contains the question and all choices for native dense
and lexical retrieval. The same alternatives accompany the bounded joint source
selector. It outputs only original source indices, not an answer choice. Search
questions and choices are not written into source memory. With no choices, the
previous query representation and selection prompt are unchanged.

Local admission allows at most32 choices and3500 total UTF-8 bytes of choice
text. The serialized retrieval view must fit8192 UTF-8 bytes. Existing4000-byte
query and image-view limits remain. Oversize inputs receive explicit4xx before
model dispatch; nothing is silently truncated to fit the choice limits.

Models remain `text-embedding-v4` and `gpt-4o-mini`, with dated LLM identity and
complete response validation. Search uses at most one GPT call. Original source
responses, user isolation, original accounting and receipt scope, uncertain-call
protection, and no POST retries remain. The hosted reranker is not used. No
calibrated answerability threshold or score gain is claimed by this source change.

Activate only via `create_options_app` and
`TRACEMEM_MEM0_OPTIONS_ENABLED=approved-native-mem0-options`, in addition to the
existing explicit native service settings. Earlier service factories keep their
original behavior.
