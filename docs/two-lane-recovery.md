# Two-lane Full continuation engineering patch

The four-lane deployment experienced worker OOM under a 700 MiB container limit.
The replacement worker then refused all requests because interrupted provider
operations remained ready/inflight. This patch caps both API execution and model
HTTP work at two lanes, releases parsed raw HTTP buffers before model waits, and
acknowledges interrupted operations only during exclusive worker startup.

The original algorithm, prompts, evidence ordering, model identities, stored
memories and accounting are unchanged. Restart acknowledgements are durable
metadata separate from the original journal. They retain exact old-row and budget
snapshots; they do not invent a model response, settle charges, reuse an old budget
attempt or discard evidence. New retries still use the original budget and limits.
The acknowledgement process uses an OS-held lifetime lease. Another worker cannot
start while the owner is alive. Legacy deployments do not use this lease, so they
must be fully stopped before the first migration; no other journal writer may run.

Deployment uses the explicit create_two_lane_retry_options_app factory and
TRACEMEM_TWO_LANE_ENABLED=approved-two-lanes-exclusive-restart. Keep one Gunicorn
worker without preload. Normal service close or process death releases the lease.
The official evaluation registration remains the original algorithm version; an
accepted checkpoint resume does not constitute independent compatibility review.

Validation: full offline suite and focused two-lane/exclusive-startup/orphan-charge
retention tests; pinned Mem0 no-network checks. Synthetic large-body memory
validation is recorded separately before deployment; it does not guarantee that
all Full workloads fit. No official protected inputs are used in tests.
