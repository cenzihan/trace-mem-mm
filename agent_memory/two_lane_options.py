"""Two concurrent API/provider calls, preserving the original memory algorithm."""
from pathlib import Path
from .parallel_options import ParallelRetryTransport, ParallelOptionsSystem
from .crash_recovery import ProcessLease, RestartAcknowledgements


class TwoLaneRetryTransport(ParallelRetryTransport):
    def __init__(self, ledger_path, authority_path, prices_path, journal_root, readers, **kwargs):
        root = Path(journal_root)
        root.mkdir(parents=True, exist_ok=True)
        self._process_lease = ProcessLease(root / 'parallel-worker.lock')
        self.restart_acknowledgements = None
        try:
            super().__init__(ledger_path, authority_path, prices_path, journal_root, readers,
                             lane_count=2, **kwargs)
            self.restart_acknowledgements = RestartAcknowledgements(self.journal, self.ledger, self.receipt_scope)
        except BaseException:
            self._process_lease.close()
            raise

    def _accounted_terminal_failure(self, row):
        if (self.restart_acknowledgements is not None and
                self.restart_acknowledgements.permits(row)):
            return True
        return super()._accounted_terminal_failure(row)

    def close(self):
        # Deployment must drain requests before close; OS also releases on death.
        self._process_lease.close()


class TwoLaneOptionsSystem(ParallelOptionsSystem):
    def capabilities(self):
        return dict(super().capabilities(), version='tracemem-mm-parallel-v4.3.13-rc1',
                    concurrency=2, provider_concurrency=2,
                    restart_recovery='exclusive-worker-original-receipts-and-charges-preserved')
