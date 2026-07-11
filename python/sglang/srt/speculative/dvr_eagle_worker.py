"""Compatibility import for the unified DVR worker.

DECODE_VERIFY_ROLLBACK_EAGLE now uses ``DecodeVerifyRollbackWorkerV2`` with an
EAGLE/MTP draft backend.  Keep the old class name importable while downstream
callers transition.
"""

from sglang.srt.speculative.dvr_worker import (
    DecodeVerifyRollbackWorkerV2 as DecodeVerifyRollbackEagleWorkerV2,
)
