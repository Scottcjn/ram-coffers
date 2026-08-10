"""Failure types, each carrying the node it happened on and whether to retry.

The whole point of naming these separately is the ``retry_safe`` flag. A fleet
of second-hand consoles fails constantly and in boring ways — a console
overheats, a switch port flaps, someone's housemate turns one off — and the
coordinator has to answer one question about every failure without thinking:
*may I send this work somewhere else?* Sending a token's expert twice is
harmless; sending it after the node already applied it to a KV cache is not.
So the answer is a property of the failure, decided where the failure is
raised, and never guessed at the call site.
"""

from __future__ import annotations

from typing import Optional


class Gen9Error(Exception):
    """Base for every failure attributable to a node."""

    #: True when the request provably did not take effect, so a retry elsewhere
    #: cannot double-apply anything.
    retry_safe = False

    def __init__(self, message: str, *, unit_id: Optional[str] = None,
                 layer: Optional[int] = None, expert: Optional[int] = None):
        self.unit_id = unit_id
        self.layer = layer
        self.expert = expert
        where = f" [{unit_id}]" if unit_id else ""
        if layer is not None:
            where += f" layer {layer}"
            if expert is not None:
                where += f" expert {expert}"
        super().__init__(message + where)


class ConnectError(Gen9Error):
    """The connection could not be established. Nothing was sent."""

    retry_safe = True


class TimeoutError_(Gen9Error):
    """No reply within the deadline.

    Not retry-safe in general: the node may be mid-computation and its reply may
    still arrive on a connection the coordinator has stopped reading. The
    coordinator's recovery is to drop the connection and re-dispatch, which is
    only correct for stateless expert work — hence
    :class:`ExpertTimeout` for the case where it is.
    """

    retry_safe = False


class ExpertTimeout(TimeoutError_):
    """A routed-expert call timed out.

    Expert evaluation is a pure function of (activation, weights): it mutates
    nothing on the node, so re-dispatching to a replica is always safe even if
    the original is still grinding away.
    """

    retry_safe = True


class ProtocolError(Gen9Error):
    """A frame that does not parse, or a reply that does not match its request.

    Never retry-safe: the connection's framing is no longer trustworthy, so the
    only correct recovery is to close it.
    """

    retry_safe = False


class ShardMissing(Gen9Error):
    """The node does not hold the (layer, expert) that was asked of it.

    Means the plan and the node's actual residency disagree — a deployment bug,
    not a transient fault. Retry-safe only in the sense that another replica may
    genuinely hold it.
    """

    retry_safe = True


class CapacityError(Gen9Error):
    """The node refused a load because it would not fit in its coffers."""

    retry_safe = False


class KernelError(Gen9Error):
    """The compute backend failed: shader compile, HIP launch, allocation."""

    retry_safe = True
