"""Per-type value codecs: how a variable's value enters and leaves the stream.

Every variable type answers two questions that are completely separate from the
block stack sitting between them:

  encode:  given the token's value, what does it contribute to the residual
           stream at layer 0?                       [B, M, P] -> [B, M, D]
  decode:  given the residual stream at layer L, what are this type's logits?
                                                    [B, M, D] -> [B, M, C]

Both halves are chosen per type and independently of each other, so the two
architectures this codebase grew by hand are now two points in one space:

  value_codec="dual"   ==  slice            the hand-built parameter stream:
                                            the value is COPIED into a reserved
                                            range of the stream and READ OUT of
                                            the same range at the end.
  value_codec="typed"  ==  linear           a learnable projection of the value
                                            into the full hidden dimension, and
                                            a linear probe on the full hidden
                                            dimension to read it back.

and `mlp` adds the obvious third: a one-hidden-layer network on either side.
Mixed specs are legal and are the interesting ones -- `slice/linear` keeps the
hand-built injection and learns the readout, `linear/slice` the reverse.

What the encoder is handed
--------------------------
A variable's value lives in its type's EXTENDED domain {T, v_1, ..., v_C}: the C
values it can take, plus a placeholder T ("top") that a masked or missing token
takes instead. T is a value like any other. There is no mask bit: a learned
encoder is handed a distribution over the extended domain -- a point mass on T
for a hidden token, on v_k for an observed one, a soft row for an LLM-labelled
one -- and maps T to whatever representation it learns, which need not look
anything like a one-hot. T sits at index 0 of that row.

The slice codec is the one place a fixed encoding of T survives: it copies the
type's HAND-BUILT row, in which T is written as 1.0 in the range's first slot
and v_k as logit_high in slot k. That is the slice codec's choice of encoding,
not a property of the type, and nothing outside it depends on it.

The encoder also decides how much of the stream it OWNS. The slice encoder owns
the range it writes into: nothing else may put anything there at layer 0, which
is why the dual stream's type centroids are feature_dim wide rather than
model_dim wide. A learned encoder owns nothing and the centroids are the full
width. That single number, `reserved_tail`, is the whole structural difference
between the two architectures -- everything else is which function maps P to D.

Spec grammar
------------
    spec  := half [ "/" half ]            one half means both halves alike
    half  := kind [ ":" opt ("," opt)* ]
    kind  := "slice" | "linear" | "mlp"   ("dual" and "typed" alias the first two)
    opt   := key "=" value | key          a bare key means key=true

Options, per half (unknown keys are an error, not a silent default):
    init=      unit (N(0,1), the default for every encoder and decoder weight)
               | scaled | orthogonal | aligned
    width=     learned codecs: write into / read from only the LAST `width`
               dimensions of the stream (default: all of them)
    freeze     hold this half at its initial value
    hidden=    mlp hidden width
    layers=    mlp Linear count (>= 2)
    source=    encoder input: hand_built (the type's fixed row) | value (the
               distribution over the extended domain)
    at=        slice offset into the stream, or "tail"
    reserve=   whether the encoder owns its range exclusively, i.e. the type
               centroids are narrowed so nothing else writes there at layer 0

Examples:
    slice                       the shipped dual stream
    linear:init=unit            the shipped typed codec
    linear:init=aligned,freeze  the typed codec that IS the dual stream
    mlp:hidden=128              an MLP both ways
    slice/mlp:hidden=128        hand-built injection, MLP readout
    linear:width=8,reserve      learned code confined to the last 8 dimensions,
                                which the centroids give up
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

KINDS = ("slice", "linear", "mlp")
# The two names the flags used before encode and decode were separable.
KIND_ALIASES = {"dual": "slice", "typed": "linear"}
# Every learned encoder and decoder weight is drawn N(0, 1) unless a spec asks
# for one of the other two loud inits or the aligned control. There is no
# N(0, 0.02) option: at that scale the value reaches the stream ~1000x quieter
# than the hand-built row and the codec cost 0.09 nats (value-codec thread B3);
# a quiet probe likewise starves the decoder's gradient.
INITS = ("unit", "scaled", "orthogonal", "aligned")
SOURCES = ("hand_built", "value")


# ── spec ─────────────────────────────────────────────────────────────────────

def _parse_bool(key: str, text: str) -> bool:
    low = text.strip().lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"codec option {key}={text!r} is not a boolean")


@dataclass(frozen=True)
class HalfSpec:
    """One half of a codec -- either the encoder or the decoder.

    Fields that do not apply to the chosen kind are left at their defaults and
    are not rendered, so `render()` round-trips through `parse()`.
    """

    kind: str = "slice"
    init: str = "unit"
    freeze: bool = False
    hidden: int = 128
    layers: int = 2
    width: int | None = None        # None -> the whole stream (learned kinds only)
    source: str | None = None       # None -> the kind's natural source
    at: int | None = None           # None -> the tail (slice only)
    reserve: bool | None = None     # None -> the kind's natural answer

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown codec kind {self.kind!r}; choose from {KINDS}")
        if self.init not in INITS:
            raise ValueError(f"unknown codec init {self.init!r}; choose from {INITS}")
        if self.source is not None and self.source not in SOURCES:
            raise ValueError(f"unknown codec source {self.source!r}; choose from {SOURCES}")
        if self.kind == "mlp" and self.layers < 2:
            raise ValueError(f"mlp codec needs layers >= 2, got {self.layers}")
        if self.kind == "mlp" and self.hidden < 1:
            raise ValueError(f"mlp codec needs hidden >= 1, got {self.hidden}")
        if self.width is not None and self.kind == "slice":
            raise ValueError("width= applies to learned codecs; a slice codec's width "
                             "is the hand-built row's")
        if self.width is not None and self.width < 1:
            raise ValueError(f"codec width must be >= 1, got {self.width}")
        if self.init == "aligned" and self.kind != "linear":
            raise ValueError("init=aligned only applies to a linear codec: it is the "
                             "linear map that reproduces the slice codec exactly")

    # The source each kind reads when the spec does not say.
    #
    # The slice codec copies, so it must be handed the row it is meant to copy:
    # the hand-built one, with logit_high in the observed class's slot. A learned
    # encoder is given the value itself and discovers its own magnitude --
    # handing it the hand-built row would mean the constant it is supposed to
    # replace is still in the loop.
    @property
    def natural_source(self) -> str:
        return "hand_built" if self.kind == "slice" else "value"

    @property
    def resolved_source(self) -> str:
        return self.source or self.natural_source

    @classmethod
    def parse(cls, text: str) -> "HalfSpec":
        text = text.strip()
        if not text:
            raise ValueError("empty codec spec")
        kind, _, opt_text = text.partition(":")
        kind = kind.strip().lower()
        kind = KIND_ALIASES.get(kind, kind)
        fields: Dict[str, Any] = {"kind": kind}
        for opt in (o for o in opt_text.split(",") if o.strip()):
            key, eq, value = opt.partition("=")
            key, value = key.strip().lower(), value.strip()
            if not eq:
                value = "true"
            if key == "init":
                fields["init"] = value.lower()
            elif key == "freeze":
                fields["freeze"] = _parse_bool(key, value)
            elif key == "hidden":
                fields["hidden"] = int(value)
            elif key == "layers":
                fields["layers"] = int(value)
            elif key == "width":
                fields["width"] = int(value)
            elif key == "source":
                fields["source"] = value.lower()
            elif key == "at":
                fields["at"] = None if value.lower() == "tail" else int(value)
            elif key == "reserve":
                fields["reserve"] = _parse_bool(key, value)
            else:
                raise ValueError(f"unknown codec option {key!r} in {text!r}")
        return cls(**fields)

    def render(self) -> str:
        """The canonical spelling, which `parse` reads back unchanged."""
        default = HalfSpec(kind=self.kind)
        opts = []
        if self.kind != "slice" and self.init != default.init:
            opts.append(f"init={self.init}")
        if self.freeze:
            opts.append("freeze=true")
        if self.kind == "mlp":
            if self.hidden != default.hidden:
                opts.append(f"hidden={self.hidden}")
            if self.layers != default.layers:
                opts.append(f"layers={self.layers}")
        if self.width is not None:
            opts.append(f"width={self.width}")
        if self.source is not None and self.source != self.natural_source:
            opts.append(f"source={self.source}")
        if self.at is not None:
            opts.append(f"at={self.at}")
        if self.reserve is not None:
            opts.append(f"reserve={str(self.reserve).lower()}")
        return self.kind + (":" + ",".join(opts) if opts else "")


@dataclass(frozen=True)
class CodecSpec:
    """A full codec: one encoder half and one decoder half."""

    encoder: HalfSpec = HalfSpec()
    decoder: HalfSpec = HalfSpec()

    def __post_init__(self) -> None:
        if self.encoder.init == "aligned" and self.decoder.kind not in ("linear", "slice"):
            raise ValueError("init=aligned needs a linear or slice decoder to read the "
                             "range the encoder writes into")

    @classmethod
    def parse(cls, text: str) -> "CodecSpec":
        """Read a spec. One half means both halves alike; `enc/dec` splits them."""
        if isinstance(text, CodecSpec):
            return text
        enc_text, slash, dec_text = str(text).partition("/")
        enc = HalfSpec.parse(enc_text)
        if not slash:
            # The decoder mirrors the encoder's kind, but the encoder-only
            # options (init, source, reserve) do not carry over -- a decoder has
            # no input row to choose and no range to own.
            dec = HalfSpec(kind=enc.kind, freeze=enc.freeze,
                           hidden=enc.hidden, layers=enc.layers,
                           width=enc.width, at=enc.at,
                           init="aligned" if enc.init == "aligned" else HalfSpec().init)
        else:
            dec = HalfSpec.parse(dec_text)
        return cls(encoder=enc, decoder=dec)

    def render(self) -> str:
        mirrored = CodecSpec.parse(self.encoder.render())
        if mirrored.decoder == self.decoder:
            return self.encoder.render()
        return f"{self.encoder.render()}/{self.decoder.render()}"

    def __str__(self) -> str:
        return self.render()


def parse_type_codecs(entries, default: str = "slice") -> tuple:
    """Read `--value-codec` and repeated `--type-codec NAME=SPEC` into specs.

    Returns (default_spec, {type_name: spec}). Both are CodecSpec objects, so a
    bad spelling fails here rather than at the first forward pass.
    """
    default_spec = CodecSpec.parse(default)
    overrides: Dict[str, CodecSpec] = {}
    if isinstance(entries, Mapping):
        entries = [entries]
    for entry in entries or ():
        if isinstance(entry, Mapping):
            items = entry.items()
        else:
            name, eq, spec = str(entry).partition("=")
            if not eq:
                raise ValueError(f"--type-codec wants NAME=SPEC, got {entry!r}")
            items = [(name, spec)]
        for name, spec in items:
            name = str(name).strip()
            if not name:
                raise ValueError(f"--type-codec wants NAME=SPEC, got {entry!r}")
            overrides[name] = CodecSpec.parse(spec)
    return default_spec, overrides


def _normal_init(module: nn.Module) -> None:
    """N(0, 1) for every Linear weight in `module`, zero for every bias.

    The one init rule for codecs: no PyTorch default (kaiming-uniform, std
    ~1/sqrt(fan_in)) and no N(0, 0.02) anywhere in an encoder or a decoder.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ── encoders ─────────────────────────────────────────────────────────────────

class ValueEncoder(nn.Module, ABC):
    """Maps one type's value row into a contribution to the residual stream.

    `input_width` is how many columns of the value row this encoder reads, and
    `reserved_tail` how many of the stream's trailing dimensions it owns
    exclusively at layer 0 (0 = none).
    """

    value_source: str
    input_width: int
    reserved_tail: int

    def initialize(self) -> None:
        """Draw any random init. Called once, after every codec is constructed."""

    @abstractmethod
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """[B, M, input_width] -> [B, M, model_dim]."""


def _into_tail(x: torch.Tensor, model_dim: int) -> torch.Tensor:
    """Place a [..., T] code in the last T dimensions of a [..., model_dim] row."""
    pad = model_dim - x.shape[-1]
    return x if pad == 0 else nn.functional.pad(x, (pad, 0))


def _resolve_width(spec: HalfSpec, model_dim: int, global_param_dim: int) -> int:
    # The aligned control occupies exactly the range the slice codec would, so
    # its natural width is the hand-built row's, not the whole stream.
    width = spec.width
    if width is None:
        width = global_param_dim if spec.init == "aligned" else model_dim
    if width > model_dim:
        raise ValueError(f"codec width {width} exceeds the stream width {model_dim}")
    if spec.init == "aligned" and width < global_param_dim:
        raise ValueError(f"init=aligned needs width >= {global_param_dim} (the "
                         f"hand-built row), got {width}")
    return int(width)


def _reserved(spec: HalfSpec, width: int, model_dim: int) -> int:
    # Reserving the whole stream would leave the centroids no room at all.
    reserve = (spec.init == "aligned") if spec.reserve is None else bool(spec.reserve)
    if reserve and width >= model_dim:
        raise ValueError("reserve needs width < model_dim: the type centroids must "
                         "keep some of the stream")
    return width if reserve else 0


class SliceEncoder(ValueEncoder):
    """Copy the type's hand-built row into a reserved range of the stream.

    This is the dual stream's layer 0. `at` defaults to the tail, where the row
    is global_param_dim wide, so the type's own entries land at the start of the
    range and the rest of it stays zero -- the layout the hand-built parameter
    stream has always had, T written as 1.0 in the range's first slot.
    """

    def __init__(self, *, model_dim: int, width: int, at: int | None,
                 reserve: bool | None, source: str):
        super().__init__()
        self.model_dim = int(model_dim)
        self.width = int(width)
        self.at = self.model_dim - self.width if at is None else int(at)
        if not (0 <= self.at and self.at + self.width <= self.model_dim):
            raise ValueError(f"slice codec range [{self.at}, {self.at + self.width}) "
                             f"does not fit in a stream of width {self.model_dim}")
        self.value_source = source
        self.input_width = self.width
        # A range that reaches the end of the stream can be fenced off simply by
        # making every other layer-0 contribution narrower. A range in the middle
        # cannot, so it shares the stream with the centroids unless asked not to
        # -- and asking is then a request this builder cannot honour.
        suffix = self.at + self.width == self.model_dim
        self.reserve = suffix if reserve is None else bool(reserve)
        if self.reserve and not suffix:
            raise ValueError("a slice codec can only own its range exclusively when the "
                             f"range ends at the stream's last dimension; at={self.at} "
                             f"width={self.width} model_dim={self.model_dim}")
        self.reserved_tail = self.width if self.reserve else 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        out = value.new_zeros(*value.shape[:-1], self.model_dim)
        out[..., self.at : self.at + self.width] = value
        return out

    def extra_repr(self) -> str:
        return (f"at={self.at}, width={self.width}, reserve={self.reserve}, "
                f"source={self.value_source}")


class LinearEncoder(ValueEncoder):
    """A learnable projection of the value into the last `width` dimensions.

    With the default width that is the whole hidden dimension -- the typed
    codec. No bias: the type centroid every token already carries is the bias,
    which keeps "centroid + something" true of every token in the stream.

    Weights are N(0, 1) by default ("unit"), whatever the width, so each
    coordinate of the code is unit scale. "scaled" and "orthogonal" instead fix
    a column's norm at logit_high.
    """

    def __init__(self, *, model_dim: int, param_dim: int, global_param_dim: int,
                 spec: HalfSpec, logit_high: float):
        super().__init__()
        self.model_dim = int(model_dim)
        self.param_dim = int(param_dim)
        self.width = _resolve_width(spec, self.model_dim, int(global_param_dim))
        self.proj = nn.Linear(self.param_dim, self.width, bias=False)
        self.init = spec.init
        self.value_source = spec.resolved_source
        self.input_width = self.param_dim
        self.reserved_tail = _reserved(spec, self.width, self.model_dim)
        self.reserve = self.reserved_tail > 0
        self._global_param_dim = int(global_param_dim)
        self._logit_high = float(logit_high)
        self._freeze = spec.freeze

    def initialize(self) -> None:
        """Draw the encoder's init. Separate from __init__ so build_codecs can
        consume the RNG in the order the typed codec always did: every Linear
        constructed first, then every encoder drawn."""
        self._init_weights(self._global_param_dim, self._logit_high)
        if self._freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def _init_weights(self, global_param_dim: int, logit_high: float) -> None:
        w = self.proj.weight.data
        if self.init == "unit":
            # N(0, 1), at every width. A column's norm is then sqrt(width): 8.9
            # at the full 80, 2.2 at width 5. Each coordinate arrives at unit
            # scale, which is what the pre-norm blocks see.
            nn.init.normal_(w, mean=0.0, std=1.0)
        elif self.init == "scaled":
            # A column's norm is logit_high, i.e. the magnitude the stream gets
            # from the hand-built row. Works, but reads logit_high to do it.
            nn.init.normal_(w, mean=0.0, std=logit_high / math.sqrt(self.width))
        elif self.init == "orthogonal":
            # Loud AND well separated: a Gaussian draw can start with two values
            # pointing nearly the same way, which is a bad code to be stuck with
            # when there is little data to correct it.
            nn.init.orthogonal_(w)
            w.mul_(logit_high)
        else:                                   # aligned
            # diag(1, logit_high, ..., logit_high) into the range the slice codec
            # owns: T to the range's first slot at weight 1.0, v_k to slot k at
            # logit_high -- the hand-built row, reproduced as a linear map.
            w.zero_()
            base = self.width - global_param_dim
            w[base, 0] = 1.0
            for j in range(self.param_dim - 1):
                w[base + 1 + j, 1 + j] = logit_high

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _into_tail(self.proj(value), self.model_dim)

    def extra_repr(self) -> str:
        return (f"init={self.init}, width={self.width}, reserve={self.reserve}, "
                f"source={self.value_source}")


class MLPEncoder(ValueEncoder):
    """A small network from the value row into the last `width` dimensions.

    Hidden layers carry a bias; the output layer does not, for the same reason
    LinearEncoder has none. Every weight is N(0, 1) and every bias zero; with
    "scaled" or "orthogonal" the output layer is then rescaled so a value
    arrives at norm logit_high, measured rather than derived since with a ReLU
    in the way it is not a closed form.
    """

    def __init__(self, *, model_dim: int, param_dim: int, global_param_dim: int,
                 spec: HalfSpec, logit_high: float):
        super().__init__()
        self.model_dim = int(model_dim)
        self.param_dim = int(param_dim)
        self.width = _resolve_width(spec, self.model_dim, int(global_param_dim))
        self.value_source = spec.resolved_source
        self.input_width = self.param_dim
        self.reserved_tail = _reserved(spec, self.width, self.model_dim)
        self.reserve = self.reserved_tail > 0
        self.init = spec.init
        widths = [self.param_dim] + [spec.hidden] * (spec.layers - 1) + [self.width]
        layers: list[nn.Module] = []
        for i in range(len(widths) - 1):
            last = i == len(widths) - 2
            layers.append(nn.Linear(widths[i], widths[i + 1], bias=not last))
            if not last:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self._logit_high = float(logit_high)
        self._freeze = spec.freeze

    def initialize(self) -> None:
        _normal_init(self.net)
        if self.init != "unit":
            self._rescale_output(self._logit_high)
        if self._freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def _rescale_output(self, logit_high: float) -> None:
        """Scale the output layer so a point-mass value arrives at the target norm."""
        target = float(logit_high)              # "scaled" / "orthogonal"
        with torch.no_grad():
            probe = torch.eye(self.param_dim)
            norm = float(self.net(probe).norm(dim=-1).mean())
            if norm > 0:
                last = [m for m in self.net if isinstance(m, nn.Linear)][-1]
                last.weight.data.mul_(target / norm)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _into_tail(self.net(value), self.model_dim)

    def extra_repr(self) -> str:
        return f"init={self.init}, width={self.width}, source={self.value_source}"


# ── decoders ─────────────────────────────────────────────────────────────────

class ValueDecoder(nn.Module, ABC):
    """Reads one type's logits out of the residual stream at layer L.

    Never over T: "is this value hidden" is not something the model is asked to
    guess, so a type with C values gets C logits.
    """

    num_values: int

    def initialize(self) -> None:
        """Draw any random init. Called once, after every codec is constructed."""

    @abstractmethod
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """[B, M, model_dim] -> [B, M, num_values]."""


class SliceDecoder(ValueDecoder):
    """Read the logits straight out of the range the value was copied into.

    This is the dual stream's readout: the value's own slots, carried the whole
    way up the stack and never re-projected.
    """

    def __init__(self, *, model_dim: int, param_dim: int, global_param_dim: int,
                 at: int | None):
        super().__init__()
        self.num_values = int(param_dim) - 1
        base = model_dim - int(global_param_dim) if at is None else int(at)
        self.start = base + 1                       # the range's first slot is T's
        if self.start + self.num_values > model_dim:
            raise ValueError(f"slice decoder range [{self.start}, "
                             f"{self.start + self.num_values}) does not fit in {model_dim}")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden[..., self.start : self.start + self.num_values]

    def extra_repr(self) -> str:
        return f"start={self.start}, num_values={self.num_values}"


class LinearDecoder(ValueDecoder):
    """A linear probe on the last `width` dimensions (default: all of them).

    Untied from the encoder.
    """

    def __init__(self, *, model_dim: int, param_dim: int, global_param_dim: int,
                 spec: HalfSpec, logit_high: float):
        super().__init__()
        self.num_values = int(param_dim) - 1
        self.width = _resolve_width(spec, int(model_dim), int(global_param_dim))
        self.probe = nn.Linear(self.width, self.num_values)
        self.init = spec.init
        self._global_param_dim = int(global_param_dim)
        self._freeze = spec.freeze

    def initialize(self) -> None:
        if self.init == "aligned":
            # The readout half of the aligned control: read the same slots the
            # aligned encoder wrote, with unit weight and no bias.
            base = self.width - self._global_param_dim
            self.probe.weight.data.zero_()
            for j in range(self.num_values):
                self.probe.weight.data[j, base + 1 + j] = 1.0
            self.probe.bias.data.zero_()
        else:
            _normal_init(self.probe)
        if self._freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.probe(hidden[..., hidden.shape[-1] - self.width :])

    def extra_repr(self) -> str:
        return f"width={self.width}"


class MLPDecoder(ValueDecoder):
    """A small network reading the logits off the last `width` dimensions."""

    def __init__(self, *, model_dim: int, param_dim: int, global_param_dim: int,
                 spec: HalfSpec):
        super().__init__()
        self.num_values = int(param_dim) - 1
        self.width = _resolve_width(spec, int(model_dim), int(global_param_dim))
        widths = [self.width] + [spec.hidden] * (spec.layers - 1) + [self.num_values]
        layers: list[nn.Module] = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self._freeze = spec.freeze

    def initialize(self) -> None:
        _normal_init(self.net)
        if self._freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden[..., hidden.shape[-1] - self.width :])

    def extra_repr(self) -> str:
        return f"width={self.width}"


# ── codec ────────────────────────────────────────────────────────────────────

class ValueCodec(nn.Module):
    """One variable type's encoder and decoder, chosen independently."""

    def __init__(self, *, type_name: str, param_dim: int, global_param_dim: int,
                 model_dim: int, spec: CodecSpec, logit_high: float):
        super().__init__()
        self.type_name = type_name
        self.spec = spec
        self.param_dim = int(param_dim)
        self.global_param_dim = int(global_param_dim)
        self.model_dim = int(model_dim)
        self.encoder = _build_encoder(
            model_dim=model_dim, param_dim=param_dim,
            global_param_dim=global_param_dim, spec=spec.encoder,
            logit_high=logit_high)
        self.decoder = _build_decoder(
            model_dim=model_dim, param_dim=param_dim,
            global_param_dim=global_param_dim, spec=spec.decoder,
            logit_high=logit_high)

    @property
    def value_source(self) -> str:
        return self.encoder.value_source

    @property
    def input_width(self) -> int:
        return self.encoder.input_width

    @property
    def reserved_tail(self) -> int:
        return self.encoder.reserved_tail

    @property
    def num_values(self) -> int:
        return self.decoder.num_values

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return self.encoder(value[..., : self.encoder.input_width])

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(hidden)

    def extra_repr(self) -> str:
        return f"{self.type_name}: {self.spec.render()}"


def _build_encoder(*, model_dim: int, param_dim: int, global_param_dim: int,
                   spec: HalfSpec, logit_high: float) -> ValueEncoder:
    if spec.kind == "slice":
        return SliceEncoder(model_dim=model_dim, width=global_param_dim, at=spec.at,
                            reserve=spec.reserve, source=spec.resolved_source)
    if spec.kind == "linear":
        return LinearEncoder(model_dim=model_dim, param_dim=param_dim,
                             global_param_dim=global_param_dim, spec=spec,
                             logit_high=logit_high)
    return MLPEncoder(model_dim=model_dim, param_dim=param_dim,
                      global_param_dim=global_param_dim, spec=spec,
                      logit_high=logit_high)


def _build_decoder(*, model_dim: int, param_dim: int, global_param_dim: int,
                   spec: HalfSpec, logit_high: float) -> ValueDecoder:
    if spec.kind == "slice":
        return SliceDecoder(model_dim=model_dim, param_dim=param_dim,
                            global_param_dim=global_param_dim, at=spec.at)
    if spec.kind == "linear":
        return LinearDecoder(model_dim=model_dim, param_dim=param_dim,
                             global_param_dim=global_param_dim, spec=spec,
                             logit_high=logit_high)
    return MLPDecoder(model_dim=model_dim, param_dim=param_dim,
                      global_param_dim=global_param_dim, spec=spec)


def build_codecs(types: Mapping[str, Any], *, model_dim: int, global_param_dim: int,
                 default_spec: CodecSpec, type_specs: Mapping[str, CodecSpec],
                 logit_high: float) -> Dict[str, ValueCodec]:
    """One codec per type that carries a value.

    Entity types have no value and no prediction, so they get no codec and
    contribute nothing but their centroid at layer 0. A spec naming a type that
    has no value is a mistake worth reporting rather than ignoring.
    """
    codecs: Dict[str, ValueCodec] = {}
    for name, t in types.items():
        if not getattr(t, "is_variable", False) or t.param_dim <= 1:
            continue
        codecs[name] = ValueCodec(
            type_name=name, param_dim=t.param_dim, global_param_dim=global_param_dim,
            model_dim=model_dim, spec=type_specs.get(name, default_spec),
            logit_high=float(getattr(t, "logit_high", logit_high)))
    for codec in codecs.values():
        codec.encoder.initialize()
    for codec in codecs.values():
        codec.decoder.initialize()
    unknown = set(type_specs) - set(codecs)
    if unknown:
        known = ", ".join(sorted(codecs)) or "(none)"
        raise ValueError(f"no value-carrying type named {sorted(unknown)}; "
                         f"types with a codec: {known}")
    return codecs


def planned_reserved_tail(types: Mapping[str, Any], *, model_dim: int,
                          global_param_dim: int, default_spec: CodecSpec,
                          type_specs: Mapping[str, CodecSpec]) -> int:
    """How many trailing dimensions the codecs will own, before building them.

    The model needs this to size the type centroids, and the centroids have to
    be drawn BEFORE the codecs for the RNG to be consumed in its historical
    order. Uses the same helpers the encoders do, so the two cannot disagree --
    and the model asserts they did not.
    """
    tail = 0
    for name, t in types.items():
        if not getattr(t, "is_variable", False) or t.param_dim <= 1:
            continue
        enc = type_specs.get(name, default_spec).encoder
        if enc.kind == "slice":
            at = model_dim - global_param_dim if enc.at is None else enc.at
            suffix = at + global_param_dim == model_dim
            reserved = global_param_dim if (suffix if enc.reserve is None else enc.reserve) else 0
        else:
            reserved = _reserved(enc, _resolve_width(enc, model_dim, global_param_dim), model_dim)
        tail = max(tail, reserved)
    return tail


def render_codecs(default_spec: CodecSpec,
                  type_specs: Mapping[str, CodecSpec]) -> Dict[str, Any]:
    """The spec pair as plain JSON, for the run config a checkpoint is scored from."""
    return {
        "value_codec": default_spec.render(),
        "type_codecs": {k: v.render() for k, v in sorted(type_specs.items())},
    }


def resolve_specs(config: Any) -> tuple:
    """The (default_spec, {type: spec}) a model config asks for.

    Reads the old flags too, so every train_config.json on disk still builds the
    model it trained:

      value_codec="dual"                     -> slice
      value_codec="typed", codec_init=X      -> linear:init=X  (+ freeze)
      use_param_output_head=True             -> the default decoder becomes an
                                                MLP (hidden 128). The old head was
                                                one MLP shared by every type; this
                                                is one per type. No run on disk
                                                ever set it.

    codec_init and codec_freeze only mean something next to "typed"; given with a
    real spec they would be silently dropped, so that is an error instead.
    """
    raw = str(getattr(config, "value_codec", "slice") or "slice").strip()
    codec_init = getattr(config, "codec_init", None)
    codec_freeze = bool(getattr(config, "codec_freeze", False))
    if raw == "typed":
        # An old typed run may record the retired N(0, 0.02) init. Rebuilding it
        # is only ever a prelude to loading its checkpoint, which overwrites
        # every codec weight, so it is rebuilt with the N(0, 1) init instead.
        # New training cannot ask for "random": train.py no longer offers it.
        init = "unit" if codec_init in (None, "random") else codec_init
        text = f"linear:init={init}" + (",freeze" if codec_freeze else "")
    else:
        if codec_freeze:
            raise ValueError("codec_freeze only applies to value_codec='typed'; with a "
                             f"codec spec ({raw!r}) write 'freeze' into the spec")
        text = raw
    default_spec, type_specs = parse_type_codecs(
        getattr(config, "type_codecs", None) or {}, default=text)
    if getattr(config, "use_param_output_head", False):
        default_spec = replace(default_spec, decoder=HalfSpec(kind="mlp", hidden=128))
    return default_spec, type_specs
