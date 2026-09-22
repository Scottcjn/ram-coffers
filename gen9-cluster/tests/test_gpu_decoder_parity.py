"""The GPU FP8 decoders must be the same decoder as the CPU ones.

``kernels/fp8.c`` and ``gen9_cluster/fp8.py`` are checked against each other
in ``test_fp8.py``. The Vulkan shader (``kernels/expert.comp``) and the HIP
kernel (``kernels/expert_hip.hip``) each carry their own ``decode_e4m3``, and
CI has no GPU to run them on. Rather than keep a hand copy of the shader in
Python — which would drift exactly the way the shader itself drifted when its
NaN branch was dropped — this test lifts the real function body out of each
GPU source, transliterates that small C/GLSL subset to Python, and runs it over
all 256 codes against ``fp8.TABLE``.

The transliteration is deliberately narrow: integer declarations, ``if`` /
``else if`` / ``else`` blocks, ``return`` with an optional ternary, ``exp2``,
``float()`` casts and the two NaN bit-pattern helpers. If a future edit uses
something outside that subset, the test fails loudly at parse time instead of
passing on a guess.
"""

import math
import pathlib
import re
import shutil
import struct
import subprocess
import unittest

import numpy as np

from gen9_cluster import fp8

KERNELS = pathlib.Path(__file__).resolve().parent.parent / "kernels"
SHADER = KERNELS / "expert.comp"
HIP = KERNELS / "expert_hip.hip"


def _extract_body(source: str) -> str:
    """The text between the braces of ``decode_e4m3``."""
    start = re.search(r"decode_e4m3\s*\([^)]*\)\s*\{", source)
    if start is None:
        raise AssertionError("decode_e4m3 not found")
    depth, i = 1, start.end()
    while depth:
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
        i += 1
    return source[start.end():i - 1]


def _bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


_KNOWN_TOKENS = re.compile(r"^[\w\s().&|<>=+\-*/?:!,\[\]]*$")
_KNOWN_CALLS = {"float", "exp2", "_bits_to_float"}


def _transliterate(body: str) -> str:
    """Turn the restricted C/GLSL subset into a Python function body."""
    lines = []
    indent = 1
    for raw in body.splitlines():
        line = re.sub(r"//.*$|/\*.*?\*/", "", raw).strip()
        if not line:
            continue
        line = line.replace("const ", "")
        line = re.sub(r"\b(unsigned int|uint|float)\s+(\w+)\s*;", "", line)
        line = re.sub(r"\b(unsigned int|uint|float)\s+(\w+\s*=)", r"\2", line)
        line = re.sub(r"\b(0[xX][0-9A-Fa-f]+|\d+)[uU]\b", r"\1", line)
        line = re.sub(r"\b(\d+\.\d*|\.\d+)[fF]\b", r"\1", line)
        line = re.sub(r"\bexp2f?\(", "math.exp2(", line)
        line = re.sub(r"\b(uintBitsToFloat|__uint_as_float|__int_as_float)\(",
                      "_bits_to_float(", line)
        line = line.replace("&&", " and ").replace("||", " or ")
        if not line:
            continue

        if line.startswith("}"):
            indent -= 1
            line = line[1:].strip()
            if not line:
                continue
        block = re.fullmatch(r"(else if|if)\s*\((.*)\)\s*\{", line)
        if block:
            keyword = "elif" if block.group(1) == "else if" else "if"
            lines.append("    " * indent + f"{keyword} {block.group(2)}:")
            indent += 1
            continue
        if re.fullmatch(r"else\s*\{", line):
            lines.append("    " * indent + "else:")
            indent += 1
            continue

        if not line.endswith(";"):
            raise AssertionError(f"unsupported statement: {raw!r}")
        line = line[:-1].strip()
        ternary = re.fullmatch(r"return\s+(.*?)\s*\?\s*(.*?)\s*:\s*(.*)", line)
        if ternary:
            cond, yes, no = ternary.groups()
            line = f"return ({yes}) if ({cond}) else ({no})"
        if not _KNOWN_TOKENS.match(line) or "?" in line:
            raise AssertionError(f"unsupported expression: {raw!r}")
        unknown = set(re.findall(r"\b(\w+)\(", line)) - _KNOWN_CALLS
        if unknown:
            raise AssertionError(f"unsupported call {unknown} in {raw!r}")
        lines.append("    " * indent + line)
    if indent != 1:
        raise AssertionError("unbalanced braces in decode_e4m3")
    return "\n".join(lines)


def _load_decoder(path: pathlib.Path):
    body = _transliterate(_extract_body(path.read_text()))
    namespace = {"math": math, "_bits_to_float": _bits_to_float}
    exec(f"def decode_e4m3(b):\n{body}\n", namespace)
    return namespace["decode_e4m3"]


class _ParityMixin:
    path: pathlib.Path

    def setUp(self):
        self.decode = _load_decoder(self.path)

    def test_every_code_decodes_identically_to_the_cpu_table(self):
        mismatches = []
        for code in range(256):
            gpu = self.decode(code)
            cpu = float(fp8.TABLE[code])
            if math.isnan(cpu):
                if not math.isnan(gpu):
                    mismatches.append(f"{code:#04x}: gpu {gpu} cpu nan")
            elif math.isnan(gpu) or np.float32(gpu) != np.float32(cpu):
                mismatches.append(f"{code:#04x}: gpu {gpu} cpu {cpu}")
        self.assertEqual(mismatches, [])

    def test_only_the_two_nan_codes_are_nan(self):
        nans = [code for code in range(256) if math.isnan(self.decode(code))]
        self.assertEqual(nans, [0x7F, 0xFF])

    def test_the_maximum_normal_is_448(self):
        finite = [self.decode(c) for c in range(256)
                  if not math.isnan(self.decode(c))]
        self.assertEqual(max(finite), 448.0)
        self.assertEqual(min(finite), -448.0)


class TestVulkanShaderDecoder(_ParityMixin, unittest.TestCase):
    path = SHADER

    def test_the_shader_still_compiles(self):
        """Editing the decoder must not break the SPIR-V build. Skipped where
        glslangValidator is absent; the parity tests above still run."""
        glslang = shutil.which("glslangValidator")
        if glslang is None:
            self.skipTest("glslangValidator not installed")
        result = subprocess.run(
            [glslang, "-V", "--target-env", "vulkan1.1", "-S", "comp",
             str(SHADER), "-o", "/dev/null"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0,
                         result.stdout + result.stderr)


class TestHipKernelDecoder(_ParityMixin, unittest.TestCase):
    path = HIP


class TestTransliterator(unittest.TestCase):
    """The lifted-body approach only earns its keep if it refuses to guess."""

    def test_a_decoder_without_the_nan_branch_is_caught(self):
        body = """
            uint sign = (b >> 7) & 0x1u;
            uint exp  = (b >> 3) & 0xFu;
            uint mant = b & 0x7u;
            float mag;
            if (exp == 0u) {
                mag = float(mant) * 0.125 * exp2(-6.0);
            } else {
                mag = (1.0 + float(mant) * 0.125) * exp2(float(exp) - 7.0);
            }
            return sign == 1u ? -mag : mag;
        """
        namespace = {"math": math, "_bits_to_float": _bits_to_float}
        exec(f"def decode_e4m3(b):\n{_transliterate(body)}\n", namespace)
        decode = namespace["decode_e4m3"]
        # This is exactly the pre-fix shader: 0x7F came out as 480, not NaN.
        self.assertEqual(decode(0x7F), 480.0)
        self.assertTrue(math.isnan(float(fp8.TABLE[0x7F])))

    def test_unknown_constructs_fail_loudly(self):
        with self.assertRaises(AssertionError):
            _transliterate("for (int i = 0; i < 4; ++i) { mag += 1.0; }")
        with self.assertRaises(AssertionError):
            _transliterate("return mystery_intrinsic(b);")


if __name__ == "__main__":
    unittest.main()
