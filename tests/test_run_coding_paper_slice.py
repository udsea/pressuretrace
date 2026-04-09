from __future__ import annotations

import unittest

from pressuretrace.behavior.run_coding_paper_slice import _generation_profile_for_coding_v1


class RunCodingPaperSliceTestCase(unittest.TestCase):
    def test_qwen3_coding_profile_is_deterministic_when_thinking_off(self) -> None:
        profile = _generation_profile_for_coding_v1("Qwen/Qwen3-14B", "off")

        self.assertEqual(profile.backend, "manual_qwen3")
        self.assertFalse(profile.do_sample)
        self.assertFalse(profile.enable_thinking)


if __name__ == "__main__":
    unittest.main()
