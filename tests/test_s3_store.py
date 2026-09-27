import unittest

from ingest_worker.s3_store import safe_filename


class S3StoreTests(unittest.TestCase):
    def test_safe_filename_strips_paths_and_unsafe_characters(self):
        self.assertEqual(safe_filename("s3://bucket/folder/a%20b.wav"), "a_b.wav")
        self.assertEqual(safe_filename("../../etc/passwd"), "passwd")

    def test_safe_filename_has_fallback(self):
        self.assertEqual(safe_filename("s3://bucket/..."), "audio.bin")


if __name__ == "__main__":
    unittest.main()
