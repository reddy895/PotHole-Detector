import unittest
from unittest.mock import patch

import main


class TestInteractiveMenu(unittest.TestCase):
    """Test interactive menu inputs."""

    @patch("builtins.input", side_effect=["1", "0"])
    def test_interactive_webcam(self, mock_input):
        args = main.interactive_menu()
        self.assertEqual(args.source, "webcam")
        self.assertEqual(args.cam_idx, 0)
        self.assertTrue(args.whatsapp)
        self.assertEqual(args.authority_phone, "+919591152862")

    @patch("builtins.input", side_effect=["2", "road.mp4"])
    @patch("main._pick_file_dialog", return_value="road.mp4")
    def test_interactive_video_picker(self, mock_picker, mock_input):
        args = main.interactive_menu()
        self.assertEqual(args.source, "video")
        self.assertEqual(args.input, "road.mp4")
        self.assertTrue(args.whatsapp)
        self.assertEqual(args.authority_phone, "+919591152862")


if __name__ == "__main__":
    unittest.main()
