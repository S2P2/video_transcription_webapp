import unittest
from collections import namedtuple
import sys
import os

# Adjust path to import from the root directory if run.py is there
# This assumes test_run.py is in the root, same as run.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Attempt to import the necessary functions from run.py
# We anticipate potential issues with heavy imports in run.py,
# but proceed as per the assumption that these specific functions are usable.
try:
    from run import format_transcription, format_time
except ImportError as e:
    # If there's an ImportError, it might be due to dependencies in run.py
    # not available in the test environment or issues with global initializations.
    # For this exercise, we'll print a warning and define stubs if needed,
    # though ideally, the module should be importable.
    print(f"Warning: Could not import from run.py: {e}. Tests might not run correctly.")
    print("This test expects 'format_transcription' and 'format_time' to be importable.")
    
    # Define a placeholder format_time if not imported, to allow tests to be defined.
    # This won't replicate the true behavior but allows test structure.
    if 'format_time' not in globals():
        def format_time(seconds: float) -> str:
            # Simplified stub if actual import fails.
            # The real format_time uses int(seconds), then gmtime.
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

    # Placeholder for format_transcription if needed, though this is the function under test.
    # If this isn't importable, the tests cannot meaningfully run.

# Define a simple Segment mock as per the prompt
Segment = namedtuple('Segment', ['start', 'end', 'text'])

class TestFormatTranscription(unittest.TestCase):

    def test_speaker_attribution_and_unknown(self):
        """Tests basic speaker attribution and handling of unknown speakers."""
        mock_whisper_segments = [
            Segment(0.0, 2.0, "Hello world"),      # Corresponds to Speaker_A
            Segment(2.5, 4.0, "How are you"),      # Corresponds to Speaker_B
            Segment(4.5, 6.0, "I am fine"),        # Corresponds to Speaker_A
            Segment(7.0, 8.0, "This is unknown")   # No RTTM overlap
        ]
        mock_rttm_data = [
            ("Speaker_A", 0.0, 2.2),  # Overlaps "Hello world" fully
            ("Speaker_B", 2.3, 4.3),  # Overlaps "How are you" fully
            ("Speaker_A", 4.4, 6.5)   # Overlaps "I am fine" fully
        ]
        
        # Expected outputs (timestamps based on int(segment.start) by format_time)
        # 0.0 -> 00:00:00
        # 2.5 -> 00:00:02
        # 4.5 -> 00:00:04
        # 7.0 -> 00:00:07
        expected_ts_output = "[00:00:00] Speaker_A: Hello world\n" \
                             "[00:00:02] Speaker_B: How are you\n" \
                             "[00:00:04] Speaker_A: I am fine\n" \
                             "[00:00:07] UnknownSpeaker: This is unknown"

        expected_plain_output = "Speaker_A: Hello world\n" \
                                "Speaker_B: How are you\n" \
                                "Speaker_A: I am fine\n" \
                                "UnknownSpeaker: This is unknown"

        plain_text, text_with_timestamps = format_transcription(mock_whisper_segments, mock_rttm_data, offset=0.0)
        
        self.assertEqual(text_with_timestamps, expected_ts_output)
        self.assertEqual(plain_text, expected_plain_output)

    def test_empty_rttm_data(self):
        """Tests behavior when RTTM data is empty; all speakers should be UnknownSpeaker."""
        mock_whisper_segments = [
            Segment(0.0, 2.0, "Hello"),
            Segment(2.1, 3.5, "Another segment")
        ]
        mock_rttm_data = [] # Empty RTTM data
        
        expected_ts_output = "[00:00:00] UnknownSpeaker: Hello\n" \
                             "[00:00:02] UnknownSpeaker: Another segment"
        expected_plain_output = "UnknownSpeaker: Hello\n" \
                                "UnknownSpeaker: Another segment"

        plain_text, text_with_timestamps = format_transcription(mock_whisper_segments, mock_rttm_data, offset=0.0)

        self.assertEqual(text_with_timestamps, expected_ts_output)
        self.assertEqual(plain_text, expected_plain_output)

    def test_offset_handling(self):
        """Tests if the offset is correctly applied to segment times before formatting."""
        mock_whisper_segments = [
            Segment(1.0, 2.0, "Offset segment")
        ]
        mock_rttm_data = [
            ("Speaker_Offset", 10.0, 12.0) # RTTM times are absolute
        ]
        video_offset = 9.5 # Whisper segment times are relative to this offset
        
        # Whisper segment: 1.0 (relative) -> 1.0 + 9.5 = 10.5 (absolute)
        # RTTM segment: Speaker_Offset from 10.0 to 12.0
        # Overlap exists. Timestamp should be for 10.5 (absolute) -> "00:00:10"
        
        expected_ts_output = "[00:00:10] Speaker_Offset: Offset segment"
        expected_plain_output = "Speaker_Offset: Offset segment"

        plain_text, text_with_timestamps = format_transcription(mock_whisper_segments, mock_rttm_data, offset=video_offset)

        self.assertEqual(text_with_timestamps, expected_ts_output)
        self.assertEqual(plain_text, expected_plain_output)

    def test_multiple_rttm_overlap_selects_max_overlap(self):
        """Tests that the RTTM segment with the largest overlap is chosen."""
        mock_whisper_segments = [
            Segment(0.0, 5.0, "Long segment with multiple overlaps") 
        ]
        mock_rttm_data = [
            ("Speaker_Brief", 0.0, 1.0),      # Overlap: 1.0s (0 to 1)
            ("Speaker_Max", 0.5, 4.5),        # Overlap: 4.0s (0.5 to 4.5) -> Max
            ("Speaker_Partial", 3.0, 5.0)     # Overlap: 2.0s (3 to 5)
        ]
        
        expected_ts_output = "[00:00:00] Speaker_Max: Long segment with multiple overlaps"
        expected_plain_output = "Speaker_Max: Long segment with multiple overlaps"

        plain_text, text_with_timestamps = format_transcription(mock_whisper_segments, mock_rttm_data, offset=0.0)
        
        self.assertEqual(text_with_timestamps, expected_ts_output)
        self.assertEqual(plain_text, expected_plain_output)

    def test_rttm_order_for_equal_max_overlap(self):
        """
        Tests that if multiple RTTM segments have the same maximum overlap, 
        the first one encountered in the list is chosen.
        """
        mock_whisper_segments = [
            Segment(0.0, 4.0, "Segment with two equal overlaps") 
        ]
        # Both Speaker_First and Speaker_Second overlap fully (4.0s)
        # Speaker_First is earlier in the list.
        mock_rttm_data = [
            ("Speaker_First", 0.0, 4.0), # Max overlap, first in list
            ("Speaker_Second", 0.0, 4.0) # Same max overlap, second
        ]
        
        expected_ts_output = "[00:00:00] Speaker_First: Segment with two equal overlaps"
        expected_plain_output = "Speaker_First: Segment with two equal overlaps"

        plain_text, text_with_timestamps = format_transcription(mock_whisper_segments, mock_rttm_data, offset=0.0)
        
        self.assertEqual(text_with_timestamps, expected_ts_output)
        self.assertEqual(plain_text, expected_plain_output)

    def test_no_overlap_between_whisper_and_rttm(self):
        """Tests scenario where whisper segments and RTTM segments do not overlap at all."""
        mock_whisper_segments = [
            Segment(0.0, 2.0, "Early segment"),
            Segment(5.0, 7.0, "Late segment")
        ]
        mock_rttm_data = [
            ("Speaker_Mid", 2.5, 4.5) # Only in the middle, no overlap
        ]
        
        expected_ts_output = "[00:00:00] UnknownSpeaker: Early segment\n" \
                             "[00:00:05] UnknownSpeaker: Late segment"
        expected_plain_output = "UnknownSpeaker: Early segment\n" \
                                "UnknownSpeaker: Late segment"

        plain_text, text_with_timestamps = format_transcription(mock_whisper_segments, mock_rttm_data, offset=0.0)
        
        self.assertEqual(text_with_timestamps, expected_ts_output)
        self.assertEqual(plain_text, expected_plain_output)

if __name__ == '__main__':
    # This allows running the tests from the command line
    unittest.main(verbosity=2)
```
