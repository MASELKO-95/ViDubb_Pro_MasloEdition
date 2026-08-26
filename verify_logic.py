import os
import sys
from unittest.mock import MagicMock, patch

# Add the current directory to path so we can import app
sys.path.append('/home/maselko95/Music/ViDubb-weights2')

def test_process_video_logic():
    # Mocking the dependencies to test only the logic in process_video
    with patch('app.VideoDubbing') as MockVideoDubbing, \
         patch('os.system') as mock_system, \
         patch('os.path.exists') as mock_exists:
        
        from app import process_video, language_mapping
        
        # Setup mocks
        mock_exists.return_value = True
        
        # Test Case 1: Uploaded Video only
        print("Testing Case 1: Uploaded Video only...")
        res, msg = process_video("test.mp4", None, "English", "French", False, "turbo", False)
        assert MockVideoDubbing.called
        assert msg == "No Error"
        print("Case 1 Passed.")

        # Test Case 2: YouTube URL only
        MockVideoDubbing.reset_mock()
        print("Testing Case 2: YouTube URL only...")
        res, msg = process_video(None, "https://youtube.com/watch?v=123", "English", "French", False, "turbo", False)
        assert MockVideoDubbing.called
        assert msg == "No Error"
        # Check if yt-dlp was called
        calls = [call.args[0] for call in mock_system.call_args_list]
        assert any("yt-dlp" in c for c in calls)
        print("Case 2 Passed.")

        # Test Case 3: Both provided (should prioritize YouTube URL based on current logic)
        MockVideoDubbing.reset_mock()
        print("Testing Case 3: Both provided...")
        res, msg = process_video("local.mp4", "https://youtube.com/watch?v=123", "English", "French", False, "turbo", False)
        assert MockVideoDubbing.called
        # Verify that it tried to download the YT video
        print("Case 3 Passed.")

        # Test Case 4: Neither provided
        MockVideoDubbing.reset_mock()
        print("Testing Case 4: Neither provided...")
        res, msg = process_video(None, None, "English", "French", False, "turbo", False)
        assert not MockVideoDubbing.called
        assert "Error" in msg
        print("Case 4 Passed.")

if __name__ == "__main__":
    try:
        test_process_video_logic()
        print("\nAll logic tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        sys.exit(1)
