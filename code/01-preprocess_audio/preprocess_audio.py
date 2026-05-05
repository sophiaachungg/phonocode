#!/usr/bin/env python3
"""
Audio Preprocessing Script for PhonoCode
Converts .webm files to .wav format (16kHz, mono) for model inference.

Usage:
    # Process all files in a directory
    python preprocess_audio.py ../data_raw/phoneme_reversal/ReXa_311
    
    # Process multiple participants
    python preprocess_audio.py ../data_raw/phoneme_reversal/ReXa_311 ../data_raw/phoneme_reversal/ReXa_221
    
    # Process entire task directory
    python preprocess_audio.py ../data_raw/phoneme_reversal --recursive
    
    # Specify custom output directory
    python preprocess_audio.py ../data_raw/phoneme_reversal/ReXa_311 --output ../custom_output
"""

import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

# Configuration
TARGET_SR = 16000  # 16kHz sample rate (standard for speech models)
DEFAULT_OUTPUT_DIR = Path("../data_processed/phoneme_reversal")


def validate_wav(wav_path):
    """
    Check if a .wav file is valid (correct sample rate, mono, non-empty).
    Returns True if valid, False otherwise.
    """
    try:
        info = sf.info(str(wav_path))
        return (
            info.samplerate == TARGET_SR and 
            info.channels == 1 and 
            info.frames > 0
        )
    except Exception:
        return False


def convert_webm_to_wav(webm_path, output_path, force=False):
    """
    Convert a single .webm file to .wav using FFmpeg.
    
    Args:
        webm_path: Path to input .webm file
        output_path: Path to output .wav file
        force: If True, reprocess even if valid .wav exists
    
    Returns:
        str: 'processed', 'skipped', or 'failed'
    """
    # Check if already processed correctly
    if output_path.exists() and not force:
        if validate_wav(output_path):
            return 'skipped'
        else:
            # Exists but invalid - reprocess
            pass
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command
    command = [
        "ffmpeg",
        "-y",                      # Overwrite if needed
        "-i", str(webm_path),      # Input file
        "-ac", "1",                # Mono (1 audio channel)
        "-ar", str(TARGET_SR),     # Sample rate (16kHz)
        "-loglevel", "error",      # Only show errors
        str(output_path)           # Output file
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return 'processed'
    except subprocess.CalledProcessError as e:
        print(f"\nFFmpeg failed on: {webm_path}")
        print(f"Error: {e.stderr}")
        return 'failed'
    except Exception as e:
        print(f"\nUnexpected error processing {webm_path}: {e}")
        return 'failed'


def process_directory(input_dir, output_dir, recursive=False, force=False):
    """
    Process all .webm files in a directory.
    
    Args:
        input_dir: Path to directory containing .webm files
        output_dir: Path to output directory
        recursive: If True, search subdirectories recursively
        force: If True, reprocess all files even if they exist
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Find all .webm files
    if recursive:
        webm_files = list(input_dir.rglob("*.webm"))
    else:
        webm_files = list(input_dir.glob("*.webm"))
    
    if not webm_files:
        print(f"Warning: No .webm files found in {input_dir}")
        return
    
    print(f"Found {len(webm_files)} .webm file(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target format: 16kHz, mono .wav")
    print("-" * 60)
    
    # Stats
    stats = {'processed': 0, 'skipped': 0, 'failed': 0}
    
    # Process each file
    for webm_path in tqdm(webm_files, desc="Processing", unit="file"):
        # Determine output path
        if recursive:
            # Preserve directory structure
            try:
                rel_path = webm_path.relative_to(input_dir)
            except ValueError:
                # File is not relative to input_dir - use name only
                rel_path = Path(webm_path.name)
            output_path = (output_dir / rel_path).with_suffix(".wav")
        else:
            # Flat structure - just use filename
            output_path = (output_dir / webm_path.name).with_suffix(".wav")
        
        # Convert
        result = convert_webm_to_wav(webm_path, output_path, force=force)
        stats[result] += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Processed:  {stats['processed']} file(s)")
    print(f"Skipped:    {stats['skipped']} file(s) (already valid)")
    print(f"Failed:     {stats['failed']} file(s)")
    print(f"Total:      {len(webm_files)} file(s)")
    print(f"\nOutput saved to: {output_dir}")
    print("=" * 60)


def process_specific_files(file_list, base_input_dir, output_dir, force=False):
    """
    Process a specific list of files.
    
    Args:
        file_list: List of relative file paths
        base_input_dir: Base directory for input files
        output_dir: Output directory
        force: If True, reprocess all files
    """
    base_input_dir = Path(base_input_dir)
    output_dir = Path(output_dir)
    
    print(f"Processing {len(file_list)} specific file(s)")
    print(f"Base input directory: {base_input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    stats = {'processed': 0, 'skipped': 0, 'failed': 0}
    
    for file_path in tqdm(file_list, desc="Processing", unit="file"):
        webm_path = base_input_dir / file_path
        
        if not webm_path.exists():
            print(f"\nWarning: File not found: {webm_path}")
            stats['failed'] += 1
            continue
        
        output_path = (output_dir / file_path).with_suffix(".wav")
        result = convert_webm_to_wav(webm_path, output_path, force=force)
        stats[result] += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Processed:  {stats['processed']} file(s)")
    print(f"Skipped:    {stats['skipped']} file(s)")
    print(f"Failed:     {stats['failed']} file(s)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert .webm audio files to .wav format for PhonoCode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single participant
  python preprocess_audio.py ../data_raw/phoneme_reversal/ReXa_311
  
  # Process multiple participants
  python preprocess_audio.py ../data_raw/phoneme_reversal/ReXa_311 ../data_raw/phoneme_reversal/ReXa_221
  
  # Process entire task directory recursively
  python preprocess_audio.py ../data_raw/phoneme_reversal --recursive
  
  # Force reprocess all files
  python preprocess_audio.py ../data_raw/phoneme_reversal/ReXa_311 --force
  
  # Specify custom output directory
  python preprocess_audio.py ../data_raw/phoneme_reversal/ReXa_311 --output ../custom_output
        """
    )
    
    parser.add_argument(
        'input_paths',
        nargs='+',
        help='Path(s) to input directory/directories containing .webm files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Search subdirectories recursively'
    )
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Force reprocess all files (even if valid .wav exists)'
    )
    parser.add_argument(
        '--check-ffmpeg',
        action='store_true',
        help='Check if FFmpeg is installed and exit'
    )
    
    args = parser.parse_args()
    
    # Check FFmpeg installation
    if args.check_ffmpeg:
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                check=True
            )
            print("✓ FFmpeg is installed")
            print(result.stdout.split('\n')[0])
            sys.exit(0)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ FFmpeg is not installed or not in PATH")
            print("Please install FFmpeg: https://ffmpeg.org/download.html")
            sys.exit(1)
    
    # Verify FFmpeg is available
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg is not installed or not in PATH")
        print("Please install FFmpeg: https://ffmpeg.org/download.html")
        print("On macOS: brew install ffmpeg")
        print("On Ubuntu: sudo apt-get install ffmpeg")
        sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = DEFAULT_OUTPUT_DIR
    
    # Process each input path
    for input_path in args.input_paths:
        process_directory(
            input_dir=input_path,
            output_dir=output_dir,
            recursive=args.recursive,
            force=args.force
        )


if __name__ == "__main__":
    main()
