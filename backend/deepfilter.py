import os
import subprocess
import shutil
from pathlib import Path

class DeepFilterInference:
    def __init__(self):
        self.bin_path = Path("bin/deep-filter.exe").resolve()
        if not self.bin_path.exists():
            raise FileNotFoundError(f"DeepFilterNet3 binary not found at {self.bin_path}. Please ensure it is downloaded.")
        print(f"Initializing DeepFilterNet3 using binary at {self.bin_path}...")

    def process(self, audio_path: str, output_path: str):
        """
        Runs the deep-filter executable on the input audio.
        Saves the output to the specified output_path.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Input audio not found at {audio_path}")
            
        temp_dir = Path("temp_df").resolve()
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Run the deep-filter executable
            result = subprocess.run(
                [str(self.bin_path), "-o", str(temp_dir), str(audio_path)],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                print(f"DeepFilterNet3 stdout: {result.stdout}")
                print(f"DeepFilterNet3 stderr: {result.stderr}")
                raise RuntimeError(f"DeepFilterNet3 failed with return code {result.returncode}")
                
            # The output file is created inside temp_dir
            # For a file named 'audio.wav', the output is usually 'audio_DeepFilterNet3.wav'
            input_stem = Path(audio_path).stem
            expected_output = temp_dir / f"{input_stem}_DeepFilterNet3.wav"
            
            if expected_output.exists():
                shutil.copy2(expected_output, output_path)
            else:
                # If exact name is different, just grab the first WAV file generated in temp_dir
                generated_files = list(temp_dir.glob("*.wav"))
                if not generated_files:
                    print(f"DeepFilterNet3 stdout: {result.stdout}")
                    print(f"DeepFilterNet3 stderr: {result.stderr}")
                    raise FileNotFoundError(f"DeepFilterNet3 did not generate an output file in {temp_dir}")
                shutil.copy2(generated_files[0], output_path)
                
            print(f"Successfully processed audio with actual DeepFilterNet3.")
                
        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        return output_path
