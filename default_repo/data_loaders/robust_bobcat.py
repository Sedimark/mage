from mage_ai.data_preparation.streaming.base import BaseStreamingPipeline
import subprocess
from mage_ai.settings.repo import get_repo_path
from typing import Generator


@streaming_source
class CustomStreamingPipeline(BaseStreamingPipeline):
    def stream(self) -> Generator:
        """
        Template for streaming data from subprocess execution
        """
        
        try:
            # Run the server script
            config_file = get_repo_path() + "/configs/fleviden_pipeline_test/config.yaml"
            process_server = subprocess.Popen(
                ['python3.11', "/home/src/default_repo/utils/fleviden/scripts/server.py", "--config", config_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Run the client script
            process_client = subprocess.Popen(
                ['python3.11', "/home/src/default_repo/utils/fleviden/scripts/client.py", "--config", config_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Stream server output
            for line in process_server.stdout:
                yield self.clean_output(line)

            # Stream client output
            for line in process_client.stdout:
                yield self.clean_output(line)

            # Check for errors
            server_errors = process_server.stderr.read()
            client_errors = process_client.stderr.read()
            
            if server_errors:
                print(f"Server Error: {server_errors}")
            if client_errors:
                print(f"Client Error: {client_errors}")

        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            yield None

    def clean_output(self, line: str) -> dict:
        """
        Clean and format the output line
        Override this method based on your output format needs
        """
        return {
            'data': line.strip()
        }


if __name__ == '__main__':
    pipeline = CustomStreamingPipeline()
    for record in pipeline.stream():
        print(record)