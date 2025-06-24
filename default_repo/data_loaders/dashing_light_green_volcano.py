from mage_ai.streaming.sources.base_python import BasePythonSource
from mage_ai.settings.repo import get_repo_path
from typing import Callable
import subprocess
import logging

if 'streaming_source' not in globals():
    from mage_ai.data_preparation.decorators import streaming_source

logger = logging.getLogger(__name__)

@streaming_source
class SubprocessSource(BasePythonSource):
    def init_client(self):
        """
        Initialize the subprocess client
        """
        self.process_server = None
        self.process_client = None
        self.config_file = get_repo_path() + "/configs/sleeping_gray_glacier/config.yaml"

    def clean_output(self, line: str) -> dict:
        """
        Clean and format the output line
        """
        return {'data': line.strip()}

    def batch_read(self, handler: Callable):
        """
        Batch read the messages from the subprocesses
        """
        try:
            # Start server process
            self.process_server = subprocess.Popen(
                ['python3.11', "/home/src/default_repo/utils/fleviden/scripts/server.py", "--config", self.config_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Start client process
            self.process_client = subprocess.Popen(
                ['python3.11', "/home/src/default_repo/utils/fleviden/scripts/client.py", "--config", self.config_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Handle server output
            for line in self.process_server.stdout:
                handler(self.clean_output(line))

            # Handle client output
            for line in self.process_client.stdout:
                handler(self.clean_output(line))

            # Check for errors
            server_errors = self.process_server.stderr.read()
            client_errors = self.process_client.stderr.read()

            if server_errors:
                logger.error(f"Server Error: {server_errors}")
            if client_errors:
                logger.error(f"Client Error: {client_errors}")

        except Exception as e:
            logger.exception("An unexpected error occurred")
            handler({'error': str(e)})

        finally:
            self.cleanup()

    def cleanup(self):
        """
        Cleanup method to terminate processes
        """
        if self.process_server:
            self.process_server.terminate()
        if self.process_client:
            self.process_client.terminate()

    def stop(self):
        """
        Stop method to ensure cleanup
        """
        self.cleanup()