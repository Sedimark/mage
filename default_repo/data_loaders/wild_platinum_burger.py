from mage_ai.data_preparation.decorators import streaming_source
from mage_ai.settings.repo import get_repo_path
from typing import Generator
import subprocess
import logging
import select
import time

logger = logging.getLogger(__name__)

@streaming_source
def stream_server_and_client(**kwargs) -> Generator:
    process_server = None
    process_client = None
    try:
        config_file = get_repo_path() + "/configs/<pipeline_name>/config.yaml"

        # Start server process
        process_server = subprocess.Popen(
            ['python3.11', "/home/src/default_repo/utils/fleviden/scripts/server.py", "--config", config_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Give server time to start up
        time.sleep(2)  # Adjust this delay based on your server startup time

        # Start client process
        process_client = subprocess.Popen(
            ['python3.11', "/home/src/default_repo/utils/fleviden/scripts/client.py", "--config", config_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Set up polling for both processes
        outputs = [process_server.stdout, process_server.stderr,
                  process_client.stdout, process_client.stderr]
        
        while True:
            # Use select to check which streams have data
            readable, _, _ = select.select(outputs, [], [], 1.0)
            
            if not readable:
                # Check if processes are still running
                if process_server.poll() is not None and process_client.poll() is not None:
                    break
                continue

            for stream in readable:
                line = stream.readline()
                if not line:
                    continue

                if stream in [process_server.stdout, process_client.stdout]:
                    yield {'data': line.strip()}
                else:
                    # Handle stderr output
                    logger.error(f"Error: {line.strip()}")

            # Check process status
            if process_server.poll() is not None:
                remaining_stderr = process_server.stderr.read()
                if remaining_stderr:
                    logger.error(f"Server Error: {remaining_stderr}")
            
            if process_client.poll() is not None:
                remaining_stderr = process_client.stderr.read()
                if remaining_stderr:
                    logger.error(f"Client Error: {remaining_stderr}")

    except Exception as e:
        logger.exception("Unexpected error occurred while streaming.")
        yield {'error': str(e)}

    finally:
        # Cleanup processes
        for process in [process_server, process_client]:
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=5)  # Wait for process to terminate
                except subprocess.TimeoutExpired:
                    process.kill()  # Force kill if process doesn't terminate
                except Exception as e:
                    logger.error(f"Error during process cleanup: {e}")