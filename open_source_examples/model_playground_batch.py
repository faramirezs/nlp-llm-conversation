#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

import yaml
from ollama import Client
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLabeler:
    def __init__(self, config_path: str):
        """Initialize the labeler with configuration."""
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = self._load_config(config_path)
        self.client = Client()
        self.model_name = self.config['model']['names']
        self.prompt = self._load_prompt()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            # If config_path is not absolute, make it relative to script directory
            if not os.path.isabs(config_path):
                config_path = os.path.join(self.script_dir, config_path)

            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            sys.exit(1)

    def _load_prompt(self) -> str:
        """Load the conversation detection prompt from file."""
        try:
            # Make prompt path relative to workspace root
            prompt_path = os.path.join(os.path.dirname(self.script_dir), self.config['prompt']['path'])
            with open(prompt_path, 'r') as f:
                content = f.read()
                # Extract the prompt string from the Python file
                return content.split('"""')[1].strip()
        except Exception as e:
            logger.error(f"Failed to load prompt file: {e}")
            sys.exit(1)

    def _read_messages(self, input_file: str) -> pd.DataFrame:
        """Read and preprocess input messages."""
        try:
            df = pd.read_csv(input_file)
            required_columns = ['id', 'text', 'timestamp', 'username']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Input file must contain columns: {required_columns}")
            return df
        except Exception as e:
            logger.error(f"Failed to read input file: {e}")
            sys.exit(1)

    def _prepare_message_batch(self, messages: pd.DataFrame, batch_idx: int) -> str:
        """Prepare a batch of messages for the model."""
        batch_size = self.config['processing']['batch_size']
        start_idx = batch_idx * batch_size
        batch = messages.iloc[start_idx:start_idx + batch_size]

        formatted_messages = []
        for _, row in batch.iterrows():
            msg = f"ID: {row['id']}\nTimestamp: {row['timestamp']}\nUser: {row['username']}\nMessage: {row['text']}\n"
            formatted_messages.append(msg)

        return "\n".join(formatted_messages)

    def _process_batch(self, messages: str) -> List[Dict[str, Any]]:
        """Process a batch of messages using the configured model."""
        try:
            # Replace [MESSAGES] placeholder in prompt with actual messages
            prompt_with_messages = self.prompt.replace('[MESSAGES]', messages)

            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt_with_messages}
                ]
            )

            # Parse CSV response into list of dictionaries
            try:
                csv_lines = response.message['content'].strip().split('\n')
                reader = csv.DictReader(csv_lines)
                results = list(reader)
                if not results:
                    logger.error(f"Failed to parse model response: {response.message['content']}")
                return results
            except Exception as e:
                logger.error(f"Failed to parse response: {e}")
                logger.error(f"Raw response: {response}")
                return []

        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            return []

    def _save_results(self, results: List[Dict[str, Any]], community_name: str) -> str:
        """Save results to a CSV file in the community's data folder."""
        timestamp = datetime.now().strftime(self.config['output']['timestamp_format'])
        output_dir = os.path.join(self.config['output']['output_dir'], community_name)
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"labels_{timestamp}_{self.model_name}_{community_name}.csv"
        output_path = os.path.join(output_dir, output_filename)

        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['message_id', 'conversation_id', 'topic', 'timestamp', 'labeler_id', 'confidence'])
                writer.writeheader()
                for result in results:
                    writer.writerow({
                        'message_id': result['message_id'],
                        'conversation_id': result['conversation_id'],
                        'topic': result['topic'],
                        'timestamp': result['timestamp'],
                        'labeler_id': self.model_name,
                        'confidence': result.get('confidence', 1.0)
                    })
            return output_path
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None

    def generate_labels(self, input_file: str) -> str:
        """Generate conversation labels for the input messages."""
        logger.info(f"Starting label generation with model: {self.model_name}")

        # Extract community name from input path
        community_name = os.path.basename(os.path.dirname(input_file))

        # Read and process messages
        messages_df = self._read_messages(input_file)
        total_batches = len(messages_df) // self.config['processing']['batch_size'] + 1

        all_results = []
        for batch_idx in range(total_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
            batch_messages = self._prepare_message_batch(messages_df, batch_idx)
            batch_results = self._process_batch(batch_messages)
            all_results.extend(batch_results)

        # Save results
        output_file = self._save_results(all_results, community_name)
        if output_file:
            logger.info(f"Labels generated and saved to: {output_file}")
            logger.info(f"To evaluate results, run: python conversation_metrics.py data/groups/{community_name}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Generate conversation labels using LLMs")
    parser.add_argument("input_file", help="Path to the input messages CSV file")
    parser.add_argument("--config", default="model_config_batch.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    labeler = ModelLabeler(args.config)
    labeler.generate_labels(args.input_file)

if __name__ == "__main__":
    main()
