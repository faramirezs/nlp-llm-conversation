#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import subprocess

import yaml
from ollama import Client
import pandas as pd
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLabeler:
    def __init__(self, config_path: str, verbose: bool = False):
        """Initialize the labeler with configuration."""
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = self._load_config(config_path)
        self.gpu_id = self._select_gpu() if self.config.get('gpu', {}).get('enabled', False) else None
        if self.gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            logger.info(f"Using GPU {self.gpu_id}")
        else:
            logger.info("Using CPU for computation")
        self.client = Client()
        self.model_name = self.config['model']['name']
        self.prompt = self._load_prompt()
        self.verbose = verbose
        
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

    def _select_gpu(self) -> int:
        """Select the least utilized GPU."""
        try:
            # Run nvidia-smi to get GPU utilization
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, check=True)
            
            # Parse the output
            gpus = []
            for line in result.stdout.strip().split('\n'):
                idx, util, mem = map(float, line.split(','))
                gpus.append({'index': int(idx), 'utilization': util, 'memory': mem})
            
            if not gpus:
                if self.config.get('gpu', {}).get('fallback_to_cpu', True):
                    logger.warning("No GPUs found, falling back to CPU")
                    return None
                else:
                    raise RuntimeError("No GPUs found and CPU fallback is disabled")
            
            # Sort by utilization and memory usage
            gpus.sort(key=lambda x: (x['utilization'], x['memory']))
            
            # Return the index of the least utilized GPU
            return gpus[0]['index']
            
        except (subprocess.SubprocessError, ValueError) as e:
            if self.config.get('gpu', {}).get('fallback_to_cpu', True):
                logger.warning(f"Failed to query GPUs ({str(e)}), falling back to CPU")
                return None
            else:
                raise RuntimeError(f"Failed to query GPUs: {str(e)}")

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
            
            # Log the prompt being sent to the model
            logger.debug(f"Sending prompt to model:\n{prompt_with_messages}")
            
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt_with_messages}
                ]
            )
            
            # Log raw response for debugging
            raw_content = response.message['content'].strip()
            logger.info(f"Raw model response:\n{raw_content}")

            # Skip lines between <think> tags in the response
            filtered_lines = []
            skip = False
            for line in raw_content.split('\n'):
                if '<think>' in line:
                    skip = True
                if not skip:
                    filtered_lines.append(line)
                if '</think>' in line:
                    skip = False
            filtered_response = '\n'.join(filtered_lines)
            
            def standardize_column_name(col: str) -> str:
                """Standardize column names to our expected format."""
                col = col.strip().lower()
                if col in ['user_id', 'user']:
                    return 'message_id'
                if col in ['conv_id', 'conversation']:
                    return 'conversation_id'
                return col
            
            def extract_csv_content(text: str) -> str:
                """Extract CSV content from text, removing markdown and extra content."""
                # Remove markdown code blocks
                text = re.sub(r'```csv\s*|\s*```', '', text)
                
                # Find CSV-like content (header row followed by data)
                csv_pattern = r'([a-zA-Z_]+,\s*[a-zA-Z_]+.*?\n.*?)(?:\n\n|$)'
                csv_match = re.search(csv_pattern, text, re.DOTALL)
                if csv_match:
                    return csv_match.group(1)
                return text
            
            def standardize_timestamp(timestamp: str) -> str:
                """Standardize timestamp format."""
                if not timestamp:
                    return datetime.now().strftime('%Y%m%d%H%M%S')
                # Remove any non-alphanumeric characters
                clean_ts = re.sub(r'[^0-9]', '', timestamp)
                # Ensure it's a valid timestamp string
                if len(clean_ts) >= 14:  # YYYYMMDDHHmmss
                    return clean_ts[:14]
                return datetime.now().strftime('%Y%m%d%H%M%S')
            
            try:
                # Extract and clean CSV content from filtered response
                csv_content = extract_csv_content(filtered_response)
                
                # Parse CSV content
                lines = csv_content.strip().split('\n')
                if not lines:
                    raise ValueError("No content found")
                
                # Get and standardize headers
                headers = [standardize_column_name(col) for col in lines[0].split(',')]
                required_fields = {'message_id', 'conversation_id', 'topic', 'timestamp'}
                
                # Check if we have all required fields
                if not all(field in headers for field in required_fields):
                    raise ValueError(f"Missing required fields. Found: {headers}")
                
                results = []
                for line in lines[1:]:
                    if not line.strip() or line.startswith('<'):  # Skip empty lines and XML-like tags
                        continue
                    
                    # Split line and pad with empty strings if needed
                    values = [v.strip() for v in line.split(',')]
                    values.extend([''] * (len(headers) - len(values)))  # Pad with empty strings
                    
                    row = {}
                    for header, value in zip(headers, values):
                        if header in required_fields:
                            row[header] = value
                    
                    # Ensure required fields exist and standardize timestamp
                    if all(field in row for field in required_fields):
                        row['timestamp'] = standardize_timestamp(row['timestamp'])
                        results.append(row)
                
                if results:
                    logger.info(f"Successfully parsed {len(results)} results")
                    return results
                
                logger.error("No valid results found after parsing")
                return []
                
            except Exception as e:
                logger.error(f"Failed to parse response: {e}")
                logger.error(f"Raw response: {raw_content}")
                return []
            
        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            return []

    def _save_results(self, results: List[Dict[str, Any]], community_name: str) -> str:
        """Save results to a CSV file in the community's data folder."""
        timestamp = datetime.now().strftime(self.config['output']['timestamp_format'])
        output_dir = os.path.join(self.config['output']['output_dir'], community_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract prompt filename without extension
        prompt_filename = os.path.splitext(os.path.basename(self.config['prompt']['path']))[0]
        
        # Combine model name with prompt filename using hyphen delimiter
        model_identifier = f"{self.model_name}-{prompt_filename}"
        
        output_filename = f"labels_{timestamp}_{model_identifier}_{community_name}.csv"
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
            if self.verbose:
                print("\nVerbose error output:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"Attempted to save to: {output_path}")
                print("\nResults that failed to save:")
                for idx, result in enumerate(results):
                    print(f"\nResult {idx + 1}:")
                    print(json.dumps(result, indent=2))
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
    parser.add_argument("--config", default="model_config.yaml", help="Path to the configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose error output")
    args = parser.parse_args()

    labeler = ModelLabeler(args.config, verbose=args.verbose)
    labeler.generate_labels(args.input_file)

if __name__ == "__main__":
    main() 