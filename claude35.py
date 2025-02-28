"""
Claude 3.5 Sonnet inference module for conversation detection.
Handles message analysis and conversation labeling using Anthropic's Claude 3.5 Sonnet model.
"""

import os
import json
import csv
import logging
import argparse
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import ConversationDetector, Message, Label

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Read the conversation detection prompt
PROMPT_FILE = os.path.join(os.path.dirname(__file__), 'open_source_examples/prompts/conversation_detection_prompt.txt')
with open(PROMPT_FILE, 'r') as f:
    exec(f.read())  # This will load CONVERSATION_DETECTION_PROMPT

# Define output directory relative to the module
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'csv', 'detections')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
BATCH_SIZE = 50  # Process messages in batches of 50 to stay within token limits
MODEL_NAME = "claude-3-5-sonnet-latest"  # Claude 3.5 Sonnet model (latest version)

class Claude35ConversationDetector(ConversationDetector):
    """
    Conversation detector using Claude 3.5 Sonnet for message analysis and topic detection.
    """
    
    def __init__(self):
        """Initialize the Claude detector with API configuration."""
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = MODEL_NAME
        self.labeler_id = "claude35s"  # Identifier for this model

    def detect(self, messages: List[Message]) -> List[Label]:
        """
        Detect conversations in a list of messages.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            List of labels assigning messages to conversations
        """
        # Convert Message objects to dict format expected by _create_analysis_prompt
        message_dicts = [
            {
                'id': msg.id,
                'text': msg.text,
                'timestamp': msg.timestamp,
                'user': msg.user
            }
            for msg in messages
        ]
        
        # Process in batches
        all_labels = []
        for i in range(0, len(message_dicts), BATCH_SIZE):
            batch = message_dicts[i:i + BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1} of {(len(message_dicts) + BATCH_SIZE - 1)//BATCH_SIZE}")
            batch_labels = self._detect_conversations(batch)
            all_labels.extend(batch_labels)
        
        return all_labels

    def _detect_conversations(self, messages: List[Dict[str, Any]]) -> List[Label]:
        """Implementation of conversation detection."""
        try:
            prompt = self._create_analysis_prompt(messages)
            
            response = self.client.messages.create(
                model=self.model,
                system="You are a conversation detection system that analyzes message patterns and content to identify distinct conversations. Your output should be in CSV format with exactly 6 columns: message_id,conversation_id,topic,timestamp,labeler_id,confidence",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000  
            )
            
            # Parse the CSV response
            csv_response = response.content[0].text.strip()
            lines = [line.strip() for line in csv_response.split('\n')]
            
            # Skip empty lines and header
            data_lines = [line for line in lines if line and not line.startswith('message_id,conversation_id')]
            
            # Convert to Label objects
            labels = []
            for line in data_lines:
                try:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) != 6:
                        logger.warning(f"Skipping line with incorrect number of columns: {line}")
                        continue
                        
                    msg_id, conv_id, topic, timestamp, labeler_id, confidence = parts
                    label = Label(
                        message_id=msg_id,
                        conversation_id=conv_id,
                        topic=topic,
                        timestamp=timestamp,
                        metadata={
                            'labeler_id': labeler_id,
                            'confidence': float(confidence)
                        }
                    )
                    labels.append(label)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing line: {line} - Error: {str(e)}")
                    continue
            
            return labels
            
        except Exception as e:
            logger.error(f"Error in Claude 3.5 conversation detection: {str(e)}")
            raise

    def _create_analysis_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Create the analysis prompt for Claude."""
        formatted_messages = "\n"
        for msg in messages:
            formatted_messages += f"Message ID: {msg['id']}\n"
            formatted_messages += f"Timestamp: {msg['timestamp']}\n"
            formatted_messages += f"User: {msg['user']['username']}\n"
            formatted_messages += f"Content: {msg['text']}\n\n"

        return CONVERSATION_DETECTION_PROMPT.replace("[MESSAGES]", formatted_messages)

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Run Claude 3.5 Sonnet conversation detection on a messages file')
    parser.add_argument('input_file', help='Path to input CSV file containing messages')
    parser.add_argument('--output', '-o', help='Path to output file (default: auto-generated in data/csv/detections/)',
                      default=None)
    
    args = parser.parse_args()
    
    # Create detector instance
    detector = Claude35ConversationDetector()
    
    # Load messages
    messages = detector.load_messages(args.input_file)
    logger.info(f"Loaded {len(messages)} messages from {args.input_file}")
    
    # Detect conversations
    labels = detector.detect(messages)
    
    # Generate output filename if not provided
    if args.output is None:
        # Extract group name from input file path
        group_name = 'origintrail'  # Hardcode for now since we know the group
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        args.output = os.path.join(OUTPUT_DIR, f'labels_{timestamp}_{detector.labeler_id}_{group_name}.csv')
    
    # Save results
    detector.save_labels(labels, args.output)
    logger.info(f"Results written to: {args.output}") 