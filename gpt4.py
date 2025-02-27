"""
GPT-4 inference module for conversation detection.
Handles message analysis and conversation labeling using OpenAI's GPT-4 model.
"""

import os
import json
import csv
import logging
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
import openai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Define base classes that were previously imported
class Message:
    def __init__(self, id: str, text: str, timestamp: str, user: Dict[str, str]):
        self.id = id
        self.text = text
        self.timestamp = timestamp
        self.user = user

class Label:
    def __init__(self, message_id: str, conversation_id: str, topic: str, timestamp: str, metadata: Dict[str, Any]):
        self.message_id = message_id
        self.conversation_id = conversation_id
        self.topic = topic
        self.timestamp = timestamp
        self.metadata = metadata

class ConversationDetector:
    def load_messages(self, input_file: str) -> List[Message]:
        """Load messages from CSV file."""
        messages = []
        with open(input_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                user = {
                    'username': row['username'],
                    'first_name': row.get('first_name', ''),
                    'last_name': row.get('last_name', '')
                }
                message = Message(
                    id=row['id'],
                    text=row['text'],
                    timestamp=row['timestamp'],
                    user=user
                )
                messages.append(message)
        return messages

    def save_labels(self, labels: List[Label], output_file: str):
        """Save labels to CSV file."""
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['message_id', 'conversation_id', 'topic', 'timestamp', 'labeler_id', 'confidence'])
            for label in labels:
                writer.writerow([
                    label.message_id,
                    label.conversation_id,
                    label.topic,
                    label.timestamp,
                    label.metadata['labeler_id'],
                    label.metadata['confidence']
                ])

# Read the conversation detection prompt
PROMPT_FILE = os.path.join(os.path.dirname(__file__), 'open_source_examples/prompts/conversation_detection_prompt.txt')
with open(PROMPT_FILE, 'r') as f:
    exec(f.read())  # This will load CONVERSATION_DETECTION_PROMPT

# Constants
BATCH_SIZE = 50  # Process messages in batches of 50 to stay within token limits
MODEL_NAME = "gpt-4-0125-preview"  # GPT-4 Turbo model

class GPT4ConversationDetector(ConversationDetector):
    """
    Conversation detector using GPT-4-Turbo for message analysis and topic detection.
    """
    
    def __init__(self):
        """Initialize the GPT-4 detector with API configuration."""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = MODEL_NAME
        self.labeler_id = "gpt4o"  

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
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a conversation detection system that analyzes message patterns and content to identify distinct conversations. Output ONLY the CSV data with no additional text or formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse the CSV response
            csv_response = response.choices[0].message.content.strip()
            
            # Remove any markdown formatting
            if csv_response.startswith('```csv'):
                csv_response = csv_response[6:]
            if csv_response.endswith('```'):
                csv_response = csv_response[:-3]
            
            # Split into lines and remove empty lines
            lines = [line.strip() for line in csv_response.split('\n') if line.strip()]
            
            # Skip header if present
            if lines and lines[0].startswith('message_id,conversation_id'):
                lines = lines[1:]
            
            # Convert to Label objects
            labels = []
            for line in lines:
                try:
                    parts = line.split(',')
                    if len(parts) != 6:
                        logger.warning(f"Skipping malformed line: {line} - Expected 6 parts, got {len(parts)}")
                        continue
                        
                    msg_id, conv_id, topic, timestamp, labeler_id, confidence = parts
                    
                    # Clean up any quotes
                    msg_id = msg_id.strip('"')
                    conv_id = conv_id.strip('"')
                    topic = topic.strip('"')
                    timestamp = timestamp.strip('"')
                    labeler_id = labeler_id.strip('"')
                    confidence = confidence.strip('"')
                    
                    try:
                        confidence = float(confidence)
                    except ValueError:
                        confidence = 0.5  # Default confidence if parsing fails
                    
                    label = Label(
                        message_id=msg_id,
                        conversation_id=conv_id,
                        topic=topic,
                        timestamp=timestamp,
                        metadata={
                            'labeler_id': labeler_id,
                            'confidence': confidence
                        }
                    )
                    labels.append(label)
                except Exception as e:
                    logger.warning(f"Skipping malformed line: {line} - Error: {str(e)}")
                    continue
            
            return labels
            
        except Exception as e:
            logger.error(f"Error in GPT-4 conversation detection: {str(e)}")
            raise

    def _create_analysis_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Create the analysis prompt for GPT-4."""
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
    
    parser = argparse.ArgumentParser(description='Run GPT-4 conversation detection on a messages file')
    parser.add_argument('input_file', help='Path to input CSV file containing messages')
    parser.add_argument('--output', '-o', help='Path to output file',
                      default='gpt4_results.csv')
    
    args = parser.parse_args()
    
    # Create detector instance
    detector = GPT4ConversationDetector()
    
    # Load messages
    messages = detector.load_messages(args.input_file)
    logger.info(f"Loaded {len(messages)} messages from {args.input_file}")
    
    # Detect conversations
    labels = detector.detect(messages)
    
    # Save results
    detector.save_labels(labels, args.output)
    logger.info(f"Results written to: {args.output}") 