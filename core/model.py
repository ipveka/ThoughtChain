import streamlit as st
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
from typing import Optional, Dict, Any
import gc
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages Chain of Thought reasoning with actual Phi-2 model."""
    
    def __init__(self, model_name="microsoft/phi-2", use_quantization=True, device=None):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.device = device or self._get_device()
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        logger.info(f"Initializing ModelManager for {model_name}")
        logger.info(f"Device: {self.device}, Quantization: {use_quantization}")
        
        # Load the model
        self._load_model()
    
    def _get_device(self):
        """Automatically detect and return the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_auth_token(self):
        """Get authentication token from environment variables."""
        token = os.getenv('HUGGINGFACE_TOKEN')
        if not token or token == 'your_token_here':
            raise ValueError(
                "Hugging Face token not found. Please:\n"
                "1. Get your token from https://huggingface.co/settings/tokens\n"
                "2. Update the .env file with: HUGGINGFACE_TOKEN=your_actual_token"
            )
        return token
    
    def _load_model(self):
        """Load the Phi-2 model and tokenizer."""
        try:
            # Get authentication token
            auth_token = self._get_auth_token()
            logger.info("Authentication token loaded successfully")
            
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left",
                token=auth_token
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Loading model...")
            
            # Configure quantization if requested
            if self.use_quantization and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                logger.info("Using 4-bit quantization")
            else:
                quantization_config = None
            
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                token=auth_token
            )
            
            # Move to device if not using device_map
            if self.device != "cuda" or quantization_config is None:
                self.model = self.model.to(self.device)
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
            raise e
    
    def generate_response(self, prompt: str, max_length: int = 1024, temperature: float = 0.3, 
                         do_sample: bool = True, top_p: float = 0.9, top_k: int = 50) -> str:
        """Generate a real Chain of Thought response using the loaded model."""
        if not self.model_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Please initialize the model first.")
        
        try:
            # Create Chain of Thought prompt
            cot_prompt = self._create_cot_prompt(prompt)
            
            logger.info(f"Generating response with max_length={max_length}, temperature={temperature}")
            
            # Tokenize input
            inputs = self.tokenizer(
                cot_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  # Increased input length limit
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    length_penalty=1.0,
                    # Remove early_stopping to avoid warnings when num_beams=1
                    num_beams=1
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove the input prompt)
            response = generated_text[len(cot_prompt):].strip()
            
            logger.info(f"Generated response length: {len(response)} characters")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise e
    
    def _create_cot_prompt(self, problem: str) -> str:
        """Create an optimized Chain of Thought prompt for the given problem."""
        problem_type = self._detect_problem_type(problem)
        
        # Base CoT prompt template
        base_prompt = """You are an AI assistant that thinks through problems step by step. Always show your reasoning process clearly.

Problem: {problem}

Let's think through this step by step:

"""
        
        # Add problem-specific guidance
        if problem_type == "math":
            guidance = "For this math problem, I'll break it down into clear steps, show my calculations, and verify my answer."
        elif problem_type == "logic":
            guidance = "For this logic problem, I'll identify the key relationships, apply logical reasoning, and draw clear conclusions."
        elif problem_type == "riddle":
            guidance = "For this riddle, I'll analyze the clues carefully, consider wordplay and metaphors, and think creatively."
        else:
            guidance = "I'll approach this systematically, breaking it down into manageable parts and thinking through each step carefully."
        
        # Combine everything
        full_prompt = base_prompt.format(problem=problem) + guidance + "\n\n"
        
        return full_prompt
    
    def _detect_problem_type(self, problem: str) -> str:
        """Detect the type of problem to optimize prompting."""
        problem_lower = problem.lower()
        
        # Math keywords
        math_keywords = ['calculate', 'solve', 'equation', 'multiply', 'divide', 'add', 'subtract', 
                        'percent', '%', 'fraction', 'decimal', 'number', 'sum', 'difference',
                        'mph', 'miles', 'hours', 'minutes', 'seconds', 'distance', 'time',
                        'workers', 'days', 'build', 'complete', 'fence', 'perimeter', 'area']
        
        # Logic keywords
        logic_keywords = ['if', 'then', 'either', 'or', 'all', 'some', 'none', 'always', 'never',
                         'taller', 'shorter', 'faster', 'slower', 'before', 'after', 'position',
                         'race', 'finished', 'student', 'passed', 'class', 'roses', 'flowers']
        
        # Riddle keywords
        riddle_keywords = ['riddle', 'what am i', 'guess', 'mystery', 'puzzle', 'keys', 'locks',
                          'space', 'room', 'enter', 'outside', 'wetter', 'dries', 'young', 'old',
                          'eye', 'cannot see', 'take', 'leave behind']
        
        if any(keyword in problem_lower for keyword in math_keywords):
            return "math"
        elif any(keyword in problem_lower for keyword in logic_keywords):
            return "logic"
        elif any(keyword in problem_lower for keyword in riddle_keywords):
            return "riddle"
        else:
            return "general"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the loaded model."""
        if not self.model_loaded:
            return {
                "name": "No model loaded",
                "device": "N/A",
                "quantized": False,
                "parameters": "N/A",
                "status": "Not loaded"
            }
        
        # Get model parameters count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "name": self.model_name,
            "device": self.device,
            "quantized": self.use_quantization,
            "total_parameters": f"{total_params:,}",
            "trainable_parameters": f"{trainable_params:,}",
            "status": "Loaded and ready",
            "model_type": "Phi-2 (2.7B parameters)"
        }
    
    def cleanup(self):
        """Clean up model resources to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        self.model_loaded = False
        logger.info("Model resources cleaned up")
    
    def is_ready(self) -> bool:
        """Check if the model is ready for inference."""
        return self.model_loaded and self.model is not None and self.tokenizer is not None
