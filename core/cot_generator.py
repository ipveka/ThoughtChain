import re
import logging
from typing import List, Dict, Optional
import time

logger = logging.getLogger(__name__)

class CoTGenerator:
    """Generates Chain of Thought reasoning using a real language model."""
    
    def __init__(self, model_manager, max_length=1024):
        self.model_manager = model_manager
        self.max_length = max_length
        
        # Enhanced CoT templates for better reasoning
        self.cot_templates = {
            "math": """Let's solve this step by step:

1) First, I need to understand what's being asked
2) I'll break down the problem into smaller parts
3) I'll perform calculations step by step
4) I'll verify my answer makes sense

Problem: {problem}

Let me think through this:""",
            
            "logic": """Let's analyze this logically:

1) I'll identify the key information given
2) I'll look for relationships and patterns
3) I'll apply logical reasoning step by step
4) I'll check if my conclusion follows from the premises

Problem: {problem}

Let me think through this:""",
            
            "riddle": """Let's solve this riddle:

1) I'll read the riddle carefully
2) I'll look for wordplay and metaphors
3) I'll consider different interpretations
4) I'll think creatively about possible answers

Problem: {problem}

Let me think through this:""",
            
            "general": """Let's think through this step by step:

1) I'll understand the problem clearly
2) I'll break it down into manageable parts
3) I'll work through each part systematically
4) I'll arrive at a well-reasoned conclusion

Problem: {problem}

Let me think through this:"""
        }
    
    def _detect_problem_type(self, problem: str) -> str:
        """Detect the type of problem to use appropriate template."""
        problem_lower = problem.lower()
        
        # Enhanced math keywords
        math_keywords = ['calculate', 'solve', 'equation', 'multiply', 'divide', 'add', 'subtract', 
                        'percent', '%', 'fraction', 'decimal', 'number', 'sum', 'difference',
                        'mph', 'miles', 'hours', 'minutes', 'seconds', 'distance', 'time',
                        'workers', 'days', 'build', 'complete', 'fence', 'perimeter', 'area',
                        'price', 'cost', 'discount', 'tax', 'total', 'amount']
        
        # Enhanced logic keywords
        logic_keywords = ['if', 'then', 'either', 'or', 'all', 'some', 'none', 'always', 'never',
                         'taller', 'shorter', 'faster', 'slower', 'before', 'after', 'position',
                         'race', 'finished', 'student', 'passed', 'class', 'roses', 'flowers',
                         'conclude', 'therefore', 'because', 'since', 'given', 'assume']
        
        # Enhanced riddle keywords
        riddle_keywords = ['riddle', 'what am i', 'guess', 'mystery', 'puzzle', 'keys', 'locks',
                          'space', 'room', 'enter', 'outside', 'wetter', 'dries', 'young', 'old',
                          'eye', 'cannot see', 'take', 'leave behind', 'clue', 'hint']
        
        if any(keyword in problem_lower for keyword in math_keywords):
            return "math"
        elif any(keyword in problem_lower for keyword in logic_keywords):
            return "logic"
        elif any(keyword in problem_lower for keyword in riddle_keywords):
            return "riddle"
        else:
            return "general"
    
    def generate_cot(self, problem: str, temperature: float = 0.3, top_p: float = 0.9) -> str:
        """Generate Chain of Thought reasoning for a given problem."""
        if not self.model_manager or not self.model_manager.is_ready():
            raise RuntimeError("Model not ready. Please initialize the model first.")
        
        try:
            # Generate response using the model manager
            response = self.model_manager.generate_response(
                problem,
                max_length=self.max_length,
                temperature=temperature,
                top_p=top_p
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating CoT: {e}")
            raise e
    
    def parse_steps(self, cot_response: str) -> List[Dict[str, str]]:
        """Parse the CoT response into individual reasoning steps with enhanced pattern matching."""
        try:
            steps = []
            
            # Enhanced step patterns for real model outputs
            step_patterns = [
                r'Step\s+\d+[:\-]?\s*',  # Step 1:, Step 1-, Step 1 
                r'\d+[\.\)]\s*',         # 1., 1), 2., 2)
                r'First[,\s]',           # First, First 
                r'Second[,\s]',          # Second, Second 
                r'Third[,\s]',           # Third, Third 
                r'Fourth[,\s]',          # Fourth, Fourth 
                r'Fifth[,\s]',           # Fifth, Fifth 
                r'Next[,\s]',            # Next, Next 
                r'Then[,\s]',            # Then, Then 
                r'Now[,\s]',             # Now, Now 
                r'Finally[,\s]',         # Finally, Finally 
                r'Therefore[,\s]',       # Therefore, Therefore 
                r'So[,\s]',              # So, So 
                r'Thus[,\s]',            # Thus, Thus 
                r'Hence[,\s]',           # Hence, Hence 
                r'As\s+a\s+result[,\s]', # As a result, As a result 
                r'In\s+conclusion[,\s]', # In conclusion, In conclusion 
                r'To\s+summarize[,\s]',  # To summarize, To summarize 
                r'Let\s+me\s+',          # Let me 
                r'I\s+need\s+to\s+',     # I need to 
                r'I\s+will\s+',          # I will 
                r'I\s+should\s+',        # I should 
                r'I\s+can\s+',           # I can 
                r'We\s+can\s+',          # We can 
                r'We\s+need\s+to\s+',    # We need to 
            ]
            
            # Clean and normalize the response
            lines = cot_response.split('\n')
            current_step = ""
            step_number = 1
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line starts a new step
                is_new_step = any(re.search(pattern, line, re.IGNORECASE) for pattern in step_patterns)
                
                if is_new_step and current_step:
                    # Save the previous step
                    step_data = self._create_step_data(current_step.strip(), step_number)
                    if step_data:
                        steps.append(step_data)
                        step_number += 1
                    current_step = line
                else:
                    # Continue building the current step
                    if current_step:
                        current_step += " " + line
                    else:
                        current_step = line
            
            # Add the last step
            if current_step:
                step_data = self._create_step_data(current_step.strip(), step_number)
                if step_data:
                    steps.append(step_data)
            
            # If no steps were found, try alternative parsing
            if not steps:
                steps = self._fallback_parsing(cot_response)
            
            logger.info(f"Parsed {len(steps)} reasoning steps")
            return steps
            
        except Exception as e:
            logger.error(f"Error parsing steps: {e}")
            # Return a single step with the raw response
            return [{
                "step_number": "1",
                "content": cot_response.strip(),
                "type": "reasoning"
            }]
    
    def _create_step_data(self, content: str, step_number: int) -> Optional[Dict[str, str]]:
        """Create step data with enhanced classification."""
        if not content or len(content.strip()) < 5:  # Skip very short content
            return None
        
        return {
            "step_number": str(step_number),
            "content": content.strip(),
            "type": self._classify_step(content)
        }
    
    def _fallback_parsing(self, cot_response: str) -> List[Dict[str, str]]:
        """Fallback parsing method for when step patterns aren't found."""
        # Split by sentences and treat each as a step
        sentences = re.split(r'[.!?]+', cot_response)
        steps = []
        
        for i, sentence in enumerate(sentences, 1):
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only include substantial sentences
                steps.append({
                    "step_number": str(i),
                    "content": sentence,
                    "type": self._classify_step(sentence)
                })
        
        return steps
    
    def _classify_step(self, step_content: str) -> str:
        """Enhanced step classification based on content analysis."""
        content_lower = step_content.lower()
        
        # Calculation indicators
        if any(word in content_lower for word in ['calculate', 'multiply', 'divide', 'add', 'subtract', 
                                                 'formula', 'equation', '=', '+', '-', '*', '/', '×', '÷']):
            return "calculation"
        
        # Conclusion indicators
        if any(word in content_lower for word in ['therefore', 'so', 'conclude', 'answer', 'result', 
                                                 'thus', 'hence', 'finally', 'in conclusion', 'as a result']):
            return "conclusion"
        
        # Assumption indicators
        if any(word in content_lower for word in ['assume', 'given', 'known', 'fact', 'premise', 
                                                 'suppose', 'let', 'if', 'since', 'because']):
            return "assumption"
        
        # Analysis indicators
        if any(word in content_lower for word in ['analyze', 'examine', 'consider', 'think', 'reason',
                                                 'understand', 'identify', 'determine', 'find']):
            return "analysis"
        
        # Default to reasoning
        return "reasoning"
    
    def get_generation_stats(self) -> Dict[str, any]:
        """Get statistics about the generation process."""
        return {
            "max_length": self.max_length,
            "model_ready": self.model_manager.is_ready() if self.model_manager else False,
            "model_info": self.model_manager.get_model_info() if self.model_manager else None
        }
