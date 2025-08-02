import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class CoTGenerator:
    """Generates Chain of Thought reasoning using a language model."""
    
    def __init__(self, model_manager, max_length=200):
        self.model_manager = model_manager
        self.max_length = max_length
        
        # CoT prompting templates
        self.cot_templates = {
            "general": "Let's think step by step.\n\nProblem: {problem}\n\nSolution:",
            "math": "Let's solve this math problem step by step.\n\nProblem: {problem}\n\nStep-by-step solution:",
            "logic": "Let's analyze this logic problem step by step.\n\nProblem: {problem}\n\nReasoning:",
            "riddle": "Let's think through this riddle step by step.\n\nRiddle: {problem}\n\nStep-by-step thinking:"
        }
    
    def _detect_problem_type(self, problem: str) -> str:
        """Detect the type of problem to use appropriate template."""
        problem_lower = problem.lower()
        
        # Math keywords
        math_keywords = ['calculate', 'solve', 'equation', 'multiply', 'divide', 'add', 'subtract', 
                        'percent', '%', 'fraction', 'decimal', 'number', 'sum', 'difference']
        
        # Logic keywords
        logic_keywords = ['if', 'then', 'either', 'or', 'all', 'some', 'none', 'always', 'never',
                         'taller', 'shorter', 'faster', 'slower', 'before', 'after']
        
        # Riddle keywords
        riddle_keywords = ['riddle', 'what am i', 'guess', 'mystery', 'puzzle']
        
        if any(keyword in problem_lower for keyword in math_keywords):
            return "math"
        elif any(keyword in problem_lower for keyword in logic_keywords):
            return "logic"
        elif any(keyword in problem_lower for keyword in riddle_keywords):
            return "riddle"
        else:
            return "general"
    
    def generate_cot(self, problem: str) -> str:
        """Generate Chain of Thought reasoning for the given problem."""
        try:
            # Detect problem type and select appropriate template
            problem_type = self._detect_problem_type(problem)
            template = self.cot_templates[problem_type]
            
            # Format the prompt
            prompt = template.format(problem=problem)
            
            # Generate response
            response = self.model_manager.generate_response(
                prompt, 
                max_length=self.max_length,
                temperature=0.7,
                do_sample=True
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating CoT: {e}")
            raise e
    
    def parse_steps(self, cot_response: str) -> List[Dict[str, str]]:
        """Parse the CoT response into individual reasoning steps."""
        try:
            steps = []
            
            # Split by common step indicators
            step_patterns = [
                r'Step \d+:',
                r'\d+\.',
                r'First,',
                r'Second,',
                r'Third,',
                r'Next,',
                r'Then,',
                r'Finally,',
                r'Therefore,',
                r'So,'
            ]
            
            # Clean the response
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
                    steps.append({
                        "step_number": str(step_number),
                        "content": current_step.strip(),
                        "type": self._classify_step(current_step)
                    })
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
                steps.append({
                    "step_number": str(step_number),
                    "content": current_step.strip(),
                    "type": self._classify_step(current_step)
                })
            
            # If no steps were found, treat the entire response as one step
            if not steps:
                steps.append({
                    "step_number": "1",
                    "content": cot_response.strip(),
                    "type": "reasoning"
                })
            
            return steps
            
        except Exception as e:
            logger.error(f"Error parsing steps: {e}")
            # Return a single step with the raw response
            return [{
                "step_number": "1",
                "content": cot_response.strip(),
                "type": "reasoning"
            }]
    
    def _classify_step(self, step_content: str) -> str:
        """Classify the type of reasoning step."""
        content_lower = step_content.lower()
        
        if any(word in content_lower for word in ['calculate', 'multiply', 'divide', 'add', 'subtract']):
            return "calculation"
        elif any(word in content_lower for word in ['therefore', 'so', 'conclude', 'answer']):
            return "conclusion"
        elif any(word in content_lower for word in ['assume', 'given', 'known', 'fact']):
            return "assumption"
        else:
            return "reasoning"
