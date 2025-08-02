class ExampleProblems:
    """Collection of example problems for different categories."""
    
    def __init__(self):
        self.math_problems = [
            {
                "question": "If a train leaves the station at 3 PM traveling at 60 mph and needs to cover 180 miles, what time will it arrive?",
                "category": "time_distance",
                "difficulty": "medium"
            },
            {
                "question": "A store offers a 25% discount on a $80 item. What is the final price after a 8% sales tax is applied?",
                "category": "percentage",
                "difficulty": "medium"
            },
            {
                "question": "If 3 workers can build a wall in 6 days, how many days will it take 9 workers to build the same wall?",
                "category": "work_rate",
                "difficulty": "medium"
            },
            {
                "question": "What is 123 × 45?",
                "category": "arithmetic",
                "difficulty": "easy"
            },
            {
                "question": "A rectangular garden is 15 feet long and 8 feet wide. If you want to put a fence around it, how many feet of fencing do you need?",
                "category": "geometry",
                "difficulty": "easy"
            }
        ]
        
        self.logic_problems = [
            {
                "question": "Alice is taller than Bob. Bob is taller than Carol. Carol is taller than David. Who is the tallest?",
                "category": "ordering",
                "difficulty": "easy"
            },
            {
                "question": "If all roses are flowers, and some flowers are red, can we conclude that some roses are red?",
                "category": "logical_reasoning",
                "difficulty": "medium"
            },
            {
                "question": "In a race, Tom finished before Jerry, Jerry finished before Spike, and Spike finished before Tyke. If there were only these 4 participants, what was Jerry's position?",
                "category": "ordering",
                "difficulty": "easy"
            },
            {
                "question": "Every student in the class passed the test. Sarah is in the class. Did Sarah pass the test?",
                "category": "deduction",
                "difficulty": "easy"
            },
            {
                "question": "If it rains, then the ground gets wet. The ground is wet. Did it rain?",
                "category": "logical_fallacy",
                "difficulty": "medium"
            }
        ]
        
        self.riddles = [
            {
                "question": "I have keys but no locks. I have space but no room. You can enter but not go outside. What am I?",
                "category": "wordplay",
                "difficulty": "medium"
            },
            {
                "question": "What gets wetter as it dries?",
                "category": "wordplay",
                "difficulty": "easy"
            },
            {
                "question": "I'm tall when I'm young and short when I'm old. What am I?",
                "category": "metaphor",
                "difficulty": "easy"
            },
            {
                "question": "What has an eye but cannot see?",
                "category": "wordplay",
                "difficulty": "easy"
            },
            {
                "question": "The more you take, the more you leave behind. What am I?",
                "category": "wordplay",
                "difficulty": "medium"
            }
        ]
    
    def get_math_problems(self):
        """Get all math problems."""
        return self.math_problems
    
    def get_logic_problems(self):
        """Get all logic problems."""
        return self.logic_problems
    
    def get_riddles(self):
        """Get all riddles."""
        return self.riddles
    
    def get_problem_by_category(self, category: str):
        """Get problems by specific category."""
        all_problems = self.math_problems + self.logic_problems + self.riddles
        return [p for p in all_problems if p.get("category") == category]
    
    def get_problems_by_difficulty(self, difficulty: str):
        """Get problems by difficulty level."""
        all_problems = self.math_problems + self.logic_problems + self.riddles
        return [p for p in all_problems if p.get("difficulty") == difficulty]
    
    def get_random_problem(self, problem_type=None):
        """Get a random problem, optionally filtered by type."""
        import random
        
        if problem_type == "math":
            return random.choice(self.math_problems)
        elif problem_type == "logic":
            return random.choice(self.logic_problems)
        elif problem_type == "riddle":
            return random.choice(self.riddles)
        else:
            all_problems = self.math_problems + self.logic_problems + self.riddles
            return random.choice(all_problems)
