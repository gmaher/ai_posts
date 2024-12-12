import os
from tools import CodeGenerator
from openai import OpenAI
system_preamble ="""
You are an intelligent software engineering AI assistant.
You write clear, concise and modular maintable code.

You have been asked to complete the following project
{goal}

The previous actions you took are
{actions}

The relevant project state context is
{context}

"""

system_planner = system_preamble + """
Now plan the next {k} steps to achieve the goal.

Format your output as:

PLAN:
1. <first step>
2. <second step>
...
"""

system_executor = system_preamble +"""
The next steps for you to execute are:
{plan}

You have access to the following tools:
- CREATE_FILE(filename): Creates a new empty file
- APPEND_TO_FILE(filename, content): Adds content to the end of a file
- MODIFY_FILE(filename, start_line, end_line, content): Replaces content between start_line and end_line
- REMOVE_FILE(filename): Moves a file to the removed folder

When using tools, wrap the tool call in <tool></tool> tags and format as JSON:
<tool>
{
    "name": "CREATE_FILE",
    "arguments": {
        "filename": "example.txt"
    }
}
</tool>

Now please execute the plan.
"""

class LLMPC:
    def __init__(self, api_key: str, goal: str):
        self.generator = CodeGenerator(api_key)
        self.goal = goal
        self.actions = []
        self.context = ""
        
    def get_system_prompt(self, template: str, **kwargs) -> str:
        return template.format(
            goal=self.goal,
            actions="\n".join(f"- {action}" for action in self.actions),
            context=self.context,
            **kwargs
        )

    def plan(self, k: int = 3) -> list:
        prompt = self.get_system_prompt(system_planner, k=k)
        response = self.generator.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract plan steps from response
        content = response.choices[0].message.content
        plan_section = content.split("PLAN:")[1].strip()
        steps = []
        for line in plan_section.split("\n"):
            if line.strip() and line[0].isdigit():
                steps.append(line.split(".", 1)[1].strip())
        return steps

    def execute(self, plan: list) -> None:
        prompt = self.get_system_prompt(system_executor, plan="\n".join(f"{i+1}. {step}" for i, step in enumerate(plan)))
        self.generator.generate(prompt, "")
        self.actions.extend(plan)

def main():
    # Replace with your OpenAI API key
    api_key = "your-api-key-here"
    goal = "create flappy bird using html javascript and css"
    
    llmpc = LLMPC(api_key, goal)
    
    # Run multiple iterations of planning and execution
    for iteration in range(3):
        print(f"\nIteration {iteration + 1}")
        print("Planning...")
        plan = llmpc.plan(k=3)  # Get next 3 steps
        
        print("\nGenerated Plan:")
        for i, step in enumerate(plan, 1):
            print(f"{i}. {step}")
            
        print("\nExecuting plan...")
        llmpc.execute(plan)
        
        # Optional: Add a pause between iterations
        if iteration < 2:
            input("\nPress Enter to continue to next iteration...")

if __name__ == "__main__":
    main()