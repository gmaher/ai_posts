import os
from openai import OpenAI

system_prompt ="""
You are an intelligent software engineering AI assistant.
You write clear, concise and modular maintable code.

You have been asked to complete the following project:
{goal}

The previous actions you took are:
{actions}

The relevant project state context is:
{context}

"""

prompt_planner = """
Now plan the next {k} steps to achieve the goal.

Your plan should be detailed and include the files you want to create or modify.

Format your output as:

PLAN:
1. <first step>
2. <second step>
...
"""

prompt_executor = """
The next steps for you to execute are:
{plan}

For each file you want to create or modify, output a code block with the filename in the language specifier.
Use the full file contents - do not use ellipses or partial updates.
Format your output exactly as

```<language specifier> <filename>
<file contents>
```
Do this for each file you want to create or update

Example:
```html index.html
<html>
  <body>
    <h1>Hello World</h1>
  </body>
</html>
```

Now please execute the plan by providing the complete content for each file that needs to be created or modified.
"""

class LLMPC:
    def __init__(self, api_key: str, goal: str):
        self.client = OpenAI(api_key=api_key)
        self.goal = goal
        self.actions = []
        self.context = ""

    def update_context(self):
        files_context = []
        files_dir = "./files"
        
        if not os.path.exists(files_dir):
            os.makedirs(files_dir)
            
        for filename in os.listdir(files_dir):
            file_path = os.path.join(files_dir, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    #numbered_lines = [f"{i} {line.rstrip()}" for i, line in enumerate(lines)]
                    files_context.append(f"File: {filename}\n" + "".join(lines))
        
        self.context = "\n\n".join(files_context)

    def get_system_prompt(self, template: str, **kwargs) -> str:
        self.update_context()  # Update context before generating prompt
        return template.format(
            goal=self.goal,
            actions="\n".join(f"- {action}" for action in self.actions),
            context=self.context,
            **kwargs
        )

    def plan(self, k: int = 3) -> list:
        prompt = self.get_system_prompt(system_prompt)
        instruction = prompt_planner.format(k=k)
        print(prompt,instruction)
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": instruction}
            ],
            temperature=0.7,
            max_tokens=4096
        )
        
        # Extract plan steps from response
        content = response.choices[0].message.content
        print(content)
        plan_section = content.split("PLAN:")[1].strip()
        steps = []
        for line in plan_section.split("\n"):
            if line.strip() and line[0].isdigit():
                steps.append(line.split(".", 1)[1].strip())
        return steps

    def execute(self, plan: list) -> None:
        plan_string = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        prompt = self.get_system_prompt(system_prompt)
        instruction = prompt_executor.format(plan=plan_string)
        print(prompt, instruction)

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": instruction}
            ],
            temperature=0.7,
            max_tokens=4096
        )

        content = response.choices[0].message.content
        print(content)
        # Extract code blocks and save files
        import re
        code_blocks = re.finditer(r'```(\w+)\s+([^\n]+)\n(.*?)```', content, re.DOTALL)
        
        for block in code_blocks:
            language, filename, code = block.groups()
            filepath = os.path.join("files", filename)
            
            # Create or overwrite file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code.strip())

        self.actions.extend(plan)

def main():
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise ValueError("OPENAI_KEY environment variable not set")

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