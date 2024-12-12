import os
from tools import CodeGenerator

def main():
    system_prompt = """You are a code generation assistant that can create and modify files.
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

Generate code step by step, explaining your actions."""

    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise ValueError("OPENAI_KEY environment variable not set")

    generator = CodeGenerator(api_key)
    generator.generate(system_prompt, "create flappy bird in html javascript and css")

if __name__ == "__main__":
    main()