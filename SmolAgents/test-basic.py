from smolagents import CodeAgent, InferenceClientModel, OpenAIServerModel, DuckDuckGoSearchTool, load_tool, GradioUI

model = OpenAIServerModel(
    model_id="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
    api_base="http://localhost:8080/v1",
    api_key=""
)

# Create an agent with no tools
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model
)

image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

# Run the agent with a task
#result = agent.run("Calculate the sum of numbers from 1 to 10")#
#print(result)

GradioUI(agent).launch()
