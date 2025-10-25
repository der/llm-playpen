from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pprint import pprint

llm = ChatOpenAI(
    model="geoffmunn/Qwen3-Coder-30B-A3B-Instruct",
    # stream_usage=True,
    # temperature=None,
    # max_tokens=None,
    # timeout=None,
    # reasoning_effort="low",
    max_retries=2,
    api_key="",
    base_url="http://localhost:8080/v1",
    # organization="...",
    # other params...
    use_responses_api= False,
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
pprint(response)
