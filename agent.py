from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.
)

from langchain.agents import initialize_agent, AgentType

from langchain.chat_models import ChatAnthropic
from langchain.chat_models.fireworks import ChatFireworks

async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()
# llm = ChatOpenAI(temperature=0)
# llm = ChatAnthropic(temperature=0)
llm = ChatFireworks(model="accounts/fireworks/models/mistral-7b")

tools_by_name = {tool.name: tool for tool in tools}
navigate_tool = tools_by_name["navigate_browser"]
get_elements_tool = tools_by_name["get_elements"]

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


async def main():
    result = await agent_chain.arun("navigate to google website and summarize the page")
    # result = "navigate to google website and summarize the page"
    print(result)


async def navigate():
    print("navigating")
    result = await navigate_tool.arun({"url": "https://google.com"})
    print(result)


import asyncio

asyncio.run(main())

# asyncio.run(navigate())
