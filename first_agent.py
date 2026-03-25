# User: "What is Machine Learning?"
#      ↓
# LLM thinks: "Ye ek concept hai, Wikipedia pe milega"
#      ↓
# Tool Selected: Wikipedia

# User: "Latest AI trends in 2025?"
#      ↓
# LLM thinks: "Ye recent/current info hai, internet search chahiye"
#      ↓
# Tool Selected: DuckDuckGo


# Keywords Jo LLM Samajhta Hai:

# Keywords	                                        Tool Selected
# "What is", "Define", "Explain", "History of"	    Wikipedia
# "Latest", "Recent", "Current", "2025", "News"	    DuckDuckGo
# "Calculate", "Math", "+", "-", "*", "/"	        Calculator
# "Current time", "Today's date"	                DateTime


from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# ==================== TOOLS ====================

# 1. Wikipedia Tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# 2. DuckDuckGo Search Tool (Internet Search)
search_tool = DuckDuckGoSearchRun()

# 3. Custom Calculator Tool
@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions. Input should be a valid math expression like '2+2' or '100/5'"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except:
        return "Error: Invalid expression"

# 4. Custom Date/Time Tool
@tool
def get_current_datetime() -> str:
    """Get current date and time"""
    from datetime import datetime
    now = datetime.now()
    return f"Current Date & Time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

# ==================== LLM ====================

llm = ChatOllama(model="llama3.2:3b")

# ==================== AGENT ====================

tools = [wiki_tool, search_tool, calculator, get_current_datetime]
agent = create_react_agent(llm, tools)

# ==================== RUN ====================

def ask_agent(query: str):
    print(f"\n{'='*60}")
    print(f"User: {query}")
    print(f"{'='*60}")

    result = agent.invoke({"messages": [("user", query)]})

    # Print all messages with tool info
    for msg in result["messages"]:
        if msg.type == "ai" and hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                print(f"\nTool Selected: {tool_call['name']}")
                print(f"Tool Input: {tool_call['args']}")
        
        if hasattr(msg, 'content') and msg.content:
            if msg.type == "tool":
                print(f"\nTool Result: {msg.content[:300]}...")
            elif msg.type == "ai" and not msg.tool_calls:
                print(f"\nFinal Answer: {msg.content}")

    return result["messages"][-1].content


# ==================== TEST QUERIES ====================

# Wikipedia Query
ask_agent("What is Machine Learning? Explain briefly.")

# Calculator Query
ask_agent("What is 125 * 48 + 320?")

# Current Time Query
ask_agent("What is current date and time?")

# Internet Search Query
ask_agent("What are latest AI trends in 2025?")