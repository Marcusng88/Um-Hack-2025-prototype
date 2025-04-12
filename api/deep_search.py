from config import GOOGLE_API_KEY
from config import TAVILY_API_KEY

def deep_search(prompt: str) -> str:
    """
    Perform a deep search using Tavily and analyze the results with Google GenAI.

    Args:
        prompt (str): The search query to be analyzed.

    Returns:
        str: The analysis result from the LLM.
    """
    from llama_index.tools.tavily_research import TavilyToolSpec
    from llama_index.llms.google_genai import GoogleGenAI

    # Initialize LLM and tools
    llm = GoogleGenAI(
        api_key= GOOGLE_API_KEY,
        model_name="models/gemma-3-27b-it"
    )
    tavily_tool = TavilyToolSpec(api_key=TAVILY_API_KEY)

    # Perform search
    search_results = tavily_tool.search(prompt, max_results=3)
    combined_results = "\n\n".join([doc.text_resource.text for doc in search_results])

    # Define prompt
    analysis_prompt = (
        "Analyze the following search results and provide a concise summary:\n\n"
        f"{combined_results}"
    )

    response = llm.complete(analysis_prompt)
    return (response.text)


