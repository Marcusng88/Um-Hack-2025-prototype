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
        api_key='AIzaSyDzyqd7nsKzFqYPUVpkq51LEwRPWLz6maw',
        model_name="models/gemma-3-27b-it"
    )
    tavily_tool = TavilyToolSpec(api_key="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")

    # Perform search
    search_results = tavily_tool.search(prompt, max_results=3)
    combined_results = "\n\n".join([doc.text_resource.text for doc in search_results])

    # Define prompt
    analysis_prompt = (
        "Analyze the following search results and provide a concise summary:\n\n"
        f"{combined_results}"
    )
    print(111111)
    # Use LLM to analyze the search results
    # response = llm.chat(messages=[{"role": "user", "content": analysis_prompt}])
    response = llm.complete(analysis_prompt)
    return (response.text)


