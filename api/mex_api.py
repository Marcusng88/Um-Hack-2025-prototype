import zipfile
import pandas as pd

import asyncio
# Ensure an event loop is created
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

path = 'api/Synthetic dataset for task 2.zip'

with zipfile.ZipFile(path) as z:
    with z.open('Synthetic dataset for task 2/items.csv') as f:
        df_items = pd.read_csv(f)
    with z.open('Synthetic dataset for task 2/keywords.csv') as f:
        df_keywords = pd.read_csv(f)
    with z.open('Synthetic dataset for task 2/merchant.csv') as f:
        df_merchant = pd.read_csv(f)
    with z.open('Synthetic dataset for task 2/transaction_data.csv') as f:
        df_transaction_data = pd.read_csv(f)
    with z.open('Synthetic dataset for task 2/transaction_items.csv') as f:
        df_transaction_items = pd.read_csv(f)

from google import genai
client = genai.Client(api_key="AIzaSyDzyqd7nsKzFqYPUVpkq51LEwRPWLz6maw")
model = "gemma-3-27b-it"

from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)

from llama_index.core import PromptTemplate

from llama_index.llms.google_genai import GoogleGenAI


llm = GoogleGenAI(api_key='AIzaSyB65urG5OL56HIsf0_Rxqof-q5e6M9Pjrg', model_name="models/gemma-3-27b-it")

import pandas as pd

df = [df_transaction_items.copy(), df_items.copy(), df_transaction_data.copy(), df_merchant.copy(), df_keywords.copy()]
df = pd.concat(df)

instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)

pandas_prompt_str = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)
response_synthesis_prompt_str = (
    "Given an input question,PLEASE IGNORE THE JSON VARIABLE. synthesize a response from the query results.\n"
    "You should analyze the result and give business insight, the user will be merchant partner from food delivery company.\n"
    "Make it interactive and informative with minimum 250 words. You may generate more\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Graph Generation Output (JSON):\n{llm3_json_output}\n\n"
    "Response: "
)

graph_generator_prompt_str = (
    "Given an input pandas dataframe information {pandas_output} , the input can be None\n"
    "Figure out whether this information needed to be plot as a graph or not"
    """Write it in a json format {
        plot: bool,
        graph_type: type,
        title: str,
        x_label: str,
        y_label: str,
        data_x: list,
        data_y: list,
        color: str,
    }\n"""
    """The plot is True if needed to be plot , False otherwise.
      graph_type determines which graph is needed to be plot.
      title will be the name for the graph.
      x_label is the label for x-axis.
      y_label is the label for y-axis.
      data_x is the list of data for x-axis.
      data_y is the list of data for y-axis.
      color is the color styling for the plotly graph.\n

    """
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
graph_generator_prompt = PromptTemplate(graph_generator_prompt_str)
llm = llm


qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
        "graph_generator_prompt": graph_generator_prompt,
        "llm3": llm,
    },
    verbose=True,
)
qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
qp.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link(
            "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
        ),
        Link(
            "pandas_output_parser",
            "response_synthesis_prompt",
            dest_key="pandas_output",
        ),
        
    ]
)

# add link from response synthesis prompt to llm2


qp.add_link(
    "pandas_output_parser",      
    "graph_generator_prompt",    
    dest_key="pandas_output"     # Key in the destination prompt template
)

qp.add_link(
    "graph_generator_prompt",    # Source component
    "llm3"                       # Destination component
)
qp.add_link(
    "llm3",                           # Source component: the LLM generating JSON
    "response_synthesis_prompt",      # Destination component: the prompt for llm2
    dest_key="llm3_json_output"       # Destination key: matches placeholder in modified prompt
)
qp.add_link("response_synthesis_prompt", "llm2")

def mex_prompt(prompt):
  chat_history = []
  user_input = prompt
  try:
    response,x = qp.run_with_intermediates(query_str=user_input+" .You may refer the chat history "+str(chat_history),)
    chat_history += "User input, "+user_input
    chat_history += "AI answer, "+response.message.content
    return response.message.content

  except Exception as e:
    print(f"An error occurred: {e}")
    print("Attempting to access results directly from pipeline state if possible (this depends on the specific QP implementation)")

  