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
# from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate

from llama_index.llms.google_genai import GoogleGenAI

# To configure model parameters use the `generation_config` parameter.
# eg. generation_config = {"temperature": 0.7, "topP": 0.8, "topK": 40}
# If you only want to set a custom temperature for the model use the
# "temperature" parameter directly.

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
    "Given an input question, synthesize a response from the query results. You should analyze the result and give business insight, the user will be merchant partner from food delivery company. Make it interactive and informative in 100 words.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
llm = llm

qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
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
qp.add_link("response_synthesis_prompt", "llm2")

def mex_prompt(prompt):
  chat_history = []
  user_input = prompt
  response = qp.run(
    query_str= user_input+" .You may refer the chat history "+str(chat_history),
  )
  chat_history += "User input, "+user_input
  chat_history += "AI answer, "+response.message.content
  return response.message.content