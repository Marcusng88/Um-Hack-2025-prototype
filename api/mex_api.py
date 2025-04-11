import zipfile
import pandas as pd
from llama_index.core.llms import ChatMessage
import plotly.graph_objects as go
import json

import asyncio


from google import genai
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


llm = GoogleGenAI(api_key='AIzaSyB65urG5OL56HIsf0_Rxqof-q5e6M9Pjrg', model_name="models/gemma-3-27b-it")

def choose_dataset(input_question):
    input_question = input_question
    choose_dataset_instruction_prompt = (
        f"1. Given an input question {input_question} , choose the most approriate dataset to be further analysed later\n"
        "2. Theres 5 datasets\n"
        """3. (df_items) has columns item_id:	A unique identifier for each food item
        cuisine_tag: The type of cuisine or food category (e.g. "Side")
        item_name:	The name of the food item (e.g., "Fried Spring Rolls").
        item_price:	The price of the food item (e.g., $3.50).
        merchant_id:	A unique identifier for the restaurant, cafe, or merchant selling the item (e.g., "1d4f2").\n
        
        (df_keywords) keyword:	The name or description of the food searched by users (e.g., "fried spring rolls").
        view:	How many times users viewed this item on the app (e.g., on the main page, in search results).
        menu:	How many times users clicked into the menu page for this item (got curious enough to check details).
        checkout:	How many times users added this item to their cart (but not necessarily completed the purchase).
        order:	How many times users successfully purchased this item.\n
        
        (df_merchant) merchant_id:	Unique identifier for the merchant (e.g., restaurant, cafe, etc.).
        merchant_name:	The name of the restaurant or food vendor.
        join_date:	The date the merchant joined the platform.
        city_id:	An internal ID for the city where the merchant operates\n
        
        (df_transaction_data) order_id:	Unique ID for each order.
        order_time:	When the eater placed the order (2023-11-07 09:49:00).
        driver_arrival_time:	When the driver arrived at the merchant to pick up the order (2023-11-07 09:58:00).
        driver_pickup_time:	When the driver actually picked up the order (2023-11-07 10:06:00).
        delivery_time:	When the order was delivered to the customer (2023-11-07 10:34:00).
        order_value:	Total amount paid for the order
        eater_id:	ID of the customer who ordered
        merchant_id:	ID of the restaurant from where the order was made\n
        
        order_id:	Unique ID of the order.
        item_id	ID: of the food item that was part of the order.
        merchant_id:	ID of the restaurant that provided the item.\n\n"""
        "3. GIVE ONLY THE DATASET NAME as output"
    )

    response = llm.chat([ChatMessage(role="user", content=choose_dataset_instruction_prompt.format(input_question=input_question))])
    response = response.message.content.replace('\n', '')
    if response == 'df_items':
        df = df_items
    elif response == 'df_keywords':
        df = df_keywords
    elif response == 'df_merchant':
        df = df_merchant
    elif response == 'df_transaction_data':
        df = df_transaction_data
    elif response == 'df_transaction_items':
        df = df_transaction_items
    else:
        df = df_items
    return df

def query_pipeline(df):
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
        "Make it interactive and informative with minimum 125 words.\n"
        "If the user query is irrelevant to the dataset ,ANSWER THEIR QUESTION IN A BUSINESS PERSPECTIVE"
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
        color is the color styling for the plotly graph. Please use interactive color\n

        """
        "Do not give other output other than the json\n"
    )

    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str, df_str=df.head(5)
    )
    pandas_output_parser = PandasInstructionParser(df)
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
    graph_generator_prompt = PromptTemplate(graph_generator_prompt_str)

    global qp
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

    qp.add_link(
        "pandas_output_parser",      
        "graph_generator_prompt",    
        dest_key="pandas_output"     
    )

    qp.add_link(
        "graph_generator_prompt",   
        "llm3"                       
    )
    qp.add_link(
        "llm3",                           
        "response_synthesis_prompt",      
        dest_key="llm3_json_output"       
    )
    qp.add_link("response_synthesis_prompt", "llm2")

def graph_generation(response):

  try:
        response = json.loads(response)
  except json.JSONDecodeError:
        print("Error: Invalid JSON string provided.")
        return None

  if response.get('plot'):
        graph_type = response.get('graph_type', 'scatter').lower()
        title = response.get('title', 'Generated Graph')
        x_label = response.get('x_label', 'X-axis')
        y_label = response.get('y_label', 'Y-axis')
        data_x = response.get('data_x', [])
        data_y = response.get('data_y', [])
        color = response.get('color')

        if not data_x or not data_y or len(data_x) != len(data_y):
            print("Error: Invalid or missing data for plotting.")
            return None

        df = pd.DataFrame({'x': data_x, 'y': data_y})
        if graph_type == 'scatter':
            fig = go.Figure(data=[
                go.Scatter(x=df['x'], y=df['y'],
                        mode='markers',
                        marker_color=color,
                        name=title)
            ])
        elif graph_type == 'line':
            fig = go.Figure(data=[
                go.Scatter(x=df['x'], y=df['y'],
                        mode='lines',
                        line=dict(color=color),
                        name=title)
            ])
        elif graph_type == 'bar':
            fig = go.Figure(data=[
                go.Bar(x=df['x'], y=df['y'],
                    marker_color=color,
                    name=title)
            ])
        elif graph_type == 'histogram':
            fig = go.Figure(data=[
                go.Histogram(x=df['x'],
                            marker_color=color,
                            name=title)
            ])
        else:
            # Fallback to a scatter plot with a warning.
            print(f"Warning: Graph type '{graph_type}' is not supported. Defaulting to scatter plot.")
            fig = go.Figure(data=[
                go.Scatter(x=df['x'], y=df['y'],
                        mode='markers',
                        marker_color=color,
                        name=title)
            ])

        # Update the layout with a title and axis labels.
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template='plotly_white'
        )
        return fig
  else:
        print("Plotting is disabled in the response.")
        return None
def mex_prompt(prompt):
  chat_history = []
  user_input = prompt
  try:
    response,x = qp.run_with_intermediates(query_str=user_input+" .You may refer the chat history "+str(chat_history),)
    chat_history.append("User input, "+user_input)
    chat_history.append("AI answer, "+response.message.content)

    graph_response_extraction = (str(x.get('llm3')).split('```'))
    graph_response = graph_response_extraction[1]
    graph_data = graph_response.replace('json', '').replace('\\n', '')
    graph = graph_generation(graph_data)
    return response.message.content, graph

  except Exception as e:
    print(f"An error occurred: {e}")
    print("Attempting to access results directly from pipeline state if possible (this depends on the specific QP implementation)")

  