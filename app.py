import os
import asyncio
from dotenv import load_dotenv
import requests
import streamlit as st
from io import BytesIO
from docx import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew
from crewai_tools import DirectorySearchTool

def verify_gemini_api_key(api_key):
    API_VERSION = 'v1'
    api_url = f"https://generativelanguage.googleapis.com/{API_VERSION}/models?key={api_key}"
    
    try:
        response = requests.get(api_url, headers={'Content-Type': 'application/json'})
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # If we get here, it means the request was successful
        return True
    
    except requests.exceptions.HTTPError as e:
        
        return False
    
    except requests.exceptions.RequestException as e:
        # For any other request-related exceptions
        raise ValueError(f"An error occurred: {str(e)}")

def verify_gpt_api_key(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Using a simple request to the models endpoint
    response = requests.get("https://api.openai.com/v1/models", headers=headers)
    
    if response.status_code == 200:
        return True
    elif response.status_code == 401:
        return False
    else:
        print(f"Unexpected status code: {response.status_code}")
        return False

# Function to handle RAG content generation
def generate_text(llm, question, rag_tool):
    inputs = {'question': question}

    writer_agent = Agent(
        role='Customer Service Specialist',
        goal='To accurately and efficiently answer customer questions',
        backstory='''
        As a seasoned Customer Service Specialist, this bot has honed its 
        skills in delivering prompt and precise solutions to customer queries.
        With a background in handling diverse customer needs across various industries,
        it ensures top-notch service with every interaction.
        ''',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    task_writer = Task(
        description=f'''Use the PDF RAG search tool to accurately and efficiently answer customer question. 
                       The customer question is: {question}
                       The task involves analyzing user queries and generating clear, concise, and accurate responses.''',
        agent=writer_agent,
        expected_output="""
        - A detailed and well-sourced answer to the customer's question.
        - No extra information. Just the answer to the question.
        - Clear and concise synthesis of the retrieved information, formatted in a user-friendly manner.
        """,
        tools=[rag_tool]
    )

    crew = Crew(
        agents=[writer_agent],
        tasks=[task_writer],
        verbose=2,
        context={"Customer Question is ": question}
    )

    result = crew.kickoff(inputs=inputs)
    return result

# Function to configure RAG tool based on selected model
def configure_tool():
    rag_tool = DirectorySearchTool(
        directory="Saved Files",
        config=dict(
            llm=dict(
                provider="openai",
                config=dict(
                    model="gpt-4o",
                    temperature=0.6
                ),
            )
        )
    )
    return rag_tool

def main():
    
    validity_model = False
    
    if 'generated_content' not in st.session_state:
        st.session_state.generated_content = ''
    
    with st.sidebar:
        with st.form('OpenAI'):
            api_key = st.text_input(f'Enter your OpenAI API key', type="password")
            submitted = st.form_submit_button("Submit")

        if api_key:
                validity_model = verify_gpt_api_key(api_key)
                if validity_model ==True:
                    st.write(f"Valid OpenAI API key")
                else:
                    st.write(f"Invalid OpenAI API key")        
        
    if validity_model:    
        async def setup_OpenAI():
            loop = asyncio.get_event_loop()
            if loop is None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            os.environ["OPENAI_API_KEY"] = api_key
            llm = ChatOpenAI(model='gpt-4-turbo',temperature=0.6, max_tokens=2000,api_key=api_key)
            print("OpenAI Configured")
            return llm

        llm = asyncio.run(setup_OpenAI())
            
        rag_tool = configure_tool()

        # Initialize Streamlit app title
        st.title("Chat Bot")
        prompt = st.text_input("What info do you need?")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.empty():
                if message["role"] == "user":
                    st.markdown(f"**User**: {message['content']}")
                elif message["role"] == "assistant":
                    st.markdown(f"**Echo Bot**: {message['content']}")

        # React to user input
        if prompt := prompt:
            # Display user message in chat message container
            st.markdown(f"**User**: {prompt}")
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Process user input and generate response
            response = generate_text(llm,prompt, rag_tool)

            # Display assistant response in chat message container
            st.markdown(f"**Assistant**: {response}")
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
if __name__ == "__main__":
    main()         
