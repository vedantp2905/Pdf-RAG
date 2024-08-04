import os
import asyncio
import requests
import streamlit as st
from io import BytesIO
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
    
def configure_tool(mod):
    config = {
        'directory': "Saved Files",
        'config': {
            'llm': {
                'provider': "google" if mod == 'Gemini' else "openai",
                'config': {
                    'model': "gemini-1.5-flash" if mod == 'Gemini' else "gpt-4o",
                    'temperature': 0.6
                },
            },
            'embedder': {
                'provider': "google",
                'config': {
                    'model': "models/embedding-001",
                    'task_type': "retrieval_document",
                    'title': "Embeddings"
                },
            },
        }
    }
    
    return DirectorySearchTool(**config)

def generate_text(llm, question, rag_tool, customer_name):
    inputs = {'question': question, 'customer_name': customer_name}

    writer_agent = Agent(
        role='Customer Service Specialist',
        goal='To accurately and efficiently answer customer questions',
        backstory=f'''
        As a seasoned Customer Service Specialist, this bot has honed its 
        skills in delivering prompt and precise solutions to customer queries.
        With a background in handling diverse customer needs across various industries,
        it ensures top-notch service with every interaction.
        ''',
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter = 5
    )

    reviewer_agent = Agent(
        role='Brand Consistency and Quality Assurance Specialist',
        goal='To ensure all responses align with brand guidelines and maintain high quality',
        backstory=f'''
        With years of experience in brand management and quality assurance,
        this specialist ensures that all customer communications are consistent
        with the company's voice, values, and guidelines. They have a keen eye
        for detail and a deep understanding of the brand's identity. 
        ''',
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter = 5
    )

    task_writer = Task(
        description=f'''Use the PDF RAG search tool to accurately and efficiently answer the customer question: {question}
                       The customer's name is {customer_name}. Personalize the response appropriately.
                       Analyze user queries and generate clear, concise, and accurate responses.
                       Only obtain information using the RAG tool and no outside sources.''',
        agent=writer_agent,
        expected_output=f"""
        - A detailed and well-sourced answer to {customer_name}'s question.
        - No extra information. Just the answer to the question.
        - Clear and concise synthesis of the retrieved information, formatted in a user-friendly manner.
        - Personalized greeting and closing using {customer_name}.
        """,
        tools=[rag_tool]
    )

    task_reviewer = Task(
        description=f'''Review the answer provided by the Customer Service Specialist for {customer_name}'s question: {question}
                       Ensure the response aligns with company's brand guidelines, which include:
                       1. Maintaining a professional yet friendly tone
                       2. Accuracy of information
                       3. Conciseness and clarity
                       4. Proper use of company terminology
                       5. Adherence to our core values of transparency, innovation, and customer-centricity
                       6. Appropriate personalization using {customer_name}''',
        agent=reviewer_agent,
        expected_output=f"""
        - A final approved answer to the point
        """,
        context=[task_writer]
    )

    crew = Crew(
        agents=[writer_agent, reviewer_agent],
        tasks=[task_writer, task_reviewer],
        verbose=2
    )

    result = crew.kickoff(inputs=inputs)
    return result


def main():
    
    st.header('ChatBot')
    validity_model = False
    
    if 'generated_content' not in st.session_state:
        st.session_state.generated_content = ''
    
    if 'customer_name' not in st.session_state:
        st.session_state.customer_name = ''
    
    with st.sidebar:
        with st.form('OpenAI'):
            model = st.radio('Choose Your LLM', ('Gemini', 'OpenAI'))
            api_key = st.text_input(f'Enter your API key', type="password")
            submitted = st.form_submit_button("Submit")

        if api_key:
            if model=='OpenAI':
                validity_model = verify_gpt_api_key(api_key)
                if validity_model ==True:
                    st.write(f"Valid {model} API key")
                else:
                    st.write(f"Invalid {model} API key")   
            elif model=='Gemini':
                validity_model = verify_gemini_api_key(api_key)
                if validity_model ==True:
                    st.write(f"Valid {model} API key")
                else:
                    st.write(f"Invalid {model} API key")   
        
    if validity_model:  
        if model == 'OpenAI':
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

        elif model == 'Gemini':
            async def setup_gemini():
                loop = asyncio.get_event_loop()
                if loop is None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                os.environ["GOOGLE_API_KEY"] = api_key
                 
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    verbose=True,
                    temperature=0.6,
                    google_api_key=api_key
                )
                print("Gemini Configured")
                return llm
            
            llm = asyncio.run(setup_gemini())

            
        rag_tool = configure_tool(model)
        
        st.session_state.customer_name = st.text_input('May I know your name?')
        
    if st.session_state.customer_name:
        # Initialize Streamlit app title
        st.title(f"Welcome, {st.session_state.customer_name}!")
        st.write("How can I help you today? I can answer queries on:")
        st.write("1. Core values, mission, and vision")
        st.write("2. Services do we offer")
        st.write("3. Different plans for our services")
        st.write("4. Pricing and Billing")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Place the text input for the user's query at the bottom
        prompt = st.chat_input("Enter your query:")

        # React to user input
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process user input and generate response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response = generate_text(llm, prompt, rag_tool, st.session_state.customer_name)
                response_placeholder.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
