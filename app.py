import os
import asyncio
import chromadb
import streamlit as st
from io import BytesIO
from docx import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew
from crewai_tools import DirectorySearchTool


def save_pdf_file(uploaded_file, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_path = os.path.join(save_folder, uploaded_file.name)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return save_path

def tool(mod):
    
    
    if mod == 'Gemini':
        rag_tool = DirectorySearchTool(
        directory="Saved Files", #path required of directory
        config=dict(
            
            llm=dict(
                provider="google",  # or google, openai, anthropic, llama2, ...
                config=dict(
                    model="gemini-1.5-flash",
                    temperature=0.6
                ),
            ),
            embedder=dict(
                provider="google",  # or openai, ollama, ...
                config=dict(
                    model="models/embedding-001",
                    task_type="retrieval_document",
                    title="Embeddings for PDF",
                    
                ),
            ),
        )
    )
    
    else:
        rag_tool = DirectorySearchTool(        
        directory="Saved Files", #path required of directory
)
        
    return rag_tool
        
    


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
        description=('''Use the PDF RAG search tool to accurately and efficiently answer customer question. 
                     The task involves analyzing user queries and generating clear, concise, and accurate responses.'''),
        agent=writer_agent,
        expected_output="""
        - A detailed and well-sourced answer to the question.
        - References to specific sections or pages in the PDFs where the information was found.
        - Clear and concise synthesis of the retrieved information, formatted in a user-friendly manner.
        """,
        tools=[rag_tool]
    )

    crew = Crew(
        agents=[writer_agent],
        tasks=[task_writer],
        verbose=2,
        context={"Customer Question ": question}
    )

    result = crew.kickoff(inputs=inputs)
    return result


st.header('RAG Content Generator')
mod = None
with st.sidebar:
    with st.form('Gemini/OpenAI'):
        model = st.radio('Choose Your LLM', ('Gemini', 'OpenAI'))
        api_key = st.text_input(f'Enter your API key', type="password")
        submitted = st.form_submit_button("Submit")

if api_key:
    if model == 'OpenAI':
        async def setup_OpenAI():
            loop = asyncio.get_event_loop()
            if loop is None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            os.environ["OPENAI_API_KEY"] = api_key
            llm = ChatOpenAI(temperature=0.6, max_tokens=2000)
            print("OpenAI Configured")
            return llm

        llm = asyncio.run(setup_OpenAI())
        mod = 'OpenAI'

    if model == 'Gemini':
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
            print(llm)
            print("Gemini Configured")
            return llm

        llm = asyncio.run(setup_gemini())
        mod = 'Gemini'

    question = st.text_input("Enter your question:")

    if st.button("Generate Answer"):
        
            rag_tool = tool(mod)
        
            #with st.spinner("Generating Answer..."):
                
            generated_content = generate_text(llm, question,rag_tool)

            st.markdown(generated_content)

            doc = Document()
            doc.add_heading(question, level=1)  # Specify the heading level
            doc.add_paragraph(generated_content)

            buffer = BytesIO()
            doc.save(buffer)
            buffer.seek(0)
            
            st.download_button(
                    label="Download as Word Document",
                    data=buffer,
                    file_name=f"{question}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
