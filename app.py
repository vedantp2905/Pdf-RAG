import os
import asyncio
import streamlit as st
from io import BytesIO
from docx import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew
from crewai_tools import PDFSearchTool

def save_pdf_file(uploaded_file, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_path = os.path.join(save_folder, uploaded_file.name)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return save_path

def tool(path, mod):
    if mod == 'Gemini':
        rag_tool = PDFSearchTool(
            pdf=path,
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-flash",
                        temperature=0.6
                    ),
                ),
                embedder=dict(
                    provider="google",
                    config=dict(
                        model="models/embedding-001",
                        task_type="retrieval_document",
                        title="Embeddings for PDF",
                    ),
                ),
            ),
        )
    else:
        rag_tool = PDFSearchTool(
            pdf=path
        )
    return rag_tool

def generate_text(llm, question, rag_tool):
    inputs = {'question': question}
    
    writer_agent = Agent(
        role='Research Specialist',
        goal='To accurately and efficiently retrieve and synthesize information from PDFs to answer user questions.',
        backstory='''
        Alex has always been passionate about research and information synthesis. With a background in library science 
        and information technology, Alex developed a keen eye for finding precise information in vast databases and documents.
        Joining the AI research team, Alex now uses advanced PDF retrieval tools to help users find accurate answers quickly.
        Alex's meticulous nature and dedication to accuracy make them the perfect fit for the role of Research Specialist, ensuring
        that every question is answered with reliable and well-sourced information.
        ''',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    task_writer = Task(
        description=('''Use the PDF RAG search tool to accurately and efficiently retrieve and synthesize information from provided PDFs to answer user questions. 
                     The task involves analyzing user queries, retrieving relevant information from the PDFs, and generating clear, concise, and accurate responses.'''),
        agent=writer_agent,
        expected_output="""
        - Clear and concise synthesis of the retrieved information, formatted in a user-friendly manner.
        - A detailed and well-sourced answer to the user's question.
        - References to specific sections or pages in the PDFs where the information was found.
        """,
        tools=[rag_tool]
    )

    crew = Crew(
        agents=[writer_agent],
        tasks=[task_writer],
        verbose=2
    )

    result = crew.kickoff(inputs=inputs)
    return result

def main():
    global llm
    global mod
    
    st.header('RAG Content Generator')

    with st.sidebar:
        with st.form('Gemini/OpenAI'):
            model = st.radio('Choose Your LLM', ('Gemini', 'OpenAI'))
            api_key = st.text_input('Enter your API key', type="password")
            submitted = st.form_submit_button("Submit")

    if api_key and submitted:
        try:
            if model == 'OpenAI':
                async def setup_OpenAI():
                    os.environ["OPENAI_API_KEY"] = api_key
                    llm = ChatOpenAI(temperature=0.6, max_tokens=2000)
                    print("OpenAI Configured")
                    global mod
                    mod = "OpenAI"
                    return llm

                llm = asyncio.run(setup_OpenAI())
                
            elif model == 'Gemini':
                async def setup_gemini():
                    os.environ["GOOGLE_API_KEY"] = api_key
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        verbose=True,
                        temperature=0.6,
                        google_api_key=api_key
                    )
                    print("Gemini Configured")
                    global mod
                    mod = "Gemini"
                    return llm

                llm = asyncio.run(setup_gemini())

        except Exception as e:
            st.error(f"Error configuring LLM: {e}")
            return

    uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')

    if uploaded_file is not None:
        try:
            save_folder = 'Saved Files'
            saved_path = save_pdf_file(uploaded_file, save_folder)
            rag_tool = tool(saved_path, mod)
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            return

        question = st.text_input("Enter your question:")

        if st.button("Generate Answer"):
            with st.spinner("Generating Answer..."):
                try:
                    generated_content = generate_text(llm, question, rag_tool)
                    st.markdown(generated_content)

                    doc = Document()
                    doc.add_heading(question, level=1)
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
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()
