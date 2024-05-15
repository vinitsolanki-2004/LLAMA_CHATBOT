from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

logging.basicConfig(level=logging.DEBUG)

def initialise_llama3():
    try:
        # Create chatbot prompt
        create_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are my personal assistant"),
                ("user", "Question: {question}")
            ]
        )

        # Initialize OpenAI LLM and output parser
        lamma_model = Ollama(model="llama3")
        format_output = StrOutputParser()

        # Create chain
        chatbot_pipeline = create_prompt | lamma_model | format_output
        return chatbot_pipeline
    except Exception as e:
        logging.error(f"Failed to initialize chatbot: {e}")
        raise

# Initialize chatbot
chatbot_pipeline = initialise_llama3()

def main():
     
     while(True):

        query_input = input("Enter a prompt : ")

        if query_input.lower() == 'end':
             break
        
        if query_input:
                try:
                    response = chatbot_pipeline.invoke({'question': query_input})
                    print(response)

                except Exception as e:
                    logging.error(f"Error during chatbot invocation: {e}")

if __name__ == '__main__':
     main()