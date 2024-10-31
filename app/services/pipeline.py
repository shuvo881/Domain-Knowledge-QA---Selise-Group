
import logging
from app.services.retrieval import retrieve_documents
from app.services.generation import generate_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_answer(question: str) -> str:
    try:
        logger.info(f"Received question: {question}")

        relevant_docs = retrieve_documents(question)
        if not relevant_docs:
            logger.warning("No relevant documents retrieved for the question.")
        else:
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents for the question.")

        context = "\n".join([doc['page_content'] for doc in relevant_docs if 'page_content' in doc])

        answer = generate_response(context, question)
        logger.info(f"Generated answer: {answer}")
        
        return answer

    except Exception as e:
        logger.error(f"Error in processing question '{question}': {e}", exc_info=True)
        return "An error occurred while processing your request."
