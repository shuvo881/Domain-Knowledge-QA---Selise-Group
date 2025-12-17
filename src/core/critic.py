from src.pydentic_models.rag_model import State

class Critic:
    """
    A Critic class to evaluate and provide feedback on generated answers.
    """


    def self_reflect(self, state: State) -> dict:
        """
        Perform self-reflection on the answer in the given state.

        Parameters:
            state (State): Current state with question, context, and answer

        Returns:
            dict: {
                'answer': original answer,
                'critique': critique text,
                'improvement_needed': True/False
            }
        """
        answer = state['answer']
        question = state['question']
        # Combine all context documents into a single string
        print(doc.page_content for doc in state['context'])

        # context_text = " "#.join(doc['content'] for doc in state['context'])
        context_text = " ".join(doc.page_content for doc in state['context'])
        critique = []
        improvement_needed = False

        # Check if answer is empty or too short
        if not answer.strip():
            critique.append("Answer is empty. No information was provided.")
            improvement_needed = True
        elif len(answer.strip()) < 20:
            critique.append("Answer is too short. More detail is needed.")
            improvement_needed = True

        # Check if answer addresses the question
        if question.lower() not in answer.lower():
            critique.append("Answer does not fully address the question.")
            improvement_needed = True

        # Check if context is referenced
        if context_text and context_text.lower() not in answer.lower():
            critique.append("Answer does not incorporate the retrieved context.")
            improvement_needed = True

        # Optional: simple grammar/structure check
        if answer.count('.') < 1:
            critique.append("Answer may be incomplete or lacks proper sentence structure.")
            improvement_needed = True

        critique_text = " | ".join(critique) if critique else "Answer is good and sufficiently informative."

        return {
            "critique": critique_text,
            "improvement_needed": improvement_needed
        }
