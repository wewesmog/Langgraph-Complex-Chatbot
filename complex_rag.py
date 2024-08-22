from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages

# Define the state with session_id, user_id, and a built-in messages key
class MessagesState(TypedDict):
    session_id: str
    user_id: str
    messages: Annotated[List[dict], add_messages]
    rewritten_question: str
    generated_questions: List[str]
    retrieved_documents: List[str]
    unique_documents: List[str]
    answer: str
    feedback: str

# Define the logic for each node
def input_node(state: MessagesState) -> MessagesState:
    state["messages"] = [{"role": "user", "content": "What is the weather today?"}]
    return state

def memory_rewrite_node(state: MessagesState) -> MessagesState:
    state["rewritten_question"] = "What's today's weather forecast?"
    return state

def question_generation_node(state: MessagesState) -> MessagesState:
    state["generated_questions"] = [
        "What is the weather like today?",
        "Can you tell me today's weather?",
        "How is the weather today?"
    ]
    return state

def document_retrieval_node(state: MessagesState) -> MessagesState:
    state["retrieved_documents"] = [
        "Document 1", "Document 2", "Document 3"
    ]
    return state

def duplicate_removal_node(state: MessagesState) -> MessagesState:
    state["unique_documents"] = ["Document 1", "Document 2"]
    return state

def answer_generation_node(state: MessagesState) -> MessagesState:
    # Generate an answer and store it in the state
    state["answer"] = "The weather today is sunny with a chance of rain."
    
    # Delete the unique_documents key as it's no longer needed
    if "unique_documents" in state:
        del state["unique_documents"]
    
    return state

def output_node(state: MessagesState) -> MessagesState:
    # Append the answer to the conversation history
    state["messages"].append({"role": "assistant", "content": state["answer"]})
    
    # Return the updated state
    return state

def ask_for_feedback_node(state: MessagesState) -> MessagesState:
    state["feedback"] = "Thank you for your feedback!"
    return state

def should_ask_for_feedback(state: MessagesState) -> str:
    # Decide whether to ask for feedback
    return "ask_for_feedback" if state["user_id"] == "user_456" else END

# Create the graph
graph = StateGraph(MessagesState)

# Add nodes to the graph
graph.add_node("input", input_node)
graph.add_node("memory_rewrite", memory_rewrite_node)
graph.add_node("question_generation", question_generation_node)
graph.add_node("document_retrieval", document_retrieval_node)
graph.add_node("duplicate_removal", duplicate_removal_node)
graph.add_node("answer_generation", answer_generation_node)
graph.add_node("output", output_node)
graph.add_node("ask_for_feedback", ask_for_feedback_node)

# Define the edges between nodes
graph.set_entry_point("input")
graph.add_edge("input", "memory_rewrite")
graph.add_edge("memory_rewrite", "question_generation")
graph.add_edge("question_generation", "document_retrieval")
graph.add_edge("document_retrieval", "duplicate_removal")
graph.add_edge("duplicate_removal", "answer_generation")
graph.add_edge("answer_generation", "output")
graph.add_conditional_edges("output", should_ask_for_feedback, {"ask_for_feedback": "ask_for_feedback", END: END})
graph.add_edge("ask_for_feedback", END)

# Compile the graph
app = graph.compile()

# Example invocation with session_id and user_id
result = app.invoke({
    "session_id": "session_1231",
    "user_id": "user_4561"
})
print(result)
