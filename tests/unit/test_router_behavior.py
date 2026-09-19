from src.llm import OpenAILLM

def test_obvious_greeting_routes_general_without_api_call():
    llm = OpenAILLM.__new__(OpenAILLM)
    assert llm.route_query("Hello, I am Maumita", [], 8) == "GENERAL"
    assert llm.route_query("Good morning", [], 8) == "GENERAL"
