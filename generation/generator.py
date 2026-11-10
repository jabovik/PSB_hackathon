def generate_answer(question: str, contexts: list[str]) -> str:
    if not contexts:
        return "нет данных"
    return contexts[0][:200]