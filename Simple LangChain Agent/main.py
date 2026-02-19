import json
import os
from datetime import datetime
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pathlib import Path

load_dotenv()

# 1. Створіть LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# 2. Створіть system prompt для агента
system_prompt = """Ви — розумний помічник для студентів українських університетів.
Ваша задача — допомагати знаходити актуальну інформацію та формувати корисні відповіді.
Відповідайте українською мовою."""

# 3. Створіть функцію пошуку
def search_web(query: str) -> str:
    """Пошук інформації через DuckDuckGo"""
    
    results_text = []
    
    with DDGS() as ddgs:
        results = ddgs.text(
            query,
            region="ua-uk",   # український регіон
            safesearch="moderate",
            max_results=5
        )
        
        for r in results:
            results_text.append(
                f"Заголовок: {r['title']}\n"
                f"Опис: {r['body']}\n"
                f"Посилання: {r['href']}\n"
            )
    
    return "\n---\n".join(results_text)

# 4. Створіть функцію збереження звіту
def save_report(content: str, filename: str = None) -> str:
    """Зберігає результати у JSON файл"""
    
    # Якщо ім'я файлу не передано — генеруємо
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.json"
    
    # Структура даних
    data = {
        "created_at": datetime.now().isoformat(),
        "content": content
    }
    
    # Запис у файл
    file_path = Path(filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return f"Звіт збережено у файл: {file_path.resolve()}"


# 5. Створіть ланцюжок (chain) prompt → llm → parser
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

# 6. Запустіть агента
topic = "Штучний інтелект в освіті України 2025"

## 1. Пошук інформації
search_results = search_web(topic)

## 2. Формування промпта для LLM
user_prompt = f"""
Знайдіть та узагальніть інформацію на тему: {topic}

Ось результати пошуку:
{search_results}

Сформуйте структурований звіт:
- короткий вступ
- основні тенденції
- приклади використання
- висновки
"""

## 3. Виклик LLM
response = llm.invoke([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
])

report_text = response.content

## 4. Збереження звіту
file_path = save_report(report_text)

print("Звіт створено:", file_path)
print("\n===== REPORT =====\n")
print(report_text)